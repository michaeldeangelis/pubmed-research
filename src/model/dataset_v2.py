"""v2 windows and batch: anchor masking, novel-entity targets, next-year split.

Windows are the v1 windows, each item tagged with its timeline gene. Evidence
tags stay on the items for the probe and never enter the batch.
"""

from __future__ import annotations

import torch

from src.config_v2 import (
    SPLIT_NONE,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    TRAIN_YEAR_MAX,
    VAL_YEAR_MAX,
)
from src.model.dataset import _chunk, _load_jsonl, build_windows, collate_windows
from src.schema import DRUGS, GENES, primary_genes


def _split_for_year(year: int) -> int:
    if year <= TRAIN_YEAR_MAX:
        return SPLIT_TRAIN
    if year <= VAL_YEAR_MAX:
        return SPLIT_VAL
    return SPLIT_TEST


def build_windows_v2(enriched_path, embeddings_path) -> list[list[dict]]:
    """v1 windows, each item with "anchor": the gene whose timeline it is from.

    v1 emits windows gene by gene in GENES order, so the anchor of each window
    follows from the per-gene timeline lengths.
    """
    windows = build_windows(enriched_path, embeddings_path)
    counts = {gene: 0 for gene in GENES}
    for paper in _load_jsonl(enriched_path):
        for gene in primary_genes(list(paper.get("genes") or [])):
            counts[gene] += 1
    anchors = [gene for gene in GENES for _ in _chunk(list(range(counts[gene])))]
    if len(anchors) != len(windows):
        raise ValueError(f"anchor count {len(anchors)} != window count {len(windows)}")
    return [
        [{**paper, "anchor": anchor} for paper in window]
        for window, anchor in zip(windows, anchors)
    ]


def collate_windows_v2(windows: list[list[dict]]) -> dict[str, torch.Tensor]:
    """v1 batch plus gene_target_mask, y_novel_genes, y_novel_drugs, split."""
    batch = collate_windows(windows)
    batch_size, length, n_genes = batch["genes"].shape
    n_drugs = len(DRUGS)

    gene_target_mask = torch.ones(batch_size, length, n_genes, dtype=torch.float32)
    y_novel_genes = torch.zeros(batch_size, length, n_genes, dtype=torch.float32)
    y_novel_drugs = torch.zeros(batch_size, length, n_drugs, dtype=torch.float32)
    split = torch.full((batch_size, length), SPLIT_NONE, dtype=torch.long)

    for batch_index, window in enumerate(windows):
        real = len(window)
        anchor = window[0].get("anchor") if window else None
        if anchor is not None:
            gene_target_mask[batch_index, :, GENES.index(anchor)] = 0.0
        seen_genes = torch.zeros(n_genes)
        seen_drugs = torch.zeros(n_drugs)
        for position in range(real - 1):
            seen_genes = torch.maximum(seen_genes, batch["genes"][batch_index, position])
            seen_drugs = torch.maximum(seen_drugs, batch["drugs"][batch_index, position])
            y_novel_genes[batch_index, position] = batch["y_genes"][batch_index, position] * (
                1.0 - seen_genes
            )
            y_novel_drugs[batch_index, position] = batch["y_drugs"][batch_index, position] * (
                1.0 - seen_drugs
            )
            split[batch_index, position] = _split_for_year(int(window[position + 1]["year"]))

    batch["gene_target_mask"] = gene_target_mask
    batch["y_novel_genes"] = y_novel_genes
    batch["y_novel_drugs"] = y_novel_drugs
    batch["split"] = split
    batch["loss_mask"] = (split == SPLIT_TRAIN).float()
    return batch
