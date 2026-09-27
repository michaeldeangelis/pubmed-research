"""Gene-timeline windows and the training batch.

Evidence tags stay on the window items for the probe. collate_windows never
copies them into the batch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.config import MIN_WINDOW, TEXT_DIM, TRAIN_YEAR_MAX, WINDOW
from src.schema import DISEASES, DRUGS, GENES, multi_hot, primary_genes


def _text_vector(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        vector = value.detach().to(dtype=torch.float32).reshape(-1)
    else:
        vector = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
    if vector.numel() != TEXT_DIM:
        raise ValueError(f"paper text has length {vector.numel()}, expected {TEXT_DIM}")
    return vector


def _clinical_flag(paper: dict) -> float:
    """Own-paper clinical bit. Unit batches store it as y_clinical."""
    if "clinical" in paper:
        return float(paper["clinical"])
    return float(paper.get("y_clinical", 0.0))


def _multi_hot_tensor(items, vocab: list[str]) -> torch.Tensor:
    return torch.tensor(multi_hot(list(items or []), vocab), dtype=torch.float32)


def collate_windows(windows: list[list[dict]]) -> dict[str, torch.Tensor]:
    """Pad windows and shift next-paper labels. No held-out tags in the batch."""
    if not windows:
        raise ValueError("collate_windows received no windows")

    batch_size = len(windows)
    length = max(len(window) for window in windows)
    n_genes = len(GENES)
    n_drugs = len(DRUGS)
    n_diseases = len(DISEASES)

    text = torch.zeros(batch_size, length, TEXT_DIM, dtype=torch.float32)
    year = torch.zeros(batch_size, length, dtype=torch.long)
    genes = torch.zeros(batch_size, length, n_genes, dtype=torch.float32)
    drugs = torch.zeros(batch_size, length, n_drugs, dtype=torch.float32)
    diseases = torch.zeros(batch_size, length, n_diseases, dtype=torch.float32)
    y_genes = torch.zeros(batch_size, length, n_genes, dtype=torch.float32)
    y_drugs = torch.zeros(batch_size, length, n_drugs, dtype=torch.float32)
    y_clinical = torch.zeros(batch_size, length, dtype=torch.float32)
    loss_mask = torch.zeros(batch_size, length, dtype=torch.float32)
    pad_mask = torch.zeros(batch_size, length, dtype=torch.float32)

    for batch_index, window in enumerate(windows):
        real = len(window)
        gene_rows = [_multi_hot_tensor(paper.get("genes"), GENES) for paper in window]
        drug_rows = [_multi_hot_tensor(paper.get("drugs"), DRUGS) for paper in window]
        disease_rows = [_multi_hot_tensor(paper.get("diseases"), DISEASES) for paper in window]
        flags = [_clinical_flag(paper) for paper in window]

        for position, paper in enumerate(window):
            text[batch_index, position] = _text_vector(paper["text"])
            year[batch_index, position] = int(paper["year"])
            genes[batch_index, position] = gene_rows[position]
            drugs[batch_index, position] = drug_rows[position]
            diseases[batch_index, position] = disease_rows[position]
            pad_mask[batch_index, position] = 1.0
            # Position t predicts paper t+1. The last real token has no target.
            if position + 1 < real:
                y_genes[batch_index, position] = gene_rows[position + 1]
                y_drugs[batch_index, position] = drug_rows[position + 1]
                y_clinical[batch_index, position] = flags[position + 1]
                if int(paper["year"]) <= TRAIN_YEAR_MAX:
                    loss_mask[batch_index, position] = 1.0

    return {
        "text": text,
        "year": year,
        "genes": genes,
        "drugs": drugs,
        "diseases": diseases,
        "y_genes": y_genes,
        "y_drugs": y_drugs,
        "y_clinical": y_clinical,
        "loss_mask": loss_mask,
        "pad_mask": pad_mask,
    }


def _load_jsonl(path) -> list[dict]:
    papers: list[dict] = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                papers.append(json.loads(line))
    return papers


def _timeline_sort_key(paper: dict) -> tuple:
    pmid = str(paper.get("pmid", ""))
    if pmid.isdigit():
        pmid_key: tuple = (0, int(pmid))
    else:
        pmid_key = (1, pmid)
    return (str(paper.get("date") or ""), pmid_key)


def _chunk(timeline: list[dict]) -> list[list[dict]]:
    stride = WINDOW // 2
    chunks: list[list[dict]] = []
    for start in range(0, len(timeline), stride):
        chunk = timeline[start : start + WINDOW]
        if len(chunk) < MIN_WINDOW:
            break
        chunks.append(chunk)
    return chunks


def build_windows(enriched_path, embeddings_path) -> list[list[dict]]:
    """Attach each paper to its gene timelines and cut causal windows.

    Item fields: pmid, year, date, text, genes, drugs, diseases, clinical,
    evidence_tags. clinical is 1 when "clinical" is in evidence_tags.
    """
    papers = _load_jsonl(enriched_path)
    embeddings = np.load(embeddings_path)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[1] != TEXT_DIM:
        raise ValueError(
            f"embeddings must be [N, {TEXT_DIM}], got shape {tuple(embeddings.shape)}"
        )
    if len(embeddings) != len(papers):
        raise ValueError(
            f"embedding rows ({len(embeddings)}) do not match papers ({len(papers)})"
        )

    timelines: dict[str, list[dict]] = {gene: [] for gene in GENES}
    for paper, embedding in zip(papers, embeddings):
        anchors = primary_genes(list(paper.get("genes") or []))
        if not anchors:
            continue
        tags = list(paper.get("evidence_tags") or [])
        item = {
            "pmid": paper.get("pmid"),
            "year": int(paper["year"]),
            "date": paper.get("date", ""),
            "text": torch.tensor(embedding, dtype=torch.float32),
            "genes": list(paper.get("genes") or []),
            "drugs": list(paper.get("drugs") or []),
            "diseases": list(paper.get("diseases") or []),
            "clinical": 1.0 if "clinical" in tags else 0.0,
            "evidence_tags": tags,
        }
        for gene in anchors:
            timelines[gene].append(
                {
                    **item,
                    "genes": list(item["genes"]),
                    "drugs": list(item["drugs"]),
                    "diseases": list(item["diseases"]),
                    "evidence_tags": list(item["evidence_tags"]),
                }
            )

    windows: list[list[dict]] = []
    for gene in GENES:
        timeline = timelines[gene]
        timeline.sort(key=_timeline_sort_key)
        windows.extend(_chunk(timeline))
    return windows
