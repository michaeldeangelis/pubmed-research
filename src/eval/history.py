"""Does gene-timeline history predict the next paper beyond the latest one?

Training tokens stop at 2018. Test targets are papers from 2021 onward.
Uncertainty resamples gene timelines, because consecutive papers are not
independent draws.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.config import (
    BATCH_SIZE,
    EMBEDDINGS_PATH,
    ENRICHED_PATH,
    EPOCHS,
    GRAD_CLIP,
    LR,
    MIN_WINDOW,
    TEXT_DIM,
    TRAIN_YEAR_MAX,
    WEIGHT_DECAY,
    WINDOW,
    YEAR_MAX,
    YEAR_MIN,
)
from src.model.dataset import collate_windows
from src.model.train import _device, prediction_loss
from src.model.transformer import PaperTransformer
from src.probe.probes import auroc
from src.schema import DISEASES, DRUGS, GENES, multi_hot, primary_genes

TEST_YEAR_MIN = 2021
N_BOOT = 1000
HISTORY_DECAY_YEARS = 3.0

YEAR_VOLUME_DIM = 3 + 1 + len(GENES) + len(DRUGS)
CURRENT_DIM = TEXT_DIM + 1 + len(GENES) + len(DRUGS) + len(DISEASES)
CURRENT_VOLUME_DIM = CURRENT_DIM + 2
HISTORY_DIM = CURRENT_VOLUME_DIM + TEXT_DIM


def load_timelines(enriched_path=ENRICHED_PATH, embeddings_path=EMBEDDINGS_PATH):
    papers = []
    with Path(enriched_path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                papers.append(json.loads(line))
    embeddings = np.asarray(np.load(embeddings_path), dtype=np.float32)
    if len(embeddings) != len(papers):
        raise ValueError(
            f"embedding rows ({len(embeddings)}) do not match papers ({len(papers)})"
        )
    timelines = {gene: [] for gene in GENES}
    unique = {}
    for paper, embedding in zip(papers, embeddings):
        item = {
            "pmid": str(paper.get("pmid", "")),
            "year": int(paper["year"]),
            "date": str(paper.get("date") or ""),
            "text": embedding,
            "genes": list(paper.get("genes") or []),
            "drugs": list(paper.get("drugs") or []),
            "diseases": list(paper.get("diseases") or []),
            "clinical": 1.0 if "clinical" in (paper.get("evidence_tags") or []) else 0.0,
        }
        unique[item["pmid"]] = item
        for gene in primary_genes(item["genes"]):
            timelines[gene].append(item)
    for gene in GENES:
        timelines[gene].sort(key=_sort_key)
    return timelines, list(unique.values())


def decision_points(timeline: list[dict], gene: str) -> list[dict]:
    """One row per consecutive pair. The label is the later paper."""
    points = []
    for index in range(len(timeline) - 1):
        target = timeline[index + 1]
        points.append(
            {
                "gene": gene,
                "index": index,
                "target_year": int(target["year"]),
                "y_clinical": float(target["clinical"]),
                "y_genes": np.asarray(multi_hot(target["genes"], GENES), dtype=np.float32),
                "y_drugs": np.asarray(multi_hot(target["drugs"], DRUGS), dtype=np.float32),
            }
        )
    return points


def split_points(points: list[dict]) -> tuple[list[dict], list[dict]]:
    train = [point for point in points if point["target_year"] <= TRAIN_YEAR_MAX]
    test = [point for point in points if point["target_year"] >= TEST_YEAR_MIN]
    return train, test


def training_windows(timelines: dict[str, list[dict]]) -> list[list[dict]]:
    """Windows whose every paper is from the training period."""
    windows = []
    stride = WINDOW // 2
    for timeline in timelines.values():
        early = [paper for paper in timeline if paper["year"] <= TRAIN_YEAR_MAX]
        for start in range(0, len(early), stride):
            chunk = early[start : start + WINDOW]
            if len(chunk) < MIN_WINDOW:
                break
            windows.append([_collate_paper(paper) for paper in chunk])
    return windows


def context_slice(timeline: list[dict], index: int) -> list[dict]:
    start = max(0, index - WINDOW + 1)
    return timeline[start : index + 1]


def shuffle_context(context: list[dict], rng: random.Random) -> list[dict]:
    """Permute earlier papers inside 3-year bins. The latest paper stays put."""
    if len(context) < 2:
        return list(context)
    history = list(context[:-1])
    bins: dict[int, list[dict]] = {}
    for paper in history:
        bins.setdefault(paper["year"] // 3, []).append(paper)
    shuffled: list[dict] = []
    for key in sorted(bins):
        group = list(bins[key])
        rng.shuffle(group)
        shuffled.extend(group)
    return shuffled + [context[-1]]


def replace_history(
    context: list[dict],
    banned_pmids: set[str],
    pool_by_year: dict[int, list[dict]],
    rng: random.Random,
) -> list[dict]:
    """Swap earlier papers for year-matched papers from outside this timeline."""
    if len(context) < 2:
        return list(context)
    replaced = [
        _year_match(paper["year"], banned_pmids, pool_by_year, rng) for paper in context[:-1]
    ]
    return replaced + [context[-1]]


def year_volume_features(timeline: list[dict], index: int) -> np.ndarray:
    context = context_slice(timeline, index)
    current = timeline[index]
    recent = sum(1 for paper in timeline[: index + 1] if paper["year"] >= current["year"] - 2)
    gene_rate = np.mean([_hot(paper["genes"], GENES) for paper in context], axis=0)
    drug_rate = np.mean([_hot(paper["drugs"], DRUGS) for paper in context], axis=0)
    clinical_rate = np.mean([paper["clinical"] for paper in context])
    return np.concatenate(
        [
            np.asarray(
                [_year_norm(current["year"]), np.log1p(index + 1), np.log1p(recent)],
                dtype=np.float32,
            ),
            np.asarray([clinical_rate], dtype=np.float32),
            gene_rate,
            drug_rate,
        ]
    ).astype(np.float32)


def current_features(timeline: list[dict], index: int) -> np.ndarray:
    current = timeline[index]
    return np.concatenate(
        [
            current["text"],
            np.asarray([_year_norm(current["year"])], dtype=np.float32),
            _hot(current["genes"], GENES),
            _hot(current["drugs"], DRUGS),
            _hot(current["diseases"], DISEASES),
        ]
    ).astype(np.float32)


def current_volume_features(timeline: list[dict], index: int) -> np.ndarray:
    current = timeline[index]
    recent = sum(1 for paper in timeline[: index + 1] if paper["year"] >= current["year"] - 2)
    return np.concatenate(
        [
            current_features(timeline, index),
            np.asarray([np.log1p(index + 1), np.log1p(recent)], dtype=np.float32),
        ]
    ).astype(np.float32)


def mean_history_features(timeline: list[dict], index: int, priors: list[dict] | None = None) -> np.ndarray:
    if priors is None:
        priors = context_slice(timeline, index)[:-1]
    mean = _mean_texts(priors)
    return np.concatenate([current_volume_features(timeline, index), mean]).astype(np.float32)


def recency_history_features(
    timeline: list[dict], index: int, priors: list[dict] | None = None
) -> np.ndarray:
    if priors is None:
        priors = context_slice(timeline, index)[:-1]
    mean = _recency_mean(priors, timeline[index]["year"])
    return np.concatenate([current_volume_features(timeline, index), mean]).astype(np.float32)


def fit_linear(features: np.ndarray, labels: np.ndarray):
    """Ridge scores on train-standardized features. The bias column is unpenalized."""
    train = torch.tensor(np.asarray(features), dtype=torch.float32)
    target = torch.tensor(np.asarray(labels), dtype=torch.float32)
    if target.ndim == 1:
        target = target.unsqueeze(-1)
    mean = train.mean(dim=0)
    std = train.std(dim=0).clamp_min(1e-6)
    design = torch.cat([(train - mean) / std, torch.ones(train.shape[0], 1)], dim=1)
    gram = design.T @ design
    penalty = torch.full((gram.shape[0],), 10.0)
    penalty[-1] = 0.0
    coef = torch.linalg.solve(gram + torch.diag(penalty), design.T @ target)
    return mean.detach(), std.detach(), coef.detach()


def linear_scores(mean: torch.Tensor, std: torch.Tensor, coef: torch.Tensor, features: np.ndarray) -> torch.Tensor:
    values = torch.tensor(np.asarray(features), dtype=torch.float32)
    design = torch.cat([(values - mean) / std, torch.ones(values.shape[0], 1)], dim=1)
    return (design @ coef).detach()


def calibrate_logits(train_scores: torch.Tensor, labels: np.ndarray, test_scores: torch.Tensor) -> torch.Tensor:
    """One-dimensional logistic map from a ridge score to a logit. Ranking follows the fit."""
    scale = torch.nn.Parameter(torch.ones(()))
    bias = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam((scale, bias), lr=0.05)
    observed = torch.tensor(labels, dtype=torch.float32)
    raw = train_scores.detach().reshape(-1)
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(raw * scale + bias, observed)
        loss.backward()
        optimizer.step()
    return (test_scores.detach().reshape(-1) * scale.detach() + bias.detach()).detach()


def cluster_bootstrap(gene_ids: list[str], statistic, n_boot: int = N_BOOT, seed: int = 0):
    """Percentile interval from resampling whole gene timelines."""
    groups: dict[str, list[int]] = {}
    for index, gene in enumerate(gene_ids):
        groups.setdefault(gene, []).append(index)
    keys = list(groups)
    point = float(statistic(list(range(len(gene_ids)))))
    rng = random.Random(seed)
    samples = []
    for _ in range(n_boot):
        chosen = [keys[rng.randrange(len(keys))] for _ in keys]
        rows = [row for key in chosen for row in groups[key]]
        value = statistic(rows)
        if value is None:
            continue
        samples.append(float(value))
    if len(samples) < 50:
        return {"point": point, "lo": None, "hi": None, "n_boot": len(samples)}
    samples.sort()
    lo = samples[int(0.025 * (len(samples) - 1))]
    hi = samples[int(0.975 * (len(samples) - 1))]
    return {"point": point, "lo": lo, "hi": hi, "n_boot": len(samples)}


def run_history_comparison(timelines=None, papers=None, epochs: int = EPOCHS, seed: int = 0) -> dict:
    if timelines is None:
        timelines, papers = load_timelines()
    points = []
    for gene in GENES:
        points.extend(decision_points(timelines[gene], gene))
    train, test = split_points(points)
    if not train or not test:
        raise RuntimeError(f"empty split: train={len(train)} test={len(test)}")

    windows = training_windows(timelines)
    train_years = [paper["year"] for window in windows for paper in window]
    if train_years and max(train_years) > TRAIN_YEAR_MAX:
        raise RuntimeError("training window contains a post-2018 paper")
    model = _train_transformer(windows, epochs, seed)

    pool = _pool_by_year(papers)
    rng = random.Random(0)
    designs = {
        "year_volume": _matrix(train, timelines, year_volume_features),
        "current": _matrix(train, timelines, current_features),
        "current_volume": _matrix(train, timelines, current_volume_features),
        "mean_history": _matrix(train, timelines, mean_history_features),
        "recency_history": _matrix(train, timelines, recency_history_features),
    }
    test_designs = {
        "year_volume": _matrix(test, timelines, year_volume_features),
        "current": _matrix(test, timelines, current_features),
        "current_volume": _matrix(test, timelines, current_volume_features),
        "mean_history": _matrix(test, timelines, mean_history_features),
        "recency_history": _matrix(test, timelines, recency_history_features),
    }
    unrelated_contexts = []
    for point in test:
        context = context_slice(timelines[point["gene"]], point["index"])
        banned = {paper["pmid"] for paper in timelines[point["gene"]]}
        unrelated_contexts.append(replace_history(context, banned, pool, rng))
    test_designs["mean_unrelated"] = np.stack(
        [
            mean_history_features(
                timelines[point["gene"]], point["index"], priors=unrelated_contexts[row][:-1]
            )
            for row, point in enumerate(test)
        ]
    ).astype(np.float32)
    y_clinical = np.asarray([point["y_clinical"] for point in train], dtype=np.float32)
    y_genes = np.stack([point["y_genes"] for point in train])
    y_drugs = np.stack([point["y_drugs"] for point in train])
    test_clinical = np.asarray([point["y_clinical"] for point in test], dtype=np.float32)
    test_genes = np.stack([point["y_genes"] for point in test])
    test_drugs = np.stack([point["y_drugs"] for point in test])
    test_gene_ids = [point["gene"] for point in test]

    scores = {}
    gene_scores = {}
    drug_scores = {}
    clinical_heads = {}
    for name, train_x in designs.items():
        clinical_heads[name] = fit_linear(train_x, y_clinical)
        train_clinical_scores = linear_scores(*clinical_heads[name], train_x).reshape(-1)
        test_clinical_scores = linear_scores(*clinical_heads[name], test_designs[name]).reshape(-1)
        scores[name] = calibrate_logits(train_clinical_scores, y_clinical, test_clinical_scores)
        gene_head = fit_linear(train_x, y_genes)
        drug_head = fit_linear(train_x, y_drugs)
        gene_scores[name] = linear_scores(*gene_head, test_designs[name])
        drug_scores[name] = linear_scores(*drug_head, test_designs[name])

    train_mean_scores = linear_scores(*clinical_heads["mean_history"], designs["mean_history"]).reshape(-1)
    test_unrelated_scores = linear_scores(
        *clinical_heads["mean_history"], test_designs["mean_unrelated"]
    ).reshape(-1)
    scores["mean_unrelated"] = calibrate_logits(train_mean_scores, y_clinical, test_unrelated_scores)

    transformer_context = [_torch_context(context_slice(timelines[point["gene"]], point["index"])) for point in test]
    shuffled_context = [
        _torch_context(shuffle_context(context_slice(timelines[point["gene"]], point["index"]), rng))
        for point in test
    ]
    trans_clinical, trans_genes, trans_drugs = _transformer_logits(model, transformer_context)
    shuf_clinical, _, _ = _transformer_logits(model, shuffled_context)
    unrel_clinical, _, _ = _transformer_logits(model, [_torch_context(context) for context in unrelated_contexts])
    scores["transformer"] = trans_clinical
    scores["transformer_shuffled"] = shuf_clinical
    scores["transformer_unrelated"] = unrel_clinical
    gene_scores["transformer"] = trans_genes
    drug_scores["transformer"] = trans_drugs

    prevalence = float(y_clinical.mean())
    prevalence_logit = torch.full(
        (len(test_clinical),),
        float(np.log(prevalence / max(1.0 - prevalence, 1e-6))),
    )

    models = {}
    for name, clinical in scores.items():
        block = {
            "clinical_auroc": _boot_auroc(test_gene_ids, clinical, torch.tensor(test_clinical)),
            "clinical_bce": _bce(clinical, torch.tensor(test_clinical)),
        }
        if name in gene_scores:
            block["gene_micro_auroc"] = _boot_micro(test_gene_ids, gene_scores[name], torch.tensor(test_genes))
            block["drug_micro_auroc"] = _boot_micro(test_gene_ids, drug_scores[name], torch.tensor(test_drugs))
        models[name] = block
    models["prevalence"] = {
        "clinical_auroc": {"point": 0.5, "lo": None, "hi": None, "n_boot": 0},
        "clinical_bce": _bce(prevalence_logit, torch.tensor(test_clinical)),
    }

    deltas = {
        "mean_history_minus_current_volume": _boot_delta(
            test_gene_ids, scores["mean_history"], scores["current_volume"], torch.tensor(test_clinical)
        ),
        "recency_history_minus_current_volume": _boot_delta(
            test_gene_ids, scores["recency_history"], scores["current_volume"], torch.tensor(test_clinical)
        ),
        "transformer_minus_current_volume": _boot_delta(
            test_gene_ids, scores["transformer"], scores["current_volume"], torch.tensor(test_clinical)
        ),
        "transformer_minus_mean_history": _boot_delta(
            test_gene_ids, scores["transformer"], scores["mean_history"], torch.tensor(test_clinical)
        ),
        "transformer_minus_shuffled": _boot_delta(
            test_gene_ids, scores["transformer"], scores["transformer_shuffled"], torch.tensor(test_clinical)
        ),
        "transformer_minus_unrelated": _boot_delta(
            test_gene_ids, scores["transformer"], scores["transformer_unrelated"], torch.tensor(test_clinical)
        ),
        "mean_history_minus_unrelated": _boot_delta(
            test_gene_ids, scores["mean_history"], scores["mean_unrelated"], torch.tensor(test_clinical)
        ),
    }
    gene_deltas = {
        "transformer_minus_current_volume": _boot_micro_delta(
            test_gene_ids,
            gene_scores["transformer"],
            gene_scores["current_volume"],
            torch.tensor(test_genes),
        ),
        "mean_history_minus_current_volume": _boot_micro_delta(
            test_gene_ids,
            gene_scores["mean_history"],
            gene_scores["current_volume"],
            torch.tensor(test_genes),
        ),
    }
    return {
        "question": (
            "On held-out years, does gene-timeline history predict the next paper "
            "beyond the latest abstract, publication volume, and calendar year?"
        ),
        "split": {
            "train_target_year_max": TRAIN_YEAR_MAX,
            "test_target_year_min": TEST_YEAR_MIN,
            "n_train": len(train),
            "n_test": len(test),
            "n_training_windows": len(windows),
            "max_training_paper_year": max(train_years) if train_years else None,
        },
        "leakage": {
            "training_tokens": "Training windows contain only papers with year <= 2018.",
            "citation_selection": (
                "The corpus is Europe PMC sorted by present-day citation count, "
                "at most 80 papers per year. That selection was not knowable at the historical cutoff."
            ),
        },
        "test_clinical_base_rate": float(test_clinical.mean()),
        "models": models,
        "deltas_clinical_auroc": deltas,
        "deltas_gene_micro_auroc": gene_deltas,
        "per_gene": _per_gene(test, scores),
    }


def _train_transformer(windows: list[list[dict]], epochs: int, seed: int = 0) -> PaperTransformer:
    if not windows:
        raise RuntimeError("no training windows")
    torch.manual_seed(seed)
    device = _device()
    model = PaperTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    order = list(range(len(windows)))
    rng = random.Random(seed)
    model.train()
    for epoch in range(epochs):
        rng.shuffle(order)
        total = 0.0
        steps = 0
        for start in range(0, len(order), BATCH_SIZE):
            batch_windows = [windows[index] for index in order[start : start + BATCH_SIZE]]
            batch = {key: value.to(device) for key, value in collate_windows(batch_windows).items()}
            optimizer.zero_grad(set_to_none=True)
            loss = prediction_loss(model(batch), batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        print(f"history epoch {epoch + 1}/{epochs} loss {total / max(steps, 1):.4f}", flush=True)
    model.eval()
    return model


def _transformer_logits(model: PaperTransformer, contexts: list[list[dict]]):
    clinical = []
    genes = []
    drugs = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, len(contexts), BATCH_SIZE):
            chunk = contexts[start : start + BATCH_SIZE]
            batch = {key: value.to(device) for key, value in collate_windows(chunk).items()}
            out = model(batch)
            for row, context in enumerate(chunk):
                last = len(context) - 1
                clinical.append(out["logits_clinical"][row, last].detach().cpu())
                genes.append(out["logits_genes"][row, last].detach().cpu())
                drugs.append(out["logits_drugs"][row, last].detach().cpu())
    return torch.stack(clinical), torch.stack(genes), torch.stack(drugs)


def _boot_auroc(gene_ids, scores, labels):
    scores = scores.detach().cpu().reshape(-1)
    labels = labels.detach().cpu().reshape(-1)

    def statistic(rows):
        if len(rows) < 2:
            return None
        observed = labels[rows]
        if float(observed.max()) == float(observed.min()):
            return None
        return auroc(scores[rows], observed)

    return cluster_bootstrap(gene_ids, statistic)


def _boot_micro(gene_ids, scores, labels):
    scores = scores.detach().cpu()
    labels = labels.detach().cpu()

    def statistic(rows):
        if len(rows) < 2:
            return None
        observed = labels[rows].reshape(-1)
        if float(observed.max()) == float(observed.min()):
            return None
        return auroc(scores[rows].reshape(-1), observed)

    return cluster_bootstrap(gene_ids, statistic)


def _boot_micro_delta(gene_ids, left, right, labels):
    left = left.detach().cpu()
    right = right.detach().cpu()
    labels = labels.detach().cpu()

    def statistic(rows):
        if len(rows) < 2:
            return None
        observed = labels[rows].reshape(-1)
        if float(observed.max()) == float(observed.min()):
            return None
        return auroc(left[rows].reshape(-1), observed) - auroc(right[rows].reshape(-1), observed)

    return cluster_bootstrap(gene_ids, statistic)


def _boot_delta(gene_ids, left, right, labels):
    left = left.detach().cpu().reshape(-1)
    right = right.detach().cpu().reshape(-1)
    labels = labels.detach().cpu().reshape(-1)

    def statistic(rows):
        if len(rows) < 2:
            return None
        observed = labels[rows]
        if float(observed.max()) == float(observed.min()):
            return None
        return auroc(left[rows], observed) - auroc(right[rows], observed)

    return cluster_bootstrap(gene_ids, statistic)


def _bce(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float(
        F.binary_cross_entropy_with_logits(
            logits.detach().cpu().reshape(-1), labels.detach().cpu().reshape(-1).float()
        )
    )


def _per_gene(test: list[dict], scores: dict[str, torch.Tensor]) -> list[dict]:
    rows = []
    by_gene: dict[str, list[int]] = {}
    for index, point in enumerate(test):
        by_gene.setdefault(point["gene"], []).append(index)
    clinical = torch.tensor([point["y_clinical"] for point in test])
    for gene, indexes in by_gene.items():
        observed = clinical[indexes]
        if len(indexes) < 8 or float(observed.max()) == float(observed.min()):
            continue
        entry = {"gene": gene, "n": len(indexes), "clinical_base_rate": float(observed.mean())}
        for name in ("current_volume", "mean_history", "transformer"):
            entry[name] = auroc(scores[name][indexes], observed)
        rows.append(entry)
    return rows


def _matrix(points, timelines, feature_fn) -> np.ndarray:
    rows = [feature_fn(timelines[point["gene"]], point["index"]) for point in points]
    return np.stack(rows).astype(np.float32)


def _pool_by_year(papers: list[dict]) -> dict[int, list[dict]]:
    pool: dict[int, list[dict]] = {}
    for paper in papers:
        pool.setdefault(int(paper["year"]), []).append(paper)
    return pool


def _year_match(year: int, banned: set[str], pool_by_year: dict[int, list[dict]], rng: random.Random) -> dict:
    for radius in range(0, 40):
        years = [year] if radius == 0 else [year - radius, year + radius]
        candidates = [
            paper
            for candidate_year in years
            for paper in pool_by_year.get(candidate_year, [])
            if paper["pmid"] not in banned
        ]
        if candidates:
            return candidates[rng.randrange(len(candidates))]
    raise RuntimeError(f"no year match for {year}")


def _mean_texts(papers: list[dict]) -> np.ndarray:
    if not papers:
        return np.zeros(TEXT_DIM, dtype=np.float32)
    return np.mean([paper["text"] for paper in papers], axis=0).astype(np.float32)


def _recency_mean(papers: list[dict], current_year: int) -> np.ndarray:
    if not papers:
        return np.zeros(TEXT_DIM, dtype=np.float32)
    gaps = np.asarray([current_year - paper["year"] for paper in papers], dtype=np.float32)
    weights = np.exp(-np.maximum(gaps, 0.0) / HISTORY_DECAY_YEARS)
    weights = weights / weights.sum()
    stacked = np.stack([paper["text"] for paper in papers])
    return (stacked * weights[:, None]).sum(axis=0).astype(np.float32)


def _hot(items: list[str], vocab: list[str]) -> np.ndarray:
    return np.asarray(multi_hot(items, vocab), dtype=np.float32)


def _year_norm(year: int) -> float:
    span = max(YEAR_MAX - YEAR_MIN, 1)
    return float(year - YEAR_MIN) / span


def _sort_key(paper: dict) -> tuple:
    pmid = str(paper.get("pmid", ""))
    pmid_key: tuple = (0, int(pmid)) if pmid.isdigit() else (1, pmid)
    return (str(paper.get("date") or ""), pmid_key)


def _collate_paper(paper: dict) -> dict:
    return {
        "pmid": paper["pmid"],
        "year": paper["year"],
        "text": torch.tensor(paper["text"], dtype=torch.float32),
        "genes": list(paper["genes"]),
        "drugs": list(paper["drugs"]),
        "diseases": list(paper["diseases"]),
        "clinical": float(paper["clinical"]),
    }


def _torch_context(context: list[dict]) -> list[dict]:
    return [_collate_paper(paper) for paper in context]
