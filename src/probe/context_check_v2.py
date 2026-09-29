"""Post hoc check on the v2 genetics lift: python -m src.probe.context_check_v2

Added after the run-2 decision; not part of the pre-registered rule. The
frozen baseline sees only the current paper, while the label is about earlier
papers, so any model that sees context should beat it. This compares the
trained mixers with each other and with untrained context features: the
current text vector concatenated with the mean text of earlier papers.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from src import config_v2 as cfg
from src.model.dataset_v2 import build_windows_v2
from src.probe.probes import auroc
from src.probe.probes_v2 import _weighted_auroc, fit_direction_v2, probe_checkpoint
from src.schema import prior_genetic_evidence

RESULT_PATH = cfg.ROOT / "results" / "probe_v2_context_check.json"


def _untrained_context_scores(
    windows, labels: np.ndarray, k: int | None, test_year_min: int | None = None
) -> np.ndarray:
    """Probe scores on [text, mean text of previous k (or all) papers, has_context]."""
    if test_year_min is None:
        test_year_min = cfg.VAL_YEAR_MAX + 1
    rows, targets, years = [], [], []
    for window in windows:
        tags = [list(paper.get("evidence_tags") or []) for paper in window]
        for i, paper in enumerate(window):
            earlier = window[:i] if k is None else window[max(0, i - k) : i]
            if earlier:
                context = torch.stack([item["text"] for item in earlier]).mean(dim=0)
            else:
                context = torch.zeros(cfg.TEXT_DIM)
            flag = torch.tensor([1.0 if earlier else 0.0])
            rows.append(torch.cat([paper["text"], context, flag]))
            targets.append(1.0 if prior_genetic_evidence(tags[:i]) else 0.0)
            years.append(int(paper["year"]))
    x = torch.stack(rows)
    y = torch.tensor(targets)
    year = torch.tensor(years)
    train = year <= cfg.TRAIN_YEAR_MAX
    test = year >= test_year_min
    if not np.array_equal(y[test].numpy(), labels):
        raise RuntimeError("untrained context rows do not align with probe test positions")
    direction = fit_direction_v2(x[train], y[train])
    return (x[test] @ direction).numpy()


def _interval(point: float, dist: np.ndarray) -> dict:
    return {
        "point": float(point),
        "lo": float(np.percentile(dist, 2.5)),
        "hi": float(np.percentile(dist, 97.5)),
    }


def main() -> None:
    windows = build_windows_v2(cfg.ENRICHED_PATH, cfg.EMBEDDINGS_PATH)
    runs = {
        (mixer, seed): probe_checkpoint(cfg.CHECKPOINTS_V2 / f"{mixer}_s{seed}.pt", windows)
        for mixer in ("attention", "mean")
        for seed in cfg.SEEDS
    }
    labels = np.asarray(runs[("attention", cfg.SEEDS[0])]["test_labels"], dtype=float)
    for run in runs.values():
        if not np.array_equal(np.asarray(run["test_labels"], dtype=float), labels):
            raise RuntimeError("test labels differ across checkpoints")

    n = len(labels)
    rng = np.random.default_rng(0)
    counts = np.zeros((cfg.BOOTSTRAP_RESAMPLES, n))
    draws = rng.integers(0, n, size=(cfg.BOOTSTRAP_RESAMPLES, n))
    np.add.at(counts, (np.arange(cfg.BOOTSTRAP_RESAMPLES)[:, None], draws), 1.0)

    def full(scores) -> float:
        return float(auroc(torch.tensor(np.asarray(scores)), torch.tensor(labels)))

    def resampled(scores) -> np.ndarray:
        return _weighted_auroc(np.asarray(scores, dtype=float), labels, counts)

    def seed_avg(mixer: str, fn):
        return np.mean(
            [fn(runs[(mixer, seed)]["test_scores_contextual"]) for seed in cfg.SEEDS], axis=0
        )

    frozen = runs[("attention", cfg.SEEDS[0])]["test_scores_frozen"]
    prev_k = _untrained_context_scores(windows, labels, cfg.TOP_K)
    all_earlier = _untrained_context_scores(windows, labels, None)
    attention_full, attention_boot = seed_avg("attention", full), seed_avg("attention", resampled)
    mean_full, mean_boot = seed_avg("mean", full), seed_avg("mean", resampled)

    report = {
        "status": "post hoc, added after the v2 run-2 decision; not part of the pre-registered rule",
        "n_test": int(n),
        "n_test_positive": int(labels.sum()),
        "auroc": {
            "frozen_current_text": full(frozen),
            "untrained_context_prev_top_k_mean_text": full(prev_k),
            "untrained_context_all_earlier_mean_text": full(all_earlier),
            "mean_mixer_seed_avg": float(mean_full),
            "attention_seed_avg": float(attention_full),
        },
        "attention_minus_mean": _interval(attention_full - mean_full, attention_boot - mean_boot),
        "attention_minus_untrained_prev_top_k": _interval(
            attention_full - full(prev_k), attention_boot - resampled(prev_k)
        ),
        "attention_minus_untrained_all_earlier": _interval(
            attention_full - full(all_earlier), attention_boot - resampled(all_earlier)
        ),
    }
    RESULT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
