"""2026 confirmation of the v2 attention-vs-mean genetics readout.

python -m src.probe.confirm_2026

The rule was fixed before any 2026 paper was fetched; see the "2026
confirmation" section of docs/specs/2026-09-29-model-v2-design.md. No model
is retrained. Nothing under data/ is written.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from src import config_v2 as cfg
from src.claims.extract import enrich_paper
from src.encode import encoder
from src.ingest.pubmed import _fetch_year
from src.model.dataset_v2 import build_windows_v2
from src.probe.context_check_v2 import _untrained_context_scores
from src.probe.probes import auroc
from src.probe.probes_v2 import _weighted_auroc, probe_checkpoint
from src.schema import paper_text

CONFIRM_YEAR = 2026
MIN_CLASS = 30
WORK_DIR = cfg.OUTPUTS_V2 / "confirm_2026"
RESULT_PATH = cfg.ROOT / "results" / "probe_v2_confirm_2026.json"


def _read_jsonl(path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _fetch_new_papers(existing: list[dict]) -> list[dict]:
    """2026 papers, same query and cap, cached so the fetch happens once."""
    path = WORK_DIR / "papers_2026.jsonl"
    if path.is_file():
        return _read_jsonl(path)
    seen = {str(paper.get("pmid")) for paper in existing}
    papers = _fetch_year(CONFIRM_YEAR, cfg.PER_YEAR, seen)
    papers = [paper for paper in papers if int(paper["year"]) == CONFIRM_YEAR]
    _write_jsonl(path, papers)
    return papers


def _encode_neural(texts: list[str]) -> np.ndarray:
    vectors = encoder.encode_texts(texts, prefer_neural=True)
    if encoder.LAST_ENCODER != cfg.ENCODER_NAME:
        raise SystemExit(f"encoder fell back to {encoder.LAST_ENCODER}; the rule requires the neural encoder")
    return np.asarray(vectors, dtype=np.float32)


def _check_encoder_matches(enriched: list[dict], embeddings: np.ndarray) -> float:
    """Re-encode a few corpus papers and compare with the stored vectors."""
    sample = list(range(0, len(enriched), max(1, len(enriched) // 5)))[:5]
    fresh = _encode_neural([paper_text(enriched[i]) for i in sample])
    cosine = float(np.min(np.sum(fresh * embeddings[sample], axis=1)))
    if cosine < 0.99:
        raise SystemExit(f"re-encoded corpus vectors differ from stored ones (min cosine {cosine:.4f})")
    return cosine


def _combined_inputs() -> tuple[int, float]:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    enriched = _read_jsonl(cfg.ENRICHED_PATH)
    embeddings = np.load(cfg.EMBEDDINGS_PATH).astype(np.float32)
    new = [enrich_paper(paper) for paper in _fetch_new_papers(enriched)]
    if not new:
        raise SystemExit("Europe PMC returned no 2026 papers")
    cosine = _check_encoder_matches(enriched, embeddings)
    new_vectors = _encode_neural([paper_text(paper) for paper in new])
    _write_jsonl(WORK_DIR / "papers_enriched_combined.jsonl", enriched + new)
    np.save(WORK_DIR / "embeddings_combined.npy", np.concatenate([embeddings, new_vectors]))
    return len(new), cosine


def _interval(point: float, dist: np.ndarray) -> dict:
    return {
        "point": float(point),
        "lo": float(np.percentile(dist, 2.5)),
        "hi": float(np.percentile(dist, 97.5)),
    }


def main() -> None:
    n_new, cosine = _combined_inputs()
    windows = build_windows_v2(
        WORK_DIR / "papers_enriched_combined.jsonl", WORK_DIR / "embeddings_combined.npy"
    )
    runs = {
        (mixer, seed): probe_checkpoint(
            cfg.CHECKPOINTS_V2 / f"{mixer}_s{seed}.pt", windows, test_year_min=CONFIRM_YEAR
        )
        for mixer in ("attention", "mean")
        for seed in cfg.SEEDS
    }
    labels = np.asarray(runs[("attention", cfg.SEEDS[0])]["test_labels"], dtype=float)
    for run in runs.values():
        if not np.array_equal(np.asarray(run["test_labels"], dtype=float), labels):
            raise RuntimeError("test labels differ across checkpoints")
    n = len(labels)
    positives = int(labels.sum())
    negatives = n - positives

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

    attention_full, attention_boot = seed_avg("attention", full), seed_avg("attention", resampled)
    mean_full, mean_boot = seed_avg("mean", full), seed_avg("mean", resampled)
    frozen = runs[("attention", cfg.SEEDS[0])]["test_scores_frozen"]
    all_earlier = _untrained_context_scores(windows, labels, None, test_year_min=CONFIRM_YEAR)

    primary = _interval(attention_full - mean_full, attention_boot - mean_boot)
    powered = positives >= MIN_CLASS and negatives >= MIN_CLASS
    if not powered:
        outcome = "inconclusive"
    elif primary["point"] > 0 and primary["lo"] > 0:
        outcome = "confirmed"
    else:
        outcome = "not confirmed"

    report = {
        "rule": "docs/specs/2026-09-29-model-v2-design.md, section '2026 confirmation'",
        "outcome": outcome,
        "n_new_papers": n_new,
        "encoder_check_min_cosine": cosine,
        "n_test": n,
        "n_test_positive": positives,
        "n_test_negative": negatives,
        "powered": powered,
        "attention_minus_mean": primary,
        "auroc": {
            "frozen_current_text": full(frozen),
            "untrained_context_all_earlier_mean_text": full(all_earlier),
            "mean_mixer_seed_avg": float(mean_full),
            "attention_seed_avg": float(attention_full),
            "per_seed": {
                f"{mixer}_s{seed}": run["contextual_auroc"] for (mixer, seed), run in runs.items()
            },
        },
        "secondary": {
            "attention_minus_frozen": _interval(
                attention_full - full(frozen), attention_boot - resampled(frozen)
            ),
            "attention_minus_untrained_all_earlier": _interval(
                attention_full - full(all_earlier), attention_boot - resampled(all_earlier)
            ),
        },
    }
    RESULT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
