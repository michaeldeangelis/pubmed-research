"""Run the v2 probes and the decision rule: python -m src.probe.v2"""

from __future__ import annotations

import json

import numpy as np

SPEC_PATH = "docs/specs/2026-09-29-model-v2-design.md"
_SCORE_KEYS = ("test_labels", "test_scores_frozen", "test_scores_contextual")
_V1_KEYS = (
    "n_test_positions",
    "prior_genetics_auroc_frozen",
    "prior_genetics_auroc_contextual",
    "prior_genetics_auroc_lift",
    "resistance_auroc_contextual",
    "head_genetics_attention",
)


def main() -> dict:
    from src import config_v2 as cfg
    from src.model.dataset_v2 import build_windows_v2
    from src.probe.probes import _encoder_name
    from src.probe.probes_v2 import _test_totals, bootstrap_lift, decide, probe_checkpoint

    training_path = cfg.OUTPUTS_V2 / "training.json"
    if not training_path.is_file():
        raise SystemExit(f"Missing {training_path}; run python -m src.model.train_v2 first.")
    attention_paths = [cfg.CHECKPOINTS_V2 / f"attention_s{seed}.pt" for seed in cfg.SEEDS]
    missing = [str(path) for path in attention_paths if not path.is_file()]
    if missing:
        raise SystemExit("Missing attention checkpoint(s): " + ", ".join(missing))
    training = json.loads(training_path.read_text(encoding="utf-8"))

    windows = build_windows_v2(cfg.ENRICHED_PATH, cfg.EMBEDDINGS_PATH)
    probes: dict[str, dict[int, dict]] = {}
    for mixer in cfg.MIXERS:
        for seed in cfg.SEEDS:
            path = cfg.CHECKPOINTS_V2 / f"{mixer}_s{seed}.pt"
            if not path.is_file():
                continue
            probes.setdefault(mixer, {})[seed] = probe_checkpoint(path, windows)

    attention_seeds = [probes["attention"][seed] for seed in cfg.SEEDS]
    bootstrap = bootstrap_lift(attention_seeds, cfg.BOOTSTRAP_RESAMPLES, 0)
    decision = decide(training, bootstrap)

    report = {
        "encoder": _encoder_name(cfg.EMBEDDING_META_PATH),
        "spec": SPEC_PATH,
        "decision": decision,
        "bootstrap": bootstrap,
        "probes": {
            mixer: {
                str(seed): {k: v for k, v in result.items() if k not in _SCORE_KEYS}
                for seed, result in by_seed.items()
            }
            for mixer, by_seed in probes.items()
        },
        "training": _training_summary(training, _test_totals(training)),
        "v1_reference": _v1_reference(cfg.ROOT / "results" / "probe.json"),
        "notes": (
            f"Probe split: fit on the position's year <= {cfg.TRAIN_YEAR_MAX}, "
            f"score on year > {cfg.VAL_YEAR_MAX} (as v1). Bootstrap: "
            f"{cfg.BOOTSTRAP_RESAMPLES} paired resamples of test positions shared "
            "across attention seeds, rng seed 0."
        ),
    }
    path = cfg.RESULTS_V2_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _print_summary(path, report)
    return report


def _training_summary(training: dict, totals: dict[str, list[float]]) -> dict:
    runs = [
        {
            "mixer": run["mixer"],
            "seed": run["seed"],
            "best_epoch": run.get("best_epoch"),
            "val": run.get("val"),
            "test": run.get("test"),
        }
        for run in training.get("runs", [])
    ]
    per_mixer = {
        mixer: {
            "mean_test_total": float(np.mean(values)),
            "std_test_total": float(np.std(values, ddof=0)),
            "n_seeds": len(values),
        }
        for mixer, values in totals.items()
    }
    return {"runs": runs, "per_mixer": per_mixer}


def _v1_reference(path) -> dict | None:
    if not path.is_file():
        return None
    v1 = json.loads(path.read_text(encoding="utf-8"))
    return {key: v1.get(key) for key in _V1_KEYS}


def _print_summary(path, report: dict) -> None:
    decision = report["decision"]
    c1 = decision["attention_beats_mean"]
    c2 = decision["genetics_lift"]
    print(f"wrote {path}")
    print(
        f"criterion 1 attention<mean: attn={c1['mean_test_total_attention']:.4f} "
        f"mean={c1['mean_test_total_mean']:.4f} gap={c1['gap']:.4f} "
        f"threshold={c1['threshold']:.4f} pass={c1['pass']}"
    )
    print(
        f"criterion 2 lift: {c2['lift']:.4f} [{c2['lo']:.4f}, {c2['hi']:.4f}] "
        f"bar={c2['lift_bar']} pass={c2['pass']}"
    )
    for mixer, by_seed in report["probes"].items():
        for seed, result in by_seed.items():
            print(
                f"  {mixer} s{seed}: frozen={result['frozen_auroc']:.4f} "
                f"contextual={result['contextual_auroc']:.4f} lift={result['lift']:.4f}"
            )
    print(f"fruitful={decision['fruitful']}")


if __name__ == "__main__":
    main()
