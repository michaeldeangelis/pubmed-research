"""Run the held-out history comparison and write results/history.json."""

from __future__ import annotations

import json

from src.config import ROOT
from src.eval.history import run_history_comparison


def main() -> None:
    report = run_history_comparison()
    path = ROOT / "results" / "history.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(path)
    for name, block in report["models"].items():
        interval = _interval(block["clinical_auroc"])
        print(f"{name:28} auroc {block['clinical_auroc']['point']:.3f} {interval} bce {block['clinical_bce']:.3f}")
    for name, block in report["deltas_clinical_auroc"].items():
        print(f"delta {name:44} {block['point']:+.3f} {_interval(block)}")


def _interval(block: dict) -> str:
    if block["lo"] is None or block["hi"] is None:
        return "[na, na]"
    return f"[{block['lo']:+.3f}, {block['hi']:+.3f}]"


if __name__ == "__main__":
    main()
