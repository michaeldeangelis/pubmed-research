"""Agreement between two independent claim-level labelings."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.config import ROOT

STATUSES = ("support", "contradict", "mixed", "unresolved")


def cohens_kappa(left: list[str], right: list[str]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("label lists must be the same non-empty length")
    n = len(left)
    agree = sum(a == b for a, b in zip(left, right)) / n
    counts_l = Counter(left)
    counts_r = Counter(right)
    expected = sum(counts_l[status] / n * counts_r[status] / n for status in STATUSES)
    if expected == 1:
        return 1.0
    return (agree - expected) / (1 - expected)


def claim_status(labels: list[dict]) -> str:
    """Collapse follow-up labels. Reviews and non-tests do not count as outcomes."""
    direct = [
        row["relation"]
        for row in labels
        if row.get("independent") and row.get("relation") in ("support", "contradict")
    ]
    has_support = "support" in direct
    has_contradict = "contradict" in direct
    if has_support and has_contradict:
        return "mixed"
    if has_support:
        return "support"
    if has_contradict:
        return "contradict"
    return "unresolved"


def _status(row: dict) -> str:
    if row.get("followups"):
        return claim_status(row["followups"])
    return row["status"]


def main() -> None:
    path_a = ROOT / "bench" / "labels_a.json"
    path_b = ROOT / "bench" / "labels_b.json"
    left = json.loads(path_a.read_text(encoding="utf-8"))["claims"]
    right = json.loads(path_b.read_text(encoding="utf-8"))["claims"]
    by_b = {row["claim_id"]: row for row in right}
    ids = [row["claim_id"] for row in left]
    pair_l = [_status(row) for row in left]
    pair_r = [_status(by_b[claim_id]) for claim_id in ids]
    report = {
        "n": len(ids),
        "kappa": cohens_kappa(pair_l, pair_r),
        "agreement": sum(a == b for a, b in zip(pair_l, pair_r)) / len(ids),
        "a": dict(Counter(pair_l)),
        "b": dict(Counter(pair_r)),
    }
    out = ROOT / "bench" / "agreement.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
