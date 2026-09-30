"""Metascience v4: design against topic. See experiments.md, "2026-09-30 metascience v4".

python -m src.meta.analysis_matched

Matched-pair (McNemar) ORs for human_samples against topic-matched cell_only
papers, beside the crude ORs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from scipy.stats import beta

from src.meta import analysis, analysis_trials

MATCH_DIR = analysis.META / "matching"
RESULT_PATH = analysis.ROOT / "results" / "meta_matched.json"
OUTCOMES = ("clin_cited_8y", "trial_bg_8y", "trial_bg_8y_drug")


def _load_outcomes(pathway_dirs: dict[str, Path], trials_dir: Path) -> dict[str, dict]:
    """pmid -> {outcome: 0/1, "parity": 0/1}, for every paper in any pathway."""
    out: dict[str, dict] = {}
    for name, meta_dir in pathway_dirs.items():
        clin = {str(r["pmid"]): int(r["clin_cited_8y"]) for r in analysis._read_jsonl(meta_dir / "outcome_b.jsonl")}
        for r in analysis._read_jsonl(trials_dir / f"outcome_{name}.jsonl"):
            pmid = str(r["pmid"])
            if pmid in out or pmid not in clin:
                continue
            out[pmid] = {"clin_cited_8y": clin[pmid], "trial_bg_8y": int(r["trial_bg_8y"]),
                         "trial_bg_8y_drug": int(r["trial_bg_8y_drug"]), "parity": int(pmid) % 2 == 0}
    return out


def matched_or(b: int, c: int) -> dict:
    """McNemar OR = b/c with an exact Clopper-Pearson CI on b/(b+c)."""
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "or": None, "lo": None, "hi": None}
    lo_p = beta.ppf(0.025, b, c + 1) if b > 0 else 0.0
    hi_p = beta.ppf(0.975, b + 1, c) if c > 0 else 1.0

    def to_or(p: float) -> float:
        return math.inf if p >= 1 else p / (1 - p)

    return {"b": b, "c": c, "or": (b / c) if c else math.inf, "lo": to_or(lo_p), "hi": to_or(hi_p)}


def crude_or(treated: list[int], control: list[int]) -> dict:
    """2x2 OR with a Woolf CI (0.5 added to every cell only if one is zero)."""
    a, b = sum(treated), len(treated) - sum(treated)
    c, d = sum(control), len(control) - sum(control)
    if 0 in (a, b, c, d):
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    log_or = math.log((a * d) / (b * c))
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return {"or": math.exp(log_or), "lo": math.exp(log_or - 1.96 * se), "hi": math.exp(log_or + 1.96 * se),
            "n_treated": len(treated), "n_control": len(control)}


def verdict(m: dict) -> str:
    if m["lo"] is None:
        return "no discordant pairs"
    if m["lo"] > 1:
        return "persists"
    if m["hi"] < 1:
        return "reversed"
    return "attenuated to null"


def _attenuation(matched: float | None, crude: float) -> float | None:
    if not matched or matched in (math.inf, 0) or crude == 1:
        return None
    return 1 - math.log(matched) / math.log(crude)


def _pair_counts(pairs: list[dict], outcomes: dict[str, dict], key: str) -> tuple[int, int]:
    b = c = 0
    for p in pairs:
        t, k = outcomes[p["treated_pmid"]][key], outcomes[p["control_pmid"]][key]
        b += int(t and not k)
        c += int(k and not t)
    return b, c


def run(match_dir: Path = MATCH_DIR, pathway_dirs: dict[str, Path] = analysis_trials.PATHWAY_DIRS,
        trials_dir: Path = analysis_trials.TRIALS_DIR) -> dict:
    pairs = analysis._read_jsonl(match_dir / "pairs.jsonl")
    for p in pairs:
        p["treated_pmid"], p["control_pmid"] = str(p["treated_pmid"]), str(p["control_pmid"])
    outcomes = _load_outcomes(pathway_dirs, trials_dir)
    n_listed = len(pairs)
    pairs = [p for p in pairs if p["treated_pmid"] in outcomes and p["control_pmid"] in outcomes]
    pool = [r for r in analysis_trials.load_pooled(pathway_dirs, trials_dir)
            if r["model_system"] in ("human_samples", "cell_only") and r["pmid"] in outcomes]
    refs = {r["pmid"]: r["n_refs"] for r in pool}
    matched_treated = [p["treated_pmid"] for p in pairs]
    matched_control = [p["control_pmid"] for p in pairs]
    report: dict = {"ledger": "experiments.md, 2026-09-30 metascience v4", "n_pairs": len(pairs),
                    "n_pairs_dropped_no_outcome": n_listed - len(pairs), "outcomes": {}}
    for key in OUTCOMES + ("parity",):
        b, c = _pair_counts(pairs, outcomes, key)
        m = matched_or(b, c)
        crude = crude_or([outcomes[r["pmid"]][key] for r in pool if r["model_system"] == "human_samples"],
                         [outcomes[r["pmid"]][key] for r in pool if r["model_system"] == "cell_only"])
        unpaired = crude_or([outcomes[x][key] for x in matched_treated], [outcomes[x][key] for x in matched_control])
        report["outcomes"][key] = {
            "matched": m, "crude_all": crude, "crude_matched_unpaired": unpaired,
            "attenuation_vs_crude": _attenuation(m["or"], crude["or"]),
            "verdict": verdict(m) if key != "parity" else ("placebo ok" if m["lo"] is not None and m["lo"] <= 1 <= m["hi"] else "placebo FAILED"),
        }
    close = [p for p in pairs if refs.get(p["treated_pmid"]) is not None and refs.get(p["control_pmid"]) is not None
             and abs(math.log1p(refs[p["treated_pmid"]]) - math.log1p(refs[p["control_pmid"]])) <= 0.5]
    report["refs_balanced_pairs"] = {"n_pairs": len(close),
                                     **{key: matched_or(*_pair_counts(close, outcomes, key)) for key in OUTCOMES}}
    report["per_pathway"] = {}
    for name in pathway_dirs:
        sub = [p for p in pairs if p.get("pathway") == name]
        report["per_pathway"][name] = {
            "n_pairs": len(sub),
            **{key: matched_or(*_pair_counts(sub, outcomes, key)) for key in OUTCOMES},
        }
    return report


def main() -> None:
    report = run()
    RESULT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"n_pairs": report["n_pairs"], **{k: {"verdict": v["verdict"], "matched": v["matched"],
                                                         "crude": v["crude_all"]["or"]}
                                                     for k, v in report["outcomes"].items()}}, indent=2))


if __name__ == "__main__":
    main()
