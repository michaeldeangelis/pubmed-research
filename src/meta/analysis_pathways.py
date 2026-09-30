"""Metascience v2: the v1 model_system test in other pathways.

python -m src.meta.analysis_pathways

See experiments.md, "2026-09-30 metascience v2: other pathways". Each pathway
is fit on its full 2000-2015 corpus. The test is model_system=human_samples
against cell_only, with the placebo check. Everything else is exploratory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.meta import analysis

PATHWAY_DIR = analysis.META / "pathways"
RESULT_PATH = analysis.ROOT / "results" / "meta_pathways.json"
# The reference gene group is the first listed for each pathway in the ledger.
GENE_REFERENCE = {"egfr": "EGFR", "pi3k": "PIK3CA"}
DROP = ("human_genetics", "multi_system")
TERM = "model_system=human_samples"


def _icite_reference_counts(rows: list[dict], meta_dir: Path) -> None:
    """Use the iCite reference count for every row (amendment, 2026-09-30).

    OpenAlex credits ran out partway through EGFR 2015, so its counts mix two
    sources inside one year. iCite covers every paper in both pathways.
    """
    icite = {str(r["pmid"]): r.get("n_refs_icite") for r in analysis._read_jsonl(meta_dir / "impact.jsonl")}
    for row in rows:
        value = icite.get(row["pmid"])
        row["n_refs"] = int(value) if value else None


def run_pathway(name: str, meta_dir: Path) -> dict:
    rows = analysis.load_rows(meta_dir)
    _icite_reference_counts(rows, meta_dir)
    levels = sorted({r["gene_group"] for r in rows} - {GENE_REFERENCE[name]})
    saved = analysis.GENE_LEVELS
    analysis.GENE_LEVELS = tuple(levels)
    try:
        rng = np.random.default_rng(0)
        fit = analysis.odds_ratios(rows, DROP, placebo=True, rng=rng)
        perm = analysis.permutation_ors(rows, DROP, rng)
        papers = analysis._read_jsonl(meta_dir / "papers.jsonl")
        journals = {str(r["pmid"]): str(r.get("journal") or "") for r in papers}
        secondary = analysis.secondary(rows, DROP, rng, journals)
    finally:
        analysis.GENE_LEVELS = saved
    term = fit["terms"].get(TERM)
    placebo = fit["terms"]["placebo_pmid_even"]
    placebo_ok = placebo["lo"] <= 1 <= placebo["hi"]
    confirmed = bool(term and term["or"] > 1 and term["lo"] > 1 and placebo_ok)
    return {
        "n": fit["n"],
        "prevalence": fit["prevalence"],
        "gene_reference": GENE_REFERENCE[name],
        "gene_levels": levels,
        "terms": fit["terms"],
        "permutation_95": {k: list(v) for k, v in perm.items()},
        "placebo_ok": bool(placebo_ok),
        "confirmed": confirmed,
        "secondary": secondary,
    }


def run(base: Path = PATHWAY_DIR, names: tuple[str, ...] = ("egfr", "pi3k")) -> dict:
    per = {name: run_pathway(name, base / name) for name in names}
    confirmed = [name for name, r in per.items() if r["confirmed"]]
    verdict = "confirmed in both" if len(confirmed) == len(names) else (
        f"confirmed in {', '.join(confirmed)} only" if confirmed else "kill")
    return {"ledger": "experiments.md, 2026-09-30 metascience v2", "pathways": per,
            "confirmed": confirmed, "verdict": verdict}


def main() -> None:
    report = run()
    RESULT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": report["verdict"], **{
        k: {"n": v["n"], TERM: v["terms"].get(TERM), "placebo_ok": v["placebo_ok"]}
        for k, v in report["pathways"].items()}}, indent=2))


if __name__ == "__main__":
    main()
