"""Metascience v3: trial background citations. See experiments.md, "2026-09-30 metascience v3".

python -m src.meta.analysis_trials

Pooled logit over RAS, EGFR and PI3K eligible preclinical papers. Gene groups
are coded per pathway ("egfr:ERBB2") against one reference ("ras:KRAS"), so
the gene dummies also absorb the pathway fixed effect.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from src.meta import analysis

TRIALS_DIR = analysis.META / "trials"
RESULT_PATH = analysis.ROOT / "results" / "meta_trials.json"
PATHWAY_DIRS = {"ras": analysis.META, "egfr": analysis.META / "pathways" / "egfr",
                "pi3k": analysis.META / "pathways" / "pi3k"}
REFERENCE_GENE = "ras:KRAS"
DROP = ("human_genetics", "multi_system")
TERM = "model_system=human_samples"
MIN_EVENTS = 100
OUTCOMES = ("trial_bg_8y", "trial_bg_8y_drug", "trial_bg_8y_ph2")


def load_pooled(pathway_dirs: dict[str, Path] = PATHWAY_DIRS, trials_dir: Path = TRIALS_DIR) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for name, meta_dir in pathway_dirs.items():
        papers = {str(r["pmid"]): r for r in analysis._read_jsonl(meta_dir / "papers.jsonl")}
        impact = {str(r["pmid"]): r for r in analysis._read_jsonl(meta_dir / "impact.jsonl")}
        outcome = {str(r["pmid"]): r for r in analysis._read_jsonl(trials_dir / f"outcome_{name}.jsonl")}
        for f in analysis._read_jsonl(meta_dir / "features.jsonl"):
            pmid = str(f["pmid"])
            if f.get("group") != "preclinical" or not f.get("eligible", True):
                continue
            if pmid in seen or pmid not in outcome or pmid not in papers:
                continue
            seen.add(pmid)
            extra = impact.get(pmid, {})
            if name == "ras":
                refs = analysis._reference_count(extra)
            else:
                value = extra.get("n_refs_icite")
                refs = int(value) if value else None
            row = {**f, "pmid": pmid, "pathway": name, "year": int(papers[pmid]["year"]),
                   "gene_group": f"{name}:{f['gene_group']}", "n_refs": refs}
            for key in OUTCOMES:
                row[key] = int(outcome[pmid][key])
            rows.append(row)
    return rows


def _with_outcome(rows: list[dict], key: str) -> list[dict]:
    # analysis.odds_ratios reads clin_cited_8y; point it at the trial outcome.
    return [{**r, "clin_cited_8y": r[key]} for r in rows]


def _gene_levels(rows: list[dict], reference: str) -> tuple[str, ...]:
    return tuple(sorted({r["gene_group"] for r in rows} - {reference}))


def _fit(rows: list[dict], key: str, rng: np.random.Generator, reference: str) -> dict:
    saved = analysis.GENE_LEVELS
    analysis.GENE_LEVELS = _gene_levels(rows, reference)
    try:
        return analysis.odds_ratios(_with_outcome(rows, key), DROP, placebo=True, rng=rng)
    finally:
        analysis.GENE_LEVELS = saved


def permutation(rows: list[dict], rng: np.random.Generator) -> dict:
    """model_system ORs with model_system shuffled within year x pathway."""
    saved = analysis.GENE_LEVELS
    analysis.GENE_LEVELS = _gene_levels(rows, REFERENCE_GENE)
    try:
        base = _with_outcome(rows, "trial_bg_8y")
        years = sorted({r["year"] for r in base})
        y = np.array([r["clin_cited_8y"] for r in base], float)
        cells: dict[tuple, list[int]] = {}
        for i, r in enumerate(base):
            cells.setdefault((r["year"], r["pathway"]), []).append(i)
        draws: dict[str, list[float]] = {}
        for _ in range(analysis.N_PERM):
            systems = [r["model_system"] for r in base]
            for idx in cells.values():
                shuffled = list(rng.permutation([systems[i] for i in idx]))
                for i, value in zip(idx, shuffled):
                    systems[i] = value
            x, names = analysis.design([{**r, "model_system": s} for r, s in zip(base, systems)], years, DROP)
            beta = analysis.fit_logit(x, y)
            for j, name in enumerate(names):
                if name.startswith("model_system"):
                    draws.setdefault(name, []).append(math.exp(beta[j]))
    finally:
        analysis.GENE_LEVELS = saved
    return {k: analysis._ci(np.array(v)) for k, v in draws.items()}


def verdict(term: dict | None, placebo: dict, events: int) -> str:
    if events < MIN_EVENTS:
        return "underpowered"
    if not (placebo["lo"] <= 1 <= placebo["hi"]):
        return "unreliable (placebo)"
    if term and term["lo"] > 1:
        return "same direction"
    if term and term["hi"] < 1:
        return "opposite"
    return "null"


def run(pathway_dirs: dict[str, Path] = PATHWAY_DIRS, trials_dir: Path = TRIALS_DIR) -> dict:
    rows = load_pooled(pathway_dirs, trials_dir)
    rng = np.random.default_rng(0)
    events = sum(r["trial_bg_8y"] for r in rows if r["model_system"] in ("cell_only", "human_samples"))
    primary = _fit(rows, "trial_bg_8y", rng, REFERENCE_GENE)
    out = {
        "ledger": "experiments.md, 2026-09-30 metascience v3",
        "n": primary["n"],
        "events_total": int(sum(r["trial_bg_8y"] for r in rows)),
        "events_cell_or_human": int(events),
        "primary": primary,
        "verdict": verdict(primary["terms"].get(TERM), primary["terms"]["placebo_pmid_even"], events),
        "exploratory": {
            "permutation_95": {k: list(v) for k, v in permutation(rows, rng).items()},
            "drug_only": _fit(rows, "trial_bg_8y_drug", rng, REFERENCE_GENE),
            "phase2_plus": _fit(rows, "trial_bg_8y_ph2", rng, REFERENCE_GENE),
            "per_pathway": {},
        },
    }
    for name in pathway_dirs:
        subset = [r for r in rows if r["pathway"] == name]
        reference = min(r["gene_group"] for r in subset if r["gene_group"].endswith((":KRAS", ":EGFR", ":PIK3CA")))
        out["exploratory"]["per_pathway"][name] = {
            "events": int(sum(r["trial_bg_8y"] for r in subset)),
            "fit": _fit(subset, "trial_bg_8y", rng, reference),
        }
    return out


def main() -> None:
    report = run()
    RESULT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("n", "events_total", "events_cell_or_human", "verdict")}
                     | {TERM: report["primary"]["terms"].get(TERM)}, indent=2))


if __name__ == "__main__":
    main()
