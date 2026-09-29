"""Outcome A reference check: Reproducibility Project: Cancer Biology.

Run as ``python -m src.meta.rpcb``. Writes ``results/meta_rpcb.json``.

Data: RP:CB final analysis tables (Errington et al. 2021, eLife), OSF project
e5nvr, licence CC-BY 4.0. Downloaded into ``data/meta/rpcb/`` if missing.

Unit and rows
-------------
* Effects whose original claim was positive (``Expected difference based on
  the original paper? == Positive``), one row per effect
  (Paper #, Experiment #, Effect #).
* Internal replications (the same effect replicated more than once, e.g. in
  several cell lines; ``Internal replication # > 1``) are combined into one
  replication estimate by a fixed-effect inverse-variance meta-analysis of
  the internal replications' SMD effect sizes and standard errors. The
  dataset's own ``Meta-analysis ... (SMD)`` columns pool the ORIGINAL with
  the replications, so they are used only for the extra criterion
  ``meta_original_plus_replication_significant`` (they are identical
  across the rows of an effect). Internal replications without numeric
  data (paper 9, experiment 1) are combined by majority vote of the
  per-row categorical outcome.

Criteria (each a 0/1 per effect, None when not computable)
----------------------------------------------------------
1. ``same_direction``: replication effect in the original direction
   (categorical ``Positive`` or ``Null-positive``; ``Null``, i.e. no
   difference and no test, counts as failure).
2. ``same_direction_p05``: same direction and p < 0.05 (categorical
   ``Positive``).
   For 1-2 the SMD-based categorical column is used when the effect has an
   SMD replication estimate, otherwise the column based on the original
   test. For combined internal replications the pooled estimate's side of
   the null and its two-sided p are used.
3. ``original_in_replication_ci``: original effect size inside the
   replication 95% CI.
4. ``replication_in_original_ci``: replication effect size inside the
   original 95% CI.
   3-4 use the SMD columns (all on the same scale per effect: SMD, or HR,
   or r where no SMD conversion exists).
5. ``meta_original_plus_replication_significant`` (extra): the dataset's
   fixed-effect meta-analysis of original + replication is in the original
   direction with p < 0.05.

Rates carry Wilson 95% CIs. Spearman correlations of the original SMD
(effect-size types Cohen's d, Cohen's dz, Glass' delta only; HR and r are on
other scales) with each success criterion are reported with Fisher-z 95%
CIs. Results are also split by experiment type (Animal, Cell-based, other =
Patient Samples + Recombinant).

The 23 papers with effect data are mapped to PMIDs (lead-supplied list),
verified against Europe PMC titles, and for each paper we record whether
RP:CB coded any experiment of type Animal, both among the replicated
experiments (with effects) and among all experiments RP:CB identified.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import logging
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests

from src import config

logger = logging.getLogger(__name__)

RPCB_DIR = config.DATA / "meta" / "rpcb"
RESULT_PATH = Path(__file__).resolve().parents[2] / "results" / "meta_rpcb.json"

OSF_FILES = {
    "effect.csv": "https://osf.io/download/39s7j/",
    "experiment.csv": "https://osf.io/download/fv6te/",
    "paper.csv": "https://osf.io/download/aydj2/",
    "effect_dict.csv": "https://osf.io/download/afexp/",
    "experiment_dict.csv": "https://osf.io/download/c2rfy/",
    "paper_dict.csv": "https://osf.io/download/wx6be/",
}
EPMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

# Lead-supplied mapping, with one correction found by the title check:
# paper 5 (Ricci-Vitiani et al., "Tumour vascularization via endothelial
# differentiation of glioblastoma stem-like cells", Nature 2010) is PMID
# 21102434. 21102433 is the companion paper by Wang et al. in the same issue
# ("Glioblastoma stem-like cells give rise to tumour endothelium").
PMID_CORRECTIONS = {"5": {"supplied": "21102433", "corrected": "21102434",
                          "reason": "21102433 is Wang et al., Nature 2010, the companion paper; "
                                    "Europe PMC title of 21102434 matches the RP:CB title exactly"}}
PAPER_PMIDS = {
    "1": "20577206", "5": "21102434", "6": "20668451", "7": "20141835",
    "8": "20130576", "9": "20418870", "12": "21107320", "15": "20378772",
    "16": "20171147", "19": "21889194", "20": "21729786", "21": "21849665",
    "24": "22000013", "28": "21240262", "29": "21964340", "37": "22460902",
    "39": "22451913", "41": "22903521", "42": "22635005", "44": "22622578",
    "47": "22343901", "48": "23021215", "50": "22009989",
}

SMD_TYPES = {"Cohen's d", "Cohen's dz", "Glass' delta"}
CRITERIA = ["same_direction", "same_direction_p05", "original_in_replication_ci",
            "replication_in_original_ci", "meta_original_plus_replication_significant"]
ATTRIBUTION = (
    "Reproducibility Project: Cancer Biology data (Errington TM, Mathur M, Soderberg CK, "
    "Denis A, Perfito N, Iorns E, Nosek BA. Investigating the replicability of preclinical "
    "cancer biology. eLife 2021;10:e71601. doi:10.7554/eLife.71601), OSF project "
    "https://osf.io/e5nvr/, licensed CC-BY 4.0 (https://creativecommons.org/licenses/by/4.0/). "
    "Derived statistics computed by this project; no changes to the source data."
)

KEY = ("Paper #", "Experiment #", "Effect #")
EXPECTED = "Expected difference based on the original paper?"
OBS = "Observed difference in replication?"
OBS_SMD = "Observed difference in replication? (SMD)"


# ---------------------------------------------------------------- stats


def wilson(k: int, n: int, z: float = 1.959963984540054) -> dict:
    if n == 0:
        return {"k": 0, "n": 0, "rate": None, "ci95": [None, None]}
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"k": k, "n": n, "rate": p, "ci95": [max(0.0, centre - half), min(1.0, centre + half)]}


def spearman(x: list[float], y: list[float]) -> dict:
    n = len(x)
    if n < 4 or len(set(y)) < 2 or len(set(x)) < 2:
        return {"n": n, "rho": None, "p": None, "ci95": [None, None]}
    from scipy.stats import spearmanr
    res = spearmanr(x, y)
    rho = float(res.correlation)
    se = 1.06 / math.sqrt(n - 3)  # Fieller et al. (1957)
    zr = math.atanh(max(min(rho, 0.999999), -0.999999))
    return {"n": n, "rho": rho, "p": float(res.pvalue),
            "ci95": [math.tanh(zr - 1.96 * se), math.tanh(zr + 1.96 * se)]}


def _num(v) -> float | None:
    if v is None:
        return None
    v = str(v).strip()
    if v in ("", "NA", "Unknown", "nan"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def fixed_effect(ests: list[float], ses: list[float]) -> tuple[float, float]:
    w = [1 / (s * s) for s in ses]
    est = sum(wi * e for wi, e in zip(w, ests)) / sum(w)
    return est, math.sqrt(1 / sum(w))


def _two_sided_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2))


# ---------------------------------------------------------------- effect logic


def null_value(es_type: str) -> float:
    return 1.0 if es_type == "Hazard ratio" else 0.0


def _side(x: float, null: float) -> int:
    return (x > null) - (x < null)


def _cat_success(cat: str) -> tuple[int, int]:
    """(same_direction, same_direction_p05) from an RP:CB categorical code."""
    cat = (cat or "").strip()
    return int(cat in ("Positive", "Null-positive")), int(cat == "Positive")


def _inside(x, lo, hi):
    if x is None or lo is None or hi is None:
        return None
    return int(min(lo, hi) <= x <= max(lo, hi))


def evaluate_effect(rows: list[dict]) -> dict:
    """Collapse one effect's rows (internal replications) and score it."""
    first = rows[0]
    es_type = (first.get("Effect size type (SMD)") or "").strip()
    null = null_value(es_type)
    orig = _num(first.get("Original effect size (SMD)"))
    orig_lo = _num(first.get("Original lower CI (SMD)"))
    orig_hi = _num(first.get("Original upper CI (SMD)"))
    reps = [(_num(r.get("Replication effect size (SMD)")), _num(r.get("Replication standard error (SMD)")),
             _num(r.get("Replication lower CI (SMD)")), _num(r.get("Replication upper CI (SMD)")),
             _num(r.get("Replication p value (SMD)"))) for r in rows]
    out: dict = {"paper": first["Paper #"], "experiment": first["Experiment #"],
                 "effect": first["Effect #"], "n_internal_replications": len(rows),
                 "es_type_smd": es_type or None, "original_smd": orig}

    if len(rows) == 1:
        rep, _, rep_lo, rep_hi, _ = reps[0]
        if rep is not None:
            cat, basis = rows[0].get(OBS_SMD), "smd_categorical"
        else:
            cat, basis = rows[0].get(OBS), "original_test_categorical"
        sd, sig = _cat_success(cat)
        out.update({"replication_smd": rep, "basis": basis,
                    "same_direction": sd, "same_direction_p05": sig})
    elif es_type in SMD_TYPES and all(e is not None and s is not None and s > 0 for e, s, *_ in reps):
        est, se = fixed_effect([e for e, s, *_ in reps], [s for e, s, *_ in reps])
        rep_lo, rep_hi = est - 1.959963984540054 * se, est + 1.959963984540054 * se
        rep = est
        ref_side = _side(orig, null) if orig is not None else 1
        same = int(_side(est, null) == ref_side)
        p = _two_sided_p((est - null) / se)
        out.update({"replication_smd": est, "basis": "fixed_effect_internal_replications",
                    "same_direction": same, "same_direction_p05": int(same and p < 0.05)})
    else:
        votes = [_cat_success(r.get(OBS_SMD) or r.get(OBS)) for r in rows]
        rep = rep_lo = rep_hi = None
        out.update({"replication_smd": None, "basis": "majority_vote_internal_replications",
                    "same_direction": int(sum(v[0] for v in votes) * 2 > len(votes)),
                    "same_direction_p05": int(sum(v[1] for v in votes) * 2 > len(votes))})

    out["original_in_replication_ci"] = _inside(orig, rep_lo, rep_hi)
    out["replication_in_original_ci"] = _inside(rep, orig_lo, orig_hi)
    meta = _num(first.get("Meta-analysis effect size (SMD)"))
    meta_p = _num(first.get("Meta-analysis p value (SMD)"))
    if meta is None or meta_p is None or orig is None:
        out["meta_original_plus_replication_significant"] = None
    else:
        out["meta_original_plus_replication_significant"] = int(
            _side(meta, null) == _side(orig, null) and meta_p < 0.05)
    return out


def load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as fh:
        return list(csv.DictReader(fh))


def experiment_type_group(t: str) -> str:
    t = (t or "").strip()
    if t == "Animal":
        return "animal"
    if t == "Cell-based":
        return "cell_based"
    return "other"


def score_effects(effects: list[dict], experiments: list[dict]) -> list[dict]:
    exp_type = {(r["Paper #"], r["Experiment #"]): r.get("Type of experiment", "") for r in experiments}
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in effects:
        if (r.get(EXPECTED) or "").strip() != "Positive":
            continue
        groups[tuple(r[k] for k in KEY)].append(r)
    scored = []
    for key in sorted(groups, key=lambda k: tuple(int(x) for x in k)):
        rows = sorted(groups[key], key=lambda r: int(_num(r.get("Internal replication #")) or 1))
        s = evaluate_effect(rows)
        s["experiment_type"] = exp_type.get(key[:2], "")
        s["type_group"] = experiment_type_group(s["experiment_type"])
        scored.append(s)
    return scored


def summarize(scored: list[dict]) -> dict:
    def rates(items):
        out = {}
        for c in CRITERIA:
            vals = [s[c] for s in items if s[c] is not None]
            out[c] = wilson(sum(vals), len(vals))
        return out

    by_type = {g: {"n_effects": sum(1 for s in scored if s["type_group"] == g),
                   **rates([s for s in scored if s["type_group"] == g])}
               for g in ("animal", "cell_based", "other")}
    smd = [s for s in scored if s["es_type_smd"] in SMD_TYPES and s["original_smd"] is not None]
    corr = {}
    for c in CRITERIA:
        pts = [(s["original_smd"], s[c]) for s in smd if s[c] is not None]
        corr[c] = spearman([p[0] for p in pts], [p[1] for p in pts])
    return {"n_effects": len(scored), "overall": rates(scored), "by_experiment_type": by_type,
            "spearman_original_smd_vs_success": corr,
            "basis_counts": dict(Counter(s["basis"] for s in scored))}


# ---------------------------------------------------------------- papers / PMIDs


def _norm_title(t: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (t or "").lower())


def title_match(a: str, b: str) -> float:
    ta, tb = set(_norm_title(a)), set(_norm_title(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def fetch_epmc_titles(pmids: list[str], cache_path: Path) -> dict[str, str]:
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    for pm in pmids:
        if pm in cache:
            continue
        resp = requests.get(EPMC_URL, params={"query": f"EXT_ID:{pm} AND SRC:MED",
                                              "format": "json", "resultType": "lite"},
                            timeout=60)
        resp.raise_for_status()
        res = resp.json().get("resultList", {}).get("result", [])
        cache[pm] = res[0].get("title", "") if res else ""
        time.sleep(0.2)
    cache_path.write_text(json.dumps(cache, indent=1))
    return cache


def paper_table(papers: list[dict], experiments: list[dict], effects: list[dict],
                titles: dict[str, str]) -> list[dict]:
    with_effects = {r["Paper #"] for r in effects}
    exp_with_effects = {(r["Paper #"], r["Experiment #"]) for r in effects}
    out = []
    for p in sorted(papers, key=lambda r: int(r["Paper #"])):
        num = p["Paper #"]
        if num not in with_effects:
            continue
        exps = [e for e in experiments if e["Paper #"] == num]
        rep = [e for e in exps if (num, e["Experiment #"]) in exp_with_effects]
        pm = PAPER_PMIDS.get(num)
        epmc = titles.get(pm, "") if pm else ""
        sim = title_match(p.get("Original paper title", ""), epmc) if pm else None
        out.append({
            "paper": num, "pmid": pm, "year": p.get("Year"),
            "journal": p.get("Original paper journal"),
            "title": p.get("Original paper title"),
            "europepmc_title": epmc or None,
            "title_similarity": sim,
            "pmid_verified": bool(sim is not None and sim >= 0.6),
            "rpcb_any_animal_replicated": any(e.get("Type of experiment") == "Animal" for e in rep),
            "rpcb_any_animal_identified": any(e.get("Type of experiment") == "Animal" for e in exps),
            "experiment_types_replicated": sorted({e.get("Type of experiment", "") for e in rep}),
            "n_experiments_replicated": len(rep),
        })
    return out


# ---------------------------------------------------------------- main


def ensure_data(rpcb_dir: Path = RPCB_DIR) -> None:
    rpcb_dir.mkdir(parents=True, exist_ok=True)
    for name, url in OSF_FILES.items():
        path = rpcb_dir / name
        if path.exists() and path.stat().st_size > 0:
            continue
        logger.info("downloading %s", url)
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        path.write_bytes(resp.content)


def main() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ensure_data()
    effects = load_csv(RPCB_DIR / "effect.csv")
    experiments = load_csv(RPCB_DIR / "experiment.csv")
    papers = load_csv(RPCB_DIR / "paper.csv")
    scored = score_effects(effects, experiments)
    summary = summarize(scored)
    titles = fetch_epmc_titles(sorted(PAPER_PMIDS.values()), RPCB_DIR / "pmid_titles.json")
    ptable = paper_table(papers, experiments, effects, titles)
    all_keys = {tuple(r[k] for k in KEY) for r in effects}
    result = {
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "source": {"osf_project": "https://osf.io/e5nvr/", "files": OSF_FILES,
                   "license": "CC-BY 4.0", "attribution": ATTRIBUTION},
        "counts": {"effect_rows": len(effects), "unique_effects": len(all_keys),
                   "unique_positive_effects": summary["n_effects"],
                   "experiments_with_effects": len({k[:2] for k in all_keys}),
                   "papers_with_effects": len({k[0] for k in all_keys})},
        "method": {
            "rows": "effects with Expected difference == Positive, one per (paper, experiment, effect)",
            "internal_replications": "pooled by fixed-effect inverse-variance meta-analysis of the internal replications' SMD (the dataset's Meta-analysis columns include the original and are used only for criterion 5); paper 9 experiment 1 has no numeric data and is combined by majority vote of per-row codes",
            "direction_and_significance": "RP:CB categorical codes: SMD-based column when the effect has a replication SMD, else the original-test column; 'Null' (no difference, no test) counts as failure",
            "ci_criteria": "SMD columns (per-effect scale: SMD, or HR / r where RP:CB did not convert)",
            "spearman": "original SMD (Cohen's d, dz, Glass' delta only) vs 0/1 success, Fisher-z CI with SE 1.06/sqrt(n-3)",
            "intervals": "Wilson 95% for rates",
            "type_groups": {"animal": "Animal", "cell_based": "Cell-based", "other": "Patient Samples, Recombinant"},
        },
        "summary": summary,
        "papers": ptable,
        "pmid_check": {"n_papers": len(ptable), "n_verified_by_title": sum(p["pmid_verified"] for p in ptable),
                       "rule": "Europe PMC title for the PMID vs RP:CB title, token Jaccard >= 0.6",
                       "corrections": PMID_CORRECTIONS},
        "effects": scored,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2))
    logger.info("wrote %s", RESULT_PATH)
    return result


if __name__ == "__main__":
    main()
