"""Preregistered metascience analysis. See experiments.md, "2026-09-29 metascience v1".

python -m src.meta.analysis [--drop FEATURE ...]

This is the only code that joins design features to outcomes. It runs once,
after the plan and the measurement gate are committed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from src.config import DATA, ROOT
from src.probe.probes import auroc

import torch

META = DATA / "meta"
RESULT_PATH = ROOT / "results" / "meta_analysis.json"
DEV_YEARS = (2000, 2011)
CONFIRM_YEARS = (2012, 2015)
N_BOOT = 1000
N_PERM = 1000
MODEL_LEVELS = ("human_samples", "animal", "other")  # reference: cell_only
GENE_LEVELS = ("BRAF", "NRAS/HRAS", "other")  # reference: KRAS
FEATURES = ("model_system", "human_genetics", "multi_system")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_rows(meta_dir: Path = META) -> list[dict]:
    """Preclinical research articles with features, year, outcome B and C joined."""
    papers = {str(r["pmid"]): r for r in _read_jsonl(meta_dir / "papers.jsonl")}
    outcome = {str(r["pmid"]): r for r in _read_jsonl(meta_dir / "outcome_b.jsonl")}
    impact_path = meta_dir / "impact.jsonl"
    impact = {str(r["pmid"]): r for r in _read_jsonl(impact_path)} if impact_path.is_file() else {}
    rows = []
    for feature in _read_jsonl(meta_dir / "features.jsonl"):
        pmid = str(feature["pmid"])
        if feature.get("group") != "preclinical" or pmid not in outcome or pmid not in papers:
            continue
        extra = impact.get(pmid, {})
        rows.append(
            {
                **feature,
                "pmid": pmid,
                "year": int(papers[pmid]["year"]),
                "clin_cited_8y": int(outcome[pmid]["clin_cited_8y"]),
                "n_clin_8y": int(outcome[pmid]["n_clin_8y"]),
                "rcr": extra.get("rcr"),
                "cd": extra.get("cd"),
                "n_refs": extra.get("n_refs_openalex"),
            }
        )
    return rows


def design(rows: list[dict], years: list[int], drop: tuple[str, ...] = (), with_features: bool = True,
           placebo: bool = False) -> tuple[np.ndarray, list[str]]:
    """Columns: intercept, features, placebo, year FE, gene group, log authors, log refs.

    Missing reference counts are set to 0 with a missing indicator.
    """
    names = ["intercept"]
    cols = [np.ones(len(rows))]
    if with_features:
        if "model_system" not in drop:
            for level in MODEL_LEVELS:
                names.append(f"model_system={level}")
                cols.append(np.array([r["model_system"] == level for r in rows], float))
        for feature in ("human_genetics", "multi_system"):
            if feature not in drop:
                names.append(feature)
                cols.append(np.array([float(r[feature]) for r in rows]))
    if placebo:
        names.append("placebo_pmid_even")
        cols.append(np.array([int(r["pmid"]) % 2 == 0 for r in rows], float))
    for year in years[1:]:
        names.append(f"year={year}")
        cols.append(np.array([r["year"] == year for r in rows], float))
    for level in GENE_LEVELS:
        names.append(f"gene={level}")
        cols.append(np.array([r["gene_group"] == level for r in rows], float))
    names.append("log_authors")
    cols.append(np.log(np.array([max(1, int(r.get("n_authors") or 1)) for r in rows], float)))
    refs = [r.get("n_refs") for r in rows]
    names.append("log1p_refs")
    cols.append(np.log1p(np.array([float(v) if v is not None else 0.0 for v in refs])))
    names.append("refs_missing")
    cols.append(np.array([v is None for v in refs], float))
    x = np.column_stack(cols)
    keep = [0] + [i for i in range(1, x.shape[1]) if x[:, i].std() > 0]
    return x[:, keep], [names[i] for i in keep]


def fit_logit(x: np.ndarray, y: np.ndarray, iters: int = 50) -> np.ndarray:
    """Logistic regression by IRLS with a tiny ridge for numerical stability."""
    beta = np.zeros(x.shape[1])
    ridge = 1e-6 * np.eye(x.shape[1])
    for _ in range(iters):
        eta = np.clip(x @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = p * (1 - p)
        step = np.linalg.solve(x.T @ (x * w[:, None]) + ridge, x.T @ (y - p) - ridge @ beta)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def _predict(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x @ beta, -30, 30)))


def _ci(values: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def odds_ratios(rows: list[dict], drop: tuple[str, ...], placebo: bool, rng: np.random.Generator) -> dict:
    years = sorted({r["year"] for r in rows})
    x, names = design(rows, years, drop, placebo=placebo)
    y = np.array([r["clin_cited_8y"] for r in rows], float)
    beta = fit_logit(x, y)
    boot = np.empty((N_BOOT, len(beta)))
    for b in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        boot[b] = fit_logit(x[idx], y[idx])
    out = {}
    for j, name in enumerate(names):
        if name.startswith(("model_system", "human_genetics", "multi_system", "placebo")):
            lo, hi = _ci(np.exp(boot[:, j]))
            out[name] = {"or": float(math.exp(beta[j])), "lo": lo, "hi": hi}
    return {"n": int(len(y)), "prevalence": float(y.mean()), "terms": out}


def permutation_ors(rows: list[dict], drop: tuple[str, ...], rng: np.random.Generator) -> dict:
    """Feature ORs with F1-F3 permuted jointly within publication year."""
    years = sorted({r["year"] for r in rows})
    y = np.array([r["clin_cited_8y"] for r in rows], float)
    by_year: dict[int, list[int]] = {}
    for i, r in enumerate(rows):
        by_year.setdefault(r["year"], []).append(i)
    draws: dict[str, list[float]] = {}
    for _ in range(N_PERM):
        perm = list(range(len(rows)))
        for idx in by_year.values():
            shuffled = list(rng.permutation(idx))
            for src, dst in zip(idx, shuffled):
                perm[src] = dst
        shuffled_rows = [
            {**rows[i], **{f: rows[perm[i]][f] for f in FEATURES}} for i in range(len(rows))
        ]
        x, names = design(shuffled_rows, years, drop)
        beta = fit_logit(x, y)
        for j, name in enumerate(names):
            if name.startswith(FEATURES):
                draws.setdefault(name, []).append(math.exp(beta[j]))
    return {name: _ci(np.array(values)) for name, values in draws.items()}


def ladder(dev: list[dict], confirm: list[dict], drop: tuple[str, ...]) -> dict:
    """AUROC on confirmation for models fit on development."""
    years = sorted({r["year"] for r in dev} | {r["year"] for r in confirm})
    y_dev = np.array([r["clin_cited_8y"] for r in dev], float)
    y_con = np.array([r["clin_cited_8y"] for r in confirm], float)
    out = {"trivial": 0.5}
    for label, with_features in (("simplest", False), ("candidate", True)):
        x_all, _ = design(dev + confirm, years, drop, with_features=with_features)
        # Year effects are not transportable across disjoint year ranges; zero them at prediction.
        x_dev, x_con = x_all[: len(dev)], x_all[len(dev):]
        beta = fit_logit(x_dev, y_dev)
        _, names = design(dev + confirm, years, drop, with_features=with_features)
        beta_pred = np.array([0.0 if n.startswith("year=") else b for n, b in zip(names, beta)])
        scores = _predict(x_con, beta_pred)
        out[label] = float(auroc(torch.tensor(scores), torch.tensor(y_con)))
    return out


def verdicts(dev: dict, confirm: dict, perm: dict) -> dict:
    out = {}
    for name, d in dev["terms"].items():
        if name.startswith("placebo"):
            continue
        c = confirm["terms"].get(name)
        excl_dev = d["lo"] > 1 or d["hi"] < 1
        excl_con = c is not None and (c["lo"] > 1 or c["hi"] < 1)
        same_dir = c is not None and (d["or"] > 1) == (c["or"] > 1)
        lo, hi = perm.get(name, (0.0, math.inf))
        outside_perm = d["or"] < lo or d["or"] > hi
        out[name] = {
            "pass": bool(excl_dev and excl_con and same_dir and outside_perm),
            "ci_excludes_1_dev": bool(excl_dev),
            "ci_excludes_1_confirm": bool(excl_con),
            "same_direction": bool(same_dir),
            "outside_permutation_95": bool(outside_perm),
            "permutation_95": [lo, hi],
        }
    return out


def _fit_linear(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _fit_negbin(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """NB2 regression by maximum likelihood; returns coefficients (dispersion dropped)."""
    from scipy.optimize import minimize
    from scipy.special import gammaln

    def nll(params):
        beta, log_alpha = params[:-1], params[-1]
        alpha = math.exp(log_alpha)
        mu = np.exp(np.clip(x @ beta, -30, 30))
        r = 1.0 / alpha
        ll = gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu))
        return -ll.sum()

    start = np.zeros(x.shape[1] + 1)
    start[0] = math.log(max(y.mean(), 1e-3))
    return minimize(nll, start, method="L-BFGS-B").x[:-1]


def _feature_terms(names: list[str], beta: np.ndarray, boot: np.ndarray, transform) -> dict:
    out = {}
    for j, name in enumerate(names):
        if name.startswith(FEATURES):
            lo, hi = _ci(transform(boot[:, j]))
            out[name] = {"estimate": float(transform(np.array([beta[j]]))[0]), "lo": lo, "hi": hi}
    return out


def secondary(rows: list[dict], drop: tuple[str, ...], rng: np.random.Generator, journals: dict[str, str]) -> dict:
    """Full 2000-2015 period; estimates with bootstrap CIs, no pass/fail."""
    years = sorted({r["year"] for r in rows})
    x, names = design(rows, years, drop)
    out = {}
    specs = {
        "n_clin_8y_negbin_rate_ratio": (
            np.array([r["n_clin_8y"] for r in rows], float), _fit_negbin, np.exp, rows, x, names
        ),
    }
    with_rcr = [i for i, r in enumerate(rows) if r.get("rcr") is not None]
    specs["log_rcr_linear"] = (
        np.log(np.array([max(float(rows[i]["rcr"]), 1e-3) for i in with_rcr])),
        _fit_linear, lambda v: v, [rows[i] for i in with_rcr], x[with_rcr], names,
    )
    with_cd = [i for i, r in enumerate(rows) if r.get("cd") is not None]
    if len(with_cd) > 50:
        specs["disruption_linear"] = (
            np.array([float(rows[i]["cd"]) for i in with_cd]), _fit_linear, lambda v: v,
            [rows[i] for i in with_cd], x[with_cd], names,
        )
    top = [j for j, _ in sorted(
        ((j, n) for j, n in _counts(journals.get(r["pmid"], "") for r in rows).items() if j),
        key=lambda item: -item[1])[:50]]
    if top:
        jcols = np.column_stack([[journals.get(r["pmid"], "") == j for r in rows] for j in top]).astype(float)
        specs["journal_sensitivity_logit_or"] = (
            np.array([r["clin_cited_8y"] for r in rows], float), fit_logit, np.exp, rows,
            np.column_stack([x, jcols]), names + [f"journal={j}" for j in top],
        )
    for label, (y, fitter, transform, _subset, xs, ns) in specs.items():
        beta = fitter(xs, y)
        n_boot = N_BOOT if fitter is not _fit_negbin else max(1, N_BOOT // 5)
        boot = np.empty((n_boot, len(beta)))
        for b in range(n_boot):
            idx = rng.integers(0, len(y), len(y))
            boot[b] = fitter(xs[idx], y[idx])
        out[label] = {"n": int(len(y)), "n_boot": n_boot, "terms": _feature_terms(ns, beta, boot, transform)}
    return out


def _counts(values) -> dict:
    out: dict = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def clinical_descriptive(meta_dir: Path) -> dict:
    """Clinical-paper rates by model system; described, not tested."""
    outcome = {str(r["pmid"]): r for r in _read_jsonl(meta_dir / "outcome_b.jsonl")}
    table: dict[str, list[int]] = {}
    for f in _read_jsonl(meta_dir / "features.jsonl"):
        pmid = str(f["pmid"])
        if f.get("group") == "clinical" and pmid in outcome:
            table.setdefault(f["model_system"], []).append(int(outcome[pmid]["clin_cited_8y"]))
    return {k: {"n": len(v), "clin_cited_8y": sum(v) / len(v)} for k, v in sorted(table.items())}


def run(meta_dir: Path = META, drop: tuple[str, ...] = ()) -> dict:
    rows = load_rows(meta_dir)
    dev = [r for r in rows if DEV_YEARS[0] <= r["year"] <= DEV_YEARS[1]]
    confirm = [r for r in rows if CONFIRM_YEARS[0] <= r["year"] <= CONFIRM_YEARS[1]]
    rng = np.random.default_rng(0)
    fit_dev = odds_ratios(dev, drop, placebo=True, rng=rng)
    fit_con = odds_ratios(confirm, drop, placebo=True, rng=rng)
    perm = permutation_ors(dev, drop, rng)
    per_term = verdicts(fit_dev, fit_con, perm)
    placebo_ok = all(
        fit["terms"]["placebo_pmid_even"]["lo"] <= 1 <= fit["terms"]["placebo_pmid_even"]["hi"]
        for fit in (fit_dev, fit_con)
    )
    passed = [name for name, v in per_term.items() if v["pass"]]
    journals = {str(r["pmid"]): str(r.get("journal") or "") for r in _read_jsonl(meta_dir / "papers.jsonl")}
    return {
        "secondary": secondary(rows, drop, rng, journals),
        "clinical_descriptive": clinical_descriptive(meta_dir),
        "ledger": "experiments.md, 2026-09-29 metascience v1",
        "dropped_by_gate": list(drop),
        "development": fit_dev,
        "confirmation": fit_con,
        "verdicts": per_term,
        "placebo_ok": bool(placebo_ok),
        "ladder_auroc_confirmation": ladder(dev, confirm, drop),
        "verdict": ("unreliable (placebo)" if not placebo_ok else "pass" if passed else "kill"),
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop", nargs="*", default=[], choices=list(FEATURES))
    args = parser.parse_args()
    report = run(drop=tuple(args.drop))
    RESULT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("verdict", "passed", "placebo_ok", "ladder_auroc_confirmation")}, indent=2))


if __name__ == "__main__":
    main()
