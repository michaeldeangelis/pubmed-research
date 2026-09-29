"""Breakthrough case histories: model systems on the citation path to approval.

Contract: ``experiments.md`` entry "2026-09-29 breakthrough case histories"
(method fixed before the run; descriptive, no pass/fail).

Run as ``python -m src.meta.cases``. Steps:

1. Verify every anchor and milestone PMID against Europe PMC (year, first
   author surname, title keywords). A PMID that fails is searched for by
   author + year + title keywords; a unique passing hit replaces it and the
   correction (old, new, reason) is recorded in the result.
2. Lineage per case from iCite ``references``: depth 1 = references of the
   anchors, depth 2 = references of depth-1 papers. Kept: iCite research
   articles with iCite year <= the case's anchor year (the latest anchor, i.e.
   the approval trial), anchors excluded. A paper kept at depth 1 is not
   counted again at depth 2. Depth 2 expands from every kept depth-1 paper.
3. Europe PMC core metadata for lineage PMIDs not in ``data/meta/papers.jsonl``
   (batched ``EXT_ID:(a OR b ...) AND SRC:MED``, cached under
   ``data/meta/cases/cache/``). ``model_system`` from ``src.meta.features``
   rules, unchanged. Status of each lineage paper, first match wins:
   ``unclassified`` (not in Europe PMC, or no MeSH), ``non_primary`` (a
   publication type in ``features.NON_PRIMARY_PUB_TYPES``), else its
   model_system. Shares are over classified primary papers only; the other
   two are counted and reported.
4. Base rate: eligible rows of ``data/meta/features.jsonl`` (metascience v1
   corpus, 2000-2015), year from ``papers.jsonl``, restricted to the
   lineage's year range intersected with the corpus years; group=all and
   group=preclinical. The lineage side of each comparison is restricted to
   the same years and the same iCite group. Share difference = lineage -
   base, Newcombe (hybrid Wilson) 95% CI.
5. Result: ``results/case_histories.json``.

This module never reads outcome files (``outcome_b.jsonl``, ``impact.jsonl``).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

from src import config
from src.meta import corpus, features
from src.schema import strip_markup

logger = logging.getLogger(__name__)

ROOT = features.ROOT
META_DIR = config.DATA / "meta"
CASES_DIR = META_DIR / "cases"
CACHE_DIR = CASES_DIR / "cache"
LINEAGE_PATH = CASES_DIR / "lineage.jsonl"
PAPERS_PATH = META_DIR / "papers.jsonl"
FEATURES_PATH = META_DIR / "features.jsonl"
RESULT_PATH = ROOT / "results" / "case_histories.json"

ICITE_FIELDS = "pmid,year,title,is_research_article,is_clinical,references"
EPMC_BATCH = 100
MODEL_SYSTEMS = features.MODEL_SYSTEMS
STATUSES = (*MODEL_SYSTEMS, "non_primary", "unclassified")
Z95 = 1.959963984540054

Getter = Callable[[str, dict], dict]
IciteFetch = Callable[[list[str]], list[dict]]

# --- cases (listed before the run; see experiments.md) -------------------------

CASES: dict[str, dict] = {
    "vemurafenib": {
        "target": "BRAF V600E",
        "anchors": ["21639808"],
        "milestones": [
            {"pmid": "12068308", "role": "discovery", "author": "Davies", "year": 2002,
             "title_kw": ["BRAF", "cancer"], "label": "Davies 2002 Nature, BRAF mutations in human cancer"},
            {"pmid": "18287029", "role": "first_selective_compound", "author": "Tsai", "year": 2008,
             "title_kw": ["selective", "B-Raf"], "label": "Tsai 2008 PNAS, PLX4720"},
            {"pmid": "20823850", "role": "preclinical_plus_clinical_proof", "author": "Bollag", "year": 2010,
             "title_kw": ["RAF inhibitor", "melanoma"], "label": "Bollag 2010 Nature, PLX4032"},
            {"pmid": "20818844", "role": "first_in_human", "author": "Flaherty", "year": 2010,
             "title_kw": ["BRAF", "melanoma"], "label": "Flaherty 2010 NEJM, phase 1"},
            {"pmid": "21639808", "role": "approval_trial", "author": "Chapman", "year": 2011,
             "title_kw": ["vemurafenib", "melanoma"], "label": "Chapman 2011 NEJM, BRIM-3"},
        ],
        "timeline": {"discovery": "12068308", "first_selective_compound": "18287029",
                     "first_in_human": "20818844", "approval_trial": "21639808"},
        "timeline_note": "Publication years. Discovery = the BRAF V600E mutation.",
    },
    "sotorasib": {
        "target": "KRAS G12C",
        "anchors": ["32955176", "34096690"],
        "milestones": [
            {"pmid": "24256730", "role": "discovery_of_pocket", "author": "Ostrem", "year": 2013,
             "title_kw": ["G12C", "allosterically"], "label": "Ostrem 2013 Nature, switch-II pocket"},
            {"pmid": "26739882", "role": "cellular_proof", "author": "Patricelli", "year": 2016,
             "title_kw": ["KRAS"], "label": "Patricelli 2016 Cancer Discov, ARS-853"},
            {"pmid": "29373830", "role": "in_vivo_proof", "author": "Janes", "year": 2018,
             "title_kw": ["G12C"], "label": "Janes 2018 Cell, ARS-1620"},
            {"pmid": "31666701", "role": "preclinical_candidate", "author": "Canon", "year": 2019,
             "title_kw": ["AMG 510"], "label": "Canon 2019 Nature, AMG 510"},
            {"pmid": "32955176", "role": "first_in_human", "author": "Hong", "year": 2020,
             "title_kw": ["Sotorasib"], "label": "Hong 2020 NEJM, CodeBreaK 100 phase 1"},
            {"pmid": "34096690", "role": "approval_trial", "author": "Skoulidis", "year": 2021,
             "title_kw": ["Sotorasib"], "label": "Skoulidis 2021 NEJM, CodeBreaK 100 phase 2"},
        ],
        "timeline": {"discovery": "24256730", "first_selective_compound": "24256730",
                     "first_in_human": "32955176", "approval_trial": "34096690"},
        "timeline_note": (
            "Publication years. Discovery = the druggable switch-II pocket (the "
            "KRAS mutation itself was known from the 1980s). Ostrem 2013 is also "
            "the first G12C-selective covalent compound series."
        ),
    },
}

# --- stats ----------------------------------------------------------------------


def wilson(k: int, n: int, z: float = Z95) -> dict:
    if n == 0:
        return {"k": 0, "n": 0, "share": None, "ci95": [None, None]}
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"k": k, "n": n, "share": round(p, 4),
            "ci95": [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]}


def newcombe_diff(k1: int, n1: int, k2: int, n2: int, z: float = Z95) -> dict:
    """p1 - p2 with the Newcombe hybrid-score (method 10) 95% CI."""
    if n1 == 0 or n2 == 0:
        return {"diff": None, "ci95": [None, None]}
    a, b = wilson(k1, n1, z), wilson(k2, n2, z)
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = a["ci95"]
    l2, u2 = b["ci95"]
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return {"diff": round(d, 4), "ci95": [round(lo, 4), round(hi, 4)]}


def shares(statuses: Iterable[str]) -> dict:
    """Model-system shares over classified primary papers, plus the others."""
    c = Counter(statuses)
    n = sum(c[m] for m in MODEL_SYSTEMS)
    return {
        "n_classified": n,
        "n_non_primary": c["non_primary"],
        "n_unclassified": c["unclassified"],
        "shares": {m: wilson(c[m], n) for m in MODEL_SYSTEMS},
    }


def compare(lineage_statuses: list[str], base_systems: list[str]) -> dict:
    lin, base = shares(lineage_statuses), shares(base_systems)
    diff = {
        m: newcombe_diff(lin["shares"][m]["k"], lin["n_classified"],
                         base["shares"][m]["k"], base["n_classified"])
        for m in MODEL_SYSTEMS
    }
    return {"lineage": lin, "base": {"n": base["n_classified"], "shares": base["shares"]},
            "diff_lineage_minus_base": diff}


# --- Europe PMC ------------------------------------------------------------------


def _year(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_epmc(item: dict) -> dict:
    row = corpus.parse_paper(item)
    row["author_string"] = str(item.get("authorString") or "")
    return row


def fetch_epmc(pmids: Iterable[str], getter: Getter = corpus.http_get_json,
               cache_dir: Path = CACHE_DIR) -> dict[str, dict]:
    """Europe PMC core records for PMIDs (SRC:MED), batched, cached by query hash."""
    ids = sorted({str(p) for p in pmids}, key=int)
    out: dict[str, dict] = {}
    for i in range(0, len(ids), EPMC_BATCH):
        batch = ids[i : i + EPMC_BATCH]
        params = {"query": f"EXT_ID:({' OR '.join(batch)}) AND SRC:MED", "format": "json",
                  "resultType": "core", "pageSize": 1000}
        key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
        payload = corpus.cached_get(cache_dir / "europepmc" / f"{key}.json", config.EUROPEPMC, params, getter)
        raw = (payload.get("resultList") or {}).get("result") or []
        if isinstance(raw, dict):
            raw = [raw]
        for item in raw:
            if isinstance(item, dict) and str(item.get("pmid") or "") in batch:
                out.setdefault(str(item["pmid"]), parse_epmc(item))
    return out


def search_epmc(query: str, getter: Getter = corpus.http_get_json, cache_dir: Path = CACHE_DIR) -> list[dict]:
    params = {"query": f"({query}) AND SRC:MED", "format": "json", "resultType": "core", "pageSize": 25}
    key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
    payload = corpus.cached_get(cache_dir / "europepmc_search" / f"{key}.json", config.EUROPEPMC, params, getter)
    raw = (payload.get("resultList") or {}).get("result") or []
    return [parse_epmc(i) for i in ([raw] if isinstance(raw, dict) else raw) if isinstance(i, dict) and i.get("pmid")]


# --- step 1: verification ----------------------------------------------------------


def first_author(author_string: str) -> str:
    first = (author_string or "").split(",", 1)[0].strip()
    return first.rsplit(" ", 1)[0] if " " in first else first


def check_record(rec: dict | None, spec: dict) -> list[str]:
    """Problems with a Europe PMC record against the expected year/author/title."""
    if rec is None:
        return ["pmid not found in Europe PMC"]
    problems = []
    if _year(rec.get("year")) != spec["year"]:
        problems.append(f"year {rec.get('year')} != {spec['year']}")
    if first_author(rec.get("author_string", "")).casefold() != spec["author"].casefold():
        problems.append(f"first author {first_author(rec.get('author_string', ''))!r} != {spec['author']!r}")
    title = strip_markup(rec.get("title") or "").casefold()
    missing = [k for k in spec["title_kw"] if k.casefold() not in title]
    if missing:
        problems.append(f"title lacks {missing}")
    return problems


def verify_cases(cases: dict, getter: Getter = corpus.http_get_json,
                 cache_dir: Path = CACHE_DIR) -> tuple[dict, list[dict], list[dict]]:
    """Returns (corrected cases, corrections, verification rows)."""
    cases = json.loads(json.dumps(cases))
    specs = [m for c in cases.values() for m in c["milestones"]]
    recs = fetch_epmc([m["pmid"] for m in specs], getter=getter, cache_dir=cache_dir)
    corrections: list[dict] = []
    rows: list[dict] = []
    remap: dict[str, str] = {}
    for spec in specs:
        old = spec["pmid"]
        if old in remap:
            spec["pmid"] = remap[old]
            continue
        problems = check_record(recs.get(old), spec)
        new = old
        if problems:
            kw = " AND ".join(f'TITLE:"{k}"' for k in spec["title_kw"])
            hits = search_epmc(f'AUTH:"{spec["author"]}" AND PUB_YEAR:{spec["year"]} AND {kw}',
                               getter=getter, cache_dir=cache_dir)
            good = [h for h in hits if not check_record(h, spec)]
            if len(good) == 1:
                new = good[0]["pmid"]
                recs[new] = good[0]
                corrections.append({"old": old, "new": new, "label": spec["label"],
                                    "reason": "; ".join(problems)})
                remap[old] = new
        spec["pmid"] = new
        rec = recs.get(new)
        rows.append({"pmid": new, "label": spec["label"], "verified": not check_record(rec, spec),
                     "problems": check_record(rec, spec),
                     "epmc": None if rec is None else {"year": rec.get("year"), "first_author": first_author(rec.get("author_string", "")),
                                                      "title": rec.get("title"), "journal": rec.get("journal")}})
    for case in cases.values():
        case["anchors"] = [remap.get(p, p) for p in case["anchors"]]
        case["timeline"] = {k: remap.get(p, p) for k, p in case["timeline"].items()}
    return cases, corrections, rows


# --- step 2: lineage ------------------------------------------------------------------


def _keep(rec: dict | None, max_year: int) -> str | None:
    """None if kept, else the drop reason."""
    if rec is None:
        return "not_in_icite"
    if not features.truthy(rec.get("is_research_article")):
        return "not_research_article"
    y = _year(rec.get("year"))
    if y is None:
        return "no_year"
    if y > max_year:
        return "after_anchor_year"
    return None


def build_lineage(anchors: list[str], max_year: int, fetch: IciteFetch) -> dict:
    """Depth-1 and depth-2 lineage from iCite references.

    Returns {"depth": {pmid: 1|2}, "icite": {pmid: rec}, "dropped": {1: Counter, 2: Counter},
    "n_refs_anchor": {...}}.
    """
    anchors = [str(a) for a in anchors]
    anchor_set = set(anchors)
    icite: dict[str, dict] = {r["pmid"]: r for r in fetch(anchors)}
    missing_anchor = [a for a in anchors if a not in icite]
    if missing_anchor:
        raise ValueError(f"anchors not in iCite: {missing_anchor}")

    def refs(pmid: str) -> set[str]:
        return {str(x) for x in icite[pmid].get("references") or []}

    depth: dict[str, int] = {}
    dropped = {1: Counter(), 2: Counter()}
    cand1 = set().union(*(refs(a) for a in anchors)) - anchor_set
    icite.update({r["pmid"]: r for r in fetch(sorted(cand1 - set(icite), key=int))})
    for p in sorted(cand1, key=int):
        reason = _keep(icite.get(p), max_year)
        if reason:
            dropped[1][reason] += 1
        else:
            depth[p] = 1
    level1 = [p for p in depth]
    cand2 = set().union(set(), *(refs(p) for p in level1)) - anchor_set - set(level1)
    icite.update({r["pmid"]: r for r in fetch(sorted(cand2 - set(icite), key=int))})
    for p in sorted(cand2, key=int):
        reason = _keep(icite.get(p), max_year)
        if reason:
            dropped[2][reason] += 1
        else:
            depth[p] = 2
    return {
        "depth": depth,
        "icite": icite,
        "dropped": {k: dict(v) for k, v in dropped.items()},
        "n_candidates": {1: len(cand1), 2: len(cand2)},
        "n_refs_anchor": {a: len(refs(a)) for a in anchors},
    }


# --- step 3: classification -------------------------------------------------------------


def is_non_primary(paper: dict) -> bool:
    return any(features._norm_pub_type(t) in features._NON_PRIMARY for t in paper.get("pub_types") or [])


def status(paper: dict | None) -> str:
    """unclassified > non_primary > model_system (features rules, unchanged)."""
    if paper is None or not features.mesh_set(paper.get("mesh")):
        return "unclassified"
    if is_non_primary(paper):
        return "non_primary"
    return features.model_system(features.systems_present(paper))


_DESIGN_MESH = features._norm_set(
    features.HUMAN_SPECIFIC_TERMS + features.ANIMAL_STRONG_TERMS + features.ANIMAL_WEAK_TERMS
    + features.CELL_TERMS + features.HUMAN_GENERIC_TERMS
)
_TRIAL_TYPE = re.compile(r"clinical trial|randomized|multicenter", re.I)


def design_line(paper: dict | None) -> str:
    """One line from MeSH, publication types and abstract cues."""
    if paper is None:
        return "no Europe PMC record"
    st = status(paper)
    present = features.systems_present(paper)
    systems = [s for s in features.SYSTEMS if present[s]] or ["none"]
    mesh = [m for m in paper.get("mesh") or [] if features.norm_term(m) in _DESIGN_MESH
            or features.norm_term(m).startswith(features._ANIMAL_PREFIXES)]
    trial = [t for t in paper.get("pub_types") or [] if _TRIAL_TYPE.search(t)]
    text = features.paper_text(paper)
    cues = []
    for pat in (features.HUMAN_SAMPLE_TEXT, features.IN_VIVO_TEXT, features.CELL_TEXT):
        cues += sorted({m.group(0).lower() for m in pat.finditer(text)})[:3]
    parts = [f"{st}", f"systems: {'+'.join(systems)}"]
    if mesh:
        parts.append("design MeSH: " + ", ".join(dict.fromkeys(mesh)))
    if trial:
        parts.append("pub types: " + ", ".join(trial))
    if cues:
        parts.append("abstract cues: " + ", ".join(cues))
    return "; ".join(parts)


# --- step 4: base rate ------------------------------------------------------------------


def base_rows(features_rows: list[dict], papers: list[dict]) -> list[dict]:
    """Eligible rows with a year from papers.jsonl."""
    year = {str(p["pmid"]): _year(p.get("year")) for p in papers}
    out = []
    for r in features_rows:
        if not r.get("eligible"):
            continue
        y = year.get(str(r["pmid"]))
        if y is not None:
            out.append({"pmid": str(r["pmid"]), "group": r["group"], "model_system": r["model_system"], "year": y})
    return out


# --- assembly ------------------------------------------------------------------------------


def lineage_rows(lin: dict, meta: dict[str, dict]) -> list[dict]:
    rows = []
    for pmid, d in sorted(lin["depth"].items(), key=lambda kv: (kv[1], int(kv[0]))):
        rec = lin["icite"][pmid]
        paper = meta.get(pmid)
        rows.append({
            "pmid": pmid,
            "depth": d,
            "year": _year(rec.get("year")),
            "group": "clinical" if features.truthy(rec.get("is_clinical")) else "preclinical",
            "status": status(paper),
            "on_topic": bool(paper) and features.on_topic(paper),
            "in_epmc": paper is not None,
        })
    return rows


def counts_by_depth(rows: list[dict]) -> dict:
    out = {}
    for label, sub in (("1", [r for r in rows if r["depth"] == 1]),
                       ("2", [r for r in rows if r["depth"] == 2]),
                       ("all", rows)):
        c = Counter(r["status"] for r in sub)
        out[label] = {"n": len(sub), **{s: c[s] for s in STATUSES}}
    return out


def case_summary(rows: list[dict], base: list[dict], corpus_years: tuple[int, int]) -> dict:
    years = [r["year"] for r in rows if r["year"] is not None]
    lo_l, hi_l = (min(years), max(years)) if years else (None, None)
    lo, hi = (max(lo_l, corpus_years[0]), min(hi_l, corpus_years[1])) if years else (None, None)
    out: dict = {
        "lineage_year_range": [lo_l, hi_l],
        "counts": counts_by_depth(rows),
        "shares_all_years": {},
        "comparison": {"year_range": [lo, hi], "corpus_years": list(corpus_years), "by_depth": {}},
    }
    depths = (("1", lambda r: r["depth"] == 1), ("2", lambda r: r["depth"] == 2), ("all", lambda r: True))
    for label, f in depths:
        sub = [r for r in rows if f(r)]
        out["shares_all_years"][label] = shares(r["status"] for r in sub)
        if lo is None or lo > hi:
            continue
        win = [r for r in sub if lo <= r["year"] <= hi]
        b = [x for x in base if lo <= x["year"] <= hi]
        out["comparison"]["by_depth"][label] = {
            "all": compare([r["status"] for r in win], [x["model_system"] for x in b]),
            "preclinical": compare([r["status"] for r in win if r["group"] == "preclinical"],
                                   [x["model_system"] for x in b if x["group"] == "preclinical"]),
            "all_on_topic_lineage": compare([r["status"] for r in win if r["on_topic"]],
                                            [x["model_system"] for x in b]),
            "n_lineage_outside_years": sum(1 for r in sub if not lo <= r["year"] <= hi),
        }
    return out


def milestone_rows(case: dict, depth: dict[str, int], meta: dict[str, dict]) -> list[dict]:
    out = []
    anchors = set(case["anchors"])
    for m in case["milestones"]:
        paper = meta.get(m["pmid"])
        out.append({
            "pmid": m["pmid"],
            "label": m["label"],
            "role": m["role"],
            "year": paper.get("year") if paper else None,
            "title": paper.get("title") if paper else None,
            "model_system": status(paper),
            "depth": "anchor" if m["pmid"] in anchors else depth.get(m["pmid"]),
            "design": design_line(paper),
        })
    return out


def timeline(case: dict, meta: dict[str, dict]) -> dict:
    t = {}
    for key, pmid in case["timeline"].items():
        paper = meta.get(pmid)
        t[key] = {"pmid": pmid, "year": paper.get("year") if paper else None}
    t["years_discovery_to_approval_trial"] = (
        t["approval_trial"]["year"] - t["discovery"]["year"]
        if t["approval_trial"]["year"] and t["discovery"]["year"] else None
    )
    t["note"] = case.get("timeline_note")
    return t


def run(
    cases: dict = CASES,
    getter: Getter = corpus.http_get_json,
    cache_dir: Path = CACHE_DIR,
    papers: list[dict] | None = None,
    features_rows: list[dict] | None = None,
    icite_fetch: IciteFetch | None = None,
) -> tuple[dict, list[dict]]:
    papers = corpus.read_jsonl(PAPERS_PATH) if papers is None else papers
    features_rows = corpus.read_jsonl(FEATURES_PATH) if features_rows is None else features_rows
    if icite_fetch is None:
        def icite_fetch(pmids: list[str]) -> list[dict]:
            return corpus.fetch_icite(pmids, getter=getter, cache_dir=cache_dir, fields=ICITE_FIELDS, subdir="icite")

    cases, corrections, verification = verify_cases(cases, getter=getter, cache_dir=cache_dir)

    lineages = {}
    for name, case in cases.items():
        anchor_recs = {r["pmid"]: r for r in icite_fetch(case["anchors"])}
        max_year = max(_year(anchor_recs[a].get("year")) for a in case["anchors"])
        lin = build_lineage(case["anchors"], max_year, icite_fetch)
        lin["max_year"] = max_year
        lineages[name] = lin

    known = {str(p["pmid"]): p for p in papers}
    wanted = {p for lin in lineages.values() for p in lin["depth"]}
    wanted |= {m["pmid"] for c in cases.values() for m in c["milestones"]}
    fetched = fetch_epmc(sorted(wanted - set(known), key=int), getter=getter, cache_dir=cache_dir)
    meta = {p: known.get(p) or fetched.get(p) for p in wanted}
    meta = {k: v for k, v in meta.items() if v is not None}

    base = base_rows(features_rows, papers)
    corpus_years = (min(x["year"] for x in base), max(x["year"] for x in base))

    result: dict = {
        "run_date": _dt.date.today().isoformat(),
        "contract": "experiments.md, 2026-09-29 breakthrough case histories",
        "method": {
            "lineage": "depth 1 = iCite references of anchors; depth 2 = iCite references of kept depth-1 papers; "
                       "kept = iCite research article with iCite year <= case anchor year (latest anchor); anchors excluded; "
                       "a depth-1 paper is not counted again at depth 2",
            "classification": "src.meta.features systems_present/model_system, unchanged; status order unclassified "
                              "(no Europe PMC record or no MeSH) > non_primary (features.NON_PRIMARY_PUB_TYPES) > model_system",
            "shares": "over classified primary papers; Wilson 95% CI; diff = lineage - base, Newcombe hybrid 95% CI",
            "base_rate": "features.jsonl eligible rows (metascience v1 corpus, on-topic RAS/MAPK, MeSH, primary), year from "
                         "papers.jsonl, restricted to the lineage years intersected with the corpus years; lineage side "
                         "restricted to the same years and iCite group. 'all_on_topic_lineage' additionally keeps only "
                         "lineage papers passing features.on_topic (sensitivity, not in the contract)",
        },
        "pmid_verification": verification,
        "pmid_corrections": corrections,
        "cases": {},
    }
    all_rows: list[dict] = []
    for name, case in cases.items():
        lin = lineages[name]
        rows = lineage_rows(lin, meta)
        for r in rows:
            all_rows.append({**r, "case": name})
        result["cases"][name] = {
            "target": case["target"],
            "anchors": [{"pmid": a, "year": _year(lin["icite"][a].get("year")),
                         "title": lin["icite"][a].get("title"), "n_icite_references": lin["n_refs_anchor"][a]}
                        for a in case["anchors"]],
            "anchor_year": lin["max_year"],
            "lineage_candidates": lin["n_candidates"],
            "lineage_dropped": lin["dropped"],
            "n_not_in_epmc": sum(1 for r in rows if not r["in_epmc"]),
            "milestones": milestone_rows(case, lin["depth"], meta),
            "timeline": timeline(case, meta),
            **case_summary(rows, base, corpus_years),
        }

    # combined lineage: union, a paper keeps its smallest depth
    comb: dict[str, dict] = {}
    for r in all_rows:
        prev = comb.get(r["pmid"])
        if prev is None or r["depth"] < prev["depth"]:
            comb[r["pmid"]] = {k: v for k, v in r.items() if k != "case"}
    overlap = {n: {r["pmid"] for r in all_rows if r["case"] == n} for n in cases}
    names = list(cases)
    result["combined"] = {
        "n_shared_between_cases": len(set.intersection(*overlap.values())) if len(names) > 1 else 0,
        **case_summary(list(comb.values()), base, corpus_years),
    }
    return result, all_rows


# --- summary ---------------------------------------------------------------------------------


def _fmt(s: dict) -> str:
    if s["share"] is None:
        return "   n/a          "
    return f"{s['share']:.2f} [{s['ci95'][0]:.2f},{s['ci95'][1]:.2f}]"


def summary_lines(result: dict) -> list[str]:
    lines = []
    if result["pmid_corrections"]:
        for c in result["pmid_corrections"]:
            lines.append(f"PMID correction {c['old']} -> {c['new']} ({c['label']}): {c['reason']}")
    else:
        bad = [v for v in result["pmid_verification"] if not v["verified"]]
        lines.append("PMIDs: all verified" if not bad else f"PMIDs unverified: {[v['pmid'] for v in bad]}")
    blocks = [(n, c) for n, c in result["cases"].items()] + [("combined", result["combined"])]
    for name, c in blocks:
        cnt = c["counts"]
        lines.append(f"\n== {name}  lineage years {c['lineage_year_range']}")
        for d in ("1", "2", "all"):
            x = cnt[d]
            lines.append(f"  depth {d:>3}: n={x['n']:5d}  " + "  ".join(f"{s}={x[s]}" for s in STATUSES))
        for m in c.get("milestones", []):
            lines.append(f"  {m['year']} {m['label'][:45]:45s} {m['model_system']:13s} depth={m['depth']}")
        if "timeline" in c:
            t = c["timeline"]
            lines.append("  timeline: " + " -> ".join(f"{k} {t[k]['year']}" for k in
                                                     ("discovery", "first_selective_compound", "first_in_human", "approval_trial")))
        cmp_ = c["comparison"]
        lines.append(f"  vs base rate, years {cmp_['year_range']}: share lineage | base | diff")
        for d, blk in cmp_["by_depth"].items():
            for g in ("all", "preclinical"):
                x = blk[g]
                lines.append(f"   depth {d:>3} {g:11s} n={x['lineage']['n_classified']:4d} vs {x['base']['n']:5d}")
                for m in MODEL_SYSTEMS:
                    df = x["diff_lineage_minus_base"][m]
                    dtxt = "n/a" if df["diff"] is None else f"{df['diff']:+.2f} [{df['ci95'][0]:+.2f},{df['ci95'][1]:+.2f}]"
                    lines.append(f"      {m:13s} {_fmt(x['lineage']['shares'][m])} | {_fmt(x['base']['shares'][m])} | {dtxt}")
    return lines


def main() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result, rows = run()
    corpus.write_jsonl(LINEAGE_PATH, rows)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\n".join(summary_lines(result)))
    print(f"\nwrote {RESULT_PATH} and {LINEAGE_PATH}")
    return result


if __name__ == "__main__":
    main()
