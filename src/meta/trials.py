"""Metascience v3: trial background citations from ClinicalTrials.gov.

Contract: ``experiments.md`` entry "2026-09-30 metascience v3: trial
background citations". Run as ``python -m src.meta.trials``. Steps:

1. Fetch every registered study from the ClinicalTrials.gov API v2
   (``/api/v2/studies``, ``pageSize=1000``, ``pageToken`` paging). ``fields``
   is restricted to the v2 field pieces ``NCTId``, ``StudyType``,
   ``StartDate``, ``Phase``, ``InterventionType`` and ``ReferencesModule``,
   which return these JSON paths:

   * ``protocolSection.identificationModule.nctId``
   * ``protocolSection.designModule.studyType``
   * ``protocolSection.statusModule.startDateStruct.date`` (``YYYY-MM-DD``
     or ``YYYY-MM``)
   * ``protocolSection.designModule.phases`` (list; ``NA``, ``EARLY_PHASE1``,
     ``PHASE1`` .. ``PHASE4``; phase 1/2 is ``["PHASE1", "PHASE2"]``)
   * ``protocolSection.armsInterventionsModule.interventions[].type``
   * ``protocolSection.referencesModule.references[]`` with ``pmid``,
     ``type`` (``BACKGROUND``, ``RESULT``, ``DERIVED``) and ``citation``.

   Each raw page is cached as ``data/meta/trials/cache/page_NNNN.json``; the
   next page's token is read from the cached previous page, so a rerun is
   offline and an interrupted fetch resumes at the first missing page (page
   tokens can expire; if a resume fails, delete the cache and refetch). A
   compact per-study table goes to ``studies.jsonl`` (with the cache page
   each study came from), counts to ``manifest.json``.
2. ``pmid_index.jsonl``: one row per PMID cited as a BACKGROUND reference by
   an INTERVENTIONAL study, with the list of those studies
   (``nct``, ``start_year``, ``phases``, ``intervention_types``). A PMID
   cited twice by the same study is listed once.
3. ``outcome_<pathway>.jsonl`` for pathway ``ras`` (v1,
   ``data/meta/features.jsonl``), ``egfr`` and ``pi3k``
   (``data/meta/pathways/<p>/features.jsonl``): one row per eligible
   preclinical paper. With ``Y`` the paper's year from that pathway's
   ``papers.jsonl`` (iCite year when missing, as in outcome B), a linked study
   counts when ``Y - 1 <= start_year <= Y + 8``.

   * ``trial_bg_8y`` / ``n_trials_bg_8y``: any / number of linked studies in
     the window.
   * ``trial_bg_8y_drug``: the same, restricted to studies with a DRUG or
     BIOLOGICAL intervention.
   * ``trial_bg_8y_ph2``: restricted to phase 2 or later: every listed phase
     is PHASE2, PHASE3 or PHASE4 (so PHASE2, PHASE3, PHASE4 and PHASE2/PHASE3
     count; PHASE1/PHASE2, EARLY_PHASE1, NA and no phase do not).
   * ``first_trial_year``: earliest start year over all dated linked studies
     (any year), null if none.
   * ``n_trials_bg_undated``: linked studies without a start date; they are
     excluded from the window and counted (also in the manifest).
4. Linkage check: 30 random linked (pmid, nct) pairs in the window (pooled
   over pathways, a paper kept in the first of ras, egfr, pi3k;
   ``numpy.random.default_rng(0)``) are written to
   ``bench/meta_validation/trial_links_sample.json`` with the trial's
   reference entry (read back from the cached page) and the corpus title, plus
   a title/citation token-overlap auto-check.

Blinding: this module reads only ``pmid``, ``group`` and ``eligible`` from
the features tables and reports marginal counts only. It never relates a
design feature to the outcome.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from src import config
from src.meta import corpus
from src.meta.corpus import Getter, read_jsonl, write_jsonl

logger = logging.getLogger(__name__)

META_DIR = config.DATA / "meta"
TRIALS_DIR = META_DIR / "trials"
CACHE_DIR = TRIALS_DIR / "cache"
STUDIES_PATH = TRIALS_DIR / "studies.jsonl"
INDEX_PATH = TRIALS_DIR / "pmid_index.jsonl"
MANIFEST_PATH = TRIALS_DIR / "manifest.json"
SAMPLE_PATH = config.DATA.parent / "bench" / "meta_validation" / "trial_links_sample.json"

API_URL = "https://clinicaltrials.gov/api/v2/studies"
PAGE_SIZE = 1000
FIELDS = "NCTId,StudyType,StartDate,Phase,InterventionType,ReferencesModule"

PATHWAYS = ("ras", "egfr", "pi3k")
WINDOW_BEFORE = 1
WINDOW_AFTER = 8
DRUG_TYPES = frozenset({"DRUG", "BIOLOGICAL"})
PH2_PHASES = frozenset({"PHASE2", "PHASE3", "PHASE4"})
N_SAMPLE = 30
MATCH_THRESHOLD = 0.6


# ---------------------------------------------------------------- step 1


def page_params(token: str | None) -> dict:
    params = {"format": "json", "pageSize": PAGE_SIZE, "fields": FIELDS, "countTotal": "true"}
    if token:
        params["pageToken"] = token
    return params


def iter_pages(getter: Getter = corpus.http_get_json, cache_dir: Path = CACHE_DIR) -> Iterator[tuple[int, dict]]:
    """(page index, payload) for every page; cached pages are read, not fetched."""
    token: str | None = None
    page = 0
    while True:
        path = cache_dir / f"page_{page:04d}.json"
        if not path.exists():
            logger.info("fetching page %d", page)
        payload = corpus.cached_get(path, API_URL, page_params(token), getter)
        yield page, payload
        nxt = payload.get("nextPageToken")
        if not payload.get("studies") or not nxt or nxt == token:
            break
        token = str(nxt)
        page += 1


def _pmid(value) -> str | None:
    s = str(value or "").strip()
    return s if s.isdigit() and int(s) > 0 else None


def start_year(date: str | None) -> int | None:
    m = re.match(r"^\s*(\d{4})", str(date or ""))
    return int(m.group(1)) if m else None


def parse_study(item: dict, page: int | None = None) -> dict | None:
    """Raw v2 study -> compact row; None without an NCT ID."""
    ps = item.get("protocolSection") or {}
    nct = (ps.get("identificationModule") or {}).get("nctId")
    if not nct:
        return None
    date = ((ps.get("statusModule") or {}).get("startDateStruct") or {}).get("date")
    design = ps.get("designModule") or {}
    interventions = (ps.get("armsInterventionsModule") or {}).get("interventions") or []
    refs = (ps.get("referencesModule") or {}).get("references") or []
    return {
        "nct": str(nct),
        "study_type": design.get("studyType"),
        "start_date": date,
        "start_year": start_year(date),
        "phases": list(design.get("phases") or []),
        "intervention_types": sorted({str(i.get("type")) for i in interventions if i.get("type")}),
        "page": page,
        "refs": [[_pmid(r.get("pmid")), r.get("type")] for r in refs],
    }


def fetch_studies(getter: Getter = corpus.http_get_json, cache_dir: Path = CACHE_DIR) -> tuple[list[dict], dict]:
    """All studies, deduplicated by NCT ID (first occurrence kept)."""
    seen: dict[str, dict] = {}
    stats = {"total_count": None, "n_pages": 0, "n_records": 0, "n_no_nct": 0, "n_duplicate_nct": 0}
    for page, payload in iter_pages(getter, cache_dir):
        if stats["total_count"] is None and payload.get("totalCount") is not None:
            stats["total_count"] = int(payload["totalCount"])
        stats["n_pages"] += 1
        for item in payload.get("studies") or []:
            stats["n_records"] += 1
            row = parse_study(item, page)
            if row is None:
                stats["n_no_nct"] += 1
            elif row["nct"] in seen:
                stats["n_duplicate_nct"] += 1
            else:
                seen[row["nct"]] = row
    studies = sorted(seen.values(), key=lambda r: r["nct"])
    stats["n_studies"] = len(studies)
    return studies, stats


def study_stats(studies: list[dict]) -> dict:
    """Marginal counts over the study table (references by type etc.)."""
    ref_types: Counter = Counter()
    ref_types_interventional: Counter = Counter()
    ref_types_with_pmid: Counter = Counter()
    for s in studies:
        for pmid, rtype in s["refs"]:
            ref_types[str(rtype)] += 1
            if pmid:
                ref_types_with_pmid[str(rtype)] += 1
            if s["study_type"] == "INTERVENTIONAL":
                ref_types_interventional[str(rtype)] += 1
    interventional = [s for s in studies if s["study_type"] == "INTERVENTIONAL"]
    bg = [s for s in interventional if any(p and t == "BACKGROUND" for p, t in s["refs"])]
    return {
        "study_types": dict(Counter(str(s["study_type"]) for s in studies).most_common()),
        "n_studies_any_reference": sum(1 for s in studies if s["refs"]),
        "n_studies_any_pmid_reference": sum(1 for s in studies if any(p for p, _ in s["refs"])),
        "n_references_by_type": dict(ref_types.most_common()),
        "n_references_with_pmid_by_type": dict(ref_types_with_pmid.most_common()),
        "n_references_by_type_interventional": dict(ref_types_interventional.most_common()),
        "n_studies_no_start_date": sum(1 for s in studies if s["start_year"] is None),
        "n_interventional": len(interventional),
        "n_interventional_with_background_pmid": len(bg),
        "n_interventional_with_background_pmid_no_start_date": sum(1 for s in bg if s["start_year"] is None),
    }


# ---------------------------------------------------------------- step 2


def build_index(studies: Iterable[dict]) -> list[dict]:
    """PMID -> interventional studies citing it as BACKGROUND (one entry per study)."""
    index: dict[str, dict[str, dict]] = {}
    for s in studies:
        if s["study_type"] != "INTERVENTIONAL":
            continue
        for pmid, rtype in s["refs"]:
            if not pmid or rtype != "BACKGROUND":
                continue
            index.setdefault(pmid, {}).setdefault(s["nct"], {
                "nct": s["nct"],
                "start_year": s["start_year"],
                "phases": list(s["phases"]),
                "intervention_types": list(s["intervention_types"]),
            })
    return [{"pmid": p, "trials": [index[p][n] for n in sorted(index[p])]} for p in sorted(index, key=int)]


# ---------------------------------------------------------------- step 3


def is_drug(trial: dict) -> bool:
    return bool(DRUG_TYPES & set(trial.get("intervention_types") or []))


def is_ph2(trial: dict) -> bool:
    phases = set(trial.get("phases") or [])
    return bool(phases) and phases <= PH2_PHASES


def in_window(trial: dict, year: int) -> bool:
    y = trial.get("start_year")
    return y is not None and year - WINDOW_BEFORE <= y <= year + WINDOW_AFTER


def outcome_row(pmid: str, year: int, trials: list[dict]) -> dict:
    window = [t for t in trials if in_window(t, year)]
    dated = [t["start_year"] for t in trials if t.get("start_year") is not None]
    return {
        "pmid": str(pmid),
        "trial_bg_8y": int(bool(window)),
        "n_trials_bg_8y": len(window),
        "trial_bg_8y_drug": int(any(is_drug(t) for t in window)),
        "trial_bg_8y_ph2": int(any(is_ph2(t) for t in window)),
        "first_trial_year": min(dated) if dated else None,
        "n_trials_bg_undated": len(trials) - len(dated),
    }


def pathway_files(pathway: str) -> dict:
    base = META_DIR if pathway == "ras" else META_DIR / "pathways" / pathway
    return {"features": base / "features.jsonl", "papers": base / "papers.jsonl", "icite": base / "icite.jsonl"}


def eligible_pmids(features_path: Path) -> list[str]:
    """Eligible preclinical PMIDs; only pmid/group/eligible are read (blinding)."""
    out = []
    for row in read_jsonl(features_path):
        if row.get("group") == "preclinical" and row.get("eligible"):
            out.append(str(row["pmid"]))
    return sorted(set(out), key=int)


def paper_years(papers_path: Path, icite_path: Path | None, pmids: set[str]) -> tuple[dict[str, int | None], int]:
    """pmid -> year (papers.jsonl, iCite year if missing); also n from iCite."""
    years: dict[str, int | None] = {}
    for p in read_jsonl(papers_path):
        if str(p["pmid"]) in pmids:
            years[str(p["pmid"])] = corpus._to_year(p.get("year"))
    missing = {p for p in pmids if not years.get(p)}
    n_icite = 0
    if missing and icite_path is not None and icite_path.exists():
        for r in read_jsonl(icite_path):
            pm = str(r.get("pmid"))
            if pm in missing:
                y = corpus._to_year(r.get("year"))
                if y:
                    years[pm] = y
                    n_icite += 1
    return years, n_icite


def compute_outcome(pmids: list[str], years: dict[str, int | None], index: dict[str, list[dict]]) -> tuple[list[dict], dict]:
    rows = []
    stats = {"no_year_skipped": 0, "n_linked_pairs_undated": 0, "n_ph12_in_window_only": 0}
    for pmid in pmids:
        y = years.get(pmid)
        if not y:
            stats["no_year_skipped"] += 1
            continue
        trials = index.get(pmid, [])
        row = outcome_row(pmid, y, trials)
        stats["n_linked_pairs_undated"] += row["n_trials_bg_undated"]
        window = [t for t in trials if in_window(t, y)]
        # How many papers would flip if phase 1/2 counted as phase 2+ (reported only).
        if not row["trial_bg_8y_ph2"] and any("PHASE2" in (t.get("phases") or []) for t in window):
            stats["n_ph12_in_window_only"] += 1
        rows.append(row)
    return rows, stats


def marginals(rows: list[dict]) -> dict:
    n = len(rows)
    k = sum(r["trial_bg_8y"] for r in rows)
    return {
        "n_eligible": n,
        "n_trial_bg_8y": k,
        "rate_trial_bg_8y": k / n if n else None,
        "n_trial_bg_8y_drug": sum(r["trial_bg_8y_drug"] for r in rows),
        "n_trial_bg_8y_ph2": sum(r["trial_bg_8y_ph2"] for r in rows),
        "n_linked_any_year": sum(1 for r in rows if r["first_trial_year"] is not None or r["n_trials_bg_undated"]),
    }


# ---------------------------------------------------------------- step 4

_STOP = frozenset(
    "the and for with from into that this are was were its their than via using use not but which after during "
    "between through under over has have been can may all our other both among two one novel new role".split()
)


def tokens(text: str) -> set[str]:
    text = re.sub(r"<[^>]+>", " ", str(text or "")).casefold()
    return {t for t in re.findall(r"[a-z0-9]+", text) if len(t) >= 3 and t not in _STOP}


def title_overlap(title: str, citation: str) -> float | None:
    """Share of the title's content tokens that appear in the citation."""
    tt = tokens(title)
    if not tt:
        return None
    return len(tt & tokens(citation)) / len(tt)


def sample_pairs(pairs: list[tuple[str, str, str]], n: int = N_SAMPLE, seed: int = 0) -> list[tuple[str, str, str]]:
    """``n`` pairs from a deterministically sorted list, numpy default_rng(seed)."""
    ordered = sorted(set(pairs), key=lambda t: (PATHWAYS.index(t[0]), int(t[1]), t[2]))
    if len(ordered) <= n:
        return ordered
    idx = np.random.default_rng(seed).choice(len(ordered), size=n, replace=False)
    return [ordered[i] for i in idx]


def reference_entries(nct: str, page: int, pmid: str, cache_dir: Path = CACHE_DIR) -> list[dict]:
    payload = json.loads((cache_dir / f"page_{page:04d}.json").read_text(encoding="utf-8"))
    for item in payload.get("studies") or []:
        ps = item.get("protocolSection") or {}
        if (ps.get("identificationModule") or {}).get("nctId") == nct:
            refs = (ps.get("referencesModule") or {}).get("references") or []
            return [r for r in refs if _pmid(r.get("pmid")) == pmid]
    return []


# ---------------------------------------------------------------- main


def _write_manifest(manifest: dict, path: Path = MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    TRIALS_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict = {}
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest.setdefault("fetch_date", _dt.date.today().isoformat())
    manifest["last_run"] = _dt.datetime.now().isoformat(timespec="seconds")

    # step 1
    studies, fstats = fetch_studies()
    write_jsonl(STUDIES_PATH, studies)
    manifest["fetch"] = {"url": API_URL, "pageSize": PAGE_SIZE, "fields": FIELDS, **fstats}
    manifest["studies"] = study_stats(studies)
    _write_manifest(manifest)
    logger.info("studies: %d (totalCount %s)", fstats["n_studies"], fstats["total_count"])

    # step 2
    index_rows = build_index(studies)
    write_jsonl(INDEX_PATH, index_rows)
    manifest["pmid_index"] = {
        "definition": "PMIDs cited as BACKGROUND by INTERVENTIONAL studies",
        "n_pmids": len(index_rows),
        "n_pairs": sum(len(r["trials"]) for r in index_rows),
    }
    _write_manifest(manifest)
    index = {r["pmid"]: r["trials"] for r in index_rows}
    page_of = {s["nct"]: s["page"] for s in studies}
    del studies

    # step 3
    manifest["outcome"] = {
        "definition": "trial_bg_8y = BACKGROUND ref of an INTERVENTIONAL study with start_year in [Y-1, Y+8]; "
        "drug = DRUG or BIOLOGICAL intervention; ph2 = all phases in {PHASE2, PHASE3, PHASE4}; "
        "studies without a start date excluded from the window",
        "per_pathway": {},
    }
    taken: set[str] = set()
    pooled: list[dict] = []
    pairs: list[tuple[str, str, str]] = []
    titles: dict[str, str] = {}
    for pathway in PATHWAYS:
        files = pathway_files(pathway)
        pmids = eligible_pmids(files["features"])
        years, n_icite = paper_years(files["papers"], files["icite"], set(pmids))
        rows, ostats = compute_outcome(pmids, years, index)
        write_jsonl(TRIALS_DIR / f"outcome_{pathway}.jsonl", rows)
        manifest["outcome"]["per_pathway"][pathway] = {**marginals(rows), "year_from_icite": n_icite, **ostats}
        for r in rows:
            if r["pmid"] in taken:
                continue
            taken.add(r["pmid"])
            pooled.append(r)
            if r["trial_bg_8y"]:
                y = years[r["pmid"]]
                pairs.extend((pathway, r["pmid"], t["nct"]) for t in index[r["pmid"]] if in_window(t, y))
        for p in read_jsonl(files["papers"]):
            if str(p["pmid"]) in index:
                titles.setdefault(str(p["pmid"]), p.get("title") or "")
        logger.info("%s: %s", pathway, manifest["outcome"]["per_pathway"][pathway])
    manifest["outcome"]["pooled_dedup"] = {**marginals(pooled), "order": list(PATHWAYS), "n_linked_pairs_in_window": len(pairs)}
    _write_manifest(manifest)

    # step 4
    sample = []
    for pathway, pmid, nct in sample_pairs(pairs):
        entries = reference_entries(nct, page_of[nct], pmid)
        title = titles.get(pmid, "")
        citation = " | ".join(str(e.get("citation") or "") for e in entries)
        ov = title_overlap(title, citation)
        sample.append({
            "pathway": pathway,
            "pmid": pmid,
            "nct": nct,
            "reference_entries": entries,
            "citation": citation,
            "paper_title": title,
            "title_token_overlap": None if ov is None else round(ov, 3),
            "auto_match": ov is not None and ov >= MATCH_THRESHOLD,
        })
    n_match = sum(s["auto_match"] for s in sample)
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_PATH.write_text(json.dumps({
        "description": "Random linked (paper, trial) pairs with trial_bg_8y=1, pooled over ras, egfr, pi3k "
        "(paper kept in its first pathway), numpy default_rng(0); for manual verification",
        "n_pairs_population": len(set(pairs)),
        "auto_match_rule": f"share of title content tokens found in the citation >= {MATCH_THRESHOLD}",
        "n_auto_match": n_match,
        "n_sample": len(sample),
        "pairs": sample,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest["linkage_sample"] = {"path": str(SAMPLE_PATH.relative_to(config.DATA.parent)),
                                  "n_sample": len(sample), "n_auto_match": n_match,
                                  "auto_match_rate": n_match / len(sample) if sample else None}
    _write_manifest(manifest)
    logger.info("linkage auto-match %d/%d", n_match, len(sample))


if __name__ == "__main__":
    main()
