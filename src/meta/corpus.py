"""Metascience corpus: PMIDs, Europe PMC metadata, iCite records, outcome B.

Run as ``python -m src.meta.corpus``. Steps:

1. Europe PMC search with the ingest query from ``src/ingest/pubmed.py`` and
   ``PUB_YEAR:[2000 TO 2015]``, cursor paging (``resultType=core``,
   ``pageSize=1000``). Writes ``data/meta/papers.jsonl`` as soon as the step
   finishes.
2. iCite ``/api/pubs`` in batches of at most 200 PMIDs (500 returns HTTP
   413). All returned fields are kept, and ``pmid`` is written as a string.
   Writes ``data/meta/icite.jsonl``.
3. Outcome B. Rows are written for iCite research articles only
   (``is_research_article`` true); other papers get no row. For each one,
   the publication years of the PMIDs in ``cited_by_clin`` are looked up in
   iCite (batched, cached in ``data/meta/citing_years.json``). With
   ``Y`` = the paper's publication year (Europe PMC ``pubYear``, falling back
   to the iCite year when that is missing), a clinical citer counts when
   ``Y - 1 <= citer_year <= Y + 8``. The ``-1`` lower bound tolerates epub
   skew: a citer's year can come before the focal paper's print year when
   the focal paper was available online first. Citers that iCite cannot
   date are ignored.
   ``clin_cited_8y`` is 1 if any citer falls in the window, ``n_clin_8y``
   is the count in the window, and ``first_clin_year`` is the minimum year
   over all dated clinical citers (any year), or null if there are none.

Raw responses are cached under ``data/meta/cache/``, so reruns do not hit
the network. Fetch date, queries and counts go to ``data/meta/manifest.json``.

Blinding: this module reports only marginal counts of its own tables. It
never relates a design feature to an outcome.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Callable, Iterable

import requests

from src import config
from src.ingest import pubmed

logger = logging.getLogger(__name__)

META_DIR = config.DATA / "meta"
CACHE_DIR = META_DIR / "cache"
PAPERS_PATH = META_DIR / "papers.jsonl"
ICITE_PATH = META_DIR / "icite.jsonl"
CITING_YEARS_PATH = META_DIR / "citing_years.json"
OUTCOME_B_PATH = META_DIR / "outcome_b.jsonl"
MANIFEST_PATH = META_DIR / "manifest.json"

ICITE_URL = "https://icite.od.nih.gov/api/pubs"
YEAR_RANGE = "[2000 TO 2015]"
QUERY = pubmed._QUERY.format(year=YEAR_RANGE)
PAGE_SIZE = 1000
ICITE_BATCH = 200
WINDOW_BEFORE = 1  # epub skew tolerance
WINDOW_AFTER = 8
SLEEP = 0.2
TRIES = 3
BACKOFF = 2.0
HEADERS = {"User-Agent": "ras-mapk-metascience-v1"}

Getter = Callable[[str, dict], dict]


# ---------------------------------------------------------------- HTTP


def http_get_json(url: str, params: dict) -> dict:
    """GET with polite sleep and retry on transient errors (3 tries)."""
    last: Exception | None = None
    for attempt in range(TRIES):
        time.sleep(SLEEP)
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=120)
            if response.status_code == 429 or response.status_code >= 500:
                raise requests.HTTPError(f"transient HTTP {response.status_code}")
            response.raise_for_status()
            return response.json()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError, ValueError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and 400 <= status < 500 and status != 429:
                raise
            last = exc
            wait = BACKOFF * (2**attempt)
            logger.warning("GET %s failed (%s); retry in %.0fs", url, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"GET {url} failed after {TRIES} tries") from last


def cached_get(cache_path: Path, url: str, params: dict, getter: Getter) -> dict:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    payload = getter(url, params)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(cache_path)
    return payload


# ---------------------------------------------------------------- step 1


def parse_paper(item: dict) -> dict:
    """Europe PMC core record -> papers.jsonl row (reuses ingest parsers)."""
    base = pubmed.parse_europepmc_result(item)
    authors = (item.get("authorList") or {}).get("author") if isinstance(item.get("authorList"), dict) else None
    doi = item.get("doi")
    return {
        "pmid": base["pmid"],
        "doi": str(doi).strip() if doi else None,
        "year": base["year"] or None,
        "journal": base["journal"],
        "title": base["title"],
        "abstract": base["abstract"],
        "mesh": base["mesh"],
        "pub_types": base["pub_types"],
        "n_authors": len(pubmed._as_list(authors)),
    }


def fetch_papers(getter: Getter = http_get_json, cache_dir: Path = CACHE_DIR) -> tuple[list[dict], int | None]:
    """All SRC:MED hits for the query, deduplicated by PMID, in PMID order."""
    cursor = "*"
    page = 0
    papers: dict[str, dict] = {}
    hit_count: int | None = None
    while True:
        params = {
            "query": QUERY,
            "format": "json",
            "resultType": "core",
            "pageSize": PAGE_SIZE,
            "cursorMark": cursor,
        }
        payload = cached_get(cache_dir / "europepmc" / f"page_{page:04d}.json", config.EUROPEPMC, params, getter)
        if hit_count is None and payload.get("hitCount") is not None:
            hit_count = int(payload["hitCount"])
        raw = (payload.get("resultList") or {}).get("result") or []
        if isinstance(raw, dict):
            raw = [raw]
        for item in raw:
            if not isinstance(item, dict) or item.get("pmid") in (None, ""):
                continue
            paper = parse_paper(item)
            papers.setdefault(paper["pmid"], paper)
        next_cursor = str(payload.get("nextCursorMark") or "")
        if not raw or not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
        page += 1
    ordered = [papers[k] for k in sorted(papers, key=int)]
    return ordered, hit_count


# ---------------------------------------------------------------- step 2


def _batches(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def fetch_icite(
    pmids: Iterable[str],
    getter: Getter = http_get_json,
    cache_dir: Path = CACHE_DIR,
    fields: str | None = None,
    subdir: str = "icite",
) -> list[dict]:
    """iCite records for PMIDs, batched by 200; pmid normalised to str."""
    ids = sorted({str(p) for p in pmids}, key=int)
    out: list[dict] = []
    for batch in _batches(ids, ICITE_BATCH):
        params = {"pmids": ",".join(batch)}
        if fields:
            params["fl"] = fields
        key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
        payload = cached_get(cache_dir / subdir / f"{key}.json", ICITE_URL, params, getter)
        for rec in payload.get("data") or []:
            rec = dict(rec)
            rec["pmid"] = str(rec.get("pmid"))
            out.append(rec)
    out.sort(key=lambda r: int(r["pmid"]))
    return out


# ---------------------------------------------------------------- step 3


def _to_year(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def citing_years(
    pmids: Iterable[str],
    getter: Getter = http_get_json,
    cache_path: Path = CITING_YEARS_PATH,
    cache_dir: Path = CACHE_DIR,
) -> dict[str, int | None]:
    """pmid -> iCite year for citers; unknown PMIDs map to None (cached too)."""
    known: dict[str, int | None] = {}
    if cache_path.exists():
        known = json.loads(cache_path.read_text(encoding="utf-8"))
    wanted = {str(p) for p in pmids}
    missing = sorted(wanted - set(known), key=int)
    if missing:
        found = fetch_icite(missing, getter=getter, cache_dir=cache_dir, fields="pmid,year", subdir="icite_citers")
        got = {r["pmid"]: _to_year(r.get("year")) for r in found}
        for p in missing:
            known[p] = got.get(p)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(dict(sorted(known.items(), key=lambda kv: int(kv[0])))), encoding="utf-8")
    return {p: known.get(p) for p in wanted}


def outcome_b_row(pmid: str, paper_year: int, citer_years: Iterable[int | None]) -> dict:
    years = [y for y in citer_years if y is not None]
    lo, hi = paper_year - WINDOW_BEFORE, paper_year + WINDOW_AFTER
    n = sum(1 for y in years if lo <= y <= hi)
    return {
        "pmid": str(pmid),
        "clin_cited_8y": int(n > 0),
        "n_clin_8y": n,
        "first_clin_year": min(years) if years else None,
    }


def compute_outcome_b(
    papers: list[dict], icite: list[dict], years: dict[str, int | None]
) -> tuple[list[dict], dict]:
    """Outcome B rows for iCite research articles only."""
    paper_year = {p["pmid"]: p.get("year") for p in papers}
    rows: list[dict] = []
    stats = {"year_from_icite": 0, "year_mismatch_epmc_vs_icite": 0, "no_year_skipped": 0}
    for rec in icite:
        if not rec.get("is_research_article"):
            continue
        pmid = rec["pmid"]
        y = paper_year.get(pmid)
        iy = _to_year(rec.get("year"))
        if y and iy and y != iy:
            stats["year_mismatch_epmc_vs_icite"] += 1
        if not y:
            y = iy
            stats["year_from_icite"] += 1
        if not y:
            stats["no_year_skipped"] += 1
            continue
        citers = [str(c) for c in rec.get("cited_by_clin") or []]
        rows.append(outcome_b_row(pmid, y, [years.get(c) for c in citers]))
    rows.sort(key=lambda r: int(r["pmid"]))
    return rows, stats


# ---------------------------------------------------------------- IO


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    META_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict = {}
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest.setdefault("fetch_date", _dt.date.today().isoformat())
    manifest["last_run"] = _dt.datetime.now().isoformat(timespec="seconds")
    manifest["europepmc"] = {"url": config.EUROPEPMC, "query": QUERY, "resultType": "core", "pageSize": PAGE_SIZE}

    papers, hit_count = fetch_papers()
    n_papers = write_jsonl(PAPERS_PATH, papers)
    manifest["europepmc"].update({"hitCount": hit_count, "n_papers": n_papers})
    _write_manifest(manifest)
    logger.info("papers.jsonl: %d rows (hitCount %s)", n_papers, hit_count)

    icite = fetch_icite([p["pmid"] for p in papers])
    n_icite = write_jsonl(ICITE_PATH, icite)
    n_research = sum(1 for r in icite if r.get("is_research_article"))
    n_clinical = sum(1 for r in icite if r.get("is_research_article") and r.get("is_clinical"))
    manifest["icite"] = {
        "url": ICITE_URL,
        "batch": ICITE_BATCH,
        "n_requested": n_papers,
        "n_returned": n_icite,
        "n_research_articles": n_research,
        "n_research_clinical": n_clinical,
        "n_research_preclinical": n_research - n_clinical,
    }
    _write_manifest(manifest)
    logger.info("icite.jsonl: %d rows, %d research articles", n_icite, n_research)

    research = [r for r in icite if r.get("is_research_article")]
    citers = {str(c) for r in research for c in r.get("cited_by_clin") or []}
    years = citing_years(citers)
    rows, stats = compute_outcome_b(papers, icite, years)
    n_b = write_jsonl(OUTCOME_B_PATH, rows)
    positive = sum(r["clin_cited_8y"] for r in rows)
    manifest["outcome_b"] = {
        "definition": "clin_cited_8y = any iCite cited_by_clin citer with year in [Y-1, Y+8], "
        "Y = Europe PMC pubYear (iCite year if missing); -1 tolerates epub skew",
        "rows": "iCite research articles only",
        "n_rows": n_b,
        "n_clinical_citers": len(citers),
        "n_clinical_citers_undated": sum(1 for c in citers if years.get(c) is None),
        "n_clin_cited_8y": positive,
        "rate_clin_cited_8y": positive / n_b if n_b else None,
        **stats,
    }
    _write_manifest(manifest)
    logger.info("outcome_b.jsonl: %d rows, clin_cited_8y rate %.4f", n_b, positive / n_b if n_b else float("nan"))


if __name__ == "__main__":
    main()
