"""Fetch RAS/MAPK papers from Europe PMC and write papers.jsonl."""

from __future__ import annotations

import json
import logging
import re
import time

import requests

from src import config
from src.schema import strip_markup

logger = logging.getLogger(__name__)

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_QUERY = (
    '(TITLE_ABS:"KRAS" OR TITLE_ABS:"NRAS" OR TITLE_ABS:"HRAS" OR TITLE_ABS:"BRAF" '
    'OR TITLE_ABS:"MAP2K1" OR TITLE_ABS:"MAP2K2" OR TITLE_ABS:"trametinib" '
    'OR TITLE_ABS:"sotorasib" OR TITLE_ABS:"vemurafenib" OR TITLE_ABS:"dabrafenib") '
    "AND PUB_YEAR:{year} AND SRC:MED"
)
_HEADERS = {"User-Agent": "ras-mapk-v1"}
_made_request = False


def _as_list(value):
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    return [value]


def _node_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if value.get("#text") is not None:
            return str(value["#text"])
        if value.get("value") is not None:
            return str(value["value"])
        return ""
    return str(value)


def _journal(item: dict) -> str:
    info = item.get("journalInfo") or {}
    if not isinstance(info, dict):
        return ""
    journal = info.get("journal") or {}
    if not isinstance(journal, dict):
        return str(journal) if journal else ""
    return str(journal.get("title") or "")


def _mesh(item: dict) -> list[str]:
    headings = (item.get("meshHeadingList") or {}).get("meshHeading")
    names: list[str] = []
    for heading in _as_list(headings):
        if isinstance(heading, dict):
            text = _node_text(heading.get("descriptorName"))
        else:
            text = _node_text(heading)
        if text:
            names.append(text)
    return names


def _pub_types(item: dict) -> list[str]:
    raw = (item.get("pubTypeList") or {}).get("pubType")
    types: list[str] = []
    for entry in _as_list(raw):
        text = _node_text(entry)
        if text:
            types.append(text)
    return types


def _date_and_year(item: dict) -> tuple[str, int]:
    raw_date = str(item.get("firstPublicationDate") or "")
    pub_year = item.get("pubYear")
    pub_year_s = "" if pub_year is None else str(pub_year).strip()
    if _DATE_RE.fullmatch(raw_date):
        date = raw_date
    elif pub_year_s:
        date = f"{pub_year_s}-01-01"
    else:
        date = ""
    year: int | None = None
    if pub_year_s:
        try:
            year = int(float(pub_year_s)) if "." in pub_year_s else int(pub_year_s)
        except ValueError:
            year = None
    if year is None and len(date) >= 4 and date[:4].isdigit():
        year = int(date[:4])
    return date, 0 if year is None else year


def parse_europepmc_result(item: dict) -> dict:
    raw_pmid = item.get("pmid", "")
    date, year = _date_and_year(item)
    return {
        "pmid": "" if raw_pmid is None else str(raw_pmid),
        "date": date,
        "year": year,
        "title": strip_markup(str(item.get("title") or "")),
        "abstract": strip_markup(str(item.get("abstractText") or "")),
        "journal": _journal(item),
        "mesh": _mesh(item),
        "pub_types": _pub_types(item),
    }


def _search(params: dict) -> dict:
    global _made_request
    if _made_request:
        time.sleep(0.15)
    _made_request = True
    response = requests.get(
        config.EUROPEPMC,
        params=params,
        headers=_HEADERS,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _fetch_year(year: int, limit: int, seen: set[str]) -> list[dict]:
    cursor = "*"
    kept: list[dict] = []
    while len(kept) < limit:
        payload = _search(
            {
                "query": _QUERY.format(year=year),
                "format": "json",
                "resultType": "core",
                "pageSize": 100,
                "cursorMark": cursor,
                "sort": "CITED desc",
            }
        )
        raw = (payload.get("resultList") or {}).get("result") or []
        if isinstance(raw, dict):
            raw = [raw]
        for item in raw:
            if len(kept) >= limit:
                break
            if not isinstance(item, dict) or item.get("pmid") in (None, ""):
                continue
            paper = parse_europepmc_result(item)
            if not paper["pmid"] or not paper["abstract"].strip():
                continue
            if paper["pmid"] in seen:
                continue
            seen.add(paper["pmid"])
            kept.append(paper)
        if len(kept) >= limit:
            break
        next_cursor = str(payload.get("nextCursorMark") or "")
        if not raw or not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
    return kept


def fetch_papers(per_year: int | None = None) -> list[dict]:
    limit = config.PER_YEAR if per_year is None else per_year
    papers: list[dict] = []
    seen: set[str] = set()
    for year in range(config.YEAR_MIN, config.YEAR_MAX + 1):
        try:
            papers.extend(_fetch_year(year, limit, seen))
        except Exception:
            logger.exception("Europe PMC fetch failed for %s", year)
    return papers


def main() -> None:
    papers = fetch_papers()
    config.PAPERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.PAPERS_PATH.open("w", encoding="utf-8") as handle:
        for paper in papers:
            handle.write(json.dumps(paper, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
