"""Freeze source papers before any later paper is examined.

Eligible years are 2006–2014 so each claim has a later window inside the corpus.
Selection uses the source record only: year, publication type, and whether the
abstract states an effect. Citation rank and subsequent papers are not inputs.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from src.config import ENRICHED_PATH, ROOT

SEED = 20260927
SOURCE_YEAR_MIN = 2006
SOURCE_YEAR_MAX = 2014
PER_YEAR = 8
MIN_ABSTRACT = 400
HORIZON_YEARS = 8

SAMPLE_PATH = ROOT / "bench" / "sample.json"

_REVIEW_TYPES = ("review", "editorial", "comment", "letter", "erratum", "retraction")
_CUE = re.compile(
    r"\b(associated|association|inhibit\w*|resistant|resistance|survival|"
    r"response|predict\w*|prognos\w*|reduc\w*|increas\w*|sensitiv\w*|"
    r"no significant|failed|failure|correlat\w*)\b",
    re.I,
)


def is_review(paper: dict) -> bool:
    types = " ".join(paper.get("pub_types") or []).lower()
    return any(label in types for label in _REVIEW_TYPES)


def is_source_candidate(paper: dict) -> bool:
    year = int(paper.get("year") or 0)
    abstract = paper.get("abstract") or ""
    if year < SOURCE_YEAR_MIN or year > SOURCE_YEAR_MAX:
        return False
    if len(abstract) < MIN_ABSTRACT:
        return False
    if not paper.get("genes"):
        return False
    if is_review(paper):
        return False
    text = f"{paper.get('title') or ''} {abstract}"
    return _CUE.search(text) is not None


def select_sources(papers: list[dict], per_year: int = PER_YEAR, seed: int = SEED) -> list[dict]:
    """Stratified sample. PMID order is the only tie break. Returns source fields only."""
    by_year: dict[int, list[dict]] = {}
    for paper in papers:
        if is_source_candidate(paper):
            by_year.setdefault(int(paper["year"]), []).append(paper)
    chosen: list[dict] = []
    for year in range(SOURCE_YEAR_MIN, SOURCE_YEAR_MAX + 1):
        pool = sorted(by_year.get(year, []), key=lambda paper: str(paper.get("pmid")))
        rng = random.Random(seed + year)
        take = min(per_year, len(pool))
        picks = pool if take == len(pool) else rng.sample(pool, take)
        chosen.extend(sorted(picks, key=lambda paper: str(paper.get("pmid"))))
    return [_source_record(paper) for paper in chosen]


def _source_record(paper: dict) -> dict:
    return {
        "pmid": str(paper.get("pmid")),
        "year": int(paper["year"]),
        "date": paper.get("date") or f"{paper['year']}-01-01",
        "title": paper.get("title") or "",
        "abstract": paper.get("abstract") or "",
        "genes": list(paper.get("genes") or []),
        "drugs": list(paper.get("drugs") or []),
        "diseases": list(paper.get("diseases") or []),
        "pub_types": list(paper.get("pub_types") or []),
        "journal": paper.get("journal") or "",
    }


def load_papers(path: Path = ENRICHED_PATH) -> list[dict]:
    papers = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                papers.append(json.loads(line))
    return papers


def main() -> None:
    sample = select_sources(load_papers())
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": SEED,
        "per_year": PER_YEAR,
        "source_years": [SOURCE_YEAR_MIN, SOURCE_YEAR_MAX],
        "outcomes_consulted": False,
        "n": len(sample),
        "papers": sample,
    }
    SAMPLE_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"{SAMPLE_PATH} n={len(sample)}")


if __name__ == "__main__":
    main()
