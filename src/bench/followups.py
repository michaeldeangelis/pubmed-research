"""Attach later papers to frozen claims. Does not assign outcomes."""

from __future__ import annotations

import json
from pathlib import Path

from src.bench.sample import HORIZON_YEARS, is_review, load_papers
from src.config import ROOT

CLAIMS_PATH = ROOT / "bench" / "claims.json"
PACKETS_PATH = ROOT / "bench" / "packets.json"
MAX_ARTICLES = 8
MAX_REVIEWS = 3


def followups_for(claim: dict, papers: list[dict]) -> dict:
    cutoff = claim["cutoff"]
    year = int(cutoff[:4])
    gene = claim["gene"]
    diseases = set(claim.get("diseases") or [])
    drugs = set(claim.get("drugs") or [])
    articles = []
    reviews = []
    for paper in papers:
        if str(paper.get("pmid")) == str(claim["pmid"]):
            continue
        if int(paper.get("year") or 0) < year:
            continue
        if int(paper.get("year") or 0) > year + HORIZON_YEARS:
            continue
        if (paper.get("date") or "9999") < cutoff and int(paper.get("year") or 0) == year:
            continue
        if gene not in (paper.get("genes") or []):
            continue
        paper_diseases = set(paper.get("diseases") or [])
        paper_drugs = set(paper.get("drugs") or [])
        disease_hit = bool(diseases and diseases & paper_diseases)
        drug_hit = bool(drugs and drugs & paper_drugs)
        text = f"{paper.get('title') or ''} {paper.get('abstract') or ''}".lower()
        mention = any(name.lower() in text for name in diseases | drugs)
        if (diseases or drugs) and not (disease_hit or drug_hit or mention):
            continue
        record = {
            "pmid": str(paper.get("pmid")),
            "date": paper.get("date") or "",
            "year": int(paper["year"]),
            "title": paper.get("title") or "",
            "abstract": paper.get("abstract") or "",
            "pub_types": list(paper.get("pub_types") or []),
            "genes": list(paper.get("genes") or []),
            "drugs": list(paper.get("drugs") or []),
            "diseases": list(paper.get("diseases") or []),
        }
        if is_review(paper):
            reviews.append(record)
        else:
            articles.append(record)
    articles.sort(key=lambda row: (row["date"], row["pmid"]))
    reviews.sort(key=lambda row: (row["date"], row["pmid"]))
    return {
        "articles": articles[:MAX_ARTICLES],
        "reviews": reviews[:MAX_REVIEWS],
        "n_articles_in_window": len(articles),
        "n_reviews_in_window": len(reviews),
    }


def build_packets(claims: list[dict], papers: list[dict]) -> list[dict]:
    packets = []
    for claim in claims:
        if claim.get("skip"):
            continue
        packets.append({**claim, "later": followups_for(claim, papers)})
    return packets


def main() -> None:
    claims = json.loads(CLAIMS_PATH.read_text(encoding="utf-8"))
    rows = claims["claims"] if isinstance(claims, dict) else claims
    packets = build_packets(rows, load_papers())
    PACKETS_PATH.write_text(json.dumps({"n": len(packets), "packets": packets}, indent=2) + "\n", encoding="utf-8")
    print(f"{PACKETS_PATH} n={len(packets)}")


if __name__ == "__main__":
    main()
