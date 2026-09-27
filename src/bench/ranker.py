"""Claim-paper ranker over the pooled retrieval candidates.

The score uses four fixed checks and no fitted weights: population,
intervention or exposure, measured outcome, and whether the passage reports
a new result. The current twenty claims are development evidence. The frozen
route lists are not rewritten.
"""

from __future__ import annotations

import re

from src.bench.graph import BUDGET, _WORD

_META = re.compile(r"meta-analysis|systematic review|literature review", re.I)
_NEW = re.compile(
    r"\b(we identified|we found|we report|our results|our data|this study|patients were)\b",
    re.I,
)
_POOL = ("same_gene", "text_similarity", "citation", "experimental", "shared_references")


def _terms(text: str) -> set[str]:
    return {word.lower() for word in _WORD.findall(text or "")}


def _coverage(needles: set[str], haystack: set[str]) -> float | None:
    if not needles:
        return None
    return len(needles & haystack) / len(needles)


def compatibility(spec: dict, paper: dict) -> dict:
    """Score one paper against one claim. Higher means a closer match."""
    blob = f"{paper.get('title') or ''} {paper.get('abstract') or ''}"
    words = _terms(blob)
    population = _coverage(_terms(spec.get("setting") or ""), words)
    intervention = _coverage(_terms(spec.get("intervention") or ""), words)
    outcome_terms = _terms(spec.get("outcome") or "") - _terms(spec.get("intervention") or "") - _terms(spec.get("setting") or "")
    outcome = _coverage(outcome_terms, words)
    if _META.search(blob):
        new_result = 0.0
    elif _NEW.search(paper.get("abstract") or ""):
        new_result = 1.0
    else:
        new_result = 0.4
    parts = [value for value in (population, intervention, outcome, new_result) if value is not None]
    score = sum(parts) / len(parts) if parts else 0.0
    return {
        "score": round(score, 4),
        "population": None if population is None else round(population, 4),
        "intervention": None if intervention is None else round(intervention, 4),
        "outcome": None if outcome is None else round(outcome, 4),
        "new_result": new_result,
    }


def select_twenty(claim: dict) -> list[str]:
    """Top 20 papers from the pooled route lists for one claim."""
    papers = {row["pmid"]: row for row in claim.get("neighbors") or []}
    text_rank = {pmid: index for index, pmid in enumerate(claim["lists"].get("text_similarity") or [])}
    pool: list[str] = []
    seen: set[str] = set()
    for method in _POOL:
        for pmid in claim["lists"].get(method) or []:
            if pmid not in seen:
                seen.add(pmid)
                pool.append(pmid)
    scored = []
    for pmid in pool:
        detail = compatibility(claim["spec"], papers.get(pmid) or {})
        scored.append((detail["score"], -text_rank.get(pmid, 10_000), pmid))
    scored.sort(reverse=True)
    return [pmid for _score, _tie, pmid in scored[:BUDGET]]


def paired_reconciliation(rows: list[dict]) -> dict:
    """Show how per-claim ranker-minus-text differences add to the total."""
    diffs = {row["claim_id"]: row["paired_difference"] for row in rows}
    highlighted_ids = ("C033", "C016", "C035")
    highlighted = {claim_id: diffs.get(claim_id, 0) for claim_id in highlighted_ids}
    others = {claim_id: value for claim_id, value in diffs.items() if claim_id not in highlighted_ids and value}
    return {
        "alk_gain": highlighted["C033"],
        "c016_loss": highlighted["C016"],
        "c035_loss": highlighted["C035"],
        "highlighted_net": sum(highlighted.values()),
        "other_claims_net": sum(others.values()),
        "other_claims": others,
        "total": sum(diffs.values()),
    }


def development_comparison(claims: list[dict], judgments: dict[str, dict[str, str]]) -> dict:
    """Compare the ranker's twenty with text similarity on audited labels.

    This is development evidence. The twenty claims already shaped the rule.
    """
    from src.bench.freeze import _count_tests

    rows = []
    ranker_tests = 0
    text_tests = 0
    ranker_claims = 0
    text_claims = 0
    for claim in claims:
        if claim["claim_id"] not in judgments:
            continue
        judged = judgments[claim["claim_id"]]
        papers = {row["pmid"]: row for row in claim.get("neighbors") or []}
        picked = select_twenty(claim)
        text_ids = list(claim["lists"].get("text_similarity") or [])[:BUDGET]
        picked_stats = _count_tests(picked, papers, judged)
        text_stats = _count_tests(text_ids, papers, judged)
        ranker_tests += picked_stats["direct_tests"]
        text_tests += text_stats["direct_tests"]
        ranker_claims += int(picked_stats["direct_tests"] > 0)
        text_claims += int(text_stats["direct_tests"] > 0)
        rows.append(
            {
                "claim_id": claim["claim_id"],
                "ranker_tests": picked_stats["direct_tests"],
                "text_tests": text_stats["direct_tests"],
                "paired_difference": picked_stats["direct_tests"] - text_stats["direct_tests"],
                "ranker_pmids": sorted(picked_stats["test_pmids"]),
                "text_pmids": sorted(text_stats["test_pmids"]),
            }
        )
    return {
        "role": "development_evidence",
        "note": (
            "The twenty evaluation claims already influenced the audit and the "
            "ranking rule. These paired differences are development evidence, "
            "not a held-out test."
        ),
        "rule": "Equal-weight mean of population, intervention, outcome word overlap, and a new-result flag. No fitted weights.",
        "budget": BUDGET,
        "n_claims": len(rows),
        "ranker_direct_tests": ranker_tests,
        "text_direct_tests": text_tests,
        "paired_difference": ranker_tests - text_tests,
        "ranker_claims_with_a_test": ranker_claims,
        "text_claims_with_a_test": text_claims,
        "reconciliation": paired_reconciliation(rows),
        "claims": rows,
    }
