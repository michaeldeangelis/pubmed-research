"""Select the next paper with a frozen relevance score and a provenance discount.

The compatibility score is not refit. A candidate is discounted only when
language establishes that it reuses evidence already selected: the same
write-up, or an abstract that both names the other PMID and says the data
are reused. A bare PMID mention can be a replication or a disagreement.
A shared trial identifier marks a shared trial; cohorts, endpoints, and
analyses inside that trial may still differ, so it does not discount the
paper. Uncertain cohort identity gets no independence bonus.
"""

from __future__ import annotations

import re

from src.bench.freeze import same_report
from src.bench.graph import BUDGET
from src.bench.ranker import _POOL, compatibility

# Fixed discount for a paper that repeats a cohort already selected.
# Not searched against the twenty development claims.
SAME_COHORT_DISCOUNT = 0.25
BRAF_CLUSTER = ("C013", "C015", "C016")
_NCT = re.compile(r"NCT\d{8}")
_REUSE = re.compile(
    r"same (?:cohort|patients|population|dataset|data set)|"
    r"re-?analysis|extended follow-up|"
    r"our previous (?:cohort|patients|series|study)|"
    r"these same patients|previously described cohort",
    re.I,
)


def _blob(paper: dict) -> str:
    return f"{paper.get('title') or ''} {paper.get('abstract') or ''}"


def evidence_relation(paper: dict, prior: dict) -> str:
    """Return same_report, same_cohort, same_trial, or uncertain.

    same_cohort requires language that the data are reused. same_trial records
    a shared registry identifier and does not by itself mean a reused cohort.
    """
    if same_report(paper, prior):
        return "same_report"
    paper_id = str(paper.get("pmid") or "")
    prior_id = str(prior.get("pmid") or "")
    if paper.get("duplicate_of") == prior_id or prior.get("duplicate_of") == paper_id:
        return "same_report"
    paper_text = paper.get("abstract") or ""
    prior_text = prior.get("abstract") or ""
    names_other = (prior_id and prior_id in paper_text) or (paper_id and paper_id in prior_text)
    if names_other and (_REUSE.search(paper_text) or _REUSE.search(prior_text)):
        return "same_cohort"
    paper_trials = set(_NCT.findall(_blob(paper)))
    prior_trials = set(_NCT.findall(_blob(prior)))
    if paper_trials and prior_trials and paper_trials & prior_trials:
        return "same_trial"
    return "uncertain"


def _value(score: float, relations: list[str]) -> float:
    if "same_report" in relations:
        return 0.0
    if "same_cohort" in relations:
        return score * SAME_COHORT_DISCOUNT
    return score


def select_with_provenance(claim: dict) -> list[str]:
    """Greedy twenty. Relevance is the frozen score. Repeats are discounted."""
    papers = {row["pmid"]: row for row in claim.get("neighbors") or []}
    text_rank = {pmid: index for index, pmid in enumerate(claim["lists"].get("text_similarity") or [])}
    pool: list[str] = []
    seen: set[str] = set()
    for method in _POOL:
        for pmid in claim["lists"].get(method) or []:
            if pmid not in seen:
                seen.add(pmid)
                pool.append(pmid)
    ranked = []
    for pmid in pool:
        detail = compatibility(claim["spec"], papers.get(pmid) or {})
        ranked.append((detail["score"], -text_rank.get(pmid, 10_000), pmid))
    ranked.sort(reverse=True)
    selected: list[str] = []
    while len(selected) < BUDGET and ranked:
        best_index = None
        best_value = -1.0
        for index, (score, _tie, pmid) in enumerate(ranked):
            relations = [
                evidence_relation(papers.get(pmid) or {"pmid": pmid}, papers.get(prior) or {"pmid": prior})
                for prior in selected
            ]
            value = _value(score, relations)
            if value > best_value:
                best_value = value
                best_index = index
        if best_index is None or best_value <= 0:
            break
        selected.append(ranked.pop(best_index)[2])
    return selected


def evidence_units(pmids: set[str] | list[str], papers: dict[str, dict]) -> dict:
    """Collapse detected repeats. Unlinked papers stay unresolved, not confirmed independent."""
    ids = list(pmids)
    parent = {pmid: pmid for pmid in ids}

    def find(pmid: str) -> str:
        while parent[pmid] != pmid:
            parent[pmid] = parent[parent[pmid]]
            pmid = parent[pmid]
        return pmid

    for index, left in enumerate(ids):
        for right in ids[index + 1 :]:
            relation = evidence_relation(papers.get(left) or {"pmid": left}, papers.get(right) or {"pmid": right})
            if relation in {"same_report", "same_cohort"}:
                parent[find(right)] = find(left)
    clusters: dict[str, list[str]] = {}
    for pmid in ids:
        clusters.setdefault(find(pmid), []).append(pmid)
    return {
        "papers": len(ids),
        "units": len(clusters),
        "collapsed_repeats": len(ids) - len(clusters),
        "unresolved_singleton_units": sum(1 for members in clusters.values() if len(members) == 1),
        "repeat_clusters": sum(1 for members in clusters.values() if len(members) > 1),
    }


def _subset(rows: list[dict], exclude: set[str]) -> dict:
    kept = [row for row in rows if row["claim_id"] not in exclude]
    def total(key: str) -> int:
        return sum(row[key] for row in kept)
    return {
        "n_claims": len(kept),
        "text_tests": total("text_tests"),
        "ranker_tests": total("ranker_tests"),
        "provenance_tests": total("provenance_tests"),
        "text_units": total("text_units"),
        "ranker_units": total("ranker_units"),
        "provenance_units": total("provenance_units"),
        "text_claims": sum(row["text_tests"] > 0 for row in kept),
        "ranker_claims": sum(row["ranker_tests"] > 0 for row in kept),
        "provenance_claims": sum(row["provenance_tests"] > 0 for row in kept),
        "provenance_minus_text": total("provenance_tests") - total("text_tests"),
        "provenance_minus_ranker": total("provenance_tests") - total("ranker_tests"),
    }


def compare_selectors(claims: list[dict], judgments: dict[str, dict[str, str]]) -> dict:
    """Text, frozen ranker, and provenance-aware selection on the same pools."""
    from src.bench.freeze import _count_tests
    from src.bench.ranker import select_twenty

    rows = []
    for claim in claims:
        if claim["claim_id"] not in judgments:
            continue
        judged = judgments[claim["claim_id"]]
        papers = {row["pmid"]: row for row in claim.get("neighbors") or []}
        text_ids = list(claim["lists"].get("text_similarity") or [])[:BUDGET]
        ranker_ids = select_twenty(claim)
        provenance_ids = select_with_provenance(claim)
        text_stats = _count_tests(text_ids, papers, judged)
        ranker_stats = _count_tests(ranker_ids, papers, judged)
        provenance_stats = _count_tests(provenance_ids, papers, judged)
        text_units = evidence_units(text_stats["test_pmids"], papers)
        ranker_units = evidence_units(ranker_stats["test_pmids"], papers)
        provenance_units = evidence_units(provenance_stats["test_pmids"], papers)
        rows.append(
            {
                "claim_id": claim["claim_id"],
                "text_tests": text_stats["direct_tests"],
                "ranker_tests": ranker_stats["direct_tests"],
                "provenance_tests": provenance_stats["direct_tests"],
                "provenance_minus_text": provenance_stats["direct_tests"] - text_stats["direct_tests"],
                "provenance_minus_ranker": provenance_stats["direct_tests"] - ranker_stats["direct_tests"],
                "ranker_minus_text": ranker_stats["direct_tests"] - text_stats["direct_tests"],
                "text_units": text_units["units"],
                "ranker_units": ranker_units["units"],
                "provenance_units": provenance_units["units"],
                "text_collapsed_repeats": text_units["collapsed_repeats"],
                "ranker_collapsed_repeats": ranker_units["collapsed_repeats"],
                "provenance_collapsed_repeats": provenance_units["collapsed_repeats"],
                "provenance_pmids": sorted(provenance_stats["test_pmids"]),
            }
        )
    full = _subset(rows, set())
    without_braf = _subset(rows, set(BRAF_CLUSTER))
    return {
        "role": "development_evidence",
        "rules_frozen": True,
        "same_cohort_discount": SAME_COHORT_DISCOUNT,
        "note": (
            "Compatibility weights are unchanged. A paper is discounted only with "
            "positive evidence of repeated evidence. Missing provenance is not counted "
            "as independence. The twenty claims remain development evidence. "
            "C013, C015, and C016 are held out together."
        ),
        "all_claims": full,
        "without_braf_cluster": without_braf,
        "claims": rows,
    }
