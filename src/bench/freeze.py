"""Freeze ranked lists and separate retrieval from a verified test.

Date rule, one version for every route. A paper in a later calendar year is
subsequent. In the source year, the order is subsequent only when both the
source and the paper have a day-level date and the paper is strictly later.
A January 1 placeholder, or the same calendar day, is order-uncertain. Those
papers stay in the retrieved list and are flagged.

A shared title with a near-copy abstract is one report. Distinct PMIDs do not
by themselves make independent evidence.

The judged proposition is the exposure, setting, and direction. The source's
exact proportion is its measurement, not a second claim.
"""

from __future__ import annotations

import json
import random
import re

from src.bench.graph import BUDGET, METHODS, PILOT_SEED
from src.bench.sample import HORIZON_YEARS

DATE_RULE = "day-level-if-both-dates-resolve-otherwise-uncertain"
_TOKEN = re.compile(r"[a-z0-9]+")
_LEADING_COUNT = re.compile(r"^\d+\s+")


def is_year_resolution(date: str) -> bool:
    return bool(date) and date.endswith("-01-01")


def order_status(paper_date: str, paper_year: int, cutoff: str) -> str:
    """Return subsequent, uncertain, or before."""
    source_year = int(cutoff[:4])
    if paper_year > source_year:
        return "subsequent"
    if paper_year < source_year:
        return "before"
    if is_year_resolution(cutoff) or is_year_resolution(paper_date or ""):
        return "uncertain"
    if (paper_date or "") > cutoff:
        return "subsequent"
    if (paper_date or "") == cutoff:
        return "uncertain"
    return "before"


def judged_proposition(spec: dict) -> dict:
    setting = _LEADING_COUNT.sub("", spec.get("setting") or spec.get("population") or "").strip()
    intervention = (spec.get("intervention") or spec.get("exposure") or "").strip()
    direction = spec.get("direction") or ""
    proposition = f"In {setting}, {intervention} ({direction})."
    return {
        "judged_proposition": proposition,
        "source_measurement": spec.get("quote") or spec.get("outcome") or "",
        "setting_class": setting,
        "intervention": intervention,
        "direction": direction,
        "scope": (
            "Judge this proposition, not the source's exact proportion. "
            "A new cohort can support the association without reproducing the source fraction. "
            "A paper is not a direct test when it only studies consequences in subjects already selected for the exposure, "
            "uses a different species or system than the setting class, "
            "or mentions the topic in a background sentence. "
            "A second write-up of the same patients or the same experiment is not independent evidence."
        ),
    }


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall((text or "").lower()))


def _title_key(title: str) -> str:
    return " ".join(_TOKEN.findall((title or "").lower()))


def same_report(left: dict, right: dict) -> bool:
    """True when two records are one underlying write-up."""
    if str(left.get("pmid")) == str(right.get("pmid")):
        return False
    left_title = _title_key(left.get("title") or "")
    right_title = _title_key(right.get("title") or "")
    if not left_title or left_title != right_title:
        return False
    left_tokens = _tokens(left.get("abstract") or "")
    right_tokens = _tokens(right.get("abstract") or "")
    if not left_tokens or not right_tokens:
        return True
    overlap = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    return overlap >= 0.55


def duplicate_groups(papers: list[dict]) -> dict[str, str]:
    """Map a later PMID onto the earliest PMID in its same-report cluster."""
    ordered = sorted(papers, key=lambda paper: (paper.get("year") or 0, paper.get("date") or "", str(paper.get("pmid"))))
    canonical: dict[str, str] = {}
    kept: list[dict] = []
    for paper in ordered:
        match = next((prior for prior in kept if same_report(prior, paper)), None)
        if match is None:
            kept.append(paper)
            continue
        canonical[str(paper["pmid"])] = str(match["pmid"])
    return canonical


def apply_freeze(claims: list[dict]) -> list[dict]:
    frozen = []
    for claim in claims:
        scope = judged_proposition(claim["spec"])
        papers = list(claim.get("neighbors") or []) + list(claim.get("reviews") or [])
        groups = duplicate_groups(papers)
        neighbors = [_annotate(paper, claim["cutoff"], groups) for paper in claim.get("neighbors") or []]
        reviews = [_annotate(paper, claim["cutoff"], groups) for paper in claim.get("reviews") or []]
        frozen.append({**claim, **scope, "date_rule": DATE_RULE, "neighbors": neighbors, "reviews": reviews})
    return frozen


def _annotate(paper: dict, cutoff: str, groups: dict[str, str]) -> dict:
    status = order_status(paper.get("date") or "", int(paper.get("year") or 0), cutoff)
    return {
        **paper,
        "order": status,
        "order_uncertain": status == "uncertain",
        "duplicate_of": groups.get(str(paper.get("pmid") or "")),
    }


def blinded_union(claims: list[dict]) -> tuple[dict, dict]:
    """Deduplicated union of the frozen lists, without route, rank, or prior label."""
    packets = []
    provenance = []
    for claim in claims:
        if not claim.get("in_labeled_eval"):
            continue
        by_pmid = {row["pmid"]: row for row in claim.get("neighbors") or []}
        ordered: list[str] = []
        seen: set[str] = set()
        for method in list(METHODS) + ["shared_references"]:
            for pmid in claim["lists"].get(method) or []:
                if pmid in seen or pmid not in by_pmid:
                    continue
                seen.add(pmid)
                ordered.append(pmid)
        random.Random(f"{PILOT_SEED}:{claim['claim_id']}").shuffle(ordered)
        candidates = []
        for index, pmid in enumerate(ordered, start=1):
            paper = by_pmid[pmid]
            candidate_id = f"{claim['claim_id']}-G{index:02d}"
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "pmid": pmid,
                    "year": paper["year"],
                    "date": paper["date"],
                    "title": paper["title"],
                    "abstract": paper["abstract"],
                    "pub_types": paper["pub_types"],
                }
            )
            provenance.append(
                {
                    "claim_id": claim["claim_id"],
                    "candidate_id": candidate_id,
                    "pmid": pmid,
                    "methods": [rel["type"] for rel in paper.get("relationships") or []],
                    "order": paper.get("order"),
                    "duplicate_of": paper.get("duplicate_of"),
                }
            )
        packets.append(
            {
                "claim_id": claim["claim_id"],
                "source_pmid": claim["pmid"],
                "cutoff": claim["cutoff"],
                "horizon_years": HORIZON_YEARS,
                "judged_proposition": claim["judged_proposition"],
                "source_measurement": claim["source_measurement"],
                "scope": claim["scope"],
                "candidates": candidates,
            }
        )
    blinded = {
        "n_claims": len(packets),
        "date_rule": DATE_RULE,
        "note": (
            "Retrieval route, rank, and any earlier support label are hidden. "
            "Judge the proposition in the packet. A retrieval passage is not a test. "
            "Agreement between model annotators is inter-agent agreement."
        ),
        "relations": [
            "direct_support",
            "direct_contradict",
            "direct_mixed",
            "partial",
            "not_test",
            "unclear_reporting",
        ],
        "claims": packets,
    }
    return blinded, {"rows": provenance}


def route_comparison(claims: list[dict], judgments: dict[str, dict[str, str]]) -> dict:
    """Unique independent direct tests inside each route's first 20 slots.

    judgments maps claim_id to pmid to support, contradict, or mixed.
    Order-uncertain papers and second copies of the same report do not add a test.
    A route that returned 2 papers used 2 of its 20 slots.
    """
    methods = list(METHODS) + ["shared_references"]
    per_method = {
        method: {
            "slots_filled": 0,
            "slots_available": 0,
            "direct_tests": 0,
            "claims_with_a_direct_test": 0,
            "direct_tests_beyond_same_gene": 0,
            "uncertain_order_not_counted": 0,
            "duplicate_reports_not_counted": 0,
        }
        for method in methods
    }
    by_claim = []
    for claim in claims:
        if judgments and claim["claim_id"] not in judgments and not claim.get("in_labeled_eval"):
            continue
        if claim["claim_id"] not in judgments:
            continue
        judged = judgments[claim["claim_id"]]
        papers = {row["pmid"]: row for row in claim.get("neighbors") or []}
        same_gene_tests = _count_tests(claim["lists"].get("same_gene") or [], papers, judged)
        row = {"claim_id": claim["claim_id"]}
        for method in methods:
            ids = list(claim["lists"].get(method) or [])[:BUDGET]
            stats = _count_tests(ids, papers, judged)
            per_method[method]["slots_filled"] += len(ids)
            per_method[method]["slots_available"] += BUDGET
            per_method[method]["direct_tests"] += stats["direct_tests"]
            per_method[method]["uncertain_order_not_counted"] += stats["uncertain"]
            per_method[method]["duplicate_reports_not_counted"] += stats["duplicates"]
            if stats["direct_tests"]:
                per_method[method]["claims_with_a_direct_test"] += 1
            beyond = 0
            if method != "same_gene":
                beyond = len(stats["test_pmids"] - same_gene_tests["test_pmids"])
                per_method[method]["direct_tests_beyond_same_gene"] += beyond
            row[method] = {
                "direct_tests": stats["direct_tests"],
                "slots_filled": len(ids),
                "beyond_same_gene": beyond,
                "test_pmids": sorted(stats["test_pmids"]),
            }
        by_claim.append(row)
    n = len(by_claim)
    for method, stats in per_method.items():
        stats["claims_judged"] = n
        stats["mean_slots_filled"] = round(stats["slots_filled"] / n, 3) if n else 0.0
    return {
        "budget": BUDGET,
        "horizon_years": HORIZON_YEARS,
        "date_rule": DATE_RULE,
        "n_claims": n,
        "methods": per_method,
        "claims": by_claim,
        "annotator_note": "If annotators are model agents, this agreement is inter-agent agreement.",
    }


def _count_tests(pmids: list[str], papers: dict[str, dict], judged: dict[str, str]) -> dict:
    seen_reports: set[str] = set()
    test_pmids: set[str] = set()
    uncertain = 0
    duplicates = 0
    for pmid in pmids:
        paper = papers.get(pmid) or {}
        report = paper.get("duplicate_of") or pmid
        if report in seen_reports:
            duplicates += 1
            continue
        seen_reports.add(report)
        if paper.get("order_uncertain"):
            if pmid in judged:
                uncertain += 1
            continue
        if pmid in judged and judged[pmid] in {"support", "contradict", "mixed"}:
            test_pmids.add(pmid)
    return {
        "direct_tests": len(test_pmids),
        "test_pmids": test_pmids,
        "uncertain": uncertain,
        "duplicates": duplicates,
    }


_DIRECT = {
    "direct_support": "support",
    "direct_contradict": "contradict",
    "direct_mixed": "mixed",
}


def _call(row: dict) -> str | None:
    mapped = _DIRECT.get(row.get("relation") or "")
    if mapped and row.get("independent", True):
        return mapped
    return None


def consensus_judgments(pairs: list[tuple[dict, dict]]) -> tuple[dict[str, dict[str, str]], dict]:
    """Keep a direct test only when both readers call it one, and agree on the direction."""
    judgments: dict[str, dict[str, str]] = {}
    left_bin: list[str] = []
    right_bin: list[str] = []
    relation_agree = 0
    n = 0
    direction_disagreements = 0
    one_sided = 0
    for left, right in pairs:
        by_right = {
            row["pmid"]: row
            for claim in right["claims"]
            for row in claim["followups"]
        }
        for claim in left["claims"]:
            bucket = judgments.setdefault(claim["claim_id"], {})
            for row in claim["followups"]:
                other = by_right[row["pmid"]]
                n += 1
                if row.get("relation") == other.get("relation"):
                    relation_agree += 1
                call_l = _call(row)
                call_r = _call(other)
                left_bin.append("direct" if call_l else "other")
                right_bin.append("direct" if call_r else "other")
                if call_l and call_r and call_l == call_r:
                    bucket[row["pmid"]] = call_l
                elif call_l and call_r:
                    direction_disagreements += 1
                elif call_l or call_r:
                    one_sided += 1
    both_direct = sum(left == "direct" and right == "direct" for left, right in zip(left_bin, right_bin))
    direct_agree = sum(left == right for left, right in zip(left_bin, right_bin)) / n if n else 0.0
    return judgments, {
        "n_papers": n,
        "relation_agreement": round(relation_agree / n, 4) if n else 0.0,
        "direct_test_agreement": round(direct_agree, 4),
        "papers_both_call_direct": both_direct,
        "direction_disagreements": direction_disagreements,
        "one_reader_only": one_sided,
        "rule": "Count a paper only when both readers call it an independent direct test and agree on support, contradiction, or mixed.",
        "annotator_note": "Both readers are model agents. This is inter-agent agreement, not an expert audit.",
    }


def write_frozen(claims: list[dict], path) -> None:
    path.write_text(json.dumps({"date_rule": DATE_RULE, "claims": claims}, indent=2) + "\n", encoding="utf-8")
