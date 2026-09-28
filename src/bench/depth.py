"""Candidate-depth comparison for the frozen text-similarity route.

The query, Europe PMC result order, cosine scorer, and eligibility screen
stay fixed. The only change is how many ordered hits are admitted before
scoring: 40, 200, or 1,000. Each pool returns up to twenty papers.

C068 checks whether the known miss enters that twenty. Fresh claims measure
whether a deeper pool adds a claim with an eligible direct test. This module
does not write the frozen explorer lists.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

from src import config
from src.bench.freeze import duplicate_groups, judged_proposition, order_status
from src.bench.graph import (
    BUDGET,
    EuropePmc,
    claim_statement,
    content_query,
    rank_text,
)
from src.bench.retrieve import screen_hit, specify_claim

DEPTHS = (40, 200, 1000)
BLIND_SEED = 20260930
REGRESSION_CLAIM = "C068"
KNOWN_MISS = "36156071"
FRESH_FAMILIES = {
    "pancreatic_kras": ["C025", "C037", "C045", "C065", "C069", "C071"],
    "colorectal_kras": ["C036", "C053", "C062", "C064", "C070"],
}
DIRECT = {"support", "contradict", "mixed"}

BLIND_PATH = config.ROOT / "results" / "depth_blind.json"
PROVENANCE_PATH = config.ROOT / "results" / "depth_provenance.json"
JUDGMENT_PATH = config.ROOT / "results" / "depth_judgments.json"
SUMMARY_PATH = config.ROOT / "results" / "depth_comparison.json"


def fresh_claim_ids() -> list[str]:
    ids: list[str] = []
    for family in FRESH_FAMILIES.values():
        ids.extend(family)
    return ids


def load_claims(claim_ids: list[str]) -> list[dict]:
    payload = json.loads((config.ROOT / "bench" / "claims.json").read_text(encoding="utf-8"))
    by_id = {row["claim_id"]: row for row in payload["claims"]}
    missing = [claim_id for claim_id in claim_ids if claim_id not in by_id]
    if missing:
        raise KeyError(", ".join(missing))
    return [by_id[claim_id] for claim_id in claim_ids]


def fetch_ordered(client: EuropePmc, query: str, limit: int) -> tuple[list[dict], int | None, dict[int, float]]:
    """First `limit` hits in the API's default order. No sort override.

    `seconds_at_depth` is the time to reach each nested prefix.
    """
    cursor = "*"
    collected: list[dict] = []
    seen: set[str] = set()
    hit_count: int | None = None
    started = time.perf_counter()
    seconds_at_depth: dict[int, float] = {}
    while len(collected) < limit:
        page_size = min(100, limit - len(collected))
        params = {
            "query": query,
            "format": "json",
            "resultType": "core",
            "pageSize": page_size,
            "cursorMark": cursor,
        }
        payload = client.search(params)
        if hit_count is None:
            raw_count = payload.get("hitCount")
            hit_count = int(raw_count) if raw_count is not None else None
        batch = _papers(payload)
        if not batch:
            break
        for paper in batch:
            pmid = str(paper.get("pmid") or "")
            if pmid and pmid in seen:
                continue
            if pmid:
                seen.add(pmid)
            collected.append(paper)
            for depth in DEPTHS:
                if depth not in seconds_at_depth and len(collected) >= depth:
                    seconds_at_depth[depth] = round(time.perf_counter() - started, 3)
            if len(collected) >= limit:
                break
        nxt = str(payload.get("nextCursorMark") or "")
        if not nxt or nxt == cursor:
            break
        cursor = nxt
    elapsed = round(time.perf_counter() - started, 3)
    for depth in DEPTHS:
        seconds_at_depth.setdefault(depth, elapsed)
    return collected[:limit], hit_count, seconds_at_depth


def eligible_in_order(hits: list[dict], claim: dict) -> list[dict]:
    flat = {"pmid": str(claim["pmid"]), "cutoff": claim["cutoff"], "gene": claim["gene"]}
    kept: list[dict] = []
    seen: set[str] = set()
    for paper in hits:
        pmid = str(paper.get("pmid") or "")
        if not pmid or pmid in seen:
            continue
        seen.add(pmid)
        if screen_hit(paper, flat, require_gene=True) is not None:
            continue
        kept.append(paper)
    return kept


def rank_pool(claim: dict, hits: list[dict], encode, depth: int, budget: int = BUDGET) -> list[tuple[dict, float]]:
    """Score the eligible prefix with the existing cosine scorer and return twenty."""
    eligible = eligible_in_order(hits[:depth], claim)
    if not eligible:
        return []
    statement = claim_statement(specify_claim(claim))
    return rank_text(statement, eligible, encode, budget=max(budget, len(eligible)))


def returned_twenty(ranked: list[tuple[dict, float]], budget: int = BUDGET) -> list[tuple[dict, float]]:
    return ranked[:budget]


def known_miss_status(hits: list[dict], ranked_by_depth: dict[int, list[tuple[dict, float]]], pmid: str = KNOWN_MISS) -> dict:
    raw_index = next((index for index, paper in enumerate(hits) if str(paper.get("pmid") or "") == pmid), None)
    status = {"pmid": pmid, "raw_index": raw_index, "in_returned_twenty": {}, "similarity_rank": {}}
    for depth, ranked in ranked_by_depth.items():
        pmids = [str(paper["pmid"]) for paper, _score in ranked]
        status["in_returned_twenty"][str(depth)] = pmid in pmids[:BUDGET]
        status["similarity_rank"][str(depth)] = pmids.index(pmid) + 1 if pmid in pmids else None
    return status


def counted_tests(
    pmids: list[str],
    papers: dict[str, dict],
    judgments: dict[str, str],
    cutoff: str,
) -> list[str]:
    """Direct tests in returned order.

    The first time a report is seen consumes it. An order-uncertain copy and a
    second copy do not count.
    """
    groups = duplicate_groups([papers[pmid] for pmid in pmids if pmid in papers])
    seen_reports: set[str] = set()
    counted: list[str] = []
    for pmid in pmids:
        paper = papers.get(pmid)
        if paper is None:
            continue
        report = groups.get(pmid) or pmid
        if report in seen_reports:
            continue
        seen_reports.add(report)
        if order_status(paper.get("date") or "", int(paper.get("year") or 0), cutoff) != "subsequent":
            continue
        if judgments.get(pmid) not in DIRECT:
            continue
        counted.append(pmid)
    return counted


def coverage_row(claim_ids: list[str], tests_by_claim: dict[str, list[str]]) -> dict:
    claims_with = [claim_id for claim_id in claim_ids if tests_by_claim.get(claim_id)]
    return {
        "claims": len(claim_ids),
        "claims_with_an_eligible_direct_test": len(claims_with),
        "direct_tests": sum(len(tests_by_claim.get(claim_id) or []) for claim_id in claim_ids),
        "claim_ids_with_a_test": claims_with,
    }


def _papers(payload: dict) -> list[dict]:
    from src.bench.graph import _result_papers

    return _result_papers(payload)


def _record(paper: dict) -> dict:
    return {
        "pmid": str(paper.get("pmid") or ""),
        "year": int(paper.get("year") or 0),
        "date": paper.get("date") or "",
        "title": paper.get("title") or "",
        "abstract": paper.get("abstract") or "",
        "pub_types": list(paper.get("pub_types") or []),
    }


def run_claim(claim: dict, client: EuropePmc, encode) -> dict:
    query = content_query(claim)
    hits, hit_count, seconds_at_depth = fetch_ordered(client, query, DEPTHS[-1])
    retrieval_seconds = {str(depth): seconds_at_depth[depth] for depth in DEPTHS}
    ranked_by_depth: dict[int, list[tuple[dict, float]]] = {}
    scoring_seconds: dict[str, float] = {}
    for depth in DEPTHS:
        score_started = time.perf_counter()
        ranked_by_depth[depth] = rank_pool(claim, hits, encode, depth)
        scoring_seconds[str(depth)] = round(time.perf_counter() - score_started, 3)
    returned = {
        str(depth): [str(paper["pmid"]) for paper, _score in returned_twenty(ranked_by_depth[depth])]
        for depth in DEPTHS
    }
    eligible_counts = {
        str(depth): len(eligible_in_order(hits[:depth], claim))
        for depth in DEPTHS
    }
    papers: dict[str, dict] = {}
    for depth in DEPTHS:
        for paper, score in returned_twenty(ranked_by_depth[depth]):
            record = papers.setdefault(str(paper["pmid"]), _record(paper))
            record.setdefault("score", {})[str(depth)] = round(score, 4)
    return {
        "claim_id": claim["claim_id"],
        "pmid": str(claim["pmid"]),
        "cutoff": claim["cutoff"],
        "query": query,
        "hit_count": hit_count,
        "fetched": len(hits),
        "retrieval_seconds": retrieval_seconds,
        "scoring_seconds": scoring_seconds,
        "eligible_at_depth": eligible_counts,
        "returned": returned,
        "papers": papers,
        "hits": hits,
        "ranked_by_depth": ranked_by_depth,
    }


def blind_packet(rows: list[dict]) -> dict:
    """Deduplicated union. Depth, score, and hit position are omitted."""
    claims = []
    for row in rows:
        claim = {
            "pmid": row["pmid"],
            "cutoff": row["cutoff"],
            "gene": row["gene"],
            "exposure": row.get("exposure") or "",
            "population": row.get("population") or "",
            "effect": row.get("effect") or "",
            "direction": row.get("direction") or "",
            "quote": row.get("quote") or "",
        }
        proposition = judged_proposition(specify_claim(claim))
        union: list[dict] = []
        seen: set[str] = set()
        for depth in DEPTHS:
            for pmid in row["returned"][str(depth)]:
                if pmid in seen:
                    continue
                seen.add(pmid)
                paper = row["papers"][pmid]
                union.append(
                    {
                        "pmid": pmid,
                        "year": paper["year"],
                        "date": paper["date"],
                        "title": paper["title"],
                        "abstract": paper["abstract"],
                        "pub_types": paper["pub_types"],
                    }
                )
        random.Random(f"{BLIND_SEED}:{row['claim_id']}").shuffle(union)
        for index, paper in enumerate(union, start=1):
            paper["candidate_id"] = f"{row['claim_id']}-U{index:02d}"
        claims.append(
            {
                "claim_id": row["claim_id"],
                "cutoff": row["cutoff"],
                "proposition": proposition,
                "candidates": union,
            }
        )
    return {
        "note": "Candidate depth is hidden. Judge whether each paper is an independent direct test of the proposition.",
        "claims": claims,
    }


def provenance_payload(fresh_rows: list[dict], regression: dict, encoder_name: str) -> dict:
    def public(row: dict) -> dict:
        return {key: value for key, value in row.items() if key not in {"hits", "ranked_by_depth", "gene", "exposure", "population", "effect", "direction", "quote"}}

    return {
        "fixed": {
            "query": "content_query",
            "ordering": "Europe PMC default order, cursor pages, no sort override",
            "scorer": "rank_text cosine between the claim statement and the title plus abstract",
            "eligibility": "screen_hit with a gene mention required",
            "budget": BUDGET,
            "depths": list(DEPTHS),
            "date_rule": "day-level-if-both-dates-resolve-otherwise-uncertain",
        },
        "encoder": encoder_name,
        "regression_claim": REGRESSION_CLAIM,
        "known_miss": regression,
        "families": FRESH_FAMILIES,
        "fresh": [public(row) for row in fresh_rows],
    }


def write_outputs(fresh_rows: list[dict], regression: dict, encoder_name: str) -> None:
    BLIND_PATH.parent.mkdir(parents=True, exist_ok=True)
    BLIND_PATH.write_text(json.dumps(blind_packet(fresh_rows), indent=2) + "\n", encoding="utf-8")
    PROVENANCE_PATH.write_text(
        json.dumps(provenance_payload(fresh_rows, regression, encoder_name), indent=2) + "\n",
        encoding="utf-8",
    )


def summarize(provenance: dict, judgments: dict[str, dict[str, str]]) -> dict:
    by_family = {}
    overall_ids: list[str] = []
    per_claim = []
    costs = {str(depth): {"retrieval_seconds": 0.0, "scoring_seconds": 0.0} for depth in DEPTHS}
    for row in provenance["fresh"]:
        overall_ids.append(row["claim_id"])
        costs_row = row["scoring_seconds"]
        retrieval = row["retrieval_seconds"]
        for depth in DEPTHS:
            costs[str(depth)]["retrieval_seconds"] += float(retrieval[str(depth)])
            costs[str(depth)]["scoring_seconds"] += float(costs_row[str(depth)])
        judged = judgments.get(row["claim_id"]) or {}
        tests = {}
        for depth in DEPTHS:
            tests[str(depth)] = counted_tests(row["returned"][str(depth)], row["papers"], judged, row["cutoff"])
        per_claim.append({"claim_id": row["claim_id"], "tests": tests, "eligible_at_depth": row["eligible_at_depth"]})
    for name, claim_ids in provenance["families"].items():
        by_family[name] = {
            str(depth): coverage_row(
                claim_ids,
                {row["claim_id"]: row["tests"][str(depth)] for row in per_claim},
            )
            for depth in DEPTHS
        }
    overall = {
        str(depth): coverage_row(
            overall_ids,
            {row["claim_id"]: row["tests"][str(depth)] for row in per_claim},
        )
        for depth in DEPTHS
    }
    for depth in DEPTHS:
        costs[str(depth)]["retrieval_seconds"] = round(costs[str(depth)]["retrieval_seconds"], 3)
        costs[str(depth)]["scoring_seconds"] = round(costs[str(depth)]["scoring_seconds"], 3)
    return {
        "primary_outcome": "claims with an eligible direct test",
        "secondary_outcome": "direct-test yield",
        "families": by_family,
        "overall": overall,
        "costs": costs,
        "claims": per_claim,
        "known_miss": provenance.get("known_miss"),
        "note": (
            "Retrieval seconds are cumulative: the time to reach that prefix of one ordered result list. "
            "Scoring seconds are the cosine scorer on each pool separately. "
            "C068 is a recovery check and is not in the fresh totals."
        ),
    }


def run(encode=None, encoder_name: str | None = None, client: EuropePmc | None = None) -> None:
    from src.bench.graph import load_encoder

    if encode is None:
        encode, encoder_name = load_encoder()
    client = client or EuropePmc(timeout=180)
    fresh_claims = load_claims(fresh_claim_ids())
    fresh_rows = []
    for index, claim in enumerate(fresh_claims, start=1):
        row = run_claim(claim, client, encode)
        row["gene"] = claim["gene"]
        row["exposure"] = claim.get("exposure") or ""
        row["population"] = claim.get("population") or ""
        row["effect"] = claim.get("effect") or ""
        row["direction"] = claim.get("direction") or ""
        row["quote"] = claim.get("quote") or ""
        fresh_rows.append(row)
        print(
            f"fresh {index}/{len(fresh_claims)} {claim['claim_id']} fetched={row['fetched']} retrieval_s={row['retrieval_seconds']['1000']}",
            flush=True,
        )
    regression_claim = load_claims([REGRESSION_CLAIM])[0]
    regression_row = run_claim(regression_claim, client, encode)
    regression = known_miss_status(regression_row["hits"], regression_row["ranked_by_depth"])
    regression["hit_count"] = regression_row["hit_count"]
    regression["fetched"] = regression_row["fetched"]
    regression["retrieval_seconds"] = regression_row["retrieval_seconds"]
    regression["scoring_seconds"] = regression_row["scoring_seconds"]
    regression["eligible_at_depth"] = regression_row["eligible_at_depth"]
    print(f"regression {REGRESSION_CLAIM} fetched={regression_row['fetched']}", flush=True)
    write_outputs(fresh_rows, regression, encoder_name or "")


def main() -> None:
    run()
    print(f"wrote {BLIND_PATH}")
    print(f"wrote {PROVENANCE_PATH}")


if __name__ == "__main__":
    main()
