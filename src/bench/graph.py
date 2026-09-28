"""Claim-centered evidence graph for retrieval, not for forecasting.

Each frozen claim retrieves neighbors through several routes. Every route
returns at most BUDGET non-review papers inside the same follow-up horizon.
Later papers are legitimate when the question is what evidence followed the
claim. They are not inputs for a forecast made at the source date.

Routes:
- same_gene: earliest papers on the claim gene
- text_similarity: cosine similarity to the claim statement
- citation: papers that cite the source
- experimental: gene, setting, and focus terms from the claim
- shared_references: papers that cite the source's own references
- combination: rerank of the union, preferring papers found by more routes

The original eight-paper packet is recorded so a route can be compared with
what chronological same-gene retrieval already showed. It is not one of the
equal-budget arms.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import requests

from src import config
from src.bench.retrieve import (
    PILOT_SEED,
    _gene_clause,
    _year_window,
    citation_query,
    component_query,
    in_horizon,
    screen_hit,
    select_pilot,
    specify_claim,
    strata_from_labels,
)
from src.bench.sample import HORIZON_YEARS, is_review
from src.encode.encoder import LAST_ENCODER, _hash_embed
from src.ingest.pubmed import parse_europepmc_result

BUDGET = 20
METHODS = (
    "same_gene",
    "text_similarity",
    "citation",
    "experimental",
    "combination",
)
ROUTE_METHODS = (
    "same_gene",
    "text_similarity",
    "citation",
    "experimental",
    "shared_references",
)

GRAPH_DIR = config.DATA / "graph"
CACHE_DIR = config.DATA / "graph_cache"
SUMMARY_PATH = config.ROOT / "results" / "graph_retrieval.json"
EXPLORER_HTML = config.ROOT / "explorer" / "index.html"
JUDGE_CANDIDATES = config.ROOT / "bench" / "graph_judge_candidates.json"
JUDGE_PROVENANCE = config.ROOT / "bench" / "graph_judge_provenance.json"

_HEADERS = {"User-Agent": "ras-mapk-v1-graph"}
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9-]{4,}")
_STOP = {
    "about", "after", "against", "associated", "association", "because", "before",
    "between", "cells", "clinical", "could", "disease", "during", "expression",
    "found", "from", "gene", "genes", "have", "human", "identified", "into",
    "model", "mutation", "mutations", "patients", "protein", "results", "should",
    "study", "studies", "their", "there", "these", "those", "through", "using",
    "were", "which", "while", "with", "without", "would",
}
_made_request = False


def claim_statement(spec: dict) -> str:
    return ". ".join(
        part.strip().rstrip(".")
        for part in (spec.get("intervention"), spec.get("setting"), spec.get("outcome"))
        if part and str(part).strip()
    )


def content_query(claim: dict) -> str:
    """Broader than the experimental query: gene plus content words, OR-ed."""
    text = " ".join(str(claim.get(key) or "") for key in ("effect", "quote", "exposure", "population"))
    words: list[str] = []
    seen: set[str] = set()
    gene = str(claim.get("gene") or "").lower()
    for word in _WORD.findall(text):
        key = word.lower()
        if key in _STOP or key == gene or key in seen:
            continue
        seen.add(key)
        words.append(word)
        if len(words) == 6:
            break
    parts = [_gene_clause(claim["gene"])]
    if words:
        parts.append("(" + " OR ".join(words) + ")")
    parts.append(f"PUB_YEAR:[{_year_window(claim)}] AND SRC:MED NOT PMID:{claim['pmid']}")
    return " AND ".join(parts)


def take_budget(papers: list[dict], claim: dict, budget: int = BUDGET, *, require_gene: bool = True) -> tuple[list[dict], list[dict]]:
    """Split a ranked fetch into a budgeted article list and a review list."""
    kept: list[dict] = []
    reviews: list[dict] = []
    seen: set[str] = set()
    flat = {"pmid": str(claim["pmid"]), "cutoff": claim["cutoff"], "gene": claim["spec"]["gene"] if "spec" in claim else claim["gene"]}
    for paper in papers:
        pmid = str(paper.get("pmid") or "")
        if not pmid or pmid in seen:
            continue
        seen.add(pmid)
        if is_review(paper) and in_horizon_ok(paper, flat):
            reviews.append(paper)
            continue
        if screen_hit(paper, flat, require_gene=require_gene) is not None:
            continue
        kept.append(paper)
        if len(kept) >= budget:
            break
    return kept, reviews


def in_horizon_ok(paper: dict, claim: dict) -> bool:
    if str(paper.get("pmid") or "") == str(claim["pmid"]):
        return False
    return in_horizon(paper, claim["cutoff"])


def rank_text(statement: str, papers: list[dict], encode, budget: int = BUDGET) -> list[tuple[dict, float]]:
    if not papers:
        return []
    texts = [statement] + [f"{paper.get('title') or ''}. {paper.get('abstract') or ''}" for paper in papers]
    vectors = np.asarray(encode(texts), dtype=np.float32)
    scores = vectors[1:] @ vectors[0]
    order = np.argsort(-scores)
    ranked: list[tuple[dict, float]] = []
    for index in order:
        ranked.append((papers[int(index)], float(scores[int(index)])))
        if len(ranked) >= budget:
            break
    return ranked


def combine_routes(routes: dict[str, list[dict]], budget: int = BUDGET) -> list[dict]:
    """Prefer papers found by more routes. Ties break toward a better mean rank."""
    slots: dict[str, dict] = {}
    for method, papers in routes.items():
        for rank, paper in enumerate(papers, start=1):
            pmid = str(paper["pmid"])
            slot = slots.get(pmid)
            if slot is None:
                slot = {"paper": paper, "ranks": {}}
                slots[pmid] = slot
            slot["ranks"][method] = rank
    ordered = sorted(
        slots.values(),
        key=lambda slot: (
            -len(slot["ranks"]),
            -sum(1.0 / rank for rank in slot["ranks"].values()),
            str(slot["paper"]["pmid"]),
        ),
    )
    return [slot["paper"] for slot in ordered[:budget]]


def _bare_term(term: str) -> str:
    return term.replace('"', "").strip()


def matched_terms(paper: dict, terms: list[str]) -> list[str]:
    text = f"{paper.get('title') or ''} {paper.get('abstract') or ''}".lower()
    found = []
    for term in terms:
        bare = _bare_term(term).lower()
        if bare and bare in text:
            found.append(_bare_term(term))
    return found


def supporting_passage(paper: dict, terms: list[str]) -> str:
    abstract = (paper.get("abstract") or "").strip()
    title = (paper.get("title") or "").strip()
    sentences = _SENTENCE.split(abstract) if abstract else []
    best = ""
    best_score = 0
    for sentence in sentences:
        low = sentence.lower()
        score = sum(1 for term in terms if _bare_term(term).lower() in low)
        if score > best_score:
            best = sentence.strip()
            best_score = score
    if best:
        return best[:700]
    if sentences:
        return sentences[0].strip()[:700]
    return title[:700]


def assemble_claim(
    claim_row: dict,
    fetched: dict[str, list[dict]],
    packet_papers: list[dict],
    encode,
) -> dict:
    """Build equal-budget lists and a neighbor record for every linked paper."""
    same_gene, reviews_gene = take_budget(fetched.get("same_gene") or [], claim_row, require_gene=True)
    experimental, reviews_exp = take_budget(fetched.get("experimental") or [], claim_row, require_gene=True)
    citation, reviews_cite = take_budget(fetched.get("citation") or [], claim_row, require_gene=False)
    shared, reviews_shared = take_budget(fetched.get("shared_references") or [], claim_row, require_gene=True)
    text_kept, _reviews_text = take_budget(fetched.get("text_pool") or [], claim_row, budget=80, require_gene=True)
    pool = _dedupe(text_kept + same_gene)
    statement = claim_statement(claim_row["spec"])
    text_ranked = rank_text(statement, pool, encode, BUDGET)
    text_papers = [paper for paper, _score in text_ranked]
    text_scores = {str(paper["pmid"]): score for paper, score in text_ranked}
    routes = {
        "same_gene": same_gene,
        "text_similarity": text_papers,
        "citation": citation,
        "experimental": experimental,
        "shared_references": shared,
    }
    combination = combine_routes(routes, BUDGET)
    lists = {method: [str(paper["pmid"]) for paper in routes[method]] for method in routes}
    lists["combination"] = [str(paper["pmid"]) for paper in combination]
    packet_ids = [str(paper["pmid"]) for paper in packet_papers]
    reviews = _dedupe(reviews_gene + reviews_exp + reviews_cite + reviews_shared + list(fetched.get("reviews") or []))
    neighbors = _neighbors(claim_row, routes, text_scores, packet_ids, combination)
    review_neighbors = _review_neighbors(reviews, packet_ids)
    return {
        "claim_id": claim_row["claim_id"],
        "pmid": str(claim_row["pmid"]),
        "cutoff": claim_row["cutoff"],
        "year": int(claim_row["cutoff"][:4]),
        "stratum": claim_row.get("stratum") or "",
        "in_labeled_eval": bool(claim_row.get("in_labeled_eval")),
        "cutoff_resolution": "year" if str(claim_row["cutoff"]).endswith("-01-01") else "day",
        "spec": {
            "gene": claim_row["spec"]["gene"],
            "intervention": claim_row["spec"]["intervention"],
            "setting": claim_row["spec"]["setting"],
            "outcome": claim_row["spec"]["outcome"],
            "direction": claim_row["spec"]["direction"],
            "quote": claim_row["spec"]["quote"],
        },
        "use": "subsequent_evidence",
        "packet_pmids": packet_ids,
        "lists": lists,
        "neighbors": neighbors,
        "reviews": review_neighbors,
    }


def _dedupe(papers: list[dict]) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for paper in papers:
        pmid = str(paper.get("pmid") or "")
        if not pmid or pmid in seen:
            continue
        seen.add(pmid)
        out.append(paper)
    return out


def _neighbors(claim_row: dict, routes: dict[str, list[dict]], text_scores: dict[str, float], packet_ids: list[str], combination: list[dict]) -> list[dict]:
    spec = claim_row["spec"]
    terms = [spec["gene"], *(spec.get("setting_terms") or []), *(spec.get("focus_terms") or [])]
    by_pmid: dict[str, dict] = {}
    combination_ids = {str(paper["pmid"]) for paper in combination}
    packet_set = set(packet_ids)
    for method, papers in routes.items():
        for rank, paper in enumerate(papers, start=1):
            pmid = str(paper["pmid"])
            slot = by_pmid.get(pmid)
            if slot is None:
                slot = {"paper": paper, "relationships": []}
                by_pmid[pmid] = slot
            hits = matched_terms(paper, terms)
            passage_terms = [] if method == "same_gene" else terms
            slot["relationships"].append(
                {
                    "type": method,
                    "rank": rank,
                    "why": _why(method, claim_row, rank, hits, text_scores.get(pmid), paper),
                    "passage": supporting_passage(paper, passage_terms),
                    "matched_terms": hits if method != "same_gene" else [],
                }
            )
    neighbors = []
    for pmid, slot in by_pmid.items():
        paper = slot["paper"]
        in_lists = [name for name, ids in (
            ("combination", combination_ids),
        ) if pmid in ids]
        in_lists.extend(rel["type"] for rel in slot["relationships"] if rel["type"] in METHODS)
        neighbors.append(_paper_row(paper, slot["relationships"], pmid in packet_set, in_lists))
    neighbors.sort(key=lambda row: (-len(row["relationships"]), row["year"], row["pmid"]))
    return neighbors


def _review_neighbors(reviews: list[dict], packet_ids: list[str]) -> list[dict]:
    packet_set = set(packet_ids)
    rows = []
    for paper in reviews:
        rows.append(
            _paper_row(
                paper,
                [
                    {
                        "type": "review",
                        "rank": None,
                        "why": "Publication type is a review, editorial, comment, or letter. Repetition here is not an independent experiment.",
                        "passage": supporting_passage(paper, []),
                        "matched_terms": [],
                    }
                ],
                str(paper.get("pmid")) in packet_set,
                [],
            )
        )
    return rows


def _paper_row(paper: dict, relationships: list[dict], in_packet: bool, in_budget: list[str]) -> dict:
    return {
        "pmid": str(paper.get("pmid") or ""),
        "year": int(paper.get("year") or 0),
        "date": paper.get("date") or "",
        "title": paper.get("title") or "",
        "abstract": paper.get("abstract") or "",
        "pub_types": list(paper.get("pub_types") or []),
        "relationships": relationships,
        "in_packet": in_packet,
        "in_budget": in_budget,
    }


def _why(method: str, claim_row: dict, rank: int, hits: list[str], score: float | None, paper: dict) -> str:
    gene = claim_row["spec"]["gene"]
    if method == "citation":
        extra = f" Matched terms: {', '.join(hits)}." if hits else " The abstract does not state the claim's setting or focus terms."
        return f"Cites PMID {claim_row['pmid']}.{extra}"
    if method == "same_gene":
        return f"Same gene {gene}. Chronological rank {rank} among non-review papers in the {HORIZON_YEARS}-year horizon."
    if method == "experimental":
        shown = ", ".join(hits) if hits else "the structured query"
        return f"Experimental query matched {shown}."
    if method == "text_similarity":
        cosine = 0.0 if score is None else score
        return f"Cosine {cosine:.2f} between this abstract and the claim statement."
    if method == "shared_references":
        shared = ", ".join(paper.get("shared_reference_pmids") or []) or "a source reference"
        return f"Cites source references: {shared}."
    return method


def retrieval_summary(claims: list[dict]) -> dict:
    methods = list(METHODS) + ["shared_references"]
    per_method = {}
    for method in methods:
        sizes = []
        in_packet = []
        novel = []
        for claim in claims:
            ids = claim["lists"].get(method) or []
            packet = set(claim["packet_pmids"])
            sizes.append(len(ids))
            in_packet.append(sum(1 for pmid in ids if pmid in packet))
            novel.append(sum(1 for pmid in ids if pmid not in packet))
        per_method[method] = {
            "mean_candidates": _mean(sizes),
            "mean_already_in_packet": _mean(in_packet),
            "mean_absent_from_packet": _mean(novel),
            "claims_with_a_paper_outside_the_packet": sum(1 for value in novel if value > 0),
        }
    return {
        "use": "subsequent_evidence",
        "temporal_note": (
            "Neighbors are for assessing evidence after the claim. "
            "They are not valid inputs for a forecast at the source date."
        ),
        "budget": BUDGET,
        "horizon_years": HORIZON_YEARS,
        "n_claims": len(claims),
        "n_labeled_eval": sum(1 for claim in claims if claim.get("in_labeled_eval")),
        "methods": list(METHODS),
        "per_method": per_method,
        "direct_test_yield": None,
        "direct_test_note": "Yield is empty until independent direct tests are judged. Overlap with the packet is not that yield.",
    }


def score_yield(claims: list[dict], judgments: dict[str, dict[str, str]]) -> dict:
    """Count independent direct tests per route.

    judgments maps claim_id to pmid to one of support, contradict, mixed.
    A missing pmid is not a direct test. Packet papers and newly judged papers
    use the same map, so older packet labels must be translated before this call.
    """
    methods = list(METHODS) + ["shared_references", "packet"]
    rows = []
    for claim in claims:
        lists = dict(claim["lists"])
        lists["packet"] = list(claim["packet_pmids"])
        judged = judgments.get(claim["claim_id"]) or {}
        row = {"claim_id": claim["claim_id"], "stratum": claim.get("stratum") or ""}
        for method in methods:
            outcomes = [judged[pmid] for pmid in lists.get(method) or [] if pmid in judged]
            row[method] = {
                "n_direct": len(outcomes),
                "support": outcomes.count("support"),
                "contradict": outcomes.count("contradict"),
                "mixed": outcomes.count("mixed"),
            }
        rows.append(row)
    summary = {}
    for method in methods:
        summary[method] = {
            "claims_with_a_direct_test": sum(1 for row in rows if row[method]["n_direct"] > 0),
            "direct_tests": sum(row[method]["n_direct"] for row in rows),
            "claims_with_a_contradiction": sum(1 for row in rows if row[method]["contradict"] or row[method]["mixed"]),
        }
    return {"n_claims": len(rows), "methods": summary, "claims": rows}


def _mean(values: list[int]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)


def judgment_export(claims: list[dict]) -> tuple[dict, dict]:
    """Blind novel budget-list papers on the labeled-eval claims.

    Papers already in the chronological packet keep their earlier labels and
    are not re-sent. Method names are omitted from the candidate file.
    """
    packets = []
    provenance = []
    for claim in claims:
        if not claim.get("in_labeled_eval"):
            continue
        packet = set(claim["packet_pmids"])
        by_pmid = {row["pmid"]: row for row in claim["neighbors"]}
        ordered: list[str] = []
        seen: set[str] = set()
        for method in list(METHODS) + ["shared_references"]:
            for pmid in claim["lists"].get(method) or []:
                if pmid in packet or pmid in seen:
                    continue
                seen.add(pmid)
                ordered.append(pmid)
        random.Random(f"{PILOT_SEED}:{claim['claim_id']}").shuffle(ordered)
        candidates = []
        for index, pmid in enumerate(ordered, start=1):
            paper = by_pmid.get(pmid)
            if paper is None:
                continue
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
                    "methods": [
                        rel["type"]
                        for rel in paper["relationships"]
                        if rel["type"] in ROUTE_METHODS or rel["type"] == "combination"
                    ],
                    "in_combination": "combination" in (paper.get("in_budget") or []),
                }
            )
        packets.append(
            {
                "claim_id": claim["claim_id"],
                "pmid": claim["pmid"],
                "cutoff": claim["cutoff"],
                "horizon_years": HORIZON_YEARS,
                "spec": claim["spec"],
                "candidates": candidates,
            }
        )
    blinded = {
        "n_claims": len(packets),
        "note": "Method is hidden. Judge whether the paper independently tests the claim. Do not treat absence as contradiction.",
        "relations": ["direct_support", "direct_contradict", "direct_mixed", "partial", "not_test", "unclear_reporting"],
        "claims": packets,
    }
    return blinded, {"rows": provenance}


class EuropePmc:
    def __init__(self, cache_dir: Path = CACHE_DIR, timeout: int = 90):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def search(self, params: dict) -> dict:
        timeout = self.timeout
        return self._json("search", params, lambda: _http_search(params, timeout=timeout))

    def references(self, pmid: str) -> list[dict]:
        payload = self._json(
            f"refs-{pmid}",
            {"pmid": pmid},
            lambda: _http_get(
                f"https://www.ebi.ac.uk/europepmc/webservices/rest/MED/{pmid}/references",
                {"format": "json", "pageSize": 25},
            ),
        )
        raw = (payload.get("referenceList") or {}).get("reference") or []
        if isinstance(raw, dict):
            raw = [raw]
        return [row for row in raw if isinstance(row, dict)]

    def _json(self, kind: str, params: dict, fetch) -> dict:
        key = hashlib.sha256(json.dumps({"kind": kind, **params}, sort_keys=True).encode()).hexdigest()[:20]
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        payload = fetch()
        path.write_text(json.dumps(payload), encoding="utf-8")
        return payload


def _http_search(params: dict, timeout: int = 90) -> dict:
    return _http_get(config.EUROPEPMC, params, timeout=timeout)


def _http_get(url: str, params: dict, timeout: int = 90) -> dict:
    global _made_request
    if _made_request:
        time.sleep(0.2)
    _made_request = True
    response = requests.get(url, params=params, headers=_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _paper(item: dict) -> dict:
    paper = parse_europepmc_result(item)
    return paper


def _result_papers(payload: dict) -> list[dict]:
    raw = (payload.get("resultList") or {}).get("result") or []
    if isinstance(raw, dict):
        raw = [raw]
    return [_paper(item) for item in raw if isinstance(item, dict)]


def _search_papers(client: EuropePmc, query: str, page_size: int, sort: str | None) -> list[dict]:
    params = {"query": query, "format": "json", "resultType": "core", "pageSize": page_size}
    if sort:
        params["sort"] = sort
    return _result_papers(client.search(params))


def _search_through_horizon(
    client: EuropePmc,
    query: str,
    sort: str,
    claim_row: dict,
    *,
    require_gene: bool,
    budget: int = BUDGET,
    page_size: int = 100,
    max_pages: int = 8,
) -> list[dict]:
    """Walk date order until the horizon contains a full budget, or the pages end.

    The first page of a year-window search is often still before the claim cutoff.
    """
    cursor = "*"
    collected: list[dict] = []
    for _ in range(max_pages):
        params = {
            "query": query,
            "format": "json",
            "resultType": "core",
            "pageSize": page_size,
            "cursorMark": cursor,
            "sort": sort,
        }
        payload = client.search(params)
        collected.extend(_result_papers(payload))
        kept, _reviews = take_budget(collected, claim_row, budget=budget, require_gene=require_gene)
        if len(kept) >= budget:
            break
        nxt = str(payload.get("nextCursorMark") or "")
        if not nxt or nxt == cursor or not _result_papers(payload):
            break
        cursor = nxt
    return collected


def fetch_routes(claim_row: dict, client: EuropePmc) -> dict[str, list[dict]]:
    claim = {
        "pmid": claim_row["pmid"],
        "cutoff": claim_row["cutoff"],
        "gene": claim_row["spec"]["gene"],
        "effect": claim_row["spec"]["outcome"],
        "quote": claim_row["spec"]["quote"],
        "exposure": claim_row["spec"]["intervention"],
        "population": claim_row["spec"]["setting"],
    }
    same_gene = _search_through_horizon(
        client, _gene_only_query(claim), "P_PDATE_D asc", claim_row, require_gene=True
    )
    experimental = _search_papers(client, claim_row["queries"]["component"], 40, None)
    text_pool = _search_papers(client, content_query(claim), 40, None)
    citation = _search_papers(client, claim_row["queries"]["citation"], 80, "P_PDATE_D asc")
    shared = _shared_reference_papers(claim, client)
    return {
        "same_gene": same_gene,
        "experimental": experimental,
        "text_pool": text_pool,
        "citation": citation,
        "shared_references": shared,
    }


def _gene_only_query(claim: dict) -> str:
    return (
        f"{_gene_clause(claim['gene'])} AND PUB_YEAR:[{_year_window(claim)}] "
        f"AND SRC:MED NOT PMID:{claim['pmid']}"
    )


def _shared_reference_papers(claim: dict, client: EuropePmc) -> list[dict]:
    refs = []
    for row in client.references(str(claim["pmid"])):
        if str(row.get("source") or "") != "MED":
            continue
        if str(row.get("match") or "") not in {"Y", "y", ""}:
            continue
        pmid = str(row.get("id") or "")
        if pmid:
            refs.append(pmid)
        if len(refs) == 4:
            break
    found: dict[str, dict] = {}
    for ref in refs:
        query = (
            f'REF:"{ref}" AND {_gene_clause(claim["gene"])} AND PUB_YEAR:[{_year_window(claim)}] '
            f"AND SRC:MED NOT PMID:{claim['pmid']}"
        )
        for paper in _search_papers(client, query, 8, "P_PDATE_D asc"):
            pmid = str(paper.get("pmid") or "")
            if not pmid:
                continue
            slot = found.get(pmid)
            if slot is None:
                paper["shared_reference_pmids"] = [ref]
                found[pmid] = paper
            elif ref not in slot["shared_reference_pmids"]:
                slot["shared_reference_pmids"].append(ref)
    return sorted(found.values(), key=lambda paper: (-len(paper.get("shared_reference_pmids") or []), paper.get("date") or "", paper["pmid"]))


def load_encoder():
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(config.ENCODER_NAME)

        def encode(texts: list[str]) -> np.ndarray:
            raw = model.encode(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            array = np.asarray(raw, dtype=np.float32)
            if array.ndim == 1:
                array = array.reshape(1, -1)
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            return array / np.maximum(norms, np.float32(1e-8))

        return encode, config.ENCODER_NAME
    except Exception:
        return _hash_embed, "hash768-v1"


def claim_rows() -> tuple[list[dict], set[str]]:
    claims = json.loads((config.ROOT / "bench" / "claims.json").read_text(encoding="utf-8"))["claims"]
    labels_a = json.loads((config.ROOT / "bench" / "labels_a.json").read_text(encoding="utf-8"))["claims"]
    labels_b = json.loads((config.ROOT / "bench" / "labels_b.json").read_text(encoding="utf-8"))["claims"]
    pilot = select_pilot(claims, strata_from_labels(labels_a, labels_b), PILOT_SEED)
    pilot_ids = {row["claim_id"] for row in pilot["claims"]}
    strata = strata_from_labels(labels_a, labels_b)
    rows = []
    for claim in claims:
        if claim.get("skip"):
            continue
        info = strata[claim["claim_id"]]
        spec = specify_claim(claim)
        rows.append(
            {
                "claim_id": claim["claim_id"],
                "pmid": str(claim["pmid"]),
                "cutoff": claim["cutoff"],
                "stratum": info["stratum"],
                "in_labeled_eval": claim["claim_id"] in pilot_ids,
                "spec": spec,
                "queries": {
                    "component": component_query(claim),
                    "citation": citation_query(claim),
                },
            }
        )
    return rows, pilot_ids


def packet_articles() -> dict[str, list[dict]]:
    packets = json.loads((config.ROOT / "bench" / "packets.json").read_text(encoding="utf-8"))["packets"]
    return {row["claim_id"]: list((row.get("later") or {}).get("articles") or []) for row in packets}


def run_graph(client: EuropePmc | None = None, encode=None, encoder_name: str | None = None, write: bool = True) -> dict:
    rows, _pilot_ids = claim_rows()
    articles = packet_articles()
    if encode is None:
        encode, encoder_name = load_encoder()
    client = client or EuropePmc()
    assembled = []
    errors = []
    for index, row in enumerate(rows, start=1):
        try:
            fetched = fetch_routes(row, client)
            assembled.append(assemble_claim(row, fetched, articles.get(row["claim_id"]) or [], encode))
            print(f"{index}/{len(rows)} {row['claim_id']} neighbors={len(assembled[-1]['neighbors'])}", flush=True)
        except Exception as exc:
            errors.append({"claim_id": row["claim_id"], "error": str(exc)})
            print(f"{index}/{len(rows)} {row['claim_id']} FAILED {exc}", flush=True)
    summary = retrieval_summary(assembled)
    summary["errors"] = errors
    summary["encoder"] = encoder_name or LAST_ENCODER
    blinded, provenance = judgment_export(assembled)
    if write:
        GRAPH_DIR.mkdir(parents=True, exist_ok=True)
        (GRAPH_DIR / "explorer.json").write_text(json.dumps({"claims": assembled}, indent=2) + "\n", encoding="utf-8")
        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        JUDGE_CANDIDATES.write_text(json.dumps(blinded, indent=2) + "\n", encoding="utf-8")
        JUDGE_PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        if assembled:
            write_explorer(assembled, EXPLORER_HTML)
    return {"claims": assembled, "summary": summary, "blinded": blinded}


def write_explorer(claims: list[dict], path: Path) -> None:
    shell = (config.ROOT / "explorer" / "shell.html").read_text(encoding="utf-8")
    public = [{key: value for key, value in claim.items() if key not in {"stratum", "in_labeled_eval"}} for claim in claims]
    payload = json.dumps({"claims": public}, ensure_ascii=False).replace("<", "\\u003c")
    html = shell.replace("/*__GRAPH__*/", f"const GRAPH = {payload};")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def main() -> None:
    result = run_graph(write=True)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
