"""Claim specifications and horizon rules for the frozen 65-claim benchmark.

The claim list is not rebuilt. Specifications are taken from the source
record only. The graph retrieval experiment lives in src/bench/graph.py.
Outcome labels are not an input to queries.
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

import requests

from src import config
from src.bench.agreement import claim_status
from src.bench.sample import HORIZON_YEARS, is_review
from src.ingest.pubmed import parse_europepmc_result
from src.schema import GENE_ALIASES

PILOT_SEED = 20260928
SUPPORT_RULE = "census_of_agreed_support"
UNRESOLVED_PER_BIN = 3
YEAR_BINS = ((2006, 2008), (2009, 2011), (2012, 2014))
COMPONENT_CAP = 25
TAIL_CAP = 15
CITATION_CAP = 25
CITATION_FETCH = 100

SAMPLE_OUT = config.ROOT / "bench" / "pilot_sample.json"
CANDIDATES_OUT = config.ROOT / "bench" / "pilot_candidates.json"
PROVENANCE_OUT = config.ROOT / "bench" / "pilot_provenance.json"

DECISION_RULE = {
    "scarce_if_claims_with_direct_test_below": 5,
    "expand_retrieval_if_gain_at_least": 4,
    "uniform_support_if_claims_with_direct_test_at_least": 10,
    "uniform_support_if_claims_with_contradiction_at_most": 1,
    "precedence": [
        "If fewer than 5 of 20 claims have an independent direct test, do not build a durability predictor.",
        "If component and citation together cover at least 4 more claims than the gene window, expand that retrieval across the 65 claims before any model.",
        "If at least 10 claims have a direct test and at most one of those claims has a contradiction, keep the supportive skew and enlarge the source cohort without selecting on later outcomes.",
        "Otherwise stop. The pilot did not justify a predictor.",
    ],
}

_HEADERS = {"User-Agent": "ras-mapk-v1-retrieval"}
_made_request = False

_SETTING_TERMS = {
    "thyroid cancer": ["thyroid"],
    "NSCLC": ["NSCLC", '"non-small cell"', '"lung cancer"'],
    "colorectal cancer": ["colorectal", '"colon cancer"'],
    "pancreatic cancer": ["pancreatic"],
    "melanoma": ["melanoma"],
    "Noonan syndrome": ["Noonan"],
    "leukemia": ["leukemia", "leukaemia"],
}
_GENE_CLAUSE = {
    "KRAS": '(KRAS OR "K-ras" OR "K-RAS")',
    "NRAS": '(NRAS OR "N-ras" OR "N-RAS")',
    "HRAS": '(HRAS OR "H-ras" OR "H-RAS")',
    "BRAF": '(BRAF OR "B-RAF" OR "B-raf")',
    "RAF1": '(RAF1 OR "C-RAF" OR CRAF)',
    "ERBB2": '(ERBB2 OR HER2 OR "HER-2")',
    "EGFR": '(EGFR OR ERBB1 OR "HER-1")',
    "PIK3CA": "(PIK3CA)",
    "MAP2K1": '(MAP2K1 OR MEK1)',
    "NF1": "(NF1)",
}
_GENE_MENTIONS = {
    "KRAS": ("kras", "k-ras"),
    "NRAS": ("nras", "n-ras"),
    "HRAS": ("hras", "h-ras"),
    "BRAF": ("braf", "b-raf"),
    "RAF1": ("raf1", "c-raf", "craf"),
    "ERBB2": ("erbb2", "her2", "her-2"),
    "EGFR": ("egfr", "erbb1"),
    "PIK3CA": ("pik3ca",),
    "MAP2K1": ("map2k1", "mek1"),
    "NF1": ("nf1",),
}
_ALLELE = re.compile(r"\b([A-Z]\d{2,4}[A-Z])\b")
_CODON = re.compile(r"\bcodons?\s+(\d+)\b", re.I)
_FUSION = re.compile(r"\b([A-Z][A-Z0-9]{1,}-[A-Z][A-Z0-9]{1,})\b")
_DOMAIN = re.compile(r"\b(CR[0-9])\b")
_SYNDROME = re.compile(r"\b([A-Z][a-z]+)\s+syndrome\b")
_PATHWAY = re.compile(r"\b(PI3K|AKT|mTOR|Stat3|heregulin|EML4-ALK)\b")
_OUTCOME = (
    (re.compile(r"hypertrophic cardiomyopathy|\bHCM\b", re.I), '"hypertrophic cardiomyopathy"'),
    (re.compile(r"mutually exclusive", re.I), '"mutually exclusive"'),
    (re.compile(r"intraepithelial|\bmPanIN\b|\bPanIN\b", re.I), "PanIN"),
    (re.compile(r"\bprognos", re.I), "prognosis"),
    (re.compile(r"response rate|progression-free|\bPFS\b", re.I), '"response rate"'),
    (re.compile(r"\brecurrent\b|\bpersistent disease\b", re.I), "recurrent"),
    (re.compile(r"\bamplification\b", re.I), "amplification"),
    (re.compile(r"\bknock-?down\b", re.I), "knockdown"),
    (re.compile(r"\bresistant\b|\bresistance\b", re.I), "resistance"),
)
_SYSTEM = (
    (re.compile(r"\b(mouse|mice|murine)\b", re.I), "mouse"),
    (re.compile(r"\bcell lines?\b", re.I), '"cell line"'),
    (re.compile(r"\bacinar\b", re.I), "acinar"),
    (re.compile(r"\bsolid tumors?\b", re.I), '"solid tumor"'),
    (re.compile(r"\bphase I\b", re.I), '"phase I"'),
)


def in_horizon(paper: dict, cutoff: str, horizon_years: int = HORIZON_YEARS) -> bool:
    year = int(paper.get("year") or 0)
    source_year = int(cutoff[:4])
    if year < source_year or year > source_year + horizon_years:
        return False
    date = paper.get("date") or f"{year}-01-01"
    if date < cutoff and year == source_year:
        return False
    return True


def strata_from_labels(labels_a: list[dict], labels_b: list[dict]) -> dict[str, dict]:
    by_b = {row["claim_id"]: row for row in labels_b}
    out: dict[str, dict] = {}
    for row in labels_a:
        claim_id = row["claim_id"]
        other = by_b[claim_id]
        left = claim_status(row["followups"]) if row.get("followups") else row["status"]
        right = claim_status(other["followups"]) if other.get("followups") else other["status"]
        if left == right == "support":
            stratum = "agreed_support"
        elif left == right == "unresolved":
            stratum = "agreed_unresolved"
        else:
            stratum = "discordant"
        out[claim_id] = {"stratum": stratum, "status_a": left, "status_b": right}
    return out


def select_pilot(claims: list[dict], strata: dict[str, dict], seed: int = PILOT_SEED) -> dict:
    """Census of agreed support, plus three unresolved claims from each year bin."""
    kept = [row for row in claims if not row.get("skip")]
    by_id = {row["claim_id"]: row for row in kept}
    support = sorted(cid for cid, info in strata.items() if info["stratum"] == "agreed_support" and cid in by_id)
    unresolved = [cid for cid, info in strata.items() if info["stratum"] == "agreed_unresolved" and cid in by_id]
    discordant = sorted(cid for cid, info in strata.items() if info["stratum"] == "discordant")
    picked_unresolved: list[str] = []
    rng = random.Random(seed)
    for start, end in YEAR_BINS:
        pool = sorted(
            cid for cid in unresolved if start <= int(by_id[cid]["cutoff"][:4]) <= end
        )
        if len(pool) < UNRESOLVED_PER_BIN:
            raise ValueError(f"year bin {start}-{end} has {len(pool)} agreed-unresolved claims")
        picked_unresolved.extend(rng.sample(pool, UNRESOLVED_PER_BIN))
    selected = []
    for claim_id in support:
        selected.append(_pilot_row(by_id[claim_id], strata[claim_id]))
    for claim_id in sorted(picked_unresolved):
        selected.append(_pilot_row(by_id[claim_id], strata[claim_id]))
    return {
        "seed": seed,
        "horizon_years": HORIZON_YEARS,
        "support_rule": SUPPORT_RULE,
        "unresolved_per_bin": UNRESOLVED_PER_BIN,
        "year_bins": [list(pair) for pair in YEAR_BINS],
        "n": len(selected),
        "n_agreed_support": len(support),
        "n_agreed_unresolved_drawn": len(picked_unresolved),
        "n_discordant_excluded": len(discordant),
        "discordant_excluded": [
            {"claim_id": cid, **strata[cid]} for cid in discordant
        ],
        "decision_rule": DECISION_RULE,
        "claims": selected,
    }


def _pilot_row(claim: dict, info: dict) -> dict:
    spec = specify_claim(claim)
    return {
        "claim_id": claim["claim_id"],
        "pmid": str(claim["pmid"]),
        "cutoff": claim["cutoff"],
        "year": int(claim["cutoff"][:4]),
        "stratum": info["stratum"],
        "prior_status_a": info["status_a"],
        "prior_status_b": info["status_b"],
        "spec": spec,
        "queries": {
            "component": component_query(claim),
            "citation": citation_query(claim),
        },
    }


def specify_claim(claim: dict) -> dict:
    setting, focus = query_terms(claim)
    return {
        "gene": claim["gene"],
        "intervention": claim.get("exposure") or "",
        "setting": claim.get("population") or "",
        "outcome": claim.get("effect") or "",
        "direction": claim.get("direction") or "",
        "conditions": [
            f"Gene is {claim['gene']}.",
            "The exposure matches the intervention named here.",
            "The biological setting matches the setting named here.",
            "The paper reports the outcome named here, so the direction can be compared.",
            "A review, comment, or republication of the source experiment is not an independent test.",
            "Partial overlap is not a direct test.",
            "A missing follow-up is unresolved, not a contradiction.",
        ],
        "setting_terms": setting,
        "focus_terms": focus,
        "quote": claim.get("quote") or "",
    }


def query_terms(claim: dict) -> tuple[list[str], list[str]]:
    text = " ".join(
        str(claim.get(key) or "")
        for key in ("population", "exposure", "effect", "quote")
    )
    setting: list[str] = []
    for disease in claim.get("diseases") or []:
        setting.extend(_SETTING_TERMS.get(disease, [disease]))
    for match in _SYNDROME.findall(text):
        if match.lower() not in {"the", "this"}:
            setting.append(match)
    if not setting:
        for pattern, token in _SYSTEM:
            if pattern.search(text):
                setting.append(token)
    focus: list[str] = []
    for drug in claim.get("drugs") or []:
        focus.append(drug)
    focus.extend(_ALLELE.findall(text))
    focus.extend(f'"codon {codon}"' for codon in _CODON.findall(text))
    focus.extend(_FUSION.findall(text))
    focus.extend(_DOMAIN.findall(text))
    focus.extend(match.group(0) for match in _PATHWAY.finditer(text))
    for pattern, token in _OUTCOME:
        if pattern.search(text):
            focus.append(token)
    return _unique(setting), _unique(focus)


def component_query(claim: dict) -> str:
    setting, focus = query_terms(claim)
    parts = [_gene_clause(claim["gene"])]
    if setting:
        parts.append("(" + " OR ".join(setting) + ")")
    if focus:
        parts.append("(" + " OR ".join(focus) + ")")
    parts.append(f"PUB_YEAR:[{_year_window(claim)}] AND SRC:MED NOT PMID:{claim['pmid']}")
    return " AND ".join(parts)


def citation_query(claim: dict) -> str:
    return (
        f'REF:"{claim["pmid"]}" AND PUB_YEAR:[{_year_window(claim)}] '
        f"AND SRC:MED NOT PMID:{claim['pmid']}"
    )


def _year_window(claim: dict) -> str:
    year = int(str(claim.get("cutoff") or claim.get("year"))[:4])
    return f"{year} TO {year + HORIZON_YEARS}"


def _gene_clause(gene: str) -> str:
    return _GENE_CLAUSE.get(gene, f"({gene})")


def mentions_gene(paper: dict, gene: str) -> bool:
    text = f"{paper.get('title') or ''} {paper.get('abstract') or ''}".lower()
    aliases = _GENE_MENTIONS.get(gene, (gene.lower(),))
    if any(alias in text for alias in aliases):
        return True
    for alias, canonical in GENE_ALIASES.items():
        if canonical == gene and alias.lower() in text:
            return True
    return False


def _unique(terms: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for term in terms:
        key = term.lower()
        if not term or key in seen:
            continue
        seen.add(key)
        out.append(term)
    return out


def screen_hit(paper: dict, claim: dict, *, require_gene: bool = True) -> str | None:
    """Return an exclusion reason, or None when the paper stays in the pool."""
    if str(paper.get("pmid") or "") == str(claim["pmid"]):
        return "source"
    if not in_horizon(paper, claim["cutoff"]):
        return "outside_horizon"
    if is_review(paper):
        return "review"
    if not (paper.get("abstract") or "").strip():
        return "no_abstract"
    if require_gene and not mentions_gene(paper, claim["gene"]):
        return "no_gene_mention"
    return None


def blind_candidates(groups: list[dict], seed: int = PILOT_SEED) -> tuple[list[dict], list[dict]]:
    """Hide retrieval method. Returns (candidate packets, provenance rows)."""
    packets = []
    provenance = []
    for group in groups:
        rng = random.Random(f"{seed}:{group['claim_id']}")
        rows = list(group["kept"])
        rng.shuffle(rows)
        candidates = []
        for index, row in enumerate(rows, start=1):
            paper = row["paper"]
            candidate_id = f"{group['claim_id']}-P{index:02d}"
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "pmid": paper["pmid"],
                    "date": paper.get("date") or "",
                    "year": int(paper.get("year") or 0),
                    "title": paper.get("title") or "",
                    "abstract": paper.get("abstract") or "",
                    "pub_types": list(paper.get("pub_types") or []),
                }
            )
            provenance.append(
                {
                    "claim_id": group["claim_id"],
                    "candidate_id": candidate_id,
                    "pmid": paper["pmid"],
                    "methods": list(row["methods"]),
                    "component_rank": row.get("component_rank"),
                    "citation_rank": row.get("citation_rank"),
                }
            )
        packets.append(
            {
                "claim_id": group["claim_id"],
                "pmid": group["pmid"],
                "cutoff": group["cutoff"],
                "horizon_years": HORIZON_YEARS,
                "spec": group["spec"],
                "candidates": candidates,
            }
        )
    return packets, provenance


def decide(n_claims_with_test: int, n_gene_claims: int, n_claims_with_contradiction: int) -> list[str]:
    actions: list[str] = []
    gain = n_claims_with_test - n_gene_claims
    scarce = n_claims_with_test < DECISION_RULE["scarce_if_claims_with_direct_test_below"]
    expand = gain >= DECISION_RULE["expand_retrieval_if_gain_at_least"]
    uniform = (
        n_claims_with_test >= DECISION_RULE["uniform_support_if_claims_with_direct_test_at_least"]
        and n_claims_with_contradiction <= DECISION_RULE["uniform_support_if_claims_with_contradiction_at_most"]
    )
    if scarce:
        actions.append("do_not_build_predictor")
    if expand:
        actions.append("expand_retrieval_to_65")
    if uniform and not scarce:
        actions.append("enlarge_source_cohort")
    if not actions:
        actions.append("stop")
    return actions


def _search(params: dict, transport=None) -> dict:
    global _made_request
    if transport is not None:
        return transport(params)
    if _made_request:
        time.sleep(0.2)
    _made_request = True
    response = requests.get(config.EUROPEPMC, params=params, headers=_HEADERS, timeout=90)
    response.raise_for_status()
    return response.json()


def _hits(payload: dict) -> list[dict]:
    raw = (payload.get("resultList") or {}).get("result") or []
    if isinstance(raw, dict):
        return [raw]
    return [row for row in raw if isinstance(row, dict)]


def _paper(item: dict) -> dict:
    paper = parse_europepmc_result(item)
    paper["pmcid"] = "" if item.get("pmcid") in (None, "") else str(item.get("pmcid"))
    return paper


def fetch_query(query: str, page_size: int, sort: str | None, transport=None) -> tuple[int, list[dict]]:
    params = {
        "query": query,
        "format": "json",
        "resultType": "core",
        "pageSize": page_size,
    }
    if sort:
        params["sort"] = sort
    payload = _search(params, transport=transport)
    hit_count = int(payload.get("hitCount") or 0)
    return hit_count, [_paper(item) for item in _hits(payload)]


def collect_claim(claim_row: dict, gene_articles: list[dict], transport=None) -> dict:
    claim = {
        "pmid": claim_row["pmid"],
        "cutoff": claim_row["cutoff"],
        "gene": claim_row["spec"]["gene"],
    }
    merged: dict[str, dict] = {}
    exclusions: dict[str, int] = {}

    def add(paper: dict, method: str, component_rank: int | None, citation_rank: int | None) -> None:
        reason = screen_hit(paper, claim)
        if reason:
            exclusions[reason] = exclusions.get(reason, 0) + 1
            return
        pmid = str(paper["pmid"])
        slot = merged.get(pmid)
        if slot is None:
            slot = {
                "paper": paper,
                "methods": [],
                "component_rank": None,
                "citation_rank": None,
            }
            merged[pmid] = slot
        if method not in slot["methods"]:
            slot["methods"].append(method)
        if component_rank is not None:
            slot["component_rank"] = component_rank
        if citation_rank is not None:
            slot["citation_rank"] = citation_rank

    for article in gene_articles:
        add(article, "gene_window", None, None)

    component_hits, component_papers = fetch_query(
        claim_row["queries"]["component"], COMPONENT_CAP + TAIL_CAP, None, transport
    )
    for rank, paper in enumerate(component_papers, start=1):
        method = "component" if rank <= COMPONENT_CAP else "component_tail"
        add(paper, method, rank, None)

    citation_hits, citation_papers = fetch_query(
        claim_row["queries"]["citation"], CITATION_FETCH, "P_PDATE_D asc", transport
    )
    examined = 0
    citation_kept = 0
    for paper in citation_papers:
        if str(paper.get("pmid") or "") == str(claim["pmid"]):
            continue
        if not in_horizon(paper, claim["cutoff"]):
            continue
        examined += 1
        before = len(merged)
        methods_before = set(merged.get(str(paper["pmid"]), {}).get("methods") or [])
        if citation_kept >= CITATION_CAP:
            break
        add(paper, "citation", None, examined)
        after_methods = set(merged.get(str(paper["pmid"]), {}).get("methods") or [])
        if "citation" in after_methods and "citation" not in methods_before:
            citation_kept += 1
        elif str(paper["pmid"]) not in merged or len(merged) == before and "citation" not in after_methods:
            pass

    kept = list(merged.values())
    for row in kept:
        row["methods"] = sorted(row["methods"])
    return {
        "claim_id": claim_row["claim_id"],
        "pmid": claim_row["pmid"],
        "cutoff": claim_row["cutoff"],
        "spec": claim_row["spec"],
        "kept": kept,
        "stats": {
            "hit_count_component": component_hits,
            "hit_count_citation": citation_hits,
            "n_gene_window_input": len(gene_articles),
            "n_component_fetched": len(component_papers),
            "n_citation_fetched": len(citation_papers),
            "n_citation_examined_in_horizon": examined,
            "n_unique_screened": len(kept),
            "exclusions": exclusions,
            "n_by_method": {
                method: sum(1 for row in kept if method in row["methods"])
                for method in ("gene_window", "component", "citation", "component_tail")
            },
        },
    }


def gene_articles_for(claim_id: str, packets: list[dict]) -> list[dict]:
    for packet in packets:
        if packet.get("claim_id") == claim_id:
            return list((packet.get("later") or {}).get("articles") or [])
    return []


def run_pilot(transport=None, write: bool = True) -> dict:
    claims = json.loads((config.ROOT / "bench" / "claims.json").read_text(encoding="utf-8"))["claims"]
    labels_a = json.loads((config.ROOT / "bench" / "labels_a.json").read_text(encoding="utf-8"))["claims"]
    labels_b = json.loads((config.ROOT / "bench" / "labels_b.json").read_text(encoding="utf-8"))["claims"]
    packets = json.loads((config.ROOT / "bench" / "packets.json").read_text(encoding="utf-8"))["packets"]
    pilot = select_pilot(claims, strata_from_labels(labels_a, labels_b))
    if write:
        SAMPLE_OUT.write_text(json.dumps(pilot, indent=2) + "\n", encoding="utf-8")
    groups = []
    stats = []
    for row in pilot["claims"]:
        collected = collect_claim(row, gene_articles_for(row["claim_id"], packets), transport)
        groups.append(collected)
        stats.append({"claim_id": row["claim_id"], "stratum": row["stratum"], **collected["stats"]})
    candidates, provenance_rows = blind_candidates(groups)
    payload = {
        "n_claims": len(candidates),
        "caps": {
            "component": COMPONENT_CAP,
            "component_tail": TAIL_CAP,
            "citation": CITATION_CAP,
            "gene_window": 8,
        },
        "judgment": {
            "variables": ["direct_test_found", "outcome_if_found"],
            "relations": [
                "direct_support",
                "direct_contradict",
                "direct_mixed",
                "partial",
                "not_test",
                "unclear_reporting",
            ],
            "note": "Hide retrieval method. Stratum is not an input to judgment. Agreement is inter-agent agreement.",
        },
        "claims": candidates,
    }
    provenance = {"stats": stats, "rows": provenance_rows}
    if write:
        CANDIDATES_OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        PROVENANCE_OUT.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return {"pilot": pilot, "candidates": payload, "provenance": provenance}


def main() -> None:
    result = run_pilot(write=True)
    screened = sum(row["n_unique_screened"] for row in result["provenance"]["stats"])
    print(f"pilot claims={result['pilot']['n']} screened={screened}")


if __name__ == "__main__":
    main()
