"""Retrieval that keeps the named perturbation in the query.

The component query drops symbols shorter than five characters, so an exposure
such as Atg5 never reaches the ranker. This query ANDs those named
perturbation symbols back in. A related gene is not added. Finding Atg7 can
suggest where to look; it does not make an Atg7 paper a test of an Atg5 claim.

The original route lists stay frozen. This module is a separate method.
"""

from __future__ import annotations

import re

from src.bench.graph import BUDGET, EuropePmc, _search_papers, take_budget
from src.bench.retrieve import _gene_clause, _year_window, component_query, query_terms

# A symbol that contains a digit, including short gene names the word
# tokenizer drops. Not fitted to a result list.
_PERTURBATION = re.compile(r"\b([A-Za-z][A-Za-z0-9]{1,8}\d[A-Za-z0-9]{0,4})\b")


def perturbation_terms(claim: dict) -> list[str]:
    """Symbols in the exposure that name the perturbation, excluding the claim gene."""
    gene = str(claim.get("gene") or "")
    found: list[str] = []
    seen: set[str] = set()
    for token in _PERTURBATION.findall(str(claim.get("exposure") or "")):
        key = token.lower()
        if key == gene.lower() or key in seen:
            continue
        seen.add(key)
        found.append(token)
    return found


def mechanism_query(claim: dict) -> str:
    """Component query, with named perturbation symbols required as well.

    When the exposure names no extra symbol, this is the existing component query.
    """
    extra = perturbation_terms(claim)
    if not extra:
        return component_query(claim)
    setting, focus = query_terms(claim)
    parts = [_gene_clause(claim["gene"])]
    if setting:
        parts.append("(" + " OR ".join(setting) + ")")
    if focus:
        parts.append("(" + " OR ".join(focus) + ")")
    symbols = []
    for token in extra:
        symbols.append(token)
        upper = token.upper()
        if upper != token:
            symbols.append(upper)
    parts.append("(" + " OR ".join(symbols) + ")")
    parts.append(f"PUB_YEAR:[{_year_window(claim)}] AND SRC:MED NOT PMID:{claim['pmid']}")
    return " AND ".join(parts)


def fetch_mechanism(claim: dict, client: EuropePmc | None = None, budget: int = BUDGET) -> list[dict]:
    """First papers that pass the existing screen. Does not write frozen lists."""
    client = client or EuropePmc()
    fetched = _search_papers(client, mechanism_query(claim), max(budget * 2, 40), None)
    row = {
        "pmid": claim["pmid"],
        "cutoff": claim["cutoff"],
        "gene": claim["gene"],
        "spec": {"gene": claim["gene"]},
    }
    kept, _reviews = take_budget(fetched, row, budget=budget, require_gene=True)
    return kept
