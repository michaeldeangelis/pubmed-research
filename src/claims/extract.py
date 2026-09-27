"""Attach genes, drugs, diseases, evidence tags, and relations to papers."""

from __future__ import annotations

import json
import re

from src import config
from src.schema import (
    DISEASE_ALIASES,
    DISEASES,
    DRUGS,
    GENE_ALIASES,
    GENES,
    RELATIONS,
    evidence_tags,
    paper_text,
)

_INHIBIT = re.compile(r"inhibit|inhibitor|blockade|antagonist", re.I)
_ACTIVATE = re.compile(r"activat|phosphorylat", re.I)
_RESIST = re.compile(r"resistant|resistance|relapse|bypass", re.I)
_ASSOCIATE = re.compile(r"associat|mutation|mutant", re.I)
_SENTENCE = re.compile(r"[.?!]+")


def _compiled(mapping: dict[str, str]) -> list[tuple[re.Pattern[str], str]]:
    terms = sorted(mapping, key=lambda term: (-len(term), term.lower()))
    return [
        (re.compile(rf"\b{re.escape(term)}\b", re.I), mapping[term]) for term in terms
    ]


def _mapping(vocab: list[str], aliases: dict[str, str] | None = None) -> dict[str, str]:
    mapping = {token: token for token in vocab}
    if aliases:
        mapping.update(aliases)
    return mapping


_GENE_PATTERNS = _compiled(_mapping(GENES, GENE_ALIASES))
_DRUG_PATTERNS = _compiled(_mapping(DRUGS))
_DISEASE_PATTERNS = _compiled(_mapping(DISEASES, DISEASE_ALIASES))


def _match(
    text: str,
    patterns: list[tuple[re.Pattern[str], str]],
    vocab: list[str],
) -> list[str]:
    occupied: list[tuple[int, int]] = []
    found: set[str] = set()
    for pattern, canonical in patterns:
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(start < prev_end and end > prev_start for prev_start, prev_end in occupied):
                continue
            occupied.append((start, end))
            found.add(canonical)
    return [token for token in vocab if token in found]


def _add(relations: list[dict], seen: set[tuple[str, str, str]], head: str, rel: str, tail: str) -> None:
    if rel not in RELATIONS:
        return
    key = (head, rel, tail)
    if key in seen:
        return
    seen.add(key)
    relations.append({"head": head, "rel": rel, "tail": tail})


def _relations(text: str) -> list[dict]:
    relations: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for sentence in _SENTENCE.split(text):
        if not sentence.strip():
            continue
        genes = _match(sentence, _GENE_PATTERNS, GENES)
        drugs = _match(sentence, _DRUG_PATTERNS, DRUGS)
        diseases = _match(sentence, _DISEASE_PATTERNS, DISEASES)
        if _INHIBIT.search(sentence):
            for drug in drugs:
                for gene in genes:
                    _add(relations, seen, drug, "inhibits", gene)
        if _ACTIVATE.search(sentence):
            for index, gene_a in enumerate(genes):
                for gene_b in genes[index + 1 :]:
                    _add(relations, seen, gene_a, "activates", gene_b)
        if _RESIST.search(sentence):
            for drug in drugs:
                for gene in genes:
                    _add(relations, seen, drug, "resistant_to", gene)
        if _ASSOCIATE.search(sentence):
            for gene in genes:
                for disease in diseases:
                    _add(relations, seen, gene, "associated_with", disease)
    return relations


def enrich_paper(paper: dict) -> dict:
    enriched = dict(paper)
    text = paper_text(paper)
    enriched["genes"] = _match(text, _GENE_PATTERNS, GENES)
    enriched["drugs"] = _match(text, _DRUG_PATTERNS, DRUGS)
    enriched["diseases"] = _match(text, _DISEASE_PATTERNS, DISEASES)
    enriched["evidence_tags"] = evidence_tags(text)
    enriched["relations"] = _relations(text)
    return enriched


def _read_jsonl(path) -> list[dict]:
    papers: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                papers.append(json.loads(line))
    return papers


def main() -> None:
    papers = _read_jsonl(config.PAPERS_PATH)
    config.ENRICHED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.ENRICHED_PATH.open("w", encoding="utf-8") as handle:
        for paper in papers:
            handle.write(json.dumps(enrich_paper(paper), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
