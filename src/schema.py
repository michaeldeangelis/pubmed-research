"""Shared vocabulary and label rules for the RAS/MAPK v1.

Training may use genes, drugs, diseases, and the clinical flag.
human_genetics and resistance tags are held out: they are never model inputs
and never loss targets. The probe asks whether contextual states still track
prior genetic evidence.
"""

from __future__ import annotations

import re

GENES = [
    "KRAS",
    "NRAS",
    "HRAS",
    "BRAF",
    "RAF1",
    "MAP2K1",
    "MAP2K2",
    "MAPK1",
    "MAPK3",
    "NF1",
    "SOS1",
    "PTPN11",
    "EGFR",
    "ERBB2",
    "MET",
    "PIK3CA",
    "STK11",
]

GENE_ALIASES = {
    "MEK1": "MAP2K1",
    "MEK2": "MAP2K2",
    "ERK1": "MAPK1",
    "ERK2": "MAPK3",
    "HER2": "ERBB2",
    "ERBB1": "EGFR",
    "C-RAF": "RAF1",
    "CRAF": "RAF1",
    "B-RAF": "BRAF",
}

DRUGS = [
    "sotorasib",
    "adagrasib",
    "divarasib",
    "trametinib",
    "cobimetinib",
    "binimetinib",
    "selumetinib",
    "dabrafenib",
    "vemurafenib",
    "encorafenib",
    "sorafenib",
    "regorafenib",
    "cetuximab",
    "panitumumab",
    "erlotinib",
    "gefitinib",
    "osimertinib",
    "alpelisib",
]

DISEASES = [
    "NSCLC",
    "colorectal cancer",
    "melanoma",
    "pancreatic cancer",
    "thyroid cancer",
    "leukemia",
    "Noonan syndrome",
]

DISEASE_ALIASES = {
    "non-small cell lung": "NSCLC",
    "non small cell lung": "NSCLC",
    "nsclc": "NSCLC",
    "lung adenocarcinoma": "NSCLC",
    "colorectal": "colorectal cancer",
    "colon cancer": "colorectal cancer",
    "rectal cancer": "colorectal cancer",
    "melanoma": "melanoma",
    "pancreatic": "pancreatic cancer",
    "pancreas cancer": "pancreatic cancer",
    "thyroid": "thyroid cancer",
    "leukemia": "leukemia",
    "leukaemia": "leukemia",
    "noonan": "Noonan syndrome",
}

EVIDENCE_TAGS = (
    "human_genetics",
    "animal_model",
    "mechanistic",
    "clinical",
    "resistance",
)

HELD_OUT_TAGS = ("human_genetics", "resistance")
TRAIN_BATCH_KEYS = (
    "text",
    "year",
    "genes",
    "drugs",
    "diseases",
    "y_genes",
    "y_drugs",
    "y_clinical",
    "loss_mask",
    "pad_mask",
)
FORBIDDEN_BATCH_KEYS = (
    "human_genetics",
    "resistance",
    "evidence_tags",
    "prior_genetic_evidence",
    "abstract",
    "title",
)

RELATIONS = ("inhibits", "activates", "associated_with", "resistant_to")

PAPER_KEYS = (
    "pmid",
    "date",
    "year",
    "title",
    "abstract",
    "journal",
    "mesh",
    "pub_types",
)

PROBE_REPORT_KEYS = (
    "encoder",
    "n_papers",
    "n_test_positions",
    "prior_genetics_auroc_frozen",
    "prior_genetics_auroc_contextual",
    "prior_genetics_auroc_lift",
    "resistance_auroc_contextual",
    "ablation_delta_clinical_bce_genetics_context",
    "ablation_delta_clinical_bce_no_genetics_context",
    "selective_ablation",
    "head_genetics_attention",
    "notes",
)

_TAG_PATTERNS = {
    "human_genetics": re.compile(
        r"\b(gwas|germline|mendelian|polymorphism|familial|odds ratio|"
        r"loss[- ]of[- ]function|variant allele)\b|\brs\d+\b",
        re.I,
    ),
    "animal_model": re.compile(
        r"\b(mouse|murine|xenografts?|patient[- ]derived xenograft|pdx|\brats?\b)\b",
        re.I,
    ),
    "mechanistic": re.compile(
        r"\b(phosphorylation|signalling|signaling|kinase assay|downstream|pathway)\b",
        re.I,
    ),
    "clinical": re.compile(
        r"\b(phase\s*(i{1,3}|[123])|clinical trial|overall survival|"
        r"progression[- ]free|patients with)\b",
        re.I,
    ),
    "resistance": re.compile(
        r"\b(resistance|resistant|relapse|bypass|reactivation|acquired mutation)\b",
        re.I,
    ),
}

_TAG_RE = re.compile(r"<[^>]+>")


def strip_markup(text: str) -> str:
    return _TAG_RE.sub("", text or "")


def paper_text(paper: dict) -> str:
    return strip_markup(f"{paper.get('title', '')} {paper.get('abstract', '')}")


def evidence_tags(text: str) -> list[str]:
    found = [name for name, pattern in _TAG_PATTERNS.items() if pattern.search(text)]
    return found


def is_clinical(text: str) -> bool:
    return "clinical" in evidence_tags(text)


def multi_hot(items: list[str], vocab: list[str]) -> list[float]:
    present = set(items)
    return [1.0 if token in present else 0.0 for token in vocab]


def primary_genes(symbols: list[str]) -> list[str]:
    """Genes that anchor a paper to a timeline, in vocabulary order."""
    present = set(symbols)
    return [gene for gene in GENES if gene in present]


def prior_genetic_evidence(tags_before: list[list[str]]) -> bool:
    """True when any earlier paper in the same timeline carries human genetics."""
    return any("human_genetics" in tags for tags in tags_before)
