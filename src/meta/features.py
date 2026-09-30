"""Metascience v1 features F1-F3, covariates, and the measurement gate.

Contract: ``docs/specs/2026-09-29-metascience-design.md``. The analysis plan is
``experiments.md`` (entry "2026-09-29 metascience v1").

Commands
--------
``python -m src.meta.features``
    Reads ``data/meta/papers.jsonl`` and ``data/meta/icite.jsonl``, writes
    ``data/meta/features.jsonl`` (one row per iCite research article, with
    eligibility), writes the v2 blind validation packets
    (``bench/meta_validation/packets_v2.json``; the v1 ``packets.json`` is
    frozen and never rewritten), and prints marginal counts only.
``python -m src.meta.features --pathway {egfr,pi3k}``
    Metascience v2: the same rules on a pathway corpus under
    ``data/meta/pathways/<name>/`` (``features.jsonl``,
    ``manifest_features.json``), with that pathway's eligibility and
    gene-group patterns (``src.meta.pathways``). Writes no packets.
``python -m src.meta.features pathway-packets``
    Metascience v2 gate packets: 150 eligible preclinical papers per pathway
    (``default_rng(2)`` over sorted PMIDs; egfr drawn first, a PMID drawn for
    egfr is removed from the pi3k pool) in
    ``bench/meta_validation/packets_pathways.json`` (pmid/title/abstract/mesh
    only) and the pmid -> pathway map in ``packets_pathways_index.json``.
``python -m src.meta.features score <extracted.json> [--packets P] [--out O]``
    Compares the rules with a blind model extraction of the packets (default
    ``bench/meta_validation/packets.json``) and writes the gate record (default
    ``results/meta_measurement_gate.json``): Cohen's kappa per feature,
    confusion matrices, invalid-label counts, gate pass at kappa >= 0.60.

Blinding: this module never reads outcome files (``outcome_b.jsonl``,
``impact.jsonl``) and never relates a feature to an outcome.

Rules
-----
All MeSH matching is on normalized descriptor names: qualifiers after "/" are
dropped, a leading or trailing "*" (major topic) is dropped, whitespace is
collapsed, and comparison is case-insensitive.

Presence of each system (used by F3, and by F1 through precedence):

* human samples: MeSH ``Humans`` AND one of: a term in
  ``HUMAN_SPECIFIC_TERMS`` (patient material, staging, cohort designs); a term
  in ``HUMAN_AGE_TERMS`` when no animal is present; a term in
  ``HUMAN_GENERIC_TERMS`` (outcome analyses, IHC, mutation analysis) or
  ``HUMAN_SEX_TERMS`` when no animal AND no cell term is present. The last
  two can therefore make F1 ``human_samples`` but never add a system to F3.
* animal: any term in ``ANIMAL_STRONG_TERMS`` or with a prefix in
  ``ANIMAL_STRONG_PREFIXES`` (in vivo designs, named strains, whole-organism
  models); OR a term in ``ANIMAL_WEAK_TERMS`` (a bare species such as
  ``Mice``) when either no cell term is present or the abstract matches
  ``IN_VIVO_TEXT``. A bare species plus a cell line (for example ``Mice`` +
  ``NIH 3T3 Cells``) is how MeSH indexes work on a mouse cell line, so it only
  counts as animal when the text shows living animals.
* cell: any term in ``CELL_TERMS`` (cell lines, cultured cells, culture
  methods, transfection, and species tags that MeSH adds only for common cell
  lines such as COS, Vero, CHO and Sf9).

When a paper has no MeSH headings at all (unindexed), presence falls back to
the text patterns ``HUMAN_SAMPLE_TEXT``, ``IN_VIVO_TEXT`` and ``CELL_TEXT``.

F1 model_system (precedence human_samples > animal > cell_only > other):

* ``human_samples``: human samples present, no animal, no cell terms.
* ``animal``: animal present (with or without cells or human samples).
* ``cell_only``: cell terms present, no animal (human samples may also be
  present; the mixture is captured by F3).
* ``other``: none of the above.

F2 human_genetics: any MeSH term in ``HUMAN_GENETICS_MESH`` OR the title or
abstract matches ``HUMAN_GENETICS_TEXT``.

F3 multi_system: at least two of {human samples, animal, cell} present.

gene_group: first case-sensitive match in the order KRAS, BRAF, NRAS/HRAS
over the title and abstract (``GENE_GROUP_PATTERNS``); ``other`` if none.

Eligibility (``eligible``, ``exclusion``; first matching reason wins):

a. ``no_mesh``: no MeSH headings (not MEDLINE-indexed).
b. ``off_topic``: no case-sensitive gene mention (KRAS, BRAF, NRAS/HRAS,
   MAP2K1/2, MEK1/2) and no drug name (trametinib, sotorasib, vemurafenib,
   dabrafenib; any case) in the title or abstract.
c. ``non_primary``: any publication type in ``NON_PRIMARY_PUB_TYPES``, even if
   the paper is also a "Journal Article".
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
META_DIR = ROOT / "data" / "meta"
PAPERS_PATH = META_DIR / "papers.jsonl"
ICITE_PATH = META_DIR / "icite.jsonl"
FEATURES_PATH = META_DIR / "features.jsonl"
VALIDATION_DIR = ROOT / "bench" / "meta_validation"
PACKETS_PATH = VALIDATION_DIR / "packets.json"  # v1: frozen, never rewritten
PACKETS_V2_PATH = VALIDATION_DIR / "packets_v2.json"
PACKETS_PATHWAYS_PATH = VALIDATION_DIR / "packets_pathways.json"  # metascience v2 pathways
PACKETS_PATHWAYS_INDEX_PATH = VALIDATION_DIR / "packets_pathways_index.json"
EXTRACT_SCHEMA_PATH = VALIDATION_DIR / "schema.json"
GATE_PATH = ROOT / "results" / "meta_measurement_gate.json"

MODEL_SYSTEMS = ("human_samples", "animal", "cell_only", "other")
SYSTEMS = ("human_samples", "animal", "cell")
GENE_GROUPS = ("KRAS", "BRAF", "NRAS/HRAS", "other")
FEATURE_KEYS = (
    "pmid",
    "group",
    "model_system",
    "human_genetics",
    "multi_system",
    "gene_group",
    "n_authors",
    "n_refs",
    "eligible",
    "exclusion",
)
EXCLUSION_REASONS = ("no_mesh", "off_topic", "non_primary")
N_VALIDATION = 300
VALIDATION_SEED = 0
VALIDATION_SEED_V2 = 1
VALIDATION_SEED_PATHWAYS = 2
N_VALIDATION_PER_PATHWAY = 150
KAPPA_GATE = 0.60
MAX_DISAGREEMENTS = 40

# --- term lists (frozen by the commit that precedes the analysis run) --------

HUMAN_TERM = "Humans"
# Human-sample evidence comes in four tiers (all require MeSH "Humans"):
#   HUMAN_SPECIFIC_TERMS - patient material, cohorts, staging: always count.
#   HUMAN_AGE_TERMS      - patient age groups: count when no animal is present.
#   HUMAN_GENERIC_TERMS  - outcome analyses and generic assays that are also
#                          indexed on animal and cell work: count only when no
#                          animal AND no cell term is present (so they can make
#                          F1 human_samples but never add a system to F3).
#   HUMAN_SEX_TERMS      - Female/Male: count only when no animal AND no cell.
HUMAN_SPECIFIC_TERMS = (
    # patient tissue, blood and specimens
    "Biopsy",
    "Biopsy, Fine-Needle",
    "Biopsy, Needle",
    "Liquid Biopsy",
    "Circulating Tumor DNA",
    "Cell-Free Nucleic Acids",
    "Neoplastic Cells, Circulating",
    "Paraffin Embedding",
    "Tissue Array Analysis",
    "Tissue Banks",
    "Specimen Handling",
    "Microdissection",
    "Laser Capture Microdissection",
    "Tissue Fixation",
    "Tissue Embedding",
    "Neoplasm Staging",
    "Neoplasm Grading",
    # patient study designs
    "Retrospective Studies",
    "Prospective Studies",
    "Cohort Studies",
    "Case-Control Studies",
    "Cross-Sectional Studies",
    "Follow-Up Studies",
    "Longitudinal Studies",
    "Patient Selection",
    "Pedigree",
    "Inpatients",
    "Outpatients",
)
# MeSH assigns age-group check tags to human subjects only.
HUMAN_AGE_TERMS = (
    "Infant",
    "Infant, Newborn",
    "Child",
    "Child, Preschool",
    "Adolescent",
    "Young Adult",
    "Adult",
    "Middle Aged",
    "Aged",
    "Aged, 80 and over",
)
# Mutation analysis is listed here, not as specific: MeSH cannot say whether
# the sequenced material was a patient specimen or a cell line, and cell-line
# papers routinely carry these terms.
HUMAN_GENERIC_TERMS = (
    "DNA Mutational Analysis",
    "Genotyping Techniques",
    "Sequence Analysis, DNA",
    "Microsatellite Instability",
    "Loss of Heterozygosity",
    "Immunohistochemistry",
    "Prognosis",
    "Survival Analysis",
    "Survival Rate",
    "Kaplan-Meier Estimate",
    "Proportional Hazards Models",
    "Disease-Free Survival",
    "Progression-Free Survival",
    "Treatment Outcome",
    "Neoplasm Recurrence, Local",
    "Patient Outcome Assessment",
)
HUMAN_SEX_TERMS = ("Female", "Male")
HUMAN_SAMPLE_TERMS = HUMAN_SPECIFIC_TERMS + HUMAN_AGE_TERMS + HUMAN_GENERIC_TERMS

# In vivo designs, named strains and whole-organism models: always animal.
ANIMAL_STRONG_TERMS = (
    "Disease Models, Animal",
    "Models, Animal",
    "Xenograft Model Antitumor Assays",
    "Neoplasm Transplantation",
    "Transplantation, Heterologous",
    "Heterografts",
    "Animals, Genetically Modified",
    "Animals, Newborn",
    "Zebrafish",
    "Drosophila",
    "Drosophila melanogaster",
    "Caenorhabditis elegans",
    "Chick Embryo",
    "Oryzias",
)
# "Mice, Nude", "Mice, Transgenic", "Mice, Knockout", "Mice, SCID",
# "Mice, Inbred C57BL", "Rats, Sprague-Dawley", ...
ANIMAL_STRONG_PREFIXES = ("Mice, ", "Rats, ")

# Bare species tags. MeSH also assigns these for cell lines of that species,
# so they count as animal only without cell terms or with in vivo text.
ANIMAL_WEAK_TERMS = (
    "Mice",
    "Rats",
    "Dogs",
    "Cats",
    "Swine",
    "Rabbits",
    "Cattle",
    "Sheep",
    "Horses",
    "Guinea Pigs",
    "Cricetinae",
    "Mesocricetus",
    "Macaca mulatta",
    "Macaca fascicularis",
    "Primates",
    "Chickens",
    "Xenopus",
    "Xenopus laevis",
)

CELL_TERMS = (
    "Cell Line",
    "Cell Line, Tumor",
    "Cell Line, Transformed",
    "Cells, Cultured",
    "Tumor Cells, Cultured",
    "Clone Cells",
    "Hybridomas",
    "Cell Culture Techniques",
    "Primary Cell Culture",
    "Coculture Techniques",
    "Organoids",
    "Spheroids, Cellular",
    "Tumor Stem Cell Assay",
    "Tissue Culture Techniques",
    "Culture Media, Conditioned",
    "Human Umbilical Vein Endothelial Cells",
    "Transfection",
    # named cell lines
    "3T3 Cells",
    "3T3-L1 Cells",
    "BALB 3T3 Cells",
    "NIH 3T3 Cells",
    "Swiss 3T3 Cells",
    "A549 Cells",
    "Caco-2 Cells",
    "CHO Cells",
    "COS Cells",
    "HCT116 Cells",
    "HEK293 Cells",
    "HL-60 Cells",
    "HT29 Cells",
    "HaCaT Cells",
    "HeLa Cells",
    "Hep G2 Cells",
    "Jurkat Cells",
    "K562 Cells",
    "MCF-7 Cells",
    "PC-3 Cells",
    "PC12 Cells",
    "Sf9 Cells",
    "THP-1 Cells",
    "U937 Cells",
    "Vero Cells",
    "Madin Darby Canine Kidney Cells",
    "LLC-PK1 Cells",
    # species tags MeSH adds only for common cell lines (COS/Vero, CHO, Sf9)
    "Chlorocebus aethiops",
    "Cercopithecus aethiops",
    "Cricetulus",
    "Spodoptera",
)

HUMAN_GENETICS_MESH = (
    "Germ-Line Mutation",
    "Genetic Predisposition to Disease",
    "Polymorphism, Single Nucleotide",
    "Polymorphism, Genetic",
    "Genome-Wide Association Study",
    "Genetic Association Studies",
    "Pedigree",
    "Family",
    "Family Health",
    "Genetic Linkage",
    "Lod Score",
    "Linkage Disequilibrium",
    "Haplotypes",
    "Inheritance Patterns",
    "Consanguinity",
    "Twins",
    "Siblings",
    "Neoplastic Syndromes, Hereditary",
    "Colorectal Neoplasms, Hereditary Nonpolyposis",
    "Genetic Testing",
)

# Stricter than the src/schema.py tag regex: "odds ratio", "loss of function"
# and "variant allele" are dropped because they occur routinely in somatic and
# preclinical work and are not germline, predisposition, polymorphism, GWAS or
# familial evidence.
HUMAN_GENETICS_TEXT = re.compile(
    r"\b(germ[- ]?line|predispos\w*|polymorphisms?|SNPs?|"
    r"single[- ]nucleotide (polymorphisms?|variants?)|GWAS|"
    r"genome[- ]wide association|familial|pedigrees?|kindreds?|mendelian|"
    r"hereditary|inherited|heritab\w*|"
    r"(genetic|inherited|cancer|disease) susceptibility|"
    r"susceptibility (locus|loci|genes?|alleles?|variants?))\b"
    r"|\brs\d{3,}\b",
    re.I,
)
# SSCP and RFLP are mutation-detection methods (e.g. PCR-RFLP for KRAS codon
# 12), not polymorphism evidence; these phrases are removed before matching.
# For the same reason MeSH "Polymorphism, Restriction Fragment Length" and
# "Polymorphism, Single-Stranded Conformational" are not in HUMAN_GENETICS_MESH.
GENETICS_TEXT_EXCLUDE = re.compile(
    r"(single[- ]strand(ed)?[- ]conformation(al)?|restriction[- ]fragment[- ]length)"
    r"[- ]polymorphisms?",
    re.I,
)

# Text rules: used to confirm a bare species tag, and as the fallback when a
# paper has no MeSH headings.
IN_VIVO_TEXT = re.compile(
    r"\b(mice|rats|xenografts?|xenografted|tumou?r[- ]bearing|orthotopic\w*|"
    r"syngeneic|transgenic|knock-?out mice|mouse models?|animal models?|"
    r"zebrafish|drosophila|genetically engineered mouse)\b",
    re.I,
)
CELL_TEXT = re.compile(
    r"\b(cell lines?|cultured cells|in vitro|transfect\w*|organoids?|"
    r"HEK ?293\w*|HeLa|NIH ?3T3)\b",
    re.I,
)
HUMAN_SAMPLE_TEXT = re.compile(
    r"\b(patients' (tumou?rs|samples|specimens|tissues?)|"
    r"(samples|specimens|tissues?|tumou?rs|biops(y|ies)) (from|of) \d* ?patients|"
    r"patient[- ](samples|specimens|tumou?rs|tissues?|cohorts?)|"
    r"(tumou?r|tissue|blood|plasma|serum|surgical|clinical|archival) (samples|specimens)|"
    r"biops(y|ies)|surgically resected|resected (tumou?rs|specimens)|"
    r"paraffin[- ]embedded|cohort of|consecutive patients|"
    r"(analy[sz]ed|screened|genotyped|sequenced|examined) \d+ "
    r"(patients|tumou?rs|cases|samples|specimens))",
    re.I,
)

# Gene mentions are CASE-SENSITIVE so that capitalized plural acronyms never
# match: "HRAs"/"NRAs" (health risk appraisals, high-risk adenomas, nanorod
# arrays, native reactive astrocytes), "KRAs" (kidney retrieval areas, key
# result areas), "nRAs" (non-remote areas), "HRaS" (heart rate above sleep).
# The "ras"/"raf" part must be all lowercase, capitalized, or all uppercase
# (ras, Ras, RAS; raf, Raf, RAF); the gene letter may be upper or lower case.
# This accepts HGNC symbols (KRAS), hyphenated forms (K-RAS, K-ras, Ki-ras,
# c-Ha-ras, B-Raf), camel case (KRas, BRaf, hRas), mouse symbols (Kras, Braf,
# Map2k1, Mek1) and zebrafish/lower-case symbols (kras, braf, nras, hras).
# An upper-case-initial symbol may follow a lower-case letter, because Europe
# PMC abstracts run section labels into the text ("ResultsKRAS mutations",
# "MethodsKras(G12D)"). A trailing lower-case letter blocks a match
# ("Krasavin"), except BRAFi/BRAFis (BRAF inhibitor).
_RAS = r"-?(?:ras|Ras|RAS)"
_RAF = r"-?(?:raf|Raf|RAF)"
_GE = r"(?![a-z])"
KRAS_PATTERN = re.compile(
    r"(?:(?<![A-Z0-9])(?:Ki|K)|(?<![A-Za-z0-9])(?:ki|k))" + _RAS + r"2?" + _GE
)
BRAF_PATTERN = re.compile(r"(?:(?<![A-Z0-9])B|(?<![A-Za-z0-9])b)" + _RAF + r"(?:is?)?" + _GE)
NRAS_HRAS_PATTERN = re.compile(
    r"(?:(?<![A-Z0-9])(?:Ha|N|H)|(?<![A-Za-z0-9])(?:ha|n|h))" + _RAS + _GE
)
MEK_PATTERN = re.compile(r"(?<![A-Z0-9])(?:MAP2K[12]|Map2k[12]|MEK[12]|Mek[12])(?![A-Za-z0-9])")
DRUG_PATTERN = re.compile(r"\b(trametinib|sotorasib|vemurafenib|dabrafenib)\b", re.I)

# gene_group: first match wins, in this order (MEK genes and drugs alone -> other).
GENE_GROUP_PATTERNS = (
    ("KRAS", KRAS_PATTERN),
    ("BRAF", BRAF_PATTERN),
    ("NRAS/HRAS", NRAS_HRAS_PATTERN),
)
# on-topic (eligibility): any gene or drug pattern matches the title+abstract.
ON_TOPIC_PATTERNS = (KRAS_PATTERN, BRAF_PATTERN, NRAS_HRAS_PATTERN, MEK_PATTERN, DRUG_PATTERN)

# Non-primary publication types (normalized: lowercase, "-"/"_" -> space).
# JATS variants from Europe PMC ("review-article", "article-commentary",
# "product-review") are included with their MEDLINE equivalents.
NON_PRIMARY_PUB_TYPES = (
    "Review",
    "review-article",
    "product-review",
    "Letter",
    "Editorial",
    "Comment",
    "article-commentary",
    "Meta-Analysis",
    "Systematic Review",
    "Practice Guideline",
    "News",
)

_MARKUP = re.compile(r"<[^>]+>")


# --- helpers ------------------------------------------------------------------


def norm_term(term: str) -> str:
    """Normalize a MeSH descriptor: drop qualifiers and '*', fold case."""
    text = str(term or "").split("/", 1)[0]
    text = text.replace("*", " ")
    return re.sub(r"\s+", " ", text).strip().casefold()


def _norm_set(terms: Iterable[str]) -> frozenset[str]:
    return frozenset(norm_term(t) for t in terms)


_HUMAN = norm_term(HUMAN_TERM)
_HUMAN_SPECIFIC = _norm_set(HUMAN_SPECIFIC_TERMS)
_HUMAN_AGE = _norm_set(HUMAN_AGE_TERMS)
_HUMAN_GENERIC = _norm_set(HUMAN_GENERIC_TERMS)
_HUMAN_SEX = _norm_set(HUMAN_SEX_TERMS)
_ANIMAL_STRONG = _norm_set(ANIMAL_STRONG_TERMS)
_ANIMAL_PREFIXES = tuple(p.casefold() for p in ANIMAL_STRONG_PREFIXES)
_ANIMAL_WEAK = _norm_set(ANIMAL_WEAK_TERMS)
_CELL = _norm_set(CELL_TERMS)
_GENETICS = _norm_set(HUMAN_GENETICS_MESH)


def mesh_set(mesh: Iterable[str] | None) -> frozenset[str]:
    return frozenset(t for t in (norm_term(m) for m in (mesh or [])) if t)


def paper_text(paper: dict) -> str:
    return _MARKUP.sub("", f"{paper.get('title') or ''} {paper.get('abstract') or ''}")


def truthy(value) -> bool:
    """iCite flags arrive as booleans, 0/1, or 'Yes'/'No' strings."""
    if isinstance(value, str):
        return value.strip().lower() in {"yes", "true", "1", "y"}
    return bool(value)


# --- rules --------------------------------------------------------------------


def systems_present(paper: dict) -> dict[str, bool]:
    """Presence (not precedence) of human samples, animal, and cell systems."""
    mesh = mesh_set(paper.get("mesh"))
    text = paper_text(paper)
    if not mesh:
        return {
            "human_samples": bool(HUMAN_SAMPLE_TEXT.search(text)),
            "animal": bool(IN_VIVO_TEXT.search(text)),
            "cell": bool(CELL_TEXT.search(text)),
        }
    cell = bool(mesh & _CELL)
    strong = bool(mesh & _ANIMAL_STRONG) or any(t.startswith(_ANIMAL_PREFIXES) for t in mesh)
    weak = bool(mesh & _ANIMAL_WEAK)
    animal = strong or (weak and (not cell or bool(IN_VIVO_TEXT.search(text))))
    human = _HUMAN in mesh and (
        bool(mesh & _HUMAN_SPECIFIC)
        or (not animal and bool(mesh & _HUMAN_AGE))
        or (not animal and not cell and bool(mesh & (_HUMAN_GENERIC | _HUMAN_SEX)))
    )
    return {"human_samples": human, "animal": animal, "cell": cell}


def model_system(present: dict[str, bool]) -> str:
    if present["human_samples"] and not present["animal"] and not present["cell"]:
        return "human_samples"
    if present["animal"]:
        return "animal"
    if present["cell"]:
        return "cell_only"
    return "other"


def human_genetics(paper: dict) -> int:
    if mesh_set(paper.get("mesh")) & _GENETICS:
        return 1
    text = GENETICS_TEXT_EXCLUDE.sub(" ", paper_text(paper))
    return int(bool(HUMAN_GENETICS_TEXT.search(text)))


def multi_system(present: dict[str, bool]) -> int:
    return int(sum(bool(present[s]) for s in SYSTEMS) >= 2)


def gene_group(paper: dict, patterns=None) -> str:
    """First matching gene group (v1 RAS groups unless ``patterns`` is given)."""
    text = paper_text(paper)
    for name, pattern in GENE_GROUP_PATTERNS if patterns is None else patterns:
        if pattern.search(text):
            return name
    return "other"


_NON_PRIMARY = frozenset(re.sub(r"[-_\s]+", " ", t).strip().casefold() for t in NON_PRIMARY_PUB_TYPES)


def _norm_pub_type(t: str) -> str:
    return re.sub(r"[-_\s]+", " ", str(t or "")).strip().casefold()


def on_topic(paper: dict, patterns=None) -> bool:
    text = paper_text(paper)
    return any(p.search(text) for p in (ON_TOPIC_PATTERNS if patterns is None else patterns))


def exclusion(paper: dict, on_topic_patterns=None) -> str | None:
    """First matching exclusion reason (no_mesh, off_topic, non_primary) or None.

    ``on_topic_patterns`` defaults to the v1 RAS/MAPK gene and drug patterns.
    """
    if not mesh_set(paper.get("mesh")):
        return "no_mesh"
    if not on_topic(paper, on_topic_patterns):
        return "off_topic"
    if any(_norm_pub_type(t) in _NON_PRIMARY for t in paper.get("pub_types") or []):
        return "non_primary"
    return None


def rule_labels(paper: dict) -> dict:
    """Rule outputs for one paper, in the extractor's label space."""
    present = systems_present(paper)
    return {
        "model_system": model_system(present),
        "human_genetics": human_genetics(paper),
        "multi_system": multi_system(present),
        "systems_present": sorted(s for s in SYSTEMS if present[s]),
    }


def _pathway_patterns(pathway: str | None) -> tuple:
    """(gene_group patterns, on_topic patterns); (None, None) means v1 RAS."""
    if pathway is None:
        return None, None
    from src.meta import pathways

    cfg = pathways.get(pathway)
    return cfg["gene_groups"], cfg["on_topic"]


def feature_row(paper: dict, icite: dict, pathway: str | None = None) -> dict:
    groups, topic = _pathway_patterns(pathway)
    labels = rule_labels(paper)
    reason = exclusion(paper, topic)
    n_authors = paper.get("n_authors")
    return {
        "pmid": str(paper["pmid"]),
        "group": "clinical" if truthy(icite.get("is_clinical")) else "preclinical",
        "model_system": labels["model_system"],
        "human_genetics": labels["human_genetics"],
        "multi_system": labels["multi_system"],
        "gene_group": gene_group(paper, groups),
        "n_authors": int(n_authors) if n_authors is not None else None,
        "n_refs": None,  # supplied by the impact builder (OpenAlex)
        "eligible": reason is None,
        "exclusion": reason,
    }


def build_features(papers: list[dict], icite: list[dict], pathway: str | None = None) -> list[dict]:
    """One row per paper that iCite marks as a research article.

    ``pathway`` (v2) swaps in that pathway's eligibility and gene-group
    patterns from ``src.meta.pathways``; the model_system, human_genetics and
    multi_system rules are unchanged.
    """
    by_pmid = {str(r.get("pmid")): r for r in icite}
    rows = []
    for paper in papers:
        rec = by_pmid.get(str(paper.get("pmid")))
        if rec is None or not truthy(rec.get("is_research_article")):
            continue
        rows.append(feature_row(paper, rec, pathway))
    rows.sort(key=lambda r: int(r["pmid"]) if r["pmid"].isdigit() else r["pmid"])
    return rows


def marginals(rows: list[dict]) -> dict:
    out: dict = {"n": len(rows)}
    for key in ("group", "model_system", "human_genetics", "multi_system", "gene_group"):
        out[key] = dict(sorted(Counter(r[key] for r in rows).items(), key=lambda kv: str(kv[0])))
    for group in ("preclinical", "clinical"):
        sub = [r for r in rows if r["group"] == group]
        out[group] = {"n": len(sub)}
        for key in ("model_system", "human_genetics", "multi_system", "gene_group"):
            out[group][key] = dict(
                sorted(Counter(r[key] for r in sub).items(), key=lambda kv: str(kv[0]))
            )
    if rows and "exclusion" in rows[0]:
        for group in ("preclinical", "clinical"):
            sub = [r for r in rows if r["group"] == group]
            out[group]["exclusion"] = {
                reason: sum(1 for r in sub if r["exclusion"] == reason)
                for reason in (*EXCLUSION_REASONS, None)
            }
            out[group]["exclusion"]["eligible"] = out[group]["exclusion"].pop(None)
        elig = [r for r in rows if r["group"] == "preclinical" and r["eligible"]]
        out["eligible_preclinical"] = {"n": len(elig)}
        for key in ("model_system", "human_genetics", "multi_system", "gene_group"):
            out["eligible_preclinical"][key] = dict(
                sorted(Counter(r[key] for r in elig).items(), key=lambda kv: str(kv[0]))
            )
    return out


# --- validation packets ---------------------------------------------------------


def sample_validation(
    rows: list[dict], n: int = N_VALIDATION, seed: int = VALIDATION_SEED
) -> list[str]:
    pmids = sorted((r["pmid"] for r in rows if r["group"] == "preclinical"), key=_pmid_key)
    rng = np.random.default_rng(seed)
    k = min(n, len(pmids))
    idx = rng.choice(len(pmids), size=k, replace=False)
    return [pmids[i] for i in sorted(idx)]


def sample_validation_v2(
    rows: list[dict],
    exclude: Iterable[str],
    n: int = N_VALIDATION,
    seed: int = VALIDATION_SEED_V2,
) -> list[str]:
    """Eligible preclinical papers not in ``exclude`` (the v1 packets)."""
    skip = {str(p) for p in exclude}
    pool = [r for r in rows if r.get("eligible") and str(r["pmid"]) not in skip]
    return sample_validation(pool, n=n, seed=seed)


def _pmid_key(pmid: str):
    return (0, int(pmid)) if str(pmid).isdigit() else (1, str(pmid))


def make_packets(papers: list[dict], pmids: list[str]) -> list[dict]:
    """Blind packets: pmid, title, abstract, mesh only. No rule outputs."""
    by_pmid = {str(p["pmid"]): p for p in papers}
    return [
        {
            "pmid": pmid,
            "title": by_pmid[pmid].get("title") or "",
            "abstract": by_pmid[pmid].get("abstract") or "",
            "mesh": list(by_pmid[pmid].get("mesh") or []),
        }
        for pmid in pmids
    ]


EXTRACTION_SCHEMA = {
    "task": (
        "For each paper (title, abstract, MeSH headings), record which research "
        "systems the reported work used. Judge from what the paper did, not from "
        "what it cites or proposes. Return a JSON list with one object per paper."
    ),
    "output_item": {
        "pmid": "string, copied from the packet",
        "model_system": "one of: human_samples, animal, cell_only, other",
        "human_genetics": "0 or 1",
        "systems_present": "list, any subset of: human_samples, animal, cell",
    },
    "definitions": {
        "systems_present": {
            "human_samples": (
                "The work analyzed material or data taken from human patients or "
                "people: tumor tissue, biopsies, blood, surgical specimens, "
                "mutation testing of patient samples, or patient cohorts. Human "
                "cell lines do not count here."
            ),
            "animal": (
                "The work used living animals: mice, rats, xenografts or "
                "transplanted tumors in animals, transgenic or knockout animals, "
                "zebrafish, flies, worms, or other whole organisms. Cells taken "
                "from an animal species and grown in a dish do not count here."
            ),
            "cell": (
                "The work used cell lines or cells grown in culture (including "
                "organoids and transfected cells), human or animal."
            ),
        },
        "model_system": {
            "rule": "Pick the first that applies, in this order.",
            "human_samples": "Human samples present, and no animal and no cell work.",
            "animal": "Animal work present, with or without cells or human samples.",
            "cell_only": "Cell work present and no animal work (human samples may also be present).",
            "other": (
                "None of the above, for example purely computational, "
                "structural, biochemical, or cell-free work."
            ),
        },
        "human_genetics": (
            "1 if the paper reports or relies on inherited human genetic evidence: "
            "germline mutations, inherited predisposition or susceptibility, "
            "polymorphisms or SNPs, genome-wide association, familial or pedigree "
            "studies. Somatic tumor mutations alone are 0."
        ),
    },
}


# --- scoring ----------------------------------------------------------------------


def cohen_kappa(a: list, b: list, labels: Iterable | None = None) -> float:
    """Cohen's kappa for two raters over the same items.

    When expected agreement is 1 (both raters use one identical label), kappa
    is undefined; this returns 1.0 if observed agreement is also 1, else 0.0.
    """
    if len(a) != len(b):
        raise ValueError("rater lists differ in length")
    if not a:
        raise ValueError("no items")
    cats = list(labels) if labels is not None else sorted(set(a) | set(b), key=str)
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def kappa(rule_labels: list, model_labels: list, labels: Iterable | None = None) -> float:
    return cohen_kappa(rule_labels, model_labels, labels)


def confusion(rule: list, model: list, labels: Iterable) -> dict:
    """rows = rule label, columns = model label."""
    cats = [str(c) for c in labels]
    mat = {r: {m: 0 for m in cats} for r in cats}
    for x, y in zip(rule, model):
        mat[str(x)][str(y)] += 1
    return mat


def _load_extracted(obj) -> dict[str, dict]:
    if isinstance(obj, dict) and "items" in obj:
        obj = obj["items"]
    if isinstance(obj, dict):
        return {str(k): v for k, v in obj.items()}
    return {str(item["pmid"]): item for item in obj}


INVALID = "invalid"
_MODEL_SYSTEM_ALIASES = {
    "human": "human_samples",
    "humans": "human_samples",
    "human_sample": "human_samples",
    "animals": "animal",
    "animal_model": "animal",
    "in_vivo": "animal",
    "cell": "cell_only",
    "cells": "cell_only",
    "cellonly": "cell_only",
    "cell_line": "cell_only",
    "cell_lines": "cell_only",
}
_SYSTEM_ALIASES = {
    "human": "human_samples",
    "humans": "human_samples",
    "human_sample": "human_samples",
    "animals": "animal",
    "animal_model": "animal",
    "in_vivo": "animal",
    "cells": "cell",
    "cell_only": "cell",
    "cell_line": "cell",
    "cell_lines": "cell",
}


def _label_token(value) -> str:
    return re.sub(r"[-\s]+", "_", str(value).strip().casefold())


def norm_model_system(value) -> str:
    if value is None:
        return INVALID
    tok = _label_token(value)
    tok = _MODEL_SYSTEM_ALIASES.get(tok, tok)
    return tok if tok in MODEL_SYSTEMS else INVALID


def norm_binary(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    if isinstance(value, str):
        tok = value.strip().casefold()
        if tok in {"1", "yes", "true", "y"}:
            return 1
        if tok in {"0", "no", "false", "n"}:
            return 0
    return INVALID


def norm_systems(value) -> tuple[list[str] | None, int]:
    """(normalized systems or None if the field is unusable, n unknown entries)."""
    if isinstance(value, str):
        value = [p for p in re.split(r"[,;|]+", value) if p.strip()]
    if not isinstance(value, (list, tuple, set)):
        return None, 0
    out, unknown = set(), 0
    for entry in value:
        tok = _label_token(entry)
        tok = _SYSTEM_ALIASES.get(tok, tok)
        if tok in SYSTEMS:
            out.add(tok)
        else:
            unknown += 1
    return sorted(out), unknown


def _norm_model_item(item: dict | None) -> dict:
    """Normalize one extracted item. Unknown or missing labels become 'invalid'."""
    item = item or {}
    present, unknown = norm_systems(item.get("systems_present"))
    return {
        "model_system": norm_model_system(item.get("model_system")),
        "human_genetics": norm_binary(item.get("human_genetics")),
        "multi_system": INVALID if present is None else int(len(present) >= 2),
        "systems_present": present or [],
        "systems_invalid": present is None,
        "systems_unknown_entries": unknown,
    }


def score(packets: list[dict], extracted, seed: int = VALIDATION_SEED) -> dict:
    """Rule vs model agreement. Missing items and bad labels count as 'invalid'
    (a disagreement), so every packet is scored and nothing crashes."""
    items = _load_extracted(extracted)
    rule, model, pmids, missing = [], [], [], []
    for packet in packets:
        pmid = str(packet["pmid"])
        if pmid not in items:
            missing.append(pmid)
        rule.append(rule_labels(packet))
        model.append(_norm_model_item(items.get(pmid)))
        pmids.append(pmid)
    if len(missing) == len(packets):
        raise ValueError("no extracted items match the packets")

    specs = {
        "model_system": list(MODEL_SYSTEMS),
        "human_genetics": [0, 1],
        "multi_system": [0, 1],
    }
    features = {}
    for name, cats in specs.items():
        r = [x[name] for x in rule]
        m = [x[name] for x in model]
        k = kappa(r, m, [*cats, INVALID])
        features[name] = {
            "kappa": round(k, 4),
            "agreement": round(sum(x == y for x, y in zip(r, m)) / len(r), 4),
            "n_invalid": sum(1 for y in m if y == INVALID),
            "confusion_rule_rows_model_cols": confusion(r, m, [*cats, INVALID]),
            "gate_pass": bool(k >= KAPPA_GATE),
        }
    diagnostics = {}
    usable = [i for i, x in enumerate(model) if not x["systems_invalid"]]
    for system in SYSTEMS:
        r = [int(system in rule[i]["systems_present"]) for i in usable]
        m = [int(system in model[i]["systems_present"]) for i in usable]
        diagnostics[f"present_{system}"] = {
            "n": len(usable),
            "kappa": round(kappa(r, m, [0, 1]), 4) if usable else None,
            "confusion_rule_rows_model_cols": confusion(r, m, [0, 1]),
        }

    disagreements = []
    for pmid, r, m in zip(pmids, rule, model):
        diff = [k for k in specs if r[k] != m[k]]
        if diff:
            disagreements.append(
                {
                    "pmid": pmid,
                    "features": diff,
                    "rule": {k: r[k] for k in (*specs, "systems_present")},
                    "model": {k: m[k] for k in (*specs, "systems_present")},
                }
            )
    rng = np.random.default_rng(seed)
    if len(disagreements) > MAX_DISAGREEMENTS:
        idx = sorted(rng.choice(len(disagreements), size=MAX_DISAGREEMENTS, replace=False))
        for_owner = [disagreements[i] for i in idx]
    else:
        for_owner = disagreements

    return {
        "n_packets": len(packets),
        "n_scored": len(pmids),
        "missing_pmids": missing,
        "invalid": {
            "model_system": features["model_system"]["n_invalid"],
            "human_genetics": features["human_genetics"]["n_invalid"],
            "systems_present_unusable": sum(1 for x in model if x["systems_invalid"]),
            "systems_present_unknown_entries": sum(x["systems_unknown_entries"] for x in model),
        },
        "gate_threshold": KAPPA_GATE,
        "features": features,
        "dropped_features": [k for k, v in features.items() if not v["gate_pass"]],
        "diagnostics_presence": diagnostics,
        "n_disagreements": len(disagreements),
        "disagreements_for_owner": for_owner,
    }


# --- io -----------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
            n += 1
    return n


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
        fh.write("\n")


def main_build() -> dict:
    papers = read_jsonl(PAPERS_PATH)
    icite = read_jsonl(ICITE_PATH)
    rows = build_features(papers, icite)
    write_jsonl(FEATURES_PATH, rows)
    # v1 packets (PACKETS_PATH) are frozen and never rewritten; v2 draws eligible
    # preclinical papers that were not in v1.
    old = []
    if PACKETS_PATH.exists():
        with open(PACKETS_PATH) as fh:
            old = [str(p["pmid"]) for p in json.load(fh)]
    pmids = sample_validation_v2(rows, exclude=old)
    write_json(PACKETS_V2_PATH, make_packets(papers, pmids))
    if not EXTRACT_SCHEMA_PATH.exists():
        write_json(EXTRACT_SCHEMA_PATH, EXTRACTION_SCHEMA)
    counts = marginals(rows)
    counts["n_papers_in"] = len(papers)
    counts["n_no_mesh_all_papers"] = sum(1 for p in papers if not p.get("mesh"))
    counts["n_validation_packets_v2"] = len(pmids)
    counts["n_v1_packets_excluded"] = len(old)
    print(json.dumps(counts, indent=2))
    return counts


def pathway_layout(name: str) -> dict:
    from src.meta import pathways

    base = pathways.pathway_dir(name)
    return {"papers": base / "papers.jsonl", "icite": base / "icite.jsonl",
            "features": base / "features.jsonl", "manifest": base / "manifest_features.json"}


def main_build_pathway(name: str) -> dict:
    """v2: features for one pathway corpus (no packets; see pathway-packets)."""
    lay = pathway_layout(name)
    papers = read_jsonl(lay["papers"])
    icite = read_jsonl(lay["icite"])
    rows = build_features(papers, icite, pathway=name)
    write_jsonl(lay["features"], rows)
    counts = marginals(rows)
    counts["pathway"] = name
    counts["n_papers_in"] = len(papers)
    counts["n_no_mesh_all_papers"] = sum(1 for p in papers if not p.get("mesh"))
    write_json(lay["manifest"], counts)
    print(json.dumps(counts, indent=2))
    return counts


def sample_pathway_packets(
    rows_by_pathway: dict[str, list[dict]],
    n: int = N_VALIDATION_PER_PATHWAY,
    seed: int = VALIDATION_SEED_PATHWAYS,
) -> dict[str, list[str]]:
    """Per pathway (in the given order): ``n`` eligible preclinical PMIDs drawn
    with ``np.random.default_rng(seed)`` over the sorted PMIDs. A PMID already
    drawn for an earlier pathway is removed from later pools, so the combined
    packet file has no duplicates."""
    taken: set[str] = set()
    out: dict[str, list[str]] = {}
    for name, rows in rows_by_pathway.items():
        pool = [r for r in rows if r.get("eligible") and str(r["pmid"]) not in taken]
        out[name] = sample_validation(pool, n=n, seed=seed)
        taken.update(out[name])
    return out


def main_pathway_packets(names: Iterable[str] | None = None) -> dict:
    """v2 gate packets: 150 eligible preclinical papers per pathway, blind
    (pmid/title/abstract/mesh), plus a separate pmid -> pathway index."""
    from src.meta import pathways

    names = list(names or pathways.NAMES)
    rows_by = {}
    papers_all: list[dict] = []
    for name in names:
        lay = pathway_layout(name)
        rows_by[name] = read_jsonl(lay["features"])
        papers_all.extend(read_jsonl(lay["papers"]))
    drawn = sample_pathway_packets(rows_by)
    packets, index = [], {}
    for name in names:
        packets.extend(make_packets(papers_all, drawn[name]))
        index.update({pmid: name for pmid in drawn[name]})
    write_json(PACKETS_PATHWAYS_PATH, packets)
    write_json(PACKETS_PATHWAYS_INDEX_PATH, index)
    counts = {
        "n_packets": len(packets),
        "per_pathway": {name: len(drawn[name]) for name in names},
        "unique_pmids": len({p["pmid"] for p in packets}),
        "seed": VALIDATION_SEED_PATHWAYS,
    }
    print(json.dumps(counts, indent=2))
    return counts


def main_score(extracted_path: str, packets_path: Path | None = None, out_path: Path | None = None) -> dict:
    packets_path = Path(packets_path) if packets_path else PACKETS_PATH
    out_path = Path(out_path) if out_path else GATE_PATH
    with open(packets_path) as fh:
        packets = json.load(fh)
    with open(extracted_path) as fh:
        extracted = json.load(fh)
    result = score(packets, extracted)
    result["extracted_file"] = str(extracted_path)
    result["packets_file"] = str(packets_path)
    write_json(out_path, result)
    print(
        json.dumps(
            {k: {"kappa": v["kappa"], "n_invalid": v["n_invalid"], "gate_pass": v["gate_pass"]}
             for k, v in result["features"].items()},
            indent=2,
        )
    )
    return result


USAGE = (
    "usage: python -m src.meta.features\n"
    "       python -m src.meta.features --pathway {egfr,pi3k}\n"
    "       python -m src.meta.features pathway-packets\n"
    "       python -m src.meta.features score <extracted.json> [--packets PATH] [--out PATH]"
)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv:
        main_build()
        return
    if argv[0] == "--pathway" or argv[0].startswith("--pathway="):
        name = argv[1] if argv[0] == "--pathway" and len(argv) > 1 else argv[0].partition("=")[2]
        from src.meta import pathways

        if name not in pathways.NAMES:
            raise SystemExit(USAGE)
        main_build_pathway(name)
        return
    if argv[0] == "pathway-packets":
        main_pathway_packets()
        return
    if argv[0] != "score":
        raise SystemExit(USAGE)
    import argparse

    parser = argparse.ArgumentParser(prog="python -m src.meta.features score", usage=USAGE)
    parser.add_argument("extracted")
    parser.add_argument("--packets", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv[1:])
    main_score(args.extracted, args.packets, args.out)


if __name__ == "__main__":
    main()
