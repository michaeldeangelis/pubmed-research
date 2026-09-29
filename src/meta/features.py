"""Metascience v1 features F1-F3, covariates, and the measurement gate.

Contract: ``docs/specs/2026-09-29-metascience-design.md``. The analysis plan is
``experiments.md`` (entry "2026-09-29 metascience v1").

Commands
--------
``python -m src.meta.features``
    Reads ``data/meta/papers.jsonl`` and ``data/meta/icite.jsonl``, writes
    ``data/meta/features.jsonl`` (one row per iCite research article), writes
    the blind validation packets to ``bench/meta_validation/``, and prints
    marginal counts only.
``python -m src.meta.features score <extracted.json>``
    Compares the rules with a blind model extraction of the packets and writes
    ``results/meta_measurement_gate.json`` (Cohen's kappa per feature,
    confusion matrices, gate pass at kappa >= 0.60).

Blinding: this module never reads outcome files (``outcome_b.jsonl``,
``impact.jsonl``) and never relates a feature to an outcome.

Rules
-----
All MeSH matching is on normalized descriptor names: qualifiers after "/" are
dropped, a leading or trailing "*" (major topic) is dropped, whitespace is
collapsed, and comparison is case-insensitive.

Presence of each system (used by F3, and by F1 through precedence):

* human samples: MeSH ``Humans`` AND at least one term in
  ``HUMAN_SAMPLE_TERMS`` (patient tissue, mutation analysis of specimens,
  patient age groups, patient study designs, patient outcome analyses), or a
  term in ``HUMAN_SEX_TERMS`` when no animal is present.
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

gene_group: first match in the order KRAS, BRAF, NRAS/HRAS over the title and
abstract (``GENE_GROUP_PATTERNS``); ``other`` if none matches.
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
PACKETS_PATH = VALIDATION_DIR / "packets.json"
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
)
N_VALIDATION = 300
VALIDATION_SEED = 0
KAPPA_GATE = 0.60
MAX_DISAGREEMENTS = 40

# --- term lists (frozen by the commit that precedes the analysis run) --------

HUMAN_TERM = "Humans"
# Sex check tags mark human subjects only when no animal is present (MeSH also
# assigns them to animals, so a xenograft paper tagged Humans + Female would
# otherwise gain a spurious human-samples system).
HUMAN_SEX_TERMS = ("Female", "Male")

# Terms that, with Humans, indicate material or data from patients.
HUMAN_SAMPLE_TERMS = (
    # mutation analysis of specimens
    "DNA Mutational Analysis",
    "Genotyping Techniques",
    "Sequence Analysis, DNA",
    "Microsatellite Instability",
    "Loss of Heterozygosity",
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
    "Immunohistochemistry",
    "Neoplasm Staging",
    "Neoplasm Grading",
    "Tissue Fixation",
    "Tissue Embedding",
    "Patient Selection",
    "Patient Outcome Assessment",
    # patient age groups (MeSH assigns these to human subjects)
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
    # patient study designs and outcome analyses
    "Retrospective Studies",
    "Prospective Studies",
    "Cohort Studies",
    "Case-Control Studies",
    "Cross-Sectional Studies",
    "Follow-Up Studies",
    "Longitudinal Studies",
    "Prognosis",
    "Survival Analysis",
    "Survival Rate",
    "Kaplan-Meier Estimate",
    "Proportional Hazards Models",
    "Disease-Free Survival",
    "Progression-Free Survival",
    "Treatment Outcome",
    "Neoplasm Recurrence, Local",
    "Pedigree",
    "Inpatients",
    "Outpatients",
)

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

# First match wins, in this order.
GENE_GROUP_PATTERNS = (
    ("KRAS", re.compile(r"\bK-?i?-?ras\d?(?=\b|[A-Z]\d)|\bKi-ras", re.I)),
    ("BRAF", re.compile(r"\bB-?raf(?=\b|[A-Z]\d)", re.I)),
    ("NRAS/HRAS", re.compile(r"\b(N-?ras|H-?ras|Ha-?ras|c-Ha-ras|c-H-ras)(?=\b|[A-Z]\d)", re.I)),
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
_HUMAN_SAMPLE = _norm_set(HUMAN_SAMPLE_TERMS)
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
        bool(mesh & _HUMAN_SAMPLE) or (not animal and bool(mesh & _HUMAN_SEX))
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


def gene_group(paper: dict) -> str:
    text = paper_text(paper)
    for name, pattern in GENE_GROUP_PATTERNS:
        if pattern.search(text):
            return name
    return "other"


def rule_labels(paper: dict) -> dict:
    """Rule outputs for one paper, in the extractor's label space."""
    present = systems_present(paper)
    return {
        "model_system": model_system(present),
        "human_genetics": human_genetics(paper),
        "multi_system": multi_system(present),
        "systems_present": sorted(s for s in SYSTEMS if present[s]),
    }


def feature_row(paper: dict, icite: dict) -> dict:
    labels = rule_labels(paper)
    n_authors = paper.get("n_authors")
    return {
        "pmid": str(paper["pmid"]),
        "group": "clinical" if truthy(icite.get("is_clinical")) else "preclinical",
        "model_system": labels["model_system"],
        "human_genetics": labels["human_genetics"],
        "multi_system": labels["multi_system"],
        "gene_group": gene_group(paper),
        "n_authors": int(n_authors) if n_authors is not None else None,
        "n_refs": None,  # supplied by the impact builder (OpenAlex)
    }


def build_features(papers: list[dict], icite: list[dict]) -> list[dict]:
    """One row per paper that iCite marks as a research article."""
    by_pmid = {str(r.get("pmid")): r for r in icite}
    rows = []
    for paper in papers:
        rec = by_pmid.get(str(paper.get("pmid")))
        if rec is None or not truthy(rec.get("is_research_article")):
            continue
        rows.append(feature_row(paper, rec))
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


def _norm_model_item(item: dict) -> dict:
    ms = str(item.get("model_system", "")).strip().lower()
    hg = int(truthy(item.get("human_genetics")))
    present = item.get("systems_present") or []
    if isinstance(present, str):
        present = [p for p in re.split(r"[,\s]+", present) if p]
    present = sorted({str(p).strip().lower() for p in present} & set(SYSTEMS))
    return {
        "model_system": ms,
        "human_genetics": hg,
        "multi_system": int(len(present) >= 2),
        "systems_present": present,
    }


def score(packets: list[dict], extracted, seed: int = VALIDATION_SEED) -> dict:
    items = _load_extracted(extracted)
    rule, model, pmids = [], [], []
    missing = []
    for packet in packets:
        pmid = str(packet["pmid"])
        if pmid not in items:
            missing.append(pmid)
            continue
        rule.append(rule_labels(packet))
        model.append(_norm_model_item(items[pmid]))
        pmids.append(pmid)
    if not pmids:
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
        k = kappa(r, m, cats)
        features[name] = {
            "kappa": round(k, 4),
            "agreement": round(sum(x == y for x, y in zip(r, m)) / len(r), 4),
            "confusion_rule_rows_model_cols": confusion(r, m, cats),
            "gate_pass": bool(k >= KAPPA_GATE),
        }
    diagnostics = {}
    for system in SYSTEMS:
        r = [int(system in x["systems_present"]) for x in rule]
        m = [int(system in x["systems_present"]) for x in model]
        diagnostics[f"present_{system}"] = {
            "kappa": round(kappa(r, m, [0, 1]), 4),
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
    pmids = sample_validation(rows)
    write_json(PACKETS_PATH, make_packets(papers, pmids))
    write_json(EXTRACT_SCHEMA_PATH, EXTRACTION_SCHEMA)
    counts = marginals(rows)
    counts["n_papers_in"] = len(papers)
    counts["n_no_mesh"] = sum(1 for p in papers if not p.get("mesh"))
    counts["n_validation_packets"] = len(pmids)
    print(json.dumps(counts, indent=2))
    return counts


def main_score(extracted_path: str) -> dict:
    with open(PACKETS_PATH) as fh:
        packets = json.load(fh)
    with open(extracted_path) as fh:
        extracted = json.load(fh)
    result = score(packets, extracted)
    result["extracted_file"] = str(extracted_path)
    write_json(GATE_PATH, result)
    print(
        json.dumps(
            {k: {"kappa": v["kappa"], "gate_pass": v["gate_pass"]} for k, v in result["features"].items()},
            indent=2,
        )
    )
    return result


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "score":
        if len(argv) != 2:
            raise SystemExit("usage: python -m src.meta.features score <extracted.json>")
        main_score(argv[1])
    elif not argv:
        main_build()
    else:
        raise SystemExit("usage: python -m src.meta.features [score <extracted.json>]")


if __name__ == "__main__":
    main()
