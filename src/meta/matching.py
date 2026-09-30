"""Metascience v4: topic matching of patient-sample and cell-line papers.

See experiments.md, "2026-09-30 metascience v4: design against topic".

python -m src.meta.matching

Outcome-blind. Reads papers.jsonl (MeSH, year, title, abstract), features.jsonl
(group, eligibility, model_system, gene_group, n_authors) and, for the balance
table only, the reference counts in impact.jsonl. It never opens an outcome
file.

Steps
-----
1. Pool: eligible preclinical papers of RAS, EGFR and PI3K, deduplicated
   keeping the first of ras, egfr, pi3k (as in v3); treated = human_samples,
   controls = cell_only.
2. Topic: MeSH descriptors with qualifiers dropped, minus every
   design-revealing descriptor (``removed_descriptors``; the list present in
   the corpus is written to removed_mesh.json before matching). TF-IDF with
   one token per descriptor, L2-normalized; cosine similarity. A paper left
   with no descriptor is unmatchable.
3. Matching: exact on pathway and gene_group, |year difference| <= 1, 1:1
   greedy without replacement. At each step the treated paper with the
   highest best available similarity is matched to that control (ties broken
   by treated PMID, then control PMID); this is greedy over all candidate
   edges sorted by descending similarity. Because greedy decisions above a
   caliper never depend on edges below it, one pass at the lowest caliper
   gives the matching for every caliper. The caliper is the largest of
   CALIPERS matching at least MIN_MATCH_RATE of all treated papers.
4. Outputs: data/meta/matching/pairs.jsonl, bench/meta_validation/
   matching_summary.json, and 50 blind judge packets plus their key.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import sparse

from src.meta import features as F

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "data" / "meta"
PATHWAY_DIRS = {"ras": META, "egfr": META / "pathways" / "egfr", "pi3k": META / "pathways" / "pi3k"}
OUT_DIR = META / "matching"
BENCH = ROOT / "bench" / "meta_validation"
TREATED, CONTROL = "human_samples", "cell_only"
CALIPERS = (0.6, 0.5, 0.4, 0.3)
MIN_MATCH_RATE = 0.5
YEAR_WINDOW = 1
N_JUDGE = 50
JUDGE_SEED = 0
DIAGNOSTIC_FLOOR = 0.1
DIAGNOSTIC_THRESHOLDS = (0.2, 0.1)

# --- removed (design-revealing) descriptors ---------------------------------------

CHECK_TAGS = (
    "Humans", "Animals", "Female", "Male", "Pregnancy",
    *F.HUMAN_AGE_TERMS, "Infant, Premature", "Aged, 80 and over", "Frail Elderly",
)
ORGANISMS = (
    "Mice", "Rats", "Cricetinae", "Mesocricetus", "Cricetulus", "Dogs", "Cats", "Swine", "Rabbits",
    "Cattle", "Sheep", "Horses", "Guinea Pigs", "Chickens", "Xenopus", "Xenopus laevis", "Primates",
    "Macaca mulatta", "Macaca fascicularis", "Chlorocebus aethiops", "Cercopithecus aethiops",
    "Spodoptera", "Zebrafish", "Drosophila", "Drosophila melanogaster", "Caenorhabditis elegans",
    "Chick Embryo", "Oryzias", "Mice, Inbred Strains", "Rats, Inbred Strains",
    "Disease Models, Animal", "Models, Animal", "Xenograft Model Antitumor Assays",
    "Neoplasm Transplantation", "Transplantation, Heterologous", "Heterografts",
    "Animals, Genetically Modified", "Animals, Newborn", "Tumor Burden",
)
CELL_SYSTEMS = (
    "Cell Line", "Cell Line, Tumor", "Cell Line, Transformed", "Cells, Cultured",
    "Tumor Cells, Cultured", "Clone Cells", "Hybridomas", "Hybrid Cells", "Feeder Cells",
    "Artificial Cells", "Cell Culture Techniques", "Primary Cell Culture", "Coculture Techniques",
    "Cell Culture Techniques, Three Dimensional", "Organoids", "Spheroids, Cellular",
    "Tumor Stem Cell Assay", "Tissue Culture Techniques", "Culture Media", "Culture Media, Conditioned",
    "Culture Media, Serum-Free", "In Vitro Techniques", "Organ Culture Techniques",
    "Human Umbilical Vein Endothelial Cells", "Cell-Free System",
    # named cell lines
    "3T3 Cells", "3T3-L1 Cells", "BALB 3T3 Cells", "NIH 3T3 Cells", "Swiss 3T3 Cells", "A549 Cells",
    "Caco-2 Cells", "CHO Cells", "COS Cells", "HCT116 Cells", "HEK293 Cells", "HL-60 Cells",
    "HT29 Cells", "HaCaT Cells", "HeLa Cells", "Hep G2 Cells", "Jurkat Cells", "K562 Cells",
    "KB Cells", "L Cells", "MCF-7 Cells", "PC-3 Cells", "PC12 Cells", "Sf9 Cells", "THP-1 Cells",
    "U937 Cells", "Vero Cells", "Madin Darby Canine Kidney Cells", "LLC-PK1 Cells", "HT-29 Cells",
    "MCF-10A Cells", "RAW 264.7 Cells", "HEK293T Cells", "Sf21 Cells",
)
STUDY_DESIGN = (
    *F.HUMAN_SPECIFIC_TERMS, *F.HUMAN_GENERIC_TERMS,
    "Survival Analysis", "Kaplan-Meier Estimate", "Prognosis", "Disease-Free Survival",
    "Progression-Free Survival", "Treatment Outcome", "Treatment Failure", "Follow-Up Studies",
    "Multivariate Analysis", "Proportional Hazards Models", "Risk Factors", "Risk Assessment",
    "Neoplasm Staging", "Neoplasm Grading", "Biopsy", "Paraffin Embedding", "Tissue Array Analysis",
    "Immunohistochemistry", "Retrospective Studies", "Prospective Studies", "Cohort Studies",
    "Case-Control Studies", "Cross-Sectional Studies", "Longitudinal Studies", "Survival Rate",
    "Recurrence", "Disease Progression", "Fatal Outcome", "Neoplasm, Residual", "Remission Induction",
    "Predictive Value of Tests", "Sensitivity and Specificity", "Reproducibility of Results",
    "ROC Curve", "Area Under Curve", "Odds Ratio", "Logistic Models", "Regression Analysis",
    "Linear Models", "Models, Statistical", "Chi-Square Distribution", "Analysis of Variance",
    "Statistics as Topic", "Statistics, Nonparametric", "Data Interpretation, Statistical",
    "Cluster Analysis", "Confidence Intervals", "Likelihood Functions", "Probability",
    "Incidence", "Prevalence", "Age Factors", "Sex Factors", "Age Distribution", "Sex Distribution",
    "Age of Onset", "Time Factors", "Registries", "Reference Values", "Observer Variation",
    "Epidemiologic Methods", "Feasibility Studies", "Pilot Projects", "Genetic Association Studies",
    "Genome-Wide Association Study", "Genetic Testing", "Mass Screening", "Early Detection of Cancer",
    "Diagnosis, Differential", "False Negative Reactions", "False Positive Reactions",
    "Limit of Detection", "Quality Control", "Tissue Banks", "Specimen Handling", "Autopsy",
    "Surveys and Questionnaires", "Research Design", "Evaluation Studies as Topic",
    "Validation Studies as Topic", "Patient Selection", "Dose-Response Relationship, Drug",
    "Tomography, X-Ray Computed", "Magnetic Resonance Imaging", "Positron-Emission Tomography",
    "Positron Emission Tomography Computed Tomography", "Fluorodeoxyglucose F18", "Ultrasonography",
    "Radiography", "Endoscopy", "Cytodiagnosis", "Cytological Techniques", "Staining and Labeling",
    "Asian People", "Asian Continental Ancestry Group", "European Continental Ancestry Group",
    "White People", "Black People", "African Americans", "Black or African American", "White",
    "Hispanic Americans", "Hispanic or Latino", "Ethnic Groups", "Ethnicity", "Racial Groups",
    "Continental Population Groups", "East Asian People",
    "Multicenter Studies as Topic", "Survivors", "Outcome Assessment, Health Care", "SEER Program",
    "Population Surveillance", "Early Diagnosis", "Diagnostic Errors", "Decision Support Techniques",
    "Socioeconomic Factors", "Cost-Benefit Analysis", "Kidney Function Tests",
    "Drug Evaluation, Preclinical", "Drug Evaluation", "Fixatives", "Immunoassay", "Isotope Labeling",
    "Reference Standards", "Transgenes", "Endoscopic Ultrasound-Guided Fine Needle Aspiration",
    "Reoperation", "Luminescent Measurements", "Molecular Probes", "Medical Records", "Image Enhancement",
    "Molecular Epidemiology", "Hospitalization", "Materials Testing", "Electroporation",
    "Genetics, Population", "Genetic Carrier Screening", "Sample Size", "Confounding Factors, Epidemiologic",
    "Pregnancy Outcome", "Diagnosis, Computer-Assisted", "Calibration", "Image-Guided Biopsy",
    "Patient Compliance", "National Health Programs", "Prenatal Diagnosis", "Support Vector Machine",
    "Laser Scanning Cytometry", "Patient Safety", "Escherichia coli", "Saccharomyces cerevisiae",
    "Baculoviridae", "Lentivirus", "Dependovirus", "Immunomagnetic Separation", "Machine Learning",
    "Neural Networks, Computer", "Databases, Genetic", "Databases, Factual", "Databases, Protein",
    "Internet", "Treatment Failure", "Neoplasm Recurrence, Local",
    "Sentinel Lymph Node Biopsy", "Automation, Laboratory", "Biological Specimen Banks", "Image Cytometry",
    "Cryoelectron Microscopy", "Diagnostic Tests, Routine", "Radiographic Image Interpretation, Computer-Assisted",
    "Maximum Tolerated Dose", "Off-Label Use",
)
LAB_TECHNIQUES = (
    "Blotting, Western", "Blotting, Northern", "Blotting, Southern", "Immunoblotting",
    "Flow Cytometry", "Immunoprecipitation", "Chromatin Immunoprecipitation",
    "Enzyme-Linked Immunosorbent Assay", "Immunoenzyme Techniques", "Fluorescent Antibody Technique",
    "Fluorescent Antibody Technique, Indirect", "Fluorescent Antibody Technique, Direct",
    "Radioimmunoassay", "Electrophoretic Mobility Shift Assay", "Electrophoresis, Polyacrylamide Gel",
    "Electrophoresis, Gel, Two-Dimensional", "Electrophoresis", "Two-Hybrid System Techniques",
    "RNA Interference", "RNA, Small Interfering", "Gene Knockdown Techniques", "Gene Knockout Techniques", "CRISPR-Cas Systems",
    "Gene Transfer Techniques", "Transfection", "Transduction, Genetic", "Genetic Vectors",
    "Plasmids", "Cloning, Molecular", "Mutagenesis, Site-Directed", "Genes, Reporter",
    "Luciferases", "Luciferases, Firefly", "Luciferases, Renilla", "Green Fluorescent Proteins",
    "Luminescent Proteins", "Recombinant Proteins", "Recombinant Fusion Proteins", "Polymerase Chain Reaction", "Reverse Transcriptase Polymerase Chain Reaction",
    "Real-Time Polymerase Chain Reaction", "DNA Primers", "Sequence Analysis, DNA",
    "Sequence Analysis, RNA", "Sequence Analysis, Protein", "DNA Mutational Analysis",
    "High-Throughput Nucleotide Sequencing", "Whole Exome Sequencing", "Exome Sequencing",
    "Whole Genome Sequencing", "Oligonucleotide Array Sequence Analysis", "Microarray Analysis",
    "Protein Array Analysis", "Tissue Array Analysis", "Gene Expression Profiling",
    "In Situ Hybridization", "In Situ Hybridization, Fluorescence", "Comparative Genomic Hybridization",
    "In Situ Nick-End Labeling", "Polymorphism, Single-Stranded Conformational",
    "Polymorphism, Restriction Fragment Length", "Genotyping Techniques",
    "Molecular Sequence Data", "Base Sequence", "Amino Acid Sequence", "Sequence Alignment",
    "Sequence Homology, Amino Acid", "Molecular Diagnostic Techniques",
    "Microscopy, Fluorescence", "Microscopy, Confocal", "Microscopy, Electron",
    "Microscopy, Electron, Transmission", "Microscopy, Electron, Scanning", "Microscopy, Immunoelectron",
    "Microscopy, Video", "Microscopy, Phase-Contrast", "Microscopy", "Time-Lapse Imaging",
    "Image Processing, Computer-Assisted", "Chromatography, Liquid", "Chromatography, High Pressure Liquid",
    "Tandem Mass Spectrometry", "Mass Spectrometry", "Spectrometry, Mass, Matrix-Assisted Laser Desorption-Ionization",
    "Spectrometry, Mass, Electrospray Ionization", "Spectrometry, Fluorescence",
    "Drug Screening Assays, Antitumor", "Inhibitory Concentration 50", "High-Throughput Screening Assays",
    "Cell Proliferation", "Cell Survival", "Cell Count", "Colony-Forming Units Assay",
    "Cell Separation", "Tetrazolium Salts", "Bromodeoxyuridine",
    "Models, Molecular", "Models, Biological", "Models, Genetic", "Computer Simulation",
    "Algorithms", "Software", "Computational Biology", "Molecular Docking Simulation",
    "Molecular Dynamics Simulation", "Precipitin Tests", "Microfluidic Analytical Techniques",
    "Image Interpretation, Computer-Assisted", "Comet Assay", "Spectrum Analysis, Raman",
    "Patch-Clamp Techniques", "Multimodal Imaging", "Reagent Kits, Diagnostic", "Immunohistochemistry", "Histocytochemistry", "Immunophenotyping", "Laser Capture Microdissection",
    "Microdissection", "Tissue Fixation", "Formaldehyde", "Tissue Embedding", "Frozen Sections",
    "Real-Time Polymerase Chain Reaction", "Nucleic Acid Amplification Techniques",
)
REMOVED_PREFIXES = (
    "mice, ", "rats, ", "cell line,", "polymerase chain reaction", "in situ hybridization",
    "blotting, ", "microscopy", "clinical trials", "clinical trial", "randomized controlled trial",
    "controlled clinical trial", "statistics", "biopsy", "tomography", "sequence analysis",
    "spectrometry, mass", "chromatography", "electrophoresis", "culture media", "survival",
    "fluorescent antibody technique", "microarray", "models, ", "hospitals", "costs", "patient", "postoperative", "preoperative",
    "intraoperative", "laborator", "clinical protocols", "pathology, clinical", "imaging, ",
)
# Method, test, imaging, survey and "as topic" descriptors by suffix.
METHOD_SUFFIX = re.compile(
    r" (techniques?|tests?|assays?|analysis|methods?|count|counts|studies|trials|staining|imaging|"
    r"surveys?|probes|allocation)( as topic)?$|as topic$"
)
# Named cell lines not in the explicit lists: "<letters/digits> Cells" with a digit or dash.
NAMED_CELL_LINE = re.compile(r"^(?!th\d)[a-z0-9 .-]*\d[a-z0-9 .-]* cells$")

# Geographic descriptors mark human populations (MeSH Z01). Only those that occur.
GEOGRAPHIC = (
    "China", "Japan", "Taiwan", "Korea", "Republic of Korea", "Hong Kong", "India", "Singapore",
    "Thailand", "Viet Nam", "Malaysia", "Iran", "Turkey", "Israel", "Saudi Arabia", "Egypt", "Tunisia",
    "Morocco", "Pakistan", "United States", "Canada", "Mexico", "Brazil", "Argentina", "Chile",
    "Colombia", "Peru", "Europe", "Germany", "France", "Italy", "Spain", "Portugal", "Greece",
    "United Kingdom", "England", "Scotland", "Ireland", "Netherlands", "Belgium", "Switzerland",
    "Austria", "Sweden", "Norway", "Denmark", "Finland", "Poland", "Czech Republic", "Hungary",
    "Russia", "Australia", "New Zealand", "South Africa", "Nigeria", "Asia", "Africa",
)

CATEGORIES = {
    "check_tags": CHECK_TAGS,
    "organisms_and_in_vivo": ORGANISMS + F.ANIMAL_STRONG_TERMS + F.ANIMAL_WEAK_TERMS,
    "cell_systems": CELL_SYSTEMS + F.CELL_TERMS,
    "study_design_statistics_clinical": STUDY_DESIGN + F.HUMAN_AGE_TERMS + F.HUMAN_SEX_TERMS,
    "lab_techniques": LAB_TECHNIQUES,
    "geography_population": GEOGRAPHIC,
}
_EXPLICIT = frozenset(F.norm_term(t) for terms in CATEGORIES.values() for t in terms) | {F.norm_term(F.HUMAN_TERM)}


def is_removed(term: str) -> bool:
    """True if a normalized descriptor reveals the experimental system or design."""
    t = F.norm_term(term)
    return (t in _EXPLICIT or t.startswith(REMOVED_PREFIXES) or bool(NAMED_CELL_LINE.match(t))
            or bool(METHOD_SUFFIX.search(t)))


def removed_descriptors(vocab: Iterable[str]) -> list[str]:
    return sorted({F.norm_term(t) for t in vocab if is_removed(t)})


def topic_terms(mesh: Iterable[str] | None) -> list[str]:
    """Normalized, de-duplicated descriptors left after removing design terms."""
    return sorted({t for t in (F.norm_term(m) for m in (mesh or [])) if t and not is_removed(t)})


# --- pool -----------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _ref_counts(path: Path, pathway: str) -> dict[str, int | None]:
    """Reference counts only; every other impact field is discarded unread."""
    if not path.is_file():
        return {}
    out = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            extra = {k: rec.get(k) for k in ("n_refs_openalex", "n_refs_icite")}
            if pathway == "ras":
                value = next((extra[k] for k in ("n_refs_openalex", "n_refs_icite") if extra[k]), None)
            else:
                value = extra["n_refs_icite"] or None
            out[str(rec["pmid"])] = int(value) if value else None
    return out


def load_pool(pathway_dirs: dict[str, Path] = PATHWAY_DIRS) -> list[dict]:
    """Eligible preclinical treated + control papers, deduplicated in pathway order."""
    rows, seen = [], set()
    for name, meta_dir in pathway_dirs.items():
        papers = {str(r["pmid"]): r for r in _read_jsonl(meta_dir / "papers.jsonl")}
        refs = _ref_counts(meta_dir / "impact.jsonl", name)
        for f in _read_jsonl(meta_dir / "features.jsonl"):
            pmid = str(f["pmid"])
            if f.get("group") != "preclinical" or not f.get("eligible", True):
                continue
            if pmid in seen or pmid not in papers:
                continue
            seen.add(pmid)
            if f.get("model_system") not in (TREATED, CONTROL):
                continue
            p = papers[pmid]
            rows.append({
                "pmid": pmid, "pathway": name, "gene_group": f"{name}:{f['gene_group']}",
                "year": int(p["year"]), "treated": int(f["model_system"] == TREATED),
                "n_authors": f.get("n_authors", p.get("n_authors")), "n_refs": refs.get(pmid),
                "raw_mesh": [F.norm_term(m) for m in (p.get("mesh") or [])],
                "terms": topic_terms(p.get("mesh")), "title": p.get("title") or "",
                "abstract": p.get("abstract") or "",
            })
    return rows


# --- topic vectors and matching -------------------------------------------------------


def tfidf(rows: list[dict]) -> sparse.csr_matrix:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(analyzer=lambda terms: terms, lowercase=False, norm="l2")
    docs = [r["terms"] for r in rows]
    if not any(docs):
        return sparse.csr_matrix((len(rows), 0))
    return vec.fit_transform(docs).tocsr()


def candidate_edges(rows: list[dict], x: sparse.csr_matrix, floor: float) -> tuple[np.ndarray, ...]:
    """(treated index, control index, similarity) for every allowed pair with similarity >= floor."""
    blocks: dict[tuple, dict[int, list[int]]] = {}
    for i, r in enumerate(rows):
        if not r["terms"]:
            continue
        blocks.setdefault((r["pathway"], r["gene_group"], r["treated"]), {}).setdefault(r["year"], []).append(i)
    ti, ci, sims = [], [], []
    for (pathway, gene, arm), by_year in blocks.items():
        if arm != 1:
            continue
        controls = blocks.get((pathway, gene, 0), {})
        for year, treated in by_year.items():
            cands = [j for y in range(year - YEAR_WINDOW, year + YEAR_WINDOW + 1) for j in controls.get(y, [])]
            if not cands:
                continue
            s = (x[treated] @ x[cands].T).toarray()
            a, b = np.nonzero(s >= floor)
            ti.append(np.asarray(treated)[a])
            ci.append(np.asarray(cands)[b])
            sims.append(s[a, b])
    if not ti:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    return np.concatenate(ti), np.concatenate(ci), np.concatenate(sims)


def greedy(rows: list[dict], ti: np.ndarray, ci: np.ndarray, sims: np.ndarray) -> list[tuple[int, int, float]]:
    """1:1 greedy without replacement over edges by (-similarity, treated PMID, control PMID)."""
    pmids = np.array([int(r["pmid"]) for r in rows], dtype=np.int64)
    tp, cp = pmids[ti], pmids[ci]
    order = np.lexsort((cp, tp, -sims))
    used_t, used_c, pairs = set(), set(), []
    for k in order:
        t, c = int(ti[k]), int(ci[k])
        if t in used_t or c in used_c:
            continue
        used_t.add(t)
        used_c.add(c)
        pairs.append((t, c, float(sims[k])))
    return pairs


def max_feasible(rows: list[dict]) -> int:
    """Maximum 1:1 pairs under exact pathway, gene_group and |year diff| <= window (bipartite matching)."""
    from scipy.sparse.csgraph import maximum_bipartite_matching

    total = 0
    blocks: dict[tuple, list[dict]] = {}
    for r in rows:
        if r["terms"]:
            blocks.setdefault((r["pathway"], r["gene_group"]), []).append(r)
    for sub in blocks.values():
        t = [r["year"] for r in sub if r["treated"]]
        c = [r["year"] for r in sub if not r["treated"]]
        if not t or not c:
            continue
        adj = sparse.csr_matrix(np.abs(np.subtract.outer(np.array(t), np.array(c))) <= YEAR_WINDOW, dtype=np.int8)
        total += int((maximum_bipartite_matching(adj, perm_type="column") >= 0).sum())
    return total


def choose_caliper(rates: dict[float, float]) -> tuple[float, bool]:
    """Largest caliper whose match rate >= MIN_MATCH_RATE; else the smallest, flagged."""
    for c in sorted(rates, reverse=True):
        if rates[c] >= MIN_MATCH_RATE:
            return c, True
    return min(rates), False


def smd(a: list[float], b: list[float]) -> float | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return None
    pooled = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return None if pooled == 0 else round(float((a.mean() - b.mean()) / pooled), 4)


def balance(treated: list[dict], controls: list[dict]) -> dict:
    out = {}
    for key, fn in (("log_authors", lambda r: math.log(r["n_authors"]) if r["n_authors"] else None),
                    ("log1p_refs", lambda r: math.log1p(r["n_refs"]) if r["n_refs"] is not None else None)):
        a = [v for v in map(fn, treated) if v is not None]
        b = [v for v in map(fn, controls) if v is not None]
        out[key] = {"smd": smd(a, b), "mean_treated": round(float(np.mean(a)), 4) if a else None,
                    "mean_control": round(float(np.mean(b)), 4) if b else None,
                    "n_treated": len(a), "n_control": len(b),
                    "missing_treated": len(treated) - len(a), "missing_control": len(controls) - len(b)}
    return out


def judge_packets(rows: list[dict], pairs: list[tuple[int, int, float]], n: int = N_JUDGE,
                  seed: int = JUDGE_SEED) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    k = min(n, len(pairs))
    idx = sorted(rng.choice(len(pairs), size=k, replace=False).tolist()) if k else []
    items, key = [], {}
    for num, i in enumerate(idx, 1):
        t, c, _ = pairs[i]
        pair_id = f"p{num:02d}"
        treated_first = bool(rng.random() < 0.5)
        a, b = (t, c) if treated_first else (c, t)
        items.append({"pair_id": pair_id,
                      "paper_a": {"title": rows[a]["title"], "abstract": rows[a]["abstract"]},
                      "paper_b": {"title": rows[b]["title"], "abstract": rows[b]["abstract"]}})
        key[pair_id] = {"treated": "a" if treated_first else "b", "pmid_a": rows[a]["pmid"],
                        "pmid_b": rows[b]["pmid"], "treated_pmid": rows[t]["pmid"],
                        "control_pmid": rows[c]["pmid"]}
    packets = {
        "task": ("For each pair, read the two titles and abstracts. Answer 'yes' if the two papers are on "
                 "the same research topic (same gene/alteration and disease/question), otherwise 'no'. "
                 "Return a JSON list of {pair_id, same_topic: yes|no}."),
        "items": items,
    }
    return packets, key


def _quantiles(values: list[float]) -> dict:
    if not values:
        return {}
    v = np.asarray(values)
    return {f"q{int(q * 100):02d}": round(float(np.quantile(v, q)), 4) for q in (0, 0.05, 0.25, 0.5, 0.75, 0.95, 1)}


def run(pathway_dirs: dict[str, Path] = PATHWAY_DIRS, out_dir: Path = OUT_DIR, bench: Path = BENCH) -> dict:
    rows = load_pool(pathway_dirs)
    out_dir.mkdir(parents=True, exist_ok=True)
    bench.mkdir(parents=True, exist_ok=True)

    # Removed list first, before any matching.
    vocab = Counter(t for r in rows for t in set(r["raw_mesh"]) if t)
    removed = removed_descriptors(vocab)
    removed_doc = {
        "ledger": "experiments.md, 2026-09-30 metascience v4",
        "normalization": "qualifiers after '/' and '*' dropped, case-folded (features.norm_term)",
        "rules": {"explicit_by_category": {k: sorted({F.norm_term(t) for t in v}) for k, v in CATEGORIES.items()},
                  "prefixes": list(REMOVED_PREFIXES), "named_cell_line_regex": NAMED_CELL_LINE.pattern,
                  "method_suffix_regex": METHOD_SUFFIX.pattern},
        "n_removed_in_corpus": len(removed),
        "removed_in_corpus": {t: vocab[t] for t in removed},
    }
    for path in (out_dir / "removed_mesh.json", bench / "removed_mesh.json"):
        path.write_text(json.dumps(removed_doc, indent=2) + "\n", encoding="utf-8")

    x = tfidf(rows)
    ti, ci, sims = candidate_edges(rows, x, min(CALIPERS))
    pairs_all = greedy(rows, ti, ci, sims)
    n_treated = sum(r["treated"] for r in rows)
    table = {}
    for c in CALIPERS:
        n = sum(1 for p in pairs_all if p[2] >= c)
        table[c] = {"n_pairs": n, "match_rate": round(n / n_treated, 4) if n_treated else 0.0}
    caliper, reached = choose_caliper({c: v["match_rate"] for c, v in table.items()})
    # Diagnostic only (not a ledger caliper): match rates below the ledger's grid.
    dti, dci, dsims = candidate_edges(rows, x, DIAGNOSTIC_FLOOR)
    diag_pairs = greedy(rows, dti, dci, dsims)
    diagnostic = {str(c): round(sum(1 for p in diag_pairs if p[2] >= c) / n_treated, 4) if n_treated else 0.0
                  for c in DIAGNOSTIC_THRESHOLDS}
    pairs = [p for p in pairs_all if p[2] >= caliper]

    with (out_dir / "pairs.jsonl").open("w", encoding="utf-8") as handle:
        for t, c, s in pairs:
            handle.write(json.dumps({"treated_pmid": rows[t]["pmid"], "control_pmid": rows[c]["pmid"],
                                     "pathway": rows[t]["pathway"], "gene_group": rows[t]["gene_group"],
                                     "similarity": round(s, 6)}) + "\n")

    pool = {}
    for name in pathway_dirs:
        sub = [r for r in rows if r["pathway"] == name]
        pool[name] = {arm: {"n": sum(1 for r in sub if r["treated"] == v),
                            "unmatchable_no_descriptors": sum(1 for r in sub if r["treated"] == v and not r["terms"])}
                      for arm, v in (("treated", 1), ("control", 0))}
    per_pathway = {}
    for name in pathway_dirs:
        n_t = pool[name]["treated"]["n"]
        n_p = sum(1 for t, _, _ in pairs if rows[t]["pathway"] == name)
        per_pathway[name] = {"n_treated": n_t, "n_pairs": n_p, "match_rate": round(n_p / n_t, 4) if n_t else None}
    treated_rows = [r for r in rows if r["treated"]]
    control_rows = [r for r in rows if not r["treated"]]
    summary = {
        "ledger": "experiments.md, 2026-09-30 metascience v4",
        "blinding": "no outcome file read; caliper chosen from similarity only",
        "pool": pool,
        "n_treated": n_treated, "n_control": len(control_rows),
        "n_topic_descriptors": x.shape[1], "n_removed_descriptors": len(removed),
        "n_candidate_edges_at_min_caliper": int(len(sims)),
        "caliper_table": {str(c): v for c, v in table.items()},
        "match_rate_denominator": "all treated papers, including unmatchable",
        "caliper": caliper, "caliper_reached_target": reached,
        "diagnostic_match_rate_below_grid": diagnostic,
        "diagnostic_note": ("not ledger calipers. Upper bound on the match rate "
                            "is set by control supply per pathway x gene_group x year window."),
        "max_feasible_pairs_ignoring_similarity": max_feasible(rows),
        "n_pairs": len(pairs), "per_pathway": per_pathway,
        "similarity_quantiles": _quantiles([s for _, _, s in pairs]),
        "balance_before": balance(treated_rows, control_rows),
        "balance_after": balance([rows[t] for t, _, _ in pairs], [rows[c] for _, c, _ in pairs]),
    }
    (bench / "matching_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    packets, key = judge_packets(rows, pairs)
    (bench / "match_judge_packets.json").write_text(json.dumps(packets, indent=2) + "\n", encoding="utf-8")
    (bench / "match_judge_key.json").write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
