"""Correct the pilot judgments without touching the frozen retrieval lists.

The two readers agreed on a direct test for 85 claim-paper pairs and disagreed
on direction for eight more. This module records an audit of those pairs, of
every one-sided call, and of one both-rejected paper per claim. Counts are
recomputed from the corrected directions. Retrieval order is left as frozen.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.bench.freeze import consensus_judgments, route_comparison

# Papers removed from the direct-test count, with the reason.
DROPS: dict[tuple[str, str], str] = {
    ("C001", "21850009"): (
        "Disputed. The abstract reports four HRAS mutations in 21 patients and "
        "then functional senescence assays. Whether the patients were selected "
        "by clinical diagnosis, and whether the cohort overlaps earlier reports, "
        "is not settled by the abstract."
    ),
    ("C013", "26223933"): "Meta-analysis. A pooled reanalysis is not a new experiment.",
    ("C013", "25244593"): "Meta-analysis. A pooled reanalysis is not a new experiment.",
    ("C013", "23055546"): "Meta-analysis. A pooled reanalysis is not a new experiment.",
    ("C014", "19020799"): (
        "RAF1 mutations co-occurred with cardiomyopathy, but the abstract does "
        "not separate CR2-domain from CR3-domain mutations."
    ),
    ("C014", "24782337"): "Single-infant case. Not a comparison of CR2 and CR3 mutations.",
    ("C017", "27736842"): (
        "Extended RAS analysis of a phase III panitumumab versus best-supportive-care "
        "trial, the same design as the source claim, rather than a new cohort."
    ),
    ("C026", "20591136"): "The new measurement is CD56 infiltrate. KRAS is background.",
    ("C033", "23936355"): "Diagnostic-algorithm paper. It does not measure driver exclusivity.",
    ("C035", "27898473"): "Two overlap-syndrome cases, not a prevalence series of Langerhans cell histiocytosis.",
    ("C035", "29649018"): "Cases were already selected for a BRAF mutation, so prevalence is not tested.",
    ("C057", "23285177"): "Assay-platform paper. It does not measure the acquired-resistance phenotype.",
    ("C063", "33202284"): (
        "Tests whether Notch protects BRAF-mutant lines from cobimetinib, not whether "
        "cobimetinib has superior efficacy in those models."
    ),
}

# Agreed direction replaced after reading the result sentence.
DIRECTION_CHANGES: dict[tuple[str, str], tuple[str, str]] = {
    ("C013", "26120069"): (
        "mixed",
        "BRAF predicted bilaterality and did not independently predict nodal metastasis or recurrence.",
    ),
    ("C024", "19501586"): (
        "mixed",
        "Acinar Kras induced pancreatic intraepithelial neoplasia, and higher Ras activity also produced invasive carcinoma.",
    ),
}

# Direction disagreements and one-sided calls promoted after reading.
ADDITIONS: dict[tuple[str, str], tuple[str, str]] = {
    ("C016", "26823860"): (
        "mixed",
        "Extrathyroidal extension was associated with BRAF; other listed clinicopathologic features were not.",
    ),
    ("C016", "26120069"): (
        "mixed",
        "BRAF predicted bilaterality and did not independently predict nodal metastasis or recurrence.",
    ),
    ("C016", "25489262"): (
        "mixed",
        "BRAF independently predicted recurrence and was not associated with invasion or nodal metastasis.",
    ),
    ("C016", "23687957"): (
        "mixed",
        "Univariate associations with size and extension were not retained in the multivariate model.",
    ),
    ("C033", "27438512"): (
        "mixed",
        "EML4-ALK and EGFR or KRAS were mostly separate, with one lesion carrying both L858R and EML4-ALK.",
    ),
    ("C057", "32912923"): (
        "mixed",
        "One vemurafenib-resistant line reactivated MEK/ERK and the paired line activated AKT.",
    ),
    ("C015", "17685465"): (
        "support",
        "In papillary carcinomas 1 cm or smaller, BRAF V600E tracked with nodal progression.",
    ),
    ("C015", "24354346"): (
        "contradict",
        "In papillary microcarcinoma, BRAF was not correlated with aggressive or recurrent disease.",
    ),
    ("C026", "19738388"): (
        "support",
        "In irinotecan-refractory Korean colorectal cancers, KRAS predicted response and survival on cetuximab plus irinotecan.",
    ),
    ("C033", "25706305"): (
        "support",
        "EML4-ALK was exclusive of EGFR and KRAS mutations in the resected series.",
    ),
    ("C033", "27086595"): (
        "contradict",
        "The series includes tumors carrying both an EGFR mutation and an ALK rearrangement.",
    ),
}

# Read and left uncounted. These are the eight direction disagreements that
# are not direct tests, plus the one-sided calls that stayed partial.
NOT_A_TEST: dict[tuple[str, str], str] = {
    ("C001", "17601930"): "Two sisters, one HRAS and one KRAS, after a Costello diagnosis that the paper revises. Same report as PMID 21686750.",
    ("C001", "21686750"): "Second write-up of PMID 17601930.",
    ("C001", "20979192"): "One mosaic HRAS case. Not a cohort test of the Costello association.",
    ("C014", "22389993"): "Two LEOPARD-syndrome cases, a different diagnosis from Noonan syndrome.",
    ("C014", "26266034"): "One preterm infant. The authors draw no genotype-phenotype conclusion.",
    ("C016", "23690767"): "miRNA profile paper. BRAF is context, not the measured association.",
    ("C017", "26508446"): "Methods comparison of sequencing panels, not a new efficacy result.",
    ("C026", "27555788"): "Argues for broader sequencing. Does not report a new KRAS efficacy result.",
    ("C033", "26711128"): "Stage I prognosis by driver. Does not test mutual exclusivity.",
    ("C033", "25360721"): "Meta-analysis of clinical correlates, not a new exclusivity experiment.",
    ("C035", "28084334"): "The authors argue the lesions are hyperplasia rather than Langerhans cell histiocytosis.",
    ("C035", "26782803"): "Thyroid fine-needle aspiration pitfall. Not an LCH prevalence series.",
    ("C035", "24147236"): "One neonate. Not a prevalence series.",
    ("C035", "26454140"): "One overlap case. Not a prevalence series.",
    ("C041", "23942080"): "The PIK3CA association with progression-free survival was not significant.",
    ("C057", "25844720"): "Studies interferon-induced PD-L1, not the acquired BRAF-inhibitor resistance phenotype.",
    ("C063", "33904516"): "An Hsp90-inhibitor combination, not a test of cobimetinib superiority.",
    ("C063", "33560788"): "One non-V600 melanoma case.",
    ("C063", "31171876"): "A triple-combination response rate, not cobimetinib superiority.",
    ("C063", "34853302"): "Intermittent versus continuous dosing, not cobimetinib superiority.",
    ("C063", "34667063"): "One pancreatic-cancer case report.",
    ("C063", "31985841"): "One ganglioglioma case.",
    ("C063", "32534646"): "Atezolizumab added to vemurafenib plus cobimetinib. The comparison is immunotherapy, not cobimetinib superiority.",
}

# One both-rejected paper per claim, drawn with seed 20260929 from text-route
# partials when available. All twenty stayed rejected after reading.
REJECT_SAMPLE: dict[str, str] = {
    "C001": "22821884",
    "C007": "24586816",
    "C013": "26230187",
    "C014": "20602484",
    "C015": "25066317",
    "C016": "26080065",
    "C017": "27085587",
    "C021": "27503890",
    "C024": "26416424",
    "C026": "26384309",
    "C028": "27835580",
    "C030": "28656062",
    "C033": "29636358",
    "C035": "29271794",
    "C041": "20937558",
    "C042": "30389925",
    "C047": "29438965",
    "C057": "32562785",
    "C063": "32898388",
    "C068": "35595057",
}


def corrected_judgments(pairs: list[tuple[dict, dict]]) -> tuple[dict[str, dict[str, str]], dict]:
    """Return claim to pmid to direction, after the audit deltas."""
    judgments, agreement = consensus_judgments(pairs)
    notes: list[dict] = []
    for (claim_id, pmid), reason in DROPS.items():
        direction = judgments.get(claim_id, {}).pop(pmid, None)
        notes.append({"claim_id": claim_id, "pmid": pmid, "action": "drop", "was": direction, "reason": reason})
    for (claim_id, pmid), (direction, reason) in DIRECTION_CHANGES.items():
        was = judgments.get(claim_id, {}).get(pmid)
        judgments.setdefault(claim_id, {})[pmid] = direction
        notes.append({"claim_id": claim_id, "pmid": pmid, "action": "direction", "was": was, "now": direction, "reason": reason})
    for (claim_id, pmid), (direction, reason) in ADDITIONS.items():
        judgments.setdefault(claim_id, {})[pmid] = direction
        notes.append({"claim_id": claim_id, "pmid": pmid, "action": "add", "now": direction, "reason": reason})
    for (claim_id, pmid), reason in NOT_A_TEST.items():
        notes.append({"claim_id": claim_id, "pmid": pmid, "action": "not_a_test", "reason": reason})
    summary = {
        "drops": len(DROPS),
        "direction_changes": len(DIRECTION_CHANGES),
        "additions": len(ADDITIONS),
        "not_a_test": len(NOT_A_TEST),
        "reject_sample_kept_rejected": len(REJECT_SAMPLE),
        "disputed": ["C001:21850009"],
        "reader_agreement": agreement,
    }
    return judgments, {"summary": summary, "notes": notes}


def load_reader_pairs(root: Path) -> list[tuple[dict, dict]]:
    pairs = []
    for index in range(1, 6):
        left = json.loads((root / "bench" / "judge_labels" / f"A{index}.json").read_text())
        right = json.loads((root / "bench" / "judge_labels" / f"B{index}.json").read_text())
        pairs.append((left, right))
    return pairs


def recompute(claims: list[dict], pairs: list[tuple[dict, dict]]) -> dict:
    judgments, audit = corrected_judgments(pairs)
    comparison = route_comparison(claims, judgments)
    return {"judgments": judgments, "audit": audit, "comparison": comparison}
