import json
from pathlib import Path

from src.bench.audit import DROPS, corrected_judgments, load_reader_pairs
from src.bench.ranker import select_twenty

ROOT = Path(__file__).resolve().parents[1]


def test_disputed_costello_paper_is_not_counted():
    judgments, audit = corrected_judgments(load_reader_pairs(ROOT))
    assert "21850009" not in judgments["C001"]
    assert "C001:21850009" in audit["summary"]["disputed"]
    assert "21 patients" in DROPS[("C001", "21850009")]


def test_meta_analyses_drop_and_missed_irinotecan_cohort_is_added():
    judgments, _audit = corrected_judgments(load_reader_pairs(ROOT))
    assert "26223933" not in judgments["C013"]
    assert judgments["C026"]["19738388"] == "support"
    assert judgments["C013"]["26120069"] == "mixed"


def test_ranker_differences_sum_to_the_headline_gap():
    from src.bench.ranker import paired_reconciliation

    rows = [
        {"claim_id": "C033", "paired_difference": 3},
        {"claim_id": "C016", "paired_difference": -3},
        {"claim_id": "C035", "paired_difference": -3},
        {"claim_id": "C057", "paired_difference": 2},
        {"claim_id": "C001", "paired_difference": 1},
        {"claim_id": "C013", "paired_difference": 1},
        {"claim_id": "C026", "paired_difference": 1},
        {"claim_id": "C041", "paired_difference": 1},
        {"claim_id": "C042", "paired_difference": 1},
        {"claim_id": "C047", "paired_difference": -1},
        {"claim_id": "C007", "paired_difference": 0},
    ]
    report = paired_reconciliation(rows)
    assert report["highlighted_net"] == -3
    assert report["other_claims_net"] == 6
    assert report["total"] == 3


def test_provenance_selection_skips_a_second_copy_of_the_same_report():
    from src.bench.provenance import select_with_provenance

    abstract = "We enrolled forty patients with Costello syndrome and found an HRAS mutation in this study."
    claim = {
        "spec": {
            "setting": "patients with Costello syndrome",
            "intervention": "HRAS mutation",
            "outcome": "HRAS mutation identified in patients with Costello syndrome",
        },
        "lists": {
            "text_similarity": ["1", "2", "3"],
            "same_gene": [],
            "citation": [],
            "experimental": [],
            "shared_references": [],
        },
        "neighbors": [
            {"pmid": "1", "title": "HRAS in Costello syndrome", "abstract": abstract},
            {"pmid": "2", "title": "HRAS in Costello syndrome", "abstract": abstract},
            {"pmid": "3", "title": "A different melanoma study", "abstract": "Unrelated melanoma cell lines were cultured."},
        ],
    }
    picked = select_with_provenance(claim)
    assert picked[0] == "1"
    assert "2" not in picked
    assert picked[1] == "3"


def test_a_pmid_mention_without_reuse_language_is_not_shared_evidence():
    from src.bench.provenance import evidence_relation

    prior = {"pmid": "24445999", "title": "Atg5 in lung cancer", "abstract": "We inactivated Atg5 in mice."}
    citation = {
        "pmid": "36156071",
        "title": "Systemic autophagy inhibition",
        "abstract": "Unlike PMID 24445999, tumor-cell Atg5 loss did not reduce established tumor growth.",
    }
    reuse = {
        "pmid": "30000000",
        "title": "Reanalysis",
        "abstract": "This reanalysis of PMID 24445999 uses the same cohort.",
    }
    assert evidence_relation(citation, prior) == "uncertain"
    assert evidence_relation(reuse, prior) == "same_cohort"


def test_a_shared_trial_identifier_does_not_discount_the_paper():
    from src.bench.provenance import evidence_relation, select_with_provenance

    left = {"pmid": "1", "title": "Arm A of the trial", "abstract": "NCT01234567 enrolled patients with colorectal cancer. We found a KRAS result in this study."}
    right = {"pmid": "2", "title": "Arm B of the trial", "abstract": "NCT01234567 enrolled a later endpoint cohort. We report overall survival in this study."}
    assert evidence_relation(left, right) == "same_trial"
    claim = {
        "spec": {
            "setting": "patients with colorectal cancer",
            "intervention": "KRAS mutation",
            "outcome": "KRAS result in patients with colorectal cancer",
        },
        "lists": {"text_similarity": ["1", "2"], "same_gene": [], "citation": [], "experimental": [], "shared_references": []},
        "neighbors": [left, right],
    }
    assert select_with_provenance(claim) == ["1", "2"]


def test_mechanism_query_requires_the_named_perturbation_and_not_a_paralog():
    from src.bench.mechanism import mechanism_query, perturbation_terms

    claim = {
        "pmid": "24445999",
        "cutoff": "2014-01-01",
        "gene": "KRAS",
        "population": "mice with KRas(G12D)-driven lung cancer",
        "exposure": "tissue-specific Atg5 inactivation",
        "effect": "Atg5 inactivation markedly impaired progression of KRas(G12D)-driven lung cancer and extended survival of tumour-bearing mice.",
        "quote": "tissue-specific inactivation of Atg5 markedly impairs the progression of KRas(G12D)-driven lung cancer.",
    }
    terms = perturbation_terms(claim)
    query = mechanism_query(claim)
    assert "Atg5" in terms
    assert "Atg7" not in query
    assert "Atg5 OR ATG5" in query


def test_ranker_does_not_rewrite_frozen_lists():
    payload = json.loads((ROOT / "data" / "graph" / "explorer.json").read_text())
    claim = next(row for row in payload["claims"] if row["claim_id"] == "C001")
    before = json.dumps(claim["lists"])
    picked = select_twenty(claim)
    assert json.dumps(claim["lists"]) == before
    assert len(picked) <= 20
    assert picked
