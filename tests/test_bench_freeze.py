from src.bench.freeze import (
    apply_freeze,
    blinded_union,
    consensus_judgments,
    judged_proposition,
    order_status,
    route_comparison,
    same_report,
)


def test_year_placeholder_in_the_source_year_is_uncertain_and_a_later_year_is_not():
    assert order_status("2006-06-01", 2006, "2006-01-01") == "uncertain"
    assert order_status("2007-01-01", 2007, "2006-01-01") == "subsequent"
    assert order_status("2010-08-01", 2010, "2010-06-01") == "subsequent"
    assert order_status("2010-06-01", 2010, "2010-06-01") == "uncertain"
    assert order_status("2010-01-01", 2010, "2010-06-15") == "uncertain"


def test_same_title_and_overlapping_abstract_are_one_report():
    left = {
        "pmid": "17601930",
        "year": 2007,
        "date": "2007-07-01",
        "title": "De novo HRAS and KRAS mutations in two siblings with short stature and neuro-cardio-facio-cutaneous features.",
        "abstract": "We describe two sisters who presented with dysmorphic features. The patients were initially diagnosed with Costello syndrome. We identified a germline HRAS mutation in one sister.",
    }
    right = {
        "pmid": "21686750",
        "year": 2009,
        "date": "2009-01-23",
        "title": "De novo HRAS and KRAS mutations in two siblings with short stature and neuro-cardio-facio-cutaneous features.",
        "abstract": "We describe two sisters who presented with dysmorphic features. The patients were initially diagnosed with Costello syndrome. We identified a germline HRAS mutation in one sister and a second mutation.",
    }
    other = {
        "pmid": "1",
        "year": 2008,
        "date": "2008-01-01",
        "title": "A mouse model for Costello syndrome",
        "abstract": "Germline activation of H-RAS causes Costello syndrome in this mouse.",
    }
    assert same_report(left, right)
    assert not same_report(left, other)
    claim = {
        "claim_id": "C001",
        "pmid": "16372351",
        "cutoff": "2006-01-01",
        "in_labeled_eval": True,
        "spec": {
            "setting": "36 patients with Costello syndrome",
            "intervention": "heterozygous HRAS missense mutations in codons 12 and 13",
            "direction": "positive",
            "quote": "identified in 33/36 (92%) patients",
            "outcome": "mutations were identified in 33 of 36 patients",
        },
        "lists": {"same_gene": ["17601930"], "experimental": ["17601930", "21686750"], "text_similarity": [], "citation": [], "combination": ["21686750"], "shared_references": []},
        "packet_pmids": [],
        "neighbors": [
            {**left, "relationships": [{"type": "experimental"}], "pub_types": [], "abstract": left["abstract"], "title": left["title"]},
            {**right, "relationships": [{"type": "experimental"}], "pub_types": [], "abstract": right["abstract"], "title": right["title"]},
        ],
        "reviews": [],
    }
    frozen = apply_freeze([claim])[0]
    assert frozen["neighbors"][1]["duplicate_of"] == "17601930"
    assert "33/36" not in frozen["judged_proposition"]
    assert "Costello" in frozen["judged_proposition"]
    blinded, provenance = blinded_union(frozen and [frozen])
    blob = str(blinded)
    assert "agreed_support" not in blob
    assert "experimental" not in blob
    assert provenance["rows"][0]["methods"]


def test_a_duplicate_and_an_uncertain_paper_do_not_add_a_direct_test():
    claim = apply_freeze(
        [
            {
                "claim_id": "C9",
                "pmid": "1",
                "cutoff": "2010-06-01",
                "in_labeled_eval": True,
                "spec": {"setting": "patients", "intervention": "KRAS mutation", "direction": "positive", "quote": "4 of 10", "outcome": ""},
                "lists": {
                    "same_gene": ["10"],
                    "experimental": ["10", "11", "12"],
                    "text_similarity": [],
                    "citation": [],
                    "combination": ["11"],
                    "shared_references": [],
                },
                "packet_pmids": [],
                "neighbors": [
                    {"pmid": "10", "year": 2011, "date": "2011-01-01", "title": "Cohort", "abstract": "KRAS cohort", "relationships": [{"type": "same_gene"}], "pub_types": []},
                    {"pmid": "11", "year": 2011, "date": "2011-02-01", "title": "Cohort", "abstract": "KRAS cohort repeat", "relationships": [{"type": "experimental"}], "pub_types": []},
                    {"pmid": "12", "year": 2010, "date": "2010-01-01", "title": "Early", "abstract": "KRAS early", "relationships": [{"type": "experimental"}], "pub_types": []},
                ],
                "reviews": [],
            }
        ]
    )[0]
    # 11 shares the title "Cohort" but abstracts differ enough that they may not cluster.
    claim["neighbors"][1]["duplicate_of"] = "10"
    scored = route_comparison([claim], {"C9": {"10": "support", "11": "support", "12": "support"}})
    experimental = scored["methods"]["experimental"]
    assert experimental["slots_filled"] == 3
    assert experimental["slots_available"] == 20
    assert experimental["direct_tests"] == 1
    assert experimental["duplicate_reports_not_counted"] == 1
    assert experimental["uncertain_order_not_counted"] == 1
    assert experimental["direct_tests_beyond_same_gene"] == 0


def test_consensus_requires_both_readers_and_the_same_direction():
    def pack(calls):
        return {"claims": [{"claim_id": "C1", "followups": [
            {"pmid": pmid, "relation": relation, "independent": independent}
            for pmid, relation, independent in calls
        ]}]}
    left = pack([("1", "direct_support", True), ("2", "direct_support", True), ("3", "direct_support", True), ("4", "not_test", True)])
    right = pack([("1", "direct_support", True), ("2", "direct_contradict", True), ("3", "not_test", True), ("4", "not_test", True)])
    judgments, stats = consensus_judgments([(left, right)])
    assert judgments["C1"] == {"1": "support"}
    assert stats["direction_disagreements"] == 1
    assert stats["one_reader_only"] == 1
    assert stats["n_papers"] == 4


def test_proposition_drops_the_source_headcount_from_the_setting():
    text = judged_proposition(
        {
            "setting": "36 patients with Costello syndrome",
            "intervention": "heterozygous HRAS missense mutations in codons 12 and 13",
            "direction": "positive",
            "quote": "33/36",
        }
    )
    assert text["judged_proposition"].startswith("In patients with Costello syndrome")
    assert text["source_measurement"] == "33/36"
