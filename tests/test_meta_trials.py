"""Synthetic tests for src.meta.trials (no network)."""

from __future__ import annotations

import json

import pytest

from src.meta import trials


def _study(nct, stype="INTERVENTIONAL", date="2010-05", phases=("PHASE2",), itypes=("DRUG",), refs=()):
    ps = {
        "identificationModule": {"nctId": nct},
        "statusModule": {"startDateStruct": {"date": date}} if date else {},
        "designModule": {"studyType": stype, **({"phases": list(phases)} if phases else {})},
        "armsInterventionsModule": {"interventions": [{"type": t} for t in itypes]},
    }
    if refs:
        ps["referencesModule"] = {
            "references": [{"pmid": p, "type": t, "citation": f"Author A. Cite {p}."} for p, t in refs]
        }
    return {"protocolSection": ps}


def test_start_year_formats():
    assert trials.start_year("2010-05") == 2010
    assert trials.start_year("2010-05-17") == 2010
    assert trials.start_year(None) is None and trials.start_year("") is None


def test_parse_study_fields():
    row = trials.parse_study(
        _study("NCT1", itypes=("DRUG", "BIOLOGICAL", "DRUG"), refs=[("12", "BACKGROUND"), (" ", "RESULT")]), page=3
    )
    assert row == {
        "nct": "NCT1",
        "study_type": "INTERVENTIONAL",
        "start_date": "2010-05",
        "start_year": 2010,
        "phases": ["PHASE2"],
        "intervention_types": ["BIOLOGICAL", "DRUG"],
        "page": 3,
        "refs": [["12", "BACKGROUND"], [None, "RESULT"]],
    }
    bare = trials.parse_study({"protocolSection": {"identificationModule": {"nctId": "NCT2"}}})
    assert bare["start_year"] is None and bare["phases"] == [] and bare["refs"] == []
    assert trials.parse_study({"protocolSection": {}}) is None


def test_fetch_pages_token_paging_dedup_and_cache(tmp_path):
    pages = {
        None: {"totalCount": 3, "nextPageToken": "t1", "studies": [_study("NCT2"), _study("NCT1")]},
        "t1": {"studies": [_study("NCT1"), _study("NCT3")]},
    }
    calls = []

    def getter(url, params):
        calls.append(params.get("pageToken"))
        assert url == trials.API_URL
        assert params["pageSize"] == 1000 and params["fields"] == trials.FIELDS
        return pages[params.get("pageToken")]

    studies, stats = trials.fetch_studies(getter, tmp_path)
    assert [s["nct"] for s in studies] == ["NCT1", "NCT2", "NCT3"]
    assert stats["total_count"] == 3 and stats["n_pages"] == 2 and stats["n_duplicate_nct"] == 1
    assert calls == [None, "t1"]
    assert (tmp_path / "page_0000.json").exists() and (tmp_path / "page_0001.json").exists()
    # rerun is offline
    again, _ = trials.fetch_studies(lambda u, p: pytest.fail("network"), tmp_path)
    assert [s["nct"] for s in again] == ["NCT1", "NCT2", "NCT3"]


def test_fetch_resumes_from_cached_token(tmp_path):
    (tmp_path / "page_0000.json").write_text(json.dumps({"nextPageToken": "t1", "studies": [_study("NCT1")]}))
    calls = []

    def getter(url, params):
        calls.append(params.get("pageToken"))
        return {"studies": [_study("NCT9")]}

    studies, _ = trials.fetch_studies(getter, tmp_path)
    assert calls == ["t1"] and [s["nct"] for s in studies] == ["NCT1", "NCT9"]


def _rows():
    return [
        trials.parse_study(s, page=0)
        for s in [
            _study("NCT1", refs=[("100", "BACKGROUND"), ("100", "BACKGROUND"), ("200", "RESULT")]),
            _study("NCT2", stype="OBSERVATIONAL", refs=[("100", "BACKGROUND")]),
            _study("NCT3", date=None, phases=(), itypes=("BEHAVIORAL",), refs=[("100", "BACKGROUND"), ("300", "DERIVED")]),
            _study("NCT4", refs=[(None, "BACKGROUND")]),
            _study("NCT5"),
        ]
    ]


def test_build_index_background_interventional_only():
    index = trials.build_index(_rows())
    assert index == [
        {
            "pmid": "100",
            "trials": [
                {"nct": "NCT1", "start_year": 2010, "phases": ["PHASE2"], "intervention_types": ["DRUG"]},
                {"nct": "NCT3", "start_year": None, "phases": [], "intervention_types": ["BEHAVIORAL"]},
            ],
        }
    ]


def test_study_stats_counts():
    stats = trials.study_stats(_rows())
    assert stats["n_studies_any_reference"] == 4
    assert stats["n_references_by_type"] == {"BACKGROUND": 5, "RESULT": 1, "DERIVED": 1}
    assert stats["n_references_with_pmid_by_type"]["BACKGROUND"] == 4
    assert stats["n_interventional"] == 4
    assert stats["n_interventional_with_background_pmid"] == 2
    assert stats["n_interventional_with_background_pmid_no_start_date"] == 1


def _t(nct, year, phases=("PHASE2",), itypes=("DRUG",)):
    return {"nct": nct, "start_year": year, "phases": list(phases), "intervention_types": list(itypes)}


def test_window_bounds():
    assert trials.in_window(_t("a", 2004), 2005) and trials.in_window(_t("a", 2013), 2005)
    assert not trials.in_window(_t("a", 2003), 2005) and not trials.in_window(_t("a", 2014), 2005)
    assert not trials.in_window(_t("a", None), 2005)


def test_phase_and_drug_rules():
    assert trials.is_ph2(_t("a", 1, ("PHASE2",)))
    assert trials.is_ph2(_t("a", 1, ("PHASE2", "PHASE3")))
    assert trials.is_ph2(_t("a", 1, ("PHASE4",)))
    assert not trials.is_ph2(_t("a", 1, ("PHASE1", "PHASE2")))
    assert not trials.is_ph2(_t("a", 1, ("NA",)))
    assert not trials.is_ph2(_t("a", 1, ()))
    assert trials.is_drug(_t("a", 1, itypes=("BIOLOGICAL", "OTHER")))
    assert not trials.is_drug(_t("a", 1, itypes=("PROCEDURE",)))


def test_outcome_row():
    ts = [
        _t("A", 2003),  # before window
        _t("B", 2006, ("PHASE1",), ("DEVICE",)),  # in window, not drug, not ph2
        _t("C", 2012, ("PHASE1", "PHASE2"), ("BIOLOGICAL",)),  # in window, drug, phase 1/2
        _t("D", None),  # undated
    ]
    assert trials.outcome_row("7", 2005, ts) == {
        "pmid": "7",
        "trial_bg_8y": 1,
        "n_trials_bg_8y": 2,
        "trial_bg_8y_drug": 1,
        "trial_bg_8y_ph2": 0,
        "first_trial_year": 2003,
        "n_trials_bg_undated": 1,
    }
    assert trials.outcome_row("8", 2005, []) == {
        "pmid": "8", "trial_bg_8y": 0, "n_trials_bg_8y": 0, "trial_bg_8y_drug": 0,
        "trial_bg_8y_ph2": 0, "first_trial_year": None, "n_trials_bg_undated": 0,
    }


def test_compute_outcome_skips_missing_year_and_counts():
    index = {"1": [_t("A", 2010, ("PHASE1", "PHASE2"))], "2": [_t("B", None)]}
    rows, stats = trials.compute_outcome(["1", "2", "3"], {"1": 2008, "2": 2008, "3": None}, index)
    assert [r["pmid"] for r in rows] == ["1", "2"]
    assert stats == {"no_year_skipped": 1, "n_linked_pairs_undated": 1, "n_ph12_in_window_only": 1}
    m = trials.marginals(rows)
    assert m["n_eligible"] == 2 and m["n_trial_bg_8y"] == 1 and m["rate_trial_bg_8y"] == 0.5
    assert m["n_linked_any_year"] == 2


def test_eligible_pmids_reads_only_preclinical_eligible(tmp_path):
    path = tmp_path / "features.jsonl"
    rows = [
        {"pmid": "3", "group": "preclinical", "eligible": True},
        {"pmid": "1", "group": "preclinical", "eligible": True},
        {"pmid": "2", "group": "clinical", "eligible": True},
        {"pmid": "4", "group": "preclinical", "eligible": False},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert trials.eligible_pmids(path) == ["1", "3"]


def test_paper_years_icite_fallback(tmp_path):
    papers = tmp_path / "papers.jsonl"
    icite = tmp_path / "icite.jsonl"
    papers.write_text(json.dumps({"pmid": "1", "year": 2004}) + "\n" + json.dumps({"pmid": "2", "year": None}) + "\n")
    icite.write_text(json.dumps({"pmid": 2, "year": 2006}) + "\n")
    years, n = trials.paper_years(papers, icite, {"1", "2"})
    assert years == {"1": 2004, "2": 2006} and n == 1


def test_title_overlap_and_tokens():
    title = "Oncogenic <i>KRAS</i> drives resistance to MEK inhibition in colorectal cancer"
    good = "Smith J. Oncogenic KRAS drives resistance to MEK inhibition in colorectal cancer. Nature. 2010."
    assert trials.title_overlap(title, good) == 1.0
    assert trials.title_overlap(title, "Doe A. A trial of aspirin in heart failure. Lancet. 2011.") == 0.0
    assert trials.title_overlap("", good) is None


def test_sample_pairs_deterministic():
    pairs = [("egfr", str(i), f"NCT{i}") for i in range(50)] + [("ras", "5", "NCT5")] * 2
    a = trials.sample_pairs(pairs, n=30, seed=0)
    assert a == trials.sample_pairs(list(reversed(pairs)), n=30, seed=0)
    assert len(a) == 30 and len(set(a)) == 30
    assert trials.sample_pairs(pairs[:3], n=30) == sorted(pairs[:3], key=lambda t: int(t[1]))


def test_reference_entries_from_cache(tmp_path):
    payload = {"studies": [_study("NCT1", refs=[("100", "BACKGROUND"), ("200", "BACKGROUND")])]}
    (tmp_path / "page_0002.json").write_text(json.dumps(payload))
    got = trials.reference_entries("NCT1", 2, "200", tmp_path)
    assert got == [{"pmid": "200", "type": "BACKGROUND", "citation": "Author A. Cite 200."}]
    assert trials.reference_entries("NCT9", 2, "200", tmp_path) == []
