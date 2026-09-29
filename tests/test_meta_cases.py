"""Synthetic tests for src.meta.cases (no network)."""

from __future__ import annotations

import re

import pytest

from src.meta import cases


# --- fakes -----------------------------------------------------------------------


def _item(pmid, year, author="Smith", title="KRAS study", mesh=("Humans",), pub_types=("Journal Article",)):
    return {
        "pmid": str(pmid),
        "pubYear": str(year),
        "title": title,
        "abstractText": "Abstract.",
        "authorString": f"{author} A, Other B.",
        "journalInfo": {"journal": {"title": "J"}},
        "meshHeadingList": {"meshHeading": [{"descriptorName": m} for m in mesh]},
        "pubTypeList": {"pubType": list(pub_types)},
    }


class FakeEPMC:
    """Answers EXT_ID batch queries and AUTH searches from a dict of items."""

    def __init__(self, items):
        self.items = {i["pmid"]: i for i in items}
        self.calls = []

    def __call__(self, url, params):
        self.calls.append(params["query"])
        q = params["query"]
        m = re.match(r"EXT_ID:\((.*)\) AND SRC:MED", q)
        if m:
            ids = m.group(1).split(" OR ")
            return {"resultList": {"result": [self.items[i] for i in ids if i in self.items]}}
        am = re.search(r'AUTH:"([^"]+)" AND PUB_YEAR:(\d+)', q)
        hits = [i for i in self.items.values()
                if i["authorString"].startswith(am.group(1)) and i["pubYear"] == am.group(2)]
        return {"resultList": {"result": hits}}


def _icite_fetch(table):
    def fetch(pmids):
        return [dict(table[p], pmid=p) for p in pmids if p in table]
    return fetch


# --- stats -----------------------------------------------------------------------


def test_wilson_known_values():
    w = cases.wilson(5, 10)
    assert w["share"] == 0.5
    assert w["ci95"] == pytest.approx([0.2366, 0.7634], abs=1e-4)
    assert cases.wilson(0, 0)["share"] is None


def test_newcombe_diff_contains_point_and_is_symmetric():
    d = cases.newcombe_diff(30, 50, 20, 50)
    assert d["diff"] == pytest.approx(0.2)
    lo, hi = d["ci95"]
    assert lo < 0.2 < hi
    e = cases.newcombe_diff(20, 50, 30, 50)
    assert e["ci95"] == pytest.approx([-hi, -lo], abs=1e-4)
    assert cases.newcombe_diff(1, 0, 1, 2)["diff"] is None


def test_shares_exclude_non_primary_and_unclassified():
    s = cases.shares(["animal", "animal", "cell_only", "non_primary", "unclassified"])
    assert s["n_classified"] == 3
    assert s["n_non_primary"] == 1 and s["n_unclassified"] == 1
    assert s["shares"]["animal"]["k"] == 2 and s["shares"]["human_samples"]["k"] == 0


# --- lineage -----------------------------------------------------------------------


def _table():
    R = {"is_research_article": True, "is_clinical": False}
    return {
        "100": {**R, "year": 2011, "references": ["1", "2", "3", "4", "5", "100"]},
        "1": {**R, "year": 2005, "references": ["2", "10", "11", "1"]},
        "2": {**R, "year": 2011, "references": ["12"]},
        "3": {"is_research_article": False, "year": 2004, "references": ["13"]},
        "4": {**R, "year": 2012, "references": ["14"]},  # after anchor year
        # "5" missing from iCite
        "10": {**R, "year": 2000, "references": []},
        "11": {**R, "year": 2013, "references": []},
        "12": {**R, "year": 1999, "is_clinical": True, "references": []},
        "13": {**R, "year": 2000, "references": []},
        "14": {**R, "year": 2000, "references": []},
    }


def test_build_lineage_depths_filters_and_dedup():
    lin = cases.build_lineage(["100"], 2011, _icite_fetch(_table()))
    # depth 1: 1 and 2 kept (2011 <= 2011); 3 not research; 4 too late; 5 not in iCite; anchor self-ref excluded
    assert {p: d for p, d in lin["depth"].items() if d == 1} == {"1": 1, "2": 1}
    assert lin["dropped"][1] == {"not_research_article": 1, "after_anchor_year": 1, "not_in_icite": 1}
    # depth 2 from kept depth-1 only: 2 (already depth 1, not recounted), 10, 11 (too late), 12; not 13/14
    assert {p: d for p, d in lin["depth"].items() if d == 2} == {"10": 2, "12": 2}
    assert lin["dropped"][2] == {"after_anchor_year": 1}
    assert "100" not in lin["depth"]


def test_build_lineage_multiple_anchors_exclude_each_other():
    t = _table()
    t["200"] = {"is_research_article": True, "year": 2012, "references": ["100", "10"]}
    lin = cases.build_lineage(["100", "200"], 2012, _icite_fetch(t))
    assert "100" not in lin["depth"] and "200" not in lin["depth"]
    assert lin["depth"]["10"] == 1  # reached at depth 1 via 200, not recounted at 2
    assert lin["depth"]["4"] == 1   # 2012 allowed with max_year 2012


def test_build_lineage_missing_anchor_raises():
    with pytest.raises(ValueError):
        cases.build_lineage(["999"], 2011, _icite_fetch(_table()))


# --- classification ---------------------------------------------------------------------


def test_status_order():
    assert cases.status(None) == "unclassified"
    assert cases.status({"mesh": [], "pub_types": ["Review"], "title": "", "abstract": "mice"}) == "unclassified"
    assert cases.status({"mesh": ["Humans", "Mice, Nude"], "pub_types": ["Review"]}) == "non_primary"
    assert cases.status({"mesh": ["Humans", "Mice, Nude"], "pub_types": ["Journal Article"]}) == "animal"
    assert cases.status({"mesh": ["Humans", "Cell Line, Tumor"], "pub_types": []}) == "cell_only"
    assert cases.status({"mesh": ["Humans", "Biopsy"], "pub_types": []}) == "human_samples"
    assert cases.status({"mesh": ["Proteins"], "pub_types": []}) == "other"


def test_design_line_mentions_mesh_and_trial():
    paper = {"mesh": ["Humans", "Middle Aged", "Biopsy"], "pub_types": ["Clinical Trial, Phase I"],
             "title": "BRAF", "abstract": "Tumor biopsies from 20 patients."}
    line = cases.design_line(paper)
    assert line.startswith("human_samples")
    assert "Biopsy" in line and "Clinical Trial, Phase I" in line
    assert cases.design_line(None) == "no Europe PMC record"


# --- verification ----------------------------------------------------------------------------


def _mini_cases(pmid="11"):
    return {"x": {"target": "T", "anchors": [pmid],
                  "milestones": [{"pmid": pmid, "role": "approval_trial", "author": "Chapman", "year": 2011,
                                  "title_kw": ["vemurafenib"], "label": "Chapman 2011"}],
                  "timeline": {"discovery": pmid, "first_selective_compound": pmid,
                               "first_in_human": pmid, "approval_trial": pmid}}}


def test_verify_accepts_correct_pmid(tmp_path):
    fake = FakeEPMC([_item(11, 2011, "Chapman", "Improved survival with vemurafenib")])
    c, corr, rows = cases.verify_cases(_mini_cases(), getter=fake, cache_dir=tmp_path)
    assert corr == [] and rows[0]["verified"]
    assert c["x"]["anchors"] == ["11"]


def test_verify_corrects_wrong_pmid(tmp_path):
    fake = FakeEPMC([_item(11, 2009, "Jones", "Something else"),
                     _item(22, 2011, "Chapman", "Improved survival with <i>vemurafenib</i>")])
    c, corr, rows = cases.verify_cases(_mini_cases(), getter=fake, cache_dir=tmp_path)
    assert corr[0]["old"] == "11" and corr[0]["new"] == "22"
    assert "year" in corr[0]["reason"]
    assert c["x"]["anchors"] == ["22"] and c["x"]["timeline"]["approval_trial"] == "22"
    assert rows[0]["verified"]


def test_verify_leaves_unresolved_flagged(tmp_path):
    fake = FakeEPMC([])
    c, corr, rows = cases.verify_cases(_mini_cases(), getter=fake, cache_dir=tmp_path)
    assert corr == [] and not rows[0]["verified"]


def test_fetch_epmc_batches_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(cases, "EPMC_BATCH", 2)
    fake = FakeEPMC([_item(i, 2000) for i in (1, 2, 3)])
    got = cases.fetch_epmc(["3", "1", "2", "4"], getter=fake, cache_dir=tmp_path)
    assert set(got) == {"1", "2", "3"} and len(fake.calls) == 2
    cases.fetch_epmc(["3", "1", "2", "4"], getter=fake, cache_dir=tmp_path)
    assert len(fake.calls) == 2  # cached


# --- end to end ----------------------------------------------------------------------------------


def test_run_end_to_end(tmp_path):
    table = _table()
    table["100"]["year"] = 2011
    items = [
        _item(100, 2011, "Chapman", "vemurafenib trial", mesh=("Humans", "Adult"), pub_types=("Clinical Trial, Phase III",)),
        _item(1, 2005, mesh=("Humans", "Mice, Nude")),
        _item(2, 2011, mesh=("Humans", "Cell Line, Tumor")),
        _item(10, 2000, mesh=()),  # no MeSH -> unclassified
        # 12 absent from Europe PMC -> unclassified
    ]
    fake = FakeEPMC(items)
    c = _mini_cases("100")
    c["x"]["milestones"][0]["title_kw"] = ["vemurafenib"]
    papers = [{"pmid": "500", "year": 2005}, {"pmid": "501", "year": 2010}, {"pmid": "502", "year": 2014},
              {"pmid": "503", "year": 2006}]
    feats = [
        {"pmid": "500", "group": "preclinical", "model_system": "animal", "eligible": True},
        {"pmid": "501", "group": "clinical", "model_system": "human_samples", "eligible": True},
        {"pmid": "502", "group": "preclinical", "model_system": "cell_only", "eligible": True},  # outside years
        {"pmid": "503", "group": "preclinical", "model_system": "cell_only", "eligible": False},
    ]
    result, rows = cases.run(cases=c, getter=fake, cache_dir=tmp_path, papers=papers,
                             features_rows=feats, icite_fetch=_icite_fetch(table))
    x = result["cases"]["x"]
    assert x["counts"]["1"] == {"n": 2, "human_samples": 0, "animal": 1, "cell_only": 1, "other": 0,
                                "non_primary": 0, "unclassified": 0}
    assert x["counts"]["2"]["unclassified"] == 2
    assert x["lineage_year_range"] == [1999, 2011]
    # base corpus years 2005-2014 (eligible rows); window = [2005, 2011]
    cmp_ = x["comparison"]
    assert cmp_["year_range"] == [2005, 2011]
    allg = cmp_["by_depth"]["all"]["all"]
    assert allg["base"]["n"] == 2 and allg["lineage"]["n_classified"] == 2
    assert allg["diff_lineage_minus_base"]["cell_only"]["diff"] == pytest.approx(0.5)
    pre = cmp_["by_depth"]["all"]["preclinical"]
    assert pre["base"]["n"] == 1
    assert x["milestones"][0]["depth"] == "anchor"
    assert x["timeline"]["approval_trial"]["year"] == 2011
    assert result["combined"]["counts"]["all"]["n"] == 4
    assert {r["pmid"] for r in rows} == {"1", "2", "10", "12"}
    assert cases.summary_lines(result)
