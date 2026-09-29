"""Unit tests for src.meta.impact on synthetic data (no network)."""

import json

import pytest

from src.meta import impact


def test_window_citers_inclusive_bounds_and_undated():
    by_year = [{"1": 2004}, {"2": 2005}, {"3": 2010}, {"4": 2011}, {"5": None}, {"2": 2005}]
    citers, undated = impact.window_citers(by_year, 2005)
    assert citers == ["2", "3"]
    assert undated == 1
    assert impact.window_citers(by_year, None) == ([], 0)


def test_disruption_nok_values():
    focal_refs = {"r1", "r2"}
    citers = {"a": {"x"}, "b": {"r1", "x"}, "c": set(), "d": {"r2"}}
    res = impact.disruption_nok(focal_refs, citers)
    assert res == {"cd": (2 - 2) / 4, "n_i": 2, "n_j": 2}
    assert impact.disruption_nok(focal_refs, {"a": {"x"}})["cd"] == 1.0
    assert impact.disruption_nok(focal_refs, {"a": {"r1"}})["cd"] == -1.0


def test_disruption_nok_nulls():
    assert impact.disruption_nok(set(), {"a": {"x"}})["cd"] is None
    assert impact.disruption_nok({"r"}, {})["cd"] is None


def test_pick_openalex_work_deterministic():
    works = [{"id": "W2", "referenced_works": ["a"]},
             {"id": "W1", "referenced_works": ["a"]},
             {"id": "W3", "referenced_works": []}]
    assert impact.pick_openalex_work(works)["id"] == "W1"
    assert impact.pick_openalex_work([]) is None
    assert impact.pmid_from_openalex({"ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/123"}}) == "123"


def test_validation_order_fixed():
    a = impact.validation_order(["3", "1", "2", "10"])
    b = impact.validation_order(["10", "2", "1", "3"])
    assert a == b and sorted(a) == ["1", "10", "2", "3"]


def test_fetch_and_compute_icite_cd(tmp_path):
    papers = [{"pmid": "100", "year": 2005}, {"pmid": "200", "year": 2005},
              {"pmid": "300", "year": 2005}, {"pmid": "400", "year": 2005}]
    icite = {
        # focal 100: refs r1,r2; citers c1 (disrupting), c2 (consolidating), c9 late
        "100": {"references": [1, 2], "citedByPmidsByYear": [{"11": 2006}, {"12": 2010}, {"19": 2012}]},
        "200": {"references": [], "citedByPmidsByYear": [{"11": 2006}]},  # no refs -> null
        "300": {"references": [1], "citedByPmidsByYear": [{"19": 2012}]},  # no 5y citers -> null
        # 400 absent from iCite
    }
    store = {"11": [5], "12": [2, 7]}
    calls = []

    def fake_fetch(pmids):
        calls.append(list(pmids))
        return [{"pmid": int(p), "year": 2006, "references": store[p]} for p in pmids if p in store]

    cache = impact.JsonlCache(tmp_path / "c.jsonl")
    impact.fetch_citer_refs(["11", "12", "13"], cache, fetch=fake_fetch, workers=1, batch=2)
    assert cache.get("13") == {"pmid": "13", "missing": True}
    # resumable: second call fetches nothing
    impact.fetch_citer_refs(["11", "12", "13"], impact.JsonlCache(tmp_path / "c.jsonl"),
                            fetch=fake_fetch, workers=1)
    assert len(calls) == 2

    out = impact.compute_icite_cd(papers, icite, impact.JsonlCache(tmp_path / "c.jsonl"))
    assert out["100"]["cd"] == 0.0 and out["100"]["n_i"] == 1 and out["100"]["n_j"] == 1
    assert out["100"]["n_citers_5y"] == 2
    assert out["200"]["cd"] is None and out["200"]["n_refs_icite"] == 0
    assert out["300"]["cd"] is None and out["300"]["n_citers_5y"] == 0
    assert out["400"]["cd"] is None

    rows = impact.build_rows(papers, {"100": {"relative_citation_ratio": 1.5}}, out, None, {}, None)
    r100 = rows[0]
    assert r100["cd_source"] == "icite_nok" and r100["rcr"] == 1.5 and r100["n_refs_openalex"] is None
    assert rows[1]["cd_source"] is None and rows[1]["cd"] is None
    for key in ("pmid", "rcr", "cd_source", "cd", "n_refs_openalex"):
        assert key in r100


class FakeOA:
    """Scripted OpenAlex responses keyed on the filter string."""

    def __init__(self, remaining=100):
        self.remaining = remaining
        self.calls = []

    def __call__(self, params):
        self.calls.append(params)
        self.remaining -= 1
        f = params["filter"]
        headers = {"x-ratelimit-remaining": str(self.remaining)}
        if f.startswith("ids.pmid:"):
            res = []
            for p in f.split(":", 1)[1].split("|"):
                if p == "1":
                    res.append({"id": "https://openalex.org/W1", "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/1"},
                                "publication_year": 2005, "referenced_works": ["https://openalex.org/R1", "https://openalex.org/R2"]})
            return {"results": res, "meta": {"count": len(res)}}, headers
        if f.startswith("cites:W1"):
            assert "publication_year:2005-2010" in f
            if params["cursor"] == "*":
                return {"results": [{"id": "https://openalex.org/C1", "referenced_works": ["https://openalex.org/R1"]}],
                        "meta": {"count": 2, "next_cursor": "n2"}}, headers
            return {"results": [{"id": "https://openalex.org/C2", "referenced_works": []}],
                    "meta": {"count": 2, "next_cursor": None}}, headers
        raise AssertionError(f)


def test_openalex_focal_and_validation(tmp_path):
    fake = FakeOA()
    oa = impact.OpenAlex(tmp_path, get=fake)
    focal = oa.focal(["1", "2"])
    assert focal.get("1")["referenced_works"] == ["R1", "R2"]
    assert focal.get("2") == {"pmid": "2", "found": False}
    papers = [{"pmid": "1", "year": 2005}, {"pmid": "2", "year": 2005}]
    out, info = impact.compute_openalex_validation(papers, {"1", "2"}, focal, oa)
    assert out["1"]["cd"] == 0.0 and out["1"]["n_i"] == 1 and out["1"]["n_j"] == 1
    assert info["completed"] == 1 and not out["1"]["capped"]
    n = len(fake.calls)
    # cached: no new calls
    impact.OpenAlex(tmp_path, get=fake).focal(["1", "2"])
    impact.compute_openalex_validation(papers, {"1"}, focal, impact.OpenAlex(tmp_path, get=fake))
    assert len(fake.calls) == n


def test_openalex_budget_stops(tmp_path):
    fake = FakeOA(remaining=impact.OPENALEX_RESERVE + 1)
    oa = impact.OpenAlex(tmp_path, get=fake)
    oa.focal(["1"])  # uses the last credit above the reserve
    with pytest.raises(impact.Budget):
        oa.call({"filter": "ids.pmid:9"})


def test_sciscinet_subset_join(tmp_path):
    lines = {
        "link": ["0\t2", "55\t100", "66\t999"],
        "papers": ["PaperID\tDOI\tDocType\tYear\tDisruption",
                   "55\t10.1/x\tJournal\t2005.0\t-0.01",
                   "77\t10.1/Y\tJournal\t2006.0\t0.2",
                   "88\t\tJournal\t2006.0\t"],
    }

    def stream(url):
        return iter(lines["link"] if url.endswith("df_magid_pubmedid.tsv") else lines["papers"])

    out = impact.sciscinet_subset({"100", "200"}, {"100": None, "200": "10.1/y"}, tmp_path, stream=stream)
    assert out["100"] == {"mag_id": "55", "join": "pmid", "sciscinet_d": -0.01}
    assert out["200"]["join"] == "doi" and out["200"]["sciscinet_d"] == 0.2
    assert json.loads((tmp_path / "papers_subset.json").read_text()) == out
