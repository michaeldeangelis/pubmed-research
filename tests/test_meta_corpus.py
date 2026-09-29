"""Synthetic tests for src.meta.corpus (no network)."""

from __future__ import annotations

import json

from src.meta import corpus


def _epmc_item(pmid, year="2005", n_auth=3, doi="10.1/x"):
    return {
        "pmid": str(pmid),
        "doi": doi,
        "pubYear": year,
        "title": "KRAS <i>mutant</i> study",
        "abstractText": "Abstract text.",
        "journalInfo": {"journal": {"title": "J Test"}},
        "meshHeadingList": {"meshHeading": [{"descriptorName": "Humans"}, {"descriptorName": "Mice"}]},
        "pubTypeList": {"pubType": ["Journal Article"]},
        "authorList": {"author": [{"fullName": f"A{i}"} for i in range(n_auth)]},
    }


def test_parse_paper_fields():
    row = corpus.parse_paper(_epmc_item(123, n_auth=4))
    assert row == {
        "pmid": "123",
        "doi": "10.1/x",
        "year": 2005,
        "journal": "J Test",
        "title": "KRAS mutant study",
        "abstract": "Abstract text.",
        "mesh": ["Humans", "Mice"],
        "pub_types": ["Journal Article"],
        "n_authors": 4,
    }


def test_parse_paper_missing_authors_and_doi():
    item = _epmc_item(5, doi=None)
    del item["authorList"]
    row = corpus.parse_paper(item)
    assert row["n_authors"] == 0 and row["doi"] is None


def test_query_uses_year_range():
    assert "PUB_YEAR:[2000 TO 2015]" in corpus.QUERY and "SRC:MED" in corpus.QUERY


def test_fetch_papers_cursor_paging_dedup_and_cache(tmp_path):
    pages = {
        "*": {"hitCount": 3, "nextCursorMark": "c1", "resultList": {"result": [_epmc_item(30), _epmc_item(10)]}},
        "c1": {"nextCursorMark": "c2", "resultList": {"result": [_epmc_item(10), _epmc_item(20), {"pmid": None}]}},
        "c2": {"nextCursorMark": "c2", "resultList": {"result": []}},
    }
    calls = []

    def getter(url, params):
        calls.append(params["cursorMark"])
        assert params["pageSize"] == 1000 and params["resultType"] == "core"
        return pages[params["cursorMark"]]

    papers, hits = corpus.fetch_papers(getter=getter, cache_dir=tmp_path)
    assert [p["pmid"] for p in papers] == ["10", "20", "30"]
    assert hits == 3
    assert calls == ["*", "c1", "c2"]

    def boom(url, params):
        raise AssertionError("network used on cached rerun")

    again, _ = corpus.fetch_papers(getter=boom, cache_dir=tmp_path)
    assert again == papers


def test_fetch_icite_batches_of_200_and_pmid_str(tmp_path):
    sizes = []

    def getter(url, params):
        ids = params["pmids"].split(",")
        sizes.append(len(ids))
        return {"data": [{"pmid": int(i), "year": 2001, "extra": [1]} for i in ids]}

    recs = corpus.fetch_icite([str(i) for i in range(1, 451)], getter=getter, cache_dir=tmp_path)
    assert sizes == [200, 200, 50]
    assert len(recs) == 450
    assert all(isinstance(r["pmid"], str) for r in recs)
    assert recs[0]["extra"] == [1]


def test_citing_years_cache_and_unknown(tmp_path):
    calls = []

    def getter(url, params):
        calls.append(params)
        assert params["fl"] == "pmid,year"
        return {"data": [{"pmid": 1, "year": 2003}, {"pmid": 2, "year": 2010}]}

    path = tmp_path / "citing_years.json"
    got = corpus.citing_years(["1", "2", "3"], getter=getter, cache_path=path, cache_dir=tmp_path)
    assert got == {"1": 2003, "2": 2010, "3": None}
    assert json.loads(path.read_text()) == {"1": 2003, "2": 2010, "3": None}
    got2 = corpus.citing_years(["1", "3"], getter=None, cache_path=path, cache_dir=tmp_path)
    assert got2 == {"1": 2003, "3": None}
    assert len(calls) == 1


def test_outcome_b_window_edges():
    # paper 2005: window [2004, 2013]
    assert corpus.outcome_b_row("1", 2005, [2004]) == {
        "pmid": "1", "clin_cited_8y": 1, "n_clin_8y": 1, "first_clin_year": 2004,
    }
    assert corpus.outcome_b_row("1", 2005, [2013, 2014])["n_clin_8y"] == 1
    r = corpus.outcome_b_row("1", 2005, [2003, 2014, None])
    assert r["clin_cited_8y"] == 0 and r["n_clin_8y"] == 0 and r["first_clin_year"] == 2003
    r = corpus.outcome_b_row("1", 2005, [])
    assert r == {"pmid": "1", "clin_cited_8y": 0, "n_clin_8y": 0, "first_clin_year": None}


def test_compute_outcome_b_research_only_and_year_fallback():
    papers = [{"pmid": "1", "year": 2005}, {"pmid": "2", "year": 2005}, {"pmid": "3", "year": None}]
    icite = [
        {"pmid": "1", "year": 2005, "is_research_article": True, "cited_by_clin": [11, 12]},
        {"pmid": "2", "year": 2005, "is_research_article": False, "cited_by_clin": [11]},
        {"pmid": "3", "year": 2010, "is_research_article": True, "cited_by_clin": [12]},
    ]
    years = {"11": 2007, "12": 2020}
    rows, stats = corpus.compute_outcome_b(papers, icite, years)
    assert [r["pmid"] for r in rows] == ["1", "3"]
    assert rows[0] == {"pmid": "1", "clin_cited_8y": 1, "n_clin_8y": 1, "first_clin_year": 2007}
    # paper 3 falls back to iCite year 2010: window [2009, 2018], 2020 is outside
    assert rows[1]["clin_cited_8y"] == 0 and rows[1]["first_clin_year"] == 2020
    assert stats["year_from_icite"] == 1


def test_write_read_jsonl_roundtrip(tmp_path):
    path = tmp_path / "x.jsonl"
    assert corpus.write_jsonl(path, [{"a": 1}, {"b": "é"}]) == 2
    assert corpus.read_jsonl(path) == [{"a": 1}, {"b": "é"}]
