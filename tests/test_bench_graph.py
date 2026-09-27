from src.bench.graph import (
    BUDGET,
    _search_through_horizon,
    assemble_claim,
    combine_routes,
    judgment_export,
    rank_text,
    score_yield,
    supporting_passage,
    take_budget,
)
from src.encode.encoder import _hash_embed


def _paper(pmid, year, abstract, pub_types=None, title=None):
    return {
        "pmid": pmid,
        "year": year,
        "date": f"{year}-06-01",
        "title": title or f"Paper {pmid}",
        "abstract": abstract,
        "pub_types": pub_types or ["Journal Article"],
    }


def _claim():
    return {
        "claim_id": "CX",
        "pmid": "100",
        "cutoff": "2010-06-01",
        "stratum": "agreed_unresolved",
        "in_labeled_eval": True,
        "spec": {
            "gene": "KRAS",
            "intervention": "intervention X",
            "setting": "setting Z",
            "outcome": "reduces outcome Y",
            "direction": "positive",
            "quote": "intervention X reduces outcome Y in setting Z",
            "setting_terms": ["setting Z"],
            "focus_terms": ["outcome Y"],
            "conditions": [],
        },
    }


def test_budget_stops_at_twenty_and_holds_reviews_out():
    claim = _claim()
    papers = [_paper(str(i), 2011, f"KRAS result number {i}") for i in range(30)]
    papers.insert(3, _paper("900", 2012, "KRAS reviewed", ["Review"], "A review"))
    kept, reviews = take_budget(papers, claim)
    assert len(kept) == BUDGET
    assert [row["pmid"] for row in reviews] == ["900"]
    assert all(row["pmid"] != "900" for row in kept)


def test_citation_can_keep_a_paper_that_omits_the_gene_name():
    claim = _claim()
    other = _paper("7", 2011, "This assay measures calcium flux in hepatocytes.")
    kept, _reviews = take_budget([other], claim, require_gene=False)
    assert [row["pmid"] for row in kept] == ["7"]
    kept_gene, _ = take_budget([other], claim, require_gene=True)
    assert kept_gene == []


def test_combination_prefers_a_paper_found_by_more_routes():
    multi = _paper("1", 2011, "KRAS")
    single = _paper("2", 2011, "KRAS")
    ranked = combine_routes({"citation": [single, multi], "experimental": [multi]})
    assert ranked[0]["pmid"] == "1"


def test_illustrative_neighborhood_separates_citer_test_and_review():
    claim = _claim()
    citer = _paper(
        "200",
        2011,
        "This KRAS study cites earlier work but measures outcome Q in an unrelated assay.",
        title="Different endpoint",
    )
    independent = _paper(
        "300",
        2012,
        "In setting Z, intervention X reduces outcome Y. KRAS was the measured gene.",
        title="Independent test",
    )
    review = _paper(
        "400",
        2013,
        "This review repeats the conclusion that intervention X reduces outcome Y.",
        ["Review"],
        title="Review of the finding",
    )
    early = _paper("500", 2010, "KRAS signaling in fibroblasts without that intervention.")
    early["date"] = "2010-08-01"
    fetched = {
        "same_gene": [early, citer, independent],
        "experimental": [independent],
        "citation": [citer, review],
        "shared_references": [],
        "text_pool": [independent, early],
    }
    assembled = assemble_claim(claim, fetched, packet_papers=[early], encode=_hash_embed)
    assert "300" in assembled["lists"]["experimental"]
    assert "200" in assembled["lists"]["citation"]
    assert "400" not in assembled["lists"]["citation"]
    assert "400" not in assembled["lists"]["combination"]
    assert any(row["pmid"] == "400" for row in assembled["reviews"])
    neighbor = next(row for row in assembled["neighbors"] if row["pmid"] == "300")
    experimental = next(rel for rel in neighbor["relationships"] if rel["type"] == "experimental")
    assert "outcome Y" in experimental["passage"] or "intervention X" in experimental["passage"]
    assert len(assembled["lists"]["combination"]) <= BUDGET
    blinded, provenance = judgment_export([assembled])
    candidate = blinded["claims"][0]["candidates"][0]
    assert "methods" not in candidate
    assert "relationships" not in candidate
    assert provenance["rows"]


def test_similarity_ranks_the_claim_wording_above_an_unrelated_abstract():
    close = _paper("1", 2011, "In setting Z, intervention X reduces outcome Y in a KRAS model.")
    far = _paper("2", 2011, "Zebrafish fin regeneration after surgical amputation of the caudal fin.")
    ranked = rank_text("intervention X reduces outcome Y in setting Z", [far, close], _hash_embed)
    assert ranked[0][0]["pmid"] == "1"
    assert ranked[0][1] > ranked[1][1]


def test_date_scan_continues_past_papers_before_the_cutoff():
    claim = _claim()

    def item(pmid, date):
        return {
            "pmid": pmid,
            "firstPublicationDate": date,
            "pubYear": date[:4],
            "title": "KRAS study",
            "abstractText": "A KRAS experiment.",
            "pubTypeList": {"pubType": ["Journal Article"]},
        }

    pages = {
        "*": {"nextCursorMark": "page-2", "resultList": {"result": [item("1", "2010-01-01"), item("2", "2010-02-01")]}},
        "page-2": {"nextCursorMark": "page-3", "resultList": {"result": [item("3", "2010-08-01"), item("4", "2011-01-01")]}},
    }

    class Client:
        def search(self, params):
            return pages[params["cursorMark"]]

    papers = _search_through_horizon(Client(), "KRAS", "P_PDATE_D asc", claim, require_gene=True, budget=2, page_size=2)
    assert [paper["pmid"] for paper in papers] == ["1", "2", "3", "4"]
    kept, _reviews = take_budget(papers, claim, budget=2)
    assert [paper["pmid"] for paper in kept] == ["3", "4"]


def test_yield_counts_a_direct_test_the_packet_did_not_contain():
    claim = _claim()
    claim["lists"] = {
        "same_gene": ["500"],
        "text_similarity": ["300"],
        "citation": ["200"],
        "experimental": ["300"],
        "combination": ["300"],
        "shared_references": [],
    }
    claim["packet_pmids"] = ["500"]
    scored = score_yield([claim], {"CX": {"300": "support", "500": "support"}})
    assert scored["methods"]["experimental"]["claims_with_a_direct_test"] == 1
    assert scored["methods"]["experimental"]["direct_tests"] == 1
    assert scored["methods"]["citation"]["direct_tests"] == 0
    assert scored["methods"]["packet"]["direct_tests"] == 1
    assert scored["methods"]["combination"]["claims_with_a_contradiction"] == 0


def test_passage_returns_the_sentence_that_carries_the_term():
    paper = _paper("1", 2011, "Background is long. intervention X reduces outcome Y in setting Z. More background.")
    assert "outcome Y" in supporting_passage(paper, ["outcome Y"])
