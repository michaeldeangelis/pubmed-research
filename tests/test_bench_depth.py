import numpy as np

from src.bench.depth import (
    BUDGET,
    DEPTHS,
    KNOWN_MISS,
    content_query,
    counted_tests,
    coverage_row,
    eligible_in_order,
    known_miss_status,
    rank_pool,
    returned_twenty,
)
from src.bench.mechanism import mechanism_query


def _paper(pmid, year, abstract, pub_types=None, title=None, date=None):
    return {
        "pmid": pmid,
        "year": year,
        "date": date or f"{year}-06-01",
        "title": title or f"Paper {pmid}",
        "abstract": abstract,
        "pub_types": pub_types or ["Journal Article"],
    }


def _claim():
    return {
        "claim_id": "CX",
        "pmid": "100",
        "cutoff": "2010-06-01",
        "gene": "KRAS",
        "population": "setting Z",
        "exposure": "TARGET",
        "effect": "reduces outcome Y",
        "direction": "positive",
        "quote": "TARGET reduces outcome Y",
        "diseases": [],
        "drugs": [],
    }


def _encode(texts):
    rows = []
    for text in texts:
        rows.append([1.0, 0.0] if "TARGET" in text else [0.0, 1.0])
    return np.asarray(rows, dtype=np.float32)


def test_depth_forty_cannot_see_a_later_hit_and_two_hundred_can():
    hits = [_paper(str(i), 2012, f"KRAS unrelated assay {i}") for i in range(80)]
    hits[3] = _paper("900", 2012, "TARGET KRAS reviewed", ["Review"], "A review")
    hits[55] = _paper("361", 2012, "TARGET KRAS in setting Z reduces outcome Y")
    claim = _claim()
    shallow = returned_twenty(rank_pool(claim, hits, _encode, 40))
    deeper = returned_twenty(rank_pool(claim, hits, _encode, 200))
    assert "361" not in [paper["pmid"] for paper, _score in shallow]
    assert "900" not in [paper["pmid"] for paper, _score in deeper]
    assert [paper["pmid"] for paper, _score in deeper][0] == "361"
    assert len(shallow) <= BUDGET
    assert len(deeper) <= BUDGET


def test_known_miss_status_is_separate_from_fresh_coverage():
    hits = [_paper(str(i), 2012, "KRAS other") for i in range(50)]
    hits[10] = _paper(KNOWN_MISS, 2012, "TARGET KRAS")
    claim = _claim()
    ranked = {depth: rank_pool(claim, hits, _encode, depth) for depth in DEPTHS}
    status = known_miss_status(hits, ranked)
    assert status["raw_index"] == 10
    assert status["in_returned_twenty"]["40"] is True
    assert coverage_row(["C025"], {"C025": []})["claims_with_an_eligible_direct_test"] == 0


def test_content_query_stays_the_text_query():
    claim = _claim()
    claim["exposure"] = "tissue-specific Atg5 inactivation"
    assert "Atg5" not in content_query(claim)
    assert "Atg5" in mechanism_query(claim)


def test_counted_tests_drop_uncertain_order_and_a_second_copy():
    papers = {
        "1": _paper("1", 2010, "KRAS cohort alpha", date="2010-01-01", title="Same title"),
        "2": _paper("2", 2011, "KRAS cohort alpha", title="Same title"),
        "3": _paper("3", 2011, "KRAS cohort beta", title="Other title"),
    }
    papers["2"]["abstract"] = papers["1"]["abstract"]
    judgments = {"1": "support", "2": "support", "3": "support", "4": "not_test"}
    counted = counted_tests(["1", "2", "3", "4"], papers, judgments, "2010-06-01")
    assert counted == ["3"]


def test_a_review_is_not_eligible():
    claim = _claim()
    hits = [_paper("900", 2012, "TARGET KRAS", ["Review"])]
    assert eligible_in_order(hits, claim) == []
