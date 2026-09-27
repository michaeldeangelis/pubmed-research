from src.bench.agreement import claim_status, cohens_kappa
from src.bench.followups import followups_for


def test_followups_stay_inside_the_horizon_and_split_reviews():
    claim = {
        "pmid": "1",
        "cutoff": "2010-06-01",
        "gene": "KRAS",
        "diseases": ["NSCLC"],
        "drugs": [],
        "skip": False,
    }
    papers = [
        {"pmid": "1", "year": 2010, "date": "2010-06-01", "genes": ["KRAS"], "diseases": ["NSCLC"], "drugs": [], "title": "source", "abstract": "NSCLC", "pub_types": []},
        {"pmid": "2", "year": 2009, "date": "2009-01-01", "genes": ["KRAS"], "diseases": ["NSCLC"], "drugs": [], "title": "earlier", "abstract": "NSCLC", "pub_types": []},
        {"pmid": "3", "year": 2012, "date": "2012-01-01", "genes": ["KRAS"], "diseases": ["NSCLC"], "drugs": [], "title": "later", "abstract": "NSCLC study", "pub_types": ["Journal Article"]},
        {"pmid": "4", "year": 2013, "date": "2013-01-01", "genes": ["KRAS"], "diseases": ["NSCLC"], "drugs": [], "title": "review", "abstract": "NSCLC review", "pub_types": ["Review"]},
        {"pmid": "5", "year": 2020, "date": "2020-01-01", "genes": ["KRAS"], "diseases": ["NSCLC"], "drugs": [], "title": "late", "abstract": "NSCLC", "pub_types": []},
        {"pmid": "6", "year": 2012, "date": "2012-05-01", "genes": ["BRAF"], "diseases": ["NSCLC"], "drugs": [], "title": "other", "abstract": "NSCLC", "pub_types": []},
    ]
    got = followups_for(claim, papers)
    assert [row["pmid"] for row in got["articles"]] == ["3"]
    assert [row["pmid"] for row in got["reviews"]] == ["4"]


def test_unresolved_when_nothing_directly_tests_the_claim():
    assert claim_status([{"relation": "not_direct", "independent": True}]) == "unresolved"
    assert claim_status([{"relation": "support", "independent": False}]) == "unresolved"
    assert claim_status([
        {"relation": "support", "independent": True},
        {"relation": "contradict", "independent": True},
    ]) == "mixed"


def test_kappa_is_one_when_labels_match():
    labels = ["support", "unresolved", "contradict", "support"]
    assert cohens_kappa(labels, labels) == 1.0
