from src.bench.sample import is_source_candidate, select_sources


def _paper(**kwargs):
    base = {
        "pmid": "1",
        "year": 2010,
        "date": "2010-03-01",
        "title": "KRAS mutation and survival",
        "abstract": "A" * 400 + " KRAS mutation was associated with shorter survival in colorectal cancer.",
        "genes": ["KRAS"],
        "drugs": [],
        "diseases": ["colorectal cancer"],
        "pub_types": ["Journal Article"],
        "journal": "J",
    }
    base.update(kwargs)
    return base


def test_reviews_and_later_papers_are_not_sources():
    assert is_source_candidate(_paper())
    assert not is_source_candidate(_paper(year=2021, pmid="2"))
    assert not is_source_candidate(_paper(pub_types=["Review"], pmid="3"))
    assert not is_source_candidate(_paper(abstract="Too short.", pmid="4"))


def test_sample_is_deterministic_and_balanced():
    papers = []
    for year in range(2006, 2015):
        for index in range(12):
            papers.append(
                _paper(
                    pmid=str(year * 100 + index),
                    year=year,
                    title=f"Finding {index}",
                )
            )
    papers.append(_paper(pmid="99999", year=2020))
    first = select_sources(papers, per_year=8, seed=20260927)
    second = select_sources(papers, per_year=8, seed=20260927)
    assert [row["pmid"] for row in first] == [row["pmid"] for row in second]
    assert all(2006 <= row["year"] <= 2014 for row in first)
    assert len(first) == 8 * 9
    assert "99999" not in {row["pmid"] for row in first}
    assert "abstract" in first[0]
    assert "followups" not in first[0]
