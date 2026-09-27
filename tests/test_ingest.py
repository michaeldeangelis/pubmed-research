from src.ingest.pubmed import parse_europepmc_result


SAMPLE = {
    "pmid": "34590053",
    "title": "Sotorasib in <i>KRAS</i> G12C NSCLC",
    "abstractText": "Phase II patients with NSCLC were treated.",
    "firstPublicationDate": "2021-08-02",
    "pubYear": "2021",
    "journalInfo": {"journal": {"title": "Example Journal"}},
    "meshHeadingList": {"meshHeading": [{"descriptorName": "Lung Neoplasms"}]},
    "pubTypeList": {"pubType": ["Journal Article"]},
}


def test_parse_core_record():
    paper = parse_europepmc_result(SAMPLE)
    assert paper["pmid"] == "34590053"
    assert paper["year"] == 2021
    assert paper["date"] == "2021-08-02"
    assert "<i>" not in paper["title"]
    assert "KRAS" in paper["title"]
    assert paper["journal"] == "Example Journal"
    assert "Lung Neoplasms" in paper["mesh"]
    assert "Journal Article" in paper["pub_types"]
    assert "Phase II" in paper["abstract"]
