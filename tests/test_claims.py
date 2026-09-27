from src.claims.extract import enrich_paper


def _paper(**kwargs):
    base = {
        "pmid": "1",
        "date": "2021-06-01",
        "year": 2021,
        "title": "",
        "abstract": "",
        "journal": "J",
        "mesh": [],
        "pub_types": [],
    }
    base.update(kwargs)
    return base


def test_enrich_links_gene_drug_disease_and_relation():
    paper = _paper(
        title="Sotorasib inhibits KRAS G12C in NSCLC",
        abstract="The inhibitor reduced signaling in patients with lung adenocarcinoma.",
    )
    out = enrich_paper(paper)
    assert "KRAS" in out["genes"]
    assert "sotorasib" in out["drugs"]
    assert "NSCLC" in out["diseases"]
    assert any(rel["rel"] == "inhibits" and rel["head"] == "sotorasib" and rel["tail"] == "KRAS" for rel in out["relations"])


def test_aliases_and_held_out_tags():
    paper = _paper(
        title="MEK1 blockade",
        abstract="A germline MAP2K1 variant and acquired resistance in a melanoma xenograft model.",
    )
    out = enrich_paper(paper)
    assert "MAP2K1" in out["genes"]
    assert "melanoma" in out["diseases"]
    assert "human_genetics" in out["evidence_tags"]
    assert "resistance" in out["evidence_tags"]
    assert "animal_model" in out["evidence_tags"]


def test_enrich_does_not_drop_source_fields():
    paper = _paper(pmid="99", title="BRAF")
    out = enrich_paper(paper)
    assert out["pmid"] == "99"
    assert "year" in out
