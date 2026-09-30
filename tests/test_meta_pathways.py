"""Metascience v2 pathway config: queries, case-sensitive patterns, wiring."""

import pytest

from src.meta import corpus, features as F, impact, pathways as P


def paper(pmid="1", mesh=("Humans",), title="", abstract="", pub_types=("Journal Article",)):
    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "mesh": list(mesh),
        "pub_types": list(pub_types),
        "n_authors": 3,
    }


def hits(pattern, text):
    return bool(pattern.search(text))


# --- queries ------------------------------------------------------------------------


def test_queries_use_exactly_the_ledger_terms():
    assert P.PATHWAYS["egfr"]["terms"] == (
        "EGFR", "ERBB2", "HER2", "ERBB3", "erlotinib", "gefitinib", "osimertinib",
        "afatinib", "lapatinib", "cetuximab", "panitumumab", "trastuzumab",
    )
    assert P.PATHWAYS["pi3k"]["terms"] == (
        "PIK3CA", "PTEN", "AKT1", "MTOR", "mTORC1", "everolimus", "temsirolimus",
        "alpelisib", "idelalisib",
    )
    q = P.PATHWAYS["pi3k"]["query"]
    assert q.startswith('(TITLE_ABS:"PIK3CA" OR TITLE_ABS:"PTEN" OR ')
    assert q.endswith(') AND PUB_YEAR:[2000 TO 2015] AND SRC:MED')
    assert q.count("TITLE_ABS:") == 9
    assert P.PATHWAYS["egfr"]["query"].count("TITLE_ABS:") == 12


def test_gene_group_levels_and_order():
    assert [n for n, _ in P.PATHWAYS["egfr"]["gene_groups"]] == ["EGFR", "ERBB2"]
    assert P.PATHWAYS["egfr"]["gene_group_levels"] == ("EGFR", "ERBB2", "other")
    assert [n for n, _ in P.PATHWAYS["pi3k"]["gene_groups"]] == ["PIK3CA", "PTEN", "AKT", "MTOR"]
    assert P.PATHWAYS["pi3k"]["gene_group_levels"][-1] == "other"
    with pytest.raises(KeyError):
        P.get("ras")


# --- EGFR/ERBB patterns ----------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "EGFR mutations", "mouse Egfr", "EGFRvIII", "pEGFR levels", "EGFRs", "EGFRi therapy",
    "ResultsEGFR was", "ErbB1", "HER1", "HER-1", "c-erbB-1", "EGFR-TKI", "EGFR(L858R)",
])
def test_egfr_positive(text):
    assert hits(P.EGFR_PATTERN, text)


@pytest.mark.parametrize("text", [
    "eGFR declined", "estimated egfr", "EGFRx", "AEGFR", "HER10",
    "EGFR1",
])
def test_egfr_negative(text):
    assert not hits(P.EGFR_PATTERN, text)


@pytest.mark.parametrize("text", [
    "HER2-positive", "HER2/neu", "Her2 amplification", "HER-2", "ERBB2", "ErbB2", "ErbB-2",
    "erbB2", "c-erbB-2", "Erbb2 mice", "HER2+", "pHER2",
])
def test_her2_positive(text):
    assert hits(P.ERBB2_PATTERN, text)


@pytest.mark.parametrize("text", [
    "neu oncogene", "rat neu", "her2nd visit", "her2 lowercase", "OTHER2", "HER21",
    "Hera2", "HER22",
])
def test_her2_negative(text):
    # "neu" alone, all-lower-case "her2", and longer tokens do not count
    assert not hits(P.ERBB2_PATTERN, text)


def test_erbb3_and_drugs():
    assert hits(P.ERBB3_PATTERN, "HER3 signalling") and hits(P.ERBB3_PATTERN, "ErbB-3")
    assert not hits(P.ERBB3_PATTERN, "HER4") and not hits(P.ERBB3_PATTERN, "her3")
    assert hits(P.EGFR_DRUG_PATTERN, "Gefitinib-resistant cells")
    assert hits(P.EGFR_DRUG_PATTERN, "TRASTUZUMAB")
    assert not hits(P.EGFR_DRUG_PATTERN, "erlotinibs")


# --- PI3K/AKT/mTOR patterns --------------------------------------------------------------


@pytest.mark.parametrize("pattern,text", [
    (P.PIK3CA_PATTERN, "PIK3CA H1047R"),
    (P.PIK3CA_PATTERN, "Pik3ca mice"),
    (P.PIK3CA_PATTERN, "p110alpha"),
    (P.PIK3CA_PATTERN, "p110-alpha"),
    (P.PIK3CA_PATTERN, "p110α isoform"),
    (P.PTEN_PATTERN, "PTEN loss"),
    (P.PTEN_PATTERN, "Pten-null mice"),
    (P.PTEN_PATTERN, "zebrafish ptena and ptenb"),
    (P.PTEN_PATTERN, "PTENP1"),
    (P.AKT_PATTERN, "AKT phosphorylation"),
    (P.AKT_PATTERN, "Akt1"),
    (P.AKT_PATTERN, "pAkt"),
    (P.AKT_PATTERN, "p-AKT(Ser473)"),
    (P.AKT_PATTERN, "PKB/Akt"),
    (P.AKT_PATTERN, "the three AKTs"),
    (P.MTOR_PATTERN, "mTOR inhibition"),
    (P.MTOR_PATTERN, "mTORC1"),
    (P.MTOR_PATTERN, "MTORC2"),
    (P.MTOR_PATTERN, "Mtor"),
    (P.MTOR_PATTERN, "mTORC1/2"),
    (P.MTOR_PATTERN, "p-mTOR"),
])
def test_pi3k_positive(pattern, text):
    assert hits(pattern, text)


@pytest.mark.parametrize("pattern,text", [
    (P.PIK3CA_PATTERN, "p110 subunit"),
    (P.PIK3CA_PATTERN, "p110beta"),
    (P.PIK3CA_PATTERN, "PIK3CB"),
    (P.PTEN_PATTERN, "Ptenlike"),
    (P.PTEN_PATTERN, "SPTEN"),
    (P.PTEN_PATTERN, "compten"),
    (P.AKT_PATTERN, "Aktuelle Therapie"),
    (P.AKT_PATTERN, "Aktivitat"),
    (P.AKT_PATTERN, "akt lowercase"),
    (P.AKT_PATTERN, "PAKT"),
    (P.AKT_PATTERN, "AKT4"),
    (P.MTOR_PATTERN, "mentor"),
    (P.MTOR_PATTERN, "amTOR"),
    (P.MTOR_PATTERN, "Mtorvik"),
])
def test_pi3k_negative(pattern, text):
    assert not hits(pattern, text)


def test_pi3k_drugs_any_case_and_no_rapamycin():
    assert hits(P.PI3K_DRUG_PATTERN, "Everolimus") and hits(P.PI3K_DRUG_PATTERN, "IDELALISIB")
    assert not hits(P.PI3K_DRUG_PATTERN, "rapamycin (sirolimus)")


# --- wiring into features -------------------------------------------------------------------


def test_gene_group_first_match_in_order():
    egfr = P.PATHWAYS["egfr"]["gene_groups"]
    assert F.gene_group(paper(abstract="HER2 and EGFR"), egfr) == "EGFR"
    assert F.gene_group(paper(abstract="HER2 only"), egfr) == "ERBB2"
    assert F.gene_group(paper(abstract="HER3 only"), egfr) == "other"
    pi3k = P.PATHWAYS["pi3k"]["gene_groups"]
    assert F.gene_group(paper(abstract="mTOR, Akt and PTEN"), pi3k) == "PTEN"
    assert F.gene_group(paper(abstract="PIK3CA and PTEN"), pi3k) == "PIK3CA"
    assert F.gene_group(paper(abstract="Akt and mTOR"), pi3k) == "AKT"
    assert F.gene_group(paper(abstract="everolimus"), pi3k) == "other"


def test_eligibility_uses_pathway_patterns():
    icite = {"pmid": "1", "is_research_article": True, "is_clinical": False}
    ras = paper(abstract="KRAS mutant cells")
    nephro = paper(abstract="eGFR fell in patients with CKD")
    her2 = paper(abstract="HER2 amplified tumours", pub_types=["Review"])
    assert F.feature_row(ras, icite, "egfr")["exclusion"] == "off_topic"
    assert F.feature_row(nephro, icite, "egfr")["exclusion"] == "off_topic"
    assert F.feature_row(her2, icite, "egfr")["exclusion"] == "non_primary"
    row = F.feature_row(paper(abstract="Pten-null mice"), icite, "pi3k")
    assert row["eligible"] and row["gene_group"] == "PTEN"
    # default (v1) behaviour is unchanged
    assert F.feature_row(ras, icite)["eligible"] and F.feature_row(ras, icite)["gene_group"] == "KRAS"
    assert F.feature_row(paper(abstract="PTEN"), icite)["exclusion"] == "off_topic"


def test_pathway_packets_are_per_pathway_seeded_and_disjoint():
    def rows(pmids, eligible=True):
        return [{"pmid": str(p), "group": "preclinical", "eligible": eligible} for p in pmids]

    a = rows(range(1, 400)) + rows(range(1000, 1010), eligible=False)
    b = rows(range(200, 700))
    drawn = F.sample_pathway_packets({"egfr": a, "pi3k": b})
    assert len(drawn["egfr"]) == 150 and len(drawn["pi3k"]) == 150
    assert not set(drawn["egfr"]) & set(drawn["pi3k"])
    assert all(int(p) < 1000 for p in drawn["egfr"])
    assert drawn["egfr"] == F.sample_validation(rows(range(1, 400)), n=150, seed=2)


def test_corpus_excludes_v1_pmids_and_layout():
    kept, n = corpus.exclude_pmids([{"pmid": "1"}, {"pmid": "2"}, {"pmid": "3"}], {"2"})
    assert [p["pmid"] for p in kept] == ["1", "3"] and n == 1
    assert corpus.layout()["papers"] == corpus.PAPERS_PATH
    lay = corpus.layout("pi3k")
    assert lay["papers"].parent == P.PATHWAYS_DIR / "pi3k"
    assert lay["query"] == P.PATHWAYS["pi3k"]["query"]


def test_layered_cache_reads_base_but_writes_own(tmp_path):
    base = impact.JsonlCache(tmp_path / "base.jsonl")
    base.add_many([{"pmid": "1", "references": ["9"]}])
    before = (tmp_path / "base.jsonl").read_text()
    cache = impact.LayeredJsonlCache(tmp_path / "own.jsonl", base=[base])
    assert "1" in cache and cache.get("1")["references"] == ["9"]
    impact.fetch_citer_refs(["1", "2"], cache,
                            fetch=lambda b: [{"pmid": p, "references": []} for p in b], workers=1)
    assert (tmp_path / "base.jsonl").read_text() == before
    assert "2" in cache and "1" not in impact.JsonlCache(tmp_path / "own.jsonl")


@pytest.mark.parametrize("pattern,text", [
    (P.EGFR_PATTERN, "EGFr expression"),
    (P.EGFR_PATTERN, "EgfR"),
    (P.EGFR_PATTERN, "anti-EGFr antibody"),
    (P.EGFR_PATTERN, "EGFr-TK inhibitors"),
    (P.MTOR_PATTERN, "mTor"),
    (P.MTOR_PATTERN, "mToR"),
    (P.MTOR_PATTERN, "mTorC1"),
    (P.MTOR_PATTERN, "mTORc1"),
    (P.MTOR_PATTERN, "mTORc2 complex"),
    (P.PTEN_PATTERN, "pTEN loss"),
])
def test_review_case_variants_accepted(pattern, text):
    assert hits(pattern, text)


@pytest.mark.parametrize("pattern,text", [
    (P.EGFR_PATTERN, "eGFR"),
    (P.EGFR_PATTERN, "egfr"),
    (P.ERBB2_PATTERN, "her2"),
    (P.ERBB2_PATTERN, "erbb2"),
    (P.AKT_PATTERN, "akt1"),
])
def test_lowercase_still_rejected(pattern, text):
    assert not hits(pattern, text)


def test_plant_akt_excluded_only_when_akt_is_the_only_match():
    icite = {"pmid": "1", "is_research_article": True, "is_clinical": False}
    plant = paper(mesh=["Arabidopsis", "Potassium Channels"], abstract="The AKT1 channel in roots")
    assert F.feature_row(plant, icite, "pi3k")["exclusion"] == "off_topic_plant"
    rice = paper(mesh=["Plant Roots"], abstract="OsAKT1 in rice")
    assert F.feature_row(rice, icite, "pi3k")["exclusion"] == "off_topic_plant"
    # weak plant signal on a mammalian paper (MEDLINE maps AKT1 to Arabidopsis Proteins)
    mammal = paper(mesh=["Humans", "Arabidopsis Proteins"], abstract="Akt1 in plant-derived compound treated cells")
    assert F.feature_row(mammal, icite, "pi3k")["eligible"]
    # another PI3K gene match keeps it on topic
    both = paper(mesh=["Arabidopsis"], abstract="AKT1 and mTOR in Arabidopsis")
    assert F.feature_row(both, icite, "pi3k")["eligible"]
    # the extra rule is PI3K-only
    assert F.feature_row(paper(mesh=["Arabidopsis"], abstract="EGFR and plants"), icite, "egfr")["eligible"]


def test_kidney_egfr_excluded_without_receptor_signal():
    icite = {"pmid": "1", "is_research_article": True, "is_clinical": False}
    kidney = paper(abstract="Estimated glomerular filtration rate (EGFR) fell to an EGFR of 45 mL/min/1.73 m2.")
    assert F.feature_row(kidney, icite, "egfr")["exclusion"] == "off_topic_kidney"
    kinase = paper(abstract="EGFR tyrosine kinase inhibition preserved glomerular filtration rate.")
    assert F.feature_row(kinase, icite, "egfr")["eligible"]
    her2 = paper(abstract="EGFR of 60 mL/min in HER2 patients")
    assert F.feature_row(her2, icite, "egfr")["eligible"]
    plain = paper(abstract="EGFR signalling in keratinocytes")
    assert F.feature_row(plain, icite, "egfr")["eligible"]


def test_marginals_count_pathway_reasons():
    rows = [
        {"pmid": "1", "group": "preclinical", "model_system": "other", "human_genetics": 0,
         "multi_system": 0, "gene_group": "other", "eligible": False, "exclusion": "off_topic_plant"},
        {"pmid": "2", "group": "preclinical", "model_system": "other", "human_genetics": 0,
         "multi_system": 0, "gene_group": "AKT", "eligible": True, "exclusion": None},
    ]
    ex = F.marginals(rows)["preclinical"]["exclusion"]
    assert ex["off_topic_plant"] == 1 and ex["eligible"] == 1


def test_openalex_scope_is_eligible_preclinical_research_articles():
    papers = [
        paper("1", abstract="PTEN loss"),
        paper("2", abstract="KRAS only"),
        paper("3", abstract="mTOR", pub_types=["Review"]),
        paper("4", abstract="Akt", mesh=()),
        paper("5", abstract="AKT1"),
        paper("6", abstract="PTEN"),
    ]
    icite = {str(i): {"is_research_article": True, "is_clinical": False} for i in range(1, 6)}
    icite["5"]["is_clinical"] = True  # clinical: out of scope; 6: no iCite record
    assert impact.pathway_openalex_scope(papers, icite, "pi3k") == ["1"]
