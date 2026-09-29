import json

import pytest

from src.meta import features as F


def paper(pmid="1", mesh=(), title="", abstract="", n_authors=5):
    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "mesh": list(mesh),
        "pub_types": ["Journal Article"],
        "n_authors": n_authors,
    }


# --- normalization -----------------------------------------------------------------


def test_norm_term_drops_qualifiers_and_major_marks():
    assert F.norm_term("Mice, Nude/genetics") == "mice, nude"
    assert F.norm_term("*Cell Line, Tumor") == "cell line, tumor"
    assert F.norm_term("  Humans ") == "humans"


# --- F1 model system ------------------------------------------------------------------


def test_human_samples_requires_humans_and_patient_term():
    p = paper(mesh=["Humans", "DNA Mutational Analysis", "Colorectal Neoplasms/genetics"])
    assert F.rule_labels(p)["model_system"] == "human_samples"
    # Humans alone (e.g. structural work) is not human samples
    assert F.rule_labels(paper(mesh=["Humans", "Protein Conformation"]))["model_system"] == "other"
    # patient term without Humans is not human samples
    assert F.rule_labels(paper(mesh=["Biopsy"]))["model_system"] == "other"


def test_human_plus_cells_is_cell_only_and_multi():
    p = paper(mesh=["Humans", "Middle Aged", "Cell Line, Tumor"])
    lab = F.rule_labels(p)
    assert lab["model_system"] == "cell_only"
    assert lab["multi_system"] == 1
    assert lab["systems_present"] == ["cell", "human_samples"]


def test_animal_takes_precedence_over_cells_and_humans():
    p = paper(mesh=["Humans", "Aged", "Cell Line, Tumor", "Mice, Nude", "Animals"])
    lab = F.rule_labels(p)
    assert lab["model_system"] == "animal"
    assert lab["multi_system"] == 1
    assert lab["systems_present"] == ["animal", "cell", "human_samples"]


def test_strong_animal_terms_and_prefixes():
    for term in ["Xenograft Model Antitumor Assays", "Disease Models, Animal", "Zebrafish",
                 "Mice, Transgenic", "Mice, Inbred C57BL", "Rats, Sprague-Dawley"]:
        assert F.rule_labels(paper(mesh=["Animals", term]))["model_system"] == "animal", term


def test_protein_descriptors_are_not_organisms():
    lab = F.rule_labels(paper(mesh=["Drosophila Proteins", "Cell Line"]))
    assert lab["model_system"] == "cell_only"


def test_bare_species_with_cell_line_needs_in_vivo_text():
    mesh = ["Animals", "Mice", "NIH 3T3 Cells"]
    in_vitro = paper(mesh=mesh, abstract="Ras transformed mouse fibroblasts in culture.")
    assert F.rule_labels(in_vitro)["model_system"] == "cell_only"
    in_vivo = paper(mesh=mesh, abstract="Transformed cells formed tumors in nude mice.")
    assert F.rule_labels(in_vivo)["model_system"] == "animal"
    assert F.rule_labels(in_vivo)["multi_system"] == 1


def test_bare_species_without_cells_is_animal():
    assert F.rule_labels(paper(mesh=["Animals", "Mice", "Lung Neoplasms"]))["model_system"] == "animal"


def test_cell_line_species_tags_are_cells_not_animals():
    lab = F.rule_labels(paper(mesh=["Animals", "COS Cells", "Chlorocebus aethiops"]))
    assert lab["model_system"] == "cell_only"
    assert lab["systems_present"] == ["cell"]


def test_other_when_no_system():
    assert F.rule_labels(paper(mesh=["Protein Conformation", "Crystallography, X-Ray"]))["model_system"] == "other"


def test_no_mesh_falls_back_to_text():
    p = paper(mesh=[], abstract="We treated xenografts in mice and KRAS mutant cell lines.")
    lab = F.rule_labels(p)
    assert lab["model_system"] == "animal"
    assert lab["systems_present"] == ["animal", "cell"]


# --- F2 human genetics ------------------------------------------------------------------


def test_human_genetics_mesh_and_text():
    assert F.human_genetics(paper(mesh=["Germ-Line Mutation"])) == 1
    assert F.human_genetics(paper(mesh=["Polymorphism, Single Nucleotide/genetics"])) == 1
    assert F.human_genetics(paper(abstract="A genome-wide association study of melanoma.")) == 1
    assert F.human_genetics(paper(abstract="Carriers of rs12345 had higher risk.")) == 1
    assert F.human_genetics(paper(title="<i>PTPN11</i> in familial Noonan syndrome")) == 1


def test_human_genetics_negatives():
    somatic = paper(
        mesh=["Mutation", "DNA Mutational Analysis"],
        abstract="Somatic KRAS mutations, loss of function of NF1, odds ratio 2.1; drug susceptibility.",
    )
    assert F.human_genetics(somatic) == 0
    methods = paper(
        mesh=["Polymorphism, Restriction Fragment Length", "Polymorphism, Single-Stranded Conformational"],
        abstract="KRAS codon 12 by PCR-restriction fragment length polymorphism and "
        "single-strand conformation polymorphism analysis.",
    )
    assert F.human_genetics(methods) == 0
    assert F.human_genetics(paper(abstract="SSCP analysis, and a TP53 polymorphism.")) == 1


# --- covariates ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,group",
    [
        ("K-ras mutations in colon cancer", "KRAS"),
        ("KRASG12C inhibitors", "KRAS"),
        ("Ki-ras codon 12", "KRAS"),
        ("BRAF V600E in melanoma", "BRAF"),
        ("BRAFV600E and B-Raf dimers", "BRAF"),
        ("NRAS-mutant melanoma", "NRAS/HRAS"),
        ("c-Ha-ras transformed cells", "NRAS/HRAS"),
        ("H-Ras signaling", "NRAS/HRAS"),
        ("Ras signaling and MEK", "other"),
        ("Krasavin et al.", "other"),
        ("BRAF and KRAS mutations are exclusive", "KRAS"),  # order KRAS > BRAF
        ("NRAS and BRAF in melanoma", "BRAF"),  # order BRAF > NRAS/HRAS
    ],
)
def test_gene_group(text, group):
    assert F.gene_group(paper(title=text)) == group


# --- rows ---------------------------------------------------------------------------------


def test_build_features_keeps_research_articles_and_sets_group():
    papers = [
        paper("3", mesh=["Humans", "Aged"], title="KRAS"),
        paper("1", mesh=["Cell Line"], title="BRAF"),
        paper("2", mesh=["Mice, Nude"]),
    ]
    icite = [
        {"pmid": "1", "is_research_article": True, "is_clinical": False},
        {"pmid": "2", "is_research_article": "No", "is_clinical": False},
        {"pmid": "3", "is_research_article": "Yes", "is_clinical": "Yes"},
    ]
    rows = F.build_features(papers, icite)
    assert [r["pmid"] for r in rows] == ["1", "3"]
    assert all(tuple(r) == F.FEATURE_KEYS for r in rows)
    assert rows[0]["group"] == "preclinical" and rows[1]["group"] == "clinical"
    assert rows[0]["model_system"] == "cell_only" and rows[0]["gene_group"] == "BRAF"
    assert rows[1]["n_refs"] is None and rows[1]["n_authors"] == 5


def test_marginals_counts_only():
    rows = [
        {"pmid": "1", "group": "preclinical", "model_system": "animal", "human_genetics": 0,
         "multi_system": 1, "gene_group": "KRAS", "n_authors": 3, "n_refs": None},
        {"pmid": "2", "group": "clinical", "model_system": "other", "human_genetics": 1,
         "multi_system": 0, "gene_group": "other", "n_authors": 3, "n_refs": None},
    ]
    m = F.marginals(rows)
    assert m["n"] == 2
    assert m["model_system"] == {"animal": 1, "other": 1}
    assert m["preclinical"]["n"] == 1


# --- validation packets ---------------------------------------------------------------------


def _rows(n, group="preclinical"):
    return [{"pmid": str(i), "group": group} for i in range(1, n + 1)]


def test_sample_is_deterministic_preclinical_only():
    rows = _rows(500) + [{"pmid": str(10_000 + i), "group": "clinical"} for i in range(100)]
    a = F.sample_validation(rows)
    b = F.sample_validation(list(reversed(rows)))
    assert a == b
    assert len(a) == 300 and len(set(a)) == 300
    assert all(int(p) <= 500 for p in a)
    assert len(F.sample_validation(_rows(10))) == 10


def test_packets_contain_no_rule_outputs():
    papers = [paper("7", mesh=["Humans"], title="t", abstract="a")]
    packets = F.make_packets(papers, ["7"])
    assert packets == [{"pmid": "7", "title": "t", "abstract": "a", "mesh": ["Humans"]}]


def test_extraction_schema_uses_same_categories():
    s = F.EXTRACTION_SCHEMA
    assert set(F.MODEL_SYSTEMS) <= set(s["definitions"]["model_system"])
    assert set(F.SYSTEMS) == set(s["definitions"]["systems_present"])
    json.dumps(s)


# --- kappa and scoring --------------------------------------------------------------------------


def test_cohen_kappa_known_values():
    # classic 2x2 example: po = 0.7, pe = 0.5 -> 0.4
    a = [1] * 20 + [1] * 5 + [0] * 10 + [0] * 15
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert F.kappa(a, b) == pytest.approx(0.4)
    assert F.kappa([0, 1, 2, 0], [0, 1, 2, 0]) == pytest.approx(1.0)
    assert F.kappa([0, 1, 0, 1], [1, 0, 1, 0]) == pytest.approx(-1.0)
    assert F.kappa([1, 1, 1], [1, 1, 1]) == 1.0  # degenerate: pe = 1
    with pytest.raises(ValueError):
        F.kappa([1], [1, 0])


def test_cohen_kappa_matches_sklearn_if_available():
    sk = pytest.importorskip("sklearn.metrics")
    import numpy as np

    rng = np.random.default_rng(1)
    a = rng.integers(0, 4, 200).tolist()
    b = [x if rng.random() < 0.7 else int(rng.integers(0, 4)) for x in a]
    assert F.kappa(a, b) == pytest.approx(sk.cohen_kappa_score(a, b))


def test_score_perfect_and_disagreements():
    packets = [
        paper("1", mesh=["Humans", "Aged", "DNA Mutational Analysis"]),
        paper("2", mesh=["Cell Line, Tumor"]),
        paper("3", mesh=["Mice, Nude", "Cell Line, Tumor"], abstract="germline"),
        paper("4", mesh=["Protein Conformation"]),
    ]
    for p in packets:
        del p["pub_types"], p["n_authors"]
    extracted = []
    for p in packets:
        lab = F.rule_labels(p)
        extracted.append({"pmid": p["pmid"], "model_system": lab["model_system"],
                          "human_genetics": lab["human_genetics"],
                          "systems_present": lab["systems_present"]})
    res = F.score(packets, extracted)
    assert res["n_scored"] == 4
    assert all(v["kappa"] == 1.0 and v["gate_pass"] for v in res["features"].values())
    assert res["dropped_features"] == [] and res["n_disagreements"] == 0

    extracted[1] = {"pmid": "2", "model_system": "other", "human_genetics": "1",
                    "systems_present": "human_samples, animal"}
    res = F.score(packets, {e["pmid"]: e for e in extracted})
    assert res["n_disagreements"] == 1
    assert set(res["disagreements_for_owner"][0]["features"]) == {
        "model_system", "human_genetics", "multi_system"}
    cm = res["features"]["model_system"]["confusion_rule_rows_model_cols"]
    assert cm["cell_only"]["other"] == 1
    assert res["features"]["model_system"]["kappa"] < 1.0


def test_score_cli_writes_gate(tmp_path, monkeypatch):
    packets = [paper(str(i), mesh=["Cell Line"] if i % 2 else ["Mice, Nude"]) for i in range(1, 7)]
    packets_path = tmp_path / "packets.json"
    packets_path.write_text(json.dumps(packets))
    extracted = [dict(pmid=p["pmid"], **{k: v for k, v in F.rule_labels(p).items()
                                         if k != "multi_system"}) for p in packets]
    ex_path = tmp_path / "extracted.json"
    ex_path.write_text(json.dumps(extracted))
    gate_path = tmp_path / "results" / "gate.json"
    monkeypatch.setattr(F, "PACKETS_PATH", packets_path)
    monkeypatch.setattr(F, "GATE_PATH", gate_path)
    F.main(["score", str(ex_path)])
    out = json.loads(gate_path.read_text())
    assert out["features"]["model_system"]["gate_pass"] is True
    assert out["gate_threshold"] == 0.60


def test_build_cli_on_synthetic_data(tmp_path, monkeypatch, capsys):
    papers = [paper(str(i), mesh=["Cell Line"], title="KRAS") for i in range(1, 21)]
    icite = [{"pmid": str(i), "is_research_article": True, "is_clinical": i > 15} for i in range(1, 21)]
    (tmp_path / "papers.jsonl").write_text("\n".join(json.dumps(p) for p in papers))
    (tmp_path / "icite.jsonl").write_text("\n".join(json.dumps(r) for r in icite))
    for name, target in [("PAPERS_PATH", "papers.jsonl"), ("ICITE_PATH", "icite.jsonl"),
                         ("FEATURES_PATH", "features.jsonl"), ("PACKETS_PATH", "v/packets.json"),
                         ("EXTRACT_SCHEMA_PATH", "v/schema.json")]:
        monkeypatch.setattr(F, name, tmp_path / target)
    counts = F.main_build()
    assert counts["group"] == {"clinical": 5, "preclinical": 15}
    packets = json.loads((tmp_path / "v/packets.json").read_text())
    assert len(packets) == 15
    assert all(set(p) == {"pmid", "title", "abstract", "mesh"} for p in packets)
    assert len(F.read_jsonl(tmp_path / "features.jsonl")) == 20


def test_sex_tags_count_for_humans_only_without_animals():
    assert F.rule_labels(paper(mesh=["Humans", "Female", "Mutation"]))["model_system"] == "human_samples"
    xeno = F.rule_labels(paper(mesh=["Humans", "Female", "Animals", "Mice, Nude", "Cell Line, Tumor"]))
    assert xeno["systems_present"] == ["animal", "cell"]


def test_text_fallback_human_samples_is_specific():
    background = paper(abstract="Patients with KRAS mutant cancer respond poorly; we study signaling.")
    assert F.rule_labels(background)["systems_present"] == []
    cohort = paper(abstract="We sequenced 120 tumors from patients; biopsy material was paraffin-embedded.")
    assert F.rule_labels(cohort)["model_system"] == "human_samples"
