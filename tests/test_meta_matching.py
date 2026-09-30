import json

import numpy as np

from src.meta import matching


def _write(tmp_path, name, rows):
    d = tmp_path / name
    d.mkdir()
    with (d / "papers.jsonl").open("w") as fp, (d / "features.jsonl").open("w") as ff, \
            (d / "impact.jsonl").open("w") as fi:
        for r in rows:
            fp.write(json.dumps({"pmid": r["pmid"], "year": r["year"], "title": f"T{r['pmid']}",
                                 "abstract": f"A{r['pmid']}", "mesh": r["mesh"]}) + "\n")
            ff.write(json.dumps({"pmid": r["pmid"], "group": r.get("group", "preclinical"),
                                 "eligible": r.get("eligible", True), "model_system": r["ms"],
                                 "gene_group": r.get("gene", "G"), "n_authors": r.get("authors", 5)}) + "\n")
            fi.write(json.dumps({"pmid": r["pmid"], "n_refs_openalex": 0, "n_refs_icite": 30,
                                 "rcr": 99}) + "\n")
    return d


def _p(pmid, ms, mesh, year=2005, gene="G", **kw):
    return {"pmid": str(pmid), "ms": ms, "mesh": mesh, "year": year, "gene": gene, **kw}


H = "human_samples"
C = "cell_only"


def test_removed_terms():
    for t in ("Humans", "Female", "Aged, 80 and over", "Mice, Nude", "Cell Line, Tumor", "HeLa Cells",
              "MCF-10A Cells", "Retrospective Studies", "Kaplan-Meier Estimate", "Immunohistochemistry/methods",
              "Blotting, Western", "RNA, Small Interfering", "Reverse Transcriptase Polymerase Chain Reaction",
              "In Situ Hybridization, Fluorescence", "Cell Proliferation", "Clinical Trials, Phase II as Topic",
              "China", "*Tissue Array Analysis"):
        assert matching.is_removed(t), t
    for t in ("Apoptosis", "Lung Neoplasms", "Gefitinib", "Receptor, Epidermal Growth Factor", "Signal Transduction",
              "Cell Lineage", "Th17 Cells", "Epithelial Cells", "Mutation", "Breast Neoplasms/genetics"):
        assert not matching.is_removed(t), t
    assert matching.topic_terms(["Humans", "Lung Neoplasms/genetics", "*Lung Neoplasms/pathology"]) == ["lung neoplasms"]


def test_pool_dedup_blocks_years_and_greedy(tmp_path):
    topic_a = ["Lung Neoplasms", "Gefitinib", "Humans", "Retrospective Studies"]
    topic_b = ["Breast Neoplasms", "Trastuzumab", "Cell Line, Tumor"]
    ras = [
        _p(1, H, topic_a), _p(2, C, ["Lung Neoplasms", "Gefitinib", "Cell Line"]),
        _p(3, H, topic_b + ["Humans"]), _p(4, C, topic_b, year=2008),  # year gap 3 -> no match
        _p(5, H, ["Humans", "Female"]),  # unmatchable: nothing left
        _p(6, "animal", topic_a), _p(7, C, topic_a, gene="OTHER"),  # wrong gene group
        _p(8, C, topic_a, eligible=False),
    ]
    egfr = [_p(1, C, topic_a), _p(20, H, topic_b), _p(21, C, topic_b, year=2006)]
    dirs = {"ras": _write(tmp_path, "ras", ras), "egfr": _write(tmp_path, "egfr", egfr)}
    rows = matching.load_pool(dirs)
    pm = {r["pmid"]: r for r in rows}
    assert pm["1"]["pathway"] == "ras" and pm["1"]["treated"] == 1
    assert "6" not in pm and "8" not in pm
    assert pm["5"]["terms"] == []
    assert pm["2"]["n_refs"] == 30  # openalex 0 falls back to icite for ras
    summary = matching.run(dirs, tmp_path / "out", tmp_path / "bench")
    pairs = [json.loads(l) for l in (tmp_path / "out" / "pairs.jsonl").read_text().splitlines()]
    got = {(p["treated_pmid"], p["control_pmid"]) for p in pairs}
    assert got == {("1", "2"), ("20", "21")}
    assert summary["pool"]["ras"]["treated"]["unmatchable_no_descriptors"] == 1
    assert summary["caliper"] == 0.6 and summary["caliper_table"]["0.6"]["n_pairs"] == 2
    removed = json.loads((tmp_path / "bench" / "removed_mesh.json").read_text())
    assert "humans" in removed["removed_in_corpus"] and "gefitinib" not in removed["removed_in_corpus"]
    key = json.loads((tmp_path / "bench" / "match_judge_key.json").read_text())
    packets = json.loads((tmp_path / "bench" / "match_judge_packets.json").read_text())
    assert len(key) == len(packets["items"]) == 2
    for item in packets["items"]:
        k = key[item["pair_id"]]
        assert item["paper_a"]["title"] == f"T{k['pmid_a']}"
        treated = k["pmid_a"] if k["treated"] == "a" else k["pmid_b"]
        assert treated == k["treated_pmid"]
        assert "pmid" not in json.dumps(item)


def test_greedy_takes_highest_edge_first_and_breaks_ties_by_pmid():
    rows = [{"pmid": p} for p in ("10", "11", "20", "21")]
    ti = np.array([0, 0, 1, 1])
    ci = np.array([2, 3, 2, 3])
    sims = np.array([0.9, 0.8, 0.95, 0.5])
    assert matching.greedy(rows, ti, ci, sims) == [(1, 2, 0.95), (0, 3, 0.8)]
    tie = matching.greedy(rows, ti, ci, np.array([0.7, 0.7, 0.7, 0.7]))
    assert tie == [(0, 2, 0.7), (1, 3, 0.7)]


def test_caliper_choice():
    assert matching.choose_caliper({0.6: 0.3, 0.5: 0.49, 0.4: 0.5, 0.3: 0.8}) == (0.4, True)
    assert matching.choose_caliper({0.6: 0.1, 0.5: 0.2, 0.4: 0.3, 0.3: 0.4}) == (0.3, False)


def test_caliper_subset_consistency(tmp_path):
    rng = np.random.default_rng(1)
    vocab = [f"Term{i}" for i in range(12)]
    rows = [_p(1000 + i, H if i % 2 else C, list(rng.choice(vocab, 3, replace=False)), year=2005 + i % 2)
            for i in range(80)]
    dirs = {"ras": _write(tmp_path, "ras", rows)}
    summary = matching.run(dirs, tmp_path / "out", tmp_path / "bench")
    n = [summary["caliper_table"][str(c)]["n_pairs"] for c in matching.CALIPERS]
    assert n == sorted(n)
    pairs = [json.loads(l) for l in (tmp_path / "out" / "pairs.jsonl").read_text().splitlines()]
    assert all(p["similarity"] >= summary["caliper"] for p in pairs)
    assert len({p["control_pmid"] for p in pairs}) == len(pairs) == len({p["treated_pmid"] for p in pairs})


def test_smd():
    assert matching.smd([1, 2, 3], [1, 2, 3]) == 0.0
    assert matching.smd([2, 3, 4], [1, 2, 3]) == 1.0
