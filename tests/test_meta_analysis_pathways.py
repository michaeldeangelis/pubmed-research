import json

import numpy as np
import pytest

from src.meta import analysis, analysis_pathways


@pytest.fixture(autouse=True)
def _small(monkeypatch):
    monkeypatch.setattr(analysis, "N_BOOT", 60)
    monkeypatch.setattr(analysis, "N_PERM", 30)


def _write(meta, n, human_log_or, genes, seed):
    rng = np.random.default_rng(seed)
    meta.mkdir(parents=True)
    rows = []
    for i in range(n):
        model = str(rng.choice(["cell_only", "animal", "human_samples", "other"], p=[0.35, 0.2, 0.35, 0.1]))
        year = int(rng.integers(2000, 2016))
        eta = -0.4 + human_log_or * (model == "human_samples")
        rows.append({"pmid": str(500000 + i), "year": year, "model_system": model,
                     "gene_group": str(rng.choice(genes)), "n_authors": int(rng.integers(1, 15)),
                     "y": int(rng.random() < 1 / (1 + np.exp(-eta)))})
    with (meta / "papers.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps({"pmid": r["pmid"], "year": r["year"], "journal": f"J{int(r['pmid']) % 5}"}) + "\n")
    with (meta / "outcome_b.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps({"pmid": r["pmid"], "clin_cited_8y": r["y"], "n_clin_8y": r["y"]}) + "\n")
    with (meta / "features.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps({"pmid": r["pmid"], "group": "preclinical", "eligible": True,
                                "model_system": r["model_system"], "human_genetics": 0, "multi_system": 0,
                                "gene_group": r["gene_group"], "n_authors": r["n_authors"]}) + "\n")
    with (meta / "impact.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps({"pmid": r["pmid"], "rcr": 1.0, "cd": 0.0, "n_refs_openalex": 30, "n_refs_icite": 28}) + "\n")


def test_confirms_where_effect_exists_and_restores_gene_levels(tmp_path):
    _write(tmp_path / "egfr", 3000, 0.8, ["EGFR", "ERBB2", "other"], 0)
    _write(tmp_path / "pi3k", 3000, 0.0, ["PIK3CA", "PTEN", "AKT", "MTOR", "other"], 1)
    before = analysis.GENE_LEVELS
    report = analysis_pathways.run(tmp_path)
    assert analysis.GENE_LEVELS == before
    assert report["pathways"]["egfr"]["confirmed"]
    assert not report["pathways"]["pi3k"]["confirmed"]
    assert report["verdict"] == "confirmed in egfr only"
    assert report["pathways"]["pi3k"]["gene_levels"] == ["AKT", "MTOR", "PTEN", "other"]
