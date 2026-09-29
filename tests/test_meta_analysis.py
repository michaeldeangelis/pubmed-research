import json

import numpy as np
import pytest

from src.meta import analysis


def _rows(n=3000, animal_log_or=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        year = int(rng.integers(2000, 2016))
        model = rng.choice(["cell_only", "animal", "human_samples", "other"], p=[0.4, 0.3, 0.2, 0.1])
        genetics = int(rng.random() < 0.2)
        multi = int(rng.random() < 0.3)
        eta = -0.3 + animal_log_or * (model == "animal") + 0.02 * (year - 2008)
        rows.append(
            {
                "pmid": str(100000 + i),
                "group": "preclinical",
                "model_system": str(model),
                "human_genetics": genetics,
                "multi_system": multi,
                "gene_group": str(rng.choice(["KRAS", "BRAF", "NRAS/HRAS", "other"])),
                "n_authors": int(rng.integers(1, 20)),
                "year": year,
                "clin_cited_8y": int(rng.random() < 1 / (1 + np.exp(-eta))),
                "n_clin_8y": 0,
                "n_refs": int(rng.integers(0, 80)) if rng.random() < 0.9 else None,
            }
        )
    return rows


@pytest.fixture(autouse=True)
def _small(monkeypatch):
    monkeypatch.setattr(analysis, "N_BOOT", 60)
    monkeypatch.setattr(analysis, "N_PERM", 60)


def _write(tmp_path, rows):
    meta = tmp_path / "meta"
    meta.mkdir()
    with (meta / "papers.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps({"pmid": r["pmid"], "year": r["year"], "journal": f"J{int(r['pmid']) % 7}"}) + "\n")
    with (meta / "outcome_b.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("pmid", "clin_cited_8y", "n_clin_8y")}) + "\n")
    with (meta / "features.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("pmid", "group", "model_system", "human_genetics",
                                                  "multi_system", "gene_group", "n_authors")}) + "\n")
    with (meta / "impact.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps({"pmid": r["pmid"], "n_refs_openalex": r["n_refs"], "rcr": 1.0, "cd": 0.0}) + "\n")
    return meta


def test_fit_logit_recovers_coefficients():
    rng = np.random.default_rng(1)
    x = np.column_stack([np.ones(20000), rng.normal(size=20000)])
    y = (rng.random(20000) < 1 / (1 + np.exp(-(0.5 + 1.2 * x[:, 1])))).astype(float)
    beta = analysis.fit_logit(x, y)
    assert beta == pytest.approx([0.5, 1.2], abs=0.06)


def test_real_effect_passes_and_null_features_do_not(tmp_path):
    report = analysis.run(_write(tmp_path, _rows(animal_log_or=1.0)))
    assert report["verdicts"]["model_system=animal"]["pass"]
    assert not report["verdicts"]["human_genetics"]["pass"]
    assert report["verdict"] == "pass"
    assert report["ladder_auroc_confirmation"]["candidate"] > report["ladder_auroc_confirmation"]["simplest"]
    sec = report["secondary"]
    assert set(sec) == {"n_clin_8y_negbin_rate_ratio", "log_rcr_plus_0_1_linear", "disruption_linear",
                        "journal_sensitivity_logit_or"}
    assert sec["journal_sensitivity_logit_or"]["terms"]["model_system=animal"]["estimate"] > 1.5


def test_null_data_is_killed(tmp_path):
    report = analysis.run(_write(tmp_path, _rows(animal_log_or=0.0, seed=3)))
    assert report["verdict"] in ("kill", "unreliable (placebo)")
    assert report["passed"] == [] or report["verdict"] != "pass"


def test_drop_removes_feature_columns(tmp_path):
    report = analysis.run(_write(tmp_path, _rows(animal_log_or=1.0)), drop=("human_genetics",))
    assert "human_genetics" not in report["verdicts"]
    assert report["dropped_by_gate"] == ["human_genetics"]
