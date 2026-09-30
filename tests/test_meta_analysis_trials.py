import json

import numpy as np
import pytest

from src.meta import analysis, analysis_trials


@pytest.fixture(autouse=True)
def _small(monkeypatch):
    monkeypatch.setattr(analysis, "N_BOOT", 40)
    monkeypatch.setattr(analysis, "N_PERM", 20)


def _write(tmp_path, name, genes, start, n, log_or, seed, shared=()):
    rng = np.random.default_rng(seed)
    meta = tmp_path / name
    meta.mkdir()
    trials = tmp_path / "trials"
    trials.mkdir(exist_ok=True)
    pmids = list(shared) + [str(start + i) for i in range(n - len(shared))]
    with (meta / "papers.jsonl").open("w") as fp, (meta / "features.jsonl").open("w") as ff, \
            (meta / "impact.jsonl").open("w") as fi, (trials / f"outcome_{name}.jsonl").open("w") as fo:
        for pmid in pmids:
            model = str(rng.choice(["cell_only", "animal", "human_samples", "other"], p=[0.35, 0.2, 0.35, 0.1]))
            y = int(rng.random() < 1 / (1 + np.exp(-(-1.5 + log_or * (model == "human_samples")))))
            fp.write(json.dumps({"pmid": pmid, "year": int(rng.integers(2000, 2016)), "journal": "J"}) + "\n")
            ff.write(json.dumps({"pmid": pmid, "group": "preclinical", "eligible": True, "model_system": model,
                                 "human_genetics": 0, "multi_system": 0, "gene_group": str(rng.choice(genes)),
                                 "n_authors": 5}) + "\n")
            fi.write(json.dumps({"pmid": pmid, "n_refs_openalex": 30, "n_refs_icite": 28}) + "\n")
            fo.write(json.dumps({"pmid": pmid, "trial_bg_8y": y, "trial_bg_8y_drug": y, "trial_bg_8y_ph2": y}) + "\n")
    return meta


def _dirs(tmp_path, log_or):
    return {
        "ras": _write(tmp_path, "ras", ["KRAS", "BRAF"], 100000, 1500, log_or, 0),
        "egfr": _write(tmp_path, "egfr", ["EGFR", "ERBB2"], 200000, 1500, log_or, 1, shared=("100000",)),
        "pi3k": _write(tmp_path, "pi3k", ["PIK3CA", "PTEN"], 300000, 1500, log_or, 2),
    }


def test_pooled_dedupes_and_codes_genes_by_pathway(tmp_path):
    dirs = _dirs(tmp_path, 0.0)
    rows = analysis_trials.load_pooled(dirs, tmp_path / "trials")
    pmids = [r["pmid"] for r in rows]
    assert len(pmids) == len(set(pmids)) == 4499
    assert next(r for r in rows if r["pmid"] == "100000")["pathway"] == "ras"
    assert {r["gene_group"] for r in rows} == {"ras:KRAS", "ras:BRAF", "egfr:EGFR", "egfr:ERBB2",
                                              "pi3k:PIK3CA", "pi3k:PTEN"}


def test_verdict_directions(tmp_path):
    assert analysis_trials.run(_dirs(tmp_path, 0.9), tmp_path / "trials")["verdict"] == "same direction"


def test_verdict_opposite_and_rules():
    placebo = {"lo": 0.9, "hi": 1.1}
    assert analysis_trials.verdict({"lo": 0.5, "hi": 0.8}, placebo, 500) == "opposite"
    assert analysis_trials.verdict({"lo": 0.8, "hi": 1.2}, placebo, 500) == "null"
    assert analysis_trials.verdict({"lo": 1.2, "hi": 1.5}, placebo, 50) == "underpowered"
    assert analysis_trials.verdict({"lo": 1.2, "hi": 1.5}, {"lo": 1.05, "hi": 1.2}, 500) == "unreliable (placebo)"
