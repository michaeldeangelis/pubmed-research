import json
import math

import pytest

from src.meta import analysis_matched as am


def test_matched_or_exact_ci():
    m = am.matched_or(30, 10)
    assert m["or"] == 3.0
    assert 1.4 < m["lo"] < 1.6 and 6 < m["hi"] < 7.5
    assert am.verdict(m) == "persists"
    assert am.verdict(am.matched_or(10, 30)) == "reversed"
    assert am.verdict(am.matched_or(12, 10)) == "attenuated to null"
    assert am.verdict(am.matched_or(0, 0)) == "no discordant pairs"


def test_crude_or_and_attenuation():
    c = am.crude_or([1] * 20 + [0] * 80, [1] * 10 + [0] * 90)
    assert c["or"] == pytest.approx((20 * 90) / (80 * 10))
    assert am._attenuation(1.5, 2.25) == pytest.approx(0.5)
    assert am._attenuation(None, 2.0) is None


def _jl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_run_counts_discordant_pairs(tmp_path):
    ras = tmp_path / "ras"
    trials = tmp_path / "trials"
    match = tmp_path / "match"
    pmids = [str(1000 + i) for i in range(8)]
    systems = ["human_samples", "cell_only"] * 4
    _jl(ras / "papers.jsonl", [{"pmid": p, "year": 2010} for p in pmids])
    _jl(ras / "features.jsonl", [{"pmid": p, "group": "preclinical", "eligible": True, "model_system": s,
                                  "gene_group": "KRAS", "n_authors": 3} for p, s in zip(pmids, systems)])
    _jl(ras / "impact.jsonl", [{"pmid": p, "n_refs_openalex": 20} for p in pmids])
    clin = [1, 0, 1, 0, 0, 1, 1, 1]
    _jl(ras / "outcome_b.jsonl", [{"pmid": p, "clin_cited_8y": y, "n_clin_8y": y} for p, y in zip(pmids, clin)])
    _jl(trials / "outcome_ras.jsonl", [{"pmid": p, "trial_bg_8y": 0, "trial_bg_8y_drug": 0, "trial_bg_8y_ph2": 0}
                                       for p in pmids])
    _jl(match / "pairs.jsonl", [{"treated_pmid": pmids[i], "control_pmid": pmids[i + 1], "pathway": "ras",
                                 "gene_group": "KRAS", "similarity": 0.9} for i in range(0, 8, 2)])
    report = am.run(match, {"ras": ras}, trials)
    m = report["outcomes"]["clin_cited_8y"]["matched"]
    assert (m["b"], m["c"]) == (2, 1)
    assert report["outcomes"]["trial_bg_8y"]["verdict"] == "no discordant pairs"
    assert report["n_pairs"] == 4
    assert not math.isnan(report["outcomes"]["clin_cited_8y"]["crude_all"]["or"])
