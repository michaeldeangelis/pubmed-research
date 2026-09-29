"""Unit tests for src.meta.rpcb on synthetic rows (no network)."""

import math

import pytest

from src.meta import rpcb


def row(paper="1", exp="1", eff="1", rep="1", expected="Positive", es_type="Cohen's d",
        orig="2.0", olo="1.0", ohi="3.0", r="1.5", rse="0.5", rlo="0.5", rhi="2.5",
        obs_smd="Positive", obs="Positive", meta="1.7", meta_p="0.001"):
    return {"Paper #": paper, "Experiment #": exp, "Effect #": eff, "Internal replication #": rep,
            rpcb.EXPECTED: expected, "Effect size type (SMD)": es_type,
            "Original effect size (SMD)": orig, "Original lower CI (SMD)": olo,
            "Original upper CI (SMD)": ohi, "Replication effect size (SMD)": r,
            "Replication standard error (SMD)": rse, "Replication lower CI (SMD)": rlo,
            "Replication upper CI (SMD)": rhi, "Replication p value (SMD)": "0.01",
            rpcb.OBS_SMD: obs_smd, rpcb.OBS: obs,
            "Meta-analysis effect size (SMD)": meta, "Meta-analysis p value (SMD)": meta_p}


def test_wilson_known_values():
    w = rpcb.wilson(5, 10)
    assert w["rate"] == 0.5
    assert w["ci95"][0] == pytest.approx(0.2366, abs=1e-3)
    assert w["ci95"][1] == pytest.approx(0.7634, abs=1e-3)
    assert rpcb.wilson(0, 0)["rate"] is None
    assert rpcb.wilson(0, 5)["ci95"][0] == 0.0


def test_single_effect_criteria():
    s = rpcb.evaluate_effect([row()])
    assert s["same_direction"] == 1 and s["same_direction_p05"] == 1
    assert s["original_in_replication_ci"] == 1  # 2.0 in [0.5, 2.5]
    assert s["replication_in_original_ci"] == 1  # 1.5 in [1, 3]
    assert s["meta_original_plus_replication_significant"] == 1
    s = rpcb.evaluate_effect([row(obs_smd="Null-positive", r="0.2", rlo="-0.5", rhi="0.9", meta_p="0.2")])
    assert (s["same_direction"], s["same_direction_p05"]) == (1, 0)
    assert s["original_in_replication_ci"] == 0 and s["replication_in_original_ci"] == 0
    assert s["meta_original_plus_replication_significant"] == 0
    s = rpcb.evaluate_effect([row(obs_smd="Null", obs="Null")])
    assert (s["same_direction"], s["same_direction_p05"]) == (0, 0)


def test_fallback_to_original_test_column_when_no_smd():
    s = rpcb.evaluate_effect([row(es_type="NA", orig="NA", olo="NA", ohi="NA", r="NA", rse="NA",
                                  rlo="NA", rhi="NA", obs_smd="Negative", obs="Null-positive",
                                  meta="NA", meta_p="NA")])
    assert s["basis"] == "original_test_categorical"
    assert (s["same_direction"], s["same_direction_p05"]) == (1, 0)
    assert s["original_in_replication_ci"] is None
    assert s["meta_original_plus_replication_significant"] is None


def test_internal_replications_pooled_fixed_effect():
    rows = [row(rep="1", r="1.0", rse="0.5", obs_smd="Null-positive"),
            row(rep="2", r="2.0", rse="0.5", obs_smd="Positive")]
    s = rpcb.evaluate_effect(rows)
    assert s["basis"] == "fixed_effect_internal_replications"
    assert s["replication_smd"] == pytest.approx(1.5)
    se = math.sqrt(1 / 8)
    assert s["same_direction_p05"] == 1  # z = 1.5 / 0.354
    assert s["original_in_replication_ci"] == int(1.5 - 1.96 * se <= 2.0 <= 1.5 + 1.96 * se)
    assert rpcb.fixed_effect([1.0, 3.0], [1.0, 1.0]) == (2.0, math.sqrt(0.5))


def test_internal_replications_majority_vote_without_numbers():
    na = dict(es_type="NA", orig="NA", olo="NA", ohi="NA", r="NA", rse="NA", rlo="NA", rhi="NA",
              meta="NA", meta_p="NA")
    rows = [row(rep="1", obs_smd="Null", obs="Null", **na), row(rep="2", **na), row(rep="3", **na)]
    s = rpcb.evaluate_effect(rows)
    assert s["basis"] == "majority_vote_internal_replications"
    assert (s["same_direction"], s["same_direction_p05"]) == (1, 1)


def test_hazard_ratio_meta_direction():
    s = rpcb.evaluate_effect([row(es_type="Hazard ratio", orig="4.0", olo="1.5", ohi="10",
                                  r="0.8", rlo="0.3", rhi="2.0", meta="1.5", meta_p="0.2",
                                  obs_smd="Null-negative")])
    assert s["same_direction"] == 0 and s["meta_original_plus_replication_significant"] == 0
    s = rpcb.evaluate_effect([row(es_type="Hazard ratio", orig="4.0", meta="1.5", meta_p="0.01")])
    assert s["meta_original_plus_replication_significant"] == 1


def test_score_filters_positive_and_groups_types():
    effects = [row(paper="1", exp="1"), row(paper="1", exp="2"), row(paper="2", exp="1"),
               row(paper="2", exp="2", expected="Null")]
    experiments = [{"Paper #": "1", "Experiment #": "1", "Type of experiment": "Animal"},
                   {"Paper #": "1", "Experiment #": "2", "Type of experiment": "Cell-based"},
                   {"Paper #": "2", "Experiment #": "1", "Type of experiment": "Recombinant"},
                   {"Paper #": "2", "Experiment #": "2", "Type of experiment": "Animal"}]
    scored = rpcb.score_effects(effects, experiments)
    assert [s["type_group"] for s in scored] == ["animal", "cell_based", "other"]
    summ = rpcb.summarize(scored)
    assert summ["n_effects"] == 3
    assert summ["overall"]["same_direction"]["n"] == 3
    assert summ["by_experiment_type"]["animal"]["n_effects"] == 1
    papers = [{"Paper #": "1", "Original paper title": "Alpha beta gamma"},
              {"Paper #": "2", "Original paper title": "Delta"},
              {"Paper #": "3", "Original paper title": "No effects"}]
    table = rpcb.paper_table(papers, experiments, effects, {rpcb.PAPER_PMIDS["1"]: "Alpha beta gamma."})
    assert [p["paper"] for p in table] == ["1", "2"]
    assert table[0]["pmid_verified"] and table[0]["rpcb_any_animal_replicated"]
    # paper 2's animal experiment only has a Null effect, still replicated/coded
    assert table[1]["rpcb_any_animal_identified"]


def test_spearman_guard():
    assert rpcb.spearman([1, 2, 3], [0, 1, 1])["rho"] is None
    r = rpcb.spearman([1, 2, 3, 4, 5, 6], [0, 0, 0, 1, 1, 1])
    assert r["rho"] > 0.8 and r["ci95"][0] < r["rho"] < r["ci95"][1]
