import numpy as np
import pytest
import torch

from src.probe.probes import auroc
from src.probe.probes_v2 import (
    _test_totals,
    _uniform_genetics_mass,
    _weighted_auroc,
    bootstrap_lift,
    decide,
)


def _seed(labels, frozen, contextual):
    return {
        "test_labels": list(labels),
        "test_scores_frozen": list(frozen),
        "test_scores_contextual": list(contextual),
    }


def test_weighted_auroc_matches_resampled_auroc():
    rng = np.random.default_rng(3)
    labels = rng.integers(0, 2, size=40).astype(float)
    scores = np.round(rng.normal(size=40), 1)  # rounding forces ties
    idx = rng.integers(0, 40, size=(5, 40))
    counts = np.zeros((5, 40))
    np.add.at(counts, (np.arange(5)[:, None], idx), 1.0)
    got = _weighted_auroc(scores, labels, counts)
    for r in range(5):
        want = auroc(torch.tensor(scores[idx[r]]), torch.tensor(labels[idx[r]]))
        assert got[r] == pytest.approx(want, abs=1e-12)


def test_bootstrap_separable_contextual_beats_random_frozen():
    rng = np.random.default_rng(0)
    labels = np.array([0, 1] * 100, dtype=float)
    per_seed = []
    for _ in range(3):
        frozen = rng.normal(size=labels.size)
        contextual = labels + 0.01 * rng.random(labels.size)
        per_seed.append(_seed(labels, frozen, contextual))
    result = bootstrap_lift(per_seed, n=300, rng_seed=0)
    assert result["n"] == 300
    assert result["lo"] > 0
    assert result["lo"] <= result["mean"] <= result["hi"]
    assert abs(result["mean"] - 0.5) < 0.1


def test_bootstrap_identical_scores_gives_zero():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, size=80).astype(float)
    scores = rng.normal(size=80)
    per_seed = [_seed(labels, scores, scores) for _ in range(2)]
    result = bootstrap_lift(per_seed, n=200, rng_seed=0)
    assert result["mean"] == pytest.approx(0.0, abs=1e-12)
    assert result["lo"] <= 0.0 <= result["hi"]


def test_bootstrap_is_seeded():
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 2, size=60).astype(float)
    per_seed = [_seed(labels, rng.normal(size=60), rng.normal(size=60))]
    a = bootstrap_lift(per_seed, n=100, rng_seed=7)
    b = bootstrap_lift(per_seed, n=100, rng_seed=7)
    assert a == b


def test_bootstrap_rejects_mismatched_labels():
    per_seed = [
        _seed([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]),
        _seed([1, 0, 0, 1], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]),
    ]
    with pytest.raises(ValueError):
        bootstrap_lift(per_seed, n=10, rng_seed=0)


def _training(attention, mean, self_totals=(1.0, 1.0, 1.0)):
    runs = []
    for mixer, totals in (("attention", attention), ("mean", mean), ("self", self_totals)):
        for seed, total in enumerate(totals):
            runs.append(
                {
                    "mixer": mixer,
                    "seed": seed,
                    "best_epoch": 3,
                    "val": {"total": total + 0.1},
                    "test": {"clinical": 0.3, "total": total},
                    "curve": [[1, 1.0, 1.0, 1.0]],
                }
            )
    return {"runs": runs}


def test_test_totals_adapter_groups_by_mixer_in_seed_order():
    training = _training([0.5, 0.6, 0.7], [0.9, 0.8, 1.0])
    training["runs"].reverse()
    totals = _test_totals(training)
    assert totals["attention"] == [0.5, 0.6, 0.7]
    assert totals["mean"] == [0.9, 0.8, 1.0]
    assert set(totals) == {"attention", "mean", "self"}


_GOOD_BOOT = {"mean": 0.05, "lo": 0.01, "hi": 0.09, "n": 1000}


def test_decide_both_pass():
    result = decide(_training([0.50, 0.51, 0.49], [0.60, 0.61, 0.59]), _GOOD_BOOT, lift_bar=0.03)
    assert result["attention_beats_mean"]["pass"] is True
    assert result["genetics_lift"]["pass"] is True
    assert result["fruitful"] is True


def test_decide_fails_when_attention_not_lower():
    result = decide(_training([0.60, 0.61, 0.59], [0.50, 0.51, 0.49]), _GOOD_BOOT, lift_bar=0.03)
    assert result["attention_beats_mean"]["pass"] is False
    assert result["fruitful"] is False


def test_decide_fails_when_gap_within_std():
    # gap 0.02, attention std ~0.082 -> not enough
    result = decide(_training([0.40, 0.50, 0.60], [0.52, 0.52, 0.52]), _GOOD_BOOT, lift_bar=0.03)
    c1 = result["attention_beats_mean"]
    assert c1["gap"] == pytest.approx(0.02)
    assert c1["threshold"] == pytest.approx(np.std([0.4, 0.5, 0.6]))
    assert c1["pass"] is False
    assert result["fruitful"] is False


def test_decide_fails_on_small_lift():
    boot = {"mean": 0.02, "lo": 0.005, "hi": 0.04, "n": 1000}
    result = decide(_training([0.50, 0.51, 0.49], [0.60, 0.61, 0.59]), boot, lift_bar=0.03)
    assert result["attention_beats_mean"]["pass"] is True
    assert result["genetics_lift"]["pass"] is False
    assert result["fruitful"] is False


def test_decide_fails_when_interval_touches_zero():
    boot = {"mean": 0.05, "lo": 0.0, "hi": 0.1, "n": 1000}
    result = decide(_training([0.50, 0.51, 0.49], [0.60, 0.61, 0.59]), boot, lift_bar=0.03)
    assert result["genetics_lift"]["pass"] is False
    assert result["fruitful"] is False


def test_uniform_reference_counts_earlier_genetics_in_last_top_k():
    genetics = [True, False, True, False, False]
    # query 4, top_k 3 -> positions 2,3,4; earlier genetics: 2 -> 1/3
    assert _uniform_genetics_mass(genetics, 4, 3) == pytest.approx(1 / 3)
    # query 2, top_k 8 -> positions 0,1,2; earlier genetics: 0 (2 is the query) -> 1/3
    assert _uniform_genetics_mass(genetics, 2, 8) == pytest.approx(1 / 3)
    assert _uniform_genetics_mass(genetics, 0, 8) == 0.0


def _synthetic_windows(n_windows=6, length=8):
    from src.config_v2 import TEXT_DIM
    from src.schema import GENES

    gen = torch.Generator().manual_seed(0)
    windows = []
    for w in range(n_windows):
        window = []
        for i in range(length):
            # Years span the probe train (<= 2018) and test (> 2020) ranges.
            year = 2014 + i if w % 2 == 0 else 2018 + i
            tags = ["human_genetics"] if (i + w) % 3 == 0 else []
            if (i + w) % 4 == 0:
                tags.append("resistance")
            if i % 2 == 0:
                tags.append("clinical")
            window.append(
                {
                    "pmid": f"{w}-{i}",
                    "year": year,
                    "date": f"{year}-01-01",
                    "text": torch.randn(TEXT_DIM, generator=gen),
                    "genes": [GENES[0], GENES[(i + 1) % len(GENES)]],
                    "drugs": [],
                    "diseases": [],
                    "clinical": 1.0 if "clinical" in tags else 0.0,
                    "evidence_tags": tags,
                    "anchor": GENES[0],
                }
            )
        windows.append(window[: length - (w % 3)])
    return windows


@pytest.mark.parametrize("mixer", ["attention", "mean"])
def test_probe_checkpoint_end_to_end(tmp_path, mixer):
    from src.config_v2 import N_HEADS, N_LAYERS
    from src.model.transformer_v2 import PaperTransformerV2
    from src.probe.probes_v2 import probe_checkpoint

    torch.manual_seed(0)
    model = PaperTransformerV2(mixer)
    path = tmp_path / f"{mixer}_s0.pt"
    torch.save({"state_dict": model.state_dict(), "mixer": mixer, "seed": 0, "best_epoch": 1}, path)

    result = probe_checkpoint(path, _synthetic_windows())
    assert result["mixer"] == mixer
    assert result["n_test"] > 0
    assert len(result["test_labels"]) == result["n_test"]
    assert len(result["test_scores_frozen"]) == result["n_test"]
    assert len(result["test_scores_contextual"]) == result["n_test"]
    assert 0.0 <= result["frozen_auroc"] <= 1.0
    assert 0.0 <= result["contextual_auroc"] <= 1.0
    assert result["lift"] == pytest.approx(result["contextual_auroc"] - result["frozen_auroc"])
    heads = result["head_genetics_attention"]
    assert len(heads) == N_LAYERS and all(len(row) == N_HEADS for row in heads)
    assert all(0.0 <= v <= 1.0 + 1e-6 for row in heads for v in row)
    assert result["n_attention_positions"] > 0
    assert 0.0 < result["uniform_reference"] <= 1.0
    if mixer == "mean":
        # The mean mixer is the uniform head, so every head equals the reference.
        for row in heads:
            for value in row:
                assert value == pytest.approx(result["uniform_reference"], abs=1e-5)

    boot = bootstrap_lift([result, result], n=50, rng_seed=0)
    assert boot["mean"] == pytest.approx(result["lift"], abs=1e-9)


def test_v2_main_fails_clearly_without_inputs(tmp_path, monkeypatch):
    from src import config_v2
    from src.probe import v2

    monkeypatch.setattr(config_v2, "OUTPUTS_V2", tmp_path / "outputs")
    monkeypatch.setattr(config_v2, "CHECKPOINTS_V2", tmp_path / "checkpoints")
    with pytest.raises(SystemExit, match="training.json"):
        v2.main()
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "training.json").write_text('{"runs": []}')
    with pytest.raises(SystemExit, match="attention_s0.pt"):
        v2.main()
