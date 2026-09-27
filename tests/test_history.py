import random

import numpy as np
import torch

from src.config import TEXT_DIM, TRAIN_YEAR_MAX
from src.eval.history import (
    CURRENT_VOLUME_DIM,
    HISTORY_DIM,
    YEAR_VOLUME_DIM,
    calibrate_logits,
    current_volume_features,
    decision_points,
    fit_linear,
    linear_scores,
    mean_history_features,
    replace_history,
    shuffle_context,
    split_points,
    training_windows,
    year_volume_features,
)
from src.probe.probes import auroc


def _paper(year, pmid, clinical=0.0, genes=None):
    text = np.zeros(TEXT_DIM, dtype=np.float32)
    text[int(pmid) % TEXT_DIM] = 1.0
    return {
        "pmid": str(pmid),
        "year": year,
        "date": f"{year}-06-01",
        "text": text,
        "genes": genes or ["KRAS"],
        "drugs": [],
        "diseases": ["NSCLC"],
        "clinical": float(clinical),
    }


def test_training_windows_exclude_later_papers():
    timeline = [_paper(year, index) for index, year in enumerate(range(2014, 2024), start=1)]
    windows = training_windows({"KRAS": timeline})
    years = [paper["year"] for window in windows for paper in window]
    assert years
    assert max(years) <= TRAIN_YEAR_MAX
    assert 2023 not in years


def test_label_is_the_next_paper_and_the_split_drops_the_gap():
    timeline = [
        _paper(2018, 1, clinical=0.0),
        _paper(2019, 2, clinical=1.0),
        _paper(2021, 3, clinical=0.0),
        _paper(2022, 4, clinical=1.0),
    ]
    points = decision_points(timeline, "KRAS")
    assert [point["target_year"] for point in points] == [2019, 2021, 2022]
    assert points[1]["y_clinical"] == 0.0
    train, test = split_points(points)
    assert [point["target_year"] for point in train] == []
    assert [point["target_year"] for point in test] == [2021, 2022]
    earlier = decision_points([_paper(2016, 1), _paper(2017, 2, clinical=1.0)], "KRAS")
    train, test = split_points(earlier)
    assert train[0]["y_clinical"] == 1.0
    assert test == []


def test_shuffle_keeps_the_latest_paper_and_the_history_multiset():
    context = [_paper(2010, 1), _paper(2010, 2), _paper(2011, 3), _paper(2020, 4)]
    shuffled = shuffle_context(context, random.Random(0))
    assert shuffled[-1]["pmid"] == "4"
    assert sorted(paper["pmid"] for paper in shuffled[:-1]) == ["1", "2", "3"]


def test_unrelated_replacement_leaves_the_timeline():
    context = [_paper(2010, 1), _paper(2011, 2), _paper(2020, 3)]
    outsider = _paper(2010, 9, genes=["EGFR"])
    pool = {2010: [_paper(2010, 1), outsider], 2011: [_paper(2011, 8, genes=["EGFR"])]}
    replaced = replace_history(context, {"1", "2", "3"}, pool, random.Random(0))
    assert replaced[-1]["pmid"] == "3"
    assert {paper["pmid"] for paper in replaced[:-1]}.isdisjoint({"1", "2", "3"})


def test_ridge_baseline_stays_finite_on_a_separable_problem():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(80, 12)).astype(np.float32)
    labels = (features[:, 0] > 0).astype(np.float32)
    fitted = fit_linear(features[:60], labels[:60])
    train_scores = linear_scores(*fitted, features[:60]).reshape(-1)
    test_scores = linear_scores(*fitted, features[60:]).reshape(-1)
    logits = calibrate_logits(train_scores, labels[:60], test_scores)
    assert torch.isfinite(logits).all()
    assert auroc(logits, torch.tensor(labels[60:])) > 0.8


def test_history_features_add_a_prior_block_the_trend_model_lacks():
    timeline = [_paper(2016, 1), _paper(2018, 2), _paper(2021, 3)]
    assert year_volume_features(timeline, 1).shape == (YEAR_VOLUME_DIM,)
    assert current_volume_features(timeline, 1).shape == (CURRENT_VOLUME_DIM,)
    assert mean_history_features(timeline, 1).shape == (HISTORY_DIM,)
    assert YEAR_VOLUME_DIM < TEXT_DIM
