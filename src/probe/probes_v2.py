"""v2 probes: per-checkpoint genetics probe, paired bootstrap, decision rule.

Genetics labels come from window evidence tags, as in v1. They are not model
inputs and not loss targets. The probe split matches v1 (fit on the position's
own year <= TRAIN_YEAR_MAX, score on year > VAL_YEAR_MAX) so v1 and v2 compare.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.probe.probes import (
    _ablation_deltas,
    _genetics_attention,
    _position_year,
    _state_dict,
    _tag_list,
    _torch_load,
    auroc,
)

_CHUNK = 16


def probe_checkpoint(path, windows) -> dict:
    """Frozen vs contextual genetics probe on one v2 checkpoint."""
    from src import config_v2 as cfg
    from src.model.dataset_v2 import collate_windows_v2
    from src.model.transformer_v2 import PaperTransformerV2
    from src.schema import prior_genetic_evidence

    checkpoint = _torch_load(path)
    if not isinstance(checkpoint, dict) or "mixer" not in checkpoint:
        raise ValueError(f"{path}: expected a v2 checkpoint dict with a 'mixer' key")
    mixer = str(checkpoint["mixer"])
    model = PaperTransformerV2(mixer)
    model.load_state_dict(_state_dict(checkpoint))
    model.eval()

    text_rows: list[torch.Tensor] = []
    hidden_rows: list[torch.Tensor] = []
    years: list[int] = []
    prior_rows: list[float] = []
    resistance_rows: list[float] = []
    abl_hidden: list[torch.Tensor] = []
    abl_y: list[float] = []
    abl_year: list[int] = []
    abl_prior: list[float] = []
    layer_mass: torch.Tensor | None = None
    uniform_mass = 0.0
    head_count = 0

    with torch.no_grad():
        for start in range(0, len(windows), _CHUNK):
            chunk = windows[start : start + _CHUNK]
            batch = collate_windows_v2(chunk)
            out = model(batch)
            hidden = out["hidden"]
            index_layers = out["attn_index_layers"]
            weight_layers = out["attn_weight_layers"]
            if layer_mass is None:
                layer_mass = torch.zeros(
                    len(index_layers), int(index_layers[0].shape[1]), dtype=torch.float64
                )
            pad = batch["pad_mask"]
            clinical_y = batch["y_clinical"]
            text = batch["text"]
            for b, window in enumerate(chunk):
                tags = [_tag_list(paper) for paper in window]
                genetics = ["human_genetics" in row for row in tags]
                n = len(window)
                for i, paper in enumerate(window):
                    if float(pad[b, i]) <= 0:
                        continue
                    year = _position_year(paper, batch, b, i)
                    prior = prior_genetic_evidence(tags[:i])
                    text_rows.append(text[b, i].detach().to("cpu"))
                    hidden_rows.append(hidden[b, i].detach().to("cpu"))
                    years.append(year)
                    prior_rows.append(1.0 if prior else 0.0)
                    resistance_rows.append(1.0 if "resistance" in tags[i] else 0.0)

                    # y_clinical at t is paper t+1's flag; the last real token has none.
                    has_next = i + 1 < n and i + 1 < pad.shape[1] and float(pad[b, i + 1]) > 0
                    if has_next:
                        abl_hidden.append(hidden[b, i].detach().to("cpu"))
                        abl_y.append(float(clinical_y[b, i]))
                        abl_year.append(year)
                        abl_prior.append(1.0 if prior else 0.0)

                    if year > cfg.VAL_YEAR_MAX and prior:
                        for layer, (idx, wt) in enumerate(zip(index_layers, weight_layers)):
                            layer_mass[layer] += _genetics_attention(
                                idx[b, :, i, :], wt[b, :, i, :], genetics, i
                            )
                        uniform_mass += _uniform_genetics_mass(genetics, i, cfg.TOP_K)
                        head_count += 1

    if hidden_rows:
        features_text = torch.stack(text_rows).float()
        features_hidden = torch.stack(hidden_rows).float()
    else:
        features_text = torch.zeros(0, cfg.TEXT_DIM)
        features_hidden = torch.zeros(0, cfg.D_MODEL)
    year_t = torch.tensor(years, dtype=torch.long)
    prior_t = torch.tensor(prior_rows, dtype=torch.float32)
    resist_t = torch.tensor(resistance_rows, dtype=torch.float32)
    train = year_t <= cfg.TRAIN_YEAR_MAX
    test = year_t > cfg.VAL_YEAR_MAX

    _, frozen_scores = _fit_and_scores(features_text, prior_t, train, test)
    hidden_direction, contextual_scores = _fit_and_scores(features_hidden, prior_t, train, test)
    _, resistance_scores = _fit_and_scores(features_hidden, resist_t, train, test)
    test_labels = prior_t[test]
    frozen_auc = auroc(frozen_scores, test_labels)
    contextual_auc = auroc(contextual_scores, test_labels)
    resistance_auc = auroc(resistance_scores, resist_t[test])

    delta_pos, delta_neg = _ablation_deltas(
        model, abl_hidden, abl_y, abl_year, abl_prior, hidden_direction, cfg.VAL_YEAR_MAX
    )
    if layer_mass is None:
        head_attention: list[list[float]] = []
    elif head_count == 0:
        head_attention = [[0.0] * layer_mass.shape[1] for _ in range(layer_mass.shape[0])]
    else:
        head_attention = (layer_mass / head_count).tolist()

    return {
        "mixer": mixer,
        "seed": checkpoint.get("seed"),
        "best_epoch": checkpoint.get("best_epoch"),
        "n_test": int(test.sum()),
        "n_test_positive": int(test_labels.sum()),
        "frozen_auroc": float(frozen_auc),
        "contextual_auroc": float(contextual_auc),
        "lift": float(contextual_auc - frozen_auc),
        "resistance_auroc": float(resistance_auc),
        "ablation_delta_clinical_bce_genetics_context": float(delta_pos),
        "ablation_delta_clinical_bce_no_genetics_context": float(delta_neg),
        "selective_ablation": bool(delta_pos > delta_neg),
        "head_genetics_attention": head_attention,
        "n_attention_positions": int(head_count),
        "uniform_reference": float(uniform_mass / head_count) if head_count else 0.0,
        "test_labels": [float(v) for v in test_labels.tolist()],
        "test_scores_frozen": [float(v) for v in frozen_scores.tolist()],
        "test_scores_contextual": [float(v) for v in contextual_scores.tolist()],
    }


def bootstrap_lift(per_seed: list[dict], n: int, rng_seed: int) -> dict:
    """Paired bootstrap of the seed-averaged lift over shared test positions."""
    if not per_seed:
        raise ValueError("bootstrap_lift needs at least one seed")
    labels = np.asarray(per_seed[0]["test_labels"], dtype=np.float64)
    for entry in per_seed[1:]:
        other = np.asarray(entry["test_labels"], dtype=np.float64)
        if other.shape != labels.shape or not np.array_equal(other, labels):
            raise ValueError("bootstrap_lift: seeds do not share the same test labels")
    size = labels.size
    scores = []
    for entry in per_seed:
        frozen = np.asarray(entry["test_scores_frozen"], dtype=np.float64)
        contextual = np.asarray(entry["test_scores_contextual"], dtype=np.float64)
        if frozen.shape != labels.shape or contextual.shape != labels.shape:
            raise ValueError("bootstrap_lift: score and label lengths differ")
        scores.append((frozen, contextual))

    full = np.ones((1, size))
    mean = float(
        np.mean([_weighted_auroc(c, labels, full)[0] - _weighted_auroc(f, labels, full)[0]
                 for f, c in scores])
    )

    rng = np.random.default_rng(rng_seed)
    counts = np.zeros((n, size))
    if size:
        draws = rng.integers(0, size, size=(n, size))
        np.add.at(counts, (np.arange(n)[:, None], draws), 1.0)
    lifts = np.zeros(n)
    for frozen, contextual in scores:
        lifts += _weighted_auroc(contextual, labels, counts) - _weighted_auroc(
            frozen, labels, counts
        )
    lifts /= len(scores)
    return {
        "mean": mean,
        "lo": float(np.percentile(lifts, 2.5)),
        "hi": float(np.percentile(lifts, 97.5)),
        "n": int(n),
    }


def decide(training: dict, probes: dict, lift_bar: float | None = None) -> dict:
    """Apply the fixed v2 decision rule.

    training: the training.json dict. probes: the bootstrap_lift result for the
    attention mixer.
    """
    if lift_bar is None:
        from src.config_v2 import LIFT_BAR

        lift_bar = LIFT_BAR
    totals = _test_totals(training)
    for mixer in ("attention", "mean"):
        if not totals.get(mixer):
            raise ValueError(f"decide: no test totals for mixer {mixer!r}")
    attention = np.asarray(totals["attention"], dtype=np.float64)
    mean = np.asarray(totals["mean"], dtype=np.float64)
    std_attention = float(np.std(attention, ddof=0))
    std_mean = float(np.std(mean, ddof=0))
    gap = float(mean.mean() - attention.mean())
    threshold = max(std_attention, std_mean)
    c1_pass = bool(attention.mean() < mean.mean() and gap > threshold)

    lift = float(probes["mean"])
    lo = float(probes["lo"])
    c2_pass = bool(lift >= lift_bar and lo > 0)
    return {
        "attention_beats_mean": {
            "mean_test_total_attention": float(attention.mean()),
            "mean_test_total_mean": float(mean.mean()),
            "std_attention": std_attention,
            "std_mean": std_mean,
            "gap": gap,
            "threshold": threshold,
            "pass": c1_pass,
        },
        "genetics_lift": {
            "lift": lift,
            "lo": lo,
            "hi": float(probes["hi"]),
            "lift_bar": float(lift_bar),
            "pass": c2_pass,
        },
        "fruitful": bool(c1_pass and c2_pass),
    }


def _test_totals(training: dict) -> dict[str, list[float]]:
    """training.json -> {mixer: [test total per seed, in seed order]}."""
    grouped: dict[str, list[tuple[int, float]]] = {}
    for run in training.get("runs", []):
        grouped.setdefault(str(run["mixer"]), []).append(
            (int(run["seed"]), float(run["test"]["total"]))
        )
    return {mixer: [total for _, total in sorted(rows)] for mixer, rows in grouped.items()}


def _uniform_genetics_mass(genetics: list[bool], query: int, top_k: int) -> float:
    """Mass a uniform head over the last top_k positions puts on earlier genetics."""
    lo = max(0, query - top_k + 1)
    span = query - lo + 1
    hits = sum(1 for j in range(lo, query) if j < len(genetics) and genetics[j])
    return hits / span


def _weighted_auroc(scores: np.ndarray, labels: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """AUROC per row of integer multiplicities; equals auroc on the resampled set.

    Ties count one half, matching the average-rank Mann-Whitney AUROC.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    counts = np.atleast_2d(np.asarray(counts, dtype=np.float64))
    if scores.size == 0:
        return np.full(counts.shape[0], 0.5)
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    positive = labels[order] >= 0.5
    starts = np.flatnonzero(np.r_[True, sorted_scores[1:] != sorted_scores[:-1]])
    weights = counts[:, order]
    pos = np.add.reduceat(weights * positive, starts, axis=1)
    neg = np.add.reduceat(weights * ~positive, starts, axis=1)
    neg_below = np.cumsum(neg, axis=1) - neg
    numerator = (pos * (neg_below + 0.5 * neg)).sum(axis=1)
    n_pos = pos.sum(axis=1)
    n_neg = neg.sum(axis=1)
    denom = n_pos * n_neg
    out = np.full(counts.shape[0], 0.5)
    ok = denom > 0
    out[ok] = numerator[ok] / denom[ok]
    return out


def fit_direction_v2(hidden: torch.Tensor, labels: torch.Tensor, l2: float = 1e-2) -> torch.Tensor:
    """Deterministic L2 logistic probe: standardized features, zero init, LBFGS.

    v1 fit_direction starts from a random init and stops after 400 SGD steps;
    on frozen text its test AUROC moves by ~0.08 across inits. Returns the
    direction in the raw feature space.
    """
    x = hidden.detach().to(dtype=torch.float64, device="cpu")
    y = labels.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
    width = int(x.shape[-1])
    if y.numel() == 0 or bool(torch.all(y == y[0])):
        return torch.zeros(width, dtype=hidden.dtype)
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp(min=1e-8)
    z = (x - mean) / std
    weight = torch.zeros(width, dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weight, bias], max_iter=500, tolerance_grad=1e-9, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(z @ weight + bias, y)
        loss = loss + 0.5 * l2 * weight.pow(2).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        optimizer.step(closure)
    return (weight.detach() / std).to(dtype=hidden.dtype)


def _fit_and_scores(features, labels, train, test) -> tuple[torch.Tensor, torch.Tensor]:
    if int(train.sum()) == 0:
        direction = torch.zeros(features.shape[-1], dtype=features.dtype)
    else:
        direction = fit_direction_v2(features[train], labels[train])
    scores = features[test] @ direction.to(dtype=features.dtype)
    return direction, scores
