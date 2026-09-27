"""Held-out linear probes, attention stats, and selective ablation.

Genetics labels are computed from window evidence tags. They are not model
inputs and they are not loss targets.
"""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F

_PROBE_STEPS = 400


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Rank AUROC via the Mann-Whitney U statistic. Ties use average ranks."""
    scores = scores.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
    labels = labels.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
    if scores.numel() == 0:
        return 0.5
    positive = labels >= 0.5
    n_pos = int(positive.sum())
    n_neg = int(scores.numel() - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = torch.argsort(scores, stable=True)
    sorted_scores = scores[order]
    n = int(scores.numel())
    change = torch.ones(n, dtype=torch.bool)
    change[1:] = sorted_scores[1:] != sorted_scores[:-1]
    group = torch.cumsum(change, dim=0) - 1
    n_groups = int(group[-1]) + 1
    counts = torch.zeros(n_groups, dtype=torch.float64)
    sums = torch.zeros(n_groups, dtype=torch.float64)
    indices = torch.arange(n, dtype=torch.float64)
    counts.scatter_add_(0, group, torch.ones(n, dtype=torch.float64))
    sums.scatter_add_(0, group, indices)
    rank_sorted = (sums / counts)[group] + 1.0
    ranks = torch.empty(n, dtype=torch.float64)
    ranks[order] = rank_sorted

    sum_pos_ranks = float(ranks[positive].sum())
    u = sum_pos_ranks - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def selective_ablation_deltas(
    clinical_bce_before: torch.Tensor,
    clinical_bce_after: torch.Tensor,
    prior_genetics: torch.Tensor,
) -> tuple[float, float]:
    """Mean BCE increase on prior-genetics rows, then on the other rows."""
    delta = (clinical_bce_after - clinical_bce_before).reshape(-1).float()
    prior = prior_genetics.reshape(-1)
    return _mean_or_zero(delta[prior == 1]), _mean_or_zero(delta[prior == 0])


def fit_direction(hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Weight vector of a bias + linear BCE probe. Zeros if only one class."""
    if hidden.ndim != 2:
        raise ValueError(f"hidden must be [N, D], got {tuple(hidden.shape)}")
    x = hidden.detach()
    y = labels.detach().reshape(-1).to(device=x.device, dtype=x.dtype)
    width = int(x.shape[-1])
    if y.numel() == 0 or bool(torch.all(y == y[0])):
        return torch.zeros(width, device=x.device, dtype=x.dtype)

    layer = torch.nn.Linear(width, 1).to(device=x.device, dtype=x.dtype)
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.05, weight_decay=1e-2)
    target = y.unsqueeze(-1)
    with torch.enable_grad():
        for _ in range(_PROBE_STEPS):
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(layer(x), target)
            loss.backward()
            optimizer.step()
    return layer.weight.detach().reshape(-1)


def project_out(hidden: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Remove the component of hidden along a unit-normalized direction."""
    unit = direction.detach().reshape(-1).to(device=hidden.device, dtype=hidden.dtype)
    norm = torch.linalg.vector_norm(unit)
    if float(norm) < 1e-8:
        return hidden
    unit = unit / norm
    coefficient = (hidden * unit).sum(dim=-1, keepdim=True)
    return hidden - coefficient * unit


def run_probe() -> dict:
    """Fit held-out probes on a frozen checkpoint and write the report."""
    from src import config
    from src.model.dataset import build_windows, collate_windows
    from src.model.transformer import PaperTransformer
    from src.schema import PROBE_REPORT_KEYS, prior_genetic_evidence

    _require_inputs(config)
    windows = list(build_windows(config.ENRICHED_PATH, config.EMBEDDINGS_PATH))
    model = _load_model(config.CHECKPOINT_PATH, PaperTransformer)
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
    head_mass = torch.zeros(config.N_HEADS, dtype=torch.float64)
    head_count = 0
    pmids: set[str] = set()
    anonymous = 0

    with torch.no_grad():
        for start in range(0, len(windows), 16):
            chunk = windows[start : start + 16]
            batch = collate_windows(chunk)
            out = model(batch)
            hidden = out["hidden"]
            attn_index = out["attn_index"]
            attn_weight = out["attn_weight"]
            pad = batch["pad_mask"]
            clinical_y = batch["y_clinical"]
            text = batch["text"]
            if attn_index.shape[1] != config.N_HEADS:
                raise RuntimeError(
                    f"attn_index has {attn_index.shape[1]} heads, expected {config.N_HEADS}"
                )
            for b, window in enumerate(chunk):
                tags = [_tag_list(paper) for paper in window]
                genetics = ["human_genetics" in row for row in tags]
                n = len(window)
                for paper in window:
                    pmid = paper.get("pmid")
                    if pmid:
                        pmids.add(str(pmid))
                    else:
                        anonymous += 1
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

                    # collate_windows stores the next paper's clinical flag at this
                    # position. The last real token has no target and is skipped.
                    has_next = (
                        i + 1 < n
                        and i + 1 < pad.shape[1]
                        and float(pad[b, i + 1]) > 0
                    )
                    if has_next:
                        abl_hidden.append(hidden[b, i].detach().to("cpu"))
                        abl_y.append(float(clinical_y[b, i]))
                        abl_year.append(year)
                        abl_prior.append(1.0 if prior else 0.0)

                    if year > config.VAL_YEAR_MAX and prior:
                        mass = _genetics_attention(
                            attn_index[b, :, i, :],
                            attn_weight[b, :, i, :],
                            genetics,
                            i,
                        )
                        head_mass += mass
                        head_count += 1

    features_text, features_hidden, year_t, prior_t, resist_t = _stack_rows(
        text_rows,
        hidden_rows,
        years,
        prior_rows,
        resistance_rows,
        config.TEXT_DIM,
        config.D_MODEL,
    )
    train = year_t <= config.TRAIN_YEAR_MAX
    test = year_t > config.VAL_YEAR_MAX
    text_direction, frozen_auc = _fit_and_score(features_text, prior_t, train, test)
    hidden_direction, contextual_auc = _fit_and_score(features_hidden, prior_t, train, test)
    _resistance_direction, resistance_auc = _fit_and_score(
        features_hidden, resist_t, train, test
    )
    del text_direction, _resistance_direction

    delta_pos, delta_neg = _ablation_deltas(
        model, abl_hidden, abl_y, abl_year, abl_prior, hidden_direction, config.VAL_YEAR_MAX
    )
    if head_count == 0:
        head_attention = [0.0] * config.N_HEADS
    else:
        head_attention = [float(value) for value in (head_mass / head_count)]

    report = {
        "encoder": _encoder_name(config.EMBEDDING_META_PATH),
        "n_papers": int(len(pmids) + anonymous),
        "n_test_positions": int(test.sum()),
        "prior_genetics_auroc_frozen": float(frozen_auc),
        "prior_genetics_auroc_contextual": float(contextual_auc),
        "prior_genetics_auroc_lift": float(contextual_auc - frozen_auc),
        "resistance_auroc_contextual": float(resistance_auc),
        "ablation_delta_clinical_bce_genetics_context": float(delta_pos),
        "ablation_delta_clinical_bce_no_genetics_context": float(delta_neg),
        "selective_ablation": bool(delta_pos > delta_neg),
        "head_genetics_attention": head_attention,
        "notes": (
            f"Split: fit on year <= {config.TRAIN_YEAR_MAX}, "
            f"evaluate on year > {config.VAL_YEAR_MAX}. "
            "Genetics was held out of the loss."
        ),
    }
    missing = [key for key in PROBE_REPORT_KEYS if key not in report]
    extra = [key for key in report if key not in PROBE_REPORT_KEYS]
    if missing or extra or len(head_attention) != config.N_HEADS:
        raise RuntimeError(f"probe report keys mismatch: missing={missing} extra={extra}")
    ordered = {key: report[key] for key in PROBE_REPORT_KEYS}
    path = config.PROBE_REPORT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8")
    return ordered


def _mean_or_zero(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.detach().float().mean())


def _require_inputs(config) -> None:
    missing = [
        str(path)
        for path in (config.CHECKPOINT_PATH, config.ENRICHED_PATH, config.EMBEDDINGS_PATH)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Cannot run the held-out probe; missing file(s): " + ", ".join(missing)
        )


def _load_model(path, model_cls):
    checkpoint = _torch_load(path)
    if isinstance(checkpoint, model_cls):
        model = checkpoint
    else:
        model = model_cls()
        model.load_state_dict(_state_dict(checkpoint))
    model.eval()
    return model


def _state_dict(checkpoint) -> dict:
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unrecognized checkpoint type: {type(checkpoint)!r}")
    payload = checkpoint
    for key in ("model", "state_dict", "model_state_dict", "model_state"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            payload = value
            break
    state = {}
    for key, value in payload.items():
        if not torch.is_tensor(value):
            continue
        if key.startswith("module."):
            key = key[len("module.") :]
        state[key] = value
    if not state:
        raise ValueError(f"Checkpoint has no tensor state dict: {_path_hint(checkpoint)}")
    return state


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        # Training checkpoints sometimes include non-tensor metadata.
        return torch.load(path, map_location="cpu", weights_only=False)


def _path_hint(checkpoint) -> str:
    if isinstance(checkpoint, dict):
        return ", ".join(list(checkpoint.keys())[:8])
    return type(checkpoint).__name__


def _encoder_name(path) -> str:
    if not path.is_file():
        return "unknown"
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "unknown"
    if isinstance(meta, str) and meta.strip():
        return meta.strip()
    if isinstance(meta, dict):
        for key in ("encoder", "encoder_name", "model", "model_name", "name"):
            value = meta.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return "unknown"


def _tag_list(paper: dict) -> list[str]:
    tags = paper.get("evidence_tags") or []
    return list(tags)


def _position_year(paper: dict, batch: dict, b: int, i: int) -> int:
    if "year" in paper:
        return int(paper["year"])
    return int(batch["year"][b, i].item())


def _genetics_attention(
    index: torch.Tensor,
    weight: torch.Tensor,
    genetics: list[bool],
    query: int,
) -> torch.Tensor:
    """Sum of attention weight on earlier human_genetics positions, per head."""
    idx = index.detach().to("cpu").long()
    attn = weight.detach().to("cpu").float()
    n = len(genetics)
    if n == 0:
        return torch.zeros(idx.shape[0], dtype=torch.float64)
    flags = torch.tensor(genetics, dtype=torch.bool)
    valid = (idx >= 0) & (idx < query) & (idx < n)
    safe = idx.clamp(min=0, max=n - 1)
    hit = flags[safe] & valid
    return (attn * hit.float()).sum(dim=-1).to(dtype=torch.float64)


def _stack_rows(text_rows, hidden_rows, years, prior_rows, resistance_rows, text_dim, d_model):
    if hidden_rows:
        text = torch.stack(text_rows).float()
        hidden = torch.stack(hidden_rows).float()
    else:
        text = torch.zeros(0, text_dim)
        hidden = torch.zeros(0, d_model)
    year = torch.tensor(years, dtype=torch.long)
    prior = torch.tensor(prior_rows, dtype=torch.float32)
    resistance = torch.tensor(resistance_rows, dtype=torch.float32)
    return text, hidden, year, prior, resistance


def _fit_and_score(features, labels, train, test) -> tuple[torch.Tensor, float]:
    if int(train.sum()) == 0:
        direction = torch.zeros(features.shape[-1], dtype=features.dtype)
    else:
        direction = fit_direction(features[train], labels[train])
    if int(test.sum()) == 0:
        return direction, 0.5
    scores = features[test] @ direction.to(dtype=features.dtype)
    return direction, auroc(scores, labels[test])


def _ablation_deltas(model, hidden_rows, targets, years, prior, direction, val_year_max):
    if not hidden_rows:
        return 0.0, 0.0
    hidden = torch.stack(hidden_rows).float()
    y = torch.tensor(targets, dtype=hidden.dtype)
    year = torch.tensor(years, dtype=torch.long)
    prior_t = torch.tensor(prior, dtype=torch.float32)
    test = year > val_year_max
    if int(test.sum()) == 0:
        return 0.0, 0.0
    with torch.no_grad():
        before = _logit_vector(model.clinical_head(hidden))
        after = _logit_vector(model.clinical_head(project_out(hidden, direction)))
        bce_before = F.binary_cross_entropy_with_logits(before, y, reduction="none")
        bce_after = F.binary_cross_entropy_with_logits(after, y, reduction="none")
    return selective_ablation_deltas(bce_before[test], bce_after[test], prior_t[test])


def _logit_vector(logit: torch.Tensor) -> torch.Tensor:
    if logit.ndim > 1 and logit.shape[-1] == 1:
        logit = logit.squeeze(-1)
    return logit.reshape(-1)
