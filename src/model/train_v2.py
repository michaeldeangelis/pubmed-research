"""Train v2 mixers x seeds with a next-year split and early stopping."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch
import torch.nn.functional as F

from src.config_v2 import (
    BATCH_SIZE,
    CHECKPOINTS_V2,
    EMBEDDINGS_PATH,
    ENRICHED_PATH,
    EPOCHS_MAX,
    GRAD_CLIP,
    LOSS_WEIGHT_CLINICAL,
    LOSS_WEIGHT_DRUG,
    LOSS_WEIGHT_GENE,
    LOSS_WEIGHT_NOVEL_DRUG,
    LOSS_WEIGHT_NOVEL_GENE,
    LR,
    MIN_WINDOW,
    MIXERS,
    OUTPUTS_V2,
    PATIENCE,
    SEEDS,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    WEIGHT_DECAY,
)
from src.model.dataset_v2 import build_windows_v2, collate_windows_v2
from src.model.transformer_v2 import PaperTransformerV2

COMPONENTS = ("clinical", "genes", "drugs", "novel_genes", "novel_drugs")
WEIGHTS = {
    "clinical": LOSS_WEIGHT_CLINICAL,
    "genes": LOSS_WEIGHT_GENE,
    "drugs": LOSS_WEIGHT_DRUG,
    "novel_genes": LOSS_WEIGHT_NOVEL_GENE,
    "novel_drugs": LOSS_WEIGHT_NOVEL_DRUG,
}
_TARGETS = {
    "clinical": ("logits_clinical", "y_clinical"),
    "genes": ("logits_genes", "y_genes"),
    "drugs": ("logits_drugs", "y_drugs"),
    "novel_genes": ("logits_novel_genes", "y_novel_genes"),
    "novel_drugs": ("logits_novel_drugs", "y_novel_drugs"),
}


def _device() -> torch.device:
    override = os.environ.get("V2_DEVICE")
    if override:
        return torch.device(override)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _loss_sums(outputs: dict, batch: dict, split: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Per component: (weighted BCE sum, weight sum) over positions in `split`."""
    mask = (batch["split"] == split).to(batch["pad_mask"].dtype)
    sums = {}
    for name in COMPONENTS:
        logit_key, target_key = _TARGETS[name]
        per_element = F.binary_cross_entropy_with_logits(
            outputs[logit_key], batch[target_key], reduction="none"
        )
        weight = mask
        while weight.ndim < per_element.ndim:
            weight = weight.unsqueeze(-1)
        weight = weight.expand_as(per_element)
        if name in ("genes", "novel_genes"):
            weight = weight * batch["gene_target_mask"]
        sums[name] = ((per_element * weight).sum(), weight.sum())
    return sums


def _combine(means: dict) -> dict:
    means["total"] = sum(WEIGHTS[name] * means[name] for name in COMPONENTS)
    return means


def prediction_loss_v2(outputs: dict, batch: dict, split: int) -> dict[str, torch.Tensor]:
    """Masked mean BCE per component on `split` positions, plus weighted total."""
    sums = _loss_sums(outputs, batch, split)
    return _combine({name: total / count.clamp(min=1.0) for name, (total, count) in sums.items()})


def _to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def _evaluate(model, windows, device, splits) -> dict[int, dict[str, float]]:
    """Exact masked means over every window for each split."""
    model.eval()
    acc = {split: {name: [0.0, 0.0] for name in COMPONENTS} for split in splits}
    for start in range(0, len(windows), BATCH_SIZE):
        batch = _to_device(collate_windows_v2(windows[start : start + BATCH_SIZE]), device)
        outputs = model(batch)
        for split in splits:
            for name, (total, count) in _loss_sums(outputs, batch, split).items():
                acc[split][name][0] += float(total)
                acc[split][name][1] += float(count)
    return {
        split: _combine({name: s / max(c, 1.0) for name, (s, c) in parts.items()})
        for split, parts in acc.items()
    }


def _save(model, mixer: str, seed: int, best_epoch: int) -> None:
    CHECKPOINTS_V2.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save(
        {"state_dict": state, "mixer": mixer, "seed": seed, "best_epoch": best_epoch},
        CHECKPOINTS_V2 / f"{mixer}_s{seed}.pt",
    )


def train_one(mixer: str, seed: int, windows: list[list[dict]]) -> dict:
    """Train one mixer/seed; keep the best-val epoch; save and return its losses."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    device = _device()
    model = PaperTransformerV2(mixer).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    order = list(range(len(windows)))

    best = None
    stale = 0
    curve: list[list[float]] = []
    for epoch in range(1, EPOCHS_MAX + 1):
        model.train()
        rng.shuffle(order)
        running, steps = 0.0, 0
        for start in range(0, len(order), BATCH_SIZE):
            batch_windows = [windows[index] for index in order[start : start + BATCH_SIZE]]
            batch = _to_device(collate_windows_v2(batch_windows), device)
            if float(batch["loss_mask"].sum()) == 0.0:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = prediction_loss_v2(model(batch), batch, SPLIT_TRAIN)["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            running += float(loss.detach())
            steps += 1
        train_total = running / max(steps, 1)

        scores = _evaluate(model, windows, device, (SPLIT_VAL, SPLIT_TEST))
        val, test = scores[SPLIT_VAL], scores[SPLIT_TEST]
        curve.append([epoch, train_total, val["total"], test["total"]])
        print(
            f"{mixer} s{seed} epoch {epoch}/{EPOCHS_MAX} train {train_total:.4f} "
            f"val {val['total']:.4f} test {test['total']:.4f}",
            flush=True,
        )
        if best is None or val["total"] < best["val"]["total"]:
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best = {"epoch": epoch, "val": val, "test": test, "state": state}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    model.load_state_dict(best["state"])
    _save(model, mixer, seed, best["epoch"])
    return {
        "mixer": mixer,
        "seed": seed,
        "best_epoch": best["epoch"],
        "val": best["val"],
        "test": best["test"],
        "curve": curve,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixers", nargs="+", default=list(MIXERS), choices=list(MIXERS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--epochs-max", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    global EPOCHS_MAX
    args = _parse_args(argv)
    if args.epochs_max is not None:
        EPOCHS_MAX = args.epochs_max
    try:
        windows = build_windows_v2(ENRICHED_PATH, EMBEDDINGS_PATH)
    except FileNotFoundError as exc:
        raise SystemExit(f"Training data not found: {exc}") from exc
    if not windows:
        raise SystemExit(
            "No training windows. Gene timelines need at least "
            f"{MIN_WINDOW} papers in {ENRICHED_PATH}."
        )

    print(f"{len(windows)} windows on {_device()}", flush=True)
    OUTPUTS_V2.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_V2 / "training.json"
    runs: list[dict] = []
    for mixer in args.mixers:
        for seed in args.seeds:
            runs.append(train_one(mixer, seed, windows))
            out_path.write_text(json.dumps({"runs": runs}, indent=2) + "\n")
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
