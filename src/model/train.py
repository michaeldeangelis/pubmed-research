"""Train the paper transformer. Data is loaded in main, not at import."""

from __future__ import annotations

import random
import sys

import torch
import torch.nn.functional as F

from src.config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    D_MODEL,
    EMBEDDINGS_PATH,
    ENRICHED_PATH,
    EPOCHS,
    GRAD_CLIP,
    LOSS_WEIGHT_CLINICAL,
    LOSS_WEIGHT_DRUG,
    LOSS_WEIGHT_GENE,
    LR,
    MIN_WINDOW,
    N_HEADS,
    N_LAYERS,
    TOP_K,
    WEIGHT_DECAY,
)
from src.model.dataset import build_windows, collate_windows
from src.model.transformer import PaperTransformer


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _masked_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_element = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weight = mask
    while weight.ndim < per_element.ndim:
        weight = weight.unsqueeze(-1)
    weight = weight.expand_as(per_element)
    return (per_element * weight).sum() / weight.sum().clamp(min=1.0)


def prediction_loss(outputs: dict, batch: dict) -> torch.Tensor:
    """BCE-with-logits on next clinical, gene, and drug labels, masked."""
    mask = batch["loss_mask"]
    return (
        LOSS_WEIGHT_CLINICAL
        * _masked_bce(outputs["logits_clinical"], batch["y_clinical"], mask)
        + LOSS_WEIGHT_GENE * _masked_bce(outputs["logits_genes"], batch["y_genes"], mask)
        + LOSS_WEIGHT_DRUG * _masked_bce(outputs["logits_drugs"], batch["y_drugs"], mask)
    )


def _save_checkpoint(model: PaperTransformer) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cpu_state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save(
        {
            "state_dict": cpu_state,
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "top_k": TOP_K,
        },
        CHECKPOINT_PATH,
    )


def main() -> None:
    try:
        windows = build_windows(ENRICHED_PATH, EMBEDDINGS_PATH)
    except FileNotFoundError as exc:
        raise SystemExit(f"Training data not found: {exc}") from exc
    if not windows:
        raise SystemExit(
            "No training windows. Gene timelines need at least "
            f"{MIN_WINDOW} papers in {ENRICHED_PATH}."
        )

    device = _device()
    model = PaperTransformer().to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    order = list(range(len(windows)))
    rng = random.Random(0)

    for epoch in range(EPOCHS):
        rng.shuffle(order)
        total = 0.0
        steps = 0
        for start in range(0, len(order), BATCH_SIZE):
            batch_windows = [windows[index] for index in order[start : start + BATCH_SIZE]]
            batch = {
                key: value.to(device)
                for key, value in collate_windows(batch_windows).items()
            }
            optimizer.zero_grad(set_to_none=True)
            loss = prediction_loss(model(batch), batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        print(f"epoch {epoch + 1}/{EPOCHS} loss {total / steps:.4f}", flush=True)

    _save_checkpoint(model)
    print(f"saved {CHECKPOINT_PATH}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
