"""Causal top-k transformer. Each token is one paper."""

from __future__ import annotations

import math

import torch
from torch import nn

from src.config import (
    D_MODEL,
    DROPOUT,
    N_HEADS,
    N_LAYERS,
    TEXT_DIM,
    TOP_K,
    YEAR_MAX,
    YEAR_MIN,
)
from src.schema import DISEASES, DRUGS, GENES


class CausalTopKAttention(nn.Module):
    """Each query keeps at most TOP_K past keys. Unused slots are index -1."""

    def __init__(self, d_model: int, n_heads: int, top_k: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} is not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.top_k = top_k
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split(self, projected: torch.Tensor) -> torch.Tensor:
        batch, length, _ = projected.shape
        return projected.view(batch, length, self.n_heads, self.d_head).permute(0, 2, 1, 3)

    def forward(
        self, hidden: torch.Tensor, pad_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, _ = hidden.shape
        query = self._split(self.q_proj(hidden))
        key = self._split(self.k_proj(hidden))
        value = self._split(self.v_proj(hidden))

        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.d_head)
        positions = torch.arange(length, device=hidden.device)
        future = positions.view(1, 1, 1, length) > positions.view(1, 1, length, 1)
        pad_keys = (pad_mask <= 0)[:, None, None, :]
        scores = scores.masked_fill(future | pad_keys, float("-inf"))

        k_keep = min(self.top_k, length)
        vals, index = torch.topk(scores, k=k_keep, dim=-1)
        if k_keep < self.top_k:
            extra = self.top_k - k_keep
            vals = torch.nn.functional.pad(vals, (0, extra), value=float("-inf"))
            index = torch.nn.functional.pad(index, (0, extra), value=-1)

        finite = torch.isfinite(vals)
        weights = torch.softmax(vals.masked_fill(~finite, float("-inf")), dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = self.dropout(weights)
        weights = weights.masked_fill(~finite, 0.0)
        index = index.masked_fill(~finite, -1)

        # Padded queries attend nowhere, even if some past keys were finite.
        query_pad = (pad_mask <= 0)[:, None, :, None]
        index = index.masked_fill(query_pad, -1)
        weights = weights.masked_fill(query_pad, 0.0)
        weights = weights.masked_fill(index < 0, 0.0)

        gathered = self._gather(value, index)
        mixed = (gathered * weights.unsqueeze(-1)).sum(dim=3)
        combined = mixed.permute(0, 2, 1, 3).contiguous().view(batch, length, -1)
        return self.out_proj(combined), index, weights

    def _gather(self, value: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        batch, heads, length, d_head = value.shape
        k_slots = index.shape[-1]
        safe = index.clamp(min=0)
        flat_value = value.reshape(batch * heads, length, d_head)
        flat_index = safe.reshape(batch * heads, length * k_slots)
        gathered = torch.gather(
            flat_value,
            1,
            flat_index.unsqueeze(-1).expand(-1, -1, d_head),
        )
        gathered = gathered.view(batch, heads, length, k_slots, d_head)
        return gathered * (index >= 0).unsqueeze(-1).to(gathered.dtype)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = CausalTopKAttention(D_MODEL, N_HEADS, TOP_K, DROPOUT)
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, 4 * D_MODEL),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(4 * D_MODEL, D_MODEL),
            nn.Dropout(DROPOUT),
        )

    def forward(
        self, hidden: torch.Tensor, pad_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_out, attn_index, attn_weight = self.attn(hidden, pad_mask)
        hidden = self.norm1(hidden + attn_out)
        hidden = self.norm2(hidden + self.ffn(hidden))
        return hidden, attn_index, attn_weight


class PaperTransformer(nn.Module):
    """Sum text, year, and entity features, then causal top-k layers."""

    def __init__(self) -> None:
        super().__init__()
        n_years = YEAR_MAX - YEAR_MIN + 1
        self.n_years = n_years
        self.text_proj = nn.Linear(TEXT_DIM, D_MODEL)
        self.year_emb = nn.Embedding(n_years + 1, D_MODEL)
        self.gene_proj = nn.Linear(len(GENES), D_MODEL)
        self.drug_proj = nn.Linear(len(DRUGS), D_MODEL)
        self.disease_proj = nn.Linear(len(DISEASES), D_MODEL)
        self.input_norm = nn.LayerNorm(D_MODEL)
        self.layers = nn.ModuleList(_Block() for _ in range(N_LAYERS))
        self.gene_head = nn.Linear(D_MODEL, len(GENES))
        self.drug_head = nn.Linear(D_MODEL, len(DRUGS))
        self.clinical_head = nn.Linear(D_MODEL, 1)

    def _year_index(self, year: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        clamped = (year - YEAR_MIN).clamp(0, self.n_years - 1)
        pad_row = torch.full_like(clamped, self.n_years)
        return torch.where(pad_mask > 0.5, clamped, pad_row)

    def forward(self, batch: dict) -> dict:
        pad_mask = batch["pad_mask"]
        hidden = (
            self.text_proj(batch["text"])
            + self.year_emb(self._year_index(batch["year"], pad_mask))
            + self.gene_proj(batch["genes"])
            + self.drug_proj(batch["drugs"])
            + self.disease_proj(batch["diseases"])
        )
        hidden = self.input_norm(hidden)

        attn_index = None
        attn_weight = None
        for layer in self.layers:
            hidden, attn_index, attn_weight = layer(hidden, pad_mask)

        return {
            "hidden": hidden,
            "attn_index": attn_index,
            "attn_weight": attn_weight,
            "logits_genes": self.gene_head(hidden),
            "logits_drugs": self.drug_head(hidden),
            "logits_clinical": self.clinical_head(hidden).squeeze(-1),
        }
