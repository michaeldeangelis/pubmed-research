"""Pre-LN paper transformer with swappable token mixers.

Mixers share one output format: index/weight [B, H, L, TOP_K], unused slots
index -1 and weight 0. `mean` and `self` keep v_proj/out_proj, so they differ
from `attention` only in how the past positions are chosen and weighted.
"""

from __future__ import annotations

import torch
from torch import nn

from src.config_v2 import (
    D_MODEL,
    DROPOUT,
    N_HEADS,
    N_LAYERS,
    TEXT_DIM,
    TOP_K,
    TRAIN_YEAR_MAX,
    YEAR_MAX,
    YEAR_MIN,
)
from src.model.transformer import CausalTopKAttention
from src.schema import DISEASES, DRUGS, GENES


class _FixedMixer(nn.Module):
    """Mixes projected values with index/weight that do not depend on content."""

    _split = CausalTopKAttention._split
    _gather = CausalTopKAttention._gather

    def __init__(self, d_model: int, n_heads: int, top_k: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} is not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.top_k = top_k
        self.d_head = d_model // n_heads
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _select(self, pad_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return index and weight [B, L, TOP_K] shared by every head."""
        raise NotImplementedError

    def forward(
        self, hidden: torch.Tensor, pad_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, _ = hidden.shape
        value = self._split(self.v_proj(hidden))
        index, weights = self._select(pad_mask)
        index = index[:, None].expand(-1, self.n_heads, -1, -1).contiguous()
        weights = self.dropout(weights[:, None].expand(-1, self.n_heads, -1, -1).contiguous())
        gathered = self._gather(value, index)
        mixed = (gathered * weights.unsqueeze(-1)).sum(dim=3)
        combined = mixed.permute(0, 2, 1, 3).contiguous().view(batch, length, -1)
        return self.out_proj(combined), index, weights


class MeanMixer(_FixedMixer):
    """Uniform weight over the last TOP_K real positions up to and including t."""

    def _select(self, pad_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length = pad_mask.shape
        positions = torch.arange(length, device=pad_mask.device)
        future = positions.view(1, length) > positions.view(length, 1)
        blocked = future[None] | (pad_mask <= 0)[:, None, :]
        scores = positions.to(pad_mask.dtype).expand(batch, length, length)
        scores = scores.masked_fill(blocked, float("-inf"))

        k_keep = min(self.top_k, length)
        vals, index = torch.topk(scores, k=k_keep, dim=-1)
        if k_keep < self.top_k:
            extra = self.top_k - k_keep
            vals = torch.nn.functional.pad(vals, (0, extra), value=float("-inf"))
            index = torch.nn.functional.pad(index, (0, extra), value=-1)
        valid = torch.isfinite(vals) & (pad_mask > 0)[:, :, None]
        index = index.masked_fill(~valid, -1)
        count = valid.sum(dim=-1, keepdim=True).clamp(min=1).to(pad_mask.dtype)
        weights = valid.to(pad_mask.dtype) / count
        return index, weights


class SelfMixer(_FixedMixer):
    """Each real position sees only itself."""

    def _select(self, pad_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length = pad_mask.shape
        real = pad_mask > 0
        positions = torch.arange(length, device=pad_mask.device).expand(batch, length)
        index = torch.full(
            (batch, length, self.top_k), -1, dtype=torch.long, device=pad_mask.device
        )
        weights = torch.zeros(batch, length, self.top_k, dtype=pad_mask.dtype, device=pad_mask.device)
        index[..., 0] = torch.where(real, positions, torch.full_like(positions, -1))
        weights[..., 0] = real.to(pad_mask.dtype)
        return index, weights


_MIXERS = {"attention": CausalTopKAttention, "mean": MeanMixer, "self": SelfMixer}


def _make_mixer(mixer: str) -> nn.Module:
    if mixer not in _MIXERS:
        raise ValueError(f"unknown mixer {mixer!r}; expected one of {sorted(_MIXERS)}")
    return _MIXERS[mixer](D_MODEL, N_HEADS, TOP_K, DROPOUT)


class _BlockV2(nn.Module):
    """Pre-LN: x + mixer(LN(x)); x + ffn(LN(x))."""

    def __init__(self, mixer: str) -> None:
        super().__init__()
        self.mixer = _make_mixer(mixer)
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
        mixed, index, weight = self.mixer(self.norm1(hidden), pad_mask)
        hidden = hidden + mixed
        hidden = hidden + self.ffn(self.norm2(hidden))
        return hidden, index, weight


class PaperTransformerV2(nn.Module):
    """v1 input features, pre-LN blocks with a chosen mixer, five heads."""

    def __init__(self, mixer: str = "attention") -> None:
        super().__init__()
        if mixer not in _MIXERS:
            raise ValueError(f"unknown mixer {mixer!r}; expected one of {sorted(_MIXERS)}")
        self.mixer = mixer
        self.d_model = D_MODEL
        n_years = YEAR_MAX - YEAR_MIN + 1
        self.n_years = n_years
        self.text_proj = nn.Linear(TEXT_DIM, D_MODEL)
        self.year_emb = nn.Embedding(n_years + 1, D_MODEL)
        self.gene_proj = nn.Linear(len(GENES), D_MODEL)
        self.drug_proj = nn.Linear(len(DRUGS), D_MODEL)
        self.disease_proj = nn.Linear(len(DISEASES), D_MODEL)
        self.input_norm = nn.LayerNorm(D_MODEL)
        self.layers = nn.ModuleList(_BlockV2(mixer) for _ in range(N_LAYERS))
        self.final_norm = nn.LayerNorm(D_MODEL)
        self.gene_head = nn.Linear(D_MODEL, len(GENES))
        self.drug_head = nn.Linear(D_MODEL, len(DRUGS))
        self.novel_gene_head = nn.Linear(D_MODEL, len(GENES))
        self.novel_drug_head = nn.Linear(D_MODEL, len(DRUGS))
        self.clinical_head = nn.Linear(D_MODEL, 1)

    def _year_index(self, year: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        # Rows after TRAIN_YEAR_MAX never get gradient, so later years reuse its row.
        clamped = (year - YEAR_MIN).clamp(0, TRAIN_YEAR_MAX - YEAR_MIN)
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

        index_layers: list[torch.Tensor] = []
        weight_layers: list[torch.Tensor] = []
        for layer in self.layers:
            hidden, attn_index, attn_weight = layer(hidden, pad_mask)
            index_layers.append(attn_index)
            weight_layers.append(attn_weight)
        hidden = self.final_norm(hidden)

        return {
            "hidden": hidden,
            "attn_index": index_layers[-1],
            "attn_weight": weight_layers[-1],
            "attn_index_layers": index_layers,
            "attn_weight_layers": weight_layers,
            "logits_genes": self.gene_head(hidden),
            "logits_drugs": self.drug_head(hidden),
            "logits_novel_genes": self.novel_gene_head(hidden),
            "logits_novel_drugs": self.novel_drug_head(hidden),
            "logits_clinical": self.clinical_head(hidden).squeeze(-1),
        }
