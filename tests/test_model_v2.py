import json

import pytest
import torch

from src.config_v2 import (
    LOSS_WEIGHT_CLINICAL,
    LOSS_WEIGHT_DRUG,
    LOSS_WEIGHT_GENE,
    LOSS_WEIGHT_NOVEL_DRUG,
    LOSS_WEIGHT_NOVEL_GENE,
    MIXERS,
    N_HEADS,
    SPLIT_NONE,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    TEXT_DIM,
    TOP_K,
)
from src.model import train_v2
from src.model.dataset_v2 import collate_windows_v2
from src.model.train_v2 import prediction_loss_v2, train_one
from src.model.transformer_v2 import PaperTransformerV2
from src.schema import DRUGS, FORBIDDEN_BATCH_KEYS, GENES


def _paper(year, genes, drugs=(), seed=0, clinical=0.0):
    generator = torch.Generator().manual_seed(seed)
    return {
        "pmid": str(seed),
        "year": year,
        "text": torch.randn(TEXT_DIM, generator=generator),
        "genes": list(genes),
        "drugs": list(drugs),
        "diseases": ["NSCLC"],
        "clinical": clinical,
        "evidence_tags": ["human_genetics"] if seed % 3 == 0 else [],
        "anchor": "KRAS",
    }


def _window(n=6, start_year=2010, offset=0):
    return [
        _paper(
            start_year + i,
            ["KRAS"] + (["BRAF"] if i % 2 else []),
            ["trametinib"] if i % 3 == 2 else [],
            seed=offset + i,
            clinical=float(i % 2),
        )
        for i in range(n)
    ]


def test_anchor_column_masked_only():
    batch = collate_windows_v2([_window(4)])
    mask = batch["gene_target_mask"]
    kras = GENES.index("KRAS")
    assert mask.shape == (1, 4, len(GENES))
    assert torch.all(mask[..., kras] == 0)
    others = [i for i in range(len(GENES)) if i != kras]
    assert torch.all(mask[..., others] == 1)


def test_no_anchor_means_all_ones():
    window = _window(4)
    for paper in window:
        paper.pop("anchor")
    batch = collate_windows_v2([window])
    assert torch.all(batch["gene_target_mask"] == 1)


def test_novel_targets():
    window = [
        _paper(2010, ["KRAS"], [], seed=1),
        _paper(2011, ["KRAS", "BRAF"], ["trametinib"], seed=2),
        _paper(2012, ["KRAS", "BRAF", "EGFR"], ["trametinib", "sotorasib"], seed=3),
    ]
    batch = collate_windows_v2([window])
    kras, braf, egfr = (GENES.index(g) for g in ("KRAS", "BRAF", "EGFR"))
    tram, soto = DRUGS.index("trametinib"), DRUGS.index("sotorasib")
    ng, nd = batch["y_novel_genes"][0], batch["y_novel_drugs"][0]
    # t=0 predicts paper 1: BRAF and trametinib are new.
    assert ng[0, braf] == 1 and ng[0, kras] == 0 and ng[0].sum() == 1
    assert nd[0, tram] == 1 and nd[0].sum() == 1
    # t=1 predicts paper 2: only EGFR and sotorasib are new.
    assert ng[1, egfr] == 1 and ng[1].sum() == 1
    assert nd[1, soto] == 1 and nd[1].sum() == 1
    # Last token has no target.
    assert ng[2].sum() == 0 and nd[2].sum() == 0
    # Plain next-paper targets still repeat the window.
    assert batch["y_genes"][0, 1, kras] == 1


def test_split_uses_next_paper_year():
    years_a = [2017, 2018, 2019, 2020]
    years_b = [2019, 2020, 2021, 2022, 2023]
    window_a = [_paper(y, ["KRAS"], seed=i) for i, y in enumerate(years_a)]
    window_b = [_paper(y, ["KRAS"], seed=10 + i) for i, y in enumerate(years_b)]
    batch = collate_windows_v2([window_a, window_b])
    split = batch["split"]
    assert split.dtype == torch.long
    # Paper 2018 predicts a 2019 paper: val, not train.
    assert split[0].tolist() == [SPLIT_TRAIN, SPLIT_VAL, SPLIT_VAL, SPLIT_NONE, SPLIT_NONE]
    # Paper 2020 predicts a 2021 paper: test.
    assert split[1].tolist() == [SPLIT_VAL, SPLIT_TEST, SPLIT_TEST, SPLIT_TEST, SPLIT_NONE]
    assert torch.equal(batch["loss_mask"], (split == SPLIT_TRAIN).float())


def test_no_forbidden_keys_and_v1_keys_present():
    batch = collate_windows_v2([_window(5)])
    for key in FORBIDDEN_BATCH_KEYS:
        assert key not in batch
    for key in ("text", "year", "genes", "drugs", "diseases", "y_genes", "y_drugs",
                "y_clinical", "loss_mask", "pad_mask"):
        assert key in batch


def test_unknown_mixer_raises():
    with pytest.raises(ValueError):
        PaperTransformerV2("nope")


def test_mean_mixer_matches_hand_average():
    torch.manual_seed(0)
    model = PaperTransformerV2("mean").eval()
    mixer = model.layers[0].mixer
    length = TOP_K + 4
    hidden = torch.randn(2, length, model.d_model)
    pad_mask = torch.ones(2, length)
    pad_mask[1, length - 3 :] = 0.0
    with torch.no_grad():
        out, index, weight = mixer(hidden, pad_mask)
        value = mixer.v_proj(hidden)
    assert index.shape == (2, N_HEADS, length, TOP_K)
    for b in range(2):
        for t in range(length):
            if pad_mask[b, t] == 0:
                assert torch.all(index[b, :, t] == -1)
                assert torch.all(weight[b, :, t] == 0)
                continue
            n = min(TOP_K, t + 1)
            expected = set(range(t + 1 - n, t + 1))
            for h in range(N_HEADS):
                got = {int(i) for i in index[b, h, t] if i >= 0}
                assert got == expected
                used = weight[b, h, t][index[b, h, t] >= 0]
                assert torch.allclose(used, torch.full_like(used, 1.0 / n))
                assert torch.all(weight[b, h, t][index[b, h, t] < 0] == 0)
            manual = mixer.out_proj(value[b, t + 1 - n : t + 1].mean(dim=0))
            assert torch.allclose(out[b, t], manual, atol=1e-5)


def test_self_mixer_index_and_weight():
    torch.manual_seed(0)
    model = PaperTransformerV2("self").eval()
    batch = collate_windows_v2([_window(6), _window(4, offset=20)])
    with torch.no_grad():
        out = model(batch)
    index, weight = out["attn_index"], out["attn_weight"]
    positions = torch.arange(6)
    for b, real in enumerate((6, 4)):
        for t in range(6):
            if t < real:
                assert torch.all(index[b, :, t, 0] == positions[t])
                assert torch.all(weight[b, :, t, 0] == 1)
            else:
                assert torch.all(index[b, :, t, 0] == -1)
                assert torch.all(weight[b, :, t, 0] == 0)
            assert torch.all(index[b, :, t, 1:] == -1)
            assert torch.all(weight[b, :, t, 1:] == 0)


@pytest.mark.parametrize("mixer", MIXERS)
def test_outputs_and_causality(mixer):
    torch.manual_seed(0)
    model = PaperTransformerV2(mixer).eval()
    batch = collate_windows_v2([_window(10), _window(7, offset=30)])
    with torch.no_grad():
        out = model(batch)
    assert len(out["attn_index_layers"]) == len(model.layers)
    assert len(out["attn_weight_layers"]) == len(model.layers)
    for key, width in (("logits_genes", len(GENES)), ("logits_novel_genes", len(GENES)),
                       ("logits_drugs", len(DRUGS)), ("logits_novel_drugs", len(DRUGS))):
        assert out[key].shape == (2, 10, width)
    assert out["logits_clinical"].shape == (2, 10)
    positions = torch.arange(10)[None, None, :, None]
    for index, weight in zip(out["attn_index_layers"], out["attn_weight_layers"]):
        assert index.shape == (2, N_HEADS, 10, TOP_K)
        valid = index >= 0
        assert torch.all(index[valid] <= positions.expand_as(index)[valid])
        assert torch.all(weight[~valid] == 0)
        # Padded queries of the shorter window attend nowhere.
        assert torch.all(index[1, :, 7:] == -1)
        assert torch.all(weight[1, :, 7:] == 0)


@pytest.mark.parametrize("mixer", MIXERS)
def test_future_text_does_not_leak(mixer):
    torch.manual_seed(0)
    model = PaperTransformerV2(mixer).eval()
    window = _window(8)
    batch = collate_windows_v2([window])
    changed = [dict(paper) for paper in window]
    changed[5]["text"] = torch.randn(TEXT_DIM) * 5
    batch_changed = collate_windows_v2([changed])
    with torch.no_grad():
        before = model(batch)["hidden"]
        after = model(batch_changed)["hidden"]
    assert torch.allclose(before[:, :5], after[:, :5], atol=1e-5)
    assert not torch.allclose(before[:, 5], after[:, 5])


def test_prediction_loss_v2_total():
    torch.manual_seed(0)
    model = PaperTransformerV2("attention")
    batch = collate_windows_v2([_window(8, start_year=2014), _window(6, start_year=2016)])
    losses = prediction_loss_v2(model(batch), batch, SPLIT_TRAIN)
    assert set(losses) == {"clinical", "genes", "drugs", "novel_genes", "novel_drugs", "total"}
    for value in losses.values():
        assert torch.isfinite(value)
    expected = (
        LOSS_WEIGHT_CLINICAL * losses["clinical"]
        + LOSS_WEIGHT_GENE * losses["genes"]
        + LOSS_WEIGHT_DRUG * losses["drugs"]
        + LOSS_WEIGHT_NOVEL_GENE * losses["novel_genes"]
        + LOSS_WEIGHT_NOVEL_DRUG * losses["novel_drugs"]
    )
    assert torch.allclose(losses["total"], expected)
    losses["total"].backward()


def test_gene_loss_ignores_anchor_column():
    torch.manual_seed(0)
    model = PaperTransformerV2("attention")
    batch = collate_windows_v2([_window(8, start_year=2010)])
    out = model(batch)
    base = prediction_loss_v2(out, batch, SPLIT_TRAIN)["genes"]
    kras = GENES.index("KRAS")
    out["logits_genes"] = out["logits_genes"].detach().clone()
    out["logits_genes"][..., kras] = -50.0
    shifted = prediction_loss_v2(out, batch, SPLIT_TRAIN)["genes"]
    assert torch.allclose(base, shifted)


def test_train_one_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("V2_DEVICE", "cpu")
    monkeypatch.setattr(train_v2, "EPOCHS_MAX", 2)
    monkeypatch.setattr(train_v2, "CHECKPOINTS_V2", tmp_path / "ckpt")
    monkeypatch.setattr(train_v2, "OUTPUTS_V2", tmp_path / "out")
    windows = [
        _window(8, start_year=2013, offset=0),
        _window(8, start_year=2015, offset=10),
        _window(6, start_year=2017, offset=20),
    ]
    result = train_one("mean", 0, windows)
    assert 1 <= result["best_epoch"] <= 2
    assert len(result["curve"]) == 2
    for part in ("val", "test"):
        assert set(result[part]) == {"clinical", "genes", "drugs", "novel_genes",
                                     "novel_drugs", "total"}
        assert all(isinstance(v, float) for v in result[part].values())
    saved = torch.load(tmp_path / "ckpt" / "mean_s0.pt", weights_only=True)
    assert saved["mixer"] == "mean" and saved["seed"] == 0
    assert saved["best_epoch"] == result["best_epoch"]
    model = PaperTransformerV2("mean")
    model.load_state_dict(saved["state_dict"])
    json.dumps(result)
