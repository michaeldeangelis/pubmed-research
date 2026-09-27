import torch

from src.config import D_MODEL, N_HEADS, TOP_K
from src.model.dataset import collate_windows
from src.model.transformer import PaperTransformer
from src.schema import FORBIDDEN_BATCH_KEYS, GENES


def _window(n=6):
    papers = []
    for i in range(n):
        genes = ["KRAS"] if i % 2 == 0 else ["BRAF"]
        papers.append(
            {
                "pmid": str(i),
                "year": 2010 + i,
                "text": torch.zeros(768),
                "genes": genes,
                "drugs": ["sotorasib"] if i == n - 1 else [],
                "diseases": ["NSCLC"],
                "y_clinical": 1.0 if i % 3 == 0 else 0.0,
            }
        )
    return papers


def test_batch_hides_held_out_labels():
    batch = collate_windows([_window()])
    for key in FORBIDDEN_BATCH_KEYS:
        assert key not in batch
    assert batch["genes"].shape[-1] == len(GENES)
    assert batch["loss_mask"].shape[0] == 1


def test_attention_is_causal_and_sparse():
    torch.manual_seed(0)
    model = PaperTransformer()
    batch = collate_windows([_window(8)])
    out = model(batch)
    index = out["attn_index"]
    weight = out["attn_weight"]
    assert index.shape[-1] == TOP_K
    assert weight.shape == index.shape
    assert out["hidden"].shape[-1] == D_MODEL
    assert out["hidden"].shape[1] == 8
    assert index.shape[1] == N_HEADS
    positions = torch.arange(index.shape[2])[None, None, :, None]
    valid = index >= 0
    assert torch.all(index[valid] <= positions.expand_as(index)[valid])
    assert torch.all(weight[~valid] == 0)
    assert int((index >= 0).sum(dim=-1).max()) <= TOP_K
