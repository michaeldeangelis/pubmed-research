import torch

from src.probe.probes import auroc, selective_ablation_deltas


def test_auroc_perfect_and_chance():
    scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
    labels = torch.tensor([0, 0, 1, 1])
    assert auroc(scores, labels) == 1.0
    tied = torch.tensor([0.5, 0.5, 0.5, 0.5])
    assert abs(auroc(tied, labels) - 0.5) < 1e-6


def test_selective_ablation_reports_both_contexts():
    delta_pos, delta_neg = selective_ablation_deltas(
        clinical_bce_before=torch.tensor([0.2, 0.2, 0.4, 0.4]),
        clinical_bce_after=torch.tensor([0.5, 0.2, 0.4, 0.9]),
        prior_genetics=torch.tensor([1, 0, 0, 1]),
    )
    assert delta_pos > delta_neg
