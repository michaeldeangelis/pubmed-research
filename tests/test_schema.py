from src.schema import (
    HELD_OUT_TAGS,
    evidence_tags,
    prior_genetic_evidence,
    strip_markup,
)


def test_markup_stripped_before_tagging():
    text = strip_markup("Germline <i>KRAS</i> variant in familial Noonan")
    assert "<i>" not in text
    assert "human_genetics" in evidence_tags(text)


def test_clinical_and_genetics_are_distinct():
    clinical = evidence_tags(
        "Phase II trial of trametinib. Patients with melanoma. Progression-free survival improved."
    )
    genetics = evidence_tags(
        "A germline KRAS polymorphism in familial Noonan syndrome, odds ratio 3.2."
    )
    assert "clinical" in clinical and "human_genetics" not in clinical
    assert "human_genetics" in genetics and "clinical" not in genetics
    assert "human_genetics" in HELD_OUT_TAGS


def test_prior_genetic_evidence_uses_only_earlier_papers():
    assert prior_genetic_evidence([["mechanistic"], ["human_genetics"]]) is True
    assert prior_genetic_evidence([["clinical"], ["animal_model"]]) is False
    assert prior_genetic_evidence([]) is False
