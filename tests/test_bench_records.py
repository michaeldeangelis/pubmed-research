from src.bench.records import disagreement_type, numbers, numeric_recall


def _rec(direction, departure=False, matches=True, claim="C1"):
    return {
        "claim_id": claim,
        "structured": {
            "direction": direction,
            "condition_departure": departure,
            "quantity_matches_claim": matches,
        },
    }


def test_numbers_normalize_and_drop_years():
    assert numbers("OR 3.5 (1.2 to 10.3), 12/6,637, P=.167 in 2013, 55.3%") == {
        "3.5",
        "1.2",
        "10.3",
        "12",
        "6637",
        "0.167",
        "55.3",
    }


def test_numeric_recall_pools_over_records():
    reference = [{"pmid": "1", "estimand": {"estimate": "OR 2.8 (0.7 to 11.8)"}}]
    extracted = {"1": {"estimand": {"uncertainty": "0.7-11.8"}, "measurement": {}}}
    out = numeric_recall(reference, extracted)
    assert out["hit"] == 2 and out["total"] == 3
    assert out["missed"] == {"1": ["2.8"]}


def test_rule_precedence_and_labels():
    assert disagreement_type([_rec("supports")]) == "consistent"
    assert disagreement_type([_rec("opposes", departure=True), _rec("null")]) == "tie"
    assert disagreement_type([_rec("opposes", departure=True), _rec("opposes", departure=True)]) == "conditions"
    assert disagreement_type([_rec("unclear", matches=False)]) == "estimand"
    assert disagreement_type([_rec("null"), _rec("supports")]) == "certainty"
    assert disagreement_type([_rec("opposes")]) == "conflict"
