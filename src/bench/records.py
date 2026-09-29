"""Spike: extract result records blind and score them against the evidence map.

Throwaway spike code. Spec: docs/specs/2026-09-29-result-records-spike.md.

python -m src.bench.records packets        write blind packets
python -m src.bench.records judge-pairs    write blinded judge pairs
python -m src.bench.records score          score extractions and judgments
"""

from __future__ import annotations

import glob
import json
import random
import re
import sys
from collections import Counter

from src.config import DATA, ROOT
from src.schema import strip_markup

MAP_PATH = ROOT / "results" / "evidence_map.json"
DIR = ROOT / "bench" / "result_records"
PACKETS_PATH = DIR / "packets.json"
EXTRACTORS = ("opus", "sonnet")
JUDGE_PAIRS_PATH = DIR / "judge_pairs.json"
JUDGE_KEY_PATH = DIR / "judge_key.json"
JUDGMENTS_PATH = DIR / "judgments.json"
RESULT_PATH = ROOT / "results" / "result_records_spike.json"
REFERENCE_LABELS = {"C024": "conditions", "C033": "estimand", "C015": "certainty"}
NUMERIC_FIELDS = (
    ("estimand", "denominator"),
    ("estimand", "estimate"),
    ("estimand", "uncertainty"),
    ("measurement", "uncertainty"),
)
JUDGED_FIELDS = (
    ("population", ("conditions", "population")),
    ("comparator", ("conditions", "comparator")),
    ("quantity", ("estimand", "quantity")),
)
BAR = 0.80

_NUMBER = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d*\.\d+|\d+")


def numbers(text: str) -> set[str]:
    """Numbers in text, normalized; four-digit years 1900-2100 dropped."""
    found: set[str] = set()
    for token in _NUMBER.findall(text or ""):
        value = float(token.replace(",", ""))
        if value.is_integer() and 1900 <= value <= 2100 and len(token) == 4:
            continue
        found.add(f"{value:g}")
    return found


def _field(record: dict, path: tuple[str, str]) -> str:
    value = (record.get(path[0]) or {}).get(path[1], "")
    return value if isinstance(value, str) else json.dumps(value)


def record_numbers(record: dict) -> set[str]:
    found: set[str] = set()
    for path in NUMERIC_FIELDS:
        found |= numbers(_field(record, path))
    return found


def numeric_recall(reference: list[dict], extracted: dict[str, dict]) -> dict:
    """Pooled share of reference numbers that appear in the extracted record."""
    hit = total = 0
    missed: dict[str, list[str]] = {}
    for record in reference:
        ref = record_numbers(record)
        got = record_numbers(extracted.get(record["pmid"], {}))
        hit += len(ref & got)
        total += len(ref)
        if ref - got:
            missed[record["pmid"]] = sorted(ref - got)
    return {"recall": hit / total if total else 1.0, "hit": hit, "total": total, "missed": missed}


def _record_class(structured: dict) -> str | None:
    direction = structured.get("direction")
    if direction == "supports":
        return None
    if structured.get("condition_departure") is True:
        return "conditions"
    if structured.get("quantity_matches_claim") is False:
        return "estimand"
    if direction == "null":
        return "certainty"
    return "conflict"


def disagreement_type(records: list[dict]) -> str:
    """Claim-level label from the non-supporting records; see the spike spec."""
    classes = [
        label
        for label in (_record_class(record.get("structured") or {}) for record in records)
        if label is not None
    ]
    if not classes:
        return "consistent"
    ranked = Counter(classes).most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return "tie"
    return ranked[0][0]


def _reference() -> list[dict]:
    evidence = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    return [
        {**experiment, "claim_id": claim["claim_id"], "proposition": claim["proposition"]}
        for claim in evidence["claims"]
        for experiment in claim["experiments"]
    ]


def _abstracts(pmids: set[str]) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for path in sorted(glob.glob(str(DATA / "graph_cache" / "*.json"))):
        try:
            payload = json.loads(open(path, encoding="utf-8").read())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        for item in (payload.get("resultList") or {}).get("result") or []:
            pmid = str(item.get("pmid", ""))
            if pmid in pmids and item.get("abstractText") and pmid not in found:
                found[pmid] = {
                    "title": strip_markup(str(item.get("title") or "")),
                    "abstract": strip_markup(str(item["abstractText"])),
                }
    missing = pmids - set(found)
    if missing:
        raise SystemExit(f"abstracts missing for {sorted(missing)}")
    return found


def write_packets() -> None:
    reference = _reference()
    abstracts = _abstracts({record["pmid"] for record in reference})
    packets = [
        {
            "claim_id": record["claim_id"],
            "proposition": record["proposition"],
            "pmid": record["pmid"],
            **abstracts[record["pmid"]],
        }
        for record in reference
    ]
    DIR.mkdir(parents=True, exist_ok=True)
    PACKETS_PATH.write_text(json.dumps(packets, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(packets)} packets to {PACKETS_PATH}")


def _extraction(name: str) -> dict[str, dict]:
    rows = json.loads((DIR / f"extracted_{name}.json").read_text(encoding="utf-8"))
    return {str(row["pmid"]): row for row in rows}


def write_judge_pairs() -> None:
    """Blinded pairs: reference vs each extractor, sides and order shuffled."""
    reference = _reference()
    rng = random.Random(0)
    pairs, key = [], {}
    for name in EXTRACTORS:
        extracted = _extraction(name)
        for record in reference:
            other = extracted.get(record["pmid"], {})
            fields = {
                label: [_field(record, path), _field(other, path)] for label, path in JUDGED_FIELDS
            }
            flip = rng.random() < 0.5
            if flip:
                fields = {label: values[::-1] for label, values in fields.items()}
            pair_id = f"P{len(pairs) + 1:03d}"
            pairs.append({"pair_id": pair_id, "proposition": record["proposition"], "fields": fields})
            key[pair_id] = {"extractor": name, "pmid": record["pmid"]}
    rng.shuffle(pairs)
    JUDGE_PAIRS_PATH.write_text(json.dumps(pairs, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    JUDGE_KEY_PATH.write_text(json.dumps(key, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(pairs)} judge pairs")


def _semantic(name: str, judgments: dict, key: dict) -> dict:
    shares = {}
    for label, _path in JUDGED_FIELDS:
        rows = [judgments[pid][label] for pid, info in key.items() if info["extractor"] == name]
        shares[label] = {
            "same": sum(row == "same" for row in rows) / len(rows),
            "counts": dict(Counter(rows)),
        }
    return shares


def score() -> None:
    reference = _reference()
    judgments = {}
    if JUDGMENTS_PATH.is_file():
        judgments = {
            row["pair_id"]: row
            for row in json.loads(JUDGMENTS_PATH.read_text(encoding="utf-8"))
        }
    key = json.loads(JUDGE_KEY_PATH.read_text(encoding="utf-8")) if judgments else {}
    report: dict = {"spec": "docs/specs/2026-09-29-result-records-spike.md", "bar": BAR, "extractors": {}}
    structured_by_name = {}
    for name in EXTRACTORS:
        extracted = _extraction(name)
        structured_by_name[name] = {pmid: row.get("structured") or {} for pmid, row in extracted.items()}
        labels = {
            claim: disagreement_type(
                [row for row in extracted.values() if row.get("claim_id") == claim]
            )
            for claim in REFERENCE_LABELS
        }
        recall = numeric_recall(reference, extracted)
        entry = {
            "n_records": len(extracted),
            "numeric_recall": recall,
            "rule_labels": labels,
            "rule_pass": labels == REFERENCE_LABELS,
            "record_classes": {
                pmid: _record_class(row.get("structured") or {}) for pmid, row in sorted(extracted.items())
            },
        }
        if judgments:
            entry["semantic"] = _semantic(name, judgments, key)
            entry["semantic_pass"] = all(v["same"] >= BAR for v in entry["semantic"].values())
            entry["all_bars"] = recall["recall"] >= BAR and entry["semantic_pass"] and entry["rule_pass"]
        report["extractors"][name] = entry
    first, second = (structured_by_name[name] for name in EXTRACTORS)
    shared = sorted(set(first) & set(second))
    report["extractor_agreement"] = {
        field: sum(first[p].get(field) == second[p].get(field) for p in shared) / len(shared)
        for field in ("direction", "significance", "condition_departure", "quantity_matches_claim")
    }
    report["reference_labels"] = REFERENCE_LABELS
    RESULT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    commands = {"packets": write_packets, "judge-pairs": write_judge_pairs, "score": score}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        raise SystemExit(f"usage: python -m src.bench.records {{{','.join(commands)}}}")
    commands[sys.argv[1]]()
