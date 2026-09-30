"""Fresh blind judge packets for pairs at similarity >= 0.6 (amendment v4-2).

python -m src.meta.match_judge2

Draws 50 pairs (rng seed 1) with similarity >= 0.6 that were not in the
first judge sample, with A/B order randomized. Titles and abstracts only.
"""

from __future__ import annotations

import json

import numpy as np

from src.config import DATA, ROOT

BENCH = ROOT / "bench" / "meta_validation"
PAIRS = DATA / "meta" / "matching" / "pairs.jsonl"
PAPER_FILES = [DATA / "meta" / "papers.jsonl", DATA / "meta" / "pathways" / "egfr" / "papers.jsonl",
               DATA / "meta" / "pathways" / "pi3k" / "papers.jsonl"]
MIN_SIM = 0.6
N = 50


def main() -> None:
    first = json.loads((BENCH / "match_judge_key.json").read_text())
    first_items = first.values() if isinstance(first, dict) else first
    judged = {(str(v["treated_pmid"]), str(v["control_pmid"])) for v in first_items}
    pairs = [json.loads(line) for line in PAIRS.open() if line.strip()]
    pool = sorted(
        (str(p["treated_pmid"]), str(p["control_pmid"]))
        for p in pairs
        if p["similarity"] >= MIN_SIM and (str(p["treated_pmid"]), str(p["control_pmid"])) not in judged
    )
    rng = np.random.default_rng(1)
    chosen = [pool[i] for i in sorted(rng.choice(len(pool), size=min(N, len(pool)), replace=False))]
    papers = {}
    for path in PAPER_FILES:
        for line in path.open():
            r = json.loads(line)
            papers.setdefault(str(r["pmid"]), r)
    packets, key = [], {}
    for i, (t, c) in enumerate(chosen):
        pid = f"Q{i + 1:03d}"
        a, b = (t, c) if rng.random() < 0.5 else (c, t)
        packets.append({"pair_id": pid,
                        "A": {"title": papers[a]["title"], "abstract": papers[a]["abstract"]},
                        "B": {"title": papers[b]["title"], "abstract": papers[b]["abstract"]}})
        key[pid] = {"treated_pmid": t, "control_pmid": c, "A": a, "B": b}
    (BENCH / "match_judge2_packets.json").write_text(json.dumps(packets, indent=1, ensure_ascii=False) + "\n")
    (BENCH / "match_judge2_key.json").write_text(json.dumps(key, indent=1) + "\n")
    print(f"pool {len(pool)} pairs >= {MIN_SIM} not yet judged; wrote {len(packets)} packets")


if __name__ == "__main__":
    main()
