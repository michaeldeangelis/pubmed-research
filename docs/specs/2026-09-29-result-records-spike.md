# Spike: can result records be extracted, and does a rule reproduce the judgments?

Plan B after transformer v2 closed. This is a feasibility spike, and its code
is throwaway. Everything below was fixed before any extraction ran.

## Question

1. Can a blind extractor rebuild the 27 result records in
   `results/evidence_map.json` from the claim and the abstract alone?
2. Given the extracted records, does a fixed rule name the disagreement type
   the map reached for each claim: C024 conditions, C033 estimand, C015
   certainty?

## Setup

- Packets (`bench/result_records/packets.json`) hold the claim ID, the
  proposition, the PMID, the title, and the abstract. They contain nothing
  from the reference records.
- Two extractors run independently: a fresh Opus subagent and a fresh Sonnet
  subagent. Each sees only the packets and `bench/result_records/schema.json`.
- A blind judge sees reference and extracted field pairs with the source
  hidden. For each record it rates population, comparator, and quantity
  estimated as `same`, `partial`, or `different`.

## Scores and bars

- **Numeric recall.** Numbers in the reference `estimand.denominator`,
  `estimand.estimate`, `estimand.uncertainty`, and
  `measurement.uncertainty`, minus four-digit years from 1900 to 2100, are
  pooled over the 27 records. Recall is the share that also appear in the
  same extracted fields. Bar: 0.80 or higher.
- **Semantic match.** The share of `same` judgments for each of population,
  comparator, and quantity. Bar: 0.80 or higher on each.
- **Rule.** `disagreement_type` in `src/bench/records.py`, run on the
  extracted structured fields, must return the reference label for all
  three claims. With n = 3 this is a sanity check, not evidence.

Each extractor is scored separately. Extraction is feasible if at least one
extractor clears all bars. Agreement between the two extractors is reported.

## Rule

Each record carries `direction` (supports, opposes, null, unclear),
`condition_departure` (bool), and `quantity_matches_claim` (bool). A record
that does not support the claim (opposes, null, or unclear) is classed in
this order: `conditions` if `condition_departure`, else `estimand` if not
`quantity_matches_claim`, else `certainty` if `direction` is null, else
`conflict`. The claim's label is the most common class among those records.
A tie returns `tie`, which counts as a miss. A claim with no such record is
`consistent`.

## Caveats

The reference records, the extractors, and the judge are all Claude models,
so agreement is partly circular. The three claims were chosen because they
disagree. They are not a sample.
