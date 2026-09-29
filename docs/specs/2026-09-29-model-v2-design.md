# Paper transformer v2

v1 is frozen. v2 lives in new files and writes new results. Nothing in
`src/config.py`, `src/schema.py`, `src/model/{dataset,transformer,train}.py`,
`src/probe/probes.py`, or `results/probe.json` changes.

## Question

Does a causal paper transformer, trained on a non-trivial objective with a
clean time split, beat a no-attention baseline and carry a readable trace of
earlier human genetic evidence?

## Decision rule (fixed before any v2 run)

Measured on test positions (next paper year >= 2021), three seeds (0, 1, 2).

1. **Attention beats mean-pooling.** Mean test loss of `attention` is lower
   than mean test loss of `mean` by more than the larger of the two
   across-seed standard deviations (population std, ddof=0).
2. **Genetics probe lift.** Seed-averaged lift (contextual AUROC minus frozen
   text AUROC, `attention` mixer) is at least 0.03, and the 2.5th percentile
   of a 1000-resample paired bootstrap of that seed-averaged lift is above 0.

v2 is fruitful only if both hold. Otherwise the record says so and work moves
to result-level records (plan B).

## Changes from v1

| v1 problem | v2 |
|---|---|
| Timeline anchor gene is in every next-paper gene target | Anchor column masked out of the gene loss |
| Objective mostly repeats the window | Added targets: next paper's genes and drugs not seen earlier in the window |
| Loss mask used the current paper's year | Split uses the next paper's year |
| No validation, last epoch kept | 2019-2020 validation, best-val epoch kept, early stopping |
| No baseline | `mean` and `self` mixers with identical data and seeds |
| Post-LN, only last layer's attention returned | Pre-LN, every layer's index and weight returned |

## Interfaces

### `src/config_v2.py`

Re-exports v1 settings it does not override. Adds:

```
SEEDS = (0, 1, 2)
MIXERS = ("attention", "mean", "self")
EPOCHS_MAX = 30
PATIENCE = 4
LOSS_WEIGHT_NOVEL_GENE = 0.5
LOSS_WEIGHT_NOVEL_DRUG = 0.5
BOOTSTRAP_RESAMPLES = 1000
LIFT_BAR = 0.03
CHECKPOINTS_V2 = CHECKPOINTS / "v2"
OUTPUTS_V2 = OUTPUTS / "v2"
RESULTS_V2_PATH = ROOT / "results" / "probe_v2.json"
SPLIT_NONE, SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = -1, 0, 1, 2
```

### `src/model/dataset_v2.py`

- `build_windows_v2(enriched_path, embeddings_path) -> list[list[dict]]`:
  v1 items plus `"anchor": <gene>` (the timeline gene).
- `collate_windows_v2(windows) -> dict[str, Tensor]`: every v1 key, plus
  - `gene_target_mask` [B, L, G]: 1, except 0 at the anchor column. All ones
    when a window has no `anchor`.
  - `y_novel_genes` [B, L, G], `y_novel_drugs` [B, L, D]: entities of paper
    t+1 absent from papers 0..t of the window.
  - `split` [B, L] long: by the year of paper t+1. `SPLIT_NONE` for padding
    and the last real token.
  - `loss_mask` equals `split == SPLIT_TRAIN` as float.
- No key from `FORBIDDEN_BATCH_KEYS` appears in the batch.

### `src/model/transformer_v2.py`

`PaperTransformerV2(mixer: str = "attention")`. Pre-LN blocks. Mixers:

- `attention`: v1 causal top-k attention.
- `mean`: uniform weight over the last `TOP_K` real positions up to and
  including t. Has no q/k projections.
- `self`: each position sees only itself.

All mixers return index/weight tensors in the v1 format
([B, H, L, TOP_K], unused slots index -1 and weight 0).

`forward(batch)` returns `hidden`, `attn_index`, `attn_weight` (last layer),
`attn_index_layers`, `attn_weight_layers` (lists, one per layer),
`logits_genes`, `logits_drugs`, `logits_novel_genes`, `logits_novel_drugs`,
`logits_clinical`.

### `src/model/train_v2.py`

- `prediction_loss_v2(outputs, batch, split: int) -> dict[str, Tensor]`:
  keys `clinical`, `genes`, `drugs`, `novel_genes`, `novel_drugs`, `total`.
  Gene loss is weighted by `gene_target_mask`. Positions are those with
  `batch["split"] == split`.
- `train_one(mixer, seed, windows) -> dict`: early stopping on val `total`,
  saves `CHECKPOINTS_V2/{mixer}_s{seed}.pt` with `{"state_dict", "mixer",
  "seed", "best_epoch"}`, returns per-component val and test losses at the
  best epoch and the epoch curve.
- `main()`: every mixer x seed, writes `OUTPUTS_V2/training.json`.

### `src/probe/probes_v2.py`

- Reuses `auroc`, `fit_direction`, `project_out` from v1.
- `probe_checkpoint(path, windows) -> dict`: frozen and contextual AUROC,
  per-position test scores and labels (for the bootstrap), resistance AUROC,
  ablation deltas, per-layer per-head genetics attention mass.
- `bootstrap_lift(per_seed, n, rng_seed) -> dict`: paired resampling of test
  positions, shared across seeds, of the seed-averaged lift. Returns mean,
  2.5th and 97.5th percentiles.
- `decide(training, probes) -> dict`: applies the decision rule and returns
  each criterion's numbers and pass/fail.
- `python -m src.probe.v2` writes `RESULTS_V2_PATH`.

## Amendments after run 1

Run 1 (`results/probe_v2_run1_year_bug.json`) is kept but not used. A code
review and a probe-noise check found three measurement faults. All three
were fixed after run 1 results had been seen. The decision rule did not change.

1. Year embedding rows after 2018 never received gradient, so every val and
   test paper carried an untrained random vector. Years after
   `TRAIN_YEAR_MAX` now share its row. On run-1 checkpoints this alone moved
   attention test loss from 1.036 to 0.951 and its seed std from 0.040 to 0.006.
2. The split used the `year` field, while windows are ordered by `date`, and
   203 papers disagree. 42 training positions saw a post-2018 input. The split
   now uses the year of the ordering date.
3. The v1 probe starts from a random init and stops after 400 SGD steps. On
   identical frozen text its test AUROC ranged 0.485-0.567 over ten inits.
   `fit_direction_v2` standardizes features and fits from zero with LBFGS,
   so it is deterministic.

`src/probe/context_check_v2.py` is a post hoc comparison added after the
run-2 decision. It is not part of the rule.

## 2026 confirmation (fixed before any 2026 paper was fetched)

The post hoc result was that attention's hidden state reads prior genetic
evidence better than the mean mixer's (+0.048 AUROC on 2021-2025). It is
tested once on papers no v2 run has seen.

- Data: Europe PMC, publication year 2026, same query, same cap of
  `PER_YEAR` = 80 cited abstracts, PMIDs already in the corpus skipped. Same
  enrichment and the same neural encoder. If the encoder falls back to hash
  vectors, the run stops. Files go to `outputs/v2/confirm_2026/`, and
  `data/` is not touched.
- Windows are rebuilt over the corpus plus the 2026 papers, so 2026 papers
  have earlier papers as context.
- Models: the six run-2 checkpoints (`attention` and `mean`, seeds 0-2)
  unchanged. No retraining. Probe directions are fit on positions from 2018
  or earlier, as before.
- Test positions: papers whose own year is 2026.
- **Rule:** confirmed if the seed-averaged attention probe AUROC minus the
  seed-averaged mean-mixer probe AUROC is above 0, and the 2.5th percentile
  of 1000 paired bootstrap resamples (rng seed 0) is above 0.
- If the 2026 test positions have fewer than 30 positives or fewer than 30
  negatives for prior genetic evidence, the result is recorded as
  inconclusive, whatever the estimate.
- Reported but not part of the rule: attention minus frozen text, and
  attention minus the untrained all-earlier context probe.

## Tests

Anchor masking, novel-entity targets, next-year split, forbidden keys,
mean-mixer equals a hand-computed average, `self` mixer ignores other
positions, causality for all mixers, loss on a tiny batch is finite,
bootstrap on synthetic scores, `decide` on synthetic pass and fail inputs.
