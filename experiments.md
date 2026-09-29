# Experiment ledger

Append-only. Never edit an earlier entry; corrections are new lines under it.

## 2026-09-29 metascience v1   (rules fixed before the run)

Question: Among preclinical RAS/MAPK research articles from 2000-2015, are
model system, human genetic evidence, or use of multiple model systems
associated with being cited by a clinical article within 8 years?
Decision it drives: Features that pass become the basis for the next phase:
case histories and a wider field. If none pass, record a negative result and
consider whether the MeSH-level features are too coarse.
Held fixed: Corpus query and dates (spec), features F1-F3 and covariates as
committed in `src/meta/features.py`, outcome B definition, analysis code
commit.
Varied: Nothing. This is an observational association study.

Model: logistic regression, clin_cited_8y ~ F1 + F2 + F3 + year FE +
gene_group + log(n_authors) + log(1 + n_refs), preclinical papers only.
Estimates are odds ratios with 95% CI from 1000 nonparametric bootstrap
resamples of papers (rng seed 0).

Split: development = publication years 2000-2011; confirmation = 2012-2015.
The same model is fit separately in each.

Baseline ladder (every row is run and reported, AUROC on confirmation with
the model fit on development):
- trivial: development prevalence
- simplest reasonable: year FE + gene_group + log authors + log refs
- candidate: simplest reasonable + F1 + F2 + F3

Negative controls (should show no association):
- placebo feature: PMID parity (even/odd) added to the candidate model
- F1-F3 permuted within publication year (1000 permutations): observed
  ORs are compared with the permutation distribution

Pass (per feature level): the OR 95% CI excludes 1 in development AND in
confirmation, with the same direction, AND the observed OR lies outside the
central 95% of its within-year permutation distribution in development. The
placebo feature's CI must include 1 in both periods, or the whole analysis
is flagged as unreliable.
Kill: No feature level passes. Record the negative result.
Confirmation: The 2012-2015 split is the confirmation. It is not used for
any choice.
Measurement gate (before outcomes are joined): a feature is dropped if the
rule-versus-model kappa on 300 sampled preclinical papers is below 0.60.
Secondary, reported, no pass/fail: the same model with n_clin_8y (negative
binomial), with log RCR (linear), and with the disruption score (linear);
the sensitivity model with 50 journal indicators; descriptive rates for
clinical papers; the RPCB reference check for outcome A.
Budget: public APIs only (Europe PMC, iCite, OpenAlex, OSF); no model API key.
Record: results/meta_analysis.json

Implementation notes, added 2026-09-29 before any feature was joined to an outcome:
- `src/meta/analysis.py` fits the logit by IRLS with a 1e-6 ridge. It was
  tested on synthetic data only.
- Missing reference counts are set to 0, with a `refs_missing` indicator.
- In the AUROC ladder, year fixed effects are fit on development and set to
  0 when scoring confirmation, because the year ranges do not overlap.
- Secondary models are fit on the full 2000-2015 period: NB2 by maximum
  likelihood for n_clin_8y (200 bootstrap resamples, because the NB fit is
  slow), linear models for log RCR and the disruption score, and the logit
  with the 50 most frequent journals as indicators. Estimates and CIs only.
- The permutation control shuffles F1-F3 jointly, keeping each paper's
  three features together, within publication year.

Measurement gate, 2026-09-29, before any outcome was joined. Blind Opus
extraction of the 300 packets was split across three subagents of 100 each.
The record is results/meta_measurement_gate.json.
- model_system: kappa 0.845 (agreement 0.90). Kept.
- multi_system: kappa 0.732 (agreement 0.90). Kept.
- human_genetics: kappa 0.415 (agreement 0.89). The rule flagged 40 and the
  model 20; they agreed on 14. Dropped from the primary analysis as the gate
  requires. It is not re-tuned on these 300 papers.
The primary analysis runs with `--drop human_genetics`.

Amendment 1, 2026-09-29, after a code review and before any outcome was
joined. The review found measurement faults in the rules. The first gate
(above) stays on record. The changes:
- Eligibility. Research articles are excluded if they have no MeSH (not
  MEDLINE-indexed; the text fallback put 69% of them in `other`), no
  case-sensitive gene mention (acronym clashes such as "HRAs" for health risk
  appraisals and "NRAs" for nanorod arrays; about 113 papers), or a
  non-primary publication type (review, letter, editorial, comment,
  meta-analysis, systematic review, guideline, news). Only eligible papers
  enter the analysis.
- Human-sample presence for F3 counts only specific patient-material and
  cohort terms. Generic outcome terms and sex tags no longer count when
  animal or cell work is present.
- gene_group is case-sensitive.
- The gate is re-run on a fresh sample of 300 eligible preclinical papers
  (rng seed 1, disjoint from the first 300), with the same 0.60 bar. It
  decides which features enter the primary model. human_genetics stays
  dropped whatever the second gate shows, because its first-gate failure
  and the review's estimate of about 45% precision both stand.
- The outcome B window is publication year minus 1 through publication year
  plus 8, inclusive. It spans 10 calendar years to tolerate epub skew. This
  was in the corpus code and is now recorded here; it affects 12 citers.
