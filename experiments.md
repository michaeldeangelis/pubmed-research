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
