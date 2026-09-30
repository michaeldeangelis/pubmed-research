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

Amendment 2, 2026-09-29, before any outcome was joined: the disruption score
(outcome C, secondary) uses cd_source "icite_nok". This is the same
n_k-free 5-year formula, computed on the iCite citation graph. SciSciNet
CD5 is gated behind a login. OpenAlex without a key allows 1,000 requests a
day, which is too slow for this corpus. On 291 papers scored from both
sources, Spearman between iCite and OpenAlex is 0.94. RPCB paper 5 is PMID
21102434, not 21102433 (corrected; see results/meta_rpcb.json).

Second measurement gate, 2026-09-29, before any outcome was joined. The
sample was 300 fresh eligible papers (rng seed 1), with the extraction again
split across three blind Opus subagents. The record is
results/meta_measurement_gate_v2.json.
- model_system: kappa 0.821 (agreement 0.88). Kept.
- multi_system: kappa 0.569 (agreement 0.86). The rule flagged 44 papers
  and the model 79; 39 of the model's positives were rule negatives. The
  stricter human-sample presence rule under-detects. Dropped, and not
  re-tuned.
- human_genetics: kappa 0.482. Dropped, as already decided.
The primary analysis runs with `--drop human_genetics multi_system`. The only
tested feature is model_system (human_samples, animal, and other, each
against cell_only).

Amendment 3, 2026-09-29, after a review of impact.py and before any outcome
was joined:
- The reference-count covariate is the OpenAlex count. When OpenAlex reports
  0 or nothing, the iCite count is used; if that is also 0 or absent, the
  value is missing (indicator). OpenAlex writes empty reference lists for
  works it never loaded. In eligible preclinical papers that was 141 false
  zeros, clustered by journal.
- The disruption score drops the focal paper from its own reference list.
  33 papers were self-listed, which forced cd = -1.
- The secondary RCR model uses log(RCR + 0.1), where it had used
  log(max(RCR, 0.001)). 108 eligible papers with RCR 0 had been extreme
  outliers.
- analysis.py now stops with an error if impact.jsonl is missing.
Known and not changed: the disruption window is Y..Y+5, with no Y-1
tolerance (391 citer links dated Y-1 are dropped). The RPCB check pools
internal replications and does not implement the prediction-interval
criterion, so its counts will not match Errington 2021's headline numbers
exactly.

### Result, 2026-09-29 (one run at commit 8f63b08, `--drop human_genetics multi_system`)

Eligible preclinical papers: 8,186 (development 2,813; confirmation 5,373).
The clin_cited_8y rate was 0.423 in development and 0.386 in confirmation.
ORs are against cell_only, with 1000-resample bootstrap 95% CIs.

| Term | Development OR [CI] | Confirmation OR [CI] | Permutation 95% | Pass |
|---|---|---|---|---|
| model_system=human_samples | 1.70 [1.39, 2.07] | 1.76 [1.50, 2.07] | [0.82, 1.23] | yes |
| model_system=animal | 0.85 [0.67, 1.10] | 0.71 [0.59, 0.85] | [0.78, 1.27] | no |
| model_system=other | 1.12 [0.76, 1.61] | 1.24 [0.91, 1.67] | [0.72, 1.44] | no |
| placebo (PMID even) | 1.04 [0.87, 1.22] | 1.11 [0.99, 1.25] | | CI includes 1 in both; placebo ok |

Ladder, AUROC on confirmation with models fit on development: trivial 0.500,
simplest 0.662, candidate 0.688. No CI was computed for the AUROC delta.

Verdict by the rule: pass, for model_system=human_samples only.

Secondary, full 2000-2015 period, estimate [95% CI], no pass/fail:
- n_clin_8y, NB rate ratio: human_samples 1.70 [1.46, 1.97]; animal 0.77
  [0.62, 0.92]; other 1.86 [1.20, 2.99].
- log(RCR + 0.1), linear: human_samples -0.095 [-0.151, -0.035]; animal
  0.056 [-0.004, 0.123]; other -0.033 [-0.138, 0.079].
- Disruption (icite_nok), linear, n=7,892: human_samples -0.128 [-0.152,
  -0.104]; animal -0.047 [-0.075, -0.021]; other -0.002 [-0.051, 0.049].
- With 50 journal indicators, logit OR: human_samples 1.97 [1.73, 2.30];
  animal 0.65 [0.55, 0.76]; other 1.30 [1.00, 1.69].
- Clinical papers, descriptive: 86% of the 552 human-sample papers were
  clinically cited, and 80% in each of the other small groups (n = 10 to 20).
- RPCB (outcome A): same direction and p<0.05 in 5/31 animal effects (16%)
  and 57/101 cell-based effects (56%) (results/meta_rpcb.json).

What this does and does not show. The population is MEDLINE-indexed primary
research on RAS/MAPK genes, 2000-2015, excluding trials and guidelines
flagged clinical by iCite. Among these papers, patient-sample studies are
about 1.7 times as likely as cell-line-only studies to be cited by a
clinical article within 8 years. The association held in both periods and
after journal adjustment. It is an association, not an effect of study
design. A patient-sample study reports facts about patients, such as
mutation prevalence or biomarker associations, which clinical articles cite
as background. So "cited clinically" partly measures topic, not the quality
or translational success of the research. The same papers score lower on
field-normalized citation and on disruption, so B and C diverge. Animal
studies had fewer clinical citations and, in RPCB, replicated worse, but the
animal term did not pass the preregistered rule. Human genetic evidence and
multiple model systems were not tested, because their rules failed the
measurement gate. Not shown: causation, approval-level translation, fields
beyond RAS/MAPK, or conduct (as opposed to reporting) of any design feature.

## 2026-09-29 breakthrough case histories   (method fixed before the run; descriptive)

Question: On the citation paths to two approvals, vemurafenib (BRAF V600E)
and sotorasib (KRAS G12C), which model systems carried the evidence, and how
does that mix compare with the field's base rate?
Decision it drives: whether the patient-sample signal from metascience v1
also holds on the path to approval, or whether cell and animal work
dominate there. The answer shapes the next wider-field study.
Cases and anchors (PMIDs verified against Europe PMC before the run):
- vemurafenib: BRIM-3, Chapman et al. 2011 NEJM.
- sotorasib: CodeBreaK 100 phase 1, Hong et al. 2020 NEJM; phase 2,
  Skoulidis et al. 2021 NEJM.
Lineage: depth 1 is the iCite references of the anchors; depth 2 is the
references of depth 1. Only research articles are kept, restricted to the
papers published before the anchor. The named milestone papers (discovery
of the mutation, the first selective inhibitor, preclinical proof) are
listed by hand, before the run, with the reason for each.
Classification: the model_system rules from src/meta/features.py, unchanged
(gate kappa 0.82). Papers without MeSH are reported as unclassified.
Comparison: the lineage's model_system shares at each depth against the
metascience v1 eligible corpus, restricted to the same publication years.
Report the share with a Wilson 95% CI. With two cases there is no pass/fail.
Record: results/case_histories.json
Milestones, listed before the run. PMIDs are from memory and are checked
against titles by the builder; corrections are recorded in the result.
- vemurafenib: Davies 2002 Nature, BRAF mutations in cancer (12068308),
  discovery; Tsai 2008 PNAS, PLX4720 selective BRAF V600E inhibitor
  (18287029), first selective tool compound; Bollag 2010 Nature, PLX4032
  clinical efficacy (20823850), preclinical plus clinical proof; Flaherty
  2010 NEJM, phase 1 (20818844), first human efficacy; Chapman 2011 NEJM,
  BRIM-3 (21639808), approval trial.
- sotorasib: Ostrem 2013 Nature, K-Ras(G12C) allosteric inhibitors
  (24256730), discovery of the druggable pocket; Patricelli 2016 Cancer
  Discov, ARS-853 (26739882), cellular proof; Janes 2018 Cell, ARS-1620
  (29373830), in vivo proof; Canon 2019 Nature, AMG 510 (31666701),
  preclinical candidate; Hong 2020 NEJM, phase 1 (32955176); Skoulidis 2021
  NEJM, phase 2 (34096690), approval trial.

### Result, 2026-09-29 (descriptive)

Record: results/case_histories.json. All 11 anchor and milestone PMIDs
matched Europe PMC; none needed correction. The lineages hold 1,578 research
articles (79 at depth 1, 1,499 at depth 2). Janes 2018 appears only at
depth 2; every other milestone is at depth 1.

Deviations from the method as written:
- Lineage papers from the same year as the anchor are included. The entry
  said "before the anchor"; the run used publication year <= anchor year.
- The sotorasib cutoff is the later anchor's year, 2021.
- The comparison window is clipped to the corpus years: 2000-2011 for
  vemurafenib, 2000-2015 for sotorasib. That leaves out 33 of 55 sotorasib
  depth-1 papers, including Patricelli 2016, Janes 2018 and Canon 2019.
- Lineage papers are not restricted to RAS/MAPK topics, while the base is.
  An on-topic-only sensitivity check was added and is labelled as extra.

Preclinical shares, lineage against base, difference with Newcombe 95% CI:
- vemurafenib, n=255 against 2,813: human_samples 0.38 against 0.52, -0.14
  [-0.20, -0.08]; cell_only 0.36 against 0.21, +0.15 [+0.09, +0.22].
- sotorasib, n=523 against 8,186: human_samples 0.41 against 0.54, -0.13
  [-0.18, -0.09]; animal +0.04 [+0.01, +0.08]; other +0.06 [+0.04, +0.09].
- combined: human_samples -0.14 [-0.18, -0.11]; cell_only +0.07
  [+0.04, +0.11].
- On-topic only: vemurafenib human_samples -0.26 [-0.33, -0.17] and
  cell_only +0.21 [+0.13, +0.30]; sotorasib differences near 0.
With trials included, only vemurafenib differs: human_samples -0.08
[-0.14, -0.02]. Sotorasib's depth-1 papers are mostly patient and trial
papers: human_samples 0.81 against 0.57, n=21.
Timelines: vemurafenib ran from discovery in 2002 to the approval trial in
2011; sotorasib from the pocket and first compound in 2013 to the approval
trial in 2021.

What this does and does not show: in two cases, the preclinical evidence
under an approval drew less on patient-sample studies and more on cell
work than the field did in the same years. The pattern is clear for
vemurafenib and weaker, and sensitive to topic restriction, for sotorasib.
Set beside metascience v1, the designs that clinical articles cite most
(patient samples) differ from the designs these approvals were built on.
Two cases establish no general rule. The model_system labels are coarse:
Davies 2002 comes out cell_only although it screened tumors.

## 2026-09-30 metascience v2: other pathways   (rules fixed before the run)

Question: Does the v1 finding, that patient-sample studies are more likely
than cell-line-only studies to be cited by a clinical article within 8
years, hold in pathways not used to form it?
Decision it drives: confirmed in both pathways means the finding
generalizes beyond RAS/MAPK and is worth a translation-outcome study. Not
confirmed means it is field-specific or topic-driven, and is recorded as
such.
Pathways, each a separate corpus, 2000-2015, Europe PMC SRC:MED:
- EGFR/ERBB: TITLE_ABS terms EGFR, ERBB2, HER2, ERBB3, erlotinib,
  gefitinib, osimertinib, afatinib, lapatinib, cetuximab, panitumumab,
  trastuzumab.
- PI3K/AKT/mTOR: TITLE_ABS terms PIK3CA, PTEN, AKT1, MTOR, mTORC1,
  everolimus, temsirolimus, alpelisib, idelalisib.
Papers already in the v1 RAS/MAPK corpus are excluded from both. A paper in
both new corpora is kept in each, and the overlap is reported.
Held fixed from v1: iCite research articles, the clinical/preclinical
split, the eligibility rules (MeSH present, primary publication type, and
a case-sensitive mention of a pathway gene or drug from that pathway's
list), the unchanged model_system rules, outcome B (clinical citation from
publication year minus 1 through plus 8), the covariates (year FE, gene
group with a pathway-specific list, log authors, the reference count as
amended), and the logit with 1000-resample bootstrap CIs (seed 0).
Measurement gate: 300 eligible preclinical papers, 150 per pathway (rng
seed 2), blind Opus extraction with the same schema, kappa >= 0.60 for
model_system. If it fails, the pathway results are reported as unreliable.
Primary test, per pathway, full 2000-2015 period with no split: confirmed
if the OR for model_system=human_samples against cell_only is > 1 with a
95% CI excluding 1, AND the placebo (PMID even) CI includes 1.
Exploratory: model_system=animal and =other; the permutation control
(1000, within year); log(RCR + 0.1); the disruption score; the journal
sensitivity model.
Kill: not confirmed in either pathway.
Record: results/meta_pathways.json

Before any outcome was joined, 2026-09-30:
- Corpora. EGFR has 52,035 papers (2,772 RAS papers excluded) and 30,144
  eligible preclinical. PI3K has 29,572 (1,514 excluded) and 21,046. The two
  corpora share 1,314 eligible preclinical papers. Most of the 5,490 EGFR
  off_topic exclusions are kidney "eGFR" papers, which a case-insensitive
  search pulls in.
- Measurement gate (results/meta_measurement_gate_pathways.json):
  model_system kappa 0.843 (agreement 0.89), a pass. For reference only,
  since neither is tested in v2: human_genetics 0.712, multi_system 0.525.
- Reference count. The OpenAlex daily credit ran out during the EGFR run,
  so 964 eligible EGFR papers use the iCite count, as the amended fallback
  rule allows. It is recorded in each pathway's manifest_impact.json.

Amendment v2-1, 2026-09-30, after a code review and before any outcome was
joined:
- Reference count. The 964 EGFR papers without an OpenAlex count are
  concentrated in 2015 (930 of 3,439 eligible), so the covariate would mix
  two sources within one year. For both pathways the covariate is now the
  iCite reference count for every paper, with missing if absent. v1 is
  unchanged.
- Eligibility fixes. Case variants (EGFr, EgfR, mTor, mTorC1, mTORc1, pTEN)
  now count as on-topic. PI3K papers whose only match is AKT and that are
  plant K+ channel papers are excluded as off_topic_plant. EGFR papers whose
  only match is EGFR used as estimated glomerular filtration rate are
  excluded as off_topic_kidney. The gate sample and its score are not
  regenerated; model_system rules are unchanged.
- Known limitation, kept for comparability with v1: the ingest markup
  stripper deletes text between "<" and ">", which removes about 280
  on-topic papers as off_topic. The PI3K on-topic rule accepts any AKT
  isoform, mTORC2 and p110alpha, which is wider than the query list; this is
  documented in src/meta/pathways.py.
- Applied (builder report): the case-variant fix made 115 EGFR and 58 PI3K
  papers eligible; 57 PI3K papers were excluded as off_topic_plant and 15
  EGFR preclinical papers as off_topic_kidney. The plant rule was split into
  strong and weak signals, because MEDLINE tags ordinary mammalian Akt1
  papers with "Arabidopsis Proteins". None of the 300 gate packets changed
  eligibility.

### Result, 2026-09-30 (one run at commit 8db3e16)

Full 2000-2015 period per pathway. ORs are against cell_only, with
1000-resample bootstrap 95% CIs. The permutation range is the central 95%
of 1000 within-year shuffles.

| Pathway | n | clin_cited_8y | human_samples OR [CI] | Permutation 95% | Placebo OR [CI] | Confirmed |
|---|---|---|---|---|---|---|
| EGFR/ERBB | 30,239 | 0.294 | 3.02 [2.84, 3.25] | [0.94, 1.06] | 1.00 [0.95, 1.04] | yes |
| PI3K/AKT/mTOR | 21,046 | 0.284 | 2.22 [2.03, 2.44] | [0.91, 1.09] | 1.03 [0.97, 1.09] | yes |

Verdict by the rule: confirmed in both pathways.

Exploratory, estimate [95% CI]:
- animal OR: EGFR 1.07 [0.98, 1.14]; PI3K 1.23 [1.14, 1.32]. In v1
  RAS/MAPK it was 0.85 and 0.71, so the animal association changes sign
  across fields and does not generalize.
- other OR: EGFR 1.08 [0.94, 1.24]; PI3K 1.09 [0.93, 1.27].
- n_clin_8y NB rate ratio, human_samples: EGFR 3.25 [2.82, 3.69]; PI3K
  2.96 [2.54, 3.34].
- log(RCR + 0.1), human_samples: EGFR -0.139 [-0.165, -0.114]; PI3K -0.195
  [-0.233, -0.158].
- Disruption, human_samples: EGFR -0.104 [-0.118, -0.091]; PI3K -0.100
  [-0.120, -0.082].
- With journal adjustment, human_samples OR: EGFR 2.82 [2.61, 3.02]; PI3K
  2.22 [2.01, 2.48].

What this does and does not show. In three cancer-signaling fields
(RAS/MAPK, EGFR/ERBB, PI3K/AKT/mTOR, 2000-2015), primary research on
patient samples is 1.7 to 3.0 times as likely as cell-line-only research to
be cited by a clinical article within 8 years. In every field, the same
papers score lower on field-normalized citation and on disruption. The
direction is stable, and the size varies by field. It remains an
association: clinical articles cite patient studies partly as background
facts about patients, so a clinical citation partly measures topic. The
animal association is field-specific. Not shown: causation, approval-level
translation, fields outside cancer signaling, or anything about how well
the studies were conducted.

## 2026-09-30 metascience v3: trial background citations   (rules fixed before the run)

Question: Is patient-sample research more or less likely than
cell-line-only research to be cited as BACKGROUND in the registration of an
interventional clinical trial? This is a direct link from a paper to a
trial, not a citation by a clinical article.
Decision it drives: the same direction as v1 and v2 means clinical uptake
of patient-sample work reaches trial design, not only clinical articles.
The opposite direction means trials build on cell or animal work, as the
case histories suggested, and that "cited clinically" and "led to a trial"
come apart. Null or underpowered means the registry link cannot answer it.
Data: ClinicalTrials.gov API v2, all registered studies, fetched once and
cached. For each study: NCT ID, study type, start date, phases,
intervention types, and references with PMID and type. Only references of
type BACKGROUND from INTERVENTIONAL studies count. RESULT and DERIVED
references are excluded, because they are the trial's own reports.
Outcome: trial_bg_8y = 1 if the paper is a BACKGROUND reference of an
interventional study whose start year is within publication year minus 1
through plus 8. Secondary: the count of such trials; restriction to trials
with a DRUG or BIOLOGICAL intervention; restriction to phase 2 or later.
Papers: the eligible preclinical papers of v1 RAS/MAPK, v2 EGFR and v2 PI3K,
with the features, eligibility and covariates each version already uses
(v1 reference count as amended; iCite reference count for EGFR and PI3K).
A paper in more than one corpus is kept once, in the first of RAS, EGFR,
PI3K.
Model: logit, trial_bg_8y ~ model_system + year FE + pathway FE +
gene_group within pathway + log authors + log(1 + refs) + refs_missing,
pooled over the three pathways, full 2000-2015. CIs are 1000-resample
bootstrap (seed 0).
Primary: the human_samples OR against cell_only. Verdict "same direction"
if the CI lies entirely above 1, "opposite" if entirely below 1, "null"
otherwise. The placebo (PMID even) CI must include 1, or the result is
flagged unreliable. If there are fewer than 100 outcome events among
cell_only plus human_samples papers, the verdict is "underpowered",
whatever the CI.
Linkage check before the join: 30 random linked pairs (paper, trial) are
verified by reading the trial record's reference entry. Report the count
of correct links.
Exploratory: the animal and other ORs; per-pathway ORs; the drug and
phase-2+ restrictions; the permutation control (1000, within year and
pathway).
Record: results/meta_trials.json

Before any outcome was joined, 2026-09-30:
- Fetch. ClinicalTrials.gov API v2 on 2026-09-29: 604,950 studies (461,679
  interventional). Interventional studies list 592,485 BACKGROUND
  references. The PMID index has 409,613 PMIDs and 550,691 (PMID, trial)
  pairs.
- Outcome marginals. RAS: 116 of 8,186 papers (1.42%). EGFR: 333 of 30,239
  (1.10%). PI3K: 241 of 21,046 (1.15%). Pooled with first-kept
  deduplication: 672 events in 58,146 papers.
- Linkage check. In 30 of 30 sampled pairs the trial's citation string
  contains the paper's full title. An independent recomputation of the RAS
  window matched the builder's output exactly (116 events, 0 mismatches).
- Definitions fixed by the builder, accepted before the run. Phase 2 or
  later means every listed phase is 2, 3 or 4; phase 1/2 is excluded. Drug
  means a DRUG or BIOLOGICAL intervention. A trial citing the same PMID
  twice counts once. Start year is the year of the start date, which may be
  anticipated. Trials with no start date are left out of the window (0, 1
  and 3 linked papers in RAS, EGFR and PI3K).

### Result, 2026-09-30 (one run at commit 31757a9)

Pooled eligible preclinical papers: 58,146, with 672 trial_bg_8y events
(1.16%). cell_only plus human_samples papers have 447 events, above the
100 minimum. ORs are against cell_only, with 1000-resample bootstrap 95%
CIs.

| Model | human_samples | animal | other | placebo |
|---|---|---|---|---|
| Primary, trial_bg_8y | 1.92 [1.54, 2.43] | 1.31 [1.06, 1.65] | 0.99 [0.60, 1.51] | 0.98 [0.84, 1.15] |
| Drug or biological trials | 1.60 [1.24, 2.07] | 1.17 [0.93, 1.49] | 0.95 [0.52, 1.49] | 1.01 [0.85, 1.19] |
| Phase 2 or later | 1.95 [1.50, 2.63] | 1.16 [0.89, 1.58] | 0.82 [0.36, 1.46] | 1.01 [0.83, 1.20] |

The permutation 95% range (within year and pathway) is [0.83, 1.20] for
human_samples and [0.81, 1.23] for animal.

Verdict by the rule: same direction.

Per pathway, exploratory, human_samples OR:
- RAS, 116 events: 0.93 [0.58, 1.52]
- EGFR, 333 events: 2.55 [1.88, 3.64]
- PI3K, 223 events after deduplication: 1.78 [1.14, 2.71]
The animal OR is 0.62 [0.31, 1.15] in RAS, 1.49 [1.06, 2.08] in EGFR and
1.41 [1.02, 1.99] in PI3K.

What this does and does not show. A direct paper-to-trial link, a BACKGROUND
reference in a trial registration, shows the same direction as clinical
citation. Patient-sample research is about twice as likely as cell-line-only
research to be cited as background by an interventional trial within 8
years. That holds for drug trials and for phase 2 or later. So the v1 and v2
association is not only a matter of what clinical articles cite; it reaches
trial design too. Two qualifications. First, the RAS pathway, where the
finding was formed and whose drug case histories rested on cell work, shows
no association on this outcome (116 events, wide CI). Second, animal work is
also above cell-only on this outcome, which it was not for clinical
citation in RAS. Being cited as background is still not the same as the
trial depending on the paper, and approval is not measured. The ORs are
associations, and a trial's background section may cite patient data for
context, much as clinical articles do.

## 2026-09-30 metascience v4: design against topic   (rules fixed before the run)

Question: Among papers on the same topic, are patient-sample papers still
more likely than cell-line-only papers to be cited clinically, or cited as
trial background? The alternative is that the v1-v3 associations reflect
topic rather than design.
Decision it drives: if the effect persists within matched topic, design
itself carries it and the next step is rigor features. If it is
attenuated, the effect is mostly topic and later work should model topic
explicitly.
Papers: eligible preclinical papers of RAS, EGFR and PI3K, deduplicated
keeping the first of RAS, EGFR, PI3K, as in v3. Only model_system
human_samples (treated) and cell_only (controls) are used.
Topic representation: TF-IDF over each paper's MeSH descriptors, with
qualifiers dropped. Every design-revealing descriptor is removed first:
every term in the model_system rule lists of src/meta/features.py, check
tags (Humans, Animals, Female, Male, age groups), organism and cell-line
descriptors, study-design and statistics descriptors (e.g. Retrospective
Studies, Cohort Studies, Survival Analysis, Kaplan-Meier Estimate,
Prognosis, Immunohistochemistry), and laboratory-technique descriptors.
The builder writes the removed list to a file before matching. Topic
similarity is cosine similarity.
Matching: 1:1 greedy nearest neighbour without replacement, exact on
pathway and gene_group, with publication year within +/-1. Treated papers
are processed in descending order of their best available similarity,
ties broken by PMID. The caliper is the largest of {0.6, 0.5, 0.4, 0.3}
that matches at least 50% of treated papers. It is chosen from similarity
distributions only, before any outcome is read.
Outcomes: clin_cited_8y (v1/v2 outcome B) and trial_bg_8y (v3).
Primary estimate, per outcome, pooled: matched-pair (McNemar) OR = b/c,
where b = pairs in which only the treated paper has the outcome and c =
pairs in which only the control does. The 95% CI is exact (Clopper-Pearson
on b/(b+c), transformed). Comparator: the crude OR over all treated and
all controls, and the crude OR over matched papers treated as unpaired.
Verdict per outcome: "persists" if the matched CI lies entirely above 1;
"reversed" if entirely below 1; otherwise "attenuated to null". Also
report attenuation = 1 - log(matched OR) / log(crude OR).
Placebo: the same matched-pair OR for PMID parity; its CI must include 1.
Balance, reported: the topic similarity of matched pairs, standardized
differences of log authors and log(1 + refs) between arms, and the match
rate per pathway.
Matching-quality check: 50 random matched pairs (rng seed 0). A blind Opus
judge sees only the two titles and abstracts, with design words not
masked, and rates "same research topic (same gene/alteration and
disease/question)" as yes or no. If fewer than 70% are yes, topic matching
is flagged weak and the verdict is reported as provisional.
Exploratory: per-pathway matched ORs; the trial outcome restricted to drug
or biological trials.
Record: results/meta_matched.json

Amendment v4-1, 2026-09-30, before any outcome was joined:
- Caliper. No caliper in {0.6, 0.5, 0.4, 0.3} matches 50% of treated
  papers. The rates were 3.4%, 7.2%, 13.8% and 24.4%. Reaching 50% would
  need a similarity near 0.1, which is close to no topic match, and the
  exact pathway, gene and year constraints alone allow only 59.9%. Caliper
  0.3 is kept (4,839 pairs: RAS 863, EGFR 3,071, PI3K 905), because topic
  closeness is the point of v4. The 50% target was set before the
  similarity distribution was known. The decision uses similarity
  distributions only (bench/meta_validation/matching_summary.json).
- Pairs where either paper lacks an outcome are dropped.
- Balance after matching: log authors std diff 0.057, and log(1 + refs)
  -0.493 (it was -0.680 before). Refs stay imbalanced, so an exploratory
  analysis restricts to pairs with |log(1 + refs) difference| <= 0.5,
  excluding pairs where either count is missing.
- Removed-descriptor list: 662 descriptors (bench/meta_validation/removed_mesh.json).
  10,094 topic descriptors remain.
