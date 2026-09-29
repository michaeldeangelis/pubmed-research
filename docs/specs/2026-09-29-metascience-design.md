# Which research designs lead to clinical use and field influence?

Metascience phase. Question from the project owner: which research
methodologies produce the best science and the breakthroughs. The analysis
plan in `experiments.md` (entry "2026-09-29 metascience v1") was fixed before
any feature was joined to any outcome.

## Outcomes

- **A, held up.** External ground truth only: the Reproducibility Project:
  Cancer Biology effect-level data (OSF e5nvr, CC-BY 4.0; 158 effects, 50
  experiments, 23 papers). Descriptive reference check, not a primary test.
- **B, led to clinical use (primary).** The paper is cited by a clinical
  article (iCite `cited_by_clin`) within 8 years of its publication year.
  Citing years come from iCite. Secondary forms: count of clinical citations
  within 8 years, and year of first clinical citation.
- **C, changed the field (secondary).** Field-normalized citation (iCite
  RCR) and a disruption score. The score is CD5 from SciSciNet when a PMID
  or DOI join exists, and otherwise the n_k-free variant
  `(n_i - n_j) / (n_i + n_j)` computed from OpenAlex. That variant uses
  5-year citers of the focal paper, and papers with no references are
  excluded. C is reported beside B and never combined with it.

## Corpus

Europe PMC, the ingest query from `src/ingest/pubmed.py`, `PUB_YEAR:[2000 TO
2015]`, `SRC:MED`. 12,211 PMIDs on 2026-09-29, and 10,047 are research
articles by iCite. The unit is one research article. Papers split into
**clinical** (iCite `is_clinical`) and **preclinical** (the rest). The
primary analysis uses preclinical papers, because for clinical papers a
clinical citation is close to automatic. Clinical papers are described but
not tested.

## Features (preclinical papers)

Primary, at most three, from MeSH headings, publication types, and fixed
text rules:

- **F1 model system**, categorical, precedence in this order: `human_samples`
  (Humans plus patient tissue or mutation analysis, with no animal or cell
  line terms), `animal` (in vivo animal terms, with or without cells),
  `cell_only` (cell line or cultured cell terms only), `other`. The
  reference level is `cell_only`.
- **F2 human genetic evidence**, binary: germline, predisposition,
  polymorphism, GWAS, or familial terms in MeSH or in the abstract.
- **F3 multiple systems**, binary: two or more of {human samples, animal,
  cell} present.

The exact term lists live in `src/meta/features.py`. They are frozen by the
commit that precedes the analysis run.

**Measurement check before outcomes.** The rules are compared with a blind
model extraction (subagents, no API key) on a random sample of 300
preclinical papers. A feature whose rule-versus-model Cohen's kappa is below
0.60 is dropped from the primary analysis before outcomes are joined, and
the drop is recorded. Disagreements, up to 40, go to the owner for
adjudication. That adjudication is reported and does not block.

## Covariates

Publication year as fixed effects, gene group (KRAS, BRAF, NRAS/HRAS,
other; from the title or abstract), log number of authors, and log
reference count. Journal is not a primary covariate, because journal choice
may sit between design and outcome. A sensitivity model adds the 50 most
frequent journals as indicators.

## Layout

| Path | Owner | Contents |
|---|---|---|
| `src/meta/corpus.py` | builder-data | PMIDs, Europe PMC core metadata, iCite, outcome B |
| `src/meta/features.py` | builder-features | F1-F3, covariates, validation packets and scoring |
| `src/meta/impact.py` | builder-impact | Outcome C: RCR, disruption |
| `src/meta/rpcb.py` | builder-impact | Outcome A reference check |
| `src/meta/analysis.py` | lead | The preregistered models |
| `tests/test_meta_*.py` | each owner | Unit tests on synthetic data |

Data goes under `data/meta/`, which is gitignored. Committed records go to
`results/meta_*.json`.

### Record schemas (JSON lines, one row per PMID)

- `data/meta/papers.jsonl`: pmid, doi, year, journal, title, abstract,
  mesh (list), pub_types (list), n_authors
- `data/meta/icite.jsonl`: iCite fields as returned, with pmid as a string
- `data/meta/outcome_b.jsonl`: pmid, clin_cited_8y (0/1), n_clin_8y,
  first_clin_year (int or null)
- `data/meta/features.jsonl`: pmid, group (clinical or preclinical),
  model_system, human_genetics (0/1), multi_system (0/1), gene_group,
  n_authors, n_refs (null if unknown)
- `data/meta/impact.jsonl`: pmid, rcr, cd_source (sciscinet, openalex_nok,
  or null), cd (float or null), n_refs_openalex

## Blinding

No builder computes an association between any feature and any outcome.
Builders may report marginal distributions of their own tables. Only
`src/meta/analysis.py`, run once by the lead after the plan is committed,
joins features to outcomes.
