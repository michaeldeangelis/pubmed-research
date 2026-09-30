# RAS/MAPK paper transformer

## Abstract

This project asked whether a computer could learn how scientific papers relate to one another. It has not shown that. What it has produced is a written record of individual experimental results.

The starting material is 65 claims about the RAS/MAPK pathway, taken from papers published between 2006 and 2014. Later papers were searched for tests of those claims. Searching with the words of the claim found many more relevant papers than taking the next papers in time order. Reading further down the search list turned up more papers for one claim that was already covered. It did not uncover tests for claims that had none.

Twenty-seven results from three claims were then written out one by one. Each result states the question, who or what was studied, what it was compared with, the conditions of the experiment, the number reported, and how uncertain that number is. A paper can contain several of these results. Whether two results agree is decided only after those details are written down.

The three claims show three different reasons that papers can seem to disagree. In the first, the experiments were set up differently, and the outcomes differed with them. In the second, percentages that look alike were calculated on different groups of patients, so they are answers to different questions. In the third, some papers reported a statistically significant association and others did not. A result that fails to reach significance is not, by itself, evidence of the opposite effect. Estimates can still be compatible once the exact comparison and its uncertainty are in view.

The record keeps those three differences separate: a change in the experiment, a change in the question being answered, and a change in how sure the estimate is. It does not yet show that one experiment caused the outcome of another, that the pattern holds beyond these cases, or that a model can predict a new result.

## Earlier probe

A small transformer over RAS/MAPK papers did not learn a readable trace of earlier human genetic evidence. On papers from 2021–2025, a linear probe for that history scored 0.518 from the model's hidden state and 0.507 from the frozen text vector. The lift is 0.011. Record: `results/probe.json`.

Tokens are papers, not words. Each head may attend to at most 8 earlier papers. The loss predicts the next paper's genes, drugs, and clinical flag. Human genetics and resistance tags are withheld from the inputs and from the loss. The probe asks whether the hidden state still carries them.

## Transformer v2

v2 asked whether the transformer beats a model without attention and whether it carries prior genetic evidence. The decision rule was fixed before the first run. `docs/specs/2026-09-29-model-v2-design.md` has the rule, the changes from v1, and the fixes below. Record: `results/probe_v2.json`.

The next-paper targets leave out the timeline's own gene and add genes and drugs not yet seen in the window. Training positions are those whose next paper is from 2018 or earlier. 2019–2020 picks the epoch. Three mixers share the same data and seeds: top-k attention, a uniform mean over the last 8 papers, and each paper alone.

v2 is not fruitful under its rule. Attention did not predict the next paper better than the mean. Mean test loss was 0.9547 for attention, 0.9474 for the mean, and 0.9541 for each paper alone. The gap of −0.0072 is larger than the seed spread of 0.0033, in the wrong direction. The genetics probe did pass its bar. It rose from 0.516 on the frozen text vector to 0.634 from attention, a lift of 0.119 with a bootstrap interval of 0.069 to 0.166. Both criteria were required.

A post hoc check asked whether that lift only reflects seeing earlier papers. Record: `results/probe_v2_context_check.json`. Attention scored 0.634, the mean mixer 0.586, and an untrained probe on the mean text of all earlier papers 0.563. Attention exceeded the mean mixer by 0.048, with an interval of 0.025 to 0.069. The check was added after the decision and reuses the same test years, so it is exploratory.

That comparison was then tested once on 2026 papers, under a rule committed before they were fetched. Record: `results/probe_v2_confirm_2026.json`. The fetch added 80 abstracts and 206 test positions, 108 of them with prior genetic evidence. The year is the publication year. First online dates run from 2024-10-14 to 2026-08-05, and the PMIDs are in `results/confirm_2026_pmids.json`. No model was retrained. Attention again exceeded the mean mixer, by 0.044, but the interval ran from −0.007 to 0.098. It was not confirmed. On the same papers, an untrained probe on the mean text of earlier papers scored 0.671, above attention at 0.601. v2 is closed.

## Which research designs reach the clinic?

This phase asked which research designs go on to be used clinically, and which change their field. The corpus is 12,211 RAS/MAPK papers from 2000 to 2015. 8,186 of them are MEDLINE-indexed primary research that iCite does not flag as a clinical study. The plan, both measurement gates, and every amendment were recorded before any design feature was joined to any outcome. Plan and record: `experiments.md`. Spec: `docs/specs/2026-09-29-metascience-design.md`. Result: `results/meta_analysis.json`.

The design feature is the model system, read from MeSH headings: patient samples, animals, cell lines only, or other. A blind model reading of 300 fresh papers agreed with the rules at kappa 0.82. Two other features, human genetic evidence and use of more than one model system, fell below the 0.60 agreement bar and were not tested.

The primary outcome is a citation from a clinical article within 8 years. Patient-sample studies were more likely to get one than cell-line-only studies: odds ratio 1.70 (1.39 to 2.07) for 2000–2011, and 1.76 (1.50 to 2.07) on the held-out 2012–2015 papers. That is the only design that passed the preregistered rule. Animal studies were lower in 2012–2015, at 0.71 (0.59 to 0.85), but not in 2000–2011, so they did not pass. A placebo feature showed no association. Adjusting for journal did not remove the patient-sample association.

The outcomes disagree. The same patient-sample papers scored lower on field-normalized citation and on disruption. They get used by clinicians more, and they change their field less. In the Reproducibility Project: Cancer Biology data, 5 of 31 animal effects replicated with the same direction and significance, against 57 of 101 cell-based effects (`results/meta_rpcb.json`).

This is an association, not an effect of the design. Patient studies report facts about patients, which clinical articles cite as background, so a clinical citation partly measures the topic rather than the quality of the work. It does not show approval-level translation, and it does not reach beyond RAS/MAPK.

## The finding in two more pathways

The patient-sample result was then tested in two pathways that played no part in forming it: EGFR/ERBB, with 30,239 eligible papers, and PI3K/AKT/mTOR, with 21,046. Papers from the RAS/MAPK corpus were left out. The rule was committed before the data was built. Record: `results/meta_pathways.json`.

It held in both. Against cell-line-only studies, the odds ratio for a clinical citation within 8 years was 3.02 (2.84 to 3.25) for EGFR and 2.22 (2.03 to 2.44) for PI3K. Placebos were null, and the within-year shuffles stayed between 0.91 and 1.09. In every pathway, the same patient-sample papers scored lower on field-normalized citation and on disruption. The animal association did not carry over: it was below 1 in RAS/MAPK, near 1 in EGFR, and 1.23 in PI3K.

Across three cancer-signaling fields, patient-sample research is 1.7 to 3 times as likely to be cited clinically, and less likely to change its field. It is still an association, and a clinical citation still partly measures topic.

## What the approvals were built on

Two approvals were traced back through their citations: vemurafenib, whose approval trial was in 2011, and sotorasib, in 2021. The lineage is the papers each approval trial cites, and the papers those cite. The same model-system rules were applied to it, then compared with the field in the same years. The method and the milestone papers were fixed before the run. Record: `results/case_histories.json`.

The preclinical work under both approvals used fewer patient-sample studies than the field did. The share was 0.38 against 0.52 for vemurafenib and 0.41 against 0.54 for sotorasib; combined, the difference is −0.14 (−0.18 to −0.11). It used more cell-line work: 0.36 against 0.21 for vemurafenib. The milestones were cell or animal work: the BRAF mutation screen, the selective tool compounds, and the preclinical candidates. Taken with the result above, clinical articles cite patient-sample studies most, while these two approvals rest more on cell work.

Two cases set no rule. The corpus ends in 2015, which leaves sotorasib's 2016–2019 preclinical papers out of its comparison. The sotorasib gap disappears when the lineage is limited to RAS/MAPK papers. The model-system labels are coarse.

## Result-record extraction spike

After v2 closed, a spike asked whether the 27 result records of the evidence map can be rebuilt blind from the claim and the abstract. A second question was whether a fixed rule then names the map's disagreement type for each claim. The bars were committed before extraction. Spec: `docs/specs/2026-09-29-result-records-spike.md`. Record: `results/result_records_spike.json`.

It is not feasible in this setup. Neither extractor cleared the bars. The bar was 0.80 for each score.

| Extractor | Numbers recovered | Population | Comparator | Quantity estimated | Rule: C024, C033, C015 |
|---|---|---|---|---|---|
| Sonnet | 0.92 | 0.93 | 0.59 | 0.48 | conditions, conditions, tie |
| Opus | 0.86 | 0.85 | 0.41 | 0.48 | consistent, conditions, tie |

The reference labels are conditions, estimand, and certainty. Most missed numbers were rates the reference had computed. The two extractors agreed on direction in 26 of 27 records and on significance in all 27. They agreed on whether conditions departed from the claim in 0.70 of records, and on whether the quantity matched the claim in 0.48.

The extractors copied the facts. They did not pin down the exact comparison and denominator, which are what the map's judgments depend on. In C033 a special population and a different denominator are the same fact, so one yes-or-no field cannot separate them. The reference, the extractors, and the judge are all Claude models. The three claims were chosen because they disagree.

The first v2 run is kept in `results/probe_v2_run1_year_bug.json` and is not used. Year embeddings after 2018 were never trained, so every test paper carried a random vector. v1 has the same fault. The v1 probe also starts from a random point. On the same frozen vectors its AUROC ranged from 0.485 to 0.567 across ten starts. The v1 lift of 0.011 is inside that range.

## Corpus and split

Europe PMC, 1998–2025, at most 80 cited PubMed abstracts per year. The fetch returned 1,958 abstracts. 1,882 mentioned a gene in the vocabulary and entered a timeline. Text vectors are from `pritamdeka/S-PubMedBert-MS-MARCO`. The probe is fit on years through 2018 and scored on years after 2020.

Training loss moved from 1.064 to 0.780 over 8 epochs. Zeroing the genetics direction raised next-clinical loss by 0.0024 when prior genetics was in the window and by 0.0017 when it was not. The largest head put 0.12 of its attention on earlier genetics papers. With 8 slots, a uniform head is already near 0.125.

The frozen encoder does separate clinical abstracts. That check is not in `results/probe.json`. It scored 0.87 on the same time cut. A genetics direction fit on the pre-2019 text vectors scored 0.70 there and 0.41 after 2020.

## Reproduce

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
python -m src.ingest.pubmed
python -m src.claims.extract
python -m src.encode.encoder
python -m src.model.train
python -m src.probe
python -m src.model.train_v2
python -m src.probe.v2
python -m src.probe.context_check_v2
python -m src.probe.confirm_2026
```

`confirm_2026` fetches from Europe PMC once and caches the papers in `outputs/v2/confirm_2026/`. The 2026 fetch recorded above ran on 2026-09-29.

The numbers above used PyTorch 2.14.0 on Apple MPS. Abstracts, embeddings, and the checkpoint stay in `data/` and `checkpoints/`, which are gitignored. `python -m src.probe` writes `outputs/probe_report.json`.

## Layout

| Path | Contents |
|---|---|
| `src/ingest/pubmed.py` | Europe PMC fetch |
| `src/claims/extract.py` | Genes, drugs, diseases, relations, evidence tags |
| `src/encode/encoder.py` | Frozen PubMedBERT vectors, hash fallback |
| `src/model/transformer.py` | Causal top-k paper transformer |
| `src/probe/probes.py` | Held-out probe and ablation |
| `results/probe.json` | The v1 run recorded above |
| `src/model/*_v2.py`, `src/probe/*v2.py` | Transformer v2, mixers, probe, decision rule |
| `results/probe_v2*.json` | v2 decision, the discarded first run, the post hoc check, the 2026 confirmation |
| `team/` | Task board and teammate notes |

## Cite

Use the Cite this repository button. It reads `CITATION.cff`.
