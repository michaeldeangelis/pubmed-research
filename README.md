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
```

The numbers above used PyTorch 2.14.0 on Apple MPS. Abstracts, embeddings, and the checkpoint stay in `data/` and `checkpoints/`, which are gitignored. `python -m src.probe` writes `outputs/probe_report.json`.

## Layout

| Path | Contents |
|---|---|
| `src/ingest/pubmed.py` | Europe PMC fetch |
| `src/claims/extract.py` | Genes, drugs, diseases, relations, evidence tags |
| `src/encode/encoder.py` | Frozen PubMedBERT vectors, hash fallback |
| `src/model/transformer.py` | Causal top-k paper transformer |
| `src/probe/probes.py` | Held-out probe and ablation |
| `results/probe.json` | The run recorded above |
| `team/` | Task board and teammate notes |

## Cite

Use the Cite this repository button. It reads `CITATION.cff`.
