# RAS/MAPK paper transformer

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
