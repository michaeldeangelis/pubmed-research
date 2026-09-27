"""Frozen v1 hyperparameters. Teammates import these; do not retune silently."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
CHECKPOINTS = ROOT / "checkpoints"

PAPERS_PATH = DATA / "papers.jsonl"
ENRICHED_PATH = DATA / "papers_enriched.jsonl"
EMBEDDINGS_PATH = DATA / "embeddings.npy"
EMBEDDING_META_PATH = DATA / "embeddings_meta.json"
CHECKPOINT_PATH = CHECKPOINTS / "model.pt"
PROBE_REPORT_PATH = OUTPUTS / "probe_report.json"

TEXT_DIM = 768
D_MODEL = 256
N_LAYERS = 4
N_HEADS = 8
TOP_K = 8
WINDOW = 24
MIN_WINDOW = 4
DROPOUT = 0.1

YEAR_MIN = 1998
YEAR_MAX = 2025
TRAIN_YEAR_MAX = 2018
VAL_YEAR_MAX = 2020
# test is year >= 2021

PER_YEAR = 80
EPOCHS = 8
LR = 3e-4
WEIGHT_DECAY = 0.01
BATCH_SIZE = 16
GRAD_CLIP = 1.0
LOSS_WEIGHT_CLINICAL = 1.0
LOSS_WEIGHT_GENE = 0.5
LOSS_WEIGHT_DRUG = 0.5

ENCODER_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"
HASH_ENCODER_NAME = "hash768-v1"

EUROPEPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
