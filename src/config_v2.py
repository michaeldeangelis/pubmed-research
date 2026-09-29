"""v2 settings. Re-exports every v1 setting; adds v2-only knobs."""

from __future__ import annotations

from src.config import *  # noqa: F401,F403
from src.config import CHECKPOINTS, OUTPUTS, ROOT

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
