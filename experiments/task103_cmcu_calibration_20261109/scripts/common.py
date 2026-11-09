"""Paths shared by the portable experiment commands (no server defaults)."""
import os
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[1]
REPOSITORY = EXPERIMENT.parents[1]
ROOT = Path(os.environ.get('TASK103_RANK_ROOT', REPOSITORY / 'work/task103')).expanduser().resolve()
MODEL_DIR = Path(os.environ.get('TASK103_MODEL_DIR', REPOSITORY / 'models/ap304_rank')).expanduser().resolve()
