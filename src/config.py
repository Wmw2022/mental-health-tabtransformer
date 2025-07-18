"""
Global configuration: paths, hyperparameters, class mappings, etc
"""
from pathlib import Path

# Root directory path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH     = PROJECT_ROOT / "src" / "data" / "data.xlsx"
OUTPUT_DIR    = PROJECT_ROOT / "outputs"

# Unified save prefix for models and charts
TIMESTAMP_FMT = "%Y%m%d_%H%M%S"

# Class mappings
LABEL_MAP = {'mild': 0, 'moderate': 1, 'severe': 2}
N_CLASSES = len(LABEL_MAP)

# Training hyperparameters
BATCH_SIZE   = 256
VAL_BATCH    = 128
TEST_BATCH   = 128
MAX_EPOCHS   = 100
PATIENCE     = 15
LR           = 2e-3
DEVICE       = "cuda:1"
