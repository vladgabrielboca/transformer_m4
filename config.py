# Data parameters
CONTEXT_LENGTH = 72
HORIZON = 18
MAX_SERIES = 10000  # None for all available series
SERIES_VAL_RATIO = 0.2
RANDOM_SEED = 42

# Architecture parameters
HIDDEN_DIM = 128
INTERMEDIATE_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 3
DROPOUT_RATE = 0.1

# Training / Inference parameters
BATCH_SIZE = 256
PREDICT_BATCH_SIZE = 512
EPOCHS = 50
LEARNING_RATE = 3e-4
AUTOREGRESSIVE_PATIENCE = 5

# File paths
TRAIN_CSV_PATH = "./data/Monthly-train.csv"

# Model naming and saving
import os

def format_float_for_name(value):
    return f"{value:g}".replace(".", "p").replace("-", "m")

DATASET_TAG = "all" if MAX_SERIES is None else str(MAX_SERIES)

RUN_NAME = (
    f"transformer"
    f"_ctx{CONTEXT_LENGTH}"
    f"_h{HIDDEN_DIM}"
    f"_ff{INTERMEDIATE_DIM}"
    f"_heads{NUM_HEADS}"
    f"_L{NUM_LAYERS}"
    f"_drop{format_float_for_name(DROPOUT_RATE)}"
    f"_lr{format_float_for_name(LEARNING_RATE)}"
    f"_bs{BATCH_SIZE}"
    f"_n{DATASET_TAG}"
    f"_seqsplit{int((1 - SERIES_VAL_RATIO) * 100)}_{int(SERIES_VAL_RATIO * 100)}"
)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(MODEL_DIR, RUN_NAME + "_best.keras")