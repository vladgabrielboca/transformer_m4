# Data parameters
CONTEXT_LENGTH = 96
HORIZON = 18
MAX_SERIES = 5000  # None for all available series

# Architecture parameters
HIDDEN_DIM = 64
INTERMEDIATE_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 3
DROPOUT_RATE = 0.1

# Training / Inference parameters
BATCH_SIZE = 128
PREDICT_BATCH_SIZE = 512
EPOCHS = 50
LEARNING_RATE = 1e-3

# File paths
TRAIN_CSV_PATH = "./data/Monthly-train.csv"
MODEL_SAVE_PATH = "transformer_ctx96_best.keras"