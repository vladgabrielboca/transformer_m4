from config import *
import os

LOG_EPSILON = 1e-6
LOG_RATIO_CLIP = 5.0

RUN_NAME = RUN_NAME.replace("transformer_", "logratio_transformer_", 1)
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, RUN_NAME + "_best.keras")
