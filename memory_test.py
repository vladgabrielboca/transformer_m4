from config import TRAIN_CSV_PATH, CONTEXT_LENGTH, HORIZON
from src.preprocessing import M4TransformerPreprocessor

preprocessor = M4TransformerPreprocessor(
    context_length=CONTEXT_LENGTH,
    horizon=HORIZON
)

dataset = preprocessor.load_from_csv(TRAIN_CSV_PATH)

eligible = []
total_windows = 0

for series in dataset:
    if len(series) >= CONTEXT_LENGTH + HORIZON + 1:
        eligible.append(series)
        train_len = len(series) - HORIZON
        total_windows += train_len - CONTEXT_LENGTH

print("Total series:", len(dataset))
print("Eligible series:", len(eligible))
print("Estimated training windows:", total_windows)

bytes_per_xy = CONTEXT_LENGTH * 1 * 4 * 2
estimated_gb = total_windows * bytes_per_xy / (1024 ** 3)

print(f"Estimated raw X+y memory: {estimated_gb:.2f} GB")
print(f"Estimated practical peak memory: {estimated_gb * 3:.2f} GB to {estimated_gb * 5:.2f} GB")