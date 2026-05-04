import numpy as np
from tensorflow import keras
from src.preprocessing import M4TransformerPreprocessor

from src.model.transformer import PositionalEmbedding
from src.model.layers import DecoderBlock

def denormalize_forecast(preds, mean, std):
    return preds * std + mean

def autoregressive_forecast(model, history_window, horizon):
    window = history_window.astype("float32").copy()
    preds = []

    for _ in range(horizon):
        model_input = np.expand_dims(window, axis=0)
        pred_seq = model.predict(model_input, verbose=0)

        next_value = pred_seq[0, -1, 0]
        preds.append(next_value)

        next_value_arr = np.array([[next_value]], dtype=np.float32)
        window = np.concatenate([window[1:], next_value_arr], axis=0)

    return np.array(preds, dtype=np.float32)


preprocessor = M4TransformerPreprocessor(context_length=48, horizon=18)
dataset = preprocessor.load_from_csv('./data/Monthly-train.csv')

X_val, y_val = preprocessor.get_validation_data()
stats = preprocessor.val_stats 

model = keras.models.load_model(
    "transformer_best_weights.keras",
    custom_objects={
        "PositionalEmbedding": PositionalEmbedding,
        "DecoderBlock": DecoderBlock
    }
)
  
all_forecasts = []
all_targets = []

for i in range(len(X_val)):
    preds_norm = autoregressive_forecast(model, X_val[i], horizon=18)

    mean, std = preprocessor.val_stats[i]
    preds = denormalize_forecast(preds_norm, mean, std)

    all_forecasts.append(preds)
    all_targets.append(y_val[i])

all_forecasts = np.asarray(all_forecasts)
all_targets = np.asarray(all_targets)

mse = np.mean((all_forecasts - all_targets) ** 2)
mae = np.mean(np.abs(all_forecasts - all_targets))

print("Validation MSE:", mse)
print("Validation MAE:", mae)