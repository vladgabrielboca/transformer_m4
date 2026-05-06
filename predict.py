import numpy as np
from tqdm import tqdm
from tensorflow import keras
from src.preprocessing import M4TransformerPreprocessor

from src.model.embedding import PositionalEmbedding
from src.model.layers import DecoderBlock

# --- BASELINE ---

def compute_metrics(y_true, y_pred):
    eps = 1e-8

    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    smape = 200 * np.mean(
        np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)
    )

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "sMAPE": smape,
    }


def naive_forecast_from_validation(X_val, val_stats, horizon=18):
    forecasts = []

    for i in range(len(X_val)):
        mean, std = val_stats[i]
        history_raw = X_val[i, :, 0] * std + mean

        last_value = history_raw[-1]
        forecast = np.repeat(last_value, horizon)

        forecasts.append(forecast)

    return np.asarray(forecasts, dtype=np.float32)


def seasonal_naive_forecast_from_validation(X_val, val_stats, horizon=18, seasonality=12):
    forecasts = []

    for i in range(len(X_val)):
        mean, std = val_stats[i]
        history_raw = X_val[i, :, 0] * std + mean

        if len(history_raw) < seasonality:
            forecast = np.repeat(history_raw[-1], horizon)
        else:
            seasonal_pattern = history_raw[-seasonality:]
            forecast = np.tile(seasonal_pattern, int(np.ceil(horizon / seasonality)))[:horizon]

        forecasts.append(forecast)

    return np.asarray(forecasts, dtype=np.float32)

# ---  MODEL PREDICTION ---

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

preprocessor.dataset = dataset[:5000]

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

for i in tqdm(range(len(X_val)), desc="Predicting"):
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

naive_preds = naive_forecast_from_validation(
    X_val,
    preprocessor.val_stats,
    horizon=18
)

seasonal_naive_preds = seasonal_naive_forecast_from_validation(
    X_val,
    preprocessor.val_stats,
    horizon=18,
    seasonality=12
)

print("Naive:", compute_metrics(y_val, naive_preds))
print("Seasonal naive:", compute_metrics(y_val, seasonal_naive_preds))
print("Transformer:", compute_metrics(y_val, all_forecasts))