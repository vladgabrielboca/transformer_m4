import numpy as np
from tqdm import tqdm
from tensorflow import keras
from src.preprocessing import M4TransformerPreprocessor

from config import (
    CONTEXT_LENGTH, HORIZON, MAX_SERIES, BATCH_SIZE, 
    EPOCHS, LEARNING_RATE, TRAIN_CSV_PATH, MODEL_SAVE_PATH, PREDICT_BATCH_SIZE
)

from src.model.embedding import PositionalEmbedding
from src.model.layers import DecoderBlock

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

def print_metrics(model_name, metrics_dict):
    print(f"\n=== {model_name} ===")

    for metric_name, value in metrics_dict.items():
        print(f"{metric_name:<6}: {value:.4f}")

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

def autoregressive_forecast_batch(model, X_val, val_stats, horizon=18, batch_size=PREDICT_BATCH_SIZE):
    window = X_val.astype(np.float32).copy()
    preds_norm = []
    
    for step in tqdm(range(horizon), desc="Predicting"):
        pred = model.predict(window, batch_size=batch_size, verbose=0)
        
        next_values = pred[:, 0]
        preds_norm.append(next_values)

        next_values = next_values[:, None, None]
        window = np.concatenate([window[:, 1:, :], next_values], axis=1)
        
        print(f"Forecast step {step + 1}/{horizon} done")
        
    preds_norm = np.stack(preds_norm, axis=1)
    means = np.asarray([mean for mean, std in val_stats], dtype=np.float32)[:, None]
    stds = np.asarray([std for mean, std in val_stats], dtype=np.float32)[:, None]
    preds_raw = preds_norm * stds + means
    
    return preds_raw

if __name__ == "__main__":
    preprocessor = M4TransformerPreprocessor(context_length=CONTEXT_LENGTH, horizon=HORIZON)
    dataset = preprocessor.load_from_csv(TRAIN_CSV_PATH)
    
    if MAX_SERIES is not None:
        preprocessor.dataset = dataset[:MAX_SERIES]
    else:
        preprocessor.dataset = dataset

    X_val, y_val = preprocessor.get_validation_data()

    print(f"Model loading from {MODEL_SAVE_PATH}...")
    model = keras.models.load_model(
        MODEL_SAVE_PATH,
        custom_objects={
            "PositionalEmbedding": PositionalEmbedding,
            "DecoderBlock": DecoderBlock
        }
    )
    
    transformer_preds = autoregressive_forecast_batch(
        model=model,
        X_val=X_val,
        val_stats=preprocessor.val_stats,
        horizon=HORIZON,
        batch_size=PREDICT_BATCH_SIZE
    )

    naive_preds = naive_forecast_from_validation(X_val, preprocessor.val_stats, horizon=HORIZON)
    seasonal_naive_preds = seasonal_naive_forecast_from_validation(X_val, preprocessor.val_stats, horizon=HORIZON, seasonality=12)

    metrics_naive = compute_metrics(y_val, naive_preds)
    metrics_snaive = compute_metrics(y_val, seasonal_naive_preds)
    metrics_transformer = compute_metrics(y_val, transformer_preds)

    print_metrics("Model: Naive", metrics_naive)
    print_metrics("Model: Seasonal Naive", metrics_snaive)
    print_metrics("Model: Transformer", metrics_transformer)