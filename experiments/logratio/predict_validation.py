import numpy as np
from tqdm import tqdm
from tensorflow import keras

from experiments.logratio.config import (
    CONTEXT_LENGTH, HORIZON, MAX_SERIES, TRAIN_CSV_PATH,
    MODEL_SAVE_PATH, PREDICT_BATCH_SIZE, SERIES_VAL_RATIO,
    LOG_RATIO_CLIP, LOG_EPSILON
)

from experiments.logratio.preprocessing import M4LogRatioPreprocessor
from src.model.embedding import PositionalEmbedding
from src.model.layers import DecoderBlock
from predict import compute_metrics, print_metrics


def naive_forecast_from_anchors(forecast_anchors, horizon=18):
    anchors = np.asarray(forecast_anchors, dtype=np.float32)
    return np.repeat(anchors[:, None], horizon, axis=1)


def seasonal_naive_forecast_from_raw_series(val_series, context_length, horizon=18, seasonality=12):
    forecasts = []

    for series in val_series:
        series = np.asarray(series, dtype=np.float32)
        history = series[-(context_length + horizon + 1) : -horizon]

        if len(history) < seasonality:
            forecast = np.repeat(history[-1], horizon)
        else:
            seasonal_pattern = history[-seasonality:]
            forecast = np.tile(seasonal_pattern, int(np.ceil(horizon / seasonality)))[:horizon]

        forecasts.append(forecast)

    return np.asarray(forecasts, dtype=np.float32)


def autoregressive_logratio_forecast_batch(
    model,
    X_val,
    forecast_anchors,
    horizon=18,
    batch_size=PREDICT_BATCH_SIZE,
    log_ratio_clip=5.0
):
    window = X_val.astype(np.float32).copy()
    current_values = np.asarray(forecast_anchors, dtype=np.float32)
    preds_raw = []

    for step in tqdm(range(horizon), desc="Predicting"):
        pred_seq = model.predict(window, batch_size=batch_size, verbose=0)
        next_log_ratios = pred_seq[:, -1, 0]

        if log_ratio_clip is not None:
            next_log_ratios = np.clip(next_log_ratios, -log_ratio_clip, log_ratio_clip)

        current_values = current_values * np.exp(next_log_ratios)
        preds_raw.append(current_values.copy())

        next_values = next_log_ratios[:, None, None].astype(np.float32)
        window = np.concatenate([window[:, 1:, :], next_values], axis=1)
        print(f"Forecast step {step + 1}/{horizon} done")

    return np.stack(preds_raw, axis=1).astype(np.float32)


if __name__ == "__main__":
    preprocessor = M4LogRatioPreprocessor(
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        epsilon=LOG_EPSILON,
        log_ratio_clip=LOG_RATIO_CLIP
    )
    dataset = preprocessor.load_from_csv(TRAIN_CSV_PATH)

    if MAX_SERIES is not None:
        preprocessor.dataset = dataset[:MAX_SERIES]
    else:
        preprocessor.dataset = dataset

    train_series, val_series = preprocessor.split_dataset_by_series(
        val_ratio=SERIES_VAL_RATIO,
        dataset=preprocessor.dataset
    )

    print("Experiment: log-ratio regression")
    print(f"Model context length: {CONTEXT_LENGTH}")

    X_val, y_val = preprocessor.get_validation_data(dataset=val_series)
    forecast_anchors = list(preprocessor.forecast_anchors)

    print(f"Total series used: {len(preprocessor.dataset)}")
    print(f"Train series: {len(train_series)}")
    print(f"Validation/Test series evaluated: {len(val_series)}")
    print(f"X_val: {X_val.shape}")
    print(f"y_val: {y_val.shape}")

    print(f"Model loading from {MODEL_SAVE_PATH}...")
    model = keras.models.load_model(
        MODEL_SAVE_PATH,
        custom_objects={
            "PositionalEmbedding": PositionalEmbedding,
            "DecoderBlock": DecoderBlock
        }
    )

    transformer_preds = autoregressive_logratio_forecast_batch(
        model=model,
        X_val=X_val,
        forecast_anchors=forecast_anchors,
        horizon=HORIZON,
        batch_size=PREDICT_BATCH_SIZE,
        log_ratio_clip=LOG_RATIO_CLIP
    )

    naive_preds = naive_forecast_from_anchors(forecast_anchors, horizon=HORIZON)
    seasonal_naive_preds = seasonal_naive_forecast_from_raw_series(
        val_series=val_series,
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        seasonality=12
    )

    metrics_naive = compute_metrics(y_val, naive_preds)
    metrics_snaive = compute_metrics(y_val, seasonal_naive_preds)
    metrics_transformer = compute_metrics(y_val, transformer_preds)

    print_metrics("Model: Naive", metrics_naive)
    print_metrics("Model: Seasonal Naive", metrics_snaive)
    print_metrics("Model: Log-ratio Transformer", metrics_transformer)
