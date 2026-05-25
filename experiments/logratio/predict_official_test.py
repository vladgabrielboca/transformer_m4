from tensorflow import keras

from experiments.logratio.config import (
    CONTEXT_LENGTH,
    HORIZON,
    TRAIN_CSV_PATH,
    TEST_CSV_PATH,
    MODEL_SAVE_PATH,
    PREDICT_BATCH_SIZE,
    LOG_RATIO_CLIP,
    LOG_EPSILON
)

from experiments.logratio.preprocessing import M4LogRatioPreprocessor
from src.model.embedding import PositionalEmbedding
from src.model.layers import DecoderBlock
from predict import compute_metrics, print_metrics

from experiments.logratio.predict_validation import (
    naive_forecast_from_anchors,
    autoregressive_logratio_forecast_batch
)

from predict_official_test import seasonal_naive_forecast_from_validation


if __name__ == "__main__":
    preprocessor = M4LogRatioPreprocessor(
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        epsilon=LOG_EPSILON,
        log_ratio_clip=LOG_RATIO_CLIP
    )

    train_dict = preprocessor.load_from_csv_with_ids(TRAIN_CSV_PATH)
    test_dict = preprocessor.load_from_csv_with_ids(TEST_CSV_PATH)

    X_test, y_test = preprocessor.get_official_test_data(
        train_dict=train_dict,
        test_dict=test_dict
    )
    forecast_anchors = list(preprocessor.forecast_anchors)

    print("Experiment: log-ratio regression")
    print("Model context length:", CONTEXT_LENGTH)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    print(f"Loading model from {MODEL_SAVE_PATH}...")

    model = keras.models.load_model(
        MODEL_SAVE_PATH,
        custom_objects={
            "PositionalEmbedding": PositionalEmbedding,
            "DecoderBlock": DecoderBlock
        }
    )

    transformer_preds = autoregressive_logratio_forecast_batch(
        model=model,
        X_val=X_test,
        forecast_anchors=forecast_anchors,
        horizon=HORIZON,
        batch_size=PREDICT_BATCH_SIZE,
        log_ratio_clip=LOG_RATIO_CLIP
    )

    naive_preds = naive_forecast_from_anchors(
        forecast_anchors=forecast_anchors,
        horizon=HORIZON
    )

    # Seasonal naive still needs the original normalized-value helper signature,
    # so use the baseline implementation only for comparability if needed.
    seasonal_naive_preds = None

    metrics_naive = compute_metrics(y_test, naive_preds)
    metrics_transformer = compute_metrics(y_test, transformer_preds)

    print_metrics("Model: Naive", metrics_naive)
    print_metrics("Model: Log-ratio Transformer", metrics_transformer)

    if seasonal_naive_preds is not None:
        metrics_snaive = compute_metrics(y_test, seasonal_naive_preds)
        print_metrics("Model: Seasonal Naive", metrics_snaive)
