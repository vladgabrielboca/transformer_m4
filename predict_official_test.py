from tensorflow import keras

from config import (
    CONTEXT_LENGTH,
    HORIZON,
    TRAIN_CSV_PATH,
    TEST_CSV_PATH,
    MODEL_SAVE_PATH,
    PREDICT_BATCH_SIZE
)

from src.preprocessing import M4TransformerPreprocessor
from src.model.embedding import PositionalEmbedding
from src.model.layers import DecoderBlock

from predict import (
    compute_metrics,
    print_metrics,
    naive_forecast_from_validation,
    seasonal_naive_forecast_from_validation,
    autoregressive_forecast_batch
)

if __name__ == "__main__":
    preprocessor = M4TransformerPreprocessor(
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON
    )

    train_dict = preprocessor.load_from_csv_with_ids(TRAIN_CSV_PATH)
    test_dict = preprocessor.load_from_csv_with_ids(TEST_CSV_PATH)

    X_test, y_test = preprocessor.get_official_test_data(
        train_dict=train_dict,
        test_dict=test_dict
    )

    test_stats = list(preprocessor.val_stats)

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

    transformer_preds = autoregressive_forecast_batch(
        model=model,
        X_val=X_test,
        val_stats=test_stats,
        horizon=HORIZON,
        batch_size=PREDICT_BATCH_SIZE
    )

    naive_preds = naive_forecast_from_validation(
        X_val=X_test,
        val_stats=test_stats,
        horizon=HORIZON
    )

    seasonal_naive_preds = seasonal_naive_forecast_from_validation(
        X_val=X_test,
        val_stats=test_stats,
        horizon=HORIZON,
        seasonality=12
    )

    metrics_naive = compute_metrics(y_test, naive_preds)
    metrics_snaive = compute_metrics(y_test, seasonal_naive_preds)
    metrics_transformer = compute_metrics(y_test, transformer_preds)

    print_metrics("Model: Naive", metrics_naive)
    print_metrics("Model: Seasonal Naive", metrics_snaive)
    print_metrics("Model: Transformer", metrics_transformer)