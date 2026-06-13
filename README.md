# Transformer M4 Monthly Forecasting

A compact decoder-only Transformer for univariate time series forecasting on the **M4 Monthly** dataset (48,000 monthly economic and financial series, 18-step horizon).

The project compares two input representations under an identical architecture and identical hyperparameters:

- **Nominal** — each series is normalized per-series (mean/std) and the model predicts the next normalized value at every step.
- **Log-ratio** — the series is converted to log-ratios `r[t] = log(x[t] / x[t-1])`; the model predicts future log-ratios, which are then reconstructed back to raw values with `x_hat[t+1] = x_hat[t] * exp(r_hat[t+1])`.

The log-ratio representation is the best-performing variant. Since both models share the same architecture and hyperparameters, the performance gap comes entirely from how the input is represented.

## Results

Official evaluation on `Monthly-test.csv` (sMAPE is the official M4 metric):

| Model | MAE | RMSE | sMAPE | Improvement over Naive |
| :--- | ---: | ---: | ---: | ---: |
| Naive | 670.87 | 1580.89 | 15.25% | — |
| Seasonal Naive | 700.02 | 1628.71 | 15.98% | — |
| Transformer (nominal) | 647.07 | 1466.14 | 14.36% | 5.86% |
| Transformer (log-ratio) | 553.93 | 1338.52 | 12.98% | 14.78% |

## Architecture

A decoder-only Transformer applied to a single channel of values per timestep:

- `PositionalEmbedding` — Dense projection of the scalar value plus a learned positional embedding.
- `N` × `DecoderBlock` — multi-head self-attention with a causal mask, followed by a feed-forward network, with residual connections and layer normalization.
- Final `LayerNormalization` and a `Dense(1)` head producing a next-step estimate at each position.

No encoder, no external covariates. The model has ~hundreds of thousands of parameters and trains comfortably on a single mid-range GPU.

## Project structure

```
transformer_m4/
├── config.py                          # Hyperparameters, dataset paths, run/model naming
├── train.py                           # Train the nominal model (checkpoint by autoregressive validation sMAPE)
├── predict.py                         # Evaluate the nominal model on the internal validation split
├── predict_official_test.py           # Evaluate the nominal model on Monthly-test.csv
├── src/
│   ├── preprocessing.py               # Per-series normalization and training window generation
│   └── model/
│       ├── embedding.py               # PositionalEmbedding
│       ├── layers.py                  # DecoderBlock
│       └── transformer.py             # Decoder-only model builder
└── experiments/
    └── logratio/                      # Log-ratio variant (best results)
        ├── config.py
        ├── preprocessing.py
        ├── train.py
        ├── predict_validation.py
        └── predict_official_test.py
```

## Requirements

- Python 3.11+
- TensorFlow / Keras
- NumPy, Pandas

```bash
pip install tensorflow numpy pandas
```

## Dataset

The dataset is **not** included in this repository due to its size. Download the M4 Monthly files from the official M4 competition repository:

https://github.com/Mcompetitions/M4-methods/tree/master/Dataset

Place both files in a `data/` folder at the project root:

```
data/
├── Monthly-train.csv
└── Monthly-test.csv
```

Paths are configured in `config.py` (`TRAIN_CSV_PATH`, `TEST_CSV_PATH`).

## Usage

All commands are run from the project root.

### Nominal model

```bash
python train.py                  # train, saving the best checkpoint to models/
python predict.py                # evaluate on the internal validation split
python predict_official_test.py  # evaluate on Monthly-test.csv
```

### Log-ratio model

The files under `experiments/logratio/` import modules from `src/` and from the root-level scripts, so they must be run as modules (`-m`), not by file path:

```bash
python -m experiments.logratio.train
python -m experiments.logratio.predict_validation
python -m experiments.logratio.predict_official_test
```

### Configuration

Experiment settings (context length, number of layers, learning rate, etc.) are edited in `config.py`. To run on a reduced subset of series, set `MAX_SERIES` to a value such as `10000`; for the full dataset, leave `MAX_SERIES = None`.

Default configuration (best setup):

| Setting | Value |
| :--- | :--- |
| `CONTEXT_LENGTH` | 60 |
| `HORIZON` | 18 |
| `HIDDEN_DIM` | 128 |
| `INTERMEDIATE_DIM` | 256 |
| `NUM_HEADS` | 4 |
| `NUM_LAYERS` | 2 |
| `DROPOUT_RATE` | 0.1 |
| `BATCH_SIZE` | 256 |
| `LEARNING_RATE` | 2e-4 |
| `AUTOREGRESSIVE_PATIENCE` | 5 |

## Training protocol

- Sequential 80/20 split at the whole-series level on `Monthly-train.csv`.
- Normalization (or log-ratio transform) is computed only on the training portion of each series to avoid information leakage.
- Checkpoint selection uses the autoregressive validation sMAPE (`val_smape_ar`), not the teacher-forced validation loss, with early stopping (`AUTOREGRESSIVE_PATIENCE`).
- Final evaluation on `Monthly-test.csv` is performed once, after the configuration is fully fixed.
