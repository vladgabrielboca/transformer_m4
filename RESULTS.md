# Transformer Experiments Log (M4 Dataset)

The corrected protocol is now:

- split the selected dataset at the **series level** using an 80/20 split,
- train on the 80% subset,
- select checkpoints using **autoregressive validation sMAPE** on the held-out series,
- evaluate Naive, Seasonal Naive, and Transformer on the **same held-out 20% subset only**.

---

## Clean Holdout Results (Current Reference)

### Experiment C1 — 10,000 series, clean 80/20 split

**Configuration**

- `context_length=48`
- `hidden_dim=64`
- `num_layers=3`
- `num_heads=4`
- `intermediate_dim=128`
- `learning_rate=1e-3`
- dataset subset: `10000` series
- evaluation: holdout-only (`~20%`, 1989 unseen series)

**Training summary**

- Train series: `7958`
- Holdout series: `1989`
- Best checkpoint selected by `val_smape_ar = 15.6467`

**Holdout metrics**

| Model | MAE | RMSE | sMAPE (%) |
| :--- | ---: | ---: | ---: |
| Naive | 613.42 | 1238.06 | 13.99 |
| Seasonal Naive | 686.81 | 1378.07 | 15.51 |
| Transformer | 691.42 | 1361.03 | 15.65 |

### Experiment C2 — 10,000 series, clean 80/20 split

**Configuration**

- `context_length=48`
- `hidden_dim=128`
- `num_layers=3`
- `num_heads=4`
- `intermediate_dim=256`
- `learning_rate=3e-4`
- dataset subset: `10000` series
- evaluation: holdout-only (`~20%`, 1989 unseen series)

**Training summary**

- Train series: `7958`
- Holdout series: `1989`
- Best checkpoint selected by `val_smape_ar = 13.5043`

**Holdout metrics**

| Model | MAE | RMSE | sMAPE (%) |
| :--- | ---: | ---: | ---: |
| Naive | 613.42 | 1238.06 | 13.99 |
| Seasonal Naive | 686.81 | 1378.07 | 15.51 |
| Transformer | 584.77 | 1171.38 | 13.50 |

---

## Experiment C3 — Larger Transformer with longer context, 10,000 series, clean 80/20 split

This experiment increased the input context from 48 months to 60 months.

Because a longer context requires longer time series, fewer series were eligible for this experiment.

### Configuration

- `context_length=60`
- `hidden_dim=128`
- `intermediate_dim=256`
- `num_heads=4`
- `num_layers=3`
- `learning_rate=3e-4`
- `MAX_SERIES=10000`
- `SERIES_VAL_RATIO=0.2`
- checkpoint selected by autoregressive validation sMAPE

### Data split

- Train series: `5937`
- Validation/Test series: `1484`
- Total eligible series: `7421`
- Evaluation: holdout-only

### Training data shapes

- `X_train`: `(1275624, 60, 1)`
- `y_train`: `(1275624, 60, 1)`
- `X_val_fit`: `(320105, 60, 1)`
- `y_val_fit`: `(320105, 60, 1)`
- `X_val_ar`: `(1484, 60, 1)`
- `y_val_ar`: `(1484, 18)`

### Holdout metrics

| Model | MAE | MSE | RMSE | sMAPE (%) |
| :--- | ---: | ---: | ---: | ---: |
| Naive | 641.4473 | 2171092.7500 | 1473.4629 | 12.8644 |
| Seasonal Naive | 658.0632 | 2657375.5000 | 1630.1459 | 12.8626 |
| Transformer | 548.2968 | 1673284.7500 | 1293.5551 | 10.5715 |

### Conclusion

This is the best clean result so far.

The Transformer strongly outperforms both baselines on the clean held-out series:

- Naive sMAPE: `12.8644`
- Seasonal Naive sMAPE: `12.8626`
- Transformer sMAPE: `10.5715`

However, this result is not directly comparable to Experiment C2 because the longer context length reduces the number of eligible series. The validation set is different.

To isolate the effect of context length, a controlled comparison was run using the same eligibility threshold for both `context_length=48` and `context_length=60`.

---

## Controlled Context-Length Comparison

To compare `context_length=48` and `context_length=60` fairly, both models were evaluated on the same subset of series eligible for `context_length=60`.

This was done using:

- `ELIGIBILITY_CONTEXT_LENGTH=60`
- same `MAX_SERIES=10000`
- same `SERIES_VAL_RATIO=0.2`
- same random seed
- same train/validation split
- same validation/test series

This ensures that the comparison is not affected by different series being filtered out due to length constraints.

---

## Experiment C4 — Context 48 evaluated on the context-60 eligible subset

### Configuration

- `context_length=48`
- `ELIGIBILITY_CONTEXT_LENGTH=60`
- `hidden_dim=128`
- `intermediate_dim=256`
- `num_heads=4`
- `num_layers=3`
- `learning_rate=3e-4`
- `MAX_SERIES=10000`
- `SERIES_VAL_RATIO=0.2`

### Data split

- Train series: `5937`
- Validation/Test series: `1484`
- Total eligible series: `7421`
- Evaluation: holdout-only

### Training data shapes

- `X_train`: `(1346868, 48, 1)`
- `y_train`: `(1346868, 48, 1)`
- `X_val_fit`: `(337913, 48, 1)`
- `y_val_fit`: `(337913, 48, 1)`
- `X_val_ar`: `(1484, 48, 1)`
- `y_val_ar`: `(1484, 18)`

### Holdout metrics

| Model | MAE | MSE | RMSE | sMAPE (%) |
| :--- | ---: | ---: | ---: | ---: |
| Naive | 641.4473 | 2171092.7500 | 1473.4629 | 12.8644 |
| Seasonal Naive | 658.0632 | 2657375.5000 | 1630.1459 | 12.8626 |
| Transformer | 570.3654 | 1703137.3750 | 1305.0431 | 11.0152 |

---

## Fair Comparison: Context 48 vs Context 60

Both experiments below use the same eligible series subset and the same holdout split.

| Experiment | Model Context | Eligibility Context | Holdout Series | Transformer MAE | Transformer RMSE | Transformer sMAPE (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| C4 | 48 | 60 | 1484 | 570.3654 | 1305.0431 | 11.0152 |
| C3 | 60 | 60 | 1484 | 548.2968 | 1293.5551 | 10.5715 |

### Interpretation

Increasing the input context from 48 to 60 months improved Transformer sMAPE from:

- `11.0152` with `context_length=48`
- to `10.5715` with `context_length=60`

Absolute improvement:

- `0.4437` sMAPE points

Relative improvement:

- approximately `4.03%`

This is a meaningful improvement, especially because the baselines are identical across both runs. That confirms the validation set did not change.

The model with `context_length=48` also had more training windows:

- Context 48: `1,346,868` training windows
- Context 60: `1,275,624` training windows

Despite having fewer training windows, the context-60 model performed better. This suggests that the additional historical context is useful, likely because it helps the model capture seasonal and longer-term temporal patterns.

---

## Current Best Clean Result

The current best methodologically reliable result is:

| Setting | Value |
| :--- | :--- |
| `context_length` | `60` |
| `hidden_dim` | `128` |
| `intermediate_dim` | `256` |
| `num_heads` | `4` |
| `num_layers` | `3` |
| `learning_rate` | `3e-4` |
| `MAX_SERIES` | `10000` |
| `SERIES_VAL_RATIO` | `0.2` |
| Evaluation | clean series-level holdout |
| Holdout series | `1484` |
| Transformer sMAPE | `10.5715` |

---
