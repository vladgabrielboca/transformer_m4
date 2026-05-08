# Transformer Experiments Log (M4 Dataset)

## Important Methodology Update

This file was revised after identifying that the earlier evaluation protocol was overly optimistic.

The original experimental pipeline had two methodological issues:

1. **Training-time validation was not fully clean.** The `fit()` validation split was created after concatenating all sliding windows, so train and validation windows could come from the same time series.
2. **Final evaluation in `predict.py` was not holdout-only.** The script evaluated on all `MAX_SERIES` series, which means reported metrics could include series that were also used during training.

Because of this, the historical results recorded below should be treated as **exploratory / optimistic** and **not as the true generalization performance on unseen series**.

The corrected protocol is now:

- split the selected dataset at the **series level** using an 80/20 split,
- train on the 80% subset,
- select checkpoints using **autoregressive validation sMAPE** on the held-out series,
- evaluate Naive, Seasonal Naive, and Transformer on the **same held-out 20% subset only**.

---

## Clean Holdout Results (Current Reference)

These are the only results that should currently be considered methodologically reliable.

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

**Conclusion:** under the corrected evaluation protocol, this configuration does **not** beat the Naive baseline on unseen-series holdout data.

---

## Legacy Results (Optimistic / Not Directly Comparable)

The experiments below are preserved for historical context, but they should **not** be interpreted as true holdout generalization results.

### Stage 1: Context Window Optimization (legacy)

*Original setup:* `hidden_dim=64`, `num_layers=3`, `num_heads=4`, `train_data=5000 series`

| Experiment | context_length | MAE | RMSE | sMAPE (%) | Model File | Legacy Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp 1** | **48** | **524.69** | **1352.66** | **10.11** | `transformer_ctx48_best.keras` | Best legacy result, but evaluated with the old optimistic pipeline. |
| **Exp 2** | 72 | 556.89 | 1397.98 | 10.49 | `transformer_ctx72_best.keras` | Worse than Exp 1 under the old pipeline. |
| **Exp 3** | 96 | 592.39 | 1503.89 | 11.16 | `transformer_ctx96_best.keras` | Further degradation under the old pipeline. |

### Stage 2: Increasing Model Capacity (legacy)

| Experiment | Architecture (dimensions) | MAE | RMSE | sMAPE (%) | Model File | Legacy Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp 4** | `hidden=128`, `inter=256`, `layers=3` | 691.40 | 1606.96 | 12.71 | `transformer_ctx48_h128_L3_best.keras` | Legacy result only. |
| **Exp 5** | `hidden=128`, `inter=256`, `layers=3`, `learning_rate=3e-4` | 561.98 | 1407.50 | 10.66 | `transformer_ctx48_h128_L3_best.keras` | Legacy result only. |

### Stage 3: Dataset Scaling (legacy)

*Original note:* `context_length=48`, evaluated under the old protocol.

| Experiment | Architecture | Eval Set | MAE | RMSE | sMAPE (%) | Legacy Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp 6** | Small (`h=64`) | 10k series | 639.98 | 1414.12 | 13.77 | Legacy full-subset evaluation, not clean holdout-only. |
|  |  | 5k series | 561.09 | 1430.77 | 10.55 | Legacy full-subset evaluation, not clean holdout-only. |
| **Exp 7** | Large (`h=128`) | 10k series | **626.07** | **1371.05** | **13.60** | Legacy full-subset evaluation, not clean holdout-only. |
|  |  | 5k series | **542.35** | **1373.65** | **10.32** | Legacy full-subset evaluation, not clean holdout-only. |

---
