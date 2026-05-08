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
| Transformer | 584.77 | 1171.38 | 13.5 |

---

## Legacy Results (Optimistic)

### Stage 1: Context Window Optimization (legacy)

*Original setup:* `hidden_dim=64`, `num_layers=3`, `num_heads=4`, `train_data=5000 series`

| Experiment | context_length | MAE | RMSE | sMAPE (%) | Model File | Legacy Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp 1** | **48** | **524.69** | **1352.66** | **10.11** | `transformer_ctx48_best.keras` | Best legacy result. |
| **Exp 2** | 72 | 556.89 | 1397.98 | 10.49 | `transformer_ctx72_best.keras` | Worse than Exp 1. |
| **Exp 3** | 96 | 592.39 | 1503.89 | 11.16 | `transformer_ctx96_best.keras` | Further degradation. |

### Stage 2: Increasing Model Capacity (legacy)

| Experiment | Architecture (dimensions) | MAE | RMSE | sMAPE (%) | Model File | Legacy Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp 4** | `hidden=128`, `inter=256`, `layers=3` | 691.40 | 1606.96 | 12.71 | `transformer_ctx48_h128_L3_best.keras` | Legacy result only. |
| **Exp 5** | `hidden=128`, `inter=256`, `layers=3`, `learning_rate=3e-4` | 561.98 | 1407.50 | 10.66 | `transformer_ctx48_h128_L3_best.keras` | Legacy result only. |

### Stage 3: Dataset Scaling (legacy)

*Original note:* `context_length=48`

| Experiment | Architecture | Eval Set | MAE | RMSE | sMAPE (%) | Legacy Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp 6** | Small (`h=64`) | 10k series | 639.98 | 1414.12 | 13.77 | Legacy full-subset evaluation, not clean holdout-only. |
|  |  | 5k series | 561.09 | 1430.77 | 10.55 | Legacy full-subset evaluation, not clean holdout-only. |
| **Exp 7** | Large (`h=128`) | 10k series | **626.07** | **1371.05** | **13.60** | Legacy full-subset evaluation, not clean holdout-only. |
|  |  | 5k series | **542.35** | **1373.65** | **10.32** | Legacy full-subset evaluation, not clean holdout-only. |

---
