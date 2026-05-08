# Experiments Log - Transformer Model (M4 Dataset)

*Note: Baseline metrics (Naive / Seasonal Naive) vary naturally depending on the size and composition of the specific subset used for evaluation.*

## Stage 1: Context Window Optimization (context_length)
*Fixed Architecture:* `hidden_dim=64`, `num_layers=3`, `num_heads=4`, `train_data=5000 series`

| Experiment | context_length | MAE | RMSE | sMAPE (%) | Model File | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp 1** | **48** | **524.69** | **1352.66** | **10.11** | `transformer_ctx48_best.keras` | **Best context.** Beats baselines (11.69% / 11.22%). |
| **Exp 2** | 72 | 556.89 | 1397.98 | 10.49 | `transformer_ctx72_best.keras` | Performance degrades. Still beats baselines. |
| **Exp 3** | 96 | 592.39 | 1503.89 | 11.16 | `transformer_ctx96_best.keras` | Degrades further. Model capacity too small for 8-year history. |

---

## Stage 2: Increasing Model Capacity (Overfitting Check)
*Fixed Data:* `context_length=48`, `train_data=5000 series`

| Experiment | Architecture | MAE | RMSE | sMAPE (%) | Model File | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp 4** | `h=128`, `inter=256`, `lr=1e-3` | 691.40 | 1606.96 | 12.71 | `transformer_ctx48_h128_L3.keras` | Severe overfitting. Model is too large for 5000 series. |
| **Exp 5** | `h=128`, `inter=256`, `lr=3e-4` | 561.98 | 1407.50 | 10.66 | `transformer_ctx48_h128_lr3e4.keras` | Lower LR acts as regularizer, but still worse than Exp 1. |

---

## Stage 3: Dataset Scaling (10,000 Series)
*Fixed Context:* `context_length=48`. Models are evaluated on two distinct sets to measure true generalization vs Stage 1 & 2.
*10k Baselines: Naive sMAPE = 14.28% | S.Naive sMAPE = 15.47%*

| Experiment | Architecture | Eval Set | MAE | RMSE | sMAPE (%) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp 6** | Small (`h=64`) | 10k series | 639.98 | 1414.12 | 13.77 | Baseline on the new, harder 10k distribution. |
| | | **5k series** | 561.09 | 1430.77 | **10.55** | Static benchmark. Worse than Exp 1 (10.11%). |
| **Exp 7** | Large (`h=128`) | 10k series | **626.07** | **1371.05** | **13.60** | **Wins on the 10k dataset.** Scaling law confirmed. |
| | | **5k series** | **542.35** | **1373.65** | **10.32** | Massive improvement vs Exp 4 (12.71%). Extra data cured the overfitting. |