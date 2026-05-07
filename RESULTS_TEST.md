# Experiments Log - Transformer Model (M4 Dataset)

## Stage 1: Context Window Optimization (context_length)
*Fixed Architecture:* `hidden_dim=64`, `num_layers=3`, `num_heads=4`, `dataset=5000 series`

| Experiment | context_length | MAE | RMSE | sMAPE (%) | Model File | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp 1** | 48 | 524.69 | 1352.66 | 10.11 | `transformer_ctx48_best.keras` | Beats baselines (Naive: 11.69%, S.Naive: 11.22%). |
| **Exp 2** | 72 | 556.89 | 1397.98 | 10.49 | `transformer_ctx72_best.keras` | Beats baselines (Naive: 11.64%, S.Naive: 11.11%). Worse than Exp 1. |
| **Exp 3** | 96 | - | - | - | `transformer_ctx96_best.keras` | To be run |

---

## Stage 2: Increasing Model Capacity
*(TO DO)*

| Experiment | Architecture (dimensions) | MAE | RMSE | sMAPE (%) | Model File | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp 4** | `hidden=128`, `inter=256`, `layers=3` | - | - | - | `transformer_cap1_best.keras` | - |
| **Exp 5** | `hidden=128`, `inter=256`, `layers=4` | - | - | - | `transformer_cap2_best.keras` | - |