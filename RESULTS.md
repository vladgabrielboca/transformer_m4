# Experiments Log - Transformer Model (M4 Dataset)

## Stage 1: Context Window Optimization (context_length)
*Fixed Architecture:* `hidden_dim=64`, `num_layers=3`, `num_heads=4`, `dataset=5000 series`

| Experiment | context_length | MAE | RMSE | sMAPE (%) | Model File | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp 1** | 48 | 524.69 | 1352.66 | 10.11 | `transformer_ctx48_best.keras` | Beats baselines (Naive: 11.69%, S.Naive: 11.22%). |
| **Exp 2** | 72 | 556.89 | 1397.98 | 10.49 | `transformer_ctx72_best.keras` | Beats baselines (Naive: 11.64%, S.Naive: 11.11%). Worse than Exp 1. |
| **Exp 3** | 96 | 592.39 | 1503.89 | 11.16 | `transformer_ctx96_best.keras` | Degrades further. MAE is worse than baselines. |

---

## Stage 2: Increasing Model Capacity

| Experiment | Architecture (dimensions) | MAE | RMSE | sMAPE (%) | Model File | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp 4** | `hidden=128`, `inter=256`, `layers=3` | 691.40 | 1606.96 | 12.71 | `transformer_ctx48_h128_L3_best.keras` | Even worse |
| **Exp 5** | `hidden=128`, `inter=256`, `layers=3`, `learning_rate=3e-4` | 561.98 | 1407.50 | 10.66 | `transformer_ctx48_h128_L3_best.keras` | Better, beats baseline, still worse than the first config: h64, context 48: sMAPE 10.1148; h128, context 48: sMAPE 10.6660 |