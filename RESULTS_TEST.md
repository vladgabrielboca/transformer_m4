# Experiments Log - Transformer Model (M4 Dataset)

*Fixed Architecture (Stage 1):* `hidden_dim=64`, `num_layers=3`, `num_heads=4`, `dataset=5000 series`

## Stage 1: Context Window Optimization (context_length)

| Experiment | Model | MAE | RMSE | sMAPE (%) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp 1**<br>*(Ctx 48)* | Naive | 582.05 | 1507.38 | 11.69 | |
| | Seasonal Naive | 582.12 | 1589.14 | 11.22 | |
| | **Transformer** | **524.69** | **1352.66** | **10.11** | Best context so far. |
| **Exp 2**<br>*(Ctx 72)* | Naive | 584.41 | 1517.64 | 11.64 | Validation subset changed due to minimum length constraint (72+18). |
| | Seasonal Naive | 582.82 | 1600.01 | 11.11 | |
| | **Transformer** | **556.89** | **1397.98** | **10.49** | Still beats baselines, but worse than Ctx 48. |
| **Exp 3**<br>*(Ctx 96)* | Naive | - | - | - | To be run |
| | Seasonal Naive | - | - | - | |
| | **Transformer** | - | - | - | |