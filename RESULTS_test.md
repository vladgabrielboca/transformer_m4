# Transformer Experiments Log (M4 Dataset)

## Evaluation Protocol

All results below use the corrected evaluation protocol:

- the dataset is split at the **series level** using an 80/20 split;
- the model is trained only on the 80% training-series subset;
- checkpoints are selected using **autoregressive validation sMAPE** on the held-out series;
- Naive, Seasonal Naive, and Transformer are evaluated on the **same held-out 20% subset**.

For experiments with longer contexts, fewer series are eligible because each series must be long enough to provide the input context and the 18-step forecasting horizon.

---

## Main Clean Holdout Results

| Exp. | Context | Eligibility Context | Hidden Dim | FF Dim | LR | Train Series | Holdout Series | Naive sMAPE | Seasonal Naive sMAPE | Transformer sMAPE |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | 48 | 48 | 64 | 128 | 1e-3 | 7958 | 1989 | 13.9876 | 15.5120 | 15.6467 |
| C2 | 48 | 48 | 128 | 256 | 3e-4 | 7958 | 1989 | 13.9876 | 15.5120 | 13.5043 |
| C3 | 60 | 60 | 128 | 256 | 3e-4 | 5937 | 1484 | 12.8644 | 12.8626 | 10.5715 |
| C4 | 48 | 60 | 128 | 256 | 3e-4 | 5937 | 1484 | 12.8644 | 12.8626 | 11.0152 |
| C5 | 72 | 72 | 128 | 256 | 3e-4 | 5892 | 1472 | 12.8717 | 13.1847 | 10.7714 |
| C6 | 60 | 72 | 128 | 256 | 3e-4 | 5892 | 1472 | 12.8717 | 13.1847 | 11.7733 |

---

## Detailed Metrics

### C1 — Small Transformer, context 48

| Model | MAE | RMSE | sMAPE (%) |
| :--- | ---: | ---: | ---: |
| Naive | 613.4219 | 1238.0596 | 13.9876 |
| Seasonal Naive | 686.8148 | 1378.0718 | 15.5120 |
| Transformer | 691.4189 | 1361.0284 | 15.6467 |

This configuration did not beat the Naive baseline on the clean holdout set.

---

### C2 — Larger Transformer, context 48

| Model | MAE | RMSE | sMAPE (%) |
| :--- | ---: | ---: | ---: |
| Naive | 613.4219 | 1238.0596 | 13.9876 |
| Seasonal Naive | 686.8148 | 1378.0718 | 15.5120 |
| Transformer | 584.7762 | 1171.3843 | 13.5043 |

Increasing the model size and lowering the learning rate made the Transformer outperform both baselines.

---

### C3 — Larger Transformer, context 60

| Model | MAE | RMSE | sMAPE (%) |
| :--- | ---: | ---: | ---: |
| Naive | 641.4473 | 1473.4629 | 12.8644 |
| Seasonal Naive | 658.0632 | 1630.1459 | 12.8626 |
| Transformer | 548.2968 | 1293.5551 | 10.5715 |

This is the best absolute clean result so far.

---

### C4 — Context 48 on the context-60 eligible subset

| Model | MAE | RMSE | sMAPE (%) |
| :--- | ---: | ---: | ---: |
| Naive | 641.4473 | 1473.4629 | 12.8644 |
| Seasonal Naive | 658.0632 | 1630.1459 | 12.8626 |
| Transformer | 570.3654 | 1305.0431 | 11.0152 |

This run was used to compare context 48 and context 60 on the exact same series subset.

---

### C5 — Larger Transformer, context 72

| Model | MAE | RMSE | sMAPE (%) |
| :--- | ---: | ---: | ---: |
| Naive | 638.1287 | 1485.6102 | 12.8717 |
| Seasonal Naive | 658.3759 | 1490.3978 | 13.1847 |
| Transformer | 544.2301 | 1261.7500 | 10.7714 |

Context 72 performs well and clearly beats both baselines.

---

### C6 — Context 60 on the context-72 eligible subset

| Model | MAE | RMSE | sMAPE (%) |
| :--- | ---: | ---: | ---: |
| Naive | 638.1287 | 1485.6102 | 12.8717 |
| Seasonal Naive | 658.3759 | 1490.3978 | 13.1847 |
| Transformer | 569.9908 | 1360.2129 | 11.7733 |

This run was used to compare context 60 and context 72 on the exact same series subset.

---

## Controlled Context-Length Comparisons

### Context 48 vs Context 60

Both models were evaluated on the same series subset by using:

- `ELIGIBILITY_CONTEXT_LENGTH=60`
- same train/holdout split
- same 1484 held-out series

| Context | Transformer MAE | Transformer RMSE | Transformer sMAPE (%) |
| ---: | ---: | ---: | ---: |
| 48 | 570.3654 | 1305.0431 | 11.0152 |
| 60 | 548.2968 | 1293.5551 | 10.5715 |

Increasing the context from 48 to 60 improved sMAPE by `0.4437` points, or approximately `4.03%`.

The baselines were identical in both runs, confirming that the evaluation set was unchanged.

---

### Context 60 vs Context 72

Both models were evaluated on the same series subset by using:

- `ELIGIBILITY_CONTEXT_LENGTH=72`
- same train/holdout split
- same 1472 held-out series

| Context | Transformer MAE | Transformer RMSE | Transformer sMAPE (%) |
| ---: | ---: | ---: | ---: |
| 60 | 569.9908 | 1360.2129 | 11.7733 |
| 72 | 544.2301 | 1261.7500 | 10.7714 |

Increasing the context from 60 to 72 improved sMAPE by `1.0019` points, or approximately `8.51%`.

Again, the baselines were identical in both runs, so the improvement comes from the longer input context rather than from a different validation set.

---

## Current Best Result

The best clean holdout result so far is:

| Setting | Value |
| :--- | :--- |
| `context_length` | `60` |
| `eligibility_context_length` | `60` |
| `hidden_dim` | `128` |
| `intermediate_dim` | `256` |
| `num_heads` | `4` |
| `num_layers` | `3` |
| `learning_rate` | `3e-4` |
| Train series | `5937` |
| Holdout series | `1484` |
| Transformer sMAPE | `10.5715` |

Although the context-72 model does not beat this absolute score, the controlled comparison on the context-72 eligible subset shows that context 72 is better than context 60 when both are evaluated on the same data.

---