# Log-ratio Transformer Experiments

## Protocol

These runs use the same best Transformer configuration as the nominal-value model, but replace the input representation with log-ratios:

```text
r[t] = log(x[t] / x[t-1])
x_hat[t+1] = x_hat[t] * exp(r_hat[t+1])
```

Fixed setup:

- `context_length=60`
- `hidden_dim=128`
- `intermediate_dim=256`
- `num_heads=4`
- `num_layers=2`
- `dropout_rate=0.1`
- `learning_rate=2e-4`
- `horizon=18`

---

## Validation Results

| Exp. | Dataset | Naive MAE | Naive RMSE | Naive sMAPE | Seasonal sMAPE | Log-ratio Transformer MAE | Log-ratio Transformer RMSE | Log-ratio Transformer sMAPE | Notes |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| LR1 | 10k validation | 882.2095 | 1811.2302 | 16.5759 | 20.9153 | 863.5809 | 1623.7988 | 16.5489 | slightly better than Naive |
| LR2 | full validation | 592.0305 | 1808.7012 | 12.4940 | 14.4777 | 513.0421 | 1588.0835 | 10.7706 | best validation result |

---

## Official Test Evaluation

Evaluation on `Monthly-test.csv` using the log-ratio model trained on the full dataset.

| Model | MAE | MSE | RMSE | sMAPE (%) |
| :--- | ---: | ---: | ---: | ---: |
| Naive | 670.0170 | 2490985.5000 | 1578.2856 | 15.2344 |
| Seasonal Naive | 699.4835 | 2649182.2500 | 1627.6309 | 15.9716 |
| Log-ratio Transformer | 553.9288 | 1791634.7500 | 1338.5197 | 12.9825 |

Official test improvement over Naive:

| Metric | Naive | Log-ratio Transformer | Difference |
| :--- | ---: | ---: | ---: |
| MAE | 670.0170 | 553.9288 | -116.0882 |
| RMSE | 1578.2856 | 1338.5197 | -239.7659 |
| sMAPE | 15.2344 | 12.9825 | -2.2519 |

Relative sMAPE improvement over Naive: approximately `14.78%`.

---

## Takeaways

- On 10k validation, log-ratio gives only a small sMAPE improvement over Naive.
- On full validation, log-ratio is much stronger than both baselines.
- On the official test set, the log-ratio Transformer clearly outperforms both Naive and Seasonal Naive.
- The log-ratio representation is currently the best-performing variant.
