# Transformer Experiments Log (M4 Monthly Dataset)

## Protocol

Current setup:

- dataset for development: `Monthly-train.csv`
- split: sequential 80/20 at series level
- no shuffle before splitting series
- context windows are created after the split
- training windows are shuffled
- checkpoint selection: best autoregressive validation sMAPE (`val_smape_ar`)
- `Monthly-test.csv` is used only for final evaluation

Legacy shuffled-series results were moved to `RESULTS_old.md`.

---

## 10k Experiments

All experiments below use `MAX_SERIES=10000`, `num_heads=4`, sequential 80/20 split, and checkpoint selection by best `val_smape_ar`.

| Exp. | Context | Hidden | FF Dim | Layers | LR | Dropout | Train Series | Val Series | Naive sMAPE | Seasonal sMAPE | Transformer MAE | Transformer RMSE | Transformer sMAPE | Notes |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| E1 | 48 | 128 | 256 | 3 | 3e-4 | 0.1 | 7958 | 1989 | 19.1662 | 23.3968 | 749.6580 | 1331.3717 | 20.7814 | worse than Naive |
| E2 | 60 | 128 | 256 | 3 | 3e-4 | 0.1 | 5937 | 1484 | 16.5884 | 20.9129 | 871.9254 | 1700.4565 | 16.6114 | close to Naive |
| E3 | 72 | 128 | 256 | 3 | 3e-4 | 0.1 | 5892 | 1472 | 16.5570 | 20.8926 | 877.9478 | 1709.2360 | 16.7807 | improves MAE/RMSE, not sMAPE |
| E4 | 60 | 128 | 256 | 1 | 3e-4 | 0.1 | 5937 | 1484 | 16.5884 | 20.9129 | 902.3423 | 1709.8749 | 17.2824 | underperforms |
| E5 | 60 | 128 | 256 | 2 | 3e-4 | 0.1 | 5937 | 1484 | 16.5884 | 20.9129 | 858.2103 | 1656.9387 | 16.2540 | best 10k result |
| E6 | 60 | 128 | 256 | 2 | 2e-4 | 0.1 | 5937 | 1484 | 16.5884 | 20.9129 | 904.9308 | 1771.8474 | 16.5562 | lower LR did not help on 10k |
| E7 | 60 | 128 | 256 | 2 | 3e-4 | 0.2 | 5937 | 1484 | 16.5884 | 20.9129 | 875.0839 | 1689.6096 | 16.3621 | higher dropout did not help |
| E8 | 60 | 128 | 512 | 2 | 3e-4 | 0.1 | 5937 | 1484 | 16.5884 | 20.9129 | 888.9582 | 1719.3309 | 16.8623 | larger FF dim did not help |

---

## Full Dataset Internal Validation

These runs use all eligible series from `Monthly-train.csv` with the same sequential 80/20 validation protocol.

| Exp. | Max Series | Context | Hidden | FF Dim | Layers | LR | Dropout | Train Series | Val Series | Naive sMAPE | Seasonal sMAPE | Transformer MAE | Transformer RMSE | Transformer sMAPE | Notes |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| F1 | all | 60 | 128 | 256 | 2 | 3e-4 | 0.1 | 29536 | 7383 | 12.4682 | 14.4563 | 677.5230 | 1641.6759 | 13.6032 | worse than Naive sMAPE |
| F2 | all | 60 | 128 | 256 | 2 | 2e-4 | 0.1 | 29536 | 7383 | 12.4682 | 14.4563 | 609.6918 | 1665.3181 | 12.3223 | best internal validation result |

Best full-dataset internal validation result: **F2**, with `sMAPE=12.3223`, compared with Naive `sMAPE=12.4682`.

---

## Official Test Evaluation

Evaluation on `Monthly-test.csv`, using the model from the full-dataset `learning_rate=2e-4` run.

| Setting | Value |
| :--- | :--- |
| Used test series | `47981` |
| Skipped short-history series | `19` |
| Missing test rows | `0` |
| Context length | `60` |

| Model | MAE | MSE | RMSE | sMAPE (%) |
| :--- | ---: | ---: | ---: | ---: |
| Naive | 670.8651 | 2499220.5000 | 1580.8923 | 15.2488 |
| Seasonal Naive | 700.0150 | 2652699.7500 | 1628.7111 | 15.9836 |
| Transformer | 647.0684 | 2149554.7500 | 1466.1360 | 14.3551 |

Official test improvement over Naive:

| Metric | Naive | Transformer | Difference |
| :--- | ---: | ---: | ---: |
| MAE | 670.8651 | 647.0684 | -23.7967 |
| RMSE | 1580.8923 | 1466.1360 | -114.7563 |
| sMAPE | 15.2488 | 14.3551 | -0.8937 |

Relative sMAPE improvement over Naive: approximately `5.86%`.

---

## Current Best Configuration

| Setting | Value |
| :--- | :--- |
| `context_length` | `60` |
| `hidden_dim` | `128` |
| `intermediate_dim` | `256` |
| `num_heads` | `4` |
| `num_layers` | `2` |
| `learning_rate` | `2e-4` |
| `dropout_rate` | `0.1` |
| Full internal validation sMAPE | `12.3223` |
| Official test sMAPE | `14.3551` |

---

## Takeaways

- On the 10k subset, the best result used `context_length=60`, `num_layers=2`, and `learning_rate=3e-4`.
- On the full dataset, `learning_rate=2e-4` performed better than `3e-4`.
- The final Transformer model beats both Naive and Seasonal Naive on the official test evaluation.
- The official test result uses only series with at least `60` historical observations; `19` short-history series were skipped.
---

## Related Experiments

Log-ratio experiments are tracked separately in `experiments/logratio/RESULTS.md`.

