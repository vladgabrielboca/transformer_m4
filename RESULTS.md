# Transformer Experiments Log (M4 Monthly Dataset)

## Protocol

Current setup:

- dataset: `Monthly-train.csv`
- split: sequential 80/20 at series level
- no shuffle before splitting series
- context windows are created after the split
- training windows are shuffled
- checkpoint selection: best autoregressive validation sMAPE (`val_smape_ar`)
- `Monthly-test.csv` is kept for final evaluation only

Legacy shuffled-series results were moved to `RESULTS_old.md`.

---

## Experiments

All experiments use:

- `MAX_SERIES = 10000`
- `num_heads = 4`
- sequential 80/20 split
- checkpoint selected by best `val_smape_ar`

| Exp. | Context | Hidden | FF Dim | Layers | LR | Dropout | Train Series | Val Series | Naive sMAPE | Seasonal sMAPE | Transformer MAE | Transformer RMSE | Transformer sMAPE | Notes |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| E1 | 48 | 128 | 256 | 3 | 3e-4 | 0.1 | 7958 | 1989 | 19.1662 | 23.3968 | 749.6580 | 1331.3717 | 20.7814 | worse than Naive |
| E2 | 60 | 128 | 256 | 3 | 3e-4 | 0.1 | 5937 | 1484 | 16.5884 | 20.9129 | 871.9254 | 1700.4565 | 16.6114 | close to Naive |
| E3 | 72 | 128 | 256 | 3 | 3e-4 | 0.1 | 5892 | 1472 | 16.5570 | 20.8926 | 877.9478 | 1709.2360 | 16.7807 | improves MAE/RMSE, not sMAPE |
| E4 | 60 | 128 | 256 | 1 | 3e-4 | 0.1 | 5937 | 1484 | 16.5884 | 20.9129 | 902.3423 | 1709.8749 | 17.2824 | underperforms |
| E5 | 60 | 128 | 256 | 2 | 3e-4 | 0.1 | 5937 | 1484 | 16.5884 | 20.9129 | 858.2103 | 1656.9387 | 16.2540 | best current |
| E6 | 60 | 128 | 256 | 2 | 2e-4 | 0.1 | 5937 | 1484 | 16.5884 | 20.9129 | 904.9308 | 1771.8474 | 16.5562 | lower LR did not help |
| E7 | 60 | 128 | 256 | 2 | 3e-4 | 0.2 | 5937 | 1484 | 16.5884 | 20.9129 | 875.0839 | 1689.6096 | 16.3621 | higher dropout did not help |
| E8 | 60 | 128 | 512 | 2 | 3e-4 | 0.1 | 5937 | 1484 | 16.5884 | 20.9129 | 888.9582 | 1719.3309 | 16.8623 | larger FF dim did not help |

---

## Best Current Configuration

| Setting | Value |
| :--- | :--- |
| `context_length` | `60` |
| `hidden_dim` | `128` |
| `intermediate_dim` | `256` |
| `num_heads` | `4` |
| `num_layers` | `2` |
| `learning_rate` | `3e-4` |
| `dropout_rate` | `0.1` |
| `MAX_SERIES` | `10000` |
| Train series | `5937` |
| Validation series | `1484` |
| Transformer sMAPE | `16.2540` |

Improvement over Naive on the same validation set:

| Metric | Naive | Transformer | Difference |
| :--- | ---: | ---: | ---: |
| MAE | 873.7679 | 858.2103 | -15.5576 |
| RMSE | 1791.7058 | 1656.9387 | -134.7671 |
| sMAPE | 16.5884 | 16.2540 | -0.3344 |

Relative sMAPE improvement over Naive: approximately `2.02%`.

---

## Current Takeaways

- The best setup so far is `context_length=60`, `num_layers=2`, `learning_rate=3e-4`, `dropout_rate=0.1`.
- Reducing the learning rate to `2e-4` did not improve the result.
- Increasing dropout to `0.2` did not improve the result.
- Increasing the feed-forward dimension to `512` did not improve the result.
- The next useful step is to evaluate the best current configuration on the full dataset.

---

## Next Step

Run the best configuration on all available training series:

| Setting | Value |
| :--- | :--- |
| `MAX_SERIES` | `None` or `48000` |
| `context_length` | `60` |
| `hidden_dim` | `128` |
| `intermediate_dim` | `256` |
| `num_heads` | `4` |
| `num_layers` | `2` |
| `learning_rate` | `3e-4` |
| `dropout_rate` | `0.1` |

After this run, use `Monthly-test.csv` only for the final evaluation.
