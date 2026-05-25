# Log-ratio Transformer Experiments

## Protocol

This experiment keeps the best current Transformer configuration unchanged and only changes the data representation.

Fixed setup:

- `context_length=60`
- `hidden_dim=128`
- `intermediate_dim=256`
- `num_heads=4`
- `num_layers=2`
- `dropout_rate=0.1`
- `learning_rate=2e-4`
- autoregressive validation with `horizon=18`

Representation:

```text
r[t] = log(x[t] / x[t-1])
x_hat[t+1] = x_hat[t] * exp(r_hat[t+1])
```

The goal is to isolate the effect of replacing nominal values with scale-free relative changes.

## Runs

| Exp. | Max Series | Context | Hidden | FF Dim | Layers | LR | Dropout | Train Series | Val Series | Naive sMAPE | Transformer sMAPE | Notes |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| LR1 | 10000 | 60 | 128 | 256 | 2 | 2e-4 | 0.1 | | | | | sanity check |
| LR2 | all | 60 | 128 | 256 | 2 | 2e-4 | 0.1 | | | | | official full run if LR1 is stable |
