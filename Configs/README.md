# Training configuration reference

Run `elman train <configuration.json>` to start training. The smoke configurations default to CUDA; `--backend cpu` selects the CPU implementation. An unavailable requested backend is an error.

| Field | Meaning |
| --- | --- |
| `format`, `version` | Required `elman.training` and integer `1`. |
| `model_name` | Required exact name from the model definition. |
| `model` | Required path to a versioned model JSON file, relative to this configuration. |
| `backend` | `cpu` or `cuda`; defaults to `cpu`. |
| `data.format` | `image_csv` for classification or `token_csv` for language modeling. |
| `data.path` | Required CSV path, relative to this configuration. |
| `data.limit_samples` | Optional maximum rows to load; `0` means every row. |
| `data.sequence_length` | Optional GPT2 data length, no greater than the model context. This changes input data length only; model dimensions and parameter count remain unchanged. |
| `training.steps` | Required positive number of optimizer updates. |
| `training.batch_size` | Samples per update; defaults to `1`. Samples repeat if the batch is larger than the dataset. |
| `training.optimizer` | `adam`, `adamw` or `sgd`; defaults to `adam`. |
| `training.learning_rate` | Positive base learning rate; default `0.001`. |
| `training.beta1`, `training.beta2` | Adam moment coefficients in `[0,1)`; defaults `0.9`, `0.999`. |
| `training.epsilon` | Positive Adam denominator epsilon; default `1e-8`. |
| `training.momentum` | SGD momentum in `[0,1)`; default `0`. |
| `training.weight_decay` | Non-negative decay; default `0`. AdamW applies decoupled decay; Adam and SGD add L2 gradients. |
| `training.clip_grad_norm` | Maximum global gradient L2 norm; `0` disables clipping, default `1`. |
| `training.seed` | Non-negative integer seed; default `42`. |
| `training.shuffle` | Shuffle sample order at dataset boundaries; default `true`. |
| `training.log_every` | Positive number of steps between console reports; default `1`. |
| `training.checkpoint_every` | Periodic checkpoint interval; `0` writes only the final checkpoint. |
| `training.lr_schedule` | `constant` or `cosine`; default `constant`. |
| `training.warmup_steps` | Linear warmup length; default `0`. |
| `training.min_lr_ratio` | Cosine final/base learning-rate ratio in `[0,1]`; default `0.1`. |
| `training.require_loss_decrease` | Fail if final dropout-free evaluation loss is not below the initial evaluation; default `false`. |
| `output.checkpoint` | Binary checkpoint destination, relative to the configuration. |
| `output.metrics` | JSON metrics destination, relative to the configuration. |
| `output.graph` | Optional first training loss's computation graph JSON destination. |

CLI `--steps N`, `--backend DEVICE`, `--checkpoint FILE` and `--output FILE` override the corresponding run settings. `--resume FILE` restores weights, optimizer moments and the step counter, then runs the configured number of additional steps. CLI paths are relative to the current directory. Configuration paths remain relative to the configuration file, independent of the working directory.

`elman infer <configuration.json>` loads `output.checkpoint`, or the explicit `--checkpoint`, and uses the configuration's dataset. It outputs classification predictions for AlexNet and teacher-forced next-token predictions for GPT2. `--output FILE` saves the inference JSON. Inference does not use dropout or retain an automatic-differentiation graph.

The supplied configurations both use the full-size model JSON files, one sample,
batch size one and one optimizer update. AlexNet consumes a 224 x 224 RGB image;
GPT2 consumes eight token IDs and predicts the next eight IDs while retaining
all 124,439,808 model parameters. Dropout retains its model-defined probability.
Finite loss/gradients, weight updates and checkpoint/inference equivalence are
the acceptance criteria; one stochastic update is not required to lower loss.
These synthetic samples verify execution, not useful model accuracy.
