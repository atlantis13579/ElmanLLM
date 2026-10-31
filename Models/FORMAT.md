# Architecture schema

Every definition is a JSON object with `format: "elman.model"`, integer `version: 1`, a string `name` and a string `architecture`. Parameters are initialized from the training seed and stored under stable names. Float32 weights are independent of backend choice.

## AlexNet

Set `architecture` to `alexnet` and provide:

| Field | Type and meaning |
| --- | --- |
| `input_shape` | Three positive integers `[channels, height, width]`. |
| `channels` | Five positive output-channel counts. |
| `kernels` | Five positive square-convolution kernel sizes. |
| `strides` | Five positive convolution strides. |
| `paddings` | Five non-negative symmetric convolution paddings. |
| `pool_kernel`, `pool_stride` | Positive max-pool settings; defaults `3`, `2`. Pooling follows convolutions 1, 2 and 5. |
| `hidden_features` | Two positive fully-connected hidden widths. |
| `num_classes` | Integer class count greater than one. |
| `dropout` | Probability in `[0,1)`, default `0.5`, before the first two linear layers. |

Every convolution and the first two linear layers are followed by ReLU. The final linear layer emits logits. Input shape is `[batch, channels, height, width]`; output shape is `[batch, num_classes]`. All convolutions have biases, unit dilation and one group. Parameter names are `features.conv1.weight`/`.bias` through `features.conv5`, then `classifier.fc1` through `classifier.fc3`.

## GPT2

Set `architecture` to `gpt2` and provide:

| Field | Type and meaning |
| --- | --- |
| `vocab_size` | Integer vocabulary size greater than one. |
| `context_length` | Positive maximum sequence length. |
| `embedding_dim` | Positive embedding width divisible by `num_heads`. |
| `num_heads` | Positive number of attention heads. |
| `num_layers` | Positive number of decoder blocks. |
| `mlp_ratio` | Positive feed-forward expansion factor, default `4`. |
| `dropout` | Probability in `[0,1)`, default `0.1`. |
| `layer_norm_epsilon` | Positive normalization epsilon, default `1e-5`. |

Input shape is `[batch, sequence]` with integer vocabulary IDs, and sequence length must not exceed the context. Output logits have shape `[batch, sequence, vocab_size]`. The model uses learned token and position embeddings, causal multi-head attention, pre-layer-normalization, residual connections, GELU and a tied token-embedding output projection. The parameter names follow `transformer.wte.weight`, `transformer.wpe.weight`, `transformer.h.<index>.*`, and `transformer.ln_f.*`.

Schema version 1 describes built-in architecture families. It does not deserialize arbitrary operation graphs. `Tensor::graph_json()` provides a separate diagnostic execution-graph representation; it does not contain model parameter data or replace the checkpoint.
