# Model format version 1

Model definitions use `format: "elman.model"`, `version: 1`, a unique `name`, and an `architecture` identifier. The current architecture registry contains `alexnet` and `gpt2`. The format is declarative at the architecture level; it is not an importer for arbitrary third-party computation graphs.

`alexnet.json` defines a full-size five-convolution, three-linear-layer AlexNet variant with ReLU, max pooling and dropout. Its 224 x 224 input leads to a 6 x 6 final feature map. Local response normalization and the original two-GPU grouped convolutions are omitted, following the common modern single-GPU variant. Convolution parameters are NCHW/OIHW and linear weights are input-by-output matrices.

`gpt2.json` defines the GPT-2 small architecture: 12 decoder blocks, 768 embedding features, 12 attention heads, a 1,024-token context and a 50,257-token vocabulary. Each block uses pre-layer-normalization, causal multi-head self-attention, residual connections, GELU and a four-times-expanded feed-forward network. Token and learned position embeddings are summed. The language-model output head shares the token-embedding parameter.

These are the definitions used by automated training and inference: AlexNet has 61,100,840 parameters and GPT2 has 124,439,808. No reduced architecture variants are provided. Weights are randomly initialized, not downloaded pretrained weights. The smoke configurations keep normal model dropout, use one sample and one update, and change only data batch/sequence lengths. GPT2 retains its 1,024-position embedding table even with eight-token inputs.

## Training and inference

Training configurations use `format: "elman.training"` and `version: 1`. `model_name` must match the referenced model's name. `model` and `data.path` are relative to the configuration's directory. `data.format` is `image_csv` or `token_csv`. `training` contains the number of steps, batch size, optimizer, learning rate, Adam coefficients, SGD momentum, weight decay, gradient clipping, seed, shuffle, reporting/checkpoint intervals, optional warmup and constant/cosine learning-rate schedule. `output` names the checkpoint, metrics and computation-graph files.

```
elman train Configs/alexnet_smoke.json --backend cuda
elman train Configs/gpt2_smoke.json --backend cuda
elman infer Configs/alexnet_smoke.json --backend cuda
elman infer Configs/gpt2_smoke.json --backend cuda
elman train Configs/gpt2_smoke.json --resume Intermediate/Runs/gpt2_full/model.elman --steps 2
```

The same commands accept `cpu`, `compute` or `metal` when that backend is available. Training fails on missing/non-finite gradients, missing parameter updates, failed checkpoint roundtrip or, when configured, a non-decreasing final evaluation loss. Metrics distinguish training loss from dropout-free initial/final evaluation loss. Inference returns class predictions or teacher-forced next-token predictions. Tokenization and autoregressive text generation are outside this first CLI version.

Optimizer updates are currently performed on the host using downloaded gradients, then uploaded to the selected device. Forward and backward tensor arithmetic executes through the selected native backend. This design prioritizes a self-contained, verifiable first version over throughput. Training supports SGD with momentum, Adam and decoupled-weight-decay AdamW.

## Checkpoint format version 1

Checkpoints are little-endian binary files with an eight-byte `ELMAN01\0` signature and a 32-bit version. They contain the model name, normalized model JSON, optimizer name, completed step count, named parameter shapes and Float32 values, and first/second optimizer moment arrays. An FNV-1a 64-bit integrity checksum covers the preceding bytes. Lengths, shapes, names, finite values, checksum and trailing bytes are validated before model parameters are assigned. Model definitions must match exactly on restore. Weight/moment blocks use buffered I/O without changing the binary layout. Files are written through a temporary sibling and replaced only after the complete write succeeds.

Resume restores parameters, optimizer moments and the completed step count; `--steps` requests additional steps. The data-order generator, batch cursor and learning-rate schedule restart for the new invocation, so resume is not a bit-exact continuation of a shuffled or scheduled run. The smoke configurations have no shuffle and a constant learning rate; model dropout remains enabled during training.
