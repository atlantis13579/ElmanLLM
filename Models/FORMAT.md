# Elman model graphs, version 2

The model files are executable layer DAGs. `create_model` constructs the graph
without dispatching on architecture names. `architecture` is descriptive metadata;
`task` selects classification, regression or language-modeling dataset semantics.
The four shipped definitions use the same graph/module runtime and optimizer path.

The design follows the separation of layer configuration, connections and weights
in [Keras serialization](https://keras.io/guides/serialization_and_saving/) and the
[Functional API](https://keras.io/guides/functional_api/). This is an Elman format,
not a Keras file. Native tensor layouts and native kernal semantics apply.

## Model contract

* `format`: `elman.model`; `version`: `2`.
* `name`: checkpoint identity. `architecture`: optional descriptive string.
* `task`: `classification`, `regression`, or `language_modeling`.
* `output_features`: class count, regression width or vocabulary size.
* `parameters`: optional JSON configuration variables, not learned tensors.
* `inputs`: named ports with `name`, `dtype`, and full `shape`, including batch.
  `-1` is dynamic. A dynamic non-batch dimension needs a positive `sample_shape`
  for the CSV trainer. `max_shape` bounds dynamic dimensions, e.g. GPT2 context.
* `outputs`: named ports with `name`, `value`, `dtype` and full `shape`.
* `layers`: layer instances and their connections.

`$name` substitutes a configuration variable recursively. Numeric expressions
`{"product":[...]}`, `{"quotient":[a,b]}` and `{"sqrt":x}` describe dependent
dimensions/scales. Cyclic variable references are rejected. Integer constructor
arguments must resolve to integers. Parameters can be changed without editing C++.

## Layers and connections

Each layer has `name`, `class_name`, optional `config`, `inputs`, and optional
`outputs`. The default output value name is the layer name. Inputs reference graph
values; multiple consumers create branches. Explicit output lists expose `split`,
`sort` and recurrent kernal outputs. The loader sorts the DAG and rejects cycles,
unresolved references, duplicate names, unknown kernals/config fields and invalid
port counts. Native kernals validate tensor dimensions and dtype constraints at
execution. Model boundary shapes/dtypes are checked on every call.

Layer names cannot contain dots. Dots separate nested module names in checkpoints.
`@layer.weight` references a registered parameter in the current graph, including
nested paths. Multiple uses share the same tensor and accumulate gradients once.
The GPT2 definition alternatively uses a zero-input `Parameter` layer as a value
consumed by both embedding lookup and the output projection.

`Graph` is a nested DAG: its config has `inputs`, `outputs` (lists of local value
names), and `layers`. Outer ports map positionally onto these local ports.

`Sequence` (alias `Sequential`) is a one-input/one-output graph. Omitted connections
chain each child to the previous result. Config `repeat` instantiates independent
copies, with stable indexed names; it does not share weights or implement a recurrent
loop. For repeated sequences use implicit connections between children, and put
explicit local branches inside a nested `Graph`, as GPT2 does. A recurrent temporal
state is represented by explicit `state` input/output ports and recurrent kernals.

## Reusable nested JSON

The root `definitions` object holds reusable JSON Graph/Sequence templates. A
`Subgraph` instance supplies a `ref` and optional configuration `parameters`:

```json
{
  "definitions": {
    "TransformerBlock": {"$include": "Modules/TransformerBlock.json"}
  },
  "layers": [
    {
      "name": "block",
      "class_name": "Subgraph",
      "inputs": ["hidden"],
      "config": {
        "ref": "TransformerBlock",
        "parameters": {"embedding_dim": 768, "num_heads": 12}
      }
    }
  ]
}
```

This is a fragment; the surrounding graph declares `hidden` and its output. The
definition value may instead be an inline object with `class_name`, `parameters`
and `config`, keeping the entire model in one JSON file. Template `parameters`
declare accepted arguments and defaults; `null` declares a required argument.
Arguments are resolved in the caller's scope, then the template is expanded in its
own local scope. Undeclared and missing arguments are rejected. Templates can call
other templates from the root `definitions`, but recursive references are rejected.
Each instance owns independent parameters; references do not imply weight sharing.
Expansion preserves the instance's name, connections and nested parameter paths.

`{"$include":"relative/path.json"}` embeds any JSON value. Paths are relative to
the containing file, including nested includes. An include object cannot have other
fields; missing files, include cycles and excessive nesting are rejected. Use
`read_model_definition(path)` before `create_model` when loading files. The training
and inference entry points do this automatically. In-memory `create_model(Json)`
accepts inline definitions without filesystem access. Checkpoints receive the
embedded definition, so the stored model contract contains the included JSON rather
than depending on its path. The usual definition-equality check still protects
against loading a checkpoint after changing a subgraph.

`Modules/TransformerBlock.json` defines the causal pre-normalization block used by
GPT2 (self-attention, two residual connections and GELU feed-forward network).
`GPT2.json` supplies its parameters and repeats the template using Sequence.

`Linear` owns `[out_features,in_features]` weights. `Conv2d` uses NCHW inputs and
OIHW weights. Stateful normalization uses registered persistent buffers. `reshape`
accepts `0` to copy the corresponding input dimension and one `-1` to infer a size.
`split_heads`/`merge_heads`, `position_ids`, `causal_mask` and `embedding` provide
dynamic sequence primitives without embedding a transformer model in C++.
Zero-input `constant`, `zeros`, `ones`, `normal` and `uniform` kernals construct
non-trainable values; `Parameter` registers a trainable weight instead.

## Kernal catalog

`Kernals.json` is the unified catalog:

* `entries`:  tensor-only kernals, trainable layers and composition primitives
  available as version-2 `class_name` values. Entries declare named tensor operands,
  optional/variadic operands, outputs, configuration types/defaults and port counts.
  `backend`, `seed`, and `training` are supplied by the execution context.
* `imported_graph_kernals`: the existing version-1 `elman.graph` importer dialect
  (ONNX/Keras/Torch aliases), with input/output arities. It retains its own attribute
  conventions and runtime checks; these aliases are not version-2 layer names.
* `native_interfaces`: typed C++ APIs requiring objects such as `QuantizedTensor`,
  `PagedAttentionMap` or sampling state, and inference layer wrappers. These are
  documented native interfaces, not falsely advertised as tensor-only bindings.

Every tensor input/output has a `shape` array listing its axis sizes in order.
Symbols are local to one kernal signature; the same symbol means the same size.
For example, matrix multiplication is `[M,N] x [N,K] -> [M,K]`: both inputs share
the contracted dimension `N`, while the output retains `M` and `K`.

The batched `matmul` signature is:

```json
{
  "inputs": [
    {"name": "a", "shape": ["...batch_a", "M", "N"]},
    {"name": "b", "shape": ["...batch_b", "N", "K"]}
  ],
  "outputs": [{"name": "output", "shape": ["...batch", "M", "K"]}]
}
```

Here `batch = broadcast(batch_a, batch_b)`. An empty batch prefix gives ordinary
matrix multiplication; `[2,3] x [3,4] -> [2,4]`. `...name` expands a dimension
list of any length, including zero. For example, ReLU uses `["...dims"]` on both
ports to preserve all dimensions. `[]` means a scalar, `[N]` a vector, and
fixed integers specify fixed sizes. Each array element may also express a single
axis size, such as `H*W` or `config.features`; `...config.shape` expands an entire
configured dimension list.

`shape_constraints` explains broadcasting, derived dimensions and conditional
shapes. `shape_alternatives` lists additional supported shapes, such as vector
operands of `matmul` or a retained size-one reduction axis. Native object/enum
arguments without tensor dimensions use `null`. These are descriptive dimension
contracts, not a shape-expression evaluator: native kernals and model ports
still validate runtime shapes. The catalog's `shape_notation` records the notation.

The catalog covers public tensor/layer APIs, not private backend kernels or their
backward dispatch implementations. Native backend/dtype/autograd restrictions still
apply, including inference-only operations. CMake embeds the catalog in the library
so deployment does not depend on the current directory. Adding an implementation
requires a C++ kernel/binding;
JSON cannot invent one.

Catalog version 2 omits redundant defaults. Data ports are `Tensor` by default;
`variadic: true` denotes a tensor list. Configuration arguments retain their JSON
types (`integer`, `number`, `boolean`, `string`, `array`, `object`). Native C++
interfaces retain `cpp_type` for non-Tensor arguments/results such as
`QuantizedTensor`, scalar values and sampling state.

An argument is required when neither `default` nor `default_expression` is present.
An optional Tensor has `default: null`, meaning the operand may be omitted, not
that the graph accepts a JSON null as a tensor value. Arity bounds are inferred
from port lists, defaults and `variadic`; explicit min/max fields only override
this inference. Missing `config` and `shape_constraints` mean an empty object and
empty list. These conventions are recorded once under the catalog's `defaults`.

## Training, inference and compatibility

CSV training uses the existing `elman.training` configuration and single input/output
interface. General multi-port callers use `GraphModel::forward(TensorMap, ...)`.
`load_model_session` exposes the declared named ports for inference. Training and
inference use the same layer graph. Evaluation disables recording and dropout;
BatchNorm statistics are saved/restored with parameters. Mixed-precision training
returns floating outputs in the declared boundary dtype through a differentiable
cast. Checkpoints retain the full definition and reject mismatched definitions.

The old version-1 architecture constructors were removed. Their old checkpoints
are not silently reinterpreted: parameter names, Linear layout and initialization
have changed. Start new runs with version-2 definitions. The separate reinforcement
learning model interfaces and external pretrained model importers retain their
existing formats.

`Tests/ModelGraphTest.cpp` checks graph branches, gradients, multi-port sessions,
stateful buffers, subgraphs, kernal catalog validation and malformed graphs using
synthetic fixtures. `Tests/ModelIO.cpp` covers JSON, datasets, checkpoint integrity
and inference reports without depending on shipped model definitions.
