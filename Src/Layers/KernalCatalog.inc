// Generated from Models/Kernals.json. Keep each literal below 16 KiB.
// Do not edit the catalog content by hand.
R"ELMANCATALOG(
{
  "format": "elman.kernals",
  "version": 2,
  "semantics": {
    "tensor_layout": "Native layouts; Linear weights [out,in], Conv2d NCHW/OIHW.",
    "dtypes": "Native kernal dtype/backend checks apply; no implicit input casts.",
    "autograd": "Native autograd support is preserved. Inference-only kernals reject unsupported backward use.",
    "context": ["backend", "seed", "training"],
    "shape": "Input/output shapes are ordered dimension lists. Equal symbols within one kernal signature denote equal dimension sizes; native kernels validate runtime sizes.",
    "scope": "Public tensor functions in source headers, stateful nn layers and graph composition primitives.",
    "imported_configuration": "Imported elman.graph v1 aliases retain their attribute conventions and native validation."
  },
  "entries": {
    "BatchNorm2d": {
      "kind": "layer",
      "inputs": [{"name": "input", "shape": ["N", "config.channels", "H", "W"]}],
      "outputs": [{"name": "output", "shape": ["N", "config.channels", "H", "W"]}],
      "config": {
        "channels": {"type": "integer"},
        "epsilon": {"type": "number", "default": 1e-05},
        "momentum": {"type": "number", "default": 0.1}
      }
    },
    "Conv2d": {
      "kind": "layer",
      "inputs": [{"name": "input", "shape": ["N", "config.in_channels", "H", "W"]}],
      "outputs": [
        {
          "name": "output",
          "shape": [
            "N",
            "config.out_channels",
            "floor((H + 2*padding - kernel_size)/stride) + 1",
            "floor((W + 2*padding - kernel_size)/stride) + 1"
          ]
        }
      ],
      "config": {
        "in_channels": {"type": "integer"},
        "out_channels": {"type": "integer"},
        "kernel_size": {"type": "integer"},
        "stride": {"type": "integer", "default": 1},
        "padding": {"type": "integer", "default": 0},
        "bias": {"type": "boolean", "default": true}
      },
      "shape_constraints": [
        "Native dtype-specific paths may accept unbatched CHW input and return unbatched CHW output; the displayed contract is the common NCHW form."
      ]
    },
    "Dropout": {
      "kind": "layer",
      "inputs": [{"name": "input", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}],
      "config": {"probability": {"type": "number", "default": 0.5}}
    },
    "Graph": {
      "kind": "layer",
      "inputs": [{"name": "input", "variadic": true, "shape": ["...input_dims[i]"]}],
      "outputs": [{"name": "output", "variadic": true, "shape": ["...output_dims[i]"]}],
      "config": {"layers": {"type": "array"}, "inputs": {"type": "array"}, "outputs": {"type": "array"}},
      "shape_constraints": ["input_dims[i] and output_dims[i] are the dimension lists of port i in the instantiated graph."]
    },
    "GroupNorm": {
      "kind": "layer",
      "inputs": [{"name": "input", "shape": ["N", "config.channels", "...spatial"]}],
      "outputs": [{"name": "output", "shape": ["N", "config.channels", "...spatial"]}],
      "config": {"channels": {"type": "integer"}, "groups": {"type": "integer"}, "epsilon": {"type": "number", "default": 1e-05}},
      "shape_constraints": ["config.channels % config.groups == 0."]
    },
    "LayerNorm": {
      "kind": "layer",
      "inputs": [{"name": "input", "shape": ["...batch", "config.features"]}],
      "outputs": [{"name": "output", "shape": ["...batch", "config.features"]}],
      "config": {"features": {"type": "integer"}, "epsilon": {"type": "number", "default": 1e-05}}
    },
    "Linear": {
      "kind": "layer",
      "inputs": [{"name": "input", "shape": ["...batch", "config.in_features"]}],
      "outputs": [{"name": "output", "shape": ["...batch", "config.out_features"]}],
      "config": {
        "in_features": {"type": "integer"},
        "out_features": {"type": "integer"},
        "bias": {"type": "boolean", "default": true}
      }
    },
    "Parameter": {
      "kind": "layer",
      "inputs": [],
      "outputs": [{"name": "output", "shape": ["...config.shape"]}],
      "config": {
        "shape": {"type": "array"},
        "initializer": {"type": "string", "default": "normal"},
        "mean": {"type": "number", "default": 0},
        "stddev": {"type": "number", "default": 0.02}
      }
    },
    "ReLU": {
      "kind": "layer",
      "inputs": [{"name": "input", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "Sequence": {
      "kind": "layer",
      "inputs": [{"name": "input", "shape": ["...input_dims"]}],
      "outputs": [{"name": "output", "shape": ["...output_dims"]}],
      "config": {"layers": {"type": "array"}, "repeat": {"type": "integer", "default": 1}},
      "shape_constraints": [
        "input_dims and output_dims are the first and last dimension lists in the instantiated single-input/single-output chain."
      ]
    },
    "Sequential": {
      "kind": "layer",
      "inputs": [{"name": "input", "shape": ["...input_dims"]}],
      "outputs": [{"name": "output", "shape": ["...output_dims"]}],
      "config": {"layers": {"type": "array"}, "repeat": {"type": "integer", "default": 1}},
      "shape_constraints": [
        "input_dims and output_dims are the first and last dimension lists in the instantiated single-input/single-output chain."
      ]
    },
    "Subgraph": {
      "kind": "layer",
      "inputs": [{"name": "input", "variadic": true, "shape": ["...input_dims[i]"]}],
      "outputs": [{"name": "output", "variadic": true, "shape": ["...output_dims[i]"]}],
      "config": {"ref": {"type": "string"}, "parameters": {"type": "object", "default": {}}},
      "phase": "definition_expansion",
      "shape_constraints": ["input_dims[i] and output_dims[i] are the dimension lists of port i in the instantiated graph."]
    },
    "abs": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "add": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "a", "shape": ["...A"]}, {"name": "b", "shape": ["...B"]}],
      "outputs": [{"name": "output", "shape": ["...broadcast_dims"]}],
      "shape_constraints": ["broadcast_dims = broadcast(A, B)."]
    },
    "add_scalar": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "a", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}],
      "config": {"b": {"type": "number"}}
    },
    "addmm": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "c", "shape": ["...C"]}, {"name": "a", "shape": ["M", "N"]}, {"name": "b", "shape": ["N", "K"]}],
      "outputs": [{"name": "output", "shape": ["M", "K"]}],
      "config": {"beta": {"type": "number", "default": 1}, "alpha": {"type": "number", "default": 1}},
      "shape_constraints": ["c must broadcast to [M,K] without expanding it."]
    },
    "amax": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...before", "D", "...after"], "shape_alternatives": [[]]}],
      "outputs": [{"name": "output", "shape": ["...before", "...after"], "shape_alternatives": [["...before", 1, "...after"]]}],
      "config": {"dim": {"type": "integer", "default": -1}, "keepdim": {"type": "boolean", "default": false}},
      "shape_constraints": [
        "before/after surround the normalized config.dim axis.",
        "The primary output shape applies when config.keepdim is false; the alternative retains a size-one axis when true.",
        "A scalar input [] remains [] for config.dim = 0 or -1, including when retaining axes."
      ]
    },
    "amin": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...before", "D", "...after"], "shape_alternatives": [[]]}],
)ELMANCATALOG",
R"ELMANCATALOG(      "outputs": [{"name": "output", "shape": ["...before", "...after"], "shape_alternatives": [["...before", 1, "...after"]]}],
      "config": {"dim": {"type": "integer", "default": -1}, "keepdim": {"type": "boolean", "default": false}},
      "shape_constraints": [
        "before/after surround the normalized config.dim axis.",
        "The primary output shape applies when config.keepdim is false; the alternative retains a size-one axis when true.",
        "A scalar input [] remains [] for config.dim = 0 or -1, including when retaining axes."
      ]
    },
    "angle": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "argsort": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}],
      "config": {"dim": {"type": "integer", "default": -1}, "descending": {"type": "boolean", "default": false}},
      "shape_constraints": ["config.dim addresses an existing input axis."]
    },
    "attention_softmax": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "scores", "shape": ["...score_dims"]}, {"name": "mask", "shape": ["...mask_dims"]}],
      "outputs": [{"name": "output", "shape": ["...score_dims"]}],
      "config": {"divisor": {"type": "number", "default": 1}, "dim": {"type": "integer", "default": -1}},
      "shape_constraints": ["mask broadcasts to scores without expanding the score shape."]
    },
    "avg_pool2d": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["N", "C", "H", "W"]}],
      "outputs": [
        {
          "name": "output",
          "shape": [
            "N",
            "C",
            "floor((H + 2*padding - kernel)/effective_stride) + 1",
            "floor((W + 2*padding - kernel)/effective_stride) + 1"
          ]
        }
      ],
      "config": {
        "kernel": {"type": "integer"},
        "stride": {"type": "integer", "default": 0},
        "padding": {"type": "integer", "default": 0}
      },
      "shape_constraints": [
        "effective_stride is config.stride, or config.kernel when stride is 0.",
        "Native dtype-specific paths may accept unbatched CHW input and return unbatched CHW output; the displayed contract is the common NCHW form."
      ]
    },
    "batch_norm": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [
        {"name": "x", "shape": ["N", "C", "...spatial"]},
        {"name": "weight", "shape": ["C"]},
        {"name": "bias", "shape": ["C"]}
      ],
      "outputs": [{"name": "output", "shape": ["N", "C", "...spatial"]}],
      "config": {"eps": {"type": "number", "default": 1e-05}}
    },
    "bmm": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "a", "shape": ["B", "M", "N"]}, {"name": "b", "shape": ["B", "N", "K"]}],
      "outputs": [{"name": "output", "shape": ["B", "M", "K"]}]
    },
    "cast": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}],
      "config": {"dtype": {"type": "string"}}
    },
    "causal_depthwise_conv1d": {
      "kind": "kernal",
      "source": "Kernals/recurrent_ops.h",
      "inputs": [
        {"name": "input", "shape": ["B", "T", "C"]},
        {"name": "weight", "shape": ["C", "K"]},
        {"name": "state", "shape": ["B", "K - 1", "C"], "default": null}
      ],
      "outputs": [{"name": "values", "shape": ["B", "T", "C"]}, {"name": "state", "shape": ["B", "K - 1", "C"]}],
      "shape_constraints": ["Omitted state is zero initialized."]
    },
    "causal_mask": {
      "kind": "kernal",
      "inputs": [{"name": "input", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": [1, 1, "input.shape[config.axis]", "input.shape[config.axis]"]}],
      "config": {"axis": {"type": "integer", "default": 1}}
    },
    "complex": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "real", "shape": ["...A"]}, {"name": "imaginary", "shape": ["...B"]}],
      "outputs": [{"name": "output", "shape": ["...broadcast_dims"]}],
      "shape_constraints": ["broadcast_dims = broadcast(A, B)."]
    },
    "concat": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "inputs", "variadic": true, "shape": ["...before", "D_i", "...after"]}],
      "outputs": [{"name": "output", "shape": ["...before", "sum(D_i)", "...after"]}],
      "config": {"dim": {"type": "integer"}},
      "shape_constraints": [
        "Each variadic input has equal rank and equal extents outside config.dim; output sums extents on config.dim.",
        "D_i is input i extent at config.dim; all before/after dimensions match across inputs."
      ]
    },
    "conj": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "constant": {
      "kind": "kernal",
      "inputs": [],
      "outputs": [{"name": "output", "shape": ["...config.shape"]}],
      "config": {"shape": {"type": "array"}, "dtype": {"type": "string", "default": "float32"}, "values": {"type": "array"}}
    },
    "contiguous_view": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...config.shape"]}],
      "config": {"shape": {"type": "array"}, "offset": {"type": "integer", "default": 0}},
      "shape_constraints": ["config.offset + product(config.shape) <= numel(x); shape dimensions are explicit."]
    },
    "conv1d": {
      "kind": "kernal",
      "source": "Kernals/audio_ops.h",
      "inputs": [
        {"name": "input", "shape": ["N", "Cin", "L"]},
        {"name": "weight", "shape": ["Cout", "Cin", "K"]},
        {"name": "bias", "shape": ["Cout"], "default": null}
      ],
      "outputs": [{"name": "output", "shape": ["N", "Cout", "floor((L + 2*padding - dilation*(K-1) - 1)/stride) + 1"]}],
      "config": {
        "stride": {"type": "integer", "default": 1},
        "padding": {"type": "integer", "default": 0},
        "dilation": {"type": "integer", "default": 1}
      }
    },
    "conv2d": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [
        {"name": "x", "shape": ["N", "Cin", "H", "W"]},
        {"name": "weight", "shape": ["Cout", "Cin", "Kh", "Kw"]},
        {"name": "bias", "shape": ["Cout"]}
      ],
      "outputs": [
        {
          "name": "output",
          "shape": ["N", "Cout", "floor((H + 2*padding - Kh)/stride) + 1", "floor((W + 2*padding - Kw)/stride) + 1"]
        }
      ],
      "config": {"stride": {"type": "integer", "default": 1}, "padding": {"type": "integer", "default": 0}},
      "shape_constraints": [
        "stride and padding come from config; native convolution has one group and unit dilation.",
        "Native dtype-specific paths may accept unbatched CHW input and return unbatched CHW output; the displayed contract is the common NCHW form."
      ]
    },
    "conv2d_with_channel_bias": {
      "kind": "kernal",
      "source": "Kernals/inference_ops.h",
      "inputs": [
        {"name": "input", "shape": ["N", "Cin", "H", "W"]},
        {"name": "weight", "shape": ["Cout", "Cin", "Kh", "Kw"]},
        {"name": "bias", "shape": ["Cout"]}
      ],
      "outputs": [
        {
          "name": "output",
          "shape": ["N", "Cout", "floor((H + 2*padding - Kh)/stride) + 1", "floor((W + 2*padding - Kw)/stride) + 1"]
        }
      ],
      "config": {"stride": {"type": "integer"}, "padding": {"type": "integer"}},
      "shape_constraints": [
)ELMANCATALOG",
R"ELMANCATALOG(        "stride and padding come from config; native convolution has one group and unit dilation.",
        "Native dtype-specific paths may accept unbatched CHW input and return unbatched CHW output; the displayed contract is the common NCHW form."
      ]
    },
    "cos": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "cross_entropy": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "logits", "shape": ["...examples", "C"]}, {"name": "targets", "shape": ["...target_dims"]}],
      "outputs": [{"name": "output", "shape": []}],
      "shape_constraints": ["product(target_dims) == product(examples); logits rank >= 2."]
    },
    "cumsum": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}],
      "config": {"dim": {"type": "integer", "default": -1}},
      "shape_constraints": ["config.dim addresses an existing input axis."]
    },
    "div": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "a", "shape": ["...A"]}, {"name": "b", "shape": ["...B"]}],
      "outputs": [{"name": "output", "shape": ["...broadcast_dims"]}],
      "shape_constraints": ["broadcast_dims = broadcast(A, B)."]
    },
    "div_scalar": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "a", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}],
      "config": {"b": {"type": "number"}}
    },
    "dropout": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}],
      "config": {"probability": {"type": "number"}}
    },
    "embedding": {
      "kind": "kernal",
      "inputs": [{"name": "ids", "shape": ["...ids"]}, {"name": "weight", "shape": ["V", "D"]}],
      "outputs": [{"name": "output", "shape": ["...ids", "D"]}]
    },
    "eq": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "a", "shape": ["...A"]}, {"name": "b", "shape": ["...B"]}],
      "outputs": [{"name": "output", "shape": ["...broadcast_dims"]}],
      "shape_constraints": ["broadcast_dims = broadcast(A, B)."]
    },
    "exact_gelu": {
      "kind": "kernal",
      "source": "Kernals/inference_ops.h",
      "inputs": [{"name": "value", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "exp": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "expand": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...config.shape"]}],
      "config": {"shape": {"type": "array"}},
      "shape_constraints": ["Only singleton dimensions can be broadcast to larger extents."]
    },
    "flatten": {
      "kind": "kernal",
      "inputs": [{"name": "input", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["product(input.shape[:axis])", "product(input.shape[axis:])"]}],
      "config": {"axis": {"type": "integer", "default": 1}}
    },
    "flux2::apply_rotary": {
      "kind": "kernal",
      "source": "Kernals/flux2_ops.h",
      "inputs": [{"name": "heads", "shape": ["B", "H", "L", 128]}, {"name": "table", "shape": ["Bt", "L", 128]}],
      "outputs": [{"name": "output", "shape": ["B", "H", "L", 128]}],
      "shape_constraints": ["Bt is 1 or B."]
    },
    "flux2::attention": {
      "kind": "kernal",
      "source": "Kernals/flux2_ops.h",
      "inputs": [
        {"name": "query", "shape": ["B", "H", "Q", 128]},
        {"name": "key", "shape": ["B", "H", "K", 128]},
        {"name": "value", "shape": ["B", "H", "K", 128]}
      ],
      "outputs": [{"name": "output", "shape": ["B", "H", "Q", 128]}]
    },
    "flux2::modulate": {
      "kind": "kernal",
      "source": "Kernals/flux2_ops.h",
      "inputs": [
        {"name": "x", "shape": ["B", "L", "D"]},
        {"name": "shift", "shape": ["B", "S", "D"]},
        {"name": "scale", "shape": ["B", "S", "D"]}
      ],
      "outputs": [{"name": "output", "shape": ["B", "L", "D"]}],
      "shape_constraints": ["S is 1 or L."]
    },
    "flux2::pack_latents": {
      "kind": "kernal",
      "source": "Kernals/flux2_ops.h",
      "inputs": [
        {"name": "posterior", "shape": ["B", "C", "2*H", "2*W"]},
        {"name": "mean", "shape": [128]},
        {"name": "variance", "shape": [128]}
      ],
      "outputs": [{"name": "output", "shape": ["B", 128, "H", "W"]}],
      "shape_constraints": ["C is 32 (mean) or 64 (posterior moments)."]
    },
    "flux2::rotary_table": {
      "kind": "kernal",
      "source": "Kernals/flux2_ops.h",
      "inputs": [{"name": "ids", "shape": ["B", "L", 4]}],
      "outputs": [{"name": "output", "shape": ["B", "L", 128]}]
    },
    "flux2::timestep_embedding": {
      "kind": "kernal",
      "source": "Kernals/flux2_ops.h",
      "inputs": [{"name": "times", "shape": ["B"]}],
      "outputs": [{"name": "output", "shape": ["B", 256]}]
    },
    "flux2::unpack_latents": {
      "kind": "kernal",
      "source": "Kernals/flux2_ops.h",
      "inputs": [
        {"name": "packed", "shape": ["B", 128, "H", "W"]},
        {"name": "mean", "shape": [128]},
        {"name": "variance", "shape": [128]}
      ],
      "outputs": [{"name": "output", "shape": ["B", 32, "2*H", "2*W"]}]
    },
    "gated_delta_rule": {
      "kind": "kernal",
      "source": "Kernals/recurrent_ops.h",
      "inputs": [
        {"name": "query", "shape": ["B", "T", "Hk", "Dk"]},
        {"name": "key", "shape": ["B", "T", "Hk", "Dk"]},
        {"name": "value", "shape": ["B", "T", "Hv", "Dv"]},
        {"name": "log_decay", "shape": ["B", "T", "Hv"]},
        {"name": "beta", "shape": ["B", "T", "Hv"]},
        {"name": "state", "shape": ["B", "Hv", "Dk", "Dv"], "default": null}
      ],
      "outputs": [{"name": "values", "shape": ["B", "T", "Hv", "Dv"]}, {"name": "state", "shape": ["B", "Hv", "Dk", "Dv"]}],
      "shape_constraints": ["Hv % Hk == 0; recurrent state is Float32; omitted state is zero initialized."]
    },
    "gather": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}, {"name": "indices", "shape": ["...index_dims"]}],
      "outputs": [{"name": "output", "shape": ["...index_dims"]}],
      "config": {"dim": {"type": "integer"}},
      "shape_constraints": ["indices has the same rank as x; non-gather extents cannot exceed x."]
    },
    "ge": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "a", "shape": ["...A"]}, {"name": "b", "shape": ["...B"]}],
      "outputs": [{"name": "output", "shape": ["...broadcast_dims"]}],
      "shape_constraints": ["broadcast_dims = broadcast(A, B)."]
    },
    "gelu": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "group_norm": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [
        {"name": "x", "shape": ["N", "C", "...spatial"]},
        {"name": "weight", "shape": ["C"]},
        {"name": "bias", "shape": ["C"]}
      ],
      "outputs": [{"name": "output", "shape": ["N", "C", "...spatial"]}],
      "config": {"groups": {"type": "integer"}, "eps": {"type": "number", "default": 1e-05}},
      "shape_constraints": ["C % config.groups == 0; rank >= 2."]
    },
    "gt": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
)ELMANCATALOG",
R"ELMANCATALOG(      "inputs": [{"name": "a", "shape": ["...A"]}, {"name": "b", "shape": ["...B"]}],
      "outputs": [{"name": "output", "shape": ["...broadcast_dims"]}],
      "shape_constraints": ["broadcast_dims = broadcast(A, B)."]
    },
    "identity": {
      "kind": "kernal",
      "inputs": [{"name": "input", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "imag": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "image_tokens": {
      "kind": "kernal",
      "source": "Kernals/inference_ops.h",
      "inputs": [{"name": "image", "shape": ["B", "C", "H", "W"]}],
      "outputs": [{"name": "output", "shape": ["B", "H * W", "C"]}]
    },
    "index_copy": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [
        {"name": "x", "shape": ["...before", "D", "...after"]},
        {"name": "indices", "shape": ["I"]},
        {"name": "source", "shape": ["...before", "I", "...after"]}
      ],
      "outputs": [{"name": "output", "shape": ["...before", "D", "...after"]}],
      "config": {"dim": {"type": "integer"}},
      "shape_constraints": ["before/after are the dimensions before/after the normalized config.dim axis; D is its extent."]
    },
    "index_select": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...before", "D", "...after"]}, {"name": "indices", "shape": ["I"]}],
      "outputs": [{"name": "output", "shape": ["...before", "I", "...after"]}],
      "config": {"dim": {"type": "integer"}},
      "shape_constraints": ["before/after are the dimensions before/after the normalized config.dim axis; D is its extent."]
    },
    "layer_norm": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...batch", "D"]}, {"name": "weight", "shape": ["D"]}, {"name": "bias", "shape": ["D"]}],
      "outputs": [{"name": "output", "shape": ["...batch", "D"]}],
      "config": {"eps": {"type": "number", "default": 1e-05}}
    },
    "le": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "a", "shape": ["...A"]}, {"name": "b", "shape": ["...B"]}],
      "outputs": [{"name": "output", "shape": ["...broadcast_dims"]}],
      "shape_constraints": ["broadcast_dims = broadcast(A, B)."]
    },
    "leaky_relu": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}],
      "config": {"slope": {"type": "number", "default": 0.01}}
    },
    "linear": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [
        {"name": "input", "shape": ["...batch", "I"]},
        {"name": "weight", "shape": ["O", "I"]},
        {"name": "bias", "shape": ["O"], "shape_alternatives": [[]], "default": null}
      ],
      "outputs": [{"name": "output", "shape": ["...batch", "O"]}],
      "shape_constraints": ["Bias may also be a scalar []."]
    },
    "log": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "log_softmax": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}],
      "config": {"dim": {"type": "integer", "default": -1}},
      "shape_constraints": ["config.dim addresses an existing input axis."]
    },
    "logical_and": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "a", "shape": ["...A"]}, {"name": "b", "shape": ["...B"]}],
      "outputs": [{"name": "output", "shape": ["...broadcast_dims"]}],
      "shape_constraints": ["broadcast_dims = broadcast(A, B)."]
    },
    "logical_not": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "logical_or": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "a", "shape": ["...A"]}, {"name": "b", "shape": ["...B"]}],
      "outputs": [{"name": "output", "shape": ["...broadcast_dims"]}],
      "shape_constraints": ["broadcast_dims = broadcast(A, B)."]
    },
    "lt": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "a", "shape": ["...A"]}, {"name": "b", "shape": ["...B"]}],
      "outputs": [{"name": "output", "shape": ["...broadcast_dims"]}],
      "shape_constraints": ["broadcast_dims = broadcast(A, B)."]
    },
    "matmul": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [
        {"name": "a", "shape": ["...batch_a", "M", "N"], "shape_alternatives": [["N"]]},
        {"name": "b", "shape": ["...batch_b", "N", "K"], "shape_alternatives": [["N"]]}
      ],
      "outputs": [
        {
          "name": "output",
          "shape": ["...batch", "M", "K"],
          "shape_alternatives": [["...batch_b", "K"], ["...batch_a", "M"], []]
        }
      ],
      "shape_constraints": [
        "batch = broadcast(batch_a, batch_b); empty batch prefixes give [M,N] x [N,K] -> [M,K].",
        "[N] x [...batch_b,N,K] -> [...batch_b,K]; [...batch_a,M,N] x [N] -> [...batch_a,M]; [N] x [N] -> []."
      ]
    },
    "matmul_transposed": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "a", "shape": ["...batch_a", "M", "N"]}, {"name": "b", "shape": ["...batch_b", "K", "N"]}],
      "outputs": [{"name": "output", "shape": ["...batch", "M", "K"]}],
      "shape_constraints": ["batch = broadcast(batch_a, batch_b); [M,N] x [K,N]^T -> [M,K]. Both operands have rank >= 2."]
    },
    "max": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...before", "D", "...after"], "shape_alternatives": [[]]}],
      "outputs": [{"name": "output", "shape": ["...before", "...after"], "shape_alternatives": [["...before", 1, "...after"]]}],
      "config": {"dim": {"type": "integer", "default": -1}, "keepdim": {"type": "boolean", "default": false}},
      "shape_constraints": [
        "before/after surround the normalized config.dim axis.",
        "The primary output shape applies when config.keepdim is false; the alternative retains a size-one axis when true.",
        "A scalar input [] remains [] for config.dim = 0 or -1, including when retaining axes."
      ]
    },
    "max_pool2d": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["N", "C", "H", "W"]}],
      "outputs": [
        {
          "name": "output",
          "shape": [
            "N",
            "C",
            "floor((H + 2*padding - kernel)/effective_stride) + 1",
            "floor((W + 2*padding - kernel)/effective_stride) + 1"
          ]
        }
      ],
      "config": {
        "kernel": {"type": "integer"},
        "stride": {"type": "integer", "default": 0},
        "padding": {"type": "integer", "default": 0}
      },
      "shape_constraints": [
        "effective_stride is config.stride, or config.kernel when stride is 0.",
        "Native dtype-specific paths may accept unbatched CHW input and return unbatched CHW output; the displayed contract is the common NCHW form."
      ]
    },
    "maximum": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "a", "shape": ["...A"]}, {"name": "b", "shape": ["...B"]}],
      "outputs": [{"name": "output", "shape": ["...broadcast_dims"]}],
      "shape_constraints": ["broadcast_dims = broadcast(A, B)."]
    },
    "mean": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
)ELMANCATALOG",
R"ELMANCATALOG(      "inputs": [{"name": "x", "shape": ["...before", "D", "...after"], "shape_alternatives": [[]]}],
      "outputs": [{"name": "output", "shape": ["...before", "...after"], "shape_alternatives": [["...before", 1, "...after"]]}],
      "config": {"dim": {"type": "integer", "default": -1}, "keepdim": {"type": "boolean", "default": false}},
      "shape_constraints": [
        "before/after surround the normalized config.dim axis.",
        "The primary output shape applies when config.keepdim is false; the alternative retains a size-one axis when true.",
        "A scalar input [] remains [] for config.dim = 0 or -1, including when retaining axes."
      ]
    },
    "mean_all": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": []}]
    },
    "merge_heads": {
      "kind": "kernal",
      "inputs": [{"name": "input", "shape": ["B", "H", "T", "D"]}],
      "outputs": [{"name": "output", "shape": ["B", "T", "H * D"]}]
    },
    "min": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...before", "D", "...after"], "shape_alternatives": [[]]}],
      "outputs": [{"name": "output", "shape": ["...before", "...after"], "shape_alternatives": [["...before", 1, "...after"]]}],
      "config": {"dim": {"type": "integer", "default": -1}, "keepdim": {"type": "boolean", "default": false}},
      "shape_constraints": [
        "before/after surround the normalized config.dim axis.",
        "The primary output shape applies when config.keepdim is false; the alternative retains a size-one axis when true.",
        "A scalar input [] remains [] for config.dim = 0 or -1, including when retaining axes."
      ]
    },
    "minimum": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "a", "shape": ["...A"]}, {"name": "b", "shape": ["...B"]}],
      "outputs": [{"name": "output", "shape": ["...broadcast_dims"]}],
      "shape_constraints": ["broadcast_dims = broadcast(A, B)."]
    },
    "mse_loss": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "prediction", "shape": ["...dims"]}, {"name": "target", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": []}],
      "shape_constraints": [
        "Prediction and target shapes must match.",
        "Low-precision storage types also permit broadcast-compatible prediction and target shapes."
      ]
    },
    "mul": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "a", "shape": ["...A"]}, {"name": "b", "shape": ["...B"]}],
      "outputs": [{"name": "output", "shape": ["...broadcast_dims"]}],
      "shape_constraints": ["broadcast_dims = broadcast(A, B)."]
    },
    "mul_scalar": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "a", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}],
      "config": {"b": {"type": "number"}}
    },
    "ne": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "a", "shape": ["...A"]}, {"name": "b", "shape": ["...B"]}],
      "outputs": [{"name": "output", "shape": ["...broadcast_dims"]}],
      "shape_constraints": ["broadcast_dims = broadcast(A, B)."]
    },
    "normal": {
      "kind": "kernal",
      "inputs": [],
      "outputs": [{"name": "output", "shape": ["...config.shape"]}],
      "config": {
        "shape": {"type": "array"},
        "dtype": {"type": "string", "default": "float32"},
        "mean": {"type": "number", "default": 0},
        "stddev": {"type": "number", "default": 1}
      }
    },
    "ones": {
      "kind": "kernal",
      "inputs": [],
      "outputs": [{"name": "output", "shape": ["...config.shape"]}],
      "config": {"shape": {"type": "array"}, "dtype": {"type": "string", "default": "float32"}}
    },
    "packed_linear": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [
        {"name": "input", "shape": ["...batch", "I"]},
        {"name": "weight", "shape": ["sum(config.output_sizes)", "I"]},
        {"name": "bias", "shape": ["sum(config.output_sizes)"], "default": null}
      ],
      "outputs": [{"name": "output", "shape": ["product(batch) * sum(config.output_sizes)"]}],
      "config": {"output_sizes": {"type": "array"}},
      "shape_constraints": ["The output is a flat concatenation of separately flattened projections."]
    },
    "permute": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...permuted_dims"]}],
      "config": {"axes": {"type": "array"}},
      "shape_constraints": ["axes is a permutation of all input axes.", "permuted_dims[i] = dims[axes[i]], with axes from config.axes."]
    },
    "position_ids": {
      "kind": "kernal",
      "inputs": [{"name": "input", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["input.shape[config.axis]"]}],
      "config": {"axis": {"type": "integer", "default": 1}}
    },
    "pow": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "a", "shape": ["...A"]}, {"name": "b", "shape": ["...B"]}],
      "outputs": [{"name": "output", "shape": ["...broadcast_dims"]}],
      "shape_constraints": ["broadcast_dims = broadcast(A, B)."]
    },
    "pow_scalar": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "a", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}],
      "config": {"b": {"type": "number"}}
    },
    "prod": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...before", "D", "...after"], "shape_alternatives": [[]]}],
      "outputs": [{"name": "output", "shape": ["...before", "...after"], "shape_alternatives": [["...before", 1, "...after"]]}],
      "config": {"dim": {"type": "integer", "default": -1}, "keepdim": {"type": "boolean", "default": false}},
      "shape_constraints": [
        "before/after surround the normalized config.dim axis.",
        "The primary output shape applies when config.keepdim is false; the alternative retains a size-one axis when true.",
        "A scalar input [] remains [] for config.dim = 0 or -1, including when retaining axes."
      ]
    },
    "quick_gelu": {
      "kind": "kernal",
      "source": "Kernals/inference_ops.h",
      "inputs": [{"name": "value", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "real": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "relu": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "repeat": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...repeated_dims"]}],
      "config": {"repeats": {"type": "array"}},
      "shape_constraints": ["Left-pad dims with 1 to match config.repeats length; repeated_dims[i] = padded_dims[i] * config.repeats[i]."]
    },
    "reshape": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...target_dims"]}],
      "config": {"shape": {"type": "array"}},
      "shape_constraints": [
        "target_dims resolves config.shape: 0 copies the matching input axis, at most one -1 is inferred, and element count is unchanged."
      ]
    },
    "rms_norm": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
)ELMANCATALOG",
R"ELMANCATALOG(      "inputs": [{"name": "x", "shape": ["...batch", "D"]}, {"name": "weight", "shape": ["D"]}],
      "outputs": [{"name": "output", "shape": ["...batch", "D"]}],
      "config": {
        "eps": {"type": "number", "default": 1e-05},
        "rounding": {"type": "string", "default": "output", "enum": ["output", "normalized"]}
      }
    },
    "rotary_embedding": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [
        {"name": "x", "shape": ["...batch", "D"]},
        {"name": "cosine", "shape": ["...cos_dims"]},
        {"name": "sine", "shape": ["...sin_dims"]}
      ],
      "outputs": [{"name": "output", "shape": ["...batch", "D"]}],
      "shape_constraints": ["D is even; cosine and sine broadcast to the full input shape without expanding it."]
    },
    "rsqrt": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "scaled_dot_product_attention": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [
        {
          "name": "query",
          "shape": [
            "..."
          ]
        },
        {
          "name": "key",
          "shape": [
            "..."
          ]
        },
        {
          "name": "value",
          "shape": [
            "..."
          ]
        },
        {
          "name": "mask",
          "shape": [
            "..."
          ],
          "default_expression": "Tensor{}"
        }
      ],
      "outputs": [
        {
          "name": "output",
          "shape": [
            "...",
            "L",
            "Ev"
          ]
        }
      ],
      "config": {
        "dropout_p": {
          "type": "number",
          "default": 0
        },
        "is_causal": {
          "type": "boolean",
          "default": false
        },
        "scale": {
          "type": "number",
          "default_expression": "1/sqrt(E)"
        },
        "enable_gqa": {
          "type": "boolean",
          "default": false
        },
        "causal": {
          "type": "boolean",
          "default": false
        }
      },
      "shape_constraints": [
        "Q/K embedding dimensions match; K/V sequence lengths match. Mask broadcasts to attention weights."
      ]
    },
    "scatter": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [
        {"name": "x", "shape": ["...dims"]},
        {"name": "indices", "shape": ["...index_dims"]},
        {"name": "source", "shape": ["...index_dims"]}
      ],
      "outputs": [{"name": "output", "shape": ["...dims"]}],
      "config": {"dim": {"type": "integer"}}
    },
    "sgn": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "sigmoid": {
      "kind": "kernal",
      "source": "Kernals/recurrent_ops.h",
      "inputs": [{"name": "input", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "sign": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "silu": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "silu_mul": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "gate", "shape": ["...A"]}, {"name": "up", "shape": ["...B"]}],
      "outputs": [{"name": "output", "shape": ["...broadcast_dims"]}],
      "shape_constraints": ["broadcast_dims = broadcast(A, B)."]
    },
    "sin": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "sinusoidal_embedding": {
      "kind": "kernal",
      "source": "Kernals/inference_ops.h",
      "inputs": [],
      "outputs": [{"name": "output", "shape": [1, "config.width"]}],
      "config": {
        "step": {"type": "integer"},
        "width": {"type": "integer"},
        "frequency_shift": {"type": "number"},
        "cosine_first": {"type": "boolean"},
        "dtype": {"type": "string"}
      }
    },
    "slice": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...before", "D", "...after"]}],
      "outputs": [{"name": "output", "shape": ["...before", "L", "...after"]}],
      "config": {
        "dim": {"type": "integer"},
        "start": {"type": "integer"},
        "stop": {"type": "integer"},
        "step": {"type": "integer", "default": 1}
      },
      "shape_constraints": [
        "before/after are the dimensions before/after the normalized config.dim axis; D is its extent.",
        "L is the nonnegative number of indices selected by config.start, config.stop and config.step after native index normalization."
      ]
    },
    "soft_target_cross_entropy": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "logits", "shape": ["...examples", "C"]}, {"name": "probabilities", "shape": ["...examples", "C"]}],
      "outputs": [{"name": "output", "shape": []}]
    },
    "softmax": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}],
      "config": {"dim": {"type": "integer", "default": -1}},
      "shape_constraints": ["config.dim addresses an existing input axis."]
    },
    "sort": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "values", "shape": ["...dims"]}, {"name": "indices", "shape": ["...dims"]}],
      "config": {"dim": {"type": "integer", "default": -1}, "descending": {"type": "boolean", "default": false}}
    },
    "split": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...before", "D", "...after"]}],
      "outputs": [
        {
          "name": "output",
          "shape": ["...before", "min(config.size, D - i * config.size)", "...after"],
          "variadic": true
        }
      ],
      "config": {"size": {"type": "integer"}, "dim": {"type": "integer", "default": 0}},
      "shape_constraints": [
        "The output descriptor applies to each output; output count is ceil(x.shape[dim] / config.size).",
        "before/after are the dimensions before/after the normalized config.dim axis; D is its extent.",
        "i is the zero-based output index, 0 <= i < ceil(D / config.size)."
      ]
    },
    "split_heads": {
      "kind": "kernal",
      "inputs": [{"name": "input", "shape": ["B", "T", "D"]}],
      "outputs": [{"name": "output", "shape": ["B", "config.heads", "T", "D / config.heads"]}],
      "config": {"heads": {"type": "integer"}},
      "shape_constraints": ["D % config.heads == 0."]
    },
    "sqrt": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "squeeze": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...before", "D", "...after"], "shape_alternatives": [[]]}],
      "outputs": [{"name": "output", "shape": ["...before", "...after"], "shape_alternatives": [["...before", "D", "...after"]]}],
      "config": {"dim": {"type": "integer"}},
      "shape_constraints": [
        "before/after are the dimensions before/after the normalized config.dim axis; D is its extent.",
        "Remove the selected axis when D = 1; otherwise preserve the input dimensions.",
)ELMANCATALOG",
R"ELMANCATALOG(        "A scalar input [] remains [] for config.dim = 0 or -1."
      ]
    },
    "squeeze_all": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...remaining_dims"]}],
      "shape_constraints": ["remaining_dims preserves dims order after removing size-one axes."]
    },
    "sub": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "a", "shape": ["...A"]}, {"name": "b", "shape": ["...B"]}],
      "outputs": [{"name": "output", "shape": ["...broadcast_dims"]}],
      "shape_constraints": ["broadcast_dims = broadcast(A, B)."]
    },
    "sub_scalar": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "a", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}],
      "config": {"b": {"type": "number"}}
    },
    "sum": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...before", "D", "...after"], "shape_alternatives": [[]]}],
      "outputs": [{"name": "output", "shape": ["...before", "...after"], "shape_alternatives": [["...before", 1, "...after"]]}],
      "config": {"dim": {"type": "integer", "default": -1}, "keepdim": {"type": "boolean", "default": false}},
      "shape_constraints": [
        "before/after surround the normalized config.dim axis.",
        "The primary output shape applies when config.keepdim is false; the alternative retains a size-one axis when true.",
        "A scalar input [] remains [] for config.dim = 0 or -1, including when retaining axes."
      ]
    },
    "sum_all": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": []}]
    },
    "swish": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "tanh": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "tiled_attention": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [
        {"name": "query", "shape": ["B", "Hq", "Q", "D"]},
        {"name": "key", "shape": ["B", "Hkv", "K", "D"]},
        {"name": "value", "shape": ["B", "Hkv", "K", "Dv"]},
        {"name": "mask", "shape": ["...mask_dims"]}
      ],
      "outputs": [{"name": "output", "shape": ["B", "Hq", "Q", "Dv"]}],
      "config": {
        "divisor": {"type": "number", "default": 1},
        "query_tile": {"type": "integer", "default": 32},
        "causal": {"type": "boolean", "default": false},
        "causal_offset": {"type": "integer", "default": 0},
        "tiling": {"type": "string", "default": "queries", "enum": ["queries", "heads"]}
      },
      "shape_constraints": ["Hq % Hkv == 0; additive mask, when present, broadcasts to [B,Hq,Q,K]."]
    },
    "token_image": {
      "kind": "kernal",
      "source": "Kernals/inference_ops.h",
      "inputs": [{"name": "tokens", "shape": ["B", "L", "C"]}],
      "outputs": [{"name": "output", "shape": ["B", "C", "config.height", "config.width"]}],
      "config": {"height": {"type": "integer"}, "width": {"type": "integer"}},
      "shape_constraints": ["L == config.height * config.width."]
    },
    "transpose": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...transposed_dims"]}],
      "config": {"a": {"type": "integer"}, "b": {"type": "integer"}},
      "shape_constraints": ["transposed_dims equals dims with dimensions at config.a and config.b exchanged."]
    },
    "tril": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}],
      "config": {"diagonal": {"type": "integer", "default": 0}},
      "shape_constraints": ["Input rank >= 2; triangular masking uses the last two dimensions."]
    },
    "triu": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}],
      "config": {"diagonal": {"type": "integer", "default": 0}},
      "shape_constraints": ["Input rank >= 2; triangular masking uses the last two dimensions."]
    },
    "uniform": {
      "kind": "kernal",
      "inputs": [],
      "outputs": [{"name": "output", "shape": ["...config.shape"]}],
      "config": {
        "shape": {"type": "array"},
        "dtype": {"type": "string", "default": "float32"},
        "low": {"type": "number", "default": 0},
        "high": {"type": "number", "default": 1}
      }
    },
    "unsqueeze": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["...before", "...after"]}],
      "outputs": [{"name": "output", "shape": ["...before", 1, "...after"]}],
      "config": {"dim": {"type": "integer"}},
      "shape_constraints": ["The new size-one dimension is inserted at config.dim; before/after retain the original dimensions."]
    },
    "upsample_nearest2d": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "x", "shape": ["N", "C", "H", "W"]}],
      "outputs": [{"name": "output", "shape": ["N", "C", "H * config.scale_height", "W * config.scale_width"]}],
      "config": {"scale_height": {"type": "integer", "default": 2}, "scale_width": {"type": "integer", "default": 2}}
    },
    "where": {
      "kind": "kernal",
      "source": "Core/Tensor.h",
      "inputs": [{"name": "condition", "shape": ["...C"]}, {"name": "a", "shape": ["...A"]}, {"name": "b", "shape": ["...B"]}],
      "outputs": [{"name": "output", "shape": ["...broadcast_dims"]}],
      "shape_constraints": ["broadcast_dims = broadcast(C, A, B)."]
    },
    "zeros": {
      "kind": "kernal",
      "inputs": [],
      "outputs": [{"name": "output", "shape": ["...config.shape"]}],
      "config": {"shape": {"type": "array"}, "dtype": {"type": "string", "default": "float32"}}
    }
  },
  "imported_graph_kernals": {
    "Identity": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "CumSum": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}],
      "config": {"axis": {"type": "integer", "default": -1}},
      "shape_constraints": ["config.axis addresses an existing input axis."]
    },
    "ArgSort": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}],
      "config": {"axis": {"type": "integer", "default": -1}, "descending": {"type": "boolean", "default": false}},
      "shape_constraints": ["config.axis addresses an existing input axis."]
    },
    "IndexSelect": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...before", "D", "...after"]}, {"name": "input1", "shape": ["I"]}],
      "outputs": [{"name": "output", "shape": ["...before", "I", "...after"]}],
      "config": {"axis": {"type": "integer", "default": 0}},
      "shape_constraints": ["before/after are the dimensions before/after the normalized config.axis axis; D is its extent."]
    },
    "Slice": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...before", "D", "...after"]}],
      "outputs": [{"name": "output", "shape": ["...before", "L", "...after"]}],
)ELMANCATALOG",
R"ELMANCATALOG(      "config": {
        "axis": {"type": "integer", "default": 0},
        "step": {"type": "integer", "default": 1},
        "start": {"type": "integer"},
        "end": {"type": "integer"}
      },
      "shape_constraints": [
        "before/after are the dimensions before/after the normalized config.axis axis; D is its extent.",
        "L is the nonnegative number of indices selected by config.start, config.end and config.step after native index normalization."
      ]
    },
    "ReduceSum": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...before", "D", "...after"], "shape_alternatives": [[]]}],
      "outputs": [{"name": "output", "shape": ["...before", "...after"], "shape_alternatives": [["...before", 1, "...after"]]}],
      "config": {"axis": {"type": "integer", "default": -1}, "keepdims": {"type": "boolean", "default": false}},
      "shape_constraints": [
        "before/after surround the normalized config.axis axis.",
        "The primary output shape applies when config.keepdims is false; the alternative retains a size-one axis when true.",
        "A scalar input [] remains [] for config.axis = 0 or -1, including when retaining axes."
      ]
    },
    "ReduceMean": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...before", "D", "...after"], "shape_alternatives": [[]]}],
      "outputs": [{"name": "output", "shape": ["...before", "...after"], "shape_alternatives": [["...before", 1, "...after"]]}],
      "config": {"axis": {"type": "integer", "default": -1}, "keepdims": {"type": "boolean", "default": false}},
      "shape_constraints": [
        "before/after surround the normalized config.axis axis.",
        "The primary output shape applies when config.keepdims is false; the alternative retains a size-one axis when true.",
        "A scalar input [] remains [] for config.axis = 0 or -1, including when retaining axes."
      ]
    },
    "MeanAll": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": []}]
    },
    "SumAll": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": []}]
    },
    "MSELoss": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...dims"]}, {"name": "input1", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": []}],
      "shape_constraints": [
        "Prediction and input1 shapes must match.",
        "Low-precision storage types also permit broadcast-compatible input0 and input1 shapes."
      ]
    },
    "Relu": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "Exp": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "Sqrt": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "Tanh": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "Abs": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "Silu": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "Gelu": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "ExactGelu": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "Log": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "Rsqrt": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "Minimum": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...A"]}, {"name": "input1", "shape": ["...B"]}],
      "outputs": [{"name": "output", "shape": ["...broadcast_dims"]}],
      "shape_constraints": ["broadcast_dims = broadcast(A, B)."]
    },
    "Maximum": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...A"]}, {"name": "input1", "shape": ["...B"]}],
      "outputs": [{"name": "output", "shape": ["...broadcast_dims"]}],
      "shape_constraints": ["broadcast_dims = broadcast(A, B)."]
    },
    "Square": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "Neg": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "LeakyRelu": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}],
      "config": {"alpha": {"type": "number"}}
    },
    "Clip": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}],
      "config": {"min": {"type": "number"}, "max": {"type": "number"}}
    },
    "Add": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...A"]}, {"name": "input1", "shape": ["...B"]}],
      "outputs": [{"name": "output", "shape": ["...broadcast_dims"]}],
      "shape_constraints": ["broadcast_dims = broadcast(A, B)."]
    },
    "Sub": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...A"]}, {"name": "input1", "shape": ["...B"]}],
      "outputs": [{"name": "output", "shape": ["...broadcast_dims"]}],
      "shape_constraints": ["broadcast_dims = broadcast(A, B)."]
    },
    "Mul": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...A"]}, {"name": "input1", "shape": ["...B"]}],
      "outputs": [{"name": "output", "shape": ["...broadcast_dims"]}],
      "shape_constraints": ["broadcast_dims = broadcast(A, B)."]
    },
    "Div": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...A"]}, {"name": "input1", "shape": ["...B"]}],
      "outputs": [{"name": "output", "shape": ["...broadcast_dims"]}],
      "shape_constraints": ["broadcast_dims = broadcast(A, B)."]
    },
    "TrueDivide": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...A"]}, {"name": "input1", "shape": ["...B"]}],
      "outputs": [{"name": "output", "shape": ["...broadcast_dims"]}],
      "shape_constraints": ["broadcast_dims = broadcast(A, B)."]
    },
    "MatMul": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [
        {
          "name": "input0",
          "shape": ["...batch_a", "config.transA ? N : M", "config.transA ? M : N"],
          "shape_alternatives": [["N"]]
        },
        {
          "name": "input1",
          "shape": ["...batch_b", "config.transB ? K : N", "config.transB ? N : K"],
)ELMANCATALOG",
R"ELMANCATALOG(          "shape_alternatives": [["N"]]
        }
      ],
      "outputs": [
        {
          "name": "output",
          "shape": ["...batch", "M", "K"],
          "shape_alternatives": [["...batch_b", "K"], ["...batch_a", "M"], []]
        }
      ],
      "config": {"adjoint": {"type": "boolean", "default": false}},
      "shape_constraints": [
        "batch = broadcast(batch_a, batch_b); empty batch prefixes give [M,N] x [N,K] -> [M,K].",
        "transA/transB default to false and swap the final two axes of the corresponding input. Adjoint changes values only.",
        "With both transpose flags false: [N] x [...batch_b,N,K] -> [...batch_b,K]; [...batch_a,M,N] x [N] -> [...batch_a,M]; [N] x [N] -> []."
      ]
    },
    "Eq": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...A"]}, {"name": "input1", "shape": ["...B"]}],
      "outputs": [{"name": "output", "shape": ["...broadcast_dims"]}],
      "shape_constraints": ["broadcast_dims = broadcast(A, B)."]
    },
    "Ne": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...A"]}, {"name": "input1", "shape": ["...B"]}],
      "outputs": [{"name": "output", "shape": ["...broadcast_dims"]}],
      "shape_constraints": ["broadcast_dims = broadcast(A, B)."]
    },
    "Gt": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...A"]}, {"name": "input1", "shape": ["...B"]}],
      "outputs": [{"name": "output", "shape": ["...broadcast_dims"]}],
      "shape_constraints": ["broadcast_dims = broadcast(A, B)."]
    },
    "Ge": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...A"]}, {"name": "input1", "shape": ["...B"]}],
      "outputs": [{"name": "output", "shape": ["...broadcast_dims"]}],
      "shape_constraints": ["broadcast_dims = broadcast(A, B)."]
    },
    "Lt": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...A"]}, {"name": "input1", "shape": ["...B"]}],
      "outputs": [{"name": "output", "shape": ["...broadcast_dims"]}],
      "shape_constraints": ["broadcast_dims = broadcast(A, B)."]
    },
    "Le": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...A"]}, {"name": "input1", "shape": ["...B"]}],
      "outputs": [{"name": "output", "shape": ["...broadcast_dims"]}],
      "shape_constraints": ["broadcast_dims = broadcast(A, B)."]
    },
    "ScalarBinary": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}],
      "config": {
        "boolean": {"type": "boolean", "default": false},
        "reverse": {"type": "boolean", "default": false},
        "integer": {"type": "string"},
        "value": {"type": "number"},
        "op": {"type": "string"}
      }
    },
    "Sigmoid": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}]
    },
    "Gemm": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [
        {"name": "input0", "shape": ["config.transA ? N : M", "config.transA ? M : N"]},
        {"name": "input1", "shape": ["config.transB ? K : N", "config.transB ? N : K"]},
        {"name": "input2", "shape": ["...bias"], "default": null}
      ],
      "outputs": [{"name": "output", "shape": ["M", "K"]}],
      "config": {
        "transA": {"type": "integer", "default": 0},
        "transB": {"type": "integer", "default": 0},
        "alpha": {"type": "number", "default": 1},
        "beta": {"type": "number", "default": 1}
      },
      "shape_constraints": ["Without transposition: [M,N] x [N,K] -> [M,K]; input2 broadcasts to [M,K]."]
    },
    "Reshape": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...dims"]}, {"name": "input1", "shape": ["rank_out"], "default": null}],
      "outputs": [{"name": "output", "shape": ["...target_dims"]}],
      "config": {"allowzero": {"type": "integer", "default": 0}, "shape": {"type": "array", "items": "integer", "default": []}},
      "shape_constraints": [
        "target_dims comes from input1 when supplied, otherwise config.shape. A zero copies the matching input axis unless config.allowzero is nonzero; at most one -1 is inferred from the unchanged element count."
      ]
    },
    "Flatten": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["product(input0.shape[:axis])", "product(input0.shape[axis:])"]}],
      "config": {"axis": {"type": "integer", "default": 1}}
    },
    "FlattenRange": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...before", "...collapsed", "...after"]}],
      "outputs": [{"name": "output", "shape": ["...before", "product(collapsed)", "...after"]}],
      "config": {
        "start": {"type": "integer", "default": 0},
        "end": {"type": "integer", "default": -1},
        "axis": {"type": "integer", "default": 1}
      },
      "shape_constraints": ["collapsed spans the normalized config.start through config.end axes, inclusive."]
    },
    "SwapAxes": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...transposed_dims"]}],
      "config": {"a": {"type": "integer"}, "b": {"type": "integer"}},
      "shape_constraints": ["transposed_dims equals dims with dimensions at config.a and config.b exchanged."]
    },
    "Size": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": []}],
      "config": {"axis": {"type": "integer"}}
    },
    "Rank": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": []}]
    },
    "AssertFalse": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": []}],
      "outputs": [{"name": "output", "shape": []}]
    },
    "Transpose": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...dims"]}, {"name": "input1", "shape": ["rank(input0)"], "default": null}],
      "outputs": [{"name": "output", "shape": ["...permuted_dims"]}],
      "config": {"perm": {"type": "array", "items": "integer", "default": []}},
      "shape_constraints": [
        "permuted_dims[i] = dims[axes[i]], with axes from input1 when supplied, otherwise config.perm, otherwise reversed axis order."
      ]
    },
    "Gather": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...before", "D", "...after"]}, {"name": "input1", "shape": ["...index_dims"]}],
      "outputs": [{"name": "output", "shape": ["...before", "...index_dims", "...after"]}],
      "config": {"axis": {"type": "integer", "default": 0}},
      "shape_constraints": ["before/after are the dimensions before/after the normalized config.axis axis; D is its extent."]
    },
    "Softmax": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}],
      "config": {"beta": {"type": "number", "default": 1}, "axis": {"type": "integer", "default": -1}}
    },
    "LogSoftmax": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}],
      "config": {"axis": {"type": "integer", "default": -1}},
      "shape_constraints": ["config.axis addresses an existing input axis."]
    },
    "Concat": {
)ELMANCATALOG",
R"ELMANCATALOG(      "source": "Inference/GraphSession.cpp",
      "max_inputs": 1024,
      "inputs": [{"name": "inputs", "variadic": true, "shape": ["...before", "D_i", "...after"]}],
      "variadic": true,
      "outputs": [{"name": "output", "shape": ["...before", "sum(D_i)", "...after"]}],
      "config": {"axis": {"type": "integer"}},
      "shape_constraints": [
        "Concat inputs match outside axis; Stack inputs have identical shapes.",
        "D_i is input i extent at config.axis; all before/after dimensions match across inputs."
      ]
    },
    "Mean": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...dims"]}, {"name": "input1", "shape": ["axes_count"]}],
      "outputs": [{"name": "output", "shape": ["...reduced_dims"]}],
      "config": {"keep_dims": {"type": "boolean", "default": false}},
      "shape_constraints": ["reduced_dims equals dims with axes listed in input1 removed, or set to 1 when config.keep_dims is true."]
    },
    "Sum": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...dims"]}, {"name": "input1", "shape": ["axes_count"]}],
      "outputs": [{"name": "output", "shape": ["...reduced_dims"]}],
      "config": {"keep_dims": {"type": "boolean", "default": false}},
      "shape_constraints": ["reduced_dims equals dims with axes listed in input1 removed, or set to 1 when config.keep_dims is true."]
    },
    "ReduceMax": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...dims"]}, {"name": "input1", "shape": ["axes_count"]}],
      "outputs": [{"name": "output", "shape": ["...reduced_dims"]}],
      "config": {"keep_dims": {"type": "boolean", "default": false}},
      "shape_constraints": ["reduced_dims equals dims with axes listed in input1 removed, or set to 1 when config.keep_dims is true."]
    },
    "ExpandDims": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...before", "...after"]}, {"name": "input1", "shape": [], "default": null}],
      "outputs": [{"name": "output", "shape": ["...before", 1, "...after"]}],
      "config": {"axis": {"type": "integer"}},
      "shape_constraints": [
        "The new size-one dimension is inserted at input1 when supplied, otherwise config.axis; before/after retain the original dimensions."
      ]
    },
    "Squeeze": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...remaining_dims"]}],
      "config": {"strict": {"type": "boolean", "default": true}, "axes": {"type": "array", "items": "integer", "default": []}},
      "shape_constraints": [
        "remaining_dims preserves dims order after removing size-one axes selected by config.axes, or all size-one axes when config.axes is absent."
      ]
    },
    "Stack": {
      "source": "Inference/GraphSession.cpp",
      "max_inputs": 1024,
      "inputs": [{"name": "inputs", "variadic": true, "shape": ["...before", "...after"]}],
      "variadic": true,
      "outputs": [{"name": "output", "shape": ["...before", "count(inputs)", "...after"]}],
      "config": {"axis": {"type": "integer", "default": 0}},
      "shape_constraints": [
        "Concat inputs match outside axis; Stack inputs have identical shapes.",
        "All inputs have equal dimensions; insert the input count at config.axis."
      ]
    },
    "StridedSlice": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [
        {"name": "input0", "shape": ["...dims"]},
        {"name": "input1", "shape": ["axes_count"]},
        {"name": "input2", "shape": ["axes_count"]},
        {"name": "input3", "shape": ["axes_count"]}
      ],
      "outputs": [{"name": "output", "shape": ["...sliced_dims"]}],
      "config": {
        "new_axis_mask": {"type": "integer", "default": 0},
        "ellipsis_mask": {"type": "integer", "default": 0},
        "begin_mask": {"type": "integer", "default": 0},
        "end_mask": {"type": "integer", "default": 0},
        "shrink_axis_mask": {"type": "integer", "default": 0}
      },
      "shape_constraints": [
        "Shrink-axis masks remove selected axes.",
        "sliced_dims contains each axis slice length from input1/input2/input3 (begin/end/stride), accounting for begin/end masks and omitting shrink_axis_mask axes; new_axis_mask and ellipsis_mask must be zero."
      ]
    },
    "Pad": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...dims"]}, {"name": "input1", "shape": ["2 * rank(input0)"], "default": null}],
      "outputs": [{"name": "output", "shape": ["...padded_dims"]}],
      "config": {"fill": {"type": "number", "default": 0}, "pads": {"type": "array", "items": "integer", "default": []}},
      "shape_constraints": [
        "padded_dims[i] = dims[i] + pads[2*i] + pads[2*i+1]; pads comes from input1 when supplied, otherwise config.pads."
      ]
    },
    "AdaptiveAvgPool2d": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...NC", "H", "W"]}],
      "outputs": [{"name": "output", "shape": ["...NC", "config.height", "config.width"]}],
      "config": {"height": {"type": "integer"}, "width": {"type": "integer"}}
    },
    "ChannelsLast": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["N", "C", "...spatial"]}],
      "outputs": [{"name": "output", "shape": ["N", "...spatial", "C"]}],
      "shape_constraints": ["Ranks below 3 are unchanged."]
    },
    "BatchReshape": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["B", "...dims"]}],
      "outputs": [{"name": "output", "shape": ["B", "...target_dims"]}],
      "config": {"shape": {"type": "array", "items": "integer", "default": []}},
      "shape_constraints": ["target_dims resolves config.shape with the batch dimension retained and the element count unchanged."]
    },
    "Cast": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["...dims"]}],
      "config": {"to": {"type": "string"}}
    },
    "Shape": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["...dims"]}],
      "outputs": [{"name": "output", "shape": ["normalized_end - normalized_start"]}],
      "config": {
        "start": {"type": "integer", "default": 0},
        "end": {"type": "integer", "default_expression": "rank"},
        "to": {"type": "string"}
      },
      "shape_constraints": ["start/end select dimensions from shape(input0)."]
    },
    "ChannelBias": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["N", "C", "...spatial"]}, {"name": "input1", "shape": ["C"]}],
      "outputs": [{"name": "output", "shape": ["N", "C", "...spatial"]}]
    },
    "DepthwiseFilter": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["H", "W", "Cin", "multiplier"]}],
      "outputs": [{"name": "output", "shape": [1, "H", "W", "Cin * multiplier"]}]
    },
    "Conv": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [
        {"name": "input0", "shape": ["N", "Cin", "H", "W"]},
        {"name": "input1", "shape": ["Cout", "Cin", "Kh", "Kw"]},
        {"name": "input2", "shape": ["Cout"], "default": null}
      ],
      "outputs": [
        {
          "name": "output",
          "shape": ["N", "Cout", "floor((H + pads[0]+pads[2]-Kh)/strides[0])+1", "floor((W + pads[1]+pads[3]-Kw)/strides[1])+1"]
        }
      ],
      "config": {
        "group": {"type": "integer", "default": 1},
        "auto_pad": {"type": "string", "default": "NOTSET"},
        "strides": {"type": "array", "items": "integer", "default": []},
        "pads": {"type": "array", "items": "integer", "default": []},
)ELMANCATALOG",
R"ELMANCATALOG(        "dilations": {"type": "array", "items": "integer", "default": []}
      }
    },
    "NhwcConv": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [
        {"name": "input0", "shape": ["N", "H", "W", "Cin"]},
        {"name": "input1", "shape": ["Cout", "Kh", "Kw", "Cin/group"]},
        {"name": "input2", "shape": ["Cout"], "default": null}
      ],
      "outputs": [{"name": "output", "shape": ["N", "Hout", "Wout", "Cout"]}],
      "config": {
        "depthwise": {"type": "boolean", "default": false},
        "group": {"type": "integer", "default": 1},
        "auto_pad": {"type": "string", "default": "NOTSET"},
        "strides": {"type": "array", "items": "integer", "default": []},
        "pads": {"type": "array", "items": "integer", "default": []},
        "dilations": {"type": "array", "items": "integer", "default": []}
      },
      "shape_constraints": [
        "SAME: Hout=ceil(H/stride_h), Wout=ceil(W/stride_w); VALID: floor((H-Kh)/stride_h)+1, floor((W-Kw)/stride_w)+1.",
        "Depthwise filters use [1,Kh,Kw,Cout], Cout is a multiple of Cin."
      ]
    },
    "NhwcPool": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["N", "H", "W", "C"]}],
      "outputs": [{"name": "output", "shape": ["N", "Hout", "Wout", "C"]}],
      "config": {"kernel": {"type": "integer"}, "maximum": {"type": "boolean"}},
      "shape_constraints": ["SAME rounds H/stride_h and W/stride_w up; VALID uses floor((extent-kernel)/stride)+1."]
    },
    "FullyConnected": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [
        {"name": "input0", "shape": ["...dims"]},
        {"name": "input1", "shape": ["O", "I"]},
        {"name": "input2", "shape": ["O"], "default": null}
      ],
      "outputs": [{"name": "output", "shape": ["product(dims) / I", "O"], "shape_alternatives": [["...batch", "O"]]}],
      "config": {"keep_dims": {"type": "boolean", "default": false}},
      "shape_constraints": [
        "The primary output shape applies when config.keep_dims is false. When true, input0 must have shape [...batch,I] and output has shape [...batch,O]."
      ]
    },
    "MaxPool": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["N", "C", "H", "W"]}],
      "outputs": [
        {
          "name": "output",
          "shape": [
            "N",
            "C",
            "floor((H+pads[0]+pads[2]-kernel_shape[0])/strides[0])+1",
            "floor((W+pads[1]+pads[3]-kernel_shape[1])/strides[1])+1"
          ]
        }
      ],
      "config": {
        "ceil_mode": {"type": "integer", "default": 0},
        "count_include_pad": {"type": "integer", "default": 0},
        "auto_pad": {"type": "string", "default": "NOTSET"},
        "kernel_shape": {"type": "array", "items": "integer", "default": []},
        "strides": {"type": "array", "items": "integer", "default": []},
        "pads": {"type": "array", "items": "integer", "default": []},
        "dilations": {"type": "array", "items": "integer", "default": []}
      }
    },
    "AveragePool": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["N", "C", "H", "W"]}],
      "outputs": [
        {
          "name": "output",
          "shape": [
            "N",
            "C",
            "floor((H+pads[0]+pads[2]-kernel_shape[0])/strides[0])+1",
            "floor((W+pads[1]+pads[3]-kernel_shape[1])/strides[1])+1"
          ]
        }
      ],
      "config": {
        "ceil_mode": {"type": "integer", "default": 0},
        "count_include_pad": {"type": "integer", "default": 0},
        "auto_pad": {"type": "string", "default": "NOTSET"},
        "kernel_shape": {"type": "array", "items": "integer", "default": []},
        "strides": {"type": "array", "items": "integer", "default": []},
        "pads": {"type": "array", "items": "integer", "default": []},
        "dilations": {"type": "array", "items": "integer", "default": []}
      }
    },
    "GlobalAveragePool": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [{"name": "input0", "shape": ["N", "C", "...spatial"]}],
      "outputs": [{"name": "output", "shape": ["N", "C", "...ones_like_spatial"]}]
    },
    "BatchNormalization": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [
        {"name": "input0", "shape": ["N", "C", "...spatial"]},
        {"name": "input1", "shape": ["C"]},
        {"name": "input2", "shape": ["C"]},
        {"name": "input3", "shape": ["C"]},
        {"name": "input4", "shape": ["C"]}
      ],
      "outputs": [{"name": "output", "shape": ["N", "C", "...spatial"]}],
      "config": {"training_mode": {"type": "integer", "default": 0}, "epsilon": {"type": "number", "default": 1e-05}}
    },
    "AffineBatchNormalization": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [
        {"name": "input0", "shape": ["...dims"]},
        {"name": "input1", "shape": ["C"]},
        {"name": "input2", "shape": ["C"]},
        {"name": "input3", "shape": ["C"], "default": null},
        {"name": "input4", "shape": ["C"], "default": null}
      ],
      "outputs": [{"name": "output", "shape": ["...dims"]}],
      "config": {"axis": {"type": "integer", "default": -1}, "epsilon": {"type": "number", "default": 0.001}},
      "shape_constraints": ["C == input0.shape[config.axis]."]
    },
    "LayerNormalization": {
      "source": "Inference/GraphSession.cpp",
      "inputs": [
        {"name": "input0", "shape": ["...batch", "D"]},
        {"name": "input1", "shape": ["D"], "default": null},
        {"name": "input2", "shape": ["D"], "default": null}
      ],
      "outputs": [{"name": "output", "shape": ["...batch", "D"]}],
      "config": {"epsilon": {"type": "number", "default": 0.001}}
    }
  },
  "native_interfaces": {
    "Core/Quantization.h": [
      {
        "name": "quantize",
        "inputs": [
          {"name": "input", "shape": ["...dims"]},
          {"name": "storage_dtype", "cpp_type": "DType", "shape": null},
          {"name": "scales", "shape": ["R", "ceil(D/group_size)"]},
          {"name": "zero_points", "shape": ["R", "ceil(D/group_size)"]},
          {"name": "group_size", "cpp_type": "int", "shape": []}
        ],
        "outputs": [{"name": "result", "cpp_type": "QuantizedTensor", "shape": ["...dims"]}],
        "binding": "native_cpp",
        "shape_constraints": [
          "For quantized values, shape is the logical codes shape. Metadata broadcasts to [R,ceil(D/group_size)] with rank <= 2, R=product(shape[:-1]), D=shape[-1]. Scalars use R=D=1."
        ]
      },
      {
        "name": "concat_rows",
        "inputs": [{"name": "values", "cpp_type": "const std::vector<QuantizedTensor>&", "shape": ["R_i", "D"]}],
        "outputs": [{"name": "result", "cpp_type": "QuantizedTensor", "shape": ["sum(R_i)", "D"]}],
        "binding": "native_cpp",
        "shape_constraints": [
          "For quantized values, shape is the logical codes shape. Metadata broadcasts to [R,ceil(D/group_size)] with rank <= 2, R=product(shape[:-1]), D=shape[-1]. Scalars use R=D=1.",
          "Each value is a matrix [R_i,D]; rows are concatenated and D matches across values."
        ]
      },
      {
        "name": "dequantize",
        "inputs": [{"name": "dtype", "cpp_type": "DType", "default_expression": "DType::Float32", "shape": null}],
        "outputs": [{"name": "result", "shape": ["...self.shape"]}],
        "binding": "native_cpp",
        "shape_constraints": [
          "For quantized values, shape is the logical codes shape. Metadata broadcasts to [R,ceil(D/group_size)] with rank <= 2, R=product(shape[:-1]), D=shape[-1]. Scalars use R=D=1."
        ]
      },
      {
        "name": "embedding",
        "inputs": [{"name": "indices", "shape": ["...ids"]}, {"name": "dtype", "cpp_type": "DType", "shape": null}],
        "outputs": [{"name": "result", "shape": ["...ids", "self.shape[-1]"]}],
)ELMANCATALOG",
R"ELMANCATALOG(        "binding": "native_cpp",
        "shape_constraints": [
          "For quantized values, shape is the logical codes shape. Metadata broadcasts to [R,ceil(D/group_size)] with rank <= 2, R=product(shape[:-1]), D=shape[-1]. Scalars use R=D=1."
        ]
      },
      {
        "name": "linear",
        "inputs": [
          {"name": "input", "shape": ["...batch", "self.shape[1]"]},
          {"name": "bias", "default": null, "shape": ["O"]},
          {"name": "workspace", "default": null, "shape": ["self.shape[0]", "self.shape[1]"]}
        ],
        "outputs": [{"name": "result", "shape": ["...batch", "self.shape[0]"]}],
        "binding": "native_cpp",
        "shape_constraints": [
          "For quantized values, shape is the logical codes shape. Metadata broadcasts to [R,ceil(D/group_size)] with rank <= 2, R=product(shape[:-1]), D=shape[-1]. Scalars use R=D=1.",
          "O = self.shape[0]; workspace, when supplied, holds the dequantized weight matrix."
        ]
      },
      {
        "name": "packed_linear",
        "inputs": [
          {"name": "input", "shape": ["...batch", "self.shape[1]"]},
          {"name": "widths", "cpp_type": "const std::vector<int>&", "shape": ["projection_count"]},
          {"name": "bias", "default": null, "shape": ["O"]},
          {"name": "workspace", "default": null, "shape": ["self.shape[0]", "self.shape[1]"]}
        ],
        "outputs": [{"name": "result", "shape": ["product(input.shape[:-1]) * sum(widths)"]}],
        "binding": "native_cpp",
        "shape_constraints": [
          "For quantized values, shape is the logical codes shape. Metadata broadcasts to [R,ceil(D/group_size)] with rank <= 2, R=product(shape[:-1]), D=shape[-1]. Scalars use R=D=1.",
          "O = self.shape[0]; workspace, when supplied, holds the dequantized weight matrix."
        ]
      },
      {
        "name": "quantized_add",
        "inputs": [
          {"name": "a", "cpp_type": "const QuantizedTensor&", "shape": ["...A"]},
          {"name": "b", "cpp_type": "const QuantizedTensor&", "shape": ["...B"]},
          {"name": "output_dtype", "cpp_type": "DType", "shape": null},
          {"name": "scales", "shape": ["R", "ceil(D/group_size)"]},
          {"name": "zero_points", "shape": ["R", "ceil(D/group_size)"]},
          {"name": "group_size", "cpp_type": "int", "shape": []}
        ],
        "outputs": [{"name": "result", "cpp_type": "QuantizedTensor", "shape": ["...broadcast_dims"]}],
        "binding": "native_cpp",
        "shape_constraints": [
          "For quantized values, shape is the logical codes shape. Metadata broadcasts to [R,ceil(D/group_size)] with rank <= 2, R=product(shape[:-1]), D=shape[-1]. Scalars use R=D=1.",
          "broadcast_dims = broadcast(A, B)."
        ]
      },
      {
        "name": "quantized_mul",
        "inputs": [
          {"name": "a", "cpp_type": "const QuantizedTensor&", "shape": ["...A"]},
          {"name": "b", "cpp_type": "const QuantizedTensor&", "shape": ["...B"]},
          {"name": "output_dtype", "cpp_type": "DType", "shape": null},
          {"name": "scales", "shape": ["R", "ceil(D/group_size)"]},
          {"name": "zero_points", "shape": ["R", "ceil(D/group_size)"]},
          {"name": "group_size", "cpp_type": "int", "shape": []}
        ],
        "outputs": [{"name": "result", "cpp_type": "QuantizedTensor", "shape": ["...broadcast_dims"]}],
        "binding": "native_cpp",
        "shape_constraints": [
          "For quantized values, shape is the logical codes shape. Metadata broadcasts to [R,ceil(D/group_size)] with rank <= 2, R=product(shape[:-1]), D=shape[-1]. Scalars use R=D=1.",
          "broadcast_dims = broadcast(A, B)."
        ]
      },
      {
        "name": "quantized_relu",
        "inputs": [{"name": "input", "cpp_type": "const QuantizedTensor&", "shape": ["...dims"]}],
        "outputs": [{"name": "result", "cpp_type": "QuantizedTensor", "shape": ["...dims"]}],
        "binding": "native_cpp",
        "shape_constraints": [
          "For quantized values, shape is the logical codes shape. Metadata broadcasts to [R,ceil(D/group_size)] with rank <= 2, R=product(shape[:-1]), D=shape[-1]. Scalars use R=D=1."
        ]
      }
    ],
    "Inference/LLM/PagedAttention.h": [
      {
        "name": "paged_attention",
        "inputs": [
          {"name": "query", "shape": ["B", "Hq", "Q", "D"]},
          {"name": "keys", "shape": ["physical_slots", "kv_heads * D"]},
          {"name": "values", "shape": ["physical_slots", "kv_heads * D"]},
          {"name": "map", "cpp_type": "const PagedAttentionMap&", "shape": null},
          {"name": "kv_heads", "cpp_type": "int", "shape": []},
          {"name": "divisor", "cpp_type": "float", "shape": []},
          {"name": "query_tile", "cpp_type": "int", "default": 32, "shape": []},
          {"name": "causal", "cpp_type": "bool", "default": true, "shape": []},
          {"name": "causal_offset", "cpp_type": "int", "default": 0, "shape": []}
        ],
        "outputs": [{"name": "result", "shape": ["B", "Hq", "Q", "D"]}],
        "binding": "native_cpp",
        "shape_constraints": ["map supplies logical-to-physical slots; Hq % kv_heads == 0."]
      }
    ],
    "Inference/LLM/Sampling.h": [
      {
        "name": "greedy_token",
        "inputs": [{"name": "logits", "shape": ["V"]}],
        "outputs": [{"name": "result", "cpp_type": "int32_t", "shape": []}],
        "binding": "native_cpp"
      },
      {
        "name": "sampling_scores",
        "inputs": [
          {"name": "logits", "shape": ["B", "V"]},
          {"name": "history", "shape": ["B", "history_length"]},
          {"name": "options", "cpp_type": "const SamplingOptions&", "shape": null}
        ],
        "outputs": [{"name": "result", "shape": ["B", "V"]}],
        "binding": "native_cpp"
      },
      {
        "name": "categorical_sample",
        "inputs": [
          {"name": "weights", "shape": ["B", "V"]},
          {"name": "uniforms", "cpp_type": "const std::vector<float>&", "shape": ["B"]}
        ],
        "outputs": [{"name": "result", "cpp_type": "std::vector<int32_t>", "shape": ["B"]}],
        "binding": "native_cpp"
      },
      {
        "name": "categorical_sample_double",
        "inputs": [
          {"name": "weights", "shape": ["B", "V"]},
          {"name": "uniforms", "cpp_type": "const std::vector<double>&", "shape": ["B"]}
        ],
        "outputs": [{"name": "result", "cpp_type": "std::vector<int32_t>", "shape": ["B"]}],
        "binding": "native_cpp"
      },
      {
        "name": "sample_tokens",
        "inputs": [
          {"name": "logits", "shape": ["B", "V"]},
          {"name": "history", "shape": ["B", "history_length"]},
          {"name": "options", "cpp_type": "const SamplingOptions&", "shape": null},
          {"name": "rng", "cpp_type": "SamplingRng&", "shape": null}
        ],
        "outputs": [{"name": "result", "cpp_type": "std::vector<int32_t>", "shape": ["B"]}],
        "binding": "native_cpp"
      }
    ],
    "Layers/Attention.h": [
      {
        "name": "operator()",
        "inputs": [
          {"name": "hidden", "shape": ["B", "Q", "query_width"]},
          {"name": "context", "shape": ["B", "K", "context_width"]},
          {"name": "causal", "cpp_type": "bool", "default": false, "shape": []}
        ],
        "outputs": [{"name": "result", "shape": ["B", "Q", "self.output.weight.shape[0]"]}],
        "binding": "native_cpp"
      }
    ],
    "Layers/Linear.h": [
      {
        "name": "operator()",
        "inputs": [{"name": "value", "shape": ["...batch", "self.weight.shape[1]"]}],
        "outputs": [{"name": "result", "shape": ["...batch", "self.weight.shape[0]"]}],
        "binding": "native_cpp"
      }
    ],
    "Layers/Conv2d.h": [
      {
        "name": "operator()",
        "inputs": [{"name": "value", "shape": ["N", "Cin", "H", "W"]}],
        "outputs": [
          {
            "name": "result",
            "shape": [
)ELMANCATALOG",
R"ELMANCATALOG(              "N",
              "self.weight.shape[0]",
              "floor((H+2*self.padding-self.weight.shape[2])/self.stride)+1",
              "floor((W+2*self.padding-self.weight.shape[3])/self.stride)+1"
            ]
          }
        ],
        "binding": "native_cpp"
      }
    ],
    "Layers/GroupNorm.h": [
      {
        "name": "operator()",
        "inputs": [{"name": "value", "shape": ["N", "C", "...spatial"]}],
        "outputs": [{"name": "result", "shape": ["N", "C", "...spatial"]}],
        "binding": "native_cpp"
      }
    ],
    "Layers/LayerNorm.h": [
      {
        "name": "operator()",
        "inputs": [{"name": "value", "shape": ["...batch", "D"]}],
        "outputs": [{"name": "result", "shape": ["...batch", "D"]}],
        "binding": "native_cpp"
      }
    ],
    "Kernals/audio_ops.h": [
      {
        "name": "conv1d",
        "inputs": [
          {"name": "input", "shape": ["N", "Cin", "L"]},
          {"name": "weight", "shape": ["Cout", "Cin", "K"]},
          {"name": "bias", "default": null, "shape": ["Cout"]},
          {"name": "stride", "cpp_type": "int", "default": 1, "shape": []},
          {"name": "padding", "cpp_type": "int", "default": 0, "shape": []},
          {"name": "dilation", "cpp_type": "int", "default": 1, "shape": []}
        ],
        "outputs": [{"name": "result", "shape": ["N", "Cout", "floor((L + 2*padding - dilation*(K-1) - 1)/stride) + 1"]}],
        "binding": "native_cpp"
      },
      {
        "name": "operator()",
        "inputs": [{"name": "samples", "shape": ["sample_count"]}],
        "outputs": [{"name": "result", "shape": ["self.mel_bins", "floor(sample_count / self.hop_length)"]}],
        "binding": "native_cpp"
      }
    ]
  },
  "shape_notation": {
    "purpose": "Dimension contracts for reading kernal signatures; symbolic dimensions and constraints are not evaluated by the loader. Native kernels and model ports validate runtime shapes.",
    "array": "An ordered list of axis sizes. [M,N] is a rank-two matrix; [] is a scalar. Tensor port shapes are always arrays.",
    "symbols": "Symbols are local to one kernal signature. Reusing a symbol requires equal sizes: [M,N] x [N,K] -> [M,K]. Numbers specify fixed sizes.",
    "variadic_dimensions": "...name expands an ordered list of zero or more dimensions, at any position. [...batch,M,N] reduces to [M,N] when batch is empty.",
    "dimension_expression": "An array element may calculate one axis size, such as H*W or config.transA ? N : M. ...config.shape expands a configured dimension list.",
    "broadcast": "broadcast(A,B,...) right-aligns dimension lists, treating missing axes as 1. Aligned extents must match or be 1; the result takes the non-one extent, including 0.",
    "constraints": "shape_constraints defines derived dimension lists and conditions for shape_alternatives; alternatives list additional supported dimension lists beyond the primary shape.",
    "configuration": "config.x refers to configuration fields; unqualified geometry names in formulas also refer to config.",
    "optional": "An optional tensor may be omitted; its shape applies when supplied.",
    "variadic_ports": "A variadic descriptor applies to each operand/output; i is its zero-based index. Other symbols remain shared within the signature.",
    "non_tensor": "A null shape denotes a native C++ object or enum without a tensor shape."
  },
  "defaults": {
    "port_type": "Tensor",
    "variadic": false,
    "binding": true,
    "config": {},
    "shape_constraints": [],
    "arguments": "Required unless default or default_expression is present. A null tensor default means omitted.",
    "arity": "Derived from ports and defaults; explicit min/max bounds only record exceptions."
  }
}
)ELMANCATALOG"
