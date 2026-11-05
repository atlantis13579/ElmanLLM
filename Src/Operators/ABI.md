# Scalar kernel ABI

Each CPU and CUDA operator contains its scalar implementation directly in its
native source file. A thread computes one output element `i`. Buffers contain
32-bit float, signed integer, or boolean words. All strides are contiguous
row-major strides.

Parameter words: `0` output element count, `1` output rank, `2..9` output shape;
`10` A rank, `11..18` A shape, `19..26` A strides; `27` B rank,
`28..35` B shape, `36..43` B strides; `44` C rank, `45..52` C shape,
`53..60` C strides. Missing inputs have rank -1. Dtypes are at 252..255 for
A, B, C, output (0 float32, 1 int32, 2 bool32). Float attributes use IEEE float
bits in an integer word. Rank is limited to eight.

These numbers identify slots in the packed CPU/CUDA ABI; tensor dimensions are
read from those slots at runtime. Native operators should use the named
`KernelParams` accessors and operator-specific attribute constants instead of
repeating absolute slot numbers.
Common image and logits axes have named constants in
`Src/Operators/tensor_axes.h`.

## Attributes (offsets are absolute parameter word indices)

| Operator | Attributes |
|---|---|
| Elementwise, comparison, logical | None; standard trailing-axis broadcasting |
| sum, mean, max, min, amax, amin, prod | 64 axis (-1 all axes), 65 keepdim |
| reduce_grad | 64 axis, 65 keepdim, 66 kind (0 sum, 1 mean, 2 max, 3 min, 4 prod); A original input, B reduced output, C upstream gradient |
| reshape, squeeze, unsqueeze | None; contiguous copy |
| permute | 64..71 permutation: output axis to input axis |
| transpose | 64 first axis, 65 second axis |
| slice, split, slice_grad | 64 axis, 65 start, 66 positive step; slice_grad A upstream, output original input shape |
| gather, index_select, scatter, scatter_add | 64 axis; gather/index_select A source, B indices; scatter A base, B indices, C updates; last update wins for scatter |
| gather_grad, index_select_grad | 64 axis; A upstream, B indices, output original source shape; deterministic sum of repeated indices |
| scatter_source_grad | 64 axis; A upstream base shape, B indices, output source shape; only the last repeated update receives a gradient |
| concat | 64 axis; pairwise concatenation of A and B |
| repeat, expand | Output shape defines repetition or broadcast |
| repeat_grad | A repeated upstream gradient, output original input shape; sums all repetitions |
| uniform, normal | 64 seed, 65 stream offset, float 66 low/mean, float 67 high/stddev |
| dropout | 64 seed, 65 stream offset, float 66 probability, 67 training flag |
| cast | Output dtype selects conversion |
| matmul, bmm | Rank >= 2, batch dimensions broadcast; transpose via separate op |
| addmm | A left matrix, B right matrix, C addend; float 64 alpha, float 65 beta |
| triu, tril | 64 diagonal offset |
| leaky_relu | float 64 negative slope |
| softmax, log_softmax | 64 axis |
| layer_norm, rms_norm | 64 number of trailing normalized dimensions, float 65 epsilon; A input, optional B weight, optional C bias |
| batch_norm | float 64 epsilon; A input NCHW, B channel mean, C channel variance; affine transform is composed separately |
| group_norm | 64 group count, float 65 epsilon; A NCHW input, optional B channel weight, optional C channel bias |
| cross_entropy | A logits [N,C], B integer targets [N]; output scalar mean |
| cross_entropy_grad | A logits [N,C], B targets, C scalar upstream; output logits shape |
| mse_loss | A predictions, B same-shape targets; output scalar mean |
| conv2d, conv2d_input_grad, conv2d_weight_grad | 64 stride H, 65 stride W, 66 padding H, 67 padding W, 68 dilation H, 69 dilation W, 70 groups. Forward A NCHW input, B OIHW weights, optional C bias. Input grad A upstream, B weights, output original input. Weight grad A original input, B upstream, output weight shape |
| max_pool2d, avg_pool2d, max_pool2d_grad, avg_pool2d_grad | 64 kernel H, 65 kernel W, 66 stride H, 67 stride W, 68 padding H, 69 padding W, 70 count_include_pad; grad A original input, B upstream, output input shape |

All reduction extrema return values. Index-based operations require validated
in-range indices; runtime kernels also guard invalid accesses. Pooling uses floor
output sizes and no dilation. Softmax and cross entropy subtract the maximum.
Convolution and normalization are straightforward reference kernels intended for
correctness and small-model training; no vendor math libraries are used.
