# Native backends

Each backend owns its buffers and executes the corresponding native operator
files. There is no automatic CPU fallback. A buffer stores 32-bit words for
Float32, Int32, or Bool. Integer conversion truncates toward zero, saturates at
the signed 32-bit limits, and maps NaN to zero. Float-based host transfer APIs
cannot represent every Int32 value exactly. Shape and copy kernels preserve
integer words when the dtype is unchanged.

| Backend | Native dependency | Validation on the development machine |
|---|---|---|
| CPU | C++17 standard library | Numerical operator and differentiation tests |
| CUDA | NVIDIA CUDA Toolkit and driver | RTX 4060; native operator parity, gradients, and full-size AlexNet/GPT2 training and inference |
| Compute | Windows SDK, Direct3D 11, D3DCompiler | Hardware compute dispatch; all native operators compared with CPU |
| Metal | macOS SDK, Foundation and Metal frameworks | Source review only; no Apple compiler or device was available |

Metal remains unverified until the Xcode preset is built and the Metal CTest
targets pass on an Apple machine. Its shader entry point is `elman_main`, with
explicit device and constant pointer address spaces. The implementation was
checked against the [Apple Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf).

Compute and Metal compile and cache shaders at runtime using platform APIs.
CMake packages the required first-party shader files beneath
`Shaders/Src` beside each executable. Runtime lookup prefers this adjacent
bundle and uses the source checkout as a development fallback. The packaged
Compute path was verified with the source fallback set to a nonexistent path.
No external shader compiler executable is required at runtime. CUDA operators
are compiled into the library at build time.

CUDA cross entropy and its derivative use cooperative row reductions, keeping
work linear in vocabulary size for the complete GPT2 vocabulary. Other
implementations use straightforward scalar reference kernels, synchronous
CUDA and Metal dispatch, and ordered D3D11 command submission. They prioritize
correctness and small-model validation over throughput. D3D11 readback waits
for completion through a staging buffer. Dispatch bounds use unsigned indices
before conversion to the shared signed indexing format; tensors support at
most eight dimensions and signed 32-bit element indices. Platform allocation
limits may be lower.
