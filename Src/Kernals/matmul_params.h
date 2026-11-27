#pragma once

namespace elman { namespace kernal_attr { namespace matmul {
// B is stored as [...,columns,shared] when this optional flag is one.
constexpr int kTransposeB = 0;
// Private CUDA backward path: A is stored as [...,shared,rows].
constexpr int kTransposeA = 2;
} } }
