#pragma once

namespace elman { namespace operator_attr { namespace matmul {
// B is stored as [...,columns,shared] when this optional flag is one.
constexpr int kTransposeB = 0;
} } }
