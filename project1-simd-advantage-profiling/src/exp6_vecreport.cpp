// Experiment 6: vectorization verification target. This file exists only to be
// compiled with -fopt-info-vec-optimized/-missed (and disassembled with
// objdump) for each ISA variant, so there is concrete evidence of what the
// compiler actually did to the kernels used by every other experiment. See
// scripts/verify_vectorization.ps1 (greps the report; never pastes full dumps).
#include "kernels.hpp"

// Explicit instantiation forces code generation for these exact signatures.
template void kernels::saxpy<float>(float*, const float*, float, size_t);
template void kernels::saxpy<double>(double*, const double*, double, size_t);
template void kernels::saxpy<uint32_t>(uint32_t*, const uint32_t*, uint32_t, size_t);
template float kernels::dot<float>(const float*, const float*, size_t);
template uint32_t kernels::dot<uint32_t>(const uint32_t*, const uint32_t*, size_t);
template void kernels::mul<float>(float*, const float*, const float*, size_t);
template void kernels::stencil3<float>(float*, const float*, float, float, float, size_t);
template void kernels::mul_strided_c<float, 2>(float*, const float*, const float*, size_t);
template void kernels::mul_strided_c<float, 4>(float*, const float*, const float*, size_t);
template void kernels::mul_strided_c<float, 8>(float*, const float*, const float*, size_t);
template void kernels::mul_strided_c<float, 16>(float*, const float*, const float*, size_t);
template void kernels::mul_gather<float>(float*, const float*, const float*, const int32_t*, size_t);

int main() { return 0; }
