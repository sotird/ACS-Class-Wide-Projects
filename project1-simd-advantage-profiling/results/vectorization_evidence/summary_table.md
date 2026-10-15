| variant | kernel | main loop vector width (bytes) | epilogue handling |
|---|---|---|---|
| scalar | saxpy | NOT VECTORIZED | - |
| scalar | dot | NOT VECTORIZED | - |
| scalar | mul | NOT VECTORIZED | - |
| scalar | stencil3 | NOT VECTORIZED | - |
| scalar | mul_strided_c | NOT VECTORIZED | - |
| scalar | mul_gather | NOT VECTORIZED | - |
| sse2 | saxpy | 16 | 8B |
| sse2 | dot | 16 |  |
| sse2 | mul | 16 | 8B |
| sse2 | stencil3 | 16 | 8B |
| sse2 | mul_strided_c | 16 |  |
| sse2 | mul_gather | NOT VECTORIZED | - |
| avx2 | saxpy | 32 | 16B |
| avx2 | dot | 32 | 16B |
| avx2 | mul | 32 | 16B |
| avx2 | stencil3 | 32 | 16B |
| avx2 | mul_strided_c | 32 | 16B |
| avx2 | mul_gather | NOT VECTORIZED | - |
| avx512 | saxpy | 64 | masked 64B |
| avx512 | dot | 64 | 16B, 32B, masked 64B |
| avx512 | mul | 64 | masked 64B |
| avx512 | stencil3 | 64 | masked 64B |
| avx512 | mul_strided_c | 64 | masked 64B |
| avx512 | mul_gather | NOT VECTORIZED | - |
| avx512_fm | saxpy | 64 | masked 64B |
| avx512_fm | dot | 64 | masked 64B |
| avx512_fm | mul | 64 | masked 64B |
| avx512_fm | stencil3 | 64 | masked 64B |
| avx512_fm | mul_strided_c | 64 | masked 64B |
| avx512_fm | mul_gather | NOT VECTORIZED | - |
| avx512_gather | saxpy | 64 | masked 64B |
| avx512_gather | dot | 64 | 16B, 32B, masked 64B |
| avx512_gather | mul | 64 | masked 64B |
| avx512_gather | stencil3 | 64 | masked 64B |
| avx512_gather | mul_strided_c | 64 | masked 64B |
| avx512_gather | mul_gather | 64 | masked 64B |
