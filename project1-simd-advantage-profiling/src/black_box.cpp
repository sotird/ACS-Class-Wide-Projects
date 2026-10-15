// Compiled as its own translation unit (no LTO) so the function below is an
// opaque call from exp8_negative_novec.cpp's point of view -- gcc cannot
// inline it or prove it has no side effects, which is exactly what blocks
// auto-vectorization in that experiment.
extern "C" float black_box(float x) {
    return x * 1.0000001f + 1e-7f;
}
