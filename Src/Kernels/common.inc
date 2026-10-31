// Shared scalar indexing and deterministic random helpers for all native backends.
EL_INLINE int elman_broadcast_index(int flat, int rank_slot, int shape_slot,
                                    int stride_slot EL_CONTEXT_ARGS) {
    int result = 0;
    int delta = P(1) - P(rank_slot);
    for (int axis = P(1) - 1; axis >= 0; --axis) {
        int coordinate = flat % P(2 + axis);
        flat /= P(2 + axis);
        int input_axis = axis - delta;
        if (input_axis >= 0 && P(shape_slot + input_axis) != 1)
            result += coordinate * P(stride_slot + input_axis);
    }
    return result;
}
EL_INLINE int elman_count(int rank_slot, int shape_slot EL_CONTEXT_ARGS) {
    if (P(rank_slot) < 0) return 0;
    int result = 1;
    for (int axis = 0; axis < P(rank_slot); ++axis) result *= P(shape_slot + axis);
    return result;
}
EL_INLINE int elman_output_stride(int axis EL_CONTEXT_ARGS) {
    int result = 1;
    for (int next = axis + 1; next < P(1); ++next) result *= P(2 + next);
    return result;
}
EL_INLINE int elman_matrix_batch(int flat, int rank_slot, int shape_slot,
                                 int stride_slot EL_CONTEXT_ARGS) {
    int result = 0;
    int delta = P(1) - P(rank_slot);
    for (int axis = P(1) - 3; axis >= 0; --axis) {
        int coordinate = flat % P(2 + axis);
        flat /= P(2 + axis);
        int input_axis = axis - delta;
        if (input_axis >= 0 && P(shape_slot + input_axis) != 1)
            result += coordinate * P(stride_slot + input_axis);
    }
    return result;
}
EL_INLINE float elman_random(int index, int seed, int offset) {
    EL_UINT value = EL_UINT(index) + EL_UINT(offset) * 747796405u + EL_UINT(seed) * 2891336453u + 1u;
    value ^= value >> 16;
    value *= 2246822519u;
    value ^= value >> 13;
    value *= 3266489917u;
    value ^= value >> 16;
    return (float(value & 16777215u) + 0.5f) * (1.0f / 16777216.0f);
}
#define EL_BINDEX(i,r,s,t) elman_broadcast_index(i,r,s,t EL_CONTEXT_PASS)
#define EL_COUNT(r,s) elman_count(r,s EL_CONTEXT_PASS)
#define EL_OSTRIDE(a) elman_output_stride(a EL_CONTEXT_PASS)
#define EL_MATRIX_BATCH(i,r,s,t) elman_matrix_batch(i,r,s,t EL_CONTEXT_PASS)
