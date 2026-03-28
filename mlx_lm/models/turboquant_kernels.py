"""Metal kernels v3: read directly from bit-packed uint32 storage.

Eliminates Python unpack step — the kernel extracts 3-bit indices
from packed uint32 words on the fly. Zero intermediate buffers.

Packing format: 10 × 3-bit values per uint32 (30/32 bits used)
  word = val0 | (val1 << 3) | (val2 << 6) | ... | (val9 << 27)
"""

import mlx.core as mx
import math

# Parallel dequant from packed storage
PACKED_DEQUANT_KERNEL = """
    uint pos = threadgroup_position_in_grid.x;
    uint elem = thread_position_in_threadgroup.x;
    uint dim = dims[0];
    uint bits = dims[1];
    uint vals_per_word = dims[2];
    uint packed_dim = dims[3];
    uint bit_mask = (1u << bits) - 1u;

    // Extract index from packed uint32
    uint word_idx = elem / vals_per_word;
    uint pos_in_word = elem % vals_per_word;
    uint word = packed[pos * packed_dim + word_idx];
    uint idx = (word >> (pos_in_word * bits)) & bit_mask;

    // Codebook lookup
    T val = centroids[idx] * scale[0];

    // Parallel WHT butterfly in threadgroup memory
    threadgroup T shared[512];
    shared[elem] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint h = 1;
    while (h < dim) {
        uint block = elem / (2 * h);
        uint offset = elem % (2 * h);
        if (offset < h) {
            uint j = block * 2 * h + offset;
            T a = shared[j];
            T b = shared[j + h];
            shared[j] = a + b;
            shared[j + h] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        h *= 2;
    }

    // Apply WHT scale, signs, and vector norm
    T result = shared[elem] * scale[0] * signs[elem] * norms[pos];
    out[pos * dim + elem] = result;
"""

# Fused Q@K^T from packed storage — no unpack, no intermediate dequant
PACKED_FUSED_QK_KERNEL = """
    uint pos = threadgroup_position_in_grid.x;
    uint head = threadgroup_position_in_grid.y;
    uint elem = thread_position_in_threadgroup.x;
    uint dim = dims[0];
    uint seq_len = dims[1];
    uint n_heads = dims[2];
    uint bits = dims[3];
    uint vals_per_word = dims[4];
    uint packed_dim = dims[5];
    uint bit_mask = (1u << bits) - 1u;

    // Extract index from packed storage
    uint kv_base = head * seq_len * packed_dim + pos * packed_dim;
    uint word_idx = elem / vals_per_word;
    uint pos_in_word = elem % vals_per_word;
    uint word = packed[kv_base + word_idx];
    uint idx = (word >> (pos_in_word * bits)) & bit_mask;

    T val = centroids[idx] * scale[0];

    // Parallel WHT butterfly
    threadgroup T shared[512];
    shared[elem] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint h = 1;
    while (h < dim) {
        uint block = elem / (2 * h);
        uint offset = elem % (2 * h);
        if (offset < h) {
            uint j = block * 2 * h + offset;
            T a = shared[j];
            T b = shared[j + h];
            shared[j] = a + b;
            shared[j + h] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        h *= 2;
    }

    // Dequant value + dot product with query
    T dequant_val = shared[elem] * scale[0] * signs[elem] * norms[head * seq_len + pos];
    T partial = dequant_val * query[head * dim + elem];

    // Parallel reduction
    shared[elem] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = dim / 2; stride > 0; stride >>= 1) {
        if (elem < stride) {
            shared[elem] += shared[elem + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (elem == 0) {
        out[head * seq_len + pos] = shared[0];
    }
"""

# Algorithm 2 packed dequant: MSE + QJL combined, float32 output
# Gaussian variant: uses S^T @ z matrix multiply
PACKED_DEQUANT_PROD_GAUSSIAN_KERNEL = """
    uint pos = threadgroup_position_in_grid.x;
    uint elem = thread_position_in_threadgroup.x;
    uint dim = dims[0];
    uint mse_bits = dims[1];
    uint mse_vpw = dims[2];
    uint mse_pdim = dims[3];
    uint qjl_pdim = dims[4];
    uint mse_bit_mask = (1u << mse_bits) - 1u;

    // MSE dequant
    uint mse_wid = elem / mse_vpw;
    uint mse_piw = elem % mse_vpw;
    uint mse_word = mse_packed[pos * mse_pdim + mse_wid];
    uint mse_idx = (mse_word >> (mse_piw * mse_bits)) & mse_bit_mask;

    T mse_val = centroids[mse_idx] * mse_scale[0];
    threadgroup T shared[512];
    shared[elem] = mse_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint h = 1;
    while (h < dim) {
        uint block = elem / (2 * h);
        uint offset = elem % (2 * h);
        if (offset < h) {
            uint j = block * 2 * h + offset;
            T a = shared[j]; T b = shared[j + h];
            shared[j] = a + b; shared[j + h] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        h *= 2;
    }

    T x_mse = shared[elem] * mse_scale[0] * signs[elem] * mse_norms[pos];

    // QJL dequant: S^T @ z * qjl_const * res_norm
    uint qjl_wid = elem / 32u;
    uint qjl_piw = elem % 32u;
    uint qjl_word = qjl_packed[pos * qjl_pdim + qjl_wid];
    T qjl_decoded = ((qjl_word >> qjl_piw) & 1u) ? (T)1.0 : (T)-1.0;

    shared[elem] = qjl_decoded;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    T x_qjl = (T)0.0;
    for (uint j = 0; j < dim; j++) {
        x_qjl += S_matrix[j * dim + elem] * shared[j];
    }
    x_qjl *= qjl_const[0] * res_norms[pos];

    out[pos * dim + elem] = x_mse + x_qjl;
"""

# Hadamard variant: uses WHT butterfly instead of matrix multiply
PACKED_DEQUANT_PROD_HADAMARD_KERNEL = """
    uint pos = threadgroup_position_in_grid.x;
    uint elem = thread_position_in_threadgroup.x;
    uint dim = dims[0];
    uint mse_bits = dims[1];
    uint mse_vpw = dims[2];
    uint mse_pdim = dims[3];
    uint qjl_pdim = dims[4];
    uint mse_bit_mask = (1u << mse_bits) - 1u;

    // MSE dequant
    uint mse_wid = elem / mse_vpw;
    uint mse_piw = elem % mse_vpw;
    uint mse_word = mse_packed[pos * mse_pdim + mse_wid];
    uint mse_idx = (mse_word >> (mse_piw * mse_bits)) & mse_bit_mask;

    T mse_val = centroids[mse_idx] * mse_scale[0];
    threadgroup T shared[512];
    shared[elem] = mse_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint h = 1;
    while (h < dim) {
        uint block = elem / (2 * h);
        uint offset = elem % (2 * h);
        if (offset < h) {
            uint j = block * 2 * h + offset;
            T a = shared[j]; T b = shared[j + h];
            shared[j] = a + b; shared[j + h] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        h *= 2;
    }

    T x_mse = shared[elem] * mse_scale[0] * signs[elem] * mse_norms[pos];

    // QJL dequant via Hadamard: WHT(qjl_signs * z) * inv_d * qjl_const * res_norm
    uint qjl_wid = elem / 32u;
    uint qjl_piw = elem % 32u;
    uint qjl_word = qjl_packed[pos * qjl_pdim + qjl_wid];
    T qjl_decoded = ((qjl_word >> qjl_piw) & 1u) ? (T)1.0 : (T)-1.0;

    shared[elem] = qjl_decoded * qjl_signs[elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    h = 1;
    while (h < dim) {
        uint block = elem / (2 * h);
        uint offset = elem % (2 * h);
        if (offset < h) {
            uint j = block * 2 * h + offset;
            T a = shared[j]; T b = shared[j + h];
            shared[j] = a + b; shared[j + h] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        h *= 2;
    }

    T inv_d = (T)1.0 / (T)dim;
    T x_qjl = shared[elem] * inv_d * qjl_const[0] * res_norms[pos];

    out[pos * dim + elem] = x_mse + x_qjl;
"""

_packed_dequant = None
_packed_fused_qk = None
_packed_dequant_prod_gaussian = None
_packed_dequant_prod_hadamard = None


def packed_dequantize(
    packed: mx.array,
    norms: mx.array,
    centroids: mx.array,
    signs: mx.array,
    dim: int,
    bits: int,
) -> mx.array:
    """Dequantize directly from packed uint32 storage via Metal."""
    assert dim <= 512, f"Head dimension {dim} exceeds Metal kernel shared memory limit of 512"
    global _packed_dequant
    if _packed_dequant is None:
        _packed_dequant = mx.fast.metal_kernel(
            name="tq_packed_dequant",
            input_names=["packed", "norms", "centroids", "signs", "scale", "dims"],
            output_names=["out"],
            source=PACKED_DEQUANT_KERNEL,
        )

    seq_len = norms.shape[0]
    p_dim = packed.shape[-1]
    vpw = {1: 32, 2: 16, 3: 10, 4: 8}[bits]
    scale = mx.array([1.0 / math.sqrt(dim)], dtype=mx.float32)
    dims_arr = mx.array([dim, bits, vpw, p_dim], dtype=mx.uint32)

    outputs = _packed_dequant(
        inputs=[packed.astype(mx.uint32).reshape(-1), norms.astype(mx.float32), centroids, signs, scale, dims_arr],
        template=[("T", mx.float32)],
        grid=(seq_len * dim, 1, 1),
        threadgroup=(dim, 1, 1),
        output_shapes=[(seq_len, dim)],
        output_dtypes=[mx.float32],
    )
    return outputs[0]


def packed_fused_qk_scores(
    query: mx.array,
    k_packed: mx.array,
    k_norms: mx.array,
    centroids: mx.array,
    signs: mx.array,
    dim: int,
    bits: int,
) -> mx.array:
    """Fused Q@K^T reading directly from packed storage."""
    global _packed_fused_qk
    if _packed_fused_qk is None:
        _packed_fused_qk = mx.fast.metal_kernel(
            name="tq_packed_fused_qk",
            input_names=["query", "packed", "norms", "centroids", "signs", "scale", "dims"],
            output_names=["out"],
            source=PACKED_FUSED_QK_KERNEL,
        )

    n_heads, seq_len = k_norms.shape
    p_dim = k_packed.shape[-1]
    vpw = {1: 32, 2: 16, 3: 10, 4: 8}[bits]
    scale = mx.array([1.0 / math.sqrt(dim)], dtype=mx.float32)
    dims_arr = mx.array([dim, seq_len, n_heads, bits, vpw, p_dim], dtype=mx.uint32)

    outputs = _packed_fused_qk(
        inputs=[
            query.astype(mx.float32).reshape(n_heads * dim),
            k_packed.astype(mx.uint32).reshape(n_heads * seq_len * p_dim),
            k_norms.astype(mx.float32).reshape(n_heads * seq_len),
            centroids, signs, scale, dims_arr,
        ],
        template=[("T", mx.float32)],
        grid=(seq_len * dim, n_heads, 1),
        threadgroup=(dim, 1, 1),
        output_shapes=[(n_heads * seq_len,)],
        output_dtypes=[mx.float32],
    )
    return outputs[0].reshape(n_heads, seq_len)


def packed_dequantize_prod(
    mse_packed: mx.array,
    qjl_packed: mx.array,
    mse_norms: mx.array,
    res_norms: mx.array,
    centroids: mx.array,
    signs: mx.array,
    qjl_matrix: mx.array,
    dim: int,
    bits: int,
    qjl_mode: str = "gaussian",
    qjl_const: float = 0.0,
) -> mx.array:
    """Dequantize Algorithm 2 from packed storage via Metal (float32 output)."""
    assert dim <= 512, f"Head dimension {dim} exceeds Metal kernel shared memory limit of 512"
    mse_bits = bits - 1
    seq_len = mse_norms.shape[0]
    mse_vpw = {1: 32, 2: 16, 3: 10, 4: 8}[mse_bits]
    mse_pdim = mse_packed.shape[-1]
    qjl_pdim = qjl_packed.shape[-1]
    mse_scale = mx.array([1.0 / math.sqrt(dim)], dtype=mx.float32)
    qjl_const_arr = mx.array([qjl_const], dtype=mx.float32)
    dims_arr = mx.array([dim, mse_bits, mse_vpw, mse_pdim, qjl_pdim], dtype=mx.uint32)

    if qjl_mode == "gaussian":
        global _packed_dequant_prod_gaussian
        if _packed_dequant_prod_gaussian is None:
            _packed_dequant_prod_gaussian = mx.fast.metal_kernel(
                name="tq_packed_dequant_prod_gaussian",
                input_names=["mse_packed", "qjl_packed", "mse_norms", "res_norms",
                             "centroids", "signs", "S_matrix", "mse_scale", "qjl_const", "dims"],
                output_names=["out"],
                source=PACKED_DEQUANT_PROD_GAUSSIAN_KERNEL,
            )
        outputs = _packed_dequant_prod_gaussian(
            inputs=[
                mse_packed.astype(mx.uint32).reshape(-1),
                qjl_packed.astype(mx.uint32).reshape(-1),
                mse_norms.astype(mx.float32),
                res_norms.astype(mx.float32),
                centroids.astype(mx.float32),
                signs.astype(mx.float32),
                qjl_matrix.astype(mx.float32).reshape(-1),
                mse_scale, qjl_const_arr, dims_arr,
            ],
            template=[("T", mx.float32)],
            grid=(seq_len * dim, 1, 1),
            threadgroup=(dim, 1, 1),
            output_shapes=[(seq_len, dim)],
            output_dtypes=[mx.float32],
        )
    else:
        global _packed_dequant_prod_hadamard
        if _packed_dequant_prod_hadamard is None:
            _packed_dequant_prod_hadamard = mx.fast.metal_kernel(
                name="tq_packed_dequant_prod_hadamard",
                input_names=["mse_packed", "qjl_packed", "mse_norms", "res_norms",
                             "centroids", "signs", "qjl_signs", "mse_scale", "qjl_const", "dims"],
                output_names=["out"],
                source=PACKED_DEQUANT_PROD_HADAMARD_KERNEL,
            )
        outputs = _packed_dequant_prod_hadamard(
            inputs=[
                mse_packed.astype(mx.uint32).reshape(-1),
                qjl_packed.astype(mx.uint32).reshape(-1),
                mse_norms.astype(mx.float32),
                res_norms.astype(mx.float32),
                centroids.astype(mx.float32),
                signs.astype(mx.float32),
                qjl_matrix.astype(mx.float32),
                mse_scale, qjl_const_arr, dims_arr,
            ],
            template=[("T", mx.float32)],
            grid=(seq_len * dim, 1, 1),
            threadgroup=(dim, 1, 1),
            output_shapes=[(seq_len, dim)],
            output_dtypes=[mx.float32],
        )

    return outputs[0]
