"""Fused Metal quantize kernel: raw fp16 vector → packed uint32 + norm.

Replaces the Python path: upcast → norm → normalize → signs → WHT → scale →
nearest centroid → pack. All in one Metal dispatch per batch of vectors.

Also includes fp16-output dequant for decode buffer writes.
"""

import mlx.core as mx
import math

# Fused quantize: one threadgroup per vector (dim threads)
# Input: fp16 vectors. Output: packed uint32 indices + float32 norms.
FUSED_QUANTIZE_KERNEL = """
    uint pos = threadgroup_position_in_grid.x;
    uint elem = thread_position_in_threadgroup.x;
    uint dim = dims[0];
    uint bits = dims[1];
    uint vals_per_word = dims[2];
    uint packed_dim = dims[3];
    uint n_centroids = dims[4];

    // Load input vector into shared memory as float32
    threadgroup float shared[512];
    shared[elem] = (float)inp[pos * dim + elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 1: Compute L2 norm via parallel reduction
    threadgroup float norm_shared[512];
    norm_shared[elem] = shared[elem] * shared[elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = dim / 2; stride > 0; stride >>= 1) {
        if (elem < stride) {
            norm_shared[elem] += norm_shared[elem + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float vec_norm = sqrt(norm_shared[0]);
    float safe_norm = max(vec_norm, 1e-8f);

    // Step 2: Normalize
    shared[elem] = shared[elem] / safe_norm;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Apply signs (randomized Hadamard = signs * WHT)
    shared[elem] = shared[elem] * signs[elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: WHT butterfly
    uint h = 1;
    while (h < dim) {
        uint block = elem / (2 * h);
        uint offset = elem % (2 * h);
        if (offset < h) {
            uint j = block * 2 * h + offset;
            float a = shared[j];
            float b = shared[j + h];
            shared[j] = a + b;
            shared[j + h] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        h *= 2;
    }

    // After raw butterfly (no 1/sqrt(d) normalization), values are already
    // in N(0,1) space: butterfly(x_unit * signs) ≈ N(0, 1)
    // No additional scaling needed — butterfly output matches codebook directly
    float scaled = shared[elem];

    // Step 6: Nearest centroid (count boundaries exceeded)
    uint idx = 0;
    for (uint b = 0; b < n_centroids - 1; b++) {
        if (scaled > boundaries[b]) {
            idx++;
        }
    }

    // Step 7: Pack indices - thread 0 of each pack group collects and packs
    // First store indices to shared memory
    threadgroup uint idx_shared[512];
    idx_shared[elem] = idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread responsible for one packed word writes it
    uint word_idx = elem / vals_per_word;
    uint pos_in_word = elem % vals_per_word;

    if (pos_in_word == 0 && word_idx < packed_dim) {
        uint word = 0;
        for (uint i = 0; i < vals_per_word && (word_idx * vals_per_word + i) < dim; i++) {
            word |= (idx_shared[word_idx * vals_per_word + i] & ((1u << bits) - 1u)) << (i * bits);
        }
        packed_out[pos * packed_dim + word_idx] = word;
    }

    // Thread 0 writes the norm
    if (elem == 0) {
        norms_out[pos] = vec_norm;
    }
"""

# fp16-output dequant: same as v3 but outputs half precision
DEQUANT_FP16_KERNEL = """
    uint pos = threadgroup_position_in_grid.x;
    uint elem = thread_position_in_threadgroup.x;
    uint dim = dims[0];
    uint bits = dims[1];
    uint vals_per_word = dims[2];
    uint packed_dim = dims[3];
    uint bit_mask = (1u << bits) - 1u;

    uint word_idx = elem / vals_per_word;
    uint pos_in_word = elem % vals_per_word;
    uint word = packed[pos * packed_dim + word_idx];
    uint idx = (word >> (pos_in_word * bits)) & bit_mask;

    float val = centroids[idx] * scale[0];

    threadgroup float shared[512];
    shared[elem] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint h = 1;
    while (h < dim) {
        uint block = elem / (2 * h);
        uint offset = elem % (2 * h);
        if (offset < h) {
            uint j = block * 2 * h + offset;
            float a = shared[j];
            float b = shared[j + h];
            shared[j] = a + b;
            shared[j + h] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        h *= 2;
    }

    float result = shared[elem] * scale[0] * signs[elem] * norms[pos];
    out[pos * dim + elem] = (half)result;
"""

# ---------------------------------------------------------------------------
# Algorithm 2 (TurboQuant_prod): MSE quantize at (b-1) bits + QJL on residual
# ---------------------------------------------------------------------------

# Gaussian QJL: sign(S @ residual) where S is d x d Gaussian matrix
FUSED_QUANTIZE_PROD_GAUSSIAN_KERNEL = """
    uint pos = threadgroup_position_in_grid.x;
    uint elem = thread_position_in_threadgroup.x;
    uint dim = dims[0];
    uint mse_bits = dims[1];
    uint mse_vpw = dims[2];
    uint mse_pdim = dims[3];
    uint n_centroids = dims[4];
    uint qjl_pdim = dims[5];

    // Load input and save original in register
    float my_orig = (float)inp[pos * dim + elem];

    threadgroup float shared[512];
    shared[elem] = my_orig;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 1: Compute L2 norm
    threadgroup float norm_shared[512];
    norm_shared[elem] = shared[elem] * shared[elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = dim / 2; stride > 0; stride >>= 1) {
        if (elem < stride) norm_shared[elem] += norm_shared[elem + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float vec_norm = sqrt(norm_shared[0]);
    float safe_norm = max(vec_norm, 1e-8f);

    // Step 2: Normalize
    shared[elem] = shared[elem] / safe_norm;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Apply MSE signs + WHT butterfly
    shared[elem] = shared[elem] * signs[elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint h = 1;
    while (h < dim) {
        uint block = elem / (2 * h);
        uint offset = elem % (2 * h);
        if (offset < h) {
            uint j = block * 2 * h + offset;
            float a = shared[j]; float b = shared[j + h];
            shared[j] = a + b; shared[j + h] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        h *= 2;
    }

    float rotated_val = shared[elem];

    // Step 4: Nearest centroid at (b-1) bits
    uint mse_idx = 0;
    for (uint b = 0; b < n_centroids - 1; b++) {
        if (rotated_val > boundaries[b]) mse_idx++;
    }

    // Step 5: Pack MSE indices
    threadgroup uint idx_shared[512];
    idx_shared[elem] = mse_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint mse_word_idx = elem / mse_vpw;
    uint mse_pos_in_word = elem % mse_vpw;
    if (mse_pos_in_word == 0 && mse_word_idx < mse_pdim) {
        uint word = 0;
        for (uint i = 0; i < mse_vpw && (mse_word_idx * mse_vpw + i) < dim; i++) {
            word |= (idx_shared[mse_word_idx * mse_vpw + i] & ((1u << mse_bits) - 1u)) << (i * mse_bits);
        }
        mse_packed_out[pos * mse_pdim + mse_word_idx] = word;
    }

    // Step 6: Compute residual in rotated space, then inverse WHT to original space
    // Residual in rotated space: r_rot = rotated_val - centroid
    float centroid_val = centroids[mse_idx];
    float r_rot = rotated_val - centroid_val;

    // Inverse WHT: B^{-1} = B/d, so inverse = butterfly then /d, then undo signs
    shared[elem] = r_rot;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    h = 1;
    while (h < dim) {
        uint block = elem / (2 * h);
        uint offset = elem % (2 * h);
        if (offset < h) {
            uint j = block * 2 * h + offset;
            float a = shared[j]; float b = shared[j + h];
            shared[j] = a + b; shared[j + h] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        h *= 2;
    }

    // residual = WHT(r_rot) / d * signs * norm  (undo normalization + rotation)
    float inv_d = 1.0f / (float)dim;
    float residual = shared[elem] * inv_d * signs[elem] * vec_norm;

    // Step 7: Compute residual L2 norm
    norm_shared[elem] = residual * residual;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = dim / 2; stride > 0; stride >>= 1) {
        if (elem < stride) norm_shared[elem] += norm_shared[elem + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float res_norm = sqrt(norm_shared[0]);

    // Step 8: QJL sign projection — sign(S @ residual)
    // All threads store residual to shared for the matrix multiply
    shared[elem] = residual;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float proj = 0.0f;
    for (uint j = 0; j < dim; j++) {
        proj += S_matrix[elem * dim + j] * shared[j];
    }
    uint sign_bit = (proj > 0.0f) ? 1u : 0u;

    // Step 9: Pack QJL sign bits (1-bit, 32 per uint32)
    idx_shared[elem] = sign_bit;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint qjl_word_idx = elem / 32u;
    uint qjl_pos = elem % 32u;
    if (qjl_pos == 0 && qjl_word_idx < qjl_pdim) {
        uint word = 0;
        for (uint i = 0; i < 32u && (qjl_word_idx * 32u + i) < dim; i++) {
            word |= (idx_shared[qjl_word_idx * 32u + i] & 1u) << i;
        }
        qjl_packed_out[pos * qjl_pdim + qjl_word_idx] = word;
    }

    // Write norms
    if (elem == 0) {
        mse_norms_out[pos] = vec_norm;
        res_norms_out[pos] = res_norm;
    }
"""

# Hadamard QJL: sign(WHT(qjl_signs * residual)) — O(d log d) instead of O(d^2)
FUSED_QUANTIZE_PROD_HADAMARD_KERNEL = """
    uint pos = threadgroup_position_in_grid.x;
    uint elem = thread_position_in_threadgroup.x;
    uint dim = dims[0];
    uint mse_bits = dims[1];
    uint mse_vpw = dims[2];
    uint mse_pdim = dims[3];
    uint n_centroids = dims[4];
    uint qjl_pdim = dims[5];

    float my_orig = (float)inp[pos * dim + elem];

    threadgroup float shared[512];
    shared[elem] = my_orig;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // L2 norm
    threadgroup float norm_shared[512];
    norm_shared[elem] = shared[elem] * shared[elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = dim / 2; stride > 0; stride >>= 1) {
        if (elem < stride) norm_shared[elem] += norm_shared[elem + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float vec_norm = sqrt(norm_shared[0]);
    float safe_norm = max(vec_norm, 1e-8f);

    // Normalize + signs + WHT
    shared[elem] = shared[elem] / safe_norm;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    shared[elem] = shared[elem] * signs[elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint h = 1;
    while (h < dim) {
        uint block = elem / (2 * h);
        uint offset = elem % (2 * h);
        if (offset < h) {
            uint j = block * 2 * h + offset;
            float a = shared[j]; float b = shared[j + h];
            shared[j] = a + b; shared[j + h] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        h *= 2;
    }

    float rotated_val = shared[elem];

    // Nearest centroid at (b-1) bits
    uint mse_idx = 0;
    for (uint b = 0; b < n_centroids - 1; b++) {
        if (rotated_val > boundaries[b]) mse_idx++;
    }

    // Pack MSE indices
    threadgroup uint idx_shared[512];
    idx_shared[elem] = mse_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint mse_word_idx = elem / mse_vpw;
    uint mse_pos_in_word = elem % mse_vpw;
    if (mse_pos_in_word == 0 && mse_word_idx < mse_pdim) {
        uint word = 0;
        for (uint i = 0; i < mse_vpw && (mse_word_idx * mse_vpw + i) < dim; i++) {
            word |= (idx_shared[mse_word_idx * mse_vpw + i] & ((1u << mse_bits) - 1u)) << (i * mse_bits);
        }
        mse_packed_out[pos * mse_pdim + mse_word_idx] = word;
    }

    // Residual in rotated space → inverse WHT → original space
    float centroid_val = centroids[mse_idx];
    float r_rot = rotated_val - centroid_val;

    shared[elem] = r_rot;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    h = 1;
    while (h < dim) {
        uint block = elem / (2 * h);
        uint offset = elem % (2 * h);
        if (offset < h) {
            uint j = block * 2 * h + offset;
            float a = shared[j]; float b = shared[j + h];
            shared[j] = a + b; shared[j + h] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        h *= 2;
    }

    float inv_d = 1.0f / (float)dim;
    float residual = shared[elem] * inv_d * signs[elem] * vec_norm;

    // Residual norm
    norm_shared[elem] = residual * residual;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = dim / 2; stride > 0; stride >>= 1) {
        if (elem < stride) norm_shared[elem] += norm_shared[elem + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float res_norm = sqrt(norm_shared[0]);

    // QJL via Hadamard: sign(WHT(qjl_signs * residual))
    shared[elem] = residual * qjl_signs[elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    h = 1;
    while (h < dim) {
        uint block = elem / (2 * h);
        uint offset = elem % (2 * h);
        if (offset < h) {
            uint j = block * 2 * h + offset;
            float a = shared[j]; float b = shared[j + h];
            shared[j] = a + b; shared[j + h] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        h *= 2;
    }

    uint sign_bit = (shared[elem] > 0.0f) ? 1u : 0u;

    // Pack QJL sign bits
    idx_shared[elem] = sign_bit;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint qjl_word_idx = elem / 32u;
    uint qjl_pos = elem % 32u;
    if (qjl_pos == 0 && qjl_word_idx < qjl_pdim) {
        uint word = 0;
        for (uint i = 0; i < 32u && (qjl_word_idx * 32u + i) < dim; i++) {
            word |= (idx_shared[qjl_word_idx * 32u + i] & 1u) << i;
        }
        qjl_packed_out[pos * qjl_pdim + qjl_word_idx] = word;
    }

    if (elem == 0) {
        mse_norms_out[pos] = vec_norm;
        res_norms_out[pos] = res_norm;
    }
"""

# Dequant Algorithm 2 with Gaussian QJL → fp16 output
DEQUANT_PROD_GAUSSIAN_FP16_KERNEL = """
    uint pos = threadgroup_position_in_grid.x;
    uint elem = thread_position_in_threadgroup.x;
    uint dim = dims[0];
    uint mse_bits = dims[1];
    uint mse_vpw = dims[2];
    uint mse_pdim = dims[3];
    uint qjl_pdim = dims[4];

    uint mse_bit_mask = (1u << mse_bits) - 1u;

    // MSE dequant: unpack → codebook → WHT → signs/scale/norm
    uint mse_wid = elem / mse_vpw;
    uint mse_piw = elem % mse_vpw;
    uint mse_word = mse_packed[pos * mse_pdim + mse_wid];
    uint mse_idx = (mse_word >> (mse_piw * mse_bits)) & mse_bit_mask;

    float mse_val = centroids[mse_idx] * mse_scale[0];

    threadgroup float shared[512];
    shared[elem] = mse_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint h = 1;
    while (h < dim) {
        uint block = elem / (2 * h);
        uint offset = elem % (2 * h);
        if (offset < h) {
            uint j = block * 2 * h + offset;
            float a = shared[j]; float b = shared[j + h];
            shared[j] = a + b; shared[j + h] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        h *= 2;
    }

    float x_mse = shared[elem] * mse_scale[0] * signs[elem] * mse_norms[pos];

    // QJL dequant: S^T @ qjl_decoded * qjl_const * res_norm
    // Unpack 1-bit QJL sign
    uint qjl_wid = elem / 32u;
    uint qjl_piw = elem % 32u;
    uint qjl_word = qjl_packed[pos * qjl_pdim + qjl_wid];
    float qjl_decoded_elem = ((qjl_word >> qjl_piw) & 1u) ? 1.0f : -1.0f;

    // Store all decoded QJL signs to shared for matrix multiply
    shared[elem] = qjl_decoded_elem;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute (S^T @ z)[elem] = sum_j S[j * dim + elem] * z[j]
    float x_qjl = 0.0f;
    for (uint j = 0; j < dim; j++) {
        x_qjl += S_matrix[j * dim + elem] * shared[j];
    }
    x_qjl *= qjl_const[0] * res_norms[pos];

    out[pos * dim + elem] = (half)(x_mse + x_qjl);
"""

# Dequant Algorithm 2 with Hadamard QJL → fp16 output
DEQUANT_PROD_HADAMARD_FP16_KERNEL = """
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

    float mse_val = centroids[mse_idx] * mse_scale[0];

    threadgroup float shared[512];
    shared[elem] = mse_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint h = 1;
    while (h < dim) {
        uint block = elem / (2 * h);
        uint offset = elem % (2 * h);
        if (offset < h) {
            uint j = block * 2 * h + offset;
            float a = shared[j]; float b = shared[j + h];
            shared[j] = a + b; shared[j + h] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        h *= 2;
    }

    float x_mse = shared[elem] * mse_scale[0] * signs[elem] * mse_norms[pos];

    // QJL dequant via Hadamard: qjl_const * res_norm * WHT(qjl_signs * z) / sqrt(d)
    // Unpack 1-bit sign
    uint qjl_wid = elem / 32u;
    uint qjl_piw = elem % 32u;
    uint qjl_word = qjl_packed[pos * qjl_pdim + qjl_wid];
    float qjl_decoded = ((qjl_word >> qjl_piw) & 1u) ? 1.0f : -1.0f;

    // Apply QJL signs then WHT
    shared[elem] = qjl_decoded * qjl_signs[elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    h = 1;
    while (h < dim) {
        uint block = elem / (2 * h);
        uint offset = elem % (2 * h);
        if (offset < h) {
            uint j = block * 2 * h + offset;
            float a = shared[j]; float b = shared[j + h];
            shared[j] = a + b; shared[j + h] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        h *= 2;
    }

    // Hadamard inverse includes 1/d for raw butterfly normalization
    float inv_d = 1.0f / (float)dim;
    float x_qjl = shared[elem] * inv_d * qjl_const[0] * res_norms[pos];

    out[pos * dim + elem] = (half)(x_mse + x_qjl);
"""

_fused_quantize_kernel = None
_dequant_fp16_kernel = None
_fused_quantize_prod_gaussian_kernel = None
_fused_quantize_prod_hadamard_kernel = None
_dequant_prod_gaussian_fp16_kernel = None
_dequant_prod_hadamard_fp16_kernel = None


def fused_quantize(
    vectors: mx.array,
    signs: mx.array,
    boundaries: mx.array,
    dim: int,
    bits: int,
) -> tuple:
    """Fused Metal quantize: raw vectors → packed uint32 + norms.

    Args:
        vectors: (n_vecs, dim) fp16/fp32 input
        signs: (dim,) rotation signs
        boundaries: (n_centroids-1,) decision boundaries
        dim: head dimension
        bits: quantization bits

    Returns:
        (packed, norms): packed uint32 (n_vecs, packed_dim), norms float32 (n_vecs,)
    """
    global _fused_quantize_kernel
    if _fused_quantize_kernel is None:
        _fused_quantize_kernel = mx.fast.metal_kernel(
            name="tq_fused_quantize",
            input_names=["inp", "signs", "boundaries", "dims"],
            output_names=["packed_out", "norms_out"],
            source=FUSED_QUANTIZE_KERNEL,
        )

    assert dim <= 512, f"Head dimension {dim} exceeds Metal kernel shared memory limit of 512"
    from mlx_lm.models.turboquant_packing import packed_dim as calc_packed_dim, VALS_PER_WORD
    n_vecs = vectors.shape[0]
    vpw = VALS_PER_WORD[bits]
    p_dim = calc_packed_dim(dim, bits)
    n_centroids = len(boundaries) + 1

    dims_arr = mx.array([dim, bits, vpw, p_dim, n_centroids], dtype=mx.uint32)

    outputs = _fused_quantize_kernel(
        inputs=[
            vectors.reshape(n_vecs * dim).astype(mx.float32),
            signs.astype(mx.float32),
            boundaries.astype(mx.float32),
            dims_arr,
        ],
        template=[],
        grid=(n_vecs * dim, 1, 1),
        threadgroup=(dim, 1, 1),
        output_shapes=[(n_vecs * p_dim,), (n_vecs,)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return outputs[0].reshape(n_vecs, p_dim), outputs[1]


def dequant_fp16(
    packed: mx.array,
    norms: mx.array,
    centroids: mx.array,
    signs: mx.array,
    dim: int,
    bits: int,
) -> mx.array:
    """Dequantize from packed to fp16 directly (no float32 intermediate)."""
    global _dequant_fp16_kernel
    if _dequant_fp16_kernel is None:
        _dequant_fp16_kernel = mx.fast.metal_kernel(
            name="tq_dequant_fp16",
            input_names=["packed", "norms", "centroids", "signs", "scale", "dims"],
            output_names=["out"],
            source=DEQUANT_FP16_KERNEL,
        )

    from mlx_lm.models.turboquant_packing import packed_dim as calc_packed_dim, VALS_PER_WORD
    seq_len = norms.shape[0]
    vpw = VALS_PER_WORD[bits]
    p_dim = calc_packed_dim(dim, bits)
    scale = mx.array([1.0 / math.sqrt(dim)], dtype=mx.float32)
    dims_arr = mx.array([dim, bits, vpw, p_dim], dtype=mx.uint32)

    outputs = _dequant_fp16_kernel(
        inputs=[packed.astype(mx.uint32).reshape(-1), norms.astype(mx.float32), centroids, signs, scale, dims_arr],
        template=[],
        grid=(seq_len * dim, 1, 1),
        threadgroup=(dim, 1, 1),
        output_shapes=[(seq_len, dim)],
        output_dtypes=[mx.float16],
    )
    return outputs[0]


def fused_quantize_prod(
    vectors: mx.array,
    signs: mx.array,
    boundaries: mx.array,
    centroids: mx.array,
    qjl_matrix: mx.array,
    dim: int,
    bits: int,
    qjl_mode: str = "gaussian",
) -> tuple:
    """Fused Algorithm 2 quantize: MSE at (b-1) bits + QJL on residual.

    Returns:
        (mse_packed, qjl_packed, mse_norms, res_norms)
    """
    assert dim <= 512, f"Head dimension {dim} exceeds Metal kernel shared memory limit of 512"
    from mlx_lm.models.turboquant_packing import packed_dim as calc_packed_dim, VALS_PER_WORD

    n_vecs = vectors.shape[0]
    mse_bits = bits - 1
    mse_vpw = VALS_PER_WORD[mse_bits]
    mse_pdim = calc_packed_dim(dim, mse_bits)
    n_centroids = len(boundaries) + 1
    qjl_pdim = calc_packed_dim(dim, 1)

    dims_arr = mx.array([dim, mse_bits, mse_vpw, mse_pdim, n_centroids, qjl_pdim], dtype=mx.uint32)

    if qjl_mode == "gaussian":
        global _fused_quantize_prod_gaussian_kernel
        if _fused_quantize_prod_gaussian_kernel is None:
            _fused_quantize_prod_gaussian_kernel = mx.fast.metal_kernel(
                name="tq_fused_quantize_prod_gaussian",
                input_names=["inp", "signs", "boundaries", "centroids", "S_matrix", "dims"],
                output_names=["mse_packed_out", "qjl_packed_out", "mse_norms_out", "res_norms_out"],
                source=FUSED_QUANTIZE_PROD_GAUSSIAN_KERNEL,
            )
        outputs = _fused_quantize_prod_gaussian_kernel(
            inputs=[
                vectors.reshape(n_vecs * dim).astype(mx.float32),
                signs.astype(mx.float32),
                boundaries.astype(mx.float32),
                centroids.astype(mx.float32),
                qjl_matrix.astype(mx.float32).reshape(-1),
                dims_arr,
            ],
            template=[],
            grid=(n_vecs * dim, 1, 1),
            threadgroup=(dim, 1, 1),
            output_shapes=[(n_vecs * mse_pdim,), (n_vecs * qjl_pdim,), (n_vecs,), (n_vecs,)],
            output_dtypes=[mx.uint32, mx.uint32, mx.float32, mx.float32],
        )
    else:
        global _fused_quantize_prod_hadamard_kernel
        if _fused_quantize_prod_hadamard_kernel is None:
            _fused_quantize_prod_hadamard_kernel = mx.fast.metal_kernel(
                name="tq_fused_quantize_prod_hadamard",
                input_names=["inp", "signs", "boundaries", "centroids", "qjl_signs", "dims"],
                output_names=["mse_packed_out", "qjl_packed_out", "mse_norms_out", "res_norms_out"],
                source=FUSED_QUANTIZE_PROD_HADAMARD_KERNEL,
            )
        outputs = _fused_quantize_prod_hadamard_kernel(
            inputs=[
                vectors.reshape(n_vecs * dim).astype(mx.float32),
                signs.astype(mx.float32),
                boundaries.astype(mx.float32),
                centroids.astype(mx.float32),
                qjl_matrix.astype(mx.float32),
                dims_arr,
            ],
            template=[],
            grid=(n_vecs * dim, 1, 1),
            threadgroup=(dim, 1, 1),
            output_shapes=[(n_vecs * mse_pdim,), (n_vecs * qjl_pdim,), (n_vecs,), (n_vecs,)],
            output_dtypes=[mx.uint32, mx.uint32, mx.float32, mx.float32],
        )

    return (
        outputs[0].reshape(n_vecs, mse_pdim),
        outputs[1].reshape(n_vecs, qjl_pdim),
        outputs[2],
        outputs[3],
    )


def dequant_prod_fp16(
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
    """Dequantize Algorithm 2 from packed to fp16."""
    from mlx_lm.models.turboquant_packing import packed_dim as calc_packed_dim, VALS_PER_WORD

    mse_bits = bits - 1
    seq_len = mse_norms.shape[0]
    mse_vpw = VALS_PER_WORD[mse_bits]
    mse_pdim = calc_packed_dim(dim, mse_bits)
    qjl_pdim = calc_packed_dim(dim, 1)
    mse_scale = mx.array([1.0 / math.sqrt(dim)], dtype=mx.float32)
    qjl_const_arr = mx.array([qjl_const], dtype=mx.float32)
    dims_arr = mx.array([dim, mse_bits, mse_vpw, mse_pdim, qjl_pdim], dtype=mx.uint32)

    if qjl_mode == "gaussian":
        global _dequant_prod_gaussian_fp16_kernel
        if _dequant_prod_gaussian_fp16_kernel is None:
            _dequant_prod_gaussian_fp16_kernel = mx.fast.metal_kernel(
                name="tq_dequant_prod_gaussian_fp16",
                input_names=["mse_packed", "qjl_packed", "mse_norms", "res_norms",
                             "centroids", "signs", "S_matrix", "mse_scale", "qjl_const", "dims"],
                output_names=["out"],
                source=DEQUANT_PROD_GAUSSIAN_FP16_KERNEL,
            )
        outputs = _dequant_prod_gaussian_fp16_kernel(
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
            template=[],
            grid=(seq_len * dim, 1, 1),
            threadgroup=(dim, 1, 1),
            output_shapes=[(seq_len, dim)],
            output_dtypes=[mx.float16],
        )
    else:
        global _dequant_prod_hadamard_fp16_kernel
        if _dequant_prod_hadamard_fp16_kernel is None:
            _dequant_prod_hadamard_fp16_kernel = mx.fast.metal_kernel(
                name="tq_dequant_prod_hadamard_fp16",
                input_names=["mse_packed", "qjl_packed", "mse_norms", "res_norms",
                             "centroids", "signs", "qjl_signs", "mse_scale", "qjl_const", "dims"],
                output_names=["out"],
                source=DEQUANT_PROD_HADAMARD_FP16_KERNEL,
            )
        outputs = _dequant_prod_hadamard_fp16_kernel(
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
            template=[],
            grid=(seq_len * dim, 1, 1),
            threadgroup=(dim, 1, 1),
            output_shapes=[(seq_len, dim)],
            output_dtypes=[mx.float16],
        )

    return outputs[0]
