"""TurboQuantKVCache: TurboQuant KV cache compression with fused Metal kernels.

Implements TurboQuant (arXiv 2504.19874) for MLX KV cache compression.

Two modes:
  - 'mse' (Algorithm 1): Hadamard rotation + Lloyd-Max codebook. Fast, b-bit MSE-optimal.
  - 'prod' (Algorithm 2): MSE at (b-1) bits + 1-bit QJL on residual. Unbiased inner products.

Bit-packed uint32 storage with fused Metal quantize/dequantize kernels.
"""

import mlx.core as mx
import math
from mlx_lm.models.turboquant_rotation import (
    random_diagonal_sign, generate_qjl_matrix, qjl_reconstruction_constant,
)
from mlx_lm.models.turboquant_packing import packed_dim, VALS_PER_WORD
from mlx_lm.models.turboquant_metal import (
    fused_quantize, dequant_fp16, fused_quantize_prod, dequant_prod_fp16,
)
from mlx_lm.models.turboquant_kernels import packed_dequantize, packed_dequantize_prod


def _compute_gaussian_codebook(bits):
    codebooks = {
        1: [-0.7979, 0.7979],
        2: [-1.5104, -0.4528, 0.4528, 1.5104],
        3: [-2.1520, -1.3440, -0.7560, -0.2451,
             0.2451, 0.7560, 1.3440, 2.1520],
        4: [-2.7326, -2.0690, -1.6180, -1.2562,
            -0.9423, -0.6568, -0.3881, -0.1284,
             0.1284, 0.3881, 0.6568, 0.9423,
             1.2562, 1.6180, 2.0690, 2.7326],
    }
    return mx.array(codebooks[bits], dtype=mx.float32)


def _compute_boundaries(centroids):
    return (centroids[:-1] + centroids[1:]) / 2.0


class _Quantizer:
    def __init__(self, dim, bits, seed):
        self.dim = dim
        self.bits = bits
        self.signs = random_diagonal_sign(dim, seed=seed)
        self.centroids = _compute_gaussian_codebook(bits)
        self.boundaries = _compute_boundaries(self.centroids)


class TurboQuantKVCache:
    """TurboQuant KV cache -- drop-in replacement for KVCache.

    Args:
        bits: Total bit budget per coordinate. Default 3.
        mode: 'mse' (Algorithm 1) or 'prod' (Algorithm 2, unbiased inner products).
        qjl_mode: 'gaussian' (paper-faithful) or 'hadamard' (faster). Only for mode='prod'.
        seed: Random seed for rotation and QJL matrices.
    """

    step = 256

    def __init__(self, bits: int = 3, mode: str = "mse",
                 qjl_mode: str = "gaussian", seed: int = 42):
        self.quant_bits = bits
        self.mode = mode
        self.qjl_mode = qjl_mode
        self.seed = seed
        self.offset = 0

        # MSE storage (both modes)
        self.k_packed = None
        self.k_norms = None
        self.v_packed = None
        self.v_norms = None

        # QJL storage (prod mode only)
        self.k_qjl_packed = None
        self.k_res_norms = None
        self.v_qjl_packed = None
        self.v_res_norms = None

        # Decode buffer
        self._k_deq_buf = None
        self._v_deq_buf = None
        self._deq_offset = 0
        self._deq_alloc = 0

        # Quantizer state (lazy init)
        self._k_q = None
        self._v_q = None
        self._k_dim = None
        self._v_dim = None
        self._k_pdim = None
        self._v_pdim = None
        self._k_qjl_pdim = None
        self._v_qjl_pdim = None
        self._k_qjl_matrix = None
        self._v_qjl_matrix = None
        self._k_qjl_const = None
        self._v_qjl_const = None

    def _ensure_quantizer(self, k_dim, v_dim):
        if self._k_q is None:
            mse_bits = self.quant_bits - 1 if self.mode == "prod" else self.quant_bits
            self._k_q = _Quantizer(k_dim, mse_bits, self.seed)
            self._k_dim = k_dim
            self._k_pdim = packed_dim(k_dim, mse_bits)
            if self.mode == "prod":
                self._k_qjl_pdim = packed_dim(k_dim, 1)
                self._k_qjl_matrix = generate_qjl_matrix(k_dim, self.seed + 2000, self.qjl_mode)
                self._k_qjl_const = qjl_reconstruction_constant(k_dim, self.qjl_mode)
        if self._v_q is None:
            mse_bits = self.quant_bits - 1 if self.mode == "prod" else self.quant_bits
            self._v_q = _Quantizer(v_dim, mse_bits, self.seed + 1)
            self._v_dim = v_dim
            self._v_pdim = packed_dim(v_dim, mse_bits)
            if self.mode == "prod":
                self._v_qjl_pdim = packed_dim(v_dim, 1)
                self._v_qjl_matrix = generate_qjl_matrix(v_dim, self.seed + 2001, self.qjl_mode)
                self._v_qjl_const = qjl_reconstruction_constant(v_dim, self.qjl_mode)

    def _ensure_storage(self, B, H, num_new):
        prev = self.offset
        needed = prev + num_new
        if self.k_packed is None or needed > self.k_packed.shape[2]:
            n = ((needed + self.step - 1) // self.step) * self.step
            if self.k_packed is not None:
                extra = n - prev
                self.k_packed = mx.concatenate([self.k_packed[..., :prev, :], mx.zeros((B, H, extra, self._k_pdim), dtype=mx.uint32)], axis=2)
                self.k_norms = mx.concatenate([self.k_norms[..., :prev], mx.zeros((B, H, extra), dtype=mx.float32)], axis=2)
                self.v_packed = mx.concatenate([self.v_packed[..., :prev, :], mx.zeros((B, H, extra, self._v_pdim), dtype=mx.uint32)], axis=2)
                self.v_norms = mx.concatenate([self.v_norms[..., :prev], mx.zeros((B, H, extra), dtype=mx.float32)], axis=2)
                if self.mode == "prod":
                    self.k_qjl_packed = mx.concatenate([self.k_qjl_packed[..., :prev, :], mx.zeros((B, H, extra, self._k_qjl_pdim), dtype=mx.uint32)], axis=2)
                    self.k_res_norms = mx.concatenate([self.k_res_norms[..., :prev], mx.zeros((B, H, extra), dtype=mx.float32)], axis=2)
                    self.v_qjl_packed = mx.concatenate([self.v_qjl_packed[..., :prev, :], mx.zeros((B, H, extra, self._v_qjl_pdim), dtype=mx.uint32)], axis=2)
                    self.v_res_norms = mx.concatenate([self.v_res_norms[..., :prev], mx.zeros((B, H, extra), dtype=mx.float32)], axis=2)
            else:
                self.k_packed = mx.zeros((B, H, n, self._k_pdim), dtype=mx.uint32)
                self.k_norms = mx.zeros((B, H, n), dtype=mx.float32)
                self.v_packed = mx.zeros((B, H, n, self._v_pdim), dtype=mx.uint32)
                self.v_norms = mx.zeros((B, H, n), dtype=mx.float32)
                if self.mode == "prod":
                    self.k_qjl_packed = mx.zeros((B, H, n, self._k_qjl_pdim), dtype=mx.uint32)
                    self.k_res_norms = mx.zeros((B, H, n), dtype=mx.float32)
                    self.v_qjl_packed = mx.zeros((B, H, n, self._v_qjl_pdim), dtype=mx.uint32)
                    self.v_res_norms = mx.zeros((B, H, n), dtype=mx.float32)

    def _full_dequant_mse(self, packed, norms, q, dim, B, H, total, dtype):
        flat_p = packed[..., :total, :].reshape(-1, packed.shape[-1])
        flat_n = norms[..., :total].reshape(-1)
        out = packed_dequantize(flat_p, flat_n, q.centroids, q.signs, dim, self.quant_bits)
        return out.reshape(B, H, total, dim).astype(dtype)

    def _full_dequant_prod(self, mse_pk, qjl_pk, mse_nrm, res_nrm, q, qjl_mat, qjl_const,
                           dim, B, H, total, dtype):
        flat_mse = mse_pk[..., :total, :].reshape(-1, mse_pk.shape[-1])
        flat_qjl = qjl_pk[..., :total, :].reshape(-1, qjl_pk.shape[-1])
        flat_mse_n = mse_nrm[..., :total].reshape(-1)
        flat_res_n = res_nrm[..., :total].reshape(-1)
        out = packed_dequantize_prod(
            flat_mse, flat_qjl, flat_mse_n, flat_res_n,
            q.centroids, q.signs, qjl_mat, dim, self.quant_bits,
            qjl_mode=self.qjl_mode, qjl_const=qjl_const,
        )
        return out.reshape(B, H, total, dim).astype(dtype)

    def update_and_fetch(self, keys, values):
        B, H, S, k_dim = keys.shape
        v_dim = values.shape[3]
        self._ensure_quantizer(k_dim, v_dim)
        self._ensure_storage(B, H, S)
        prev = self.offset

        if self.mode == "prod":
            return self._update_and_fetch_prod(keys, values, B, H, S, k_dim, v_dim, prev)
        else:
            return self._update_and_fetch_mse(keys, values, B, H, S, k_dim, v_dim, prev)

    def _update_and_fetch_mse(self, keys, values, B, H, S, k_dim, v_dim, prev):
        k_pk, k_nrm = fused_quantize(keys.reshape(-1, k_dim), self._k_q.signs, self._k_q.boundaries, k_dim, self.quant_bits)
        k_pk = k_pk.reshape(B, H, S, self._k_pdim)
        v_pk, v_nrm = fused_quantize(values.reshape(-1, v_dim), self._v_q.signs, self._v_q.boundaries, v_dim, self.quant_bits)
        v_pk = v_pk.reshape(B, H, S, self._v_pdim)

        self.k_packed[..., prev:prev+S, :] = k_pk
        self.k_norms[..., prev:prev+S] = k_nrm.reshape(B, H, S)
        self.v_packed[..., prev:prev+S, :] = v_pk
        self.v_norms[..., prev:prev+S] = v_nrm.reshape(B, H, S)
        self.offset += S
        total = self.offset

        # Incremental decode
        if S <= 4 and self._v_deq_buf is not None and self._deq_offset == prev:
            if total > self._deq_alloc:
                na = ((total + self.step - 1) // self.step) * self.step
                self._k_deq_buf = mx.concatenate([self._k_deq_buf[..., :self._deq_offset, :],
                    mx.zeros((B, H, na - self._deq_alloc, k_dim), dtype=keys.dtype)], axis=2)
                self._v_deq_buf = mx.concatenate([self._v_deq_buf[..., :self._deq_offset, :],
                    mx.zeros((B, H, na - self._deq_alloc, v_dim), dtype=values.dtype)], axis=2)
                self._deq_alloc = na

            nk = dequant_fp16(k_pk.reshape(-1, self._k_pdim), k_nrm, self._k_q.centroids, self._k_q.signs, k_dim, self.quant_bits).reshape(B, H, S, k_dim)
            nv = dequant_fp16(v_pk.reshape(-1, self._v_pdim), v_nrm, self._v_q.centroids, self._v_q.signs, v_dim, self.quant_bits).reshape(B, H, S, v_dim)
            self._k_deq_buf[..., prev:total, :] = nk
            self._v_deq_buf[..., prev:total, :] = nv
            self._deq_offset = total
            return self._k_deq_buf[..., :total, :], self._v_deq_buf[..., :total, :]

        # Full dequant (prefill)
        all_k = self._full_dequant_mse(self.k_packed, self.k_norms, self._k_q, k_dim, B, H, total, keys.dtype)
        all_v = self._full_dequant_mse(self.v_packed, self.v_norms, self._v_q, v_dim, B, H, total, values.dtype)
        alloc = ((total + self.step - 1) // self.step) * self.step
        self._k_deq_buf = mx.zeros((B, H, alloc, k_dim), dtype=keys.dtype)
        self._v_deq_buf = mx.zeros((B, H, alloc, v_dim), dtype=values.dtype)
        self._k_deq_buf[..., :total, :] = all_k
        self._v_deq_buf[..., :total, :] = all_v
        self._deq_offset = total
        self._deq_alloc = alloc
        return all_k, all_v

    def _update_and_fetch_prod(self, keys, values, B, H, S, k_dim, v_dim, prev):
        # Algorithm 2: MSE at (b-1) bits + QJL on residual
        k_mse_pk, k_qjl_pk, k_mse_nrm, k_res_nrm = fused_quantize_prod(
            keys.reshape(-1, k_dim), self._k_q.signs, self._k_q.boundaries,
            self._k_q.centroids, self._k_qjl_matrix,
            k_dim, self.quant_bits, qjl_mode=self.qjl_mode,
        )
        v_mse_pk, v_qjl_pk, v_mse_nrm, v_res_nrm = fused_quantize_prod(
            values.reshape(-1, v_dim), self._v_q.signs, self._v_q.boundaries,
            self._v_q.centroids, self._v_qjl_matrix,
            v_dim, self.quant_bits, qjl_mode=self.qjl_mode,
        )

        k_mse_pk = k_mse_pk.reshape(B, H, S, self._k_pdim)
        k_qjl_pk = k_qjl_pk.reshape(B, H, S, self._k_qjl_pdim)
        v_mse_pk = v_mse_pk.reshape(B, H, S, self._v_pdim)
        v_qjl_pk = v_qjl_pk.reshape(B, H, S, self._v_qjl_pdim)

        self.k_packed[..., prev:prev+S, :] = k_mse_pk
        self.k_norms[..., prev:prev+S] = k_mse_nrm.reshape(B, H, S)
        self.k_qjl_packed[..., prev:prev+S, :] = k_qjl_pk
        self.k_res_norms[..., prev:prev+S] = k_res_nrm.reshape(B, H, S)
        self.v_packed[..., prev:prev+S, :] = v_mse_pk
        self.v_norms[..., prev:prev+S] = v_mse_nrm.reshape(B, H, S)
        self.v_qjl_packed[..., prev:prev+S, :] = v_qjl_pk
        self.v_res_norms[..., prev:prev+S] = v_res_nrm.reshape(B, H, S)
        self.offset += S
        total = self.offset

        # Incremental decode
        if S <= 4 and self._v_deq_buf is not None and self._deq_offset == prev:
            if total > self._deq_alloc:
                na = ((total + self.step - 1) // self.step) * self.step
                self._k_deq_buf = mx.concatenate([self._k_deq_buf[..., :self._deq_offset, :],
                    mx.zeros((B, H, na - self._deq_alloc, k_dim), dtype=keys.dtype)], axis=2)
                self._v_deq_buf = mx.concatenate([self._v_deq_buf[..., :self._deq_offset, :],
                    mx.zeros((B, H, na - self._deq_alloc, v_dim), dtype=values.dtype)], axis=2)
                self._deq_alloc = na

            nk = dequant_prod_fp16(
                k_mse_pk.reshape(-1, self._k_pdim), k_qjl_pk.reshape(-1, self._k_qjl_pdim),
                k_mse_nrm, k_res_nrm,
                self._k_q.centroids, self._k_q.signs, self._k_qjl_matrix,
                k_dim, self.quant_bits, qjl_mode=self.qjl_mode, qjl_const=self._k_qjl_const,
            ).reshape(B, H, S, k_dim)
            nv = dequant_prod_fp16(
                v_mse_pk.reshape(-1, self._v_pdim), v_qjl_pk.reshape(-1, self._v_qjl_pdim),
                v_mse_nrm, v_res_nrm,
                self._v_q.centroids, self._v_q.signs, self._v_qjl_matrix,
                v_dim, self.quant_bits, qjl_mode=self.qjl_mode, qjl_const=self._v_qjl_const,
            ).reshape(B, H, S, v_dim)
            self._k_deq_buf[..., prev:total, :] = nk
            self._v_deq_buf[..., prev:total, :] = nv
            self._deq_offset = total
            return self._k_deq_buf[..., :total, :], self._v_deq_buf[..., :total, :]

        # Full dequant (prefill)
        all_k = self._full_dequant_prod(
            self.k_packed, self.k_qjl_packed, self.k_norms, self.k_res_norms,
            self._k_q, self._k_qjl_matrix, self._k_qjl_const,
            k_dim, B, H, total, keys.dtype,
        )
        all_v = self._full_dequant_prod(
            self.v_packed, self.v_qjl_packed, self.v_norms, self.v_res_norms,
            self._v_q, self._v_qjl_matrix, self._v_qjl_const,
            v_dim, B, H, total, values.dtype,
        )
        alloc = ((total + self.step - 1) // self.step) * self.step
        self._k_deq_buf = mx.zeros((B, H, alloc, k_dim), dtype=keys.dtype)
        self._v_deq_buf = mx.zeros((B, H, alloc, v_dim), dtype=values.dtype)
        self._k_deq_buf[..., :total, :] = all_k
        self._v_deq_buf[..., :total, :] = all_v
        self._deq_offset = total
        self._deq_alloc = alloc
        return all_k, all_v

    def empty(self):
        return self.k_packed is None

    @property
    def nbytes(self):
        if self.k_packed is None:
            return 0
        t = self.offset
        total = (self.k_packed[..., :t, :].nbytes + self.v_packed[..., :t, :].nbytes +
                 self.k_norms[..., :t].nbytes + self.v_norms[..., :t].nbytes)
        if self.mode == "prod" and self.k_qjl_packed is not None:
            total += (self.k_qjl_packed[..., :t, :].nbytes + self.v_qjl_packed[..., :t, :].nbytes +
                      self.k_res_norms[..., :t].nbytes + self.v_res_norms[..., :t].nbytes)
        return total

    @property
    def state(self):
        if self.k_packed is None:
            return []
        t = self.offset
        base = [self.k_packed[..., :t, :], self.k_norms[..., :t],
                self.v_packed[..., :t, :], self.v_norms[..., :t]]
        if self.mode == "prod" and self.k_qjl_packed is not None:
            base.extend([self.k_qjl_packed[..., :t, :], self.k_res_norms[..., :t],
                         self.v_qjl_packed[..., :t, :], self.v_res_norms[..., :t]])
        return base

    @state.setter
    def state(self, v):
        if not v:
            return
        self.k_packed, self.k_norms, self.v_packed, self.v_norms = v[0], v[1], v[2], v[3]
        self.offset = self.k_packed.shape[2]
        if len(v) > 4:
            self.k_qjl_packed, self.k_res_norms = v[4], v[5]
            self.v_qjl_packed, self.v_res_norms = v[6], v[7]

    @property
    def meta_state(self):
        return (f"{self.offset},{self.quant_bits},{self.seed},"
                f"{self._k_dim or 0},{self._v_dim or 0},{self.mode},{self.qjl_mode}")

    @meta_state.setter
    def meta_state(self, v):
        parts = v.split(",")
        self.offset = int(parts[0])
        self.quant_bits = int(parts[1])
        self.seed = int(parts[2])
        self._k_dim = int(parts[3]) or None
        self._v_dim = int(parts[4]) or None
        if len(parts) > 5:
            self.mode = parts[5]
            self.qjl_mode = parts[6]

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def size(self):
        return self.offset

    def make_mask(self, *args, **kwargs):
        from mlx_lm.models.cache import create_attention_mask
        return create_attention_mask(*args, offset=self.offset, **kwargs)
