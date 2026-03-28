"""Outlier channel splitting for fractional bit rates.

Wraps two TurboQuantKVCache instances: one for outlier channels (higher bits)
and one for regular channels (lower bits). Achieves fractional effective bit
rates like 2.5 or 3.5 bits per channel.

Example: n_outlier=32, bits_outlier=3, bits_regular=2 with dim=128
  effective = (32*3 + 96*2) / 128 = 2.5 bits
"""

import mlx.core as mx
from mlx_lm.models.turboquant_cache import TurboQuantKVCache


class OutlierSplitKVCache:
    """KV cache with outlier/regular channel splitting.

    First ``n_outlier`` channels are quantized at ``bits_outlier`` bits,
    remaining channels at ``bits_regular`` bits.

    Args:
        bits_outlier: Bit width for outlier channels.
        bits_regular: Bit width for regular channels.
        n_outlier: Number of outlier channels (first N dimensions).
        mode: 'mse' or 'prod' (passed to both sub-caches).
        qjl_mode: 'gaussian' or 'hadamard' (passed to both sub-caches).
        seed: Random seed.
    """

    def __init__(self, bits_outlier: int = 3, bits_regular: int = 2,
                 n_outlier: int = 32, mode: str = "mse",
                 qjl_mode: str = "gaussian", seed: int = 42):
        self.bits_outlier = bits_outlier
        self.bits_regular = bits_regular
        self.n_outlier = n_outlier
        self.mode = mode
        self.qjl_mode = qjl_mode
        self.seed = seed
        self._dim = None

        self.cache_outlier = TurboQuantKVCache(
            bits=bits_outlier, mode=mode, qjl_mode=qjl_mode, seed=seed,
        )
        self.cache_regular = TurboQuantKVCache(
            bits=bits_regular, mode=mode, qjl_mode=qjl_mode, seed=seed + 100,
        )

    def _split(self, tensor):
        return tensor[..., :self.n_outlier], tensor[..., self.n_outlier:]

    def _merge(self, outlier, regular, dtype):
        return mx.concatenate([outlier.astype(dtype), regular.astype(dtype)], axis=-1)

    def update_and_fetch(self, keys, values):
        if self._dim is None:
            self._dim = keys.shape[-1]

        k_out, k_reg = self._split(keys)
        v_out, v_reg = self._split(values)

        k_out_deq, v_out_deq = self.cache_outlier.update_and_fetch(k_out, v_out)
        k_reg_deq, v_reg_deq = self.cache_regular.update_and_fetch(k_reg, v_reg)

        return self._merge(k_out_deq, k_reg_deq, keys.dtype), self._merge(v_out_deq, v_reg_deq, values.dtype)

    @property
    def offset(self):
        return self.cache_outlier.offset

    def empty(self):
        return self.cache_outlier.empty()

    @property
    def nbytes(self):
        return self.cache_outlier.nbytes + self.cache_regular.nbytes

    def effective_bits(self):
        if self._dim is None:
            return 0.0
        n_reg = self._dim - self.n_outlier
        return (self.n_outlier * self.bits_outlier + n_reg * self.bits_regular) / self._dim

    @property
    def state(self):
        s_out = self.cache_outlier.state
        s_reg = self.cache_regular.state
        if not s_out:
            return []
        return s_out + s_reg

    @state.setter
    def state(self, v):
        if not v:
            return
        n_out = 4 if self.mode == "mse" else 8
        self.cache_outlier.state = v[:n_out]
        self.cache_regular.state = v[n_out:]

    @property
    def meta_state(self):
        return (f"{self.bits_outlier},{self.bits_regular},{self.n_outlier},"
                f"{self.mode},{self.qjl_mode},{self.seed},{self._dim or 0}|"
                f"{self.cache_outlier.meta_state}|{self.cache_regular.meta_state}")

    @meta_state.setter
    def meta_state(self, v):
        parts = v.split("|")
        header = parts[0].split(",")
        self.bits_outlier = int(header[0])
        self.bits_regular = int(header[1])
        self.n_outlier = int(header[2])
        self.mode = header[3]
        self.qjl_mode = header[4]
        self.seed = int(header[5])
        self._dim = int(header[6]) or None
        if len(parts) > 1:
            self.cache_outlier.meta_state = parts[1]
        if len(parts) > 2:
            self.cache_regular.meta_state = parts[2]

    def is_trimmable(self):
        return True

    def trim(self, n):
        n1 = self.cache_outlier.trim(n)
        self.cache_regular.trim(n)
        return n1

    def size(self):
        return self.cache_outlier.size()

    def make_mask(self, *args, **kwargs):
        return self.cache_outlier.make_mask(*args, **kwargs)
