"""Tests for TurboQuant KV cache compression.

Covers packing, Algorithm 1 (MSE), Algorithm 2 (prod/QJL), outlier splitting,
and cache interface compliance.
"""

import math
import unittest
import mlx.core as mx


class TestPacking(unittest.TestCase):
    """Test bit-packing roundtrip for all supported bit widths."""

    def _roundtrip(self, bits, dim):
        from mlx_lm.models.turboquant_packing import pack_indices, unpack_indices
        n_vecs = 8
        max_val = (1 << bits) - 1
        indices = mx.random.randint(0, max_val + 1, shape=(n_vecs, dim)).astype(mx.uint8)
        packed = pack_indices(indices, bits)
        unpacked = unpack_indices(packed, bits, dim)
        self.assertTrue(mx.array_equal(indices, unpacked),
                        f"Roundtrip failed for bits={bits}, dim={dim}")

    def test_pack_unpack_1bit(self):
        self._roundtrip(1, 128)

    def test_pack_unpack_2bit(self):
        self._roundtrip(2, 128)

    def test_pack_unpack_3bit(self):
        self._roundtrip(3, 128)

    def test_pack_unpack_4bit(self):
        self._roundtrip(4, 128)

    def test_pack_unpack_non_aligned(self):
        # dim=100 doesn't evenly divide into groups of 10 (3-bit)
        self._roundtrip(3, 100)

    def test_packed_dim_formula(self):
        from mlx_lm.models.turboquant_packing import packed_dim, VALS_PER_WORD
        for bits in [1, 2, 3, 4]:
            vpw = VALS_PER_WORD[bits]
            for dim in [64, 100, 128, 256]:
                pdim = packed_dim(dim, bits)
                self.assertEqual(pdim, (dim + vpw - 1) // vpw)


class TestRotation(unittest.TestCase):
    """Test Hadamard transform and QJL matrix generation."""

    def test_wht_roundtrip(self):
        from mlx_lm.models.turboquant_rotation import walsh_hadamard_transform
        x = mx.random.normal(shape=(4, 128))
        y = walsh_hadamard_transform(x)
        # WHT is self-inverse (up to scaling): WHT(WHT(x)) = x
        z = walsh_hadamard_transform(y)
        self.assertTrue(mx.allclose(x, z, atol=1e-5),
                        "WHT is not self-inverse")

    def test_wht_preserves_norm(self):
        from mlx_lm.models.turboquant_rotation import walsh_hadamard_transform
        x = mx.random.normal(shape=(128,))
        y = walsh_hadamard_transform(x)
        self.assertTrue(mx.allclose(mx.sqrt(mx.sum(x**2)), mx.sqrt(mx.sum(y**2)), atol=1e-5),
                        "WHT should preserve L2 norm")

    def test_random_sign_deterministic(self):
        from mlx_lm.models.turboquant_rotation import random_diagonal_sign
        s1 = random_diagonal_sign(128, seed=42)
        s2 = random_diagonal_sign(128, seed=42)
        self.assertTrue(mx.array_equal(s1, s2))

    def test_generate_qjl_matrix_gaussian(self):
        from mlx_lm.models.turboquant_rotation import generate_qjl_matrix
        S = generate_qjl_matrix(64, seed=42, mode="gaussian")
        self.assertEqual(S.shape, (64, 64))
        self.assertEqual(S.dtype, mx.float32)

    def test_generate_qjl_matrix_hadamard(self):
        from mlx_lm.models.turboquant_rotation import generate_qjl_matrix
        s = generate_qjl_matrix(64, seed=42, mode="hadamard")
        self.assertEqual(s.shape, (64,))
        # Should be all +1 or -1
        self.assertTrue(mx.all(mx.abs(s) == 1.0).item())

    def test_qjl_reconstruction_constant(self):
        from mlx_lm.models.turboquant_rotation import qjl_reconstruction_constant
        d = 128
        c_gauss = qjl_reconstruction_constant(d, "gaussian")
        c_had = qjl_reconstruction_constant(d, "hadamard")
        self.assertAlmostEqual(c_gauss, math.sqrt(math.pi / 2.0) / d, places=8)
        self.assertAlmostEqual(c_had, math.sqrt(math.pi / (2.0 * d)), places=8)


class TestAlgorithm1(unittest.TestCase):
    """Test MSE-optimal quantization (Algorithm 1)."""

    def test_cache_mse_basic(self):
        from mlx_lm.models.turboquant_cache import TurboQuantKVCache
        cache = TurboQuantKVCache(bits=3, mode="mse")
        B, H, S, D = 1, 4, 16, 128
        keys = mx.random.normal(shape=(B, H, S, D)).astype(mx.float16)
        values = mx.random.normal(shape=(B, H, S, D)).astype(mx.float16)
        k_out, v_out = cache.update_and_fetch(keys, values)
        self.assertEqual(k_out.shape, (B, H, S, D))
        self.assertEqual(v_out.shape, (B, H, S, D))
        self.assertEqual(cache.offset, S)

    def test_cache_incremental_decode(self):
        from mlx_lm.models.turboquant_cache import TurboQuantKVCache
        cache = TurboQuantKVCache(bits=3, mode="mse")
        B, H, D = 1, 4, 128

        # Prefill
        keys = mx.random.normal(shape=(B, H, 32, D)).astype(mx.float16)
        values = mx.random.normal(shape=(B, H, 32, D)).astype(mx.float16)
        cache.update_and_fetch(keys, values)
        mx.eval(cache.k_packed, cache.v_packed)

        # Decode step
        k1 = mx.random.normal(shape=(B, H, 1, D)).astype(mx.float16)
        v1 = mx.random.normal(shape=(B, H, 1, D)).astype(mx.float16)
        k_out, v_out = cache.update_and_fetch(k1, v1)
        self.assertEqual(k_out.shape, (B, H, 33, D))
        self.assertEqual(cache.offset, 33)

    def test_quantize_dequant_mse_accuracy(self):
        from mlx_lm.models.turboquant_cache import TurboQuantKVCache
        cache = TurboQuantKVCache(bits=3, mode="mse")
        B, H, S, D = 1, 2, 8, 128
        keys = mx.random.normal(shape=(B, H, S, D)).astype(mx.float16)
        values = mx.random.normal(shape=(B, H, S, D)).astype(mx.float16)
        k_out, v_out = cache.update_and_fetch(keys, values)
        mx.eval(k_out, v_out)

        # MSE should be bounded (not zero, but not huge)
        k_mse = mx.mean((keys.astype(mx.float32) - k_out.astype(mx.float32)) ** 2).item()
        self.assertLess(k_mse, 0.5, "MSE too large for 3-bit quantization")
        self.assertGreater(k_mse, 0.0, "MSE should not be zero (lossy)")

    def test_storage_allocation_fixed(self):
        from mlx_lm.models.turboquant_cache import TurboQuantKVCache
        cache = TurboQuantKVCache(bits=3, mode="mse")
        B, H, D = 1, 4, 128

        # Fill to 200 tokens
        keys = mx.random.normal(shape=(B, H, 200, D)).astype(mx.float16)
        values = mx.random.normal(shape=(B, H, 200, D)).astype(mx.float16)
        cache.update_and_fetch(keys, values)
        mx.eval(cache.k_packed)

        # Storage should be 256 (one step block), not 200+256=456
        self.assertEqual(cache.k_packed.shape[2], 256)

        # Add more to trigger expansion
        k2 = mx.random.normal(shape=(B, H, 100, D)).astype(mx.float16)
        v2 = mx.random.normal(shape=(B, H, 100, D)).astype(mx.float16)
        cache.update_and_fetch(k2, v2)
        mx.eval(cache.k_packed)

        # Should be 512 (two step blocks), not 256+512=768
        self.assertEqual(cache.k_packed.shape[2], 512)


class TestAlgorithm2(unittest.TestCase):
    """Test inner-product-optimal quantization (Algorithm 2 / QJL)."""

    def test_cache_prod_basic(self):
        from mlx_lm.models.turboquant_cache import TurboQuantKVCache
        cache = TurboQuantKVCache(bits=3, mode="prod", qjl_mode="gaussian")
        B, H, S, D = 1, 2, 8, 64
        keys = mx.random.normal(shape=(B, H, S, D)).astype(mx.float16)
        values = mx.random.normal(shape=(B, H, S, D)).astype(mx.float16)
        k_out, v_out = cache.update_and_fetch(keys, values)
        mx.eval(k_out, v_out)
        self.assertEqual(k_out.shape, (B, H, S, D))
        self.assertIsNotNone(cache.k_qjl_packed)

    def test_cache_prod_hadamard(self):
        from mlx_lm.models.turboquant_cache import TurboQuantKVCache
        cache = TurboQuantKVCache(bits=3, mode="prod", qjl_mode="hadamard")
        B, H, S, D = 1, 2, 8, 64
        keys = mx.random.normal(shape=(B, H, S, D)).astype(mx.float16)
        values = mx.random.normal(shape=(B, H, S, D)).astype(mx.float16)
        k_out, v_out = cache.update_and_fetch(keys, values)
        mx.eval(k_out, v_out)
        self.assertEqual(k_out.shape, (B, H, S, D))

    def test_qjl_inner_product_unbiased(self):
        """Statistical test: E[<y, dequant(quant(x))>] should approximate <y, x>.

        Uses unit-normalized vectors to reduce variance and a z-test at 3-sigma
        to avoid false failures while still catching real bias.
        """
        from mlx_lm.models.turboquant_cache import TurboQuantKVCache

        n_trials = 500
        D = 128
        errors = []

        for i in range(n_trials):
            cache = TurboQuantKVCache(bits=3, mode="prod", qjl_mode="gaussian", seed=i)
            # Use unit-normalized vectors to match paper's assumption
            x = mx.random.normal(shape=(1, 1, 1, D)).astype(mx.float32)
            x = x / mx.sqrt(mx.sum(x ** 2, keepdims=True))
            y = mx.random.normal(shape=(1, 1, 1, D)).astype(mx.float32)
            y = y / mx.sqrt(mx.sum(y ** 2, keepdims=True))
            x = x.astype(mx.float16)
            y = y.astype(mx.float16)

            x_deq, _ = cache.update_and_fetch(x, x)
            mx.eval(x_deq)

            ip_true = mx.sum(x.astype(mx.float32) * y.astype(mx.float32)).item()
            ip_est = mx.sum(x_deq.astype(mx.float32) * y.astype(mx.float32)).item()
            errors.append(ip_est - ip_true)

        mean_error = sum(errors) / len(errors)
        std_error = (sum((e - mean_error) ** 2 for e in errors) / len(errors)) ** 0.5
        se = std_error / (len(errors) ** 0.5)

        # z-test: mean should be within 3 sigma of 0
        z_score = abs(mean_error) / max(se, 1e-10)
        self.assertLess(z_score, 3.0,
                        f"QJL inner product appears biased: mean={mean_error:.4f}, "
                        f"SE={se:.4f}, z={z_score:.2f}")

    def test_cache_prod_state_save_load(self):
        from mlx_lm.models.turboquant_cache import TurboQuantKVCache
        cache = TurboQuantKVCache(bits=3, mode="prod", qjl_mode="gaussian")
        B, H, S, D = 1, 2, 4, 64
        keys = mx.random.normal(shape=(B, H, S, D)).astype(mx.float16)
        values = mx.random.normal(shape=(B, H, S, D)).astype(mx.float16)
        cache.update_and_fetch(keys, values)
        mx.eval(cache.k_packed, cache.k_qjl_packed)

        state = cache.state
        meta = cache.meta_state

        # state should have 8 arrays for prod mode
        self.assertEqual(len(state), 8)
        self.assertIn("prod", meta)

        # Restore
        cache2 = TurboQuantKVCache(bits=3, mode="prod", qjl_mode="gaussian")
        cache2.meta_state = meta
        cache2.state = state
        self.assertEqual(cache2.offset, S)
        self.assertTrue(mx.array_equal(cache.k_packed[..., :S, :], cache2.k_packed[..., :S, :]))

    def test_prod_storage_sizes(self):
        from mlx_lm.models.turboquant_cache import TurboQuantKVCache
        cache = TurboQuantKVCache(bits=3, mode="prod", qjl_mode="gaussian")
        B, H, S, D = 1, 2, 8, 128
        keys = mx.random.normal(shape=(B, H, S, D)).astype(mx.float16)
        values = mx.random.normal(shape=(B, H, S, D)).astype(mx.float16)
        cache.update_and_fetch(keys, values)
        mx.eval(cache.k_packed)

        nbytes = cache.nbytes
        self.assertGreater(nbytes, 0)
        # Should be less than FP16 size
        fp16_size = 2 * B * H * S * D * 2  # 2 for K+V
        self.assertLess(nbytes, fp16_size)


class TestOutlierSplit(unittest.TestCase):
    """Test outlier channel splitting for fractional bit rates."""

    def test_effective_bitrate(self):
        from mlx_lm.models.turboquant_outlier import OutlierSplitKVCache
        cache = OutlierSplitKVCache(bits_outlier=3, bits_regular=2, n_outlier=32)
        B, H, S, D = 1, 2, 8, 128
        keys = mx.random.normal(shape=(B, H, S, D)).astype(mx.float16)
        values = mx.random.normal(shape=(B, H, S, D)).astype(mx.float16)
        cache.update_and_fetch(keys, values)

        eff = cache.effective_bits()
        expected = (32 * 3 + 96 * 2) / 128.0
        self.assertAlmostEqual(eff, expected, places=4)

    def test_outlier_cache_basic(self):
        from mlx_lm.models.turboquant_outlier import OutlierSplitKVCache
        cache = OutlierSplitKVCache(bits_outlier=3, bits_regular=2, n_outlier=32)
        B, H, S, D = 1, 2, 8, 128
        keys = mx.random.normal(shape=(B, H, S, D)).astype(mx.float16)
        values = mx.random.normal(shape=(B, H, S, D)).astype(mx.float16)
        k_out, v_out = cache.update_and_fetch(keys, values)
        mx.eval(k_out, v_out)
        self.assertEqual(k_out.shape, (B, H, S, D))
        self.assertEqual(cache.offset, S)

    def test_outlier_with_prod_mode(self):
        from mlx_lm.models.turboquant_outlier import OutlierSplitKVCache
        cache = OutlierSplitKVCache(
            bits_outlier=3, bits_regular=2, n_outlier=32,
            mode="prod", qjl_mode="hadamard",
        )
        B, H, S, D = 1, 2, 4, 128
        keys = mx.random.normal(shape=(B, H, S, D)).astype(mx.float16)
        values = mx.random.normal(shape=(B, H, S, D)).astype(mx.float16)
        k_out, v_out = cache.update_and_fetch(keys, values)
        mx.eval(k_out, v_out)
        self.assertEqual(k_out.shape, (B, H, S, D))

    def test_gather_scatter_preserves_dim(self):
        from mlx_lm.models.turboquant_outlier import OutlierSplitKVCache
        cache = OutlierSplitKVCache(bits_outlier=4, bits_regular=3, n_outlier=64)
        D = 128
        x = mx.random.normal(shape=(1, 1, 1, D))
        out, reg = cache._split(x)
        self.assertEqual(out.shape[-1], 64)
        self.assertEqual(reg.shape[-1], 64)
        merged = cache._merge(out, reg, x.dtype)
        self.assertTrue(mx.array_equal(x, merged))

    def test_outlier_state_roundtrip(self):
        from mlx_lm.models.turboquant_outlier import OutlierSplitKVCache
        cache = OutlierSplitKVCache(bits_outlier=3, bits_regular=2, n_outlier=32)
        B, H, S, D = 1, 2, 4, 128
        keys = mx.random.normal(shape=(B, H, S, D)).astype(mx.float16)
        values = mx.random.normal(shape=(B, H, S, D)).astype(mx.float16)
        cache.update_and_fetch(keys, values)
        mx.eval(cache.cache_outlier.k_packed)

        meta = cache.meta_state
        self.assertIn("|", meta)

        cache2 = OutlierSplitKVCache(bits_outlier=3, bits_regular=2, n_outlier=32)
        cache2.meta_state = meta
        self.assertEqual(cache2.bits_outlier, 3)
        self.assertEqual(cache2.bits_regular, 2)


if __name__ == "__main__":
    unittest.main()
