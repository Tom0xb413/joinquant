"""研究统计方法测试。"""

from __future__ import annotations

import unittest

import numpy as np

from crypto_lab.research import _block_bootstrap_cagr


class ResearchTests(unittest.TestCase):
    """验证 Bootstrap 输出可复现且方向正确。"""

    def test_positive_constant_returns_have_positive_interval(self):
        summary = _block_bootstrap_cagr(
            np.full(140, 0.001),
            block_size=14,
            samples=100,
            seed=7,
        )
        self.assertEqual(summary["positive_resample_fraction"], 1.0)
        self.assertGreater(summary["cagr_ci_95"][0], 0)
        self.assertEqual(summary["samples"], 100)

    def test_too_short_sample_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "过短"):
            _block_bootstrap_cagr(np.zeros(10), block_size=14, samples=10)


if __name__ == "__main__":
    unittest.main()
