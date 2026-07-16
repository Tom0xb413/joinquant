"""量价指标边界条件测试。"""

from __future__ import annotations

import unittest

import numpy as np

from crypto_lab.indicators import rsi


class IndicatorTests(unittest.TestCase):
    """验证指标不会在常量或单边行情上产生异常值。"""

    def test_flat_market_rsi_is_neutral(self):
        close = np.full((20, 2), 100.0)
        np.testing.assert_allclose(rsi(close, 19, 14), [50.0, 50.0])

    def test_one_way_market_rsi_has_expected_bounds(self):
        close = np.column_stack([np.arange(1.0, 21.0), np.arange(20.0, 0.0, -1.0)])
        np.testing.assert_allclose(rsi(close, 19, 14), [100.0, 0.0])


if __name__ == "__main__":
    unittest.main()
