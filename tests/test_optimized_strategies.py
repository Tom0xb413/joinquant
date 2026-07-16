"""优化策略行为测试。"""

from __future__ import annotations

import unittest
from datetime import date, timedelta

import numpy as np

from crypto_lab.data import MarketData
from crypto_lab.indicators import inverse_volatility_weights
from crypto_lab.optimized_strategies import BtcDualMomentum, CoreSatelliteVolScaled


def rising_market(days: int = 250) -> MarketData:
    """构造 BTC 明确上升、山寨分化的合成行情。"""

    dates = tuple(date(2023, 1, 1) + timedelta(days=index) for index in range(days))
    symbols = ("BTC-USDT", "ETH-USDT", "SOL-USDT", "ADA-USDT")
    close = np.zeros((days, 4), dtype=float)
    close[:, 0] = 100 + np.arange(days) * 0.5
    close[:, 1] = 100 + np.arange(days) * 0.4
    close[:, 2] = 100 + np.arange(days) * 0.8
    close[:, 3] = 100 - np.arange(days) * 0.1
    volume = np.full_like(close, 1_000_000.0)
    return MarketData(dates, symbols, close.copy(), close.copy(), close.copy(), close, volume)


class OptimizedStrategyTests(unittest.TestCase):
    """验证门控与加权逻辑的关键性质。"""

    def test_dual_momentum_goes_to_cash_below_btc_trend(self):
        data = rising_market()
        # 把后半段 BTC 压到均线下方
        data.close[200:, 0] = 50.0
        strategy = BtcDualMomentum(regime_window=100, lookback=60, top_k=2, rebalance_days=1)
        weights = strategy.target_weights(data, 220, np.zeros(4))
        np.testing.assert_allclose(weights, np.zeros(4))

    def test_dual_momentum_prefers_positive_relative_winners(self):
        data = rising_market()
        strategy = BtcDualMomentum(regime_window=100, lookback=60, top_k=2, rebalance_days=1)
        weights = strategy.target_weights(data, 220, np.zeros(4))
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertGreater(weights[2], 0.0)  # SOL 最强
        self.assertEqual(weights[3], 0.0)  # ADA 负动量

    def test_inverse_volatility_weights_sum_to_exposure(self):
        vols = np.array([0.2, 0.4, 0.8])
        weights = inverse_volatility_weights(vols, np.array([0, 1]), 3, gross_exposure=0.7)
        self.assertAlmostEqual(float(weights.sum()), 0.7)
        self.assertGreater(weights[0], weights[1])

    def test_core_satellite_respects_rebalance_schedule(self):
        data = rising_market()
        strategy = CoreSatelliteVolScaled(rebalance_days=30)
        first = strategy.target_weights(data, 210, np.zeros(4))
        held = strategy.target_weights(data, 211, first)
        np.testing.assert_allclose(held, first)


if __name__ == "__main__":
    unittest.main()
