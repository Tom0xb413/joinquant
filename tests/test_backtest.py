"""回测时序、成本和指标测试。"""

from __future__ import annotations

import unittest
from datetime import date, timedelta

import numpy as np

from crypto_lab.backtest import run_backtest
from crypto_lab.data import MarketData


class FixedStrategy:
    """始终满仓第一列的确定性测试策略。"""

    name = "fixed"
    source_ids = ("test",)

    def target_weights(self, data, signal_index, previous):
        weights = np.zeros(len(data.symbols))
        weights[0] = 1.0
        return weights


class InvalidStrategy:
    """故意输出杠杆权重，用于验证输入保护。"""

    name = "invalid"
    source_ids = ("test",)

    def target_weights(self, data, signal_index, previous):
        return np.array([1.1, 0.0])


def market_data() -> MarketData:
    """构造两资产、四天的无缺失行情。"""

    dates = tuple(date(2024, 1, 1) + timedelta(days=index) for index in range(4))
    close = np.array([[100.0, 100.0], [110.0, 100.0], [99.0, 100.0], [108.9, 100.0]])
    open_prices = close.copy()
    volume = np.full_like(close, 1_000_000.0)
    return MarketData(dates, ("BTC-USDT", "ETH-USDT"), open_prices, close, close, close, volume)


class BacktestTests(unittest.TestCase):
    """验证引擎的关键不变量。"""

    def test_signal_is_applied_to_next_daily_return(self):
        result = run_backtest(market_data(), FixedStrategy(), fee_rate=0.0, slippage_rate=0.0)
        np.testing.assert_allclose(result.daily_returns, [0.0, 0.1, -0.1, 0.1])
        self.assertAlmostEqual(result.equity[-1], 1.089)

    def test_initial_trade_cost_is_charged_once(self):
        result = run_backtest(market_data(), FixedStrategy(), fee_rate=0.001, slippage_rate=0.0005)
        self.assertAlmostEqual(result.turnover[1], 1.0)
        self.assertAlmostEqual(result.costs[1], 0.0015)
        self.assertLess(result.costs[2], 1e-12)

    def test_leverage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "不能超过 1"):
            run_backtest(market_data(), InvalidStrategy())

    def test_metrics_use_requested_range(self):
        result = run_backtest(market_data(), FixedStrategy(), fee_rate=0.0, slippage_rate=0.0)
        metrics = result.metrics(date(2024, 1, 2), date(2024, 1, 4))
        self.assertEqual(metrics.observations, 3)
        self.assertEqual(metrics.start, "2024-01-02")
        self.assertEqual(metrics.end, "2024-01-04")


if __name__ == "__main__":
    unittest.main()

