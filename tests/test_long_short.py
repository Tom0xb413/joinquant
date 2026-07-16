"""多空回测与加密增强策略测试。"""

from __future__ import annotations

import unittest
from datetime import date, timedelta

import numpy as np

from crypto_lab.backtest import run_backtest
from crypto_lab.crypto_alpha import BtcGateAltHedge, BtcTrendTopMomentum
from crypto_lab.data import MarketData
from crypto_lab.long_short import LongShortLimits, run_long_short_backtest


class LongOnlyStub:
    name = "long"
    source_ids = ("t",)

    def target_weights(self, data, signal_index, previous):
        weights = np.zeros(len(data.symbols))
        weights[0] = 1.0
        return weights


class ShortStub:
    name = "short"
    source_ids = ("t",)

    def target_weights(self, data, signal_index, previous):
        weights = np.zeros(len(data.symbols))
        weights[1] = -0.5
        weights[0] = 0.5
        return weights


def sample_market(days: int = 260) -> MarketData:
    dates = tuple(date(2023, 1, 1) + timedelta(days=i) for i in range(days))
    symbols = ("BTC-USDT", "ETH-USDT", "SOL-USDT")
    close = np.column_stack(
        [
            100 + np.arange(days) * 0.4,
            100 + np.arange(days) * 0.2,
            100 + np.sin(np.arange(days) / 8.0) * 5,
        ]
    )
    volume = np.full_like(close, 1e6)
    return MarketData(dates, symbols, close.copy(), close.copy(), close.copy(), close, volume)


class LongShortTests(unittest.TestCase):
    def test_short_weights_are_accepted(self):
        result = run_long_short_backtest(sample_market(), ShortStub(), fee_rate=0.0, slippage_rate=0.0)
        self.assertTrue(np.any(result.weights < 0))

    def test_excessive_gross_is_clipped(self):
        class TooGross(ShortStub):
            def target_weights(self, data, signal_index, previous):
                return np.array([1.0, -1.0, 0.0])

        result = run_long_short_backtest(
            sample_market(),
            TooGross(),
            fee_rate=0.0,
            slippage_rate=0.0,
            limits=LongShortLimits(max_gross_exposure=1.5, max_net_exposure=1.5, max_short_exposure=1.0),
        )
        gross = np.sum(np.abs(result.weights[1:]), axis=1)
        self.assertTrue(np.all(gross <= 1.5 + 1e-9))

    def test_long_only_engine_still_rejects_shorts(self):
        with self.assertRaisesRegex(ValueError, "只允许多头"):
            run_backtest(sample_market(), ShortStub())

    def test_trend_top_momentum_cash_when_btc_breaks(self):
        data = sample_market()
        data.close[200:, 0] = 20.0
        strategy = BtcTrendTopMomentum(trend_window=100, lookback=60, rebalance_days=1)
        weights = strategy.target_weights(data, 230, np.zeros(3))
        np.testing.assert_allclose(weights, np.zeros(3))

    def test_hedge_strategy_can_go_short_off_trend(self):
        data = sample_market()
        data.close[200:, 0] = 20.0
        strategy = BtcGateAltHedge(trend_window=100, lookback=30, rebalance_days=1, off_short_weight=0.4)
        weights = strategy.target_weights(data, 230, np.zeros(3))
        self.assertLess(float(weights.sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
