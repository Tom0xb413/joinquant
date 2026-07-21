"""EMA 数据与策略单元测试。"""

from __future__ import annotations

import unittest
from datetime import date, datetime, timezone

import numpy as np

from crypto_lab.data import Candle
from crypto_lab.ema_backtest import bars_per_year, run_ema_backtest
from crypto_lab.ema_data import (
    aggregate_4h_to_8h,
    candles_to_bar_series,
    deviation_rate,
    ema,
    ema_slope,
)
from crypto_lab.ema_strategies import EmaCrossBasic, EmaFullFilter


def rising_series(n: int = 400):
    close = 100 + np.arange(n) * 0.2 + np.sin(np.arange(n) / 12.0)
    ts = np.arange(n) * 4 * 3_600_000 + 1_600_000_000_000
    return candles_to_bar_series(
        "BTC-USDT",
        "4H",
        [
            Candle(
                timestamp_ms=int(ts[i]),
                open=float(close[i]),
                high=float(close[i]) + 1,
                low=float(close[i]) - 1,
                close=float(close[i]),
                volume_base=1.0,
                volume_quote=100.0,
            )
            for i in range(n)
        ],
    )


class EmaTests(unittest.TestCase):
    def test_ema_is_causal(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])
        out = ema(values, 2)
        self.assertEqual(out[0], 1.0)
        self.assertGreater(out[-1], out[0])

    def test_slope_and_deviation(self):
        values = np.linspace(100, 120, 30)
        slope = ema_slope(values, 5)
        self.assertTrue(np.isnan(slope[4]))
        self.assertGreater(slope[-1], 0)
        dev = deviation_rate(values, values * 0.9)
        self.assertAlmostEqual(float(dev[-1]), 1 / 0.9 - 1.0, places=6)

    def test_aggregate_4h_to_8h(self):
        base = 1_600_945_920_000  # aligned-ish
        base = base - (base % (8 * 3_600_000))
        candles = []
        for i in range(4):
            ts = base + i * 4 * 3_600_000
            candles.append(
                Candle(ts, 10 + i, 11 + i, 9 + i, 10.5 + i, 1.0, 10.0)
            )
        eight = aggregate_4h_to_8h(candles)
        self.assertEqual(len(eight), 2)
        self.assertEqual(eight[0].open, candles[0].open)
        self.assertEqual(eight[0].close, candles[1].close)

    def test_basic_cross_backtest_runs(self):
        series = rising_series()
        result = run_ema_backtest(series, EmaCrossBasic(), fee_rate=0.0, slippage_rate=0.0)
        self.assertEqual(result.timeframe, "4H")
        self.assertAlmostEqual(bars_per_year("4H"), 365 * 6)
        self.assertTrue(np.isfinite(result.equity).all())
        self.assertGreater(result.equity[-1], 0)

    def test_full_filter_stays_finite(self):
        series = rising_series(500)
        result = run_ema_backtest(series, EmaFullFilter(), fee_rate=0.001, slippage_rate=0.0005)
        metrics = result.metrics()
        self.assertTrue(np.isfinite(metrics.sharpe))


if __name__ == "__main__":
    unittest.main()
