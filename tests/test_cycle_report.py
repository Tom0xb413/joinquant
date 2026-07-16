"""全周期报告与交易提取测试。"""

from __future__ import annotations

import unittest
from datetime import date, timedelta

import numpy as np

from crypto_lab.cycle_report import (
    drawdown_series,
    extract_trades,
    run_cycle_report,
)
from crypto_lab.data import MarketData
from crypto_lab.long_short import run_long_short_backtest
from crypto_lab.crypto_alpha import BtcTrendTopMomentum


def sample_market(days: int = 420) -> MarketData:
    dates = tuple(date(2021, 1, 1) + timedelta(days=i) for i in range(days))
    symbols = ("BTC-USDT", "ETH-USDT", "SOL-USDT")
    trend = np.linspace(100, 180, days)
    close = np.column_stack(
        [
            trend + np.sin(np.arange(days) / 11.0) * 3,
            trend * 0.8 + np.cos(np.arange(days) / 9.0) * 4,
            trend * 0.5 + np.sin(np.arange(days) / 7.0) * 6,
        ]
    )
    # 制造一段明显回撤，便于回撤序列测试
    close[250:320, :] *= np.linspace(1.0, 0.7, 70)[:, None]
    volume = np.full_like(close, 1e6)
    return MarketData(dates, symbols, close.copy(), close.copy(), close.copy(), close, volume)


class CycleReportTests(unittest.TestCase):
    def test_drawdown_series_is_non_positive(self):
        equity = np.array([1.0, 1.2, 1.1, 0.9, 1.0])
        dd = drawdown_series(equity)
        self.assertTrue(np.all(dd <= 1e-12))
        self.assertAlmostEqual(float(dd.min()), 0.9 / 1.2 - 1.0, places=6)

    def test_extract_trades_from_weight_changes(self):
        data = sample_market()
        strategy = BtcTrendTopMomentum(
            trend_window=50,
            lookback=30,
            top_k=1,
            rebalance_days=7,
            vol_target=0.4,
        )
        result = run_long_short_backtest(data, strategy, fee_rate=0.0, slippage_rate=0.0)
        trades = extract_trades(result, data, min_delta=0.01)
        self.assertGreater(len(trades), 0)
        self.assertTrue(all(abs(trade.weight_delta) >= 0.01 for trade in trades))

    def test_cycle_report_contains_regimes_and_trades(self):
        data = sample_market(500)
        report = run_cycle_report(data, fee_rate=0.0, slippage_rate=0.0)
        self.assertIn("ranking_full_cycle", report)
        self.assertIn("regimes", report)
        self.assertIn("trades", report)
        self.assertIn("btc_breadth_top_momentum", report["trades"])
        self.assertGreater(len(report["regimes"]), 3)


if __name__ == "__main__":
    unittest.main()
