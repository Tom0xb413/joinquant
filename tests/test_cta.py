"""机构 CTA 指标与聚合测试。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from crypto_lab.cta_data import PanelData, aggregate_4h_to_12h
from crypto_lab.cta_engine import FastInstitutionalCTA, run_cta_backtest
from crypto_lab.cta_indicators import kdj, macd, rsi
from crypto_lab.data import Candle


class CtaIndicatorTests(unittest.TestCase):
    def test_rsi_bounds(self):
        close = np.linspace(100, 130, 80)
        values = rsi(close, 14)
        self.assertTrue(np.nanmax(values[20:]) <= 100)
        self.assertTrue(np.nanmin(values[20:]) >= 0)

    def test_macd_shape(self):
        close = np.cumsum(np.random.default_rng(0).normal(0, 1, 200)) + 100
        line, sig, hist = macd(close)
        self.assertEqual(line.shape, close.shape)
        self.assertTrue(np.isfinite(hist[-1]))

    def test_kdj_finite_late(self):
        high = np.linspace(110, 140, 60)
        low = high - 5
        close = high - 2
        k, d, j = kdj(high, low, close)
        self.assertTrue(np.isfinite(k[-1]))
        self.assertTrue(np.isfinite(d[-1]))

    def test_aggregate_12h(self):
        base = 1_609_459_200_000
        base -= base % (12 * 3_600_000)
        candles = []
        for i in range(6):
            ts = base + i * 4 * 3_600_000
            candles.append(Candle(ts, 10 + i, 11 + i, 9 + i, 10.5 + i, 1.0, 10.0))
        out = aggregate_4h_to_12h(candles)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].open, candles[0].open)
        self.assertEqual(out[0].close, candles[2].close)

    def test_higher_tf_mapping_has_no_lookahead(self):
        from pathlib import Path

        from crypto_lab.cta_data import CTA_TOP15, TF_MS, load_panel, map_higher_tf_to_base

        data_dir = Path("data/okx_cta")
        if not (data_dir / "BTC-USDT_4H.csv").exists():
            self.skipTest("CTA 数据未准备")
        p4 = load_panel(data_dir, "4H", CTA_TOP15[:3])
        p1 = load_panel(data_dir, "1D", CTA_TOP15[:3])
        mapped = map_higher_tf_to_base(p4, p1)["index"]
        for i in range(0, len(mapped), 37):
            i1 = int(mapped[i])
            if i1 < 0:
                continue
            self.assertLessEqual(
                int(p1.timestamps_ms[i1]) + TF_MS["1D"],
                int(p4.timestamps_ms[i]) + TF_MS["4H"],
            )

    def test_rebalance_schedule_uses_absolute_utc_phase(self):
        strategy = object.__new__(FastInstitutionalCTA)
        base = 1_609_459_200_000
        strategy.panel_4h = SimpleNamespace(
            timestamps_ms=np.asarray([base + i * 4 * 3_600_000 for i in range(13)])
        )
        strategy.rebalance_bars = 12
        strategy.rebalance_phase = 0
        self.assertTrue(strategy._is_rebalance(0))
        self.assertFalse(strategy._is_rebalance(1))
        self.assertTrue(strategy._is_rebalance(12))

    def test_asset_cap_redistributes_excess_weight(self):
        strategy = object.__new__(FastInstitutionalCTA)
        strategy.max_asset_weight = 0.45
        capped = strategy._cap_asset_weights(np.asarray([0.8, 0.1, 0.1]), 1.0)
        np.testing.assert_allclose(capped, np.asarray([0.45, 0.275, 0.275]))
        self.assertAlmostEqual(float(capped.sum()), 1.0)

    def test_drawdown_scale_does_not_compound_base_position(self):
        timestamps = np.asarray([1_609_459_200_000 + i * 4 * 3_600_000 for i in range(5)])
        close = np.asarray([[100.0], [94.0], [94.0], [94.0], [94.0]])
        panel = PanelData(
            timeframe="4H",
            symbols=("X-USDT",),
            timestamps_ms=timestamps,
            open=close.copy(),
            high=close.copy(),
            low=close.copy(),
            close=close,
            volume_quote=np.ones_like(close),
        )

        class HoldStrategy:
            panel_4h = panel
            name = "hold"
            dd_soft = 0.05
            dd_hard = 0.50
            dd_reentry = 0.01
            dd_min_scale = 0.20
            dd_cooldown_bars = 2
            dd_recover_scale = 1.0

            def target_weights(self, index, previous, state):
                return np.ones(1) if previous[0] <= 0 else previous.copy(), state

        result = run_cta_backtest(HoldStrategy(), fee_rate=0.0, slippage_rate=0.0)
        self.assertAlmostEqual(float(result.weights[2, 0]), float(result.weights[3, 0]))


if __name__ == "__main__":
    unittest.main()
