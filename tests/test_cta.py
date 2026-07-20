"""机构 CTA 指标与聚合测试。"""

from __future__ import annotations

import unittest

import numpy as np

from crypto_lab.cta_data import aggregate_4h_to_12h
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


if __name__ == "__main__":
    unittest.main()
