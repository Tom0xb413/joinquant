"""行情对齐和完整性校验测试。"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from crypto_lab.data import Candle, align_market_data, load_candles, save_candles


def candle(day: int, close: float) -> Candle:
    """构造 2024 年 1 月指定日期的测试 K 线。"""

    timestamp = int(datetime(2024, 1, day, tzinfo=timezone.utc).timestamp() * 1000)
    return Candle(timestamp, close, close, close, close, 10.0, close * 10.0)


class DataTests(unittest.TestCase):
    """验证缓存往返和共同日期对齐。"""

    def test_csv_round_trip(self):
        values = [candle(1, 100.0), candle(2, 101.0)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "BTC-USDT.csv"
            save_candles(path, values)
            self.assertEqual(load_candles(path), values)

    def test_alignment_rejects_too_short_history(self):
        series = {
            "BTC-USDT": [candle(day, 100.0 + day) for day in range(1, 10)],
            "ETH-USDT": [candle(day, 50.0 + day) for day in range(1, 10)],
        }
        with self.assertRaisesRegex(ValueError, "无法进行稳健回测"):
            align_market_data(series)

    def test_negative_price_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            save_candles(path, [candle(1, -1.0)])
            with self.assertRaisesRegex(ValueError, "非正价格"):
                load_candles(path)


if __name__ == "__main__":
    unittest.main()

