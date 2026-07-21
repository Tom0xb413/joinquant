"""单资产 K 线序列、多周期下载与 8H 聚合。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from .data import Candle, OkxDataClient, _date_to_ms, load_candles, save_candles


EIGHT_H_MS = 8 * 3_600_000
FOUR_H_MS = 4 * 3_600_000
DAY_MS = 86_400_000


@dataclass(frozen=True)
class BarSeries:
    """单标的、单周期的 OHLCV 序列。"""

    symbol: str
    timeframe: str
    timestamps_ms: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume_quote: np.ndarray

    @property
    def size(self) -> int:
        return int(len(self.close))

    def times_utc(self) -> list[datetime]:
        return [
            datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc)
            for ts in self.timestamps_ms
        ]


def candles_to_bar_series(symbol: str, timeframe: str, candles: list[Candle]) -> BarSeries:
    """将 Candle 列表转为数值矩阵。"""

    if len(candles) < 50:
        raise ValueError(f"{symbol} {timeframe} 样本过少：{len(candles)}")
    return BarSeries(
        symbol=symbol,
        timeframe=timeframe,
        timestamps_ms=np.asarray([c.timestamp_ms for c in candles], dtype=np.int64),
        open=np.asarray([c.open for c in candles], dtype=float),
        high=np.asarray([c.high for c in candles], dtype=float),
        low=np.asarray([c.low for c in candles], dtype=float),
        close=np.asarray([c.close for c in candles], dtype=float),
        volume_quote=np.asarray([c.volume_quote for c in candles], dtype=float),
    )


def load_bar_series(path: Path, symbol: str, timeframe: str) -> BarSeries:
    """从 CSV 加载单资产 K 线。"""

    return candles_to_bar_series(symbol, timeframe, load_candles(path))


class OkxBarClient(OkxDataClient):
    """扩展支持任意 OKX bar（如 1Dutc、4H）。"""

    def fetch_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        bar: str,
        bar_ms: int,
        max_retries: int = 4,
    ) -> list[Candle]:
        """倒序分页下载已确认 K 线。"""

        if start > end:
            raise ValueError("start 不能晚于 end")
        if bar_ms <= 0:
            raise ValueError("bar_ms 必须为正")
        start_ms = _date_to_ms(start)
        end_ms = _date_to_ms(end) + DAY_MS - 1
        cursor = end_ms + 1
        candles: dict[int, Candle] = {}

        while cursor >= start_ms:
            payload = self._request(
                {"instId": symbol, "bar": bar, "limit": "300", "after": str(cursor)},
                max_retries,
            )
            rows = payload.get("data", [])
            if not rows:
                break
            oldest = cursor
            for row in rows:
                timestamp_ms = int(row[0])
                oldest = min(oldest, timestamp_ms)
                if start_ms <= timestamp_ms <= end_ms and row[8] == "1":
                    candles[timestamp_ms] = Candle(
                        timestamp_ms=timestamp_ms,
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                        volume_base=float(row[5]),
                        volume_quote=float(row[7]),
                    )
            if oldest >= cursor or oldest < start_ms:
                break
            cursor = oldest
            time.sleep(self.pause_seconds)
        return [candles[key] for key in sorted(candles)]


def aggregate_4h_to_8h(candles: list[Candle]) -> list[Candle]:
    """将 OKX 4H K 线按 UTC 0/8/16 对齐聚合成 8H。

    OKX 不提供原生 8H；4H 边界为 0/4/8/12/16/20，两两合并即可得到 8H。
    """

    by_ts = {c.timestamp_ms: c for c in candles}
    if not by_ts:
        return []
    start = min(by_ts)
    aligned = start - (start % EIGHT_H_MS)
    end = max(by_ts)
    aggregated: list[Candle] = []
    ts = aligned
    while ts <= end:
        first = by_ts.get(ts)
        second = by_ts.get(ts + FOUR_H_MS)
        if first is not None and second is not None:
            aggregated.append(
                Candle(
                    timestamp_ms=ts,
                    open=first.open,
                    high=max(first.high, second.high),
                    low=min(first.low, second.low),
                    close=second.close,
                    volume_base=first.volume_base + second.volume_base,
                    volume_quote=first.volume_quote + second.volume_quote,
                )
            )
        ts += EIGHT_H_MS
    return aggregated


def ensure_symbol_bars(
    data_dir: Path,
    symbol: str,
    start: date,
    end: date,
    refresh: bool = False,
) -> dict[str, Path]:
    """确保 BTC/ETH 的 4H / 8H / 1D 缓存存在。"""

    data_dir.mkdir(parents=True, exist_ok=True)
    client = OkxBarClient()
    paths: dict[str, Path] = {}

    daily_path = data_dir / f"{symbol}_1D.csv"
    if refresh or not daily_path.exists():
        candles = client.fetch_bars(symbol, start, end, bar="1Dutc", bar_ms=DAY_MS)
        if not candles:
            raise RuntimeError(f"{symbol} 1D 未下载到行情")
        save_candles(daily_path, candles)
    paths["1D"] = daily_path

    h4_path = data_dir / f"{symbol}_4H.csv"
    h8_path = data_dir / f"{symbol}_8H.csv"
    if refresh or not h4_path.exists() or not h8_path.exists():
        four_h = client.fetch_bars(symbol, start, end, bar="4H", bar_ms=FOUR_H_MS)
        if not four_h:
            raise RuntimeError(f"{symbol} 4H 未下载到行情")
        save_candles(h4_path, four_h)
        eight_h = aggregate_4h_to_8h(four_h)
        if not eight_h:
            raise RuntimeError(f"{symbol} 8H 聚合结果为空")
        save_candles(h8_path, eight_h)
    paths["4H"] = h4_path
    paths["8H"] = h8_path
    return paths


def ema(values: np.ndarray, span: int) -> np.ndarray:
    """标准 EMA；以首价为种子，避免前视。"""

    if span < 1:
        raise ValueError("EMA span 必须 >= 1")
    if len(values) == 0:
        return values.copy()
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(values, dtype=float)
    out[0] = float(values[0])
    for index in range(1, len(values)):
        out[index] = alpha * float(values[index]) + (1.0 - alpha) * out[index - 1]
    return out


def ema_slope(values: np.ndarray, lookback: int = 5) -> np.ndarray:
    """EMA 斜率：相对 lookback 根 K 线的变化率。"""

    out = np.full_like(values, np.nan, dtype=float)
    if lookback < 1:
        raise ValueError("lookback 必须 >= 1")
    for index in range(lookback, len(values)):
        base = values[index - lookback]
        if abs(base) < 1e-12:
            continue
        out[index] = values[index] / base - 1.0
    return out


def deviation_rate(price: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """价格相对均线的偏离率 (P-EMA)/EMA。"""

    out = np.full_like(price, np.nan, dtype=float)
    valid = np.isfinite(price) & np.isfinite(basis) & (np.abs(basis) > 1e-12)
    out[valid] = price[valid] / basis[valid] - 1.0
    return out
