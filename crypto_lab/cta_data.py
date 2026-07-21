"""机构级动量 CTA：TOP15 多周期数据下载与面板对齐。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from .data import Candle, _date_to_ms, load_candles, save_candles
from .ema_data import (
    DAY_MS,
    FOUR_H_MS,
    OkxBarClient,
    candles_to_bar_series,
)


TWELVE_H_MS = 12 * 3_600_000

# 流动性较好且自 2021 起有完整 OKX 现货历史的主流币（15 个）
CTA_TOP15 = (
    "BTC-USDT",
    "ETH-USDT",
    "SOL-USDT",
    "XRP-USDT",
    "ADA-USDT",
    "DOGE-USDT",
    "LTC-USDT",
    "BCH-USDT",
    "LINK-USDT",
    "DOT-USDT",
    "AVAX-USDT",
    "UNI-USDT",
    "ATOM-USDT",
    "NEAR-USDT",
    "AAVE-USDT",
)


@dataclass(frozen=True)
class PanelData:
    """多标的、单周期对齐后的 OHLCV 面板。"""

    timeframe: str
    symbols: tuple[str, ...]
    timestamps_ms: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume_quote: np.ndarray

    @property
    def size(self) -> int:
        return int(len(self.timestamps_ms))

    @property
    def n_assets(self) -> int:
        return int(len(self.symbols))

    def symbol_index(self, symbol: str) -> int:
        try:
            return self.symbols.index(symbol)
        except ValueError as exc:
            raise KeyError(symbol) from exc


def aggregate_4h_to_12h(candles: list[Candle]) -> list[Candle]:
    """将 4H K 线按 UTC 0/12 对齐聚合为 12H。"""

    by_ts = {c.timestamp_ms: c for c in candles}
    if not by_ts:
        return []
    start = min(by_ts) - (min(by_ts) % TWELVE_H_MS)
    end = max(by_ts)
    out: list[Candle] = []
    ts = start
    while ts <= end:
        parts = [by_ts.get(ts + i * FOUR_H_MS) for i in range(3)]
        if all(p is not None for p in parts):
            out.append(
                Candle(
                    timestamp_ms=ts,
                    open=parts[0].open,
                    high=max(p.high for p in parts),
                    low=min(p.low for p in parts),
                    close=parts[-1].close,
                    volume_base=sum(p.volume_base for p in parts),
                    volume_quote=sum(p.volume_quote for p in parts),
                )
            )
        ts += TWELVE_H_MS
    return out


def ensure_cta_bars(
    data_dir: Path,
    symbols: Iterable[str] = CTA_TOP15,
    start: date = date(2021, 1, 1),
    end: date | None = None,
    refresh: bool = False,
) -> dict[str, dict[str, Path]]:
    """下载并缓存 TOP 币种的 4H / 12H / 1D 数据。"""

    end = end or date.today()
    data_dir.mkdir(parents=True, exist_ok=True)
    client = OkxBarClient(pause_seconds=0.10)
    paths: dict[str, dict[str, Path]] = {}
    for symbol in symbols:
        symbol_paths: dict[str, Path] = {}
        d1 = data_dir / f"{symbol}_1D.csv"
        h4 = data_dir / f"{symbol}_4H.csv"
        h12 = data_dir / f"{symbol}_12H.csv"
        if refresh or not d1.exists():
            candles = client.fetch_bars(symbol, start, end, bar="1Dutc", bar_ms=DAY_MS)
            if len(candles) < 200:
                raise RuntimeError(f"{symbol} 1D 样本不足：{len(candles)}")
            save_candles(d1, candles)
        if refresh or not h4.exists() or not h12.exists():
            four = client.fetch_bars(symbol, start, end, bar="4H", bar_ms=FOUR_H_MS)
            if len(four) < 500:
                raise RuntimeError(f"{symbol} 4H 样本不足：{len(four)}")
            save_candles(h4, four)
            save_candles(h12, aggregate_4h_to_12h(four))
        symbol_paths["1D"] = d1
        symbol_paths["4H"] = h4
        symbol_paths["12H"] = h12
        paths[symbol] = symbol_paths
        print(f"[cta-data] {symbol} ready")
    return paths


def load_panel(data_dir: Path, timeframe: str, symbols: Iterable[str] = CTA_TOP15) -> PanelData:
    """加载并对齐多标的面板；仅保留全部标的都有的时间戳。"""

    series = {}
    for symbol in symbols:
        path = data_dir / f"{symbol}_{timeframe}.csv"
        series[symbol] = candles_to_bar_series(symbol, timeframe, load_candles(path))
    common = None
    for bar in series.values():
        stamps = set(int(x) for x in bar.timestamps_ms.tolist())
        common = stamps if common is None else (common & stamps)
    if not common or len(common) < 300:
        raise ValueError(f"{timeframe} 公共样本过少：{0 if not common else len(common)}")
    ordered = tuple(sorted(series.keys()))
    stamps = np.asarray(sorted(common), dtype=np.int64)
    n, m = len(stamps), len(ordered)
    open_ = np.zeros((n, m))
    high = np.zeros((n, m))
    low = np.zeros((n, m))
    close = np.zeros((n, m))
    volume = np.zeros((n, m))
    for col, symbol in enumerate(ordered):
        bar = series[symbol]
        index = {int(ts): i for i, ts in enumerate(bar.timestamps_ms.tolist())}
        rows = [index[int(ts)] for ts in stamps]
        open_[:, col] = bar.open[rows]
        high[:, col] = bar.high[rows]
        low[:, col] = bar.low[rows]
        close[:, col] = bar.close[rows]
        volume[:, col] = bar.volume_quote[rows]
    return PanelData(timeframe, ordered, stamps, open_, high, low, close, volume)


TF_MS = {
    "4H": FOUR_H_MS,
    "12H": TWELVE_H_MS,
    "1D": DAY_MS,
    "8H": 8 * 3_600_000,
}


def map_higher_tf_to_base(
    base: PanelData,
    higher: PanelData,
) -> dict[str, np.ndarray]:
    """将更高周期指标对齐到基准周期，仅使用在 base K 线收盘前已收盘的 higher bar。

    例如 4H 在 T 开盘、T+4H 收盘发信号时，只能使用 close_time <= T+4H 的日线/12H。
    """

    base_ms = TF_MS[base.timeframe]
    higher_ms = TF_MS[higher.timeframe]
    base_close = base.timestamps_ms + base_ms
    higher_close = higher.timestamps_ms + higher_ms
    mapped_idx = np.searchsorted(higher_close, base_close, side="right") - 1
    mapped_idx = np.clip(mapped_idx, -1, len(higher.timestamps_ms) - 1)
    mapped_idx[mapped_idx < 0] = -1
    return {"index": mapped_idx}
