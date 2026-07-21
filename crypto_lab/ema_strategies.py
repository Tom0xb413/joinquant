"""BTC/ETH 的 EMA 均线策略族（4H / 8H / 1D）。

核心信号：EMA50/100 多头排列（交叉状态，非仅交叉瞬间）。
增强条件：
1. EMA200 方向与位置过滤（大趋势）
2. 快线/趋势线斜率过滤（避免横盘钝刀）
3. 偏离率过滤（拒绝过度追高）或回撤再入场
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ema_data import BarSeries, deviation_rate, ema, ema_slope


def _ready_index(*windows: int) -> int:
    return int(max(windows))


@dataclass
class EmaCrossBasic:
    """EMA50/100：快线上慢线做多，否则空仓。"""

    fast: int = 50
    slow: int = 100
    name: str = "ema_cross_50_100"
    ideas: tuple[str, ...] = ("EMA50/100交叉",)

    def positions(self, series: BarSeries) -> np.ndarray:
        fast = ema(series.close, self.fast)
        slow = ema(series.close, self.slow)
        pos = np.zeros(series.size, dtype=float)
        ready = _ready_index(self.fast, self.slow)
        for index in range(ready, series.size):
            pos[index] = 1.0 if fast[index] > slow[index] else 0.0
        return pos


@dataclass
class EmaCrossAbove200:
    """EMA50/100 多头 + 收盘站上 EMA200。"""

    fast: int = 50
    slow: int = 100
    trend: int = 200
    name: str = "ema_cross_above_200"
    ideas: tuple[str, ...] = ("EMA50/100交叉", "EMA200过滤")

    def positions(self, series: BarSeries) -> np.ndarray:
        fast = ema(series.close, self.fast)
        slow = ema(series.close, self.slow)
        trend = ema(series.close, self.trend)
        pos = np.zeros(series.size, dtype=float)
        ready = _ready_index(self.fast, self.slow, self.trend)
        for index in range(ready, series.size):
            if fast[index] > slow[index] and series.close[index] > trend[index]:
                pos[index] = 1.0
        return pos


@dataclass
class EmaCrossSlope200:
    """EMA50/100 多头 + 站上 EMA200 + EMA200 斜率向上。"""

    fast: int = 50
    slow: int = 100
    trend: int = 200
    slope_lookback: int = 5
    min_slope: float = 0.0
    name: str = "ema_cross_slope_200"
    ideas: tuple[str, ...] = ("EMA50/100交叉", "EMA200斜率")

    def positions(self, series: BarSeries) -> np.ndarray:
        fast = ema(series.close, self.fast)
        slow = ema(series.close, self.slow)
        trend = ema(series.close, self.trend)
        slope = ema_slope(trend, self.slope_lookback)
        pos = np.zeros(series.size, dtype=float)
        ready = _ready_index(self.fast, self.slow, self.trend, self.slope_lookback)
        for index in range(ready, series.size):
            if (
                fast[index] > slow[index]
                and series.close[index] > trend[index]
                and np.isfinite(slope[index])
                and slope[index] > self.min_slope
            ):
                pos[index] = 1.0
        return pos


@dataclass
class EmaCrossDevFilter:
    """交叉趋势 + EMA200/斜率 + 偏离率上限（防追高）。"""

    fast: int = 50
    slow: int = 100
    trend: int = 200
    slope_lookback: int = 5
    min_slope: float = 0.0
    max_deviation: float = 0.08
    require_fast_slope: bool = True
    name: str = "ema_cross_dev_filter"
    ideas: tuple[str, ...] = ("EMA50/100交叉", "EMA200斜率", "偏离率过滤")

    def positions(self, series: BarSeries) -> np.ndarray:
        fast = ema(series.close, self.fast)
        slow = ema(series.close, self.slow)
        trend = ema(series.close, self.trend)
        slope200 = ema_slope(trend, self.slope_lookback)
        slope_fast = ema_slope(fast, self.slope_lookback)
        dev = deviation_rate(series.close, fast)
        pos = np.zeros(series.size, dtype=float)
        ready = _ready_index(self.fast, self.slow, self.trend, self.slope_lookback)
        for index in range(ready, series.size):
            if not (
                fast[index] > slow[index]
                and series.close[index] > trend[index]
                and np.isfinite(slope200[index])
                and slope200[index] > self.min_slope
                and np.isfinite(dev[index])
                and abs(dev[index]) <= self.max_deviation
            ):
                continue
            if self.require_fast_slope and (
                not np.isfinite(slope_fast[index]) or slope_fast[index] <= 0
            ):
                continue
            pos[index] = 1.0
        return pos


@dataclass
class EmaTrendPullback:
    """趋势向上时，等待相对 EMA50 的负偏离回撤后再做多。"""

    fast: int = 50
    slow: int = 100
    trend: int = 200
    slope_lookback: int = 5
    min_slope: float = 0.0
    pullback: float = -0.02
    reentry: float = -0.005
    name: str = "ema_trend_pullback"
    ideas: tuple[str, ...] = ("EMA趋势", "EMA200斜率", "偏离率回撤")

    def positions(self, series: BarSeries) -> np.ndarray:
        fast = ema(series.close, self.fast)
        slow = ema(series.close, self.slow)
        trend = ema(series.close, self.trend)
        slope = ema_slope(trend, self.slope_lookback)
        dev = deviation_rate(series.close, fast)
        pos = np.zeros(series.size, dtype=float)
        ready = _ready_index(self.fast, self.slow, self.trend, self.slope_lookback)
        armed = False
        holding = False
        for index in range(ready, series.size):
            trend_on = (
                fast[index] > slow[index]
                and series.close[index] > trend[index]
                and np.isfinite(slope[index])
                and slope[index] > self.min_slope
            )
            if not trend_on or not np.isfinite(dev[index]):
                armed = False
                holding = False
                pos[index] = 0.0
                continue
            if dev[index] <= self.pullback:
                armed = True
            if holding:
                if fast[index] < slow[index] or series.close[index] < trend[index]:
                    holding = False
                    armed = False
                else:
                    pos[index] = 1.0
                continue
            if armed and dev[index] >= self.reentry:
                holding = True
                armed = False
                pos[index] = 1.0
        return pos


@dataclass
class EmaFullFilter:
    """综合版：EMA50/100 + EMA200 位置/斜率 + 快线斜率 + 偏离率带。

    仅在“趋势确认且未过度偏离”时持仓；这是本轮默认优化主策略。
    """

    fast: int = 50
    slow: int = 100
    trend: int = 200
    slope_lookback: int = 5
    min_slope_200: float = 0.0
    min_slope_fast: float = 0.0
    max_deviation: float = 0.06
    min_deviation: float = -0.04
    name: str = "ema_full_filter"
    ideas: tuple[str, ...] = ("EMA50/100交叉", "EMA200参考", "斜率过滤", "偏离率带")

    def positions(self, series: BarSeries) -> np.ndarray:
        fast = ema(series.close, self.fast)
        slow = ema(series.close, self.slow)
        trend = ema(series.close, self.trend)
        slope200 = ema_slope(trend, self.slope_lookback)
        slope_fast = ema_slope(fast, self.slope_lookback)
        dev = deviation_rate(series.close, fast)
        pos = np.zeros(series.size, dtype=float)
        ready = _ready_index(self.fast, self.slow, self.trend, self.slope_lookback)
        for index in range(ready, series.size):
            if not (
                fast[index] > slow[index]
                and fast[index] > trend[index]
                and series.close[index] > trend[index]
                and np.isfinite(slope200[index])
                and slope200[index] > self.min_slope_200
                and np.isfinite(slope_fast[index])
                and slope_fast[index] > self.min_slope_fast
                and np.isfinite(dev[index])
                and self.min_deviation <= dev[index] <= self.max_deviation
            ):
                continue
            pos[index] = 1.0
        return pos


def ema_strategy_catalog() -> dict[str, type]:
    """返回 EMA 策略目录。"""

    return {
        "ema_cross_50_100": EmaCrossBasic,
        "ema_cross_above_200": EmaCrossAbove200,
        "ema_cross_slope_200": EmaCrossSlope200,
        "ema_cross_dev_filter": EmaCrossDevFilter,
        "ema_trend_pullback": EmaTrendPullback,
        "ema_full_filter": EmaFullFilter,
    }
