"""面向加密市场的 BTC 门控、轮动与对冲策略。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import MarketData
from .indicators import (
    equal_weights,
    finite_top,
    trailing_mean,
    trailing_return,
    trailing_volatility,
)


def _btc_on(data: MarketData, index: int, window: int) -> bool:
    """判断 BTC 是否处于上升趋势。"""

    btc = data.symbol_index("BTC-USDT")
    mean = trailing_mean(data.close, index, window)[btc]
    return bool(np.isfinite(mean) and data.close[index, btc] > mean)


def _scale_to_vol_target(
    weights: np.ndarray,
    volatility: np.ndarray,
    vol_target: float,
    max_gross: float,
) -> np.ndarray:
    """按组合波动率目标缩放权重；净空头组合只降杠杆不放大。"""

    if vol_target <= 0:
        return weights
    port_vol = float(np.nansum(np.abs(weights) * volatility))
    if port_vol <= 1e-8:
        return weights
    gross = float(np.sum(np.abs(weights)))
    if gross <= 1e-12:
        return weights
    scale = vol_target / port_vol
    if float(weights.sum()) <= 0:
        scale = min(scale, 1.0)
    scale = min(scale, max_gross / gross)
    return weights * scale


@dataclass
class BtcTrendTopMomentum:
    """BTC 趋势门控 + 正动量 Top-K + 波动率目标。

    风险开：只持有“自身动量为正且站上均线”的最强币；
    风险关：空仓（现金）。这是冲击夏普>1 的核心多头方案。
    """

    trend_window: int = 150
    lookback: int = 90
    top_k: int = 2
    rebalance_days: int = 14
    vol_target: float = 0.30
    max_gross: float = 1.0
    name: str = "btc_trend_top_momentum"
    ideas: tuple[str, ...] = ("BTC趋势门控", "主流/山寨动量轮动", "波动率目标")

    def target_weights(self, data: MarketData, signal_index: int, previous: np.ndarray) -> np.ndarray:
        if signal_index % self.rebalance_days != 0:
            return previous
        if not _btc_on(data, signal_index, self.trend_window):
            return np.zeros(len(data.symbols))
        momentum = trailing_return(data.close, signal_index, self.lookback)
        own_ma = trailing_mean(data.close, signal_index, self.lookback)
        volatility = trailing_volatility(data.close, signal_index, 30)
        score = np.where(
            np.isfinite(momentum)
            & np.isfinite(own_ma)
            & (momentum > 0)
            & (data.close[signal_index] > own_ma),
            momentum,
            np.nan,
        )
        weights = equal_weights(finite_top(score, self.top_k), len(data.symbols))
        return _scale_to_vol_target(weights, volatility, self.vol_target, self.max_gross)


@dataclass
class BtcStyleVolRotation:
    """BTC 门控下的主流/山寨风格轮动，并做波动缩放。

    强化版全天候：比较 BTC+ETH 与山寨篮子的相对动量，只做强势一侧。
    """

    trend_window: int = 100
    style_window: int = 60
    top_k: int = 2
    rebalance_days: int = 21
    vol_target: float = 0.45
    max_gross: float = 1.0
    name: str = "btc_style_vol_rotation"
    ideas: tuple[str, ...] = ("BTC趋势门控", "主流山寨风格轮动", "波动率目标")

    def target_weights(self, data: MarketData, signal_index: int, previous: np.ndarray) -> np.ndarray:
        if signal_index % self.rebalance_days != 0:
            return previous
        if not _btc_on(data, signal_index, self.trend_window):
            return np.zeros(len(data.symbols))
        momentum = trailing_return(data.close, signal_index, self.style_window)
        if not np.isfinite(momentum).all():
            return np.zeros(len(data.symbols))
        btc = data.symbol_index("BTC-USDT")
        eth = data.symbol_index("ETH-USDT")
        majors = np.array([btc, eth])
        alts = np.array([index for index in range(len(data.symbols)) if index not in majors])
        major_score = float(np.mean(momentum[majors]))
        alt_score = float(np.mean(momentum[alts]))
        if max(major_score, alt_score) <= 0:
            return np.zeros(len(data.symbols))
        universe = majors if major_score >= alt_score else alts
        selected = universe[np.argsort(momentum[universe], kind="stable")[::-1][: self.top_k]]
        weights = equal_weights(selected, len(data.symbols))
        volatility = trailing_volatility(data.close, signal_index, 30)
        return _scale_to_vol_target(weights, volatility, self.vol_target, self.max_gross)


@dataclass
class BtcCoreAltSatellite:
    """大部分时间只做 BTC，卫星仓轮动最强山寨。"""

    trend_window: int = 120
    lookback: int = 40
    rebalance_days: int = 14
    btc_weight: float = 0.70
    alt_weight: float = 0.30
    vol_target: float = 0.45
    max_gross: float = 1.2
    name: str = "btc_core_alt_satellite"
    ideas: tuple[str, ...] = ("BTC为主", "山寨卫星轮动", "波动率目标")

    def target_weights(self, data: MarketData, signal_index: int, previous: np.ndarray) -> np.ndarray:
        if signal_index % self.rebalance_days != 0:
            return previous
        if not _btc_on(data, signal_index, self.trend_window):
            return np.zeros(len(data.symbols))
        btc = data.symbol_index("BTC-USDT")
        momentum = trailing_return(data.close, signal_index, self.lookback)
        weights = np.zeros(len(data.symbols), dtype=float)
        weights[btc] = self.btc_weight
        alts = np.array([index for index in range(len(data.symbols)) if index != btc])
        best = int(alts[np.argmax(momentum[alts])])
        if np.isfinite(momentum[best]) and momentum[best] > 0:
            weights[best] = self.alt_weight
        else:
            weights[btc] = min(1.0, self.btc_weight + self.alt_weight)
        volatility = trailing_volatility(data.close, signal_index, 30)
        return _scale_to_vol_target(weights, volatility, self.vol_target, self.max_gross)


@dataclass
class BtcGateAltHedge:
    """风险开：多 BTC + 最强山寨；风险关/弱势：做空最弱山寨对冲。"""

    trend_window: int = 100
    lookback: int = 40
    rebalance_days: int = 14
    btc_weight: float = 0.60
    alt_weight: float = 0.40
    short_weight: float = 0.30
    off_short_weight: float = 0.40
    vol_target: float = 0.45
    max_gross: float = 1.4
    name: str = "btc_gate_alt_hedge"
    ideas: tuple[str, ...] = ("BTC趋势门控", "山寨轮动", "弱势做空对冲")

    def target_weights(self, data: MarketData, signal_index: int, previous: np.ndarray) -> np.ndarray:
        if signal_index % self.rebalance_days != 0:
            return previous
        btc = data.symbol_index("BTC-USDT")
        momentum = trailing_return(data.close, signal_index, self.lookback)
        volatility = trailing_volatility(data.close, signal_index, 30)
        alts = np.array([index for index in range(len(data.symbols)) if index != btc])
        weights = np.zeros(len(data.symbols), dtype=float)
        if _btc_on(data, signal_index, self.trend_window):
            weights[btc] = self.btc_weight
            best = int(alts[np.argmax(momentum[alts])])
            worst = int(alts[np.argmin(momentum[alts])])
            if np.isfinite(momentum[best]) and momentum[best] > 0:
                weights[best] = self.alt_weight
            if np.isfinite(momentum[worst]) and momentum[worst] < 0 and self.short_weight > 0:
                weights[worst] = -self.short_weight
        elif self.off_short_weight > 0:
            weak = alts[np.argsort(momentum[alts], kind="stable")[:2]]
            weights[weak] = -self.off_short_weight / len(weak)
        return _scale_to_vol_target(weights, volatility, self.vol_target, self.max_gross)


def crypto_alpha_catalog() -> dict[str, type]:
    """返回加密增强策略目录。"""

    return {
        "btc_trend_top_momentum": BtcTrendTopMomentum,
        "btc_style_vol_rotation": BtcStyleVolRotation,
        "btc_core_alt_satellite": BtcCoreAltSatellite,
        "btc_gate_alt_hedge": BtcGateAltHedge,
    }
