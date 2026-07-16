"""面向加密市场的 BTC 门控、轮动与对冲策略（夏普优先版）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import MarketData
from .indicators import (
    finite_top,
    inverse_volatility_weights,
    trailing_mean,
    trailing_return,
    trailing_volatility,
)


def _btc_on(data: MarketData, index: int, window: int) -> bool:
    """判断 BTC 是否站上趋势均线（风险开）。"""

    btc = data.symbol_index("BTC-USDT")
    mean = trailing_mean(data.close, index, window)[btc]
    return bool(np.isfinite(mean) and data.close[index, btc] > mean)


def _market_breadth(data: MarketData, index: int, window: int) -> float:
    """计算站上自身均线的资产占比，作为风险开的次级确认。"""

    mean = trailing_mean(data.close, index, window)
    finite = np.isfinite(mean)
    if not finite.any():
        return 0.0
    return float(np.mean(data.close[index, finite] > mean[finite]))


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


def _positive_momentum_score(
    data: MarketData,
    signal_index: int,
    lookback: int,
) -> np.ndarray:
    """正动量且站上自身均线的资产得分，其余为 NaN。"""

    momentum = trailing_return(data.close, signal_index, lookback)
    own_ma = trailing_mean(data.close, signal_index, lookback)
    return np.where(
        np.isfinite(momentum)
        & np.isfinite(own_ma)
        & (momentum > 0)
        & (data.close[signal_index] > own_ma),
        momentum,
        np.nan,
    )


@dataclass
class BtcTrendTopMomentum:
    """BTC 趋势门控 + 正动量 Top-K + 逆波动加权 + 波动率目标。

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
        score = _positive_momentum_score(data, signal_index, self.lookback)
        volatility = trailing_volatility(data.close, signal_index, 30)
        selected = finite_top(score, self.top_k)
        weights = inverse_volatility_weights(volatility, selected, len(data.symbols), 1.0)
        return _scale_to_vol_target(weights, volatility, self.vol_target, self.max_gross)


@dataclass
class BtcBreadthTopMomentum:
    """BTC 趋势 + 市场广度双重门控的 Top 动量。

    广度过低时即使 BTC 站上均线也空仓，避免伪突破后的高波动回撤。
    """

    trend_window: int = 150
    lookback: int = 90
    top_k: int = 2
    rebalance_days: int = 14
    vol_target: float = 0.32
    breadth_min: float = 0.25
    max_gross: float = 1.0
    name: str = "btc_breadth_top_momentum"
    ideas: tuple[str, ...] = ("BTC趋势门控", "广度过滤", "逆波动动量")

    def target_weights(self, data: MarketData, signal_index: int, previous: np.ndarray) -> np.ndarray:
        if signal_index % self.rebalance_days != 0:
            return previous
        if not _btc_on(data, signal_index, self.trend_window):
            return np.zeros(len(data.symbols))
        if _market_breadth(data, signal_index, self.lookback) < self.breadth_min:
            return np.zeros(len(data.symbols))
        score = _positive_momentum_score(data, signal_index, self.lookback)
        volatility = trailing_volatility(data.close, signal_index, 30)
        selected = finite_top(score, self.top_k)
        weights = inverse_volatility_weights(volatility, selected, len(data.symbols), 1.0)
        return _scale_to_vol_target(weights, volatility, self.vol_target, self.max_gross)


@dataclass
class BtcDualConfirmMomentum:
    """快慢双均线 BTC 门控 + Top 动量，进一步过滤假突破。"""

    fast_trend: int = 100
    slow_trend: int = 150
    lookback: int = 90
    top_k: int = 3
    rebalance_days: int = 14
    vol_target: float = 0.32
    max_gross: float = 1.0
    name: str = "btc_dual_confirm_momentum"
    ideas: tuple[str, ...] = ("双均线BTC门控", "Top动量", "波动率目标")

    def target_weights(self, data: MarketData, signal_index: int, previous: np.ndarray) -> np.ndarray:
        if signal_index % self.rebalance_days != 0:
            return previous
        if not (
            _btc_on(data, signal_index, self.fast_trend)
            and _btc_on(data, signal_index, self.slow_trend)
        ):
            return np.zeros(len(data.symbols))
        score = _positive_momentum_score(data, signal_index, self.lookback)
        volatility = trailing_volatility(data.close, signal_index, 30)
        selected = finite_top(score, self.top_k)
        weights = inverse_volatility_weights(volatility, selected, len(data.symbols), 1.0)
        return _scale_to_vol_target(weights, volatility, self.vol_target, self.max_gross)


@dataclass
class BtcStyleVolRotation:
    """BTC 门控下的主流/山寨风格轮动：资产需正动量且站上均线，逆波动加权。"""

    trend_window: int = 150
    style_window: int = 60
    top_k: int = 2
    rebalance_days: int = 21
    vol_target: float = 0.30
    max_gross: float = 1.0
    name: str = "btc_style_vol_rotation"
    ideas: tuple[str, ...] = ("BTC趋势门控", "主流山寨风格轮动", "波动率目标")

    def target_weights(self, data: MarketData, signal_index: int, previous: np.ndarray) -> np.ndarray:
        if signal_index % self.rebalance_days != 0:
            return previous
        if not _btc_on(data, signal_index, self.trend_window):
            return np.zeros(len(data.symbols))
        momentum = trailing_return(data.close, signal_index, self.style_window)
        own_ma = trailing_mean(data.close, signal_index, self.style_window)
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
        score = np.full(len(data.symbols), np.nan)
        for index in universe:
            if (
                np.isfinite(momentum[index])
                and momentum[index] > 0
                and np.isfinite(own_ma[index])
                and data.close[signal_index, index] > own_ma[index]
            ):
                score[index] = momentum[index]
        selected = finite_top(score, self.top_k)
        if not len(selected):
            return np.zeros(len(data.symbols))
        volatility = trailing_volatility(data.close, signal_index, 30)
        weights = inverse_volatility_weights(volatility, selected, len(data.symbols), 1.0)
        return _scale_to_vol_target(weights, volatility, self.vol_target, self.max_gross)


@dataclass
class BtcCoreAltSatellite:
    """BTC 为核心，但按相对动量自适应抬升山寨卫星权重。

    OOS 段 BTC 近似零收益时，固定高 BTC 权重会拖累年化；
    自适应后在山寨显著更强时提高卫星仓，更贴近加密风格轮动现实。
    """

    trend_window: int = 150
    lookback: int = 90
    rebalance_days: int = 14
    base_btc: float = 0.40
    max_alt: float = 0.70
    vol_target: float = 0.30
    max_gross: float = 1.0
    name: str = "btc_core_alt_satellite"
    ideas: tuple[str, ...] = ("BTC为主自适应", "山寨卫星轮动", "波动率目标")

    def target_weights(self, data: MarketData, signal_index: int, previous: np.ndarray) -> np.ndarray:
        if signal_index % self.rebalance_days != 0:
            return previous
        if not _btc_on(data, signal_index, self.trend_window):
            return np.zeros(len(data.symbols))
        btc = data.symbol_index("BTC-USDT")
        momentum = trailing_return(data.close, signal_index, self.lookback)
        own_ma = trailing_mean(data.close, signal_index, self.lookback)
        volatility = trailing_volatility(data.close, signal_index, 30)
        alts = np.array([index for index in range(len(data.symbols)) if index != btc])
        best = int(alts[np.argmax(np.where(np.isfinite(momentum[alts]), momentum[alts], -np.inf))])
        btc_mom = float(momentum[btc]) if np.isfinite(momentum[btc]) else -1.0
        alt_mom = float(momentum[best]) if np.isfinite(momentum[best]) else -1.0
        weights = np.zeros(len(data.symbols), dtype=float)
        alt_ok = (
            alt_mom > 0
            and np.isfinite(own_ma[best])
            and data.close[signal_index, best] > own_ma[best]
        )
        if alt_ok and alt_mom > btc_mom:
            edge = min(1.0, max(0.0, alt_mom - max(btc_mom, 0.0)))
            alt_weight = min(self.max_alt, 0.30 + 0.55 * edge)
            weights[best] = alt_weight
            weights[btc] = max(self.base_btc * 0.5, 1.0 - alt_weight)
        elif btc_mom > 0 and data.close[signal_index, btc] > own_ma[btc]:
            weights[btc] = 1.0
        else:
            return np.zeros(len(data.symbols))
        return _scale_to_vol_target(weights, volatility, self.vol_target, self.max_gross)


@dataclass
class BtcGateAltHedge:
    """风险开：多 BTC + 最强山寨；极弱山寨才小额做空；风险关默认空仓。

    弱市盲目做空山寨会因高相关反弹与借券成本显著伤害夏普，
    因此默认 off_short_weight=0，仅在 short_threshold 触发时对冲。
    """

    trend_window: int = 150
    lookback: int = 60
    rebalance_days: int = 14
    btc_weight: float = 0.55
    alt_weight: float = 0.45
    short_weight: float = 0.20
    short_threshold: float = -0.20
    off_short_weight: float = 0.0
    vol_target: float = 0.30
    max_gross: float = 1.2
    name: str = "btc_gate_alt_hedge"
    ideas: tuple[str, ...] = ("BTC趋势门控", "山寨轮动", "严格条件做空对冲")

    def target_weights(self, data: MarketData, signal_index: int, previous: np.ndarray) -> np.ndarray:
        if signal_index % self.rebalance_days != 0:
            return previous
        btc = data.symbol_index("BTC-USDT")
        momentum = trailing_return(data.close, signal_index, self.lookback)
        own_ma = trailing_mean(data.close, signal_index, self.lookback)
        volatility = trailing_volatility(data.close, signal_index, 30)
        alts = np.array([index for index in range(len(data.symbols)) if index != btc])
        weights = np.zeros(len(data.symbols), dtype=float)
        if not _btc_on(data, signal_index, self.trend_window):
            if self.off_short_weight > 0 and np.isfinite(momentum[alts]).any():
                weak = alts[np.argsort(momentum[alts], kind="stable")[:2]]
                weights[weak] = -self.off_short_weight / len(weak)
            return _scale_to_vol_target(weights, volatility, self.vol_target, self.max_gross)
        weights[btc] = self.btc_weight
        best = int(alts[np.argmax(np.where(np.isfinite(momentum[alts]), momentum[alts], -np.inf))])
        worst = int(alts[np.argmin(np.where(np.isfinite(momentum[alts]), momentum[alts], np.inf))])
        if (
            np.isfinite(momentum[best])
            and momentum[best] > 0
            and np.isfinite(own_ma[best])
            and data.close[signal_index, best] > own_ma[best]
        ):
            weights[best] = self.alt_weight
        else:
            weights[btc] = min(1.0, self.btc_weight + self.alt_weight)
        if (
            self.short_weight > 0
            and np.isfinite(momentum[worst])
            and momentum[worst] < self.short_threshold
        ):
            weights[worst] = -self.short_weight
        return _scale_to_vol_target(weights, volatility, self.vol_target, self.max_gross)


@dataclass
class BtcProtectiveHedge:
    """以 Top 动量为主仓；仅当组合波动过高且存在极弱山寨时启动保护性空头。"""

    trend_window: int = 150
    lookback: int = 90
    top_k: int = 2
    rebalance_days: int = 14
    vol_target: float = 0.30
    hedge_vol_trigger: float = 0.55
    short_weight: float = 0.20
    max_gross: float = 1.2
    name: str = "btc_protective_hedge"
    ideas: tuple[str, ...] = ("BTC趋势门控", "Top动量", "波动触发保护性对冲")

    def target_weights(self, data: MarketData, signal_index: int, previous: np.ndarray) -> np.ndarray:
        if signal_index % self.rebalance_days != 0:
            return previous
        if not _btc_on(data, signal_index, self.trend_window):
            return np.zeros(len(data.symbols))
        score = _positive_momentum_score(data, signal_index, self.lookback)
        volatility = trailing_volatility(data.close, signal_index, 30)
        selected = finite_top(score, self.top_k)
        weights = inverse_volatility_weights(volatility, selected, len(data.symbols), 1.0)
        port_vol = float(np.nansum(np.abs(weights) * volatility))
        if port_vol >= self.hedge_vol_trigger and self.short_weight > 0 and len(selected):
            btc = data.symbol_index("BTC-USDT")
            momentum = trailing_return(data.close, signal_index, self.lookback)
            held = set(int(index) for index in selected.tolist())
            candidates = [
                index
                for index in range(len(data.symbols))
                if index != btc and index not in held
            ]
            if candidates:
                worst = min(
                    candidates,
                    key=lambda index: momentum[index] if np.isfinite(momentum[index]) else 0.0,
                )
                if np.isfinite(momentum[worst]) and momentum[worst] < 0:
                    weights[worst] = -self.short_weight
        return _scale_to_vol_target(weights, volatility, self.vol_target, self.max_gross)


def crypto_alpha_catalog() -> dict[str, type]:
    """返回加密增强策略目录。"""

    return {
        "btc_trend_top_momentum": BtcTrendTopMomentum,
        "btc_breadth_top_momentum": BtcBreadthTopMomentum,
        "btc_dual_confirm_momentum": BtcDualConfirmMomentum,
        "btc_style_vol_rotation": BtcStyleVolRotation,
        "btc_core_alt_satellite": BtcCoreAltSatellite,
        "btc_gate_alt_hedge": BtcGateAltHedge,
        "btc_protective_hedge": BtcProtectiveHedge,
    }
