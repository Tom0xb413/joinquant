"""基于上一轮失败教训重新设计的低换手优化策略。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import MarketData
from .indicators import (
    equal_weights,
    finite_top,
    inverse_volatility_weights,
    trailing_mean,
    trailing_return,
    trailing_volatility,
)


def _btc_trend_on(data: MarketData, signal_index: int, window: int) -> bool:
    """BTC 收盘价是否位于自身趋势均线上方。"""

    btc = data.symbol_index("BTC-USDT")
    mean = trailing_mean(data.close, signal_index, window)[btc]
    return bool(np.isfinite(mean) and data.close[signal_index, btc] > mean)


@dataclass
class BtcDualMomentum:
    """BTC 趋势门控 + 长短期双动量 + 自身均线过滤。

    设计思想来自策略 01 的动量轮动与策略 02 的风险开关：
    仅在 BTC 多头时持仓；资产需同时满足长期、短期动量为正且站上自身均线。
    """

    regime_window: int = 120
    lookback: int = 90
    fast_lookback: int = 45
    top_k: int = 3
    rebalance_days: int = 21
    name: str = "btc_dual_momentum"
    ideas: tuple[str, ...] = ("01-趋势轮动", "02-风险开关")

    def target_weights(self, data: MarketData, signal_index: int, previous: np.ndarray) -> np.ndarray:
        if signal_index % self.rebalance_days != 0:
            return previous
        if not _btc_trend_on(data, signal_index, self.regime_window):
            return np.zeros(len(data.symbols))
        slow = trailing_return(data.close, signal_index, self.lookback)
        fast = trailing_return(data.close, signal_index, self.fast_lookback)
        own_ma = trailing_mean(data.close, signal_index, self.lookback)
        score = 0.7 * slow + 0.3 * fast
        eligible = (
            np.isfinite(score)
            & np.isfinite(own_ma)
            & (slow > 0)
            & (fast > 0)
            & (data.close[signal_index] > own_ma)
        )
        return equal_weights(finite_top(np.where(eligible, score, np.nan), self.top_k), len(data.symbols))


@dataclass
class BreadthRegimeRotation:
    """市场广度门控的分档仓位轮动。

    借鉴策略 02/05/07 的大盘择时思想：用“站上均线的资产占比”衡量风险偏好，
    弱市空仓、中性只持 BTC、强市才做横截面动量。
    """

    ma_window: int = 100
    lookback: int = 90
    top_k: int = 3
    rebalance_days: int = 21
    low_breadth: float = 0.35
    high_breadth: float = 0.55
    name: str = "breadth_regime_rotation"
    ideas: tuple[str, ...] = ("02-全天候", "05/07-大盘择时")

    def target_weights(self, data: MarketData, signal_index: int, previous: np.ndarray) -> np.ndarray:
        if signal_index % self.rebalance_days != 0:
            return previous
        moving_average = trailing_mean(data.close, signal_index, self.ma_window)
        above = np.isfinite(moving_average) & (data.close[signal_index] > moving_average)
        breadth = float(np.mean(above))
        btc = data.symbol_index("BTC-USDT")
        if breadth < self.low_breadth:
            return np.zeros(len(data.symbols))
        if breadth < self.high_breadth:
            if above[btc]:
                return equal_weights(np.array([btc], dtype=int), len(data.symbols))
            return np.zeros(len(data.symbols))
        momentum = trailing_return(data.close, signal_index, self.lookback)
        score = np.where(above & np.isfinite(momentum) & (momentum > 0), momentum, np.nan)
        return equal_weights(finite_top(score, self.top_k), len(data.symbols))


@dataclass
class CoreSatelliteVolScaled:
    """BTC/ETH 核心 + 山寨卫星，并按波动率缩放总敞口。

    借鉴策略 02 的风格切换与策略 03/04 的动态仓位：核心用逆波动率配置，
    卫星只分配给正动量山寨；高波动时主动降仓，换手保持很低。
    """

    regime_window: int = 120
    lookback: int = 60
    satellite_count: int = 2
    rebalance_days: int = 30
    core_weight: float = 0.70
    vol_scale: float = 0.70
    vol_window: int = 30
    name: str = "core_satellite_vol_scaled"
    ideas: tuple[str, ...] = ("02-风格轮动", "03/04-动态仓位")

    def target_weights(self, data: MarketData, signal_index: int, previous: np.ndarray) -> np.ndarray:
        if signal_index % self.rebalance_days != 0:
            return previous
        if not _btc_trend_on(data, signal_index, self.regime_window):
            return np.zeros(len(data.symbols))
        btc = data.symbol_index("BTC-USDT")
        eth = data.symbol_index("ETH-USDT")
        volatility = trailing_volatility(data.close, signal_index, self.vol_window)
        momentum = trailing_return(data.close, signal_index, self.lookback)
        weights = np.zeros(len(data.symbols), dtype=float)
        core = np.array([index for index in (btc, eth) if np.isfinite(volatility[index]) and volatility[index] > 0])
        if len(core):
            weights += inverse_volatility_weights(
                volatility,
                core,
                len(data.symbols),
                gross_exposure=self.core_weight,
            )
        alt_score = momentum.copy()
        alt_score[[btc, eth]] = np.nan
        alt_score = np.where(np.isfinite(alt_score) & (alt_score > 0), alt_score, np.nan)
        satellites = finite_top(alt_score, self.satellite_count)
        if len(satellites):
            weights[satellites] = (1.0 - self.core_weight) / len(satellites)
        portfolio_vol = float(np.nansum(np.abs(weights) * volatility))
        if portfolio_vol > 1e-8:
            weights *= min(1.0, self.vol_scale / portfolio_vol)
        return weights


@dataclass
class MajorsAltsRegime:
    """主流币与山寨币相对强弱切换，并叠加 BTC 趋势过滤。

    这是对策略 02 唯一弱正信号的强化版：保留大小盘风格切换，
    额外要求 BTC 处于上升趋势，且改为更低换手的月频再平衡。
    """

    style_window: int = 40
    trend_window: int = 100
    top_k: int = 3
    rebalance_days: int = 30
    name: str = "majors_alts_regime"
    ideas: tuple[str, ...] = ("02-全天候强化",)

    def target_weights(self, data: MarketData, signal_index: int, previous: np.ndarray) -> np.ndarray:
        if signal_index % self.rebalance_days != 0:
            return previous
        if not _btc_trend_on(data, signal_index, self.trend_window):
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
        return equal_weights(selected, len(data.symbols))


def optimized_strategy_catalog() -> dict[str, type]:
    """返回优化策略名称到实现类型的映射。"""

    return {
        "btc_dual_momentum": BtcDualMomentum,
        "breadth_regime_rotation": BreadthRegimeRotation,
        "core_satellite_vol_scaled": CoreSatelliteVolScaled,
        "majors_alts_regime": MajorsAltsRegime,
    }
