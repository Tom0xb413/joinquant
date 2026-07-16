"""由 12 个聚宽策略归并而来的 Crypto 策略族。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .data import MarketData
from .indicators import (
    cross_sectional_zscore,
    equal_weights,
    finite_top,
    rsi,
    trailing_mean,
    trailing_return,
    trailing_volatility,
)


class Strategy(Protocol):
    """回测引擎所需的最小策略接口。"""

    name: str
    source_ids: tuple[str, ...]

    def target_weights(
        self,
        data: MarketData,
        signal_index: int,
        previous: np.ndarray,
    ) -> np.ndarray:
        """仅用 signal_index 及之前数据生成下一日目标权重。"""


@dataclass
class TrendRotation:
    """迁移策略 01：均线与横截面动量轮动。"""

    lookback: int = 20
    short_window: int = 5
    top_k: int = 2
    rank_limit: int = 4
    min_return: float = 0.03
    max_return: float = 0.30
    name: str = "trend_rotation"
    source_ids: tuple[str, ...] = ("01",)

    def target_weights(self, data: MarketData, signal_index: int, previous: np.ndarray) -> np.ndarray:
        momentum = trailing_return(data.close, signal_index, self.lookback)
        short_ma = trailing_mean(data.close, signal_index, self.short_window)
        long_ma = trailing_mean(data.close, signal_index, self.lookback)
        eligible = (
            np.isfinite(momentum)
            & (short_ma > long_ma)
            & (momentum > self.min_return)
            & (momentum < self.max_return)
        )
        ranked = finite_top(np.where(eligible, momentum, np.nan), self.rank_limit)
        retained = [index for index in ranked if previous[index] > 0]
        candidates = [index for index in ranked if eligible[index] and index not in retained]
        selected = np.array((retained + candidates)[: self.top_k], dtype=int)
        return equal_weights(selected, len(data.symbols))


@dataclass
class AllWeatherRotation:
    """迁移策略 02：主流币、山寨币与现金之间的风险状态切换。"""

    regime_window: int = 20
    top_k: int = 4
    rebalance_days: int = 30
    name: str = "all_weather_rotation"
    source_ids: tuple[str, ...] = ("02",)

    def target_weights(self, data: MarketData, signal_index: int, previous: np.ndarray) -> np.ndarray:
        if signal_index % self.rebalance_days != 0:
            return previous
        momentum = trailing_return(data.close, signal_index, self.regime_window)
        if not np.isfinite(momentum).all():
            return np.zeros(len(data.symbols))
        btc = data.symbol_index("BTC-USDT")
        eth = data.symbol_index("ETH-USDT")
        major = np.array([btc, eth])
        alt = np.array([index for index in range(len(data.symbols)) if index not in major])
        major_score = float(np.mean(momentum[major]))
        alt_score = float(np.mean(momentum[alt]))
        if max(major_score, alt_score) <= 0:
            return np.zeros(len(data.symbols))
        universe = major if major_score >= alt_score else alt
        local = universe[np.argsort(momentum[universe], kind="stable")[::-1][: self.top_k]]
        return equal_weights(local, len(data.symbols))


@dataclass
class SmallLiquidityRotation:
    """迁移策略 03/04/06/10-A：在可交易池中捕捉“小盘”代理效应。"""

    liquidity_window: int = 30
    liquid_pool: int = 10
    top_k: int = 5
    regime_window: int = 80
    name: str = "small_liquidity_rotation"
    source_ids: tuple[str, ...] = ("03", "04", "06", "10-A")

    def target_weights(self, data: MarketData, signal_index: int, previous: np.ndarray) -> np.ndarray:
        if signal_index % 7 != 0:
            return previous
        btc = data.symbol_index("BTC-USDT")
        btc_ma = trailing_mean(data.close, signal_index, self.regime_window)[btc]
        if not np.isfinite(btc_ma) or data.close[signal_index, btc] <= btc_ma:
            return np.zeros(len(data.symbols))
        liquidity = trailing_mean(data.volume_quote, signal_index, self.liquidity_window)
        momentum = trailing_return(data.close, signal_index, self.liquidity_window)
        liquid = finite_top(liquidity, self.liquid_pool)
        candidates = liquid[np.argsort(liquidity[liquid], kind="stable")]
        candidates = candidates[momentum[candidates] > 0]
        return equal_weights(candidates[: self.top_k], len(data.symbols))


@dataclass
class CompositeFactorRotation:
    """迁移策略 05/07/10-C/10-D/11/12：可交易量价多因子组合。"""

    top_k: int = 5
    regime_window: int = 100
    name: str = "composite_factor_rotation"
    source_ids: tuple[str, ...] = ("05", "07", "10-C", "10-D", "11", "12")

    def target_weights(self, data: MarketData, signal_index: int, previous: np.ndarray) -> np.ndarray:
        if signal_index % 7 != 0:
            return previous
        btc = data.symbol_index("BTC-USDT")
        btc_ma = trailing_mean(data.close, signal_index, self.regime_window)[btc]
        if not np.isfinite(btc_ma) or data.close[signal_index, btc] <= btc_ma:
            return np.zeros(len(data.symbols))
        momentum_30 = trailing_return(data.close, signal_index, 30)
        momentum_90 = trailing_return(data.close, signal_index, 90)
        volatility = trailing_volatility(data.close, signal_index, 30)
        volume_7 = trailing_mean(data.volume_quote, signal_index, 7)
        volume_30 = trailing_mean(data.volume_quote, signal_index, 30)
        volume_growth = np.log(np.maximum(volume_7, 1.0) / np.maximum(volume_30, 1.0))
        liquidity = np.log(np.maximum(volume_30, 1.0))
        score = (
            0.35 * cross_sectional_zscore(momentum_30)
            + 0.25 * cross_sectional_zscore(momentum_90)
            + 0.20 * cross_sectional_zscore(volume_growth)
            - 0.10 * cross_sectional_zscore(volatility)
            + 0.10 * cross_sectional_zscore(liquidity)
        )
        return equal_weights(finite_top(score, self.top_k), len(data.symbols))


@dataclass
class RsiFactorRotation:
    """迁移策略 08：慢频筛选与快频 RSI 退出。"""

    top_k: int = 6
    entry_rsi: float = 55.0
    exit_rsi: float = 72.0
    rebalance_days: int = 30
    name: str = "rsi_factor_rotation"
    source_ids: tuple[str, ...] = ("08",)

    def target_weights(self, data: MarketData, signal_index: int, previous: np.ndarray) -> np.ndarray:
        current_rsi = rsi(data.close, signal_index, 14)
        if signal_index % self.rebalance_days != 0:
            target = previous.copy()
            target[(previous > 0) & (current_rsi > self.exit_rsi)] = 0.0
            return target
        momentum = trailing_return(data.close, signal_index, 90)
        volatility = trailing_volatility(data.close, signal_index, 30)
        liquidity = trailing_mean(data.volume_quote, signal_index, 30)
        quality = (
            0.45 * cross_sectional_zscore(momentum)
            - 0.30 * cross_sectional_zscore(volatility)
            + 0.25 * cross_sectional_zscore(np.log(np.maximum(liquidity, 1.0)))
        )
        quality[current_rsi >= self.entry_rsi] = np.nan
        return equal_weights(finite_top(quality, self.top_k), len(data.symbols))


@dataclass
class RollingRidgeRotation:
    """迁移策略 09：使用滚动岭回归替代高过拟合 XGBoost。"""

    top_k: int = 5
    horizon: int = 7
    train_window: int = 365
    ridge: float = 1.0
    name: str = "rolling_ridge_rotation"
    source_ids: tuple[str, ...] = ("09",)

    def target_weights(self, data: MarketData, signal_index: int, previous: np.ndarray) -> np.ndarray:
        if signal_index % self.horizon != 0:
            return previous
        minimum = max(90, self.horizon * 8)
        if signal_index < minimum + self.horizon:
            return np.zeros(len(data.symbols))
        samples: list[np.ndarray] = []
        labels: list[float] = []
        start = max(90, signal_index - self.train_window)
        for index in range(start, signal_index - self.horizon, self.horizon):
            features = self._features(data, index)
            future = trailing_return(data.close, index + self.horizon, self.horizon)
            relative = future - np.nanmean(future)
            valid = np.isfinite(features).all(axis=1) & np.isfinite(relative)
            samples.extend(features[valid])
            labels.extend(relative[valid])
        if len(samples) < len(data.symbols) * 4:
            return np.zeros(len(data.symbols))
        x = np.asarray(samples, dtype=float)
        y = np.asarray(labels, dtype=float)
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-12] = 1.0
        standardized = (x - mean) / scale
        design = np.column_stack([np.ones(len(standardized)), standardized])
        penalty = np.eye(design.shape[1]) * self.ridge
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        current = self._features(data, signal_index)
        valid = np.isfinite(current).all(axis=1)
        predictions = np.full(len(data.symbols), np.nan)
        predictions[valid] = np.column_stack(
            [np.ones(valid.sum()), (current[valid] - mean) / scale]
        ) @ coefficients
        predictions[predictions <= 0] = np.nan
        return equal_weights(finite_top(predictions, self.top_k), len(data.symbols))

    @staticmethod
    def _features(data: MarketData, index: int) -> np.ndarray:
        """构造仅含价格、成交量和风险的可复现特征。"""

        momentum_7 = trailing_return(data.close, index, 7)
        momentum_30 = trailing_return(data.close, index, 30)
        momentum_90 = trailing_return(data.close, index, 90)
        volatility = trailing_volatility(data.close, index, 30)
        volume_short = trailing_mean(data.volume_quote, index, 7)
        volume_long = trailing_mean(data.volume_quote, index, 30)
        volume_growth = np.log(np.maximum(volume_short, 1.0) / np.maximum(volume_long, 1.0))
        return np.column_stack(
            [
                cross_sectional_zscore(momentum_7),
                cross_sectional_zscore(momentum_30),
                cross_sectional_zscore(momentum_90),
                cross_sectional_zscore(volatility),
                cross_sectional_zscore(volume_growth),
            ]
        )


def strategy_catalog() -> dict[str, type]:
    """返回稳定的策略名称到实现类型映射。"""

    return {
        "trend_rotation": TrendRotation,
        "all_weather_rotation": AllWeatherRotation,
        "small_liquidity_rotation": SmallLiquidityRotation,
        "composite_factor_rotation": CompositeFactorRotation,
        "rsi_factor_rotation": RsiFactorRotation,
        "rolling_ridge_rotation": RollingRidgeRotation,
    }

