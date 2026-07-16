"""支持多空权重的回测引擎扩展。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backtest import BacktestResult, _drift_weights
from .data import MarketData
from .strategies import Strategy


@dataclass(frozen=True)
class LongShortLimits:
    """多空组合的敞口与成本约束。"""

    max_gross_exposure: float = 1.5
    max_net_exposure: float = 1.5
    max_short_exposure: float = 0.8
    borrow_rate_daily: float = 0.00005  # 约 1.8%/年，近似永续空头持有成本


def run_long_short_backtest(
    data: MarketData,
    strategy: Strategy,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    limits: LongShortLimits | None = None,
) -> BacktestResult:
    """按 T-1 信号持有 T 日收益，允许多空并计入借券/资金费率近似成本。"""

    limits = limits or LongShortLimits()
    if fee_rate < 0 or slippage_rate < 0:
        raise ValueError("费率和滑点不能为负")
    rows, columns = data.close.shape
    if rows < 2:
        raise ValueError("回测至少需要两天行情")
    weights = np.zeros((rows, columns), dtype=float)
    daily_returns = np.zeros(rows, dtype=float)
    turnover = np.zeros(rows, dtype=float)
    costs = np.zeros(rows, dtype=float)
    current = np.zeros(columns, dtype=float)
    trade_cost_rate = fee_rate + slippage_rate

    for index in range(1, rows):
        target = np.asarray(strategy.target_weights(data, index - 1, current.copy()), dtype=float)
        _validate_long_short_weights(target, columns, limits)
        turnover[index] = float(np.sum(np.abs(target - current)))
        trade_cost = turnover[index] * trade_cost_rate
        short_notional = float(np.sum(np.maximum(-target, 0.0)))
        borrow_cost = short_notional * limits.borrow_rate_daily
        costs[index] = trade_cost + borrow_cost
        asset_returns = data.close[index] / data.close[index - 1] - 1.0
        gross_return = float(target @ asset_returns)
        daily_returns[index] = max((1.0 + gross_return) * (1.0 - trade_cost) - 1.0 - borrow_cost, -1.0)
        weights[index] = target
        current = _drift_weights(target, asset_returns, gross_return)

    return BacktestResult(
        strategy=strategy.name,
        dates=data.dates,
        daily_returns=daily_returns,
        equity=np.cumprod(1.0 + daily_returns),
        weights=weights,
        turnover=turnover,
        costs=costs,
    )


def _validate_long_short_weights(
    weights: np.ndarray,
    expected_size: int,
    limits: LongShortLimits,
) -> None:
    """校验多空权重的有限性与敞口上限。"""

    if weights.shape != (expected_size,):
        raise ValueError(f"目标权重形状应为 {(expected_size,)}，实际为 {weights.shape}")
    if not np.isfinite(weights).all():
        raise ValueError("目标权重含 NaN 或无穷值")
    gross = float(np.sum(np.abs(weights)))
    net = float(np.sum(weights))
    short = float(np.sum(np.maximum(-weights, 0.0)))
    if gross > limits.max_gross_exposure + 1e-9:
        raise ValueError(f"总敞口 {gross:.4f} 超过上限 {limits.max_gross_exposure}")
    if abs(net) > limits.max_net_exposure + 1e-9:
        raise ValueError(f"净敞口 {net:.4f} 超过上限 {limits.max_net_exposure}")
    if short > limits.max_short_exposure + 1e-9:
        raise ValueError(f"空头敞口 {short:.4f} 超过上限 {limits.max_short_exposure}")
