"""无未来函数、含交易成本的多资产日频回测引擎。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

import numpy as np

from .data import MarketData
from .strategies import Strategy


@dataclass(frozen=True)
class PerformanceMetrics:
    """一段回测区间的核心风险收益指标。"""

    start: str
    end: str
    observations: int
    total_return: float
    cagr: float
    annual_volatility: float
    sharpe: float
    max_drawdown: float
    calmar: float
    turnover: float
    cost_paid: float


@dataclass(frozen=True)
class BacktestResult:
    """完整净值、权重和指标结果。"""

    strategy: str
    dates: tuple[date, ...]
    daily_returns: np.ndarray
    equity: np.ndarray
    weights: np.ndarray
    turnover: np.ndarray
    costs: np.ndarray

    def metrics(self, start: date | None = None, end: date | None = None) -> PerformanceMetrics:
        """计算指定日期区间指标；区间边界均包含。"""

        mask = np.ones(len(self.dates), dtype=bool)
        if start is not None:
            mask &= np.asarray(self.dates) >= start
        if end is not None:
            mask &= np.asarray(self.dates) <= end
        indices = np.flatnonzero(mask)
        if len(indices) < 2:
            raise ValueError("指标区间至少需要两个观测值")
        returns = self.daily_returns[indices]
        equity = np.cumprod(1.0 + returns)
        years = len(returns) / 365.0
        total_return = float(equity[-1] - 1.0)
        cagr = float(equity[-1] ** (1.0 / years) - 1.0) if equity[-1] > 0 else -1.0
        volatility = float(np.std(returns, ddof=1) * np.sqrt(365.0))
        sharpe = float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(365.0)) if volatility else 0.0
        peaks = np.maximum.accumulate(equity)
        drawdowns = equity / peaks - 1.0
        max_drawdown = float(-np.min(drawdowns))
        calmar = cagr / max_drawdown if max_drawdown > 1e-12 else 0.0
        return PerformanceMetrics(
            start=self.dates[indices[0]].isoformat(),
            end=self.dates[indices[-1]].isoformat(),
            observations=len(indices),
            total_return=total_return,
            cagr=cagr,
            annual_volatility=volatility,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            calmar=calmar,
            turnover=float(np.sum(self.turnover[indices])),
            cost_paid=float(np.sum(self.costs[indices])),
        )

    def metrics_dict(self, start: date | None = None, end: date | None = None) -> dict:
        """返回适合 JSON 序列化的指标字典。"""

        return asdict(self.metrics(start, end))


def run_backtest(
    data: MarketData,
    strategy: Strategy,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
) -> BacktestResult:
    """按 T-1 收盘信号持有 T 日收益，计入双向调仓成本。"""

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
    cost_rate = fee_rate + slippage_rate

    for index in range(1, rows):
        target = np.asarray(strategy.target_weights(data, index - 1, current.copy()), dtype=float)
        _validate_weights(target, columns)
        turnover[index] = float(np.sum(np.abs(target - current)))
        costs[index] = turnover[index] * cost_rate
        asset_returns = data.close[index] / data.close[index - 1] - 1.0
        gross_return = float(target @ asset_returns)
        daily_returns[index] = max(gross_return - costs[index], -1.0)
        weights[index] = target
        current = _drift_weights(target, asset_returns, gross_return)

    equity = np.cumprod(1.0 + daily_returns)
    return BacktestResult(
        strategy=strategy.name,
        dates=data.dates,
        daily_returns=daily_returns,
        equity=equity,
        weights=weights,
        turnover=turnover,
        costs=costs,
    )


def buy_and_hold(
    data: MarketData,
    symbol: str = "BTC-USDT",
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
) -> BacktestResult:
    """生成一次买入后持有的基准，初始建仓同样计成本。"""

    column = data.symbol_index(symbol)
    weights = np.zeros_like(data.close)
    weights[1:, column] = 1.0
    returns = np.zeros(len(data.dates))
    returns[1:] = data.close[1:, column] / data.close[:-1, column] - 1.0
    turnover = np.zeros(len(data.dates))
    turnover[1] = 1.0
    costs = turnover * (fee_rate + slippage_rate)
    returns -= costs
    return BacktestResult(
        strategy=f"{symbol}_buy_hold",
        dates=data.dates,
        daily_returns=returns,
        equity=np.cumprod(1.0 + returns),
        weights=weights,
        turnover=turnover,
        costs=costs,
    )


def _validate_weights(weights: np.ndarray, expected_size: int) -> None:
    """拒绝空头、杠杆和非有限权重，避免静默产生错误结果。"""

    if weights.shape != (expected_size,):
        raise ValueError(f"目标权重形状应为 {(expected_size,)}，实际为 {weights.shape}")
    if not np.isfinite(weights).all():
        raise ValueError("目标权重含 NaN 或无穷值")
    if np.any(weights < -1e-12):
        raise ValueError("当前引擎只允许多头权重")
    if float(weights.sum()) > 1.0 + 1e-9:
        raise ValueError("目标权重总和不能超过 1")


def _drift_weights(weights: np.ndarray, returns: np.ndarray, portfolio_return: float) -> np.ndarray:
    """按当日资产涨跌更新收盘后权重，剩余部分视为现金。"""

    denominator = 1.0 + portfolio_return
    if denominator <= 0:
        return np.zeros_like(weights)
    drifted = weights * (1.0 + returns) / denominator
    drifted[drifted < 1e-14] = 0.0
    return drifted

