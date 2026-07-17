"""单资产 EMA 策略回测引擎。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np

from .ema_data import BarSeries


@dataclass(frozen=True)
class EmaMetrics:
    """单资产回测指标。"""

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
    time_in_market: float
    trade_count: int
    bars_per_year: float


@dataclass(frozen=True)
class EmaBacktestResult:
    """单资产回测结果。"""

    strategy: str
    symbol: str
    timeframe: str
    timestamps_ms: np.ndarray
    positions: np.ndarray
    daily_returns: np.ndarray
    equity: np.ndarray
    turnover: np.ndarray
    costs: np.ndarray
    bars_per_year: float

    def metrics(self, start_index: int = 0, end_index: int | None = None) -> EmaMetrics:
        """计算指定下标区间（含）的绩效。"""

        end = len(self.daily_returns) if end_index is None else end_index + 1
        if end - start_index < 2:
            raise ValueError("指标区间至少需要两个观测")
        returns = self.daily_returns[start_index:end]
        equity = np.cumprod(1.0 + returns)
        years = len(returns) / self.bars_per_year
        total_return = float(equity[-1] - 1.0)
        cagr = float(equity[-1] ** (1.0 / years) - 1.0) if equity[-1] > 0 and years > 0 else -1.0
        volatility = float(np.std(returns, ddof=1) * np.sqrt(self.bars_per_year))
        sharpe = (
            float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(self.bars_per_year))
            if volatility > 0
            else 0.0
        )
        anchored = np.concatenate(([1.0], equity))
        peaks = np.maximum.accumulate(anchored)
        max_drawdown = float(-np.min(anchored / peaks - 1.0))
        calmar = cagr / max_drawdown if max_drawdown > 1e-12 else 0.0
        positions = self.positions[start_index:end]
        turnover = self.turnover[start_index:end]
        trade_count = int(np.sum(np.abs(np.diff(np.concatenate(([0.0], positions)))) > 1e-9))
        return EmaMetrics(
            start=_ts_iso(int(self.timestamps_ms[start_index])),
            end=_ts_iso(int(self.timestamps_ms[end - 1])),
            observations=len(returns),
            total_return=total_return,
            cagr=cagr,
            annual_volatility=volatility,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            calmar=calmar,
            turnover=float(np.sum(turnover)),
            cost_paid=float(np.sum(self.costs[start_index:end])),
            time_in_market=float(np.mean(np.abs(positions) > 1e-9)),
            trade_count=trade_count,
            bars_per_year=self.bars_per_year,
        )

    def metrics_dict(self, start_index: int = 0, end_index: int | None = None) -> dict:
        return asdict(self.metrics(start_index, end_index))


def bars_per_year(timeframe: str) -> float:
    """估算年化因子。"""

    mapping = {
        "1D": 365.0,
        "8H": 365.0 * 3.0,
        "4H": 365.0 * 6.0,
    }
    if timeframe not in mapping:
        raise ValueError(f"未知周期：{timeframe}")
    return mapping[timeframe]


def run_ema_backtest(
    series: BarSeries,
    strategy,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
) -> EmaBacktestResult:
    """T 根收盘信号，持有 T->T+1 收益；换手计费。"""

    if fee_rate < 0 or slippage_rate < 0:
        raise ValueError("费率不能为负")
    raw = np.asarray(strategy.positions(series), dtype=float)
    if raw.shape != (series.size,):
        raise ValueError("仓位长度必须与 K 线一致")
    if not np.isfinite(raw).all():
        raise ValueError("仓位含 NaN")
    if np.any(np.abs(raw) > 1.0 + 1e-9):
        raise ValueError("单资产仓位绝对值不能超过 1")

    # 信号延后一根，避免同根成交
    positions = np.zeros_like(raw)
    positions[1:] = raw[:-1]

    returns = np.zeros(series.size, dtype=float)
    asset_returns = np.zeros(series.size, dtype=float)
    asset_returns[1:] = series.close[1:] / series.close[:-1] - 1.0
    turnover = np.zeros(series.size, dtype=float)
    costs = np.zeros(series.size, dtype=float)
    cost_rate = fee_rate + slippage_rate
    previous = 0.0
    for index in range(1, series.size):
        target = float(positions[index])
        turnover[index] = abs(target - previous)
        costs[index] = turnover[index] * cost_rate
        gross = target * float(asset_returns[index])
        returns[index] = max((1.0 + gross) * (1.0 - costs[index]) - 1.0, -1.0)
        previous = target

    return EmaBacktestResult(
        strategy=strategy.name,
        symbol=series.symbol,
        timeframe=series.timeframe,
        timestamps_ms=series.timestamps_ms.copy(),
        positions=positions,
        daily_returns=returns,
        equity=np.cumprod(1.0 + returns),
        turnover=turnover,
        costs=costs,
        bars_per_year=bars_per_year(series.timeframe),
    )


def buy_and_hold_series(
    series: BarSeries,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
) -> EmaBacktestResult:
    """买入持有基准。"""

    class _Hold:
        name = f"{series.symbol}_buy_hold"

        def positions(self, data: BarSeries) -> np.ndarray:
            return np.ones(data.size, dtype=float)

    return run_ema_backtest(series, _Hold(), fee_rate, slippage_rate)


def _ts_iso(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()
