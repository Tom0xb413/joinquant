"""历史买卖点清单：回测权重差分与运行时成交统一输出。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..data import align_market_data, load_candles
from ..long_short import LongShortLimits, run_long_short_backtest
from .models import Fill, TradePoint
from .registry import build_strategy, strategy_catalog


def build_backtest_trade_points(
    *,
    deployment_id: str,
    strategy_key: str,
    parameters: dict[str, Any],
    data_dir: Path,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    min_delta: float = 0.01,
) -> list[TradePoint]:
    """对缓存行情跑一遍策略并提取买卖点清单。

    使用与研究相同的多空引擎，保证清单与回测路径一致；仅输出权重变化
    超过阈值的调仓，避免漂移噪声。

    @author Cursor
    @since 0.3.0
    """

    spec = strategy_catalog()[strategy_key]
    series = {
        symbol: load_candles(data_dir / f"{symbol}.csv")
        for symbol in spec.symbols
    }
    data = align_market_data(series)
    strategy = build_strategy(strategy_key, parameters)
    result = run_long_short_backtest(
        data,
        strategy,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        limits=LongShortLimits(
            max_gross_exposure=1.5,
            max_net_exposure=1.5,
            max_short_exposure=0.30,
        ),
    )
    points: list[TradePoint] = []
    previous = np.zeros(len(data.symbols), dtype=float)
    for index, day in enumerate(result.dates):
        current = np.asarray(result.weights[index], dtype=float)
        if float(result.turnover[index]) < min_delta:
            previous = current
            continue
        deltas = current - previous
        for column, delta in enumerate(deltas):
            if abs(delta) < min_delta:
                continue
            points.append(
                TradePoint(
                    date=day.isoformat(),
                    deployment_id=deployment_id,
                    strategy=strategy_key,
                    mode="backtest",
                    symbol=data.symbols[column],
                    side=_side_from_weights(previous[column], current[column]),
                    price=float(data.close[index, column]),
                    weight_from=float(previous[column]),
                    weight_to=float(current[column]),
                    weight_delta=float(delta),
                    equity_before=float(result.equity[index - 1]) if index else 1.0,
                    note="backtest rebalance",
                )
            )
        previous = current
    return points


def fills_to_trade_points(fills: list[Fill]) -> list[TradePoint]:
    """把运行时成交转换为买卖点记录。"""

    return [
        TradePoint(
            date=fill.timestamp[:10],
            deployment_id=fill.deployment_id,
            strategy=fill.strategy,
            mode=fill.mode,
            symbol=fill.symbol,
            side=fill.side,
            price=fill.price,
            weight_from=fill.weight_from,
            weight_to=fill.weight_to,
            weight_delta=fill.weight_to - fill.weight_from,
            equity_before=fill.equity_before,
            note=f"{fill.status}: {fill.note}",
        )
        for fill in fills
    ]


def summarize_trade_points(points: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总买卖点：次数、标的分布与多空方向占比。"""

    if not points:
        return {
            "count": 0,
            "buy_like": 0,
            "sell_like": 0,
            "symbols": {},
            "modes": {},
        }
    symbols: dict[str, int] = {}
    modes: dict[str, int] = {}
    buy_like = 0
    sell_like = 0
    for point in points:
        symbols[point["symbol"]] = symbols.get(point["symbol"], 0) + 1
        modes[point["mode"]] = modes.get(point["mode"], 0) + 1
        side = point["side"]
        if "buy" in side or side == "cover":
            buy_like += 1
        if "sell" in side or "short" in side:
            sell_like += 1
    return {
        "count": len(points),
        "buy_like": buy_like,
        "sell_like": sell_like,
        "symbols": symbols,
        "modes": modes,
    }


def _side_from_weights(weight_from: float, weight_to: float) -> str:
    """由权重变化推断买卖方向标签。"""

    delta = weight_to - weight_from
    if weight_from >= 0 and weight_to >= 0:
        return "buy" if delta > 0 else "sell"
    if weight_from <= 0 and weight_to <= 0:
        return "short" if delta < 0 else "cover"
    if weight_from < 0 < weight_to:
        return "cover_and_buy"
    if weight_from > 0 > weight_to:
        return "sell_and_short"
    return "rebalance"
