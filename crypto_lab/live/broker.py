"""模拟盘与实盘干跑经纪商及组合状态。"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol

import numpy as np

from .models import Fill, Position


def utc_now_iso() -> str:
    """返回 UTC ISO 时间戳字符串。"""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Broker(Protocol):
    """经纪商最小接口：按目标权重调仓并返回成交。"""

    def rebalance(
        self,
        *,
        deployment_id: str,
        strategy: str,
        mode: str,
        symbols: tuple[str, ...],
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        quantities: dict[str, float],
        prices: dict[str, float],
        equity: float,
        fee_rate: float,
        slippage_rate: float,
    ) -> list[Fill]:
        """执行或模拟调仓。"""


def _quantity_deltas(
    symbols: tuple[str, ...],
    target_weights: np.ndarray,
    quantities: dict[str, float],
    prices: dict[str, float],
    equity: float,
    min_delta_weight: float,
) -> list[tuple[int, str, float, float, float]]:
    """把目标权重转换为相对当前持仓的数量差额。

    返回 (index, symbol, delta_qty, weight_from, weight_to)。按数量差额交易
    可避免“权重 1.0”因费用/价格变化无法精确平仓的问题。
    """

    rows: list[tuple[int, str, float, float, float]] = []
    if equity <= 0:
        return rows
    for index, symbol in enumerate(symbols):
        price = float(prices.get(symbol, 0.0))
        if price <= 0:
            continue
        current_qty = float(quantities.get(symbol, 0.0))
        weight_from = current_qty * price / equity
        weight_to = float(target_weights[index])
        if abs(weight_to - weight_from) < min_delta_weight and abs(weight_to) > 1e-12:
            continue
        target_qty = weight_to * equity / price
        delta_qty = target_qty - current_qty
        if abs(weight_to) <= 1e-12:
            # 目标空仓时直接平掉全部数量，消除残余仓位
            delta_qty = -current_qty
        if abs(delta_qty) * price / equity < min_delta_weight and abs(weight_to) > 1e-12:
            continue
        if abs(delta_qty) < 1e-12:
            continue
        rows.append((index, symbol, delta_qty, weight_from, weight_to))
    return rows


@dataclass
class PaperBroker:
    """本地模拟经纪商：按最新价加减滑点成交并扣费。

    不连接交易所，只根据实盘（或缓存）价格更新账户，适合策略验证与
    持续盯盘。成交数量由目标权重对应的目标仓位与当前持仓差额决定。

    @author Cursor
    @since 0.3.0
    """

    min_delta: float = 0.005

    def rebalance(
        self,
        *,
        deployment_id: str,
        strategy: str,
        mode: str,
        symbols: tuple[str, ...],
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        quantities: dict[str, float],
        prices: dict[str, float],
        equity: float,
        fee_rate: float,
        slippage_rate: float,
    ) -> list[Fill]:
        """按持仓数量差额生成模拟成交清单。"""

        del current_weights
        fills: list[Fill] = []
        for _, symbol, delta_qty, weight_from, weight_to in _quantity_deltas(
            symbols,
            target_weights,
            quantities,
            prices,
            equity,
            self.min_delta,
        ):
            price = float(prices[symbol])
            trade_price = price * (1.0 + slippage_rate if delta_qty > 0 else 1.0 - slippage_rate)
            notional = abs(delta_qty) * trade_price
            fills.append(
                Fill(
                    id=str(uuid.uuid4()),
                    deployment_id=deployment_id,
                    strategy=strategy,
                    mode=mode,
                    timestamp=utc_now_iso(),
                    symbol=symbol,
                    side=_side_label(weight_from, weight_to),
                    quantity=delta_qty,
                    price=trade_price,
                    notional=notional,
                    fee=notional * fee_rate,
                    weight_from=weight_from,
                    weight_to=weight_to,
                    equity_before=equity,
                    note="paper fill",
                    status="filled",
                )
            )
        return fills


@dataclass
class LiveBroker:
    """实盘经纪商：默认干跑记录意向单，显式开启才允许真实下单。

    真实下单需要交易所私钥与签名实现；当前版本在 ``allow_orders=False``
    时只产出 status=intent 的记录，防止误触发送。即便开启，未完成签名
    适配前仍拒绝发单，只返回明确拒绝原因。

    @author Cursor
    @since 0.3.0
    """

    allow_orders: bool = False
    min_delta: float = 0.005
    api_key: str = ""
    api_secret: str = ""
    passphrase: str = ""

    def rebalance(
        self,
        *,
        deployment_id: str,
        strategy: str,
        mode: str,
        symbols: tuple[str, ...],
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        quantities: dict[str, float],
        prices: dict[str, float],
        equity: float,
        fee_rate: float,
        slippage_rate: float,
    ) -> list[Fill]:
        """生成实盘意向或（若开启）尝试提交订单。"""

        del current_weights, slippage_rate
        fills: list[Fill] = []
        for _, symbol, delta_qty, weight_from, weight_to in _quantity_deltas(
            symbols,
            target_weights,
            quantities,
            prices,
            equity,
            self.min_delta,
        ):
            price = float(prices[symbol])
            notional = abs(delta_qty) * price
            side = _side_label(weight_from, weight_to)
            status = "intent"
            note = "live dry-run intent"
            if self.allow_orders:
                submitted = self._submit_order(symbol, side, abs(delta_qty), price)
                status = submitted["status"]
                note = submitted["note"]
            fills.append(
                Fill(
                    id=str(uuid.uuid4()),
                    deployment_id=deployment_id,
                    strategy=strategy,
                    mode=mode,
                    timestamp=utc_now_iso(),
                    symbol=symbol,
                    side=side,
                    quantity=delta_qty,
                    price=price,
                    notional=notional,
                    fee=notional * fee_rate,
                    weight_from=weight_from,
                    weight_to=weight_to,
                    equity_before=equity,
                    note=note,
                    status=status,
                )
            )
        return fills

    def _submit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
    ) -> dict[str, str]:
        """提交真实订单的扩展点；当前拒绝执行以保护资金安全。"""

        if not (self.api_key and self.api_secret and self.passphrase):
            return {
                "status": "rejected",
                "note": "缺少 OKX API 凭证，拒绝实盘发单",
            }
        return {
            "status": "rejected",
            "note": (
                f"实盘发单适配器尚未启用：{symbol} {side} qty={quantity:.6f} "
                f"@ {price:.4f}。请先完成签名下单与风控联调。"
            ),
        }


@dataclass
class PortfolioState:
    """部署级组合状态：现金、持仓数量、权益曲线与已实现盈亏。

    记账规则：
    - 买入（quantity>0）支付 notional+fee，按加权均价更新多头成本；
    - 卖出（quantity<0）收回 notional-fee，并对减少的多头仓计算已实现盈亏；
    - 空头开平仓同样支持，便于后续打开做空模块。

    @author Cursor
    @since 0.3.0
    """

    initial_equity: float
    cash: float
    quantities: dict[str, float] = field(default_factory=dict)
    avg_prices: dict[str, float] = field(default_factory=dict)
    realized_pnl: float = 0.0
    equity_curve: list[float] = field(default_factory=list)
    daily_returns: list[float] = field(default_factory=list)

    @classmethod
    def create(cls, initial_equity: float, symbols: tuple[str, ...]) -> "PortfolioState":
        """以全现金初始化组合。"""

        return cls(
            initial_equity=initial_equity,
            cash=initial_equity,
            quantities={symbol: 0.0 for symbol in symbols},
            avg_prices={symbol: 0.0 for symbol in symbols},
            equity_curve=[initial_equity],
        )

    def apply_fills(self, fills: list[Fill]) -> None:
        """应用已成交记录，更新现金、持仓与已实现盈亏。"""

        for fill in fills:
            if fill.status != "filled":
                continue
            qty = float(fill.quantity)
            price = float(fill.price)
            fee = float(fill.fee)
            held = float(self.quantities.get(fill.symbol, 0.0))
            avg = float(self.avg_prices.get(fill.symbol, 0.0))

            if qty > 0:
                self.cash -= qty * price + fee
                if held >= 0:
                    new_held = held + qty
                    self.avg_prices[fill.symbol] = (
                        (held * avg + qty * price) / new_held if new_held else 0.0
                    )
                    self.quantities[fill.symbol] = new_held
                else:
                    # 平空后再反手
                    cover = min(qty, -held)
                    self.realized_pnl += (avg - price) * cover
                    remaining = qty - cover
                    held = held + cover
                    if remaining > 0:
                        held = remaining
                        avg = price
                    elif abs(held) < 1e-12:
                        held = 0.0
                        avg = 0.0
                    self.quantities[fill.symbol] = held
                    self.avg_prices[fill.symbol] = avg
            else:
                sell_qty = -qty
                self.cash += sell_qty * price - fee
                if held <= 0:
                    new_held = held - sell_qty
                    self.avg_prices[fill.symbol] = (
                        (abs(held) * avg + sell_qty * price) / abs(new_held)
                        if abs(new_held) > 1e-12
                        else 0.0
                    )
                    self.quantities[fill.symbol] = new_held
                else:
                    close = min(sell_qty, held)
                    self.realized_pnl += (price - avg) * close
                    remaining = sell_qty - close
                    held = held - close
                    if remaining > 0:
                        held = -remaining
                        avg = price
                    elif abs(held) < 1e-12:
                        held = 0.0
                        avg = 0.0
                    self.quantities[fill.symbol] = held
                    self.avg_prices[fill.symbol] = avg

    def valuation(
        self,
        prices: dict[str, float],
    ) -> tuple[float, float, list[Position]]:
        """只读估值：返回权益、浮动盈亏与持仓列表，不改写曲线。"""

        positions: list[Position] = []
        market_value = 0.0
        unrealized = 0.0
        for symbol, qty in self.quantities.items():
            if abs(qty) < 1e-12:
                continue
            price = float(prices.get(symbol, 0.0))
            avg = float(self.avg_prices.get(symbol, 0.0))
            value = qty * price
            pnl = (price - avg) * qty
            market_value += value
            unrealized += pnl
            positions.append(
                Position(
                    symbol=symbol,
                    quantity=qty,
                    avg_price=avg,
                    market_price=price,
                    market_value=value,
                    unrealized_pnl=pnl,
                    weight=0.0,
                )
            )
        equity = self.cash + market_value
        for position in positions:
            position.weight = position.market_value / equity if equity else 0.0
        return equity, unrealized, positions

    def record_equity(self, equity: float) -> None:
        """把最新权益写入曲线并计算区间收益。"""

        if self.equity_curve:
            previous = self.equity_curve[-1]
            self.daily_returns.append(equity / previous - 1.0 if previous > 0 else 0.0)
        self.equity_curve.append(equity)

    def current_weights(self, symbols: tuple[str, ...], prices: dict[str, float]) -> np.ndarray:
        """根据持仓市值估算当前权重向量。"""

        equity, _, _ = self.valuation(prices)
        weights = np.zeros(len(symbols), dtype=float)
        if equity <= 0:
            return weights
        for index, symbol in enumerate(symbols):
            qty = self.quantities.get(symbol, 0.0)
            price = float(prices.get(symbol, 0.0))
            weights[index] = qty * price / equity
        return weights


def performance_metrics(equity_curve: list[float], daily_returns: list[float]) -> dict[str, float]:
    """从权益曲线计算总收益、最大回撤与近似夏普。"""

    if not equity_curve:
        return {"total_return": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}
    total_return = equity_curve[-1] / equity_curve[0] - 1.0 if equity_curve[0] else 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        max_dd = max(max_dd, 1.0 - value / peak if peak > 0 else 0.0)
    sharpe = 0.0
    if len(daily_returns) >= 2:
        arr = np.asarray(daily_returns, dtype=float)
        vol = float(np.std(arr, ddof=1))
        if vol > 1e-12:
            sharpe = float(np.mean(arr) / vol * np.sqrt(365.0))
    return {
        "total_return": float(total_return),
        "max_drawdown": float(max_dd),
        "sharpe": float(sharpe),
    }


def _side_label(weight_from: float, weight_to: float) -> str:
    """根据权重前后变化生成买卖方向标签。"""

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
