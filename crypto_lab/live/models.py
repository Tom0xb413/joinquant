"""运行时领域模型：成交、持仓、账户快照与买卖点。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Fill:
    """一笔已确认成交或意向单。

    ``status`` 取值：
    - ``filled``：模拟或实盘已成交；
    - ``intent``：实盘干跑意向，尚未发往交易所；
    - ``rejected``：风控或交易所拒绝。

    @author Cursor
    @since 0.3.0
    """

    id: str
    deployment_id: str
    strategy: str
    mode: str
    timestamp: str
    symbol: str
    side: str
    quantity: float
    price: float
    notional: float
    fee: float
    weight_from: float
    weight_to: float
    equity_before: float
    note: str = ""
    status: str = "filled"

    def to_dict(self) -> dict[str, Any]:
        """序列化为 JSON 友好字典。"""

        return asdict(self)


@dataclass
class Position:
    """单个标的的当前持仓与浮动盈亏。

    数量以基础币计，正多为空负；成本价按成交均价滚动更新，便于 UI 展示。

    @author Cursor
    @since 0.3.0
    """

    symbol: str
    quantity: float
    avg_price: float
    market_price: float
    market_value: float
    unrealized_pnl: float
    weight: float

    def to_dict(self) -> dict[str, Any]:
        """序列化为 JSON 友好字典。"""

        return asdict(self)


@dataclass
class AccountSnapshot:
    """某一时刻账户权益与绩效快照。

    同时保存权重向量与权益序列摘要，供 Web 实时刷新与通知摘要复用。

    @author Cursor
    @since 0.3.0
    """

    deployment_id: str
    strategy: str
    mode: str
    updated_at: str
    cash: float
    equity: float
    initial_equity: float
    total_return: float
    day_pnl: float
    unrealized_pnl: float
    realized_pnl: float
    max_drawdown: float
    sharpe: float
    positions: list[Position] = field(default_factory=list)
    target_weights: dict[str, float] = field(default_factory=dict)
    last_signal_date: str = ""
    status: str = "idle"
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """序列化为嵌套字典，持仓一并展开。"""

        payload = asdict(self)
        return payload


@dataclass
class TradePoint:
    """历史买卖点清单中的一条记录。

    来自回测权重差分或实盘/模拟成交，统一字段方便 Web 筛选与导出。

    @author Cursor
    @since 0.3.0
    """

    date: str
    deployment_id: str
    strategy: str
    mode: str
    symbol: str
    side: str
    price: float
    weight_from: float
    weight_to: float
    weight_delta: float
    equity_before: float
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        """序列化为 JSON 友好字典。"""

        return asdict(self)
