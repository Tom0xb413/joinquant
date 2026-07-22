"""可插拔策略注册表，支撑后续多策略回测/模拟/实盘部署。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..core_top5 import CORE_TOP5_SYMBOLS, CoreTop5RegimeRotation
from ..crypto_alpha import BtcTrendTopMomentum
from ..strategies import Strategy


@dataclass(frozen=True)
class StrategySpec:
    """描述一个可在控制台部署的策略蓝图。

    factory 只接收参数字典并返回满足 Strategy Protocol 的实例；symbols
    给出该策略默认交易宇宙，便于行情预热与 UI 展示。

    @author Cursor
    @since 0.3.0
    """

    key: str
    title: str
    description: str
    symbols: tuple[str, ...]
    factory: Callable[[dict[str, Any]], Strategy]
    default_parameters: dict[str, Any]


def strategy_catalog() -> dict[str, StrategySpec]:
    """返回当前可用策略目录，后续新增策略只需在此注册一项。

    控制台、引擎和买卖点清单都通过该目录解析策略，避免散落 if/else。
    """

    return {
        "core_top5_regime_rotation": StrategySpec(
            key="core_top5_regime_rotation",
            title="TOP5 激进牛熊轮动",
            description="固定五币核心池，牛市轮动+关键位杠杆，熊市做空可降级现金",
            symbols=CORE_TOP5_SYMBOLS,
            factory=lambda params: CoreTop5RegimeRotation(**params),
            default_parameters={
                "top_k": 1,
                "rebalance_days": 14,
                "vol_target": 0.45,
                "breakout_min_gross": 1.2,
                "leveraged_max_gross": 1.3,
                "short_gross": 0.0,
            },
        ),
        "btc_trend_top_momentum": StrategySpec(
            key="btc_trend_top_momentum",
            title="BTC门控 Top 动量",
            description="BTC 趋势开启后持有正动量 Top-K，风险关空仓",
            symbols=CORE_TOP5_SYMBOLS,
            factory=lambda params: BtcTrendTopMomentum(**params),
            default_parameters={
                "trend_window": 150,
                "lookback": 90,
                "top_k": 2,
                "rebalance_days": 14,
                "vol_target": 0.30,
                "max_gross": 1.0,
            },
        ),
    }


def build_strategy(key: str, parameters: dict[str, Any] | None = None) -> Strategy:
    """按注册表构造策略实例，未知 key 时立即失败。"""

    catalog = strategy_catalog()
    if key not in catalog:
        raise KeyError(f"未知策略：{key}；可选：{', '.join(catalog)}")
    spec = catalog[key]
    merged = dict(spec.default_parameters)
    if parameters:
        merged.update(parameters)
    return spec.factory(merged)
