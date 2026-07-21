"""TOP5 核心资产的牛熊分档轮动、条件杠杆与自适应做空策略。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backtest import _drift_weights
from .crypto_alpha import _scale_to_vol_target
from .data import MarketData
from .indicators import (
    finite_top,
    inverse_volatility_weights,
    trailing_mean,
    trailing_return,
    trailing_volatility,
)


CORE_TOP5_SYMBOLS = (
    "BTC-USDT",
    "ETH-USDT",
    "SOL-USDT",
    "XRP-USDT",
    "DOGE-USDT",
)


@dataclass(frozen=True)
class ShortEdgeStats:
    """保存因果影子空头模块的滚动评价结果。

    影子模块只使用评价日及以前的价格，模拟满额空头袖套的收益和成本。
    策略据此决定下一调仓期是否值得承担做空风险，避免熊市中机械追空。

    @author Cursor
    @since 0.2.0
    """

    active_days: int
    trades: int
    total_return: float
    win_rate: float
    approved: bool


@dataclass
class CoreTop5RegimeRotation:
    """在固定 TOP5 核心池内执行牛熊分档轮动。

    基础牛市由 BTC 长趋势和正动量确认，宽度不足时保留 BTC 核心仓；
    快慢均线与核心池广度进一步确认后才轮动至动量最强资产，并只在 BTC
    突破此前关键高点时将总敞口放大到杠杆区间。熊市只考虑跌破趋势且负
    动量足够强的弱势标的；做空前先评价过去一段时间的影子空头净收益
    与胜率，任一门槛不达标便返回全现金。

    固定核心池能明确控制可交易边界，但它仍有幸存者偏差，并不代表这
    五个币会永久保持市值或流动性排名。杠杆上限是名义敞口约束，不模拟
    强平、保证金阶梯和逐币种资金费率，实盘必须另设交易所级风险控制。

    @author Cursor
    @since 0.2.0
    """

    core_symbols: tuple[str, ...] = CORE_TOP5_SYMBOLS
    slow_trend_window: int = 200
    fast_trend_window: int = 50
    momentum_window: int = 90
    fast_momentum_window: int = 30
    selection_trend_window: int = 60
    volatility_window: int = 30
    top_k: int = 2
    rebalance_days: int = 7
    breadth_min: float = 0.60
    vol_target: float = 0.40
    base_min_gross: float = 1.0
    base_max_gross: float = 1.0
    breakout_window: int = 55
    breakout_buffer: float = 0.0
    breakout_min_gross: float = 1.20
    leveraged_max_gross: float = 1.5
    short_top_k: int = 1
    short_gross: float = 0.30
    short_momentum_threshold: float = -0.12
    short_edge_window: int = 90
    short_min_trades: int = 3
    short_min_return: float = 0.01
    short_min_win_rate: float = 0.50
    shadow_cost_rate: float = 0.0015
    shadow_borrow_rate_daily: float = 0.00005
    name: str = "core_top5_regime_rotation"
    ideas: tuple[str, ...] = (
        "固定TOP5核心池",
        "牛市动量集中轮动",
        "关键位分档杠杆",
        "影子收益门控做空",
        "不稳则现金",
    )

    def __post_init__(self) -> None:
        """校验参数边界，防止无效窗口或失控敞口进入回测。

        校验在构造时执行，使错误配置尽早失败；特别限制基础敞口不高于
        杠杆敞口，并要求核心池恰好包含五个互不重复的标的。
        """

        if len(self.core_symbols) != 5 or len(set(self.core_symbols)) != 5:
            raise ValueError("core_symbols 必须恰好包含五个互不重复的标的")
        windows = (
            self.slow_trend_window,
            self.fast_trend_window,
            self.momentum_window,
            self.fast_momentum_window,
            self.selection_trend_window,
            self.volatility_window,
            self.breakout_window,
            self.short_edge_window,
        )
        if min(windows) < 2:
            raise ValueError("所有指标窗口必须至少为 2")
        if self.top_k < 1 or self.top_k > len(self.core_symbols):
            raise ValueError("top_k 必须在 1 到核心池大小之间")
        if self.short_top_k < 1 or self.short_top_k > len(self.core_symbols):
            raise ValueError("short_top_k 必须在 1 到核心池大小之间")
        if self.rebalance_days < 1:
            raise ValueError("rebalance_days 必须为正整数")
        if not 0.0 <= self.breadth_min <= 1.0:
            raise ValueError("breadth_min 必须位于 [0, 1]")
        if not 0.0 <= self.short_min_win_rate <= 1.0:
            raise ValueError("short_min_win_rate 必须位于 [0, 1]")
        if self.short_min_trades < 1:
            raise ValueError("short_min_trades 必须为正整数")
        if self.vol_target <= 0 or self.base_min_gross <= 0:
            raise ValueError("波动率目标和基础总敞口必须为正数")
        if self.base_min_gross > self.base_max_gross:
            raise ValueError("基础最小敞口不能高于基础最大敞口")
        if self.leveraged_max_gross < self.base_max_gross:
            raise ValueError("杠杆总敞口不能低于基础总敞口")
        if not self.base_max_gross <= self.breakout_min_gross <= self.leveraged_max_gross:
            raise ValueError("突破最小敞口必须位于基础和杠杆最大敞口之间")
        if not 0.0 <= self.short_gross <= self.leveraged_max_gross:
            raise ValueError("short_gross 必须位于 [0, leveraged_max_gross]")
        if min(self.shadow_cost_rate, self.shadow_borrow_rate_daily) < 0:
            raise ValueError("影子交易成本不能为负")

    def target_weights(
        self,
        data: MarketData,
        signal_index: int,
        previous: np.ndarray,
    ) -> np.ndarray:
        """根据截至 ``signal_index`` 的信息生成下一交易日目标权重。

        非调仓日保留已有核心池权重；调仓日先识别市场状态，再分别调用
        牛市轮动或熊市空头模块。中性状态、历史不足、做空边际不达标时
        均返回现金，从而保证所有风险暴露都有明确且可审计的触发条件。
        """

        core = self._core_indices(data)
        if signal_index % self.rebalance_days != 0:
            retained = np.asarray(previous, dtype=float).copy()
            non_core = np.ones(len(data.symbols), dtype=bool)
            non_core[core] = False
            retained[non_core] = 0.0
            return retained

        regime = self.market_regime(data, signal_index)
        if regime == "bull":
            return self._bull_weights(data, signal_index, core)
        if regime != "bear" or self.short_gross <= 0:
            return np.zeros(len(data.symbols), dtype=float)

        edge = self.short_edge_stats(data, signal_index)
        if not edge.approved:
            return np.zeros(len(data.symbols), dtype=float)
        return self._raw_short_weights(data, signal_index, core, self.short_gross)

    def market_regime(self, data: MarketData, signal_index: int) -> str:
        """以 BTC 快慢趋势、长动量和核心池广度识别牛、熊或中性状态。

        所有均线都包含且仅包含 ``signal_index`` 以前的数据。基础牛市
        需要 BTC 价格站上慢均线且长动量为正；横截面轮动另由独立方法做
        快均线和广度确认。熊市需要价格、快均线和长动量同时转弱，其余
        冲突状态归为中性并持有现金。
        """

        core = self._core_indices(data)
        btc = data.symbol_index("BTC-USDT")
        slow = trailing_mean(data.close, signal_index, self.slow_trend_window)
        fast = trailing_mean(data.close, signal_index, self.fast_trend_window)
        momentum = trailing_return(data.close, signal_index, self.momentum_window)
        own_trend = trailing_mean(data.close, signal_index, self.selection_trend_window)
        required = (slow[btc], fast[btc], momentum[btc], *own_trend[core])
        if not np.isfinite(required).all():
            return "neutral"

        bull = (
            data.close[signal_index, btc] > slow[btc]
            and momentum[btc] > 0
        )
        if bull:
            return "bull"
        bear = (
            data.close[signal_index, btc] < slow[btc]
            and fast[btc] < slow[btc]
            and momentum[btc] < 0
        )
        return "bear" if bear else "neutral"

    def rotation_confirmed(self, data: MarketData, signal_index: int) -> bool:
        """判断核心池是否具备从 BTC 核心仓切换到横截面轮动的条件。

        BTC 快均线必须高于慢均线，且至少 ``breadth_min`` 比例的核心资产
        站上自身趋势线。基础牛市但确认不足时仍持有 BTC，而不是完全离场，
        以降低宽度暂时收缩时错过 BTC 主升浪的风险。
        """

        core = self._core_indices(data)
        btc = data.symbol_index("BTC-USDT")
        slow = trailing_mean(data.close, signal_index, self.slow_trend_window)
        fast = trailing_mean(data.close, signal_index, self.fast_trend_window)
        own_trend = trailing_mean(data.close, signal_index, self.selection_trend_window)
        required = (slow[btc], fast[btc], *own_trend[core])
        if not np.isfinite(required).all():
            return False
        breadth = float(np.mean(data.close[signal_index, core] > own_trend[core]))
        return bool(fast[btc] > slow[btc] and breadth >= self.breadth_min)

    def is_key_breakout(self, data: MarketData, signal_index: int) -> bool:
        """判断 BTC 是否有效突破此前关键区间高点。

        关键位定义为不含当日的 ``breakout_window`` 日最高收盘价，避免
        把当日价格同时用于阈值和比较而弱化突破条件。调用方还会要求处于
        基础牛市并通过轮动确认，才会把组合总敞口放大至杠杆区间。
        """

        if signal_index < self.breakout_window:
            return False
        btc = data.symbol_index("BTC-USDT")
        prior_high = float(
            np.max(data.close[signal_index - self.breakout_window : signal_index, btc])
        )
        return bool(
            data.close[signal_index, btc] > prior_high * (1.0 + self.breakout_buffer)
        )

    def short_edge_stats(self, data: MarketData, signal_index: int) -> ShortEdgeStats:
        """回放最近窗口的影子空头并判断其净收益是否稳定。

        影子组合使用与真实策略相同的调仓相位、空头敞口、权重漂移和成本，
        并按调仓持有区间统计独立交易胜率。至少满足交易次数、累计净收益
        及胜率三个门槛才批准下一期真实小额空头，否则策略转为现金。回放
        不读取 ``signal_index`` 之后的数据，因此不会泄漏未来空头收益。
        """

        core = self._core_indices(data)
        start = max(1, signal_index - self.short_edge_window + 1)
        simulation_start = max(1, start - self.rebalance_days)
        previous = np.zeros(len(data.symbols), dtype=float)
        shadow_returns: list[float] = []
        trade_returns: list[float] = []
        current_trade_returns: list[float] | None = None
        active_days = 0
        for held_day in range(simulation_start, signal_index + 1):
            signal_day = held_day - 1
            is_rebalance = signal_day % self.rebalance_days == 0
            if is_rebalance and self.market_regime(data, signal_day) == "bear":
                target = self._raw_short_weights(
                    data,
                    signal_day,
                    core,
                    self.short_gross,
                )
            elif is_rebalance:
                target = np.zeros(len(data.symbols), dtype=float)
            else:
                target = previous.copy()
            target = self._cap_short_gross(target)
            turnover = float(np.sum(np.abs(target - previous)))
            asset_returns = data.close[held_day] / data.close[held_day - 1] - 1.0
            gross_return = float(target @ asset_returns)
            short_notional = float(np.sum(np.maximum(-target, 0.0)))
            net_return = (
                (1.0 + gross_return) * (1.0 - turnover * self.shadow_cost_rate)
                - 1.0
                - short_notional * self.shadow_borrow_rate_daily
            )
            if held_day >= start:
                if is_rebalance and current_trade_returns:
                    trade_returns.append(
                        float(np.prod(1.0 + np.asarray(current_trade_returns)) - 1.0)
                    )
                    current_trade_returns = None
                shadow_returns.append(max(net_return, -1.0))
                if short_notional > 0:
                    active_days += 1
                    if current_trade_returns is None:
                        current_trade_returns = []
                    current_trade_returns.append(net_return)
            previous = self._cap_short_gross(
                _drift_weights(target, asset_returns, gross_return)
            )
        if current_trade_returns:
            trade_returns.append(
                float(np.prod(1.0 + np.asarray(current_trade_returns)) - 1.0)
            )

        total_return = (
            float(np.prod(1.0 + np.asarray(shadow_returns, dtype=float)) - 1.0)
            if shadow_returns
            else 0.0
        )
        trades = len(trade_returns)
        win_rate = (
            float(np.mean(np.asarray(trade_returns, dtype=float) > 0))
            if trade_returns
            else 0.0
        )
        approved = (
            trades >= self.short_min_trades
            and total_return >= self.short_min_return
            and win_rate >= self.short_min_win_rate
        )
        return ShortEdgeStats(active_days, trades, total_return, win_rate, approved)

    def _cap_short_gross(self, weights: np.ndarray) -> np.ndarray:
        """把影子空头漂移权重裁剪回真实模块的名义空头上限。

        空头亏损会使权重绝对值被动放大；真实回测引擎会按敞口约束裁剪，
        影子模块必须执行同样处理，否则门控评价的是另一套风险更高的策略。
        """

        capped = np.asarray(weights, dtype=float).copy()
        short = float(np.sum(np.maximum(-capped, 0.0)))
        if short > self.short_gross and short > 0:
            capped[capped < 0] *= self.short_gross / short
        return capped

    def _core_indices(self, data: MarketData) -> np.ndarray:
        """解析固定核心池列索引，并在缺少任一标的时立即失败。

        显式失败比悄悄缩小币池更安全，因为后者会改变广度、排名和风险
        预算含义，导致不同数据集上的回测结果不可直接比较。
        """

        missing = [symbol for symbol in self.core_symbols if symbol not in data.symbols]
        if missing:
            raise ValueError(f"行情缺少核心池标的：{', '.join(missing)}")
        return np.asarray([data.symbol_index(symbol) for symbol in self.core_symbols], dtype=int)

    def _bull_weights(
        self,
        data: MarketData,
        signal_index: int,
        core: np.ndarray,
    ) -> np.ndarray:
        """在核心池内选择正趋势强动量标的并按逆波动率分配。

        长短动量按 65%/35% 合成，兼顾趋势持续性与近期加速度；资产还必须
        保持长动量为正且站上自身趋势线。突破关键位时才允许使用杠杆上限，
        其他牛市阶段即使波动率目标建议放大，也把总敞口限制在一倍以内。
        """

        if not self.rotation_confirmed(data, signal_index):
            btc_weights = np.zeros(len(data.symbols), dtype=float)
            btc_weights[data.symbol_index("BTC-USDT")] = 1.0
            return self._apply_bull_risk_budget(
                btc_weights,
                trailing_volatility(
                    data.close,
                    signal_index,
                    self.volatility_window,
                ),
                leveraged=False,
            )

        long_momentum = trailing_return(data.close, signal_index, self.momentum_window)
        fast_momentum = trailing_return(
            data.close,
            signal_index,
            self.fast_momentum_window,
        )
        own_trend = trailing_mean(data.close, signal_index, self.selection_trend_window)
        score = np.full(len(data.symbols), np.nan, dtype=float)
        eligible = (
            np.isfinite(long_momentum[core])
            & np.isfinite(fast_momentum[core])
            & np.isfinite(own_trend[core])
            & (long_momentum[core] > 0)
            & (data.close[signal_index, core] > own_trend[core])
        )
        eligible_core = core[eligible]
        score[eligible_core] = (
            0.65 * long_momentum[eligible_core] + 0.35 * fast_momentum[eligible_core]
        )
        selected = finite_top(score, self.top_k)
        if not len(selected):
            return np.zeros(len(data.symbols), dtype=float)

        volatility = trailing_volatility(
            data.close,
            signal_index,
            self.volatility_window,
        )
        weights = inverse_volatility_weights(
            volatility,
            selected,
            len(data.symbols),
            1.0,
        )
        return self._apply_bull_risk_budget(
            weights,
            volatility,
            leveraged=self.is_key_breakout(data, signal_index),
        )

    def _apply_bull_risk_budget(
        self,
        weights: np.ndarray,
        volatility: np.ndarray,
        leveraged: bool,
    ) -> np.ndarray:
        """应用波动率目标，并确保牛市和突破仓位具有预设进攻强度。

        波动率目标先控制极端风险，之后将名义总敞口限制在分档最小值与
        最大值之间。普通牛市默认保持一倍敞口；确认突破至少使用 1.2 倍，
        这样杠杆不再只是从未触及的理论上限，同时仍受多空引擎硬裁剪。
        """

        max_gross = self.leveraged_max_gross if leveraged else self.base_max_gross
        min_gross = self.breakout_min_gross if leveraged else self.base_min_gross
        scaled = _scale_to_vol_target(weights, volatility, self.vol_target, max_gross)
        gross = float(np.sum(np.abs(scaled)))
        if gross <= 1e-12:
            return scaled
        target_gross = min(max(gross, min_gross), max_gross)
        return scaled * (target_gross / gross)

    def _raw_short_weights(
        self,
        data: MarketData,
        signal_index: int,
        core: np.ndarray,
        gross: float,
    ) -> np.ndarray:
        """生成未经影子收益门控的弱势核心资产空头权重。

        候选必须同时满足长动量低于阈值并跌破自身趋势线，再从中选择动量
        最弱的 ``short_top_k`` 个标的等权做空。没有足够弱势候选时返回
        现金，避免为了保持空头仓位而交易没有显著下行趋势的资产。
        """

        weights = np.zeros(len(data.symbols), dtype=float)
        if gross <= 0:
            return weights
        momentum = trailing_return(data.close, signal_index, self.momentum_window)
        own_trend = trailing_mean(data.close, signal_index, self.selection_trend_window)
        score = np.full(len(data.symbols), np.nan, dtype=float)
        eligible = (
            np.isfinite(momentum[core])
            & np.isfinite(own_trend[core])
            & (momentum[core] <= self.short_momentum_threshold)
            & (data.close[signal_index, core] < own_trend[core])
        )
        eligible_core = core[eligible]
        score[eligible_core] = momentum[eligible_core]
        selected = finite_top(score, self.short_top_k, descending=False)
        if len(selected):
            weights[selected] = -gross / len(selected)
        return weights
