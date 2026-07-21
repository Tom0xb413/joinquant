"""TOP5 核心资产的牛熊分档轮动、条件杠杆与自适应做空策略。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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

    observations: int
    total_return: float
    win_rate: float
    approved: bool


@dataclass
class CoreTop5RegimeRotation:
    """在固定 TOP5 核心池内执行牛熊分档轮动。

    牛市必须同时满足 BTC 长趋势、快慢均线、正动量和核心池广度确认；
    组合仅持有动量最强的少数核心资产，并只在 BTC 突破此前关键高点时
    将波动率目标允许的总敞口放大到杠杆上限。熊市只考虑跌破趋势且负
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
    base_max_gross: float = 1.0
    breakout_window: int = 55
    breakout_buffer: float = 0.0
    leveraged_max_gross: float = 1.5
    short_top_k: int = 1
    short_gross: float = 0.30
    short_momentum_threshold: float = -0.12
    short_edge_window: int = 90
    short_min_observations: int = 12
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
        if self.short_min_observations < 1:
            raise ValueError("short_min_observations 必须为正整数")
        if self.vol_target <= 0 or self.base_max_gross <= 0:
            raise ValueError("波动率目标和基础总敞口必须为正数")
        if self.leveraged_max_gross < self.base_max_gross:
            raise ValueError("杠杆总敞口不能低于基础总敞口")
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

        所有均线都包含且仅包含 ``signal_index`` 以前的数据。牛市需要
        四项确认同时成立；熊市需要 BTC 价格、快均线和长动量同时转弱。
        其余冲突状态归为中性并持有现金，以减少震荡区间的反复交易。
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

        breadth = float(np.mean(data.close[signal_index, core] > own_trend[core]))
        bull = (
            data.close[signal_index, btc] > slow[btc]
            and fast[btc] > slow[btc]
            and momentum[btc] > 0
            and breadth >= self.breadth_min
        )
        if bull:
            return "bull"
        bear = (
            data.close[signal_index, btc] < slow[btc]
            and fast[btc] < slow[btc]
            and momentum[btc] < 0
        )
        return "bear" if bear else "neutral"

    def is_key_breakout(self, data: MarketData, signal_index: int) -> bool:
        """判断 BTC 是否有效突破此前关键区间高点。

        关键位定义为不含当日的 ``breakout_window`` 日最高收盘价，避免
        把当日价格同时用于阈值和比较而弱化突破条件。只有牛市状态且核心
        池广度达到门槛时，这个信号才会放宽组合总敞口至杠杆上限。
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

        对窗口内每个持有日，仅用前一日信号生成一倍名义空头目标，再扣除
        换手成本和每日借券成本。至少满足有效持仓天数、累计净收益及胜率
        三个门槛才批准下一期真实小额空头，否则策略转为现金。该回放不读取
        ``signal_index`` 之后的数据，因此不会把未来空头收益泄漏进决策。
        """

        core = self._core_indices(data)
        start = max(1, signal_index - self.short_edge_window + 1)
        previous = np.zeros(len(data.symbols), dtype=float)
        shadow_returns: list[float] = []
        active_returns: list[float] = []
        for held_day in range(start, signal_index + 1):
            signal_day = held_day - 1
            if self.market_regime(data, signal_day) == "bear":
                target = self._raw_short_weights(data, signal_day, core, 1.0)
            else:
                target = np.zeros(len(data.symbols), dtype=float)
            turnover = float(np.sum(np.abs(target - previous)))
            asset_returns = data.close[held_day] / data.close[held_day - 1] - 1.0
            gross_return = float(target @ asset_returns)
            short_notional = float(np.sum(np.maximum(-target, 0.0)))
            net_return = (
                (1.0 + gross_return) * (1.0 - turnover * self.shadow_cost_rate)
                - 1.0
                - short_notional * self.shadow_borrow_rate_daily
            )
            shadow_returns.append(max(net_return, -1.0))
            if short_notional > 0:
                active_returns.append(net_return)
            previous = target

        total_return = (
            float(np.prod(1.0 + np.asarray(shadow_returns, dtype=float)) - 1.0)
            if shadow_returns
            else 0.0
        )
        observations = len(active_returns)
        win_rate = (
            float(np.mean(np.asarray(active_returns, dtype=float) > 0))
            if active_returns
            else 0.0
        )
        approved = (
            observations >= self.short_min_observations
            and total_return >= self.short_min_return
            and win_rate >= self.short_min_win_rate
        )
        return ShortEdgeStats(observations, total_return, win_rate, approved)

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
        max_gross = (
            self.leveraged_max_gross
            if self.is_key_breakout(data, signal_index)
            else self.base_max_gross
        )
        return _scale_to_vol_target(weights, volatility, self.vol_target, max_gross)

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
