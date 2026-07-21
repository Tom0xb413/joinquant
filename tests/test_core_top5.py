"""TOP5 核心池轮动、杠杆门控和自适应做空测试。"""

from __future__ import annotations

import unittest
from datetime import date, timedelta

import numpy as np

from crypto_lab.backtest import BacktestResult
from crypto_lab.core_top5 import CORE_TOP5_SYMBOLS, CoreTop5RegimeRotation
from crypto_lab.core_top5_research import _regime_metrics
from crypto_lab.data import MarketData
from crypto_lab.long_short import LongShortLimits, run_long_short_backtest


def sample_bull_market(days: int = 520) -> MarketData:
    """构造含一个池外强势币的长期牛市，验证核心池边界和关键位突破。

    各核心资产使用不同趋势和周期扰动以产生有限波动率；池外 ADA 刻意
    设置为最强趋势，若策略错误地在全市场排名，它会被选中并使测试失败。
    """

    dates = tuple(date(2022, 1, 1) + timedelta(days=index) for index in range(days))
    time = np.arange(days, dtype=float)
    symbols = (*CORE_TOP5_SYMBOLS, "ADA-USDT")
    slopes = np.array([0.0015, 0.0018, 0.0022, 0.0012, 0.0016, 0.0040])
    phases = np.arange(len(symbols), dtype=float)
    close = np.column_stack(
        [
            100.0
            * np.exp(
                slopes[column] * time
                + 0.018 * np.sin(time / (8.0 + column) + phases[column])
            )
            for column in range(len(symbols))
        ]
    )
    close[-20:, 0] *= np.linspace(1.0, 1.08, 20)
    volume = np.full_like(close, 1_000_000.0)
    return MarketData(
        dates,
        symbols,
        close.copy(),
        close.copy(),
        close.copy(),
        close,
        volume,
    )


def sample_bear_market(days: int = 520) -> MarketData:
    """构造持续下跌市场，使扣费后的影子空头拥有明确正收益。

    不加入反弹噪声是为了隔离做空门控本身，确保有效观察数、累计净收益
    与胜率三个审批条件都可被确定性验证。
    """

    dates = tuple(date(2022, 1, 1) + timedelta(days=index) for index in range(days))
    time = np.arange(days, dtype=float)
    symbols = (*CORE_TOP5_SYMBOLS, "ADA-USDT")
    slopes = np.array([-0.0015, -0.0018, -0.0028, -0.0012, -0.0020, -0.0040])
    close = np.column_stack(
        [100.0 * np.exp(slopes[column] * time) for column in range(len(symbols))]
    )
    volume = np.full_like(close, 1_000_000.0)
    return MarketData(
        dates,
        symbols,
        close.copy(),
        close.copy(),
        close.copy(),
        close,
        volume,
    )


def compact_strategy(**overrides) -> CoreTop5RegimeRotation:
    """创建短窗口测试策略，同时保留生产规则之间的相对约束。

    缩短窗口能减少合成数据长度，却仍覆盖快慢趋势、双动量、突破和影子
    做空逻辑；调用方只覆盖与单个断言直接相关的参数。
    """

    params = {
        "slow_trend_window": 100,
        "fast_trend_window": 20,
        "momentum_window": 60,
        "fast_momentum_window": 20,
        "selection_trend_window": 30,
        "volatility_window": 20,
        "rebalance_days": 1,
        "breakout_window": 30,
        "short_edge_window": 60,
    }
    params.update(overrides)
    return CoreTop5RegimeRotation(**params)


class CoreTop5Tests(unittest.TestCase):
    """验证 TOP5 策略最关键且容易失效的风险不变量。

    @author Cursor
    @since 0.2.0
    """

    def test_bull_rotation_never_uses_non_core_asset(self):
        """即使池外资产动量最强，牛市权重也必须严格限制在五币核心池。"""

        data = sample_bull_market()
        weights = compact_strategy(vol_target=2.0).target_weights(
            data,
            len(data.dates) - 1,
            np.zeros(len(data.symbols)),
        )
        self.assertGreater(float(np.sum(weights)), 0.0)
        self.assertAlmostEqual(float(weights[-1]), 0.0)
        self.assertLessEqual(int(np.sum(weights > 0)), 2)

    def test_key_breakout_allows_but_caps_leverage(self):
        """关键突破可把总敞口放大到 1.5 倍，但绝不能越过硬上限。"""

        data = sample_bull_market()
        strategy = compact_strategy(
            vol_target=2.0,
            base_max_gross=1.0,
            leveraged_max_gross=1.5,
        )
        index = len(data.dates) - 1
        self.assertTrue(strategy.is_key_breakout(data, index))
        weights = strategy.target_weights(data, index, np.zeros(len(data.symbols)))
        self.assertAlmostEqual(float(np.sum(np.abs(weights))), 1.5, places=8)

    def test_non_breakout_bull_stays_unlevered(self):
        """未突破关键位时，即使波动率目标很高也只能使用一倍敞口。"""

        data = sample_bull_market()
        strategy = compact_strategy(
            vol_target=2.0,
            breakout_buffer=0.50,
            base_max_gross=1.0,
            leveraged_max_gross=1.5,
        )
        index = len(data.dates) - 1
        self.assertFalse(strategy.is_key_breakout(data, index))
        weights = strategy.target_weights(data, index, np.zeros(len(data.symbols)))
        self.assertAlmostEqual(float(np.sum(np.abs(weights))), 1.0, places=8)

    def test_bear_market_falls_back_to_cash_without_short_evidence(self):
        """影子空头连续持仓 episode 不足时，确认熊市也必须返回全现金。"""

        data = sample_bear_market()
        strategy = compact_strategy(
            short_momentum_threshold=-0.01,
            short_min_episodes=100,
        )
        weights = strategy.target_weights(
            data,
            len(data.dates) - 1,
            np.zeros(len(data.symbols)),
        )
        np.testing.assert_allclose(weights, np.zeros(len(data.symbols)))

    def test_profitable_shadow_short_enables_small_core_short(self):
        """持续熊市中影子收益达标后，只允许小额做空最弱核心标的。"""

        data = sample_bear_market()
        strategy = compact_strategy(
            short_gross=0.30,
            rebalance_days=7,
            short_momentum_threshold=-0.01,
            short_min_episodes=1,
            short_min_return=0.0,
            short_min_win_rate=0.50,
        )
        index = (len(data.dates) - 1) // strategy.rebalance_days * strategy.rebalance_days
        edge = strategy.short_edge_stats(data, index)
        self.assertTrue(edge.approved)
        weights = strategy.target_weights(data, index, np.zeros(len(data.symbols)))
        self.assertAlmostEqual(float(np.sum(np.maximum(-weights, 0.0))), 0.30)
        self.assertAlmostEqual(float(weights[-1]), 0.0)
        self.assertLess(float(weights.sum()), 0.0)

    def test_future_prices_cannot_change_short_edge_approval(self):
        """修改评价日之后价格不得改变影子空头统计或审批结论。"""

        data = sample_bear_market()
        strategy = compact_strategy(
            rebalance_days=7,
            short_momentum_threshold=-0.01,
            short_min_episodes=1,
            short_min_return=0.0,
        )
        index = 420
        before = strategy.short_edge_stats(data, index)
        changed = sample_bear_market()
        changed.close[index + 1 :] *= 0.05
        changed.high[index + 1 :] *= 0.05
        changed.low[index + 1 :] *= 0.05
        changed.open[index + 1 :] *= 0.05
        after = strategy.short_edge_stats(changed, index)
        self.assertEqual(before, after)

    def test_shadow_short_return_matches_real_engine_path(self):
        """影子窗口净收益必须逐日匹配相同规则在真实多空引擎中的结果。"""

        data = sample_bear_market()
        strategy = compact_strategy(
            rebalance_days=7,
            short_momentum_threshold=-0.01,
            short_min_episodes=1,
            short_min_return=-1.0,
        )

        class RawShortStrategy:
            """绕过收益门控，仅执行影子模块的原始空头目标。

            @author Cursor
            @since 0.2.0
            """

            name = "raw_short_path"

            def target_weights(self, market, signal_index, previous):
                """按生产调仓相位生成原始空头，用于与影子回放逐日对照。"""

                if signal_index % strategy.rebalance_days != 0:
                    return previous
                if strategy.market_regime(market, signal_index) != "bear":
                    return np.zeros(len(market.symbols))
                return strategy._raw_short_weights(
                    market,
                    signal_index,
                    strategy._core_indices(market),
                    strategy.short_gross,
                )

        result = run_long_short_backtest(
            data,
            RawShortStrategy(),
            fee_rate=0.001,
            slippage_rate=0.0005,
            limits=LongShortLimits(
                max_gross_exposure=1.5,
                max_net_exposure=1.5,
                max_short_exposure=0.30,
                borrow_rate_daily=0.00005,
            ),
        )
        index = 420
        start = index - strategy.short_edge_window + 1
        expected = float(np.prod(1.0 + result.daily_returns[start : index + 1]) - 1.0)
        actual = strategy.short_edge_stats(data, index).total_return
        self.assertAlmostEqual(actual, expected, places=12)

    def test_future_prices_cannot_change_current_signal(self):
        """修改信号日之后的所有价格不得改变当前目标权重，防止未来泄漏。"""

        data = sample_bull_market()
        strategy = compact_strategy(vol_target=0.4)
        index = 420
        before = strategy.target_weights(data, index, np.zeros(len(data.symbols)))
        changed = sample_bull_market()
        changed.close[index + 1 :] *= 7.0
        changed.high[index + 1 :] *= 7.0
        changed.low[index + 1 :] *= 7.0
        changed.open[index + 1 :] *= 7.0
        after = strategy.target_weights(changed, index, np.zeros(len(data.symbols)))
        np.testing.assert_allclose(before, after)

    def test_missing_core_symbol_fails_explicitly(self):
        """数据缺少任一固定核心标的时必须明确失败，不能静默缩小币池。"""

        full = sample_bull_market()
        data = MarketData(
            full.dates,
            full.symbols[:-2],
            full.open[:, :-2],
            full.high[:, :-2],
            full.low[:, :-2],
            full.close[:, :-2],
            full.volume_quote[:, :-2],
        )
        with self.assertRaisesRegex(ValueError, "行情缺少核心池标的"):
            compact_strategy().target_weights(
                data,
                len(data.dates) - 1,
                np.zeros(len(data.symbols)),
            )

    def test_regime_cagr_keeps_non_state_calendar_days(self):
        """状态贡献 CAGR 必须按完整日历年化，不能压缩掉非状态现金日。"""

        class AlternatingRegime:
            """提供确定性交替状态，隔离研究指标的日历口径。

            @author Cursor
            @since 0.2.0
            """

            def market_regime(self, data, signal_index):
                """偶数信号日返回 bull，奇数日返回 neutral。"""

                return "bull" if signal_index % 2 == 0 else "neutral"

        data = sample_bull_market()
        daily_returns = np.full(len(data.dates), 0.01)
        result = BacktestResult(
            strategy="calendar_metric",
            dates=data.dates,
            daily_returns=daily_returns,
            equity=np.cumprod(1.0 + daily_returns),
            weights=np.zeros_like(data.close),
            turnover=np.zeros(len(data.dates)),
            costs=np.zeros(len(data.dates)),
        )
        metrics = _regime_metrics(
            result,
            AlternatingRegime(),
            data,
            400,
            499,
            "bull",
        )
        expected_total = 1.01**50 - 1.0
        expected_cagr = (1.0 + expected_total) ** (365.0 / 100.0) - 1.0
        self.assertEqual(metrics["observations"], 100)
        self.assertEqual(metrics["state_observations"], 50)
        self.assertAlmostEqual(metrics["cagr"], expected_cagr)


if __name__ == "__main__":
    unittest.main()
