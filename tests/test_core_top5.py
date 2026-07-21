"""TOP5 核心池轮动、杠杆门控和自适应做空测试。"""

from __future__ import annotations

import unittest
from datetime import date, timedelta

import numpy as np

from crypto_lab.core_top5 import CORE_TOP5_SYMBOLS, CoreTop5RegimeRotation
from crypto_lab.data import MarketData


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
        """影子空头有效天数不足时，确认熊市也必须返回全现金。"""

        data = sample_bear_market()
        strategy = compact_strategy(
            short_momentum_threshold=-0.01,
            short_min_observations=100,
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
            short_momentum_threshold=-0.01,
            short_min_observations=10,
            short_min_return=0.0,
            short_min_win_rate=0.50,
        )
        index = len(data.dates) - 1
        edge = strategy.short_edge_stats(data, index)
        self.assertTrue(edge.approved)
        weights = strategy.target_weights(data, index, np.zeros(len(data.symbols)))
        self.assertAlmostEqual(float(np.sum(np.maximum(-weights, 0.0))), 0.30)
        self.assertAlmostEqual(float(weights[-1]), 0.0)
        self.assertLess(float(weights.sum()), 0.0)

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


if __name__ == "__main__":
    unittest.main()
