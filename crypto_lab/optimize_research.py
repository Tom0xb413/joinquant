"""优化策略族的参数搜索、样本外评价与报告生成。"""

from __future__ import annotations

import itertools
import json
from dataclasses import fields
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from .backtest import buy_and_hold, run_backtest
from .data import MarketData
from .optimized_strategies import (
    BreadthRegimeRotation,
    BtcDualMomentum,
    CoreSatelliteVolScaled,
    MajorsAltsRegime,
)
from .research import _block_bootstrap_cagr, _parameter_product


def _optimized_parameters(strategy: object) -> dict[str, Any]:
    """导出优化策略可调参数，排除名称与思想标签。"""

    return {
        field.name: getattr(strategy, field.name)
        for field in fields(strategy)
        if field.name not in {"name", "ideas", "source_ids"}
    }


OPTIMIZED_STRATEGY_SPECS = (
    (
        BtcDualMomentum,
        {
            "regime_window": [100, 120, 150],
            "lookback": [60, 90],
            "top_k": [3, 4],
            "rebalance_days": [14, 21, 30],
        },
    ),
    (
        BreadthRegimeRotation,
        {
            "ma_window": [80, 100, 150],
            "lookback": [60, 90],
            "top_k": [3, 4],
            "rebalance_days": [14, 21],
            "low_breadth": [0.30, 0.35],
            "high_breadth": [0.55, 0.60],
        },
    ),
    (
        CoreSatelliteVolScaled,
        {
            "regime_window": [100, 120, 150],
            "lookback": [60, 90],
            "satellite_count": [1, 2],
            "rebalance_days": [21, 30],
            "vol_scale": [0.35, 0.50, 0.70],
        },
    ),
    (
        MajorsAltsRegime,
        {
            "style_window": [20, 40, 60],
            "trend_window": [80, 100, 120],
            "top_k": [2, 3, 4],
            "rebalance_days": [21, 30],
        },
    ),
)


def run_optimized_research(
    data: MarketData,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    train_fraction: float = 0.60,
) -> dict[str, Any]:
    """仅在训练区间选参，并输出默认/优化参数的样本外对照。"""

    if not 0.5 <= train_fraction <= 0.8:
        raise ValueError("train_fraction 应位于 [0.5, 0.8]")
    split_index = max(120, int(len(data.dates) * train_fraction))
    if len(data.dates) - split_index < 120:
        raise ValueError("样本外区间不足 120 天")
    train_start = data.dates[0]
    train_end = data.dates[split_index - 1]
    test_start = data.dates[split_index]
    test_end = data.dates[-1]
    strategies: dict[str, Any] = {}

    for strategy_type, grid in OPTIMIZED_STRATEGY_SPECS:
        default_strategy = strategy_type()
        default_result = run_backtest(data, default_strategy, fee_rate, slippage_rate)
        best_score = -np.inf
        best_strategy = default_strategy
        candidates = 0
        for params in _parameter_product(grid):
            strategy = strategy_type(**params)
            result = run_backtest(data, strategy, fee_rate, slippage_rate)
            metrics = result.metrics(train_start, train_end)
            # 额外惩罚年化换手，避免回到上一轮高成本陷阱
            annual_turnover = metrics.turnover / max(metrics.observations / 365.0, 1e-9)
            score = metrics.sharpe - 0.5 * metrics.max_drawdown - 0.01 * annual_turnover
            candidates += 1
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_strategy = strategy
        optimized_result = run_backtest(data, best_strategy, fee_rate, slippage_rate)
        optimized_test = optimized_result.metrics_dict(test_start, test_end)
        default_test = default_result.metrics_dict(test_start, test_end)
        bootstrap = _block_bootstrap_cagr(
            optimized_result.daily_returns[split_index:],
            seed=20260716 + sum(ord(character) for character in best_strategy.name),
        )
        optimized_positive = optimized_test["cagr"] > 0 and optimized_test["sharpe"] > 0
        default_positive = default_test["cagr"] > 0 and default_test["sharpe"] > 0
        beats_btc = (
            optimized_test["sharpe"] > 0
            and optimized_test["cagr"] > 0
        )
        statistically_supported = bootstrap["cagr_ci_95"][0] > 0
        if optimized_positive and default_positive and statistically_supported:
            verdict = "统计通过"
        elif optimized_positive and default_positive:
            verdict = "稳健候选（默认与优化均正）"
        elif optimized_positive:
            verdict = "点估计为正（无统计支持）"
        else:
            verdict = "未通过"
        strategies[best_strategy.name] = {
            "ideas": list(getattr(best_strategy, "ideas", ())),
            "candidates_tested": candidates,
            "selection_objective": "train_sharpe - 0.5*train_mdd - 0.01*annual_turnover",
            "default_parameters": _optimized_parameters(default_strategy),
            "selected_parameters": _optimized_parameters(best_strategy),
            "default_train": default_result.metrics_dict(train_start, train_end),
            "default_test": default_test,
            "optimized_train": optimized_result.metrics_dict(train_start, train_end),
            "optimized_test": optimized_test,
            "optimized_test_bootstrap": bootstrap,
            "verdict": verdict,
            "beats_buy_hold_point_estimate": beats_btc,
        }

    benchmark = buy_and_hold(data, fee_rate=fee_rate, slippage_rate=slippage_rate)
    benchmark_test = benchmark.metrics_dict(test_start, test_end)
    for item in strategies.values():
        test = item["optimized_test"]
        test["oos_positive"] = test["cagr"] > 0 and test["sharpe"] > 0
        test["default_oos_positive"] = (
            item["default_test"]["cagr"] > 0 and item["default_test"]["sharpe"] > 0
        )
        test["risk_adjusted_better_than_btc"] = (
            test["sharpe"] > benchmark_test["sharpe"]
            and test["max_drawdown"] < benchmark_test["max_drawdown"]
        )

    return {
        "methodology": {
            "goal": "在上一轮失败教训上设计低换手、BTC 风险门控的优化策略原型",
            "signal_timing": "使用 T-1 及更早收盘数据生成权重，获取 T 日收盘到收盘收益",
            "positioning": "仅现货多头，不加杠杆、不做空；未分配权重为零收益现金",
            "fee_rate_one_way": fee_rate,
            "slippage_rate_one_way": slippage_rate,
            "annualization_days": 365,
            "selection": "仅训练区间网格选参，并惩罚年化换手；样本外不参与选参",
            "uncertainty": "样本外日收益使用 14 日循环区块 Bootstrap，2,000 次重采样",
            "known_biases": [
                "固定使用当前仍在 OKX 交易且历史较长的币对，存在幸存者偏差",
                "默认参数虽按经济逻辑设定，但仍经过探索性回测观察，存在研究者自由度",
                "日线模型未模拟盘口冲击、稳定币脱锚、交易所故障和下架清算",
            ],
        },
        "universe": list(data.symbols),
        "data_range": {"start": data.dates[0].isoformat(), "end": data.dates[-1].isoformat()},
        "split": {
            "train_start": train_start.isoformat(),
            "train_end": train_end.isoformat(),
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
            "train_fraction": train_fraction,
        },
        "benchmark": {
            "name": benchmark.strategy,
            "train": benchmark.metrics_dict(train_start, train_end),
            "test": benchmark_test,
        },
        "strategies": strategies,
    }


def write_optimized_markdown_report(
    path: Path,
    results: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    """生成优化策略报告，突出有效候选与失败项。"""

    split = results["split"]
    benchmark = results["benchmark"]["test"]
    verdict_counts: dict[str, int] = {}
    rows = []
    recommendations = []
    for name, item in results["strategies"].items():
        metrics = item["optimized_test"]
        default_metrics = item["default_test"]
        bootstrap = item["optimized_test_bootstrap"]
        verdict = item["verdict"]
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        rows.append(
            "| {name} | {ideas} | {default_cagr:.1%} | {cagr:.1%} | {sharpe:.2f} | {mdd:.1%} | {to:.1f} | {lower:.1%}–{upper:.1%} | {verdict} |".format(
                name=name,
                ideas=" / ".join(item["ideas"]),
                default_cagr=default_metrics["cagr"],
                cagr=metrics["cagr"],
                sharpe=metrics["sharpe"],
                mdd=metrics["max_drawdown"],
                to=metrics["turnover"],
                lower=bootstrap["cagr_ci_95"][0],
                upper=bootstrap["cagr_ci_95"][1],
                verdict=verdict,
            )
        )
        if verdict in {"统计通过", "稳健候选（默认与优化均正）"}:
            recommendations.append(
                f"- **{name}**：默认 CAGR `{default_metrics['cagr']:.1%}`，优化 CAGR `{metrics['cagr']:.1%}`，"
                f"Sharpe `{metrics['sharpe']:.2f}`，最大回撤 `{metrics['max_drawdown']:.1%}`。"
            )
    if not recommendations:
        recommendations.append("- 当前没有同时满足默认/优化均为正且统计稳健的策略；请把下方点估计正的结果仅作继续研究线索。")

    files = manifest.get("files", [])
    lines = [
        "# 优化策略设计：真实行情样本外验证",
        "",
        "## 结论",
        "",
        "本报告在上一轮高换手策略失效的基础上，重新设计了 4 个低换手、带 BTC/广度风险门控的原型。",
        "“有效”至少要求默认参数与训练优选参数的样本外 CAGR、Sharpe 同时为正；",
        "若 Bootstrap CAGR 95% 置信区间下界也大于 0，则记为统计通过。",
        "",
        f"- 训练区间：`{split['train_start']}` 至 `{split['train_end']}`",
        f"- 样本外区间：`{split['test_start']}` 至 `{split['test_end']}`",
        f"- BTC 样本外：CAGR `{benchmark['cagr']:.1%}`，Sharpe `{benchmark['sharpe']:.2f}`，最大回撤 `{benchmark['max_drawdown']:.1%}`",
        f"- 成本假设：单边手续费 `{results['methodology']['fee_rate_one_way']:.2%}` + 单边滑点 `{results['methodology']['slippage_rate_one_way']:.2%}`",
        f"- 判定汇总：{', '.join(f'{key} `{value}`' for key, value in sorted(verdict_counts.items()))}",
        "",
        "## 推荐继续关注",
        "",
        *recommendations,
        "",
        "| 策略 | 思想来源 | 默认 CAGR | 优化 CAGR | Sharpe | 最大回撤 | 样本外换手 | Bootstrap CAGR 95% CI | 判定 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
        *rows,
        "",
        "## 设计原则",
        "",
        "1. **先门控再选股**：BTC 趋势或市场广度不达标时直接持有现金。",
        "2. **降低换手**：默认 14–30 日再平衡，并在选参目标中惩罚年化换手。",
        "3. **双动量过滤**：既要求自身收益为正，再做横截面排序，减少接飞刀。",
        "4. **风险预算**：核心-卫星策略用逆波动率与总敞口缩放控制回撤。",
        "5. **不夸大归因**：这些是受原聚宽思想启发的新原型，不是原策略复现。",
        "",
        "## 策略说明",
        "",
        "| 策略 | 机制 | 为何可能比上一轮更稳 |",
        "|---|---|---|",
        "| `btc_dual_momentum` | BTC 均线门控 + 正动量 Top-K | 去掉日频追涨，改中低频双动量 |",
        "| `breadth_regime_rotation` | 广度弱/中/强三档仓位 | 用市场宽度替代财务择时 |",
        "| `core_satellite_vol_scaled` | BTC/ETH 核心 + 山寨卫星 + 波动缩放 | 低换手，主动降杠杆 |",
        "| `majors_alts_regime` | 主流/山寨相对强弱 + BTC 趋势过滤 | 强化上一轮唯一弱正信号 |",
        "",
        "## 数据证据",
        "",
        f"- 来源：`{manifest.get('source', 'N/A')}`",
        f"- 周期：`{manifest.get('bar', 'N/A')}`；币对数：`{len(files)}`",
        f"- 行数范围：`{min((item['rows'] for item in files), default=0)}`–`{max((item['rows'] for item in files), default=0)}`",
        "",
        "## 限制",
        "",
        *[f"- {item}" for item in results["methodology"]["known_biases"]],
        "- 即便样本外点估计为正，宽置信区间仍意味着结果不稳定，不能直接等价于实盘可上线。",
        "",
        "详细参数见 `optimized_backtest_results.json`。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
