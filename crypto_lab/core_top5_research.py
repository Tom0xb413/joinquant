"""TOP5 激进轮动策略的训练期选型、做空降级与开发后验证报告。"""

from __future__ import annotations

from dataclasses import asdict, fields, replace
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from .backtest import BacktestResult, buy_and_hold
from .core_top5 import CORE_TOP5_SYMBOLS, CoreTop5RegimeRotation
from .data import MarketData
from .long_short import LongShortLimits, run_long_short_backtest


LONG_CONFIGS: tuple[dict[str, Any], ...] = tuple(
    {
        "top_k": top_k,
        "rebalance_days": rebalance_days,
        "vol_target": vol_target,
        "leveraged_max_gross": leveraged_max_gross,
    }
    for top_k in (1, 2)
    for rebalance_days in (7, 14)
    for vol_target in (0.45, 0.55)
    for leveraged_max_gross in (1.3, 1.5)
)

DEFAULT_TRAIN_END = date(2024, 4, 27)
VALIDATION_MACRO_BULL_END = date(2025, 10, 6)


def run_core_top5_research(
    data: MarketData,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    train_end: date = DEFAULT_TRAIN_END,
) -> dict[str, Any]:
    """仅用固定训练段选择参数和做空开关，再评价开发后验证集。

    第一阶段比较有限的持仓数、调仓频率、波动率目标和杠杆上限，以两个
    训练折中较弱的因果 risk-on 日历贡献超额 CAGR 为主要目标。第二阶段
    固定多头参数，对比自适应做空与熊市现金；只有做空在两个训练折及完整
    训练期均改善熊市收益，且未明显恶化全局夏普和回撤时才启用。固定
    固定截止日之后的结果不参与程序化选择，但开发过程已查看该区间，因此
    最终报告将其诚实标记为开发后验证，而不是未触碰 forward OOS。
    """

    _validate_research_inputs(data, train_end)
    train_indices = [index for index, day in enumerate(data.dates) if day <= train_end]
    split_index = train_indices[-1] + 1
    if len(data.dates) - split_index < 180:
        raise ValueError("开发后验证区间至少需要 180 天")
    warmup = 220
    mid_index = warmup + (split_index - warmup) // 2
    folds = ((warmup, mid_index - 1), (mid_index, split_index - 1))
    limits = LongShortLimits(
        max_gross_exposure=1.5,
        max_net_exposure=1.5,
        max_short_exposure=0.30,
    )
    benchmark = buy_and_hold(data, fee_rate=fee_rate, slippage_rate=slippage_rate)

    ranked: list[
        tuple[float, bool, CoreTop5RegimeRotation, BacktestResult, dict[str, Any]]
    ] = []
    for config in LONG_CONFIGS:
        strategy = CoreTop5RegimeRotation(
            short_gross=0.0,
            shadow_cost_rate=fee_rate + slippage_rate,
            shadow_borrow_rate_daily=limits.borrow_rate_daily,
            **config,
        )
        result = run_long_short_backtest(
            data,
            strategy,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            limits=limits,
        )
        fold_details = []
        risk_on_excesses = []
        fold_sharpes = []
        fold_drawdowns = []
        for start, end in folds:
            strategy_risk_on = _regime_metrics(result, strategy, data, start, end, "bull")
            benchmark_risk_on = _regime_metrics(benchmark, strategy, data, start, end, "bull")
            overall = result.metrics(data.dates[start], data.dates[end])
            excess = strategy_risk_on["cagr"] - benchmark_risk_on["cagr"]
            risk_on_excesses.append(excess)
            fold_sharpes.append(overall.sharpe)
            fold_drawdowns.append(overall.max_drawdown)
            fold_details.append(
                {
                    "start": data.dates[start].isoformat(),
                    "end": data.dates[end].isoformat(),
                    "risk_on_strategy": strategy_risk_on,
                    "risk_on_benchmark": benchmark_risk_on,
                    "risk_on_excess_cagr": excess,
                    "overall": asdict(overall),
                }
            )
        qualified = min(risk_on_excesses) > 0 and min(fold_sharpes) > 0
        score = (
            min(risk_on_excesses)
            + 0.40 * float(np.mean(risk_on_excesses))
            + 0.15 * min(fold_sharpes)
            - 0.25 * max(fold_drawdowns)
        )
        ranked.append(
            (
                score,
                qualified,
                strategy,
                result,
                {"folds": fold_details, "score": score, "qualified": qualified},
            )
        )

    eligible = [item for item in ranked if item[1]] or ranked
    eligible.sort(key=lambda item: item[0], reverse=True)
    _, long_qualified, cash_strategy, cash_result, selected_training = eligible[0]
    short_strategy = replace(cash_strategy, short_gross=0.30)
    short_result = run_long_short_backtest(
        data,
        short_strategy,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        limits=limits,
    )

    short_decision = _select_short_module(
        data,
        cash_strategy,
        cash_result,
        short_strategy,
        short_result,
        folds,
        warmup,
        split_index - 1,
    )
    if short_decision["enabled"]:
        recommended_strategy = short_strategy
        recommended_result = short_result
    else:
        recommended_strategy = cash_strategy
        recommended_result = cash_result

    train_start = data.dates[warmup]
    locked_train_end = data.dates[split_index - 1]
    test_start = data.dates[split_index]
    test_end = data.dates[-1]
    recommended_test = recommended_result.metrics_dict(test_start, test_end)
    benchmark_test = benchmark.metrics_dict(test_start, test_end)
    recommended_risk_on_test = _regime_metrics(
        recommended_result,
        recommended_strategy,
        data,
        split_index,
        len(data.dates) - 1,
        "bull",
    )
    benchmark_risk_on_test = _regime_metrics(
        benchmark,
        recommended_strategy,
        data,
        split_index,
        len(data.dates) - 1,
        "bull",
    )
    recommended_bear_test = _regime_metrics(
        recommended_result,
        recommended_strategy,
        data,
        split_index,
        len(data.dates) - 1,
        "bear",
    )
    cash_bear_test = _regime_metrics(
        cash_result,
        cash_strategy,
        data,
        split_index,
        len(data.dates) - 1,
        "bear",
    )
    short_bear_test = _regime_metrics(
        short_result,
        short_strategy,
        data,
        split_index,
        len(data.dates) - 1,
        "bear",
    )
    risk_on_excess = (
        recommended_risk_on_test["cagr"] - benchmark_risk_on_test["cagr"]
    )
    macro_bull_end = min(VALIDATION_MACRO_BULL_END, test_end)
    if macro_bull_end > test_start:
        recommended_macro_bull = recommended_result.metrics_dict(
            test_start,
            macro_bull_end,
        )
        benchmark_macro_bull = benchmark.metrics_dict(test_start, macro_bull_end)
    else:
        recommended_macro_bull = None
        benchmark_macro_bull = None
    exposure = _exposure_stats(recommended_result, split_index, len(data.dates) - 1)

    return {
        "methodology": {
            "goal": "连续宏观牛市 CAGR 领先 BTC；确认突破可使用训练候选允许的杠杆",
            "selection": "仅用训练期双折；先选牛市超额，再独立决定做空或熊市现金",
            "signal_timing": "T-1 收盘生成权重，持有 T 日 close-to-close 收益",
            "core_pool_policy": "固定五币核心池，不在全样本上按事后收益或成交量换池",
            "key_level": "BTC 收盘突破不含当日的过去 55 日最高收盘价",
            "short_fallback": "影子空头滚动净收益/胜率不达标则在线空仓；训练期贡献不稳则全局禁用做空",
            "risk_on_metric": "非风险开启日按现金零收益保留在完整日历中，再计算贡献 CAGR；另列条件累计收益",
            "macro_bull_metric": "2024-04-28 至 2025-10-06 为事后历史分段，只用于最终评价、不参与选参",
            "evaluation_status": "开发后验证集；规则曾在查看该区间后修订，不能视为未触碰 forward OOS",
            "fee_rate_one_way": fee_rate,
            "slippage_rate_one_way": slippage_rate,
            "borrow_rate_daily": limits.borrow_rate_daily,
        },
        "universe": list(data.symbols),
        "core_pool": list(CORE_TOP5_SYMBOLS),
        "data_range": {
            "start": data.dates[0].isoformat(),
            "end": data.dates[-1].isoformat(),
        },
        "split": {
            "warmup_end": data.dates[warmup - 1].isoformat(),
            "train_start": train_start.isoformat(),
            "train_end": locked_train_end.isoformat(),
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
            "split_policy": "固定训练截止日，追加未来数据不会回流训练集",
        },
        "selection": {
            "long_candidates_tested": len(ranked),
            "unique_return_paths": len(
                {item[3].daily_returns.tobytes() for item in ranked}
            ),
            "long_candidates_qualified_both_folds": sum(int(item[1]) for item in ranked),
            "selected_was_qualified": long_qualified,
            "objective": "min(risk-on excess)+0.4*mean(risk-on excess)+0.15*min(sharpe)-0.25*max(MDD)",
            "selected_training": selected_training,
            "qualification_status": (
                "qualified" if long_qualified else "exploratory_not_qualified"
            ),
        },
        "selected_parameters": _strategy_params(recommended_strategy),
        "short_module": {
            **short_decision,
            "cash_bear_test": cash_bear_test,
            "adaptive_short_bear_test": short_bear_test,
            "cash_full_test": cash_result.metrics_dict(test_start, test_end),
            "adaptive_short_full_test": short_result.metrics_dict(test_start, test_end),
        },
        "benchmark": {
            "train": benchmark.metrics_dict(train_start, locked_train_end),
            "test": benchmark_test,
            "risk_on_test": benchmark_risk_on_test,
            "macro_bull_test": benchmark_macro_bull,
        },
        "strategy": {
            "train": recommended_result.metrics_dict(train_start, locked_train_end),
            "test": recommended_test,
            "risk_on_test": recommended_risk_on_test,
            "macro_bull_test": recommended_macro_bull,
            "bear_test": recommended_bear_test,
            "risk_on_excess_cagr": risk_on_excess,
            "risk_on_beats_btc": risk_on_excess > 0,
            "macro_bull_excess_cagr": (
                recommended_macro_bull["cagr"] - benchmark_macro_bull["cagr"]
                if recommended_macro_bull is not None and benchmark_macro_bull is not None
                else None
            ),
            "macro_bull_beats_btc": (
                recommended_macro_bull["cagr"] > benchmark_macro_bull["cagr"]
                if recommended_macro_bull is not None and benchmark_macro_bull is not None
                else None
            ),
            "full_test_beats_btc": recommended_test["cagr"] > benchmark_test["cagr"],
            "oos_exposure": exposure,
        },
        "equity_curves": {
            "BTC buy-and-hold": [float(value) for value in benchmark.equity],
            "TOP5 cash-fallback": [float(value) for value in cash_result.equity],
            "TOP5 adaptive-short": [float(value) for value in short_result.equity],
            "TOP5 selected": [float(value) for value in recommended_result.equity],
        },
        "dates": [day.isoformat() for day in data.dates],
    }


def write_core_top5_report(path: Path, results: dict[str, Any]) -> None:
    """生成包含规则、训练选择、开发后验证表现和风险边界的报告。

    报告明确区分训练选择与开发后验证，并同时展示空仓及自适应做空结果，
    使读者能检查做空模块是否真正增加熊市收益，而非只看最终组合总收益。
    """

    strategy = results["strategy"]
    benchmark = results["benchmark"]
    short = results["short_module"]
    params = results["selected_parameters"]
    exposure = strategy["oos_exposure"]
    decision = "启用自适应做空" if short["enabled"] else "做空未通过，熊市退回现金"
    qualification_passed = results["selection"]["selected_was_qualified"]
    title_suffix = "" if qualification_passed else "（探索性、训练资格未通过）"
    macro_row = (
        _comparison_row(
            "事后宏观牛段",
            strategy["macro_bull_test"],
            benchmark["macro_bull_test"],
        )
        if strategy["macro_bull_test"] is not None
        and benchmark["macro_bull_test"] is not None
        else "| 事后宏观牛段 | N/A | N/A | N/A | N/A | N/A |"
    )
    lines = [
        f"# TOP5 核心池激进牛熊轮动策略{title_suffix}",
        "",
        "## 最终规则",
        "",
        f"- 核心池：`{', '.join(results['core_pool'])}`；池外权重恒为零。",
        "- 牛市基础仓：BTC 站上慢趋势且长动量为正；宽度不足时持有 BTC，避免错过主升浪。",
        "- 轮动：快均线和核心池广度确认后，按长短动量排名并逆波动持有最强标的。",
        (
            f"- 杠杆：BTC 在调仓信号日突破此前 55 日最高收盘价时，名义敞口进入 "
            f"`{params['breakout_min_gross']:.2f}–{params['leveraged_max_gross']:.2f}` 倍区间，"
            f"最长持有至下一次 `{params['rebalance_days']}` 日调仓。"
        ),
        "- 熊市：只做空跌破趋势且长动量显著为负的最弱核心资产。",
        "- 双重熔断：滚动影子空头不达收益/胜率门槛则当期空仓；训练贡献不稳则全局禁用。",
        "",
        "## 训练选择",
        "",
        f"- 多头候选 `{results['selection']['long_candidates_tested']}` 组；"
        f"形成 `{results['selection']['unique_return_paths']}` 条不同收益路径；"
        f"双折均跑赢 BTC 的候选 `{results['selection']['long_candidates_qualified_both_folds']}` 组。",
        f"- 做空结论：**{decision}**。",
        (
            "- 训练资格：**通过，可进入后续 forward 验证**。"
            if qualification_passed
            else "- 训练资格：**未通过**；以下仅为最高分探索性候选，不构成正式推荐。"
        ),
        f"- 探索性选中参数：`{params}`",
        "",
        "## 开发后验证集结果",
        "",
        "| 区间/状态 | 策略 CAGR | BTC CAGR | 超额 | 策略 Sharpe | 策略最大回撤 |",
        "|---|---:|---:|---:|---:|---:|",
        _comparison_row("完整开发后验证集", strategy["test"], benchmark["test"]),
        macro_row,
        _comparison_row(
            "因果 risk-on 日历贡献",
            strategy["risk_on_test"],
            benchmark["risk_on_test"],
        ),
        "",
        (
            f"- 事后宏观牛段是否领先 BTC："
            f"`{'是' if strategy['macro_bull_beats_btc'] else '否'}`；"
            f"超额 CAGR `{strategy['macro_bull_excess_cagr']:+.1%}`。"
            if strategy["macro_bull_beats_btc"] is not None
            else "- 当前开发后验证集没有完整宏观牛段。"
        ),
        f"- 因果 risk-on 的完整日历贡献超额 CAGR `{strategy['risk_on_excess_cagr']:+.1%}`；"
        f"其中活跃 `{strategy['risk_on_test']['state_observations']}` 天，"
        f"条件累计收益 `{strategy['risk_on_test']['conditional_total_return']:.1%}`。",
        f"- 熊市状态策略 CAGR `{strategy['bear_test']['cagr']:.1%}`；"
        f"现金方案 `{short['cash_bear_test']['cagr']:.1%}`；"
        f"自适应做空方案 `{short['adaptive_short_bear_test']['cagr']:.1%}`。",
        f"- 完整验证集现金方案 CAGR/Sharpe "
        f"`{short['cash_full_test']['cagr']:.1%}/{short['cash_full_test']['sharpe']:.2f}`；"
        f"自适应做空 `{short['adaptive_short_full_test']['cagr']:.1%}/"
        f"{short['adaptive_short_full_test']['sharpe']:.2f}`。",
        f"- 验证集实际最大总敞口 `{exposure['max_gross']:.3f}` 倍；"
        f"杠杆日 `{exposure['leveraged_days']}`；最大空头 `{exposure['max_short']:.3f}` 倍。",
        "",
        "## 风险边界",
        "",
        "- 这是研究原型，不是收益承诺；激进目标不等于每个牛市切片都必然跑赢 BTC。",
        "- 固定当期大市值币池存在幸存者偏差，不能把当前 TOP5 身份回填为历史事实。",
        "- 日线回测无法模拟盘中强平、跳空、交易所保证金阶梯和逐币种资金费率。",
        "- 当前验证区间已在开发过程中被查看；只有 2026-07-15 之后新增数据可作为新的 forward OOS。",
        "- 验证集仍只有一个完整市场路径；上线前应做滚动仿真、资金容量和极端滑点压力测试。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_core_top5_research(path: Path, results: dict[str, Any]) -> Path:
    """绘制全样本净值与开发后验证集 CAGR，提供可快速审阅的验证图。

    左图使用对数净值比较 BTC、熊市现金和自适应做空方案；右图只展示
    固定训练截止日后的完整、宏观牛段与 risk-on 贡献 CAGR。由于开发中
    已查看过该区间，图标题明确标为开发后验证而非未触碰样本外。
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dates = np.arange(len(results["dates"]))
    curves = results["equity_curves"]
    strategy = results["strategy"]
    benchmark = results["benchmark"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name in ("BTC buy-and-hold", "TOP5 cash-fallback", "TOP5 adaptive-short"):
        axes[0].plot(dates, curves[name], label=name, linewidth=1.8)
    axes[0].set_yscale("log")
    axes[0].set_title("Full-sample equity (log scale)")
    axes[0].set_xlabel("Trading day index")
    axes[0].set_ylabel("Equity")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=8)

    labels = ["Full dev validation", "Macro bull", "Risk-on contribution"]
    strategy_cagr = [
        strategy["test"]["cagr"] * 100,
        (
            strategy["macro_bull_test"]["cagr"] * 100
            if strategy["macro_bull_test"] is not None
            else 0.0
        ),
        strategy["risk_on_test"]["cagr"] * 100,
    ]
    benchmark_cagr = [
        benchmark["test"]["cagr"] * 100,
        (
            benchmark["macro_bull_test"]["cagr"] * 100
            if benchmark["macro_bull_test"] is not None
            else 0.0
        ),
        benchmark["risk_on_test"]["cagr"] * 100,
    ]
    x = np.arange(len(labels))
    width = 0.36
    axes[1].bar(x - width / 2, benchmark_cagr, width, label="BTC / cash baseline")
    axes[1].bar(x + width / 2, strategy_cagr, width, label="TOP5 strategy")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("CAGR %")
    axes[1].set_title("Post-development validation (not untouched OOS)")
    axes[1].axhline(0, color="#555555", linewidth=1)
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _select_short_module(
    data: MarketData,
    cash_strategy: CoreTop5RegimeRotation,
    cash_result: BacktestResult,
    short_strategy: CoreTop5RegimeRotation,
    short_result: BacktestResult,
    folds: tuple[tuple[int, int], ...],
    train_start: int,
    train_end: int,
) -> dict[str, Any]:
    """比较做空和现金在训练熊市中的贡献，并执行全局降级决策。

    做空必须在两个训练折及完整训练区间都提高熊市总收益，同时全局夏普
    最多下降 0.05、最大回撤最多恶化 3 个百分点。严格的 AND 规则优先
    控制错误启用风险；任何一项失败都把熊市动作降级为现金。
    """

    fold_rows = []
    fold_improvements = []
    for start, end in folds:
        cash = _regime_metrics(cash_result, cash_strategy, data, start, end, "bear")
        short = _regime_metrics(short_result, short_strategy, data, start, end, "bear")
        improvement = short["total_return"] - cash["total_return"]
        fold_improvements.append(improvement)
        fold_rows.append(
            {
                "start": data.dates[start].isoformat(),
                "end": data.dates[end].isoformat(),
                "cash": cash,
                "adaptive_short": short,
                "total_return_improvement": improvement,
            }
        )

    cash_bear = _regime_metrics(
        cash_result,
        cash_strategy,
        data,
        train_start,
        train_end,
        "bear",
    )
    short_bear = _regime_metrics(
        short_result,
        short_strategy,
        data,
        train_start,
        train_end,
        "bear",
    )
    cash_overall = cash_result.metrics(data.dates[train_start], data.dates[train_end])
    short_overall = short_result.metrics(data.dates[train_start], data.dates[train_end])
    enabled = (
        min(fold_improvements) > 0
        and short_bear["total_return"] > cash_bear["total_return"] + 0.01
        and short_overall.sharpe >= cash_overall.sharpe - 0.05
        and short_overall.max_drawdown <= cash_overall.max_drawdown + 0.03
    )
    failed_checks = []
    if min(fold_improvements) <= 0:
        failed_checks.append("至少一个训练折的熊市收益未改善")
    if short_bear["total_return"] <= cash_bear["total_return"] + 0.01:
        failed_checks.append("完整训练期熊市收益改善不足 1 个百分点")
    if short_overall.sharpe < cash_overall.sharpe - 0.05:
        failed_checks.append("训练期整体 Sharpe 恶化超过 0.05")
    if short_overall.max_drawdown > cash_overall.max_drawdown + 0.03:
        failed_checks.append("训练期最大回撤恶化超过 3 个百分点")
    return {
        "enabled": enabled,
        "decision": "adaptive_short" if enabled else "cash_fallback",
        "failed_checks": failed_checks,
        "training_folds": fold_rows,
        "cash_training_bear": cash_bear,
        "adaptive_short_training_bear": short_bear,
        "cash_training_overall": asdict(cash_overall),
        "adaptive_short_training_overall": asdict(short_overall),
    }


def _regime_metrics(
    result: BacktestResult,
    regime_strategy: CoreTop5RegimeRotation,
    data: MarketData,
    start: int,
    end: int,
    regime: str,
) -> dict[str, Any]:
    """按前一日可知的市场状态计算完整日历贡献和条件收益。

    第 ``i`` 日收益对应第 ``i-1`` 日信号，因此状态标签也取前一日，保持
    与回测持仓完全一致。非目标状态日以现金零收益保留在原日历中，CAGR、
    波动率和回撤因此仍按完整连续区间计算；另附活跃日条件累计及年化值，
    但明确不把后者称为连续市场阶段 CAGR。
    """

    calendar_indices = np.arange(
        max(1, start),
        min(end, len(data.dates) - 1) + 1,
        dtype=int,
    )
    state_mask = np.asarray(
        [
            regime_strategy.market_regime(data, int(index) - 1) == regime
            for index in calendar_indices
        ],
        dtype=bool,
    )
    conditional_returns = result.daily_returns[calendar_indices[state_mask]]
    calendar_returns = np.zeros(len(calendar_indices), dtype=float)
    calendar_returns[state_mask] = conditional_returns
    metrics = _return_metrics(calendar_returns)
    conditional_total_return = (
        float(np.prod(1.0 + conditional_returns) - 1.0)
        if len(conditional_returns)
        else 0.0
    )
    conditional_years = len(conditional_returns) / 365.0
    conditional_annualized_return = (
        float((1.0 + conditional_total_return) ** (1.0 / conditional_years) - 1.0)
        if conditional_years > 0 and conditional_total_return > -1.0
        else 0.0
    )
    return {
        **metrics,
        "state_observations": int(state_mask.sum()),
        "conditional_total_return": conditional_total_return,
        "conditional_annualized_return": conditional_annualized_return,
    }


def _return_metrics(returns: np.ndarray) -> dict[str, Any]:
    """从连续日历收益数组计算统一年化指标。

    数组中的现金日应显式写为零，以确保 CAGR 使用真实日历长度。空数组
    返回零值；两天以上才计算样本波动率，避免单观测自由度导致 NaN。
    """

    observations = len(returns)
    if observations == 0:
        return {
            "observations": 0,
            "total_return": 0.0,
            "cagr": 0.0,
            "annual_volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }
    equity = np.cumprod(1.0 + returns)
    total_return = float(equity[-1] - 1.0)
    years = observations / 365.0
    cagr = float(equity[-1] ** (1.0 / years) - 1.0) if equity[-1] > 0 else -1.0
    volatility = (
        float(np.std(returns, ddof=1) * np.sqrt(365.0)) if observations > 1 else 0.0
    )
    sharpe = (
        float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(365.0))
        if volatility > 0
        else 0.0
    )
    anchored = np.concatenate(([1.0], equity))
    peaks = np.maximum.accumulate(anchored)
    max_drawdown = float(-np.min(anchored / peaks - 1.0))
    return {
        "observations": observations,
        "total_return": total_return,
        "cagr": cagr,
        "annual_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def _exposure_stats(result: BacktestResult, start: int, end: int) -> dict[str, Any]:
    """汇总指定区间的真实总敞口、空头和杠杆使用情况。

    统计直接读取经过引擎裁剪后的每日权重，而不是策略配置上限，因此可
    揭示理论杠杆帽是否真正被触发，并防止报告把“允许”误写成“已使用”。
    """

    weights = np.asarray(result.weights[start : end + 1], dtype=float)
    gross = np.sum(np.abs(weights), axis=1)
    short = np.sum(np.maximum(-weights, 0.0), axis=1)
    active = gross > 1e-8
    return {
        "max_gross": float(np.max(gross)) if len(gross) else 0.0,
        "leveraged_days": int(np.sum(gross > 1.0 + 1e-8)),
        "active_days": int(np.sum(active)),
        "average_active_gross": float(np.mean(gross[active])) if active.any() else 0.0,
        "max_short": float(np.max(short)) if len(short) else 0.0,
    }


def _comparison_row(label: str, strategy: dict[str, Any], benchmark: dict[str, Any]) -> str:
    """格式化策略与 BTC 的单行 Markdown 对比，统一百分比精度。

    集中格式化可避免完整区间和牛市状态使用不同口径，并让报告中展示的
    超额始终由同一对 CAGR 直接相减得到。
    """

    excess = strategy["cagr"] - benchmark["cagr"]
    return (
        f"| {label} | {strategy['cagr']:.1%} | {benchmark['cagr']:.1%} | "
        f"{excess:+.1%} | {strategy['sharpe']:.2f} | {strategy['max_drawdown']:.1%} |"
    )


def _strategy_params(strategy: CoreTop5RegimeRotation) -> dict[str, Any]:
    """导出可复现实例的构造参数，排除仅用于展示的名称和思想字段。

    使用 dataclass 字段反射避免新增参数后报告遗漏，元组会由 JSON 编码器
    稳定写成数组，能够直接用于后续配置转换。
    """

    return {
        field.name: getattr(strategy, field.name)
        for field in fields(strategy)
        if field.name not in {"name", "ideas"}
    }


def _validate_research_inputs(data: MarketData, train_end: date) -> None:
    """验证研究所需核心池、样本长度和固定训练截止日。

    研究至少需要 900 天，以覆盖 200 日慢趋势预热、两个训练折和独立样本
    开发后验证区间；截止日必须落在样本内部，确保程序化选择区和评价区不重叠。
    """

    missing = [symbol for symbol in CORE_TOP5_SYMBOLS if symbol not in data.symbols]
    if missing:
        raise ValueError(f"研究数据缺少核心池标的：{', '.join(missing)}")
    if len(data.dates) < 900:
        raise ValueError("TOP5 研究至少需要 900 天行情")
    train_days = sum(day <= train_end for day in data.dates)
    test_days = len(data.dates) - train_days
    if train_days < 500 or test_days < 180:
        raise ValueError("固定训练截止日必须保留至少 500 天训练和 180 天验证数据")
