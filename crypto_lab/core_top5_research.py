"""TOP5 激进轮动策略的训练期选型、做空降级与样本外报告。"""

from __future__ import annotations

from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

import numpy as np

from .backtest import BacktestResult, buy_and_hold
from .core_top5 import CORE_TOP5_SYMBOLS, CoreTop5RegimeRotation
from .data import MarketData
from .long_short import LongShortLimits, run_long_short_backtest


LONG_CONFIGS: tuple[dict[str, Any], ...] = tuple(
    {
        "rebalance_days": rebalance_days,
        "vol_target": vol_target,
        "leveraged_max_gross": leveraged_max_gross,
    }
    for rebalance_days in (7, 14)
    for vol_target in (0.35, 0.40, 0.45)
    for leveraged_max_gross in (1.3, 1.5)
)


def run_core_top5_research(
    data: MarketData,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    train_fraction: float = 0.60,
) -> dict[str, Any]:
    """仅用训练期选择牛市参数和是否启用做空，再做一次样本外评价。

    第一阶段比较有限的调仓频率、波动率目标和杠杆上限，以两个训练折中
    较弱的牛市超额 CAGR 为主要目标。第二阶段固定多头参数，对比自适应
    做空与熊市现金；只有做空在两个训练折及完整训练期均改善熊市收益，
    且未明显恶化全局夏普和回撤时才启用。样本外结果不参与任何选择。
    """

    _validate_research_inputs(data, train_fraction)
    split_index = max(500, int(len(data.dates) * train_fraction))
    if len(data.dates) - split_index < 180:
        raise ValueError("样本外区间至少需要 180 天")
    warmup = 220
    mid_index = warmup + (split_index - warmup) // 2
    folds = ((warmup, mid_index - 1), (mid_index, split_index - 1))
    limits = LongShortLimits(
        max_gross_exposure=1.5,
        max_net_exposure=1.5,
        max_short_exposure=0.5,
    )
    benchmark = buy_and_hold(data, fee_rate=fee_rate, slippage_rate=slippage_rate)

    ranked: list[tuple[float, bool, CoreTop5RegimeRotation, BacktestResult, dict[str, Any]]] = []
    for config in LONG_CONFIGS:
        strategy = CoreTop5RegimeRotation(short_gross=0.0, **config)
        result = run_long_short_backtest(
            data,
            strategy,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            limits=limits,
        )
        fold_details = []
        bull_excesses = []
        fold_sharpes = []
        fold_drawdowns = []
        for start, end in folds:
            strategy_bull = _regime_metrics(result, strategy, data, start, end, "bull")
            benchmark_bull = _regime_metrics(benchmark, strategy, data, start, end, "bull")
            overall = result.metrics(data.dates[start], data.dates[end])
            excess = strategy_bull["cagr"] - benchmark_bull["cagr"]
            bull_excesses.append(excess)
            fold_sharpes.append(overall.sharpe)
            fold_drawdowns.append(overall.max_drawdown)
            fold_details.append(
                {
                    "start": data.dates[start].isoformat(),
                    "end": data.dates[end].isoformat(),
                    "bull_strategy": strategy_bull,
                    "bull_benchmark": benchmark_bull,
                    "bull_excess_cagr": excess,
                    "overall": asdict(overall),
                }
            )
        qualified = min(bull_excesses) > 0 and min(fold_sharpes) > 0
        score = (
            min(bull_excesses)
            + 0.40 * float(np.mean(bull_excesses))
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
    train_end = data.dates[split_index - 1]
    test_start = data.dates[split_index]
    test_end = data.dates[-1]
    recommended_test = recommended_result.metrics_dict(test_start, test_end)
    benchmark_test = benchmark.metrics_dict(test_start, test_end)
    recommended_bull_test = _regime_metrics(
        recommended_result,
        recommended_strategy,
        data,
        split_index,
        len(data.dates) - 1,
        "bull",
    )
    benchmark_bull_test = _regime_metrics(
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
    bull_excess = recommended_bull_test["cagr"] - benchmark_bull_test["cagr"]

    return {
        "methodology": {
            "goal": "牛市风险开阶段 CAGR 大幅领先 BTC；关键突破可用最高 1.5 倍名义敞口",
            "selection": "仅用训练期双折；先选牛市超额，再独立决定做空或熊市现金",
            "signal_timing": "T-1 收盘生成权重，持有 T 日 close-to-close 收益",
            "core_pool_policy": "固定五币核心池，不在全样本上按事后收益或成交量换池",
            "key_level": "BTC 收盘突破不含当日的过去 55 日最高收盘价",
            "short_fallback": "影子空头滚动净收益/胜率不达标则在线空仓；训练期贡献不稳则全局禁用做空",
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
            "train_end": train_end.isoformat(),
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
            "train_fraction": train_fraction,
        },
        "selection": {
            "long_candidates_tested": len(ranked),
            "long_candidates_qualified_both_folds": sum(int(item[1]) for item in ranked),
            "selected_was_qualified": long_qualified,
            "objective": "min(bull excess)+0.4*mean(bull excess)+0.15*min(sharpe)-0.25*max(MDD)",
            "selected_training": selected_training,
        },
        "recommended_parameters": _strategy_params(recommended_strategy),
        "short_module": {
            **short_decision,
            "cash_bear_test": cash_bear_test,
            "adaptive_short_bear_test": short_bear_test,
        },
        "benchmark": {
            "train": benchmark.metrics_dict(train_start, train_end),
            "test": benchmark_test,
            "bull_test": benchmark_bull_test,
        },
        "strategy": {
            "train": recommended_result.metrics_dict(train_start, train_end),
            "test": recommended_test,
            "bull_test": recommended_bull_test,
            "bear_test": recommended_bear_test,
            "bull_excess_cagr": bull_excess,
            "bull_beats_btc": bull_excess > 0,
            "full_test_beats_btc": recommended_test["cagr"] > benchmark_test["cagr"],
        },
        "equity_curves": {
            "BTC buy-and-hold": [float(value) for value in benchmark.equity],
            "TOP5 cash-fallback": [float(value) for value in cash_result.equity],
            "TOP5 adaptive-short": [float(value) for value in short_result.equity],
            "TOP5 recommended": [float(value) for value in recommended_result.equity],
        },
        "dates": [day.isoformat() for day in data.dates],
    }


def write_core_top5_report(path: Path, results: dict[str, Any]) -> None:
    """生成包含规则、训练选择、样本外表现和风险边界的 Markdown 报告。

    报告明确区分训练选择与样本外评价，并同时展示空仓及自适应做空结果，
    使读者能检查做空模块是否真正增加熊市收益，而非只看最终组合总收益。
    """

    strategy = results["strategy"]
    benchmark = results["benchmark"]
    short = results["short_module"]
    params = results["recommended_parameters"]
    decision = "启用自适应做空" if short["enabled"] else "做空未通过，熊市退回现金"
    lines = [
        "# TOP5 核心池激进牛熊轮动策略",
        "",
        "## 最终规则",
        "",
        f"- 核心池：`{', '.join(results['core_pool'])}`；池外权重恒为零。",
        "- 牛市：BTC 站上慢趋势、快均线高于慢均线、长动量为正且核心池广度达标。",
        "- 轮动：长短动量合成排名，持有最强标的并按逆波动率分配。",
        "- 杠杆：仅 BTC 突破此前 55 日最高收盘价时，将名义总敞口上限放宽至 1.5 倍。",
        "- 熊市：只做空跌破趋势且长动量显著为负的最弱核心资产。",
        "- 双重熔断：滚动影子空头不达收益/胜率门槛则当期空仓；训练贡献不稳则全局禁用。",
        "",
        "## 训练选择",
        "",
        f"- 多头候选 `{results['selection']['long_candidates_tested']}` 组；"
        f"双折均跑赢 BTC 的候选 `{results['selection']['long_candidates_qualified_both_folds']}` 组。",
        f"- 做空结论：**{decision}**。",
        f"- 推荐参数：`{params}`",
        "",
        "## 样本外结果",
        "",
        "| 区间/状态 | 策略 CAGR | BTC CAGR | 超额 | 策略 Sharpe | 策略最大回撤 |",
        "|---|---:|---:|---:|---:|---:|",
        _comparison_row("完整样本外", strategy["test"], benchmark["test"]),
        _comparison_row("因果牛市状态", strategy["bull_test"], benchmark["bull_test"]),
        "",
        f"- 牛市状态是否领先 BTC：`{'是' if strategy['bull_beats_btc'] else '否'}`；"
        f"超额 CAGR `{strategy['bull_excess_cagr']:+.1%}`。",
        f"- 熊市状态策略 CAGR `{strategy['bear_test']['cagr']:.1%}`；"
        f"现金方案 `{short['cash_bear_test']['cagr']:.1%}`；"
        f"自适应做空方案 `{short['adaptive_short_bear_test']['cagr']:.1%}`。",
        "",
        "## 风险边界",
        "",
        "- 这是研究原型，不是收益承诺；激进目标不等于每个牛市切片都必然跑赢 BTC。",
        "- 固定当期大市值币池存在幸存者偏差，不能把当前 TOP5 身份回填为历史事实。",
        "- 日线回测无法模拟盘中强平、跳空、交易所保证金阶梯和逐币种资金费率。",
        "- 样本外仍只有一个完整市场路径；上线前应做滚动仿真、资金容量和极端滑点压力测试。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_core_top5_research(path: Path, results: dict[str, Any]) -> Path:
    """绘制全样本净值与样本外状态 CAGR，提供可快速审阅的验证图。

    左图使用对数净值比较 BTC、熊市现金和自适应做空方案；右图只展示
    已锁定参数后的样本外完整、牛市与熊市 CAGR，避免把训练拟合图误当
    成最终证据。
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

    labels = ["Full OOS", "Bull state", "Bear state"]
    strategy_cagr = [
        strategy["test"]["cagr"] * 100,
        strategy["bull_test"]["cagr"] * 100,
        strategy["bear_test"]["cagr"] * 100,
    ]
    benchmark_cagr = [
        benchmark["test"]["cagr"] * 100,
        benchmark["bull_test"]["cagr"] * 100,
        0.0,
    ]
    x = np.arange(len(labels))
    width = 0.36
    axes[1].bar(x - width / 2, benchmark_cagr, width, label="BTC / cash baseline")
    axes[1].bar(x + width / 2, strategy_cagr, width, label="TOP5 strategy")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("CAGR %")
    axes[1].set_title("Locked-parameter out-of-sample results")
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
    """按前一日可知的市场状态抽取收益并计算状态内风险收益指标。

    第 ``i`` 日收益对应第 ``i-1`` 日信号，因此状态标签也取前一日，保持
    与回测持仓完全一致。非连续状态日按日收益拼接，仅用于条件表现比较，
    不应解释为可直接投资的连续净值路径。
    """

    indices = [
        index
        for index in range(max(1, start), min(end, len(data.dates) - 1) + 1)
        if regime_strategy.market_regime(data, index - 1) == regime
    ]
    returns = result.daily_returns[np.asarray(indices, dtype=int)] if indices else np.array([])
    return _return_metrics(returns)


def _return_metrics(returns: np.ndarray) -> dict[str, Any]:
    """从可为非连续状态日的收益数组计算统一年化指标。

    空状态返回零值而不是抛错，使训练折在没有对应市场状态时可被明确判为
    没有证据；两天以上才计算样本波动率，避免单观测自由度导致 NaN。
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


def _validate_research_inputs(data: MarketData, train_fraction: float) -> None:
    """验证研究所需核心池、样本长度和时间切分比例。

    研究至少需要 900 天，以覆盖 200 日慢趋势预热、两个训练折和独立样本
    外区间；提前拒绝短样本可避免把大量空仓预热期误当成稳健表现。
    """

    missing = [symbol for symbol in CORE_TOP5_SYMBOLS if symbol not in data.symbols]
    if missing:
        raise ValueError(f"研究数据缺少核心池标的：{', '.join(missing)}")
    if len(data.dates) < 900:
        raise ValueError("TOP5 研究至少需要 900 天行情")
    if not 0.50 <= train_fraction <= 0.80:
        raise ValueError("train_fraction 必须位于 [0.50, 0.80]")
