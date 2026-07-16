"""参数优化、样本外评估与研究报告生成。"""

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
from .strategies import (
    AllWeatherRotation,
    CompositeFactorRotation,
    RollingRidgeRotation,
    RsiFactorRotation,
    SmallLiquidityRotation,
    TrendRotation,
)


STRATEGY_SPECS = (
    (
        TrendRotation,
        {
            "lookback": [20, 30, 60],
            "short_window": [5, 10],
            "top_k": [2, 3],
        },
    ),
    (
        AllWeatherRotation,
        {
            "regime_window": [10, 20, 40],
            "top_k": [3, 4],
        },
    ),
    (
        SmallLiquidityRotation,
        {
            "liquidity_window": [20, 30],
            "top_k": [4, 6],
            "regime_window": [50, 100],
        },
    ),
    (
        CompositeFactorRotation,
        {
            "top_k": [3, 5, 7],
            "regime_window": [50, 100, 150],
        },
    ),
    (
        RsiFactorRotation,
        {
            "top_k": [4, 6],
            "entry_rsi": [45.0, 55.0],
            "exit_rsi": [65.0, 75.0],
        },
    ),
    (
        RollingRidgeRotation,
        {
            "top_k": [3, 5],
            "train_window": [180, 365],
            "ridge": [0.1, 1.0, 10.0],
        },
    ),
)


def run_research(
    data: MarketData,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    train_fraction: float = 0.60,
) -> dict[str, Any]:
    """在前段数据选参，只在锁定参数后评价后段样本。"""

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

    for strategy_type, grid in STRATEGY_SPECS:
        default_strategy = strategy_type()
        default_result = run_backtest(data, default_strategy, fee_rate, slippage_rate)
        best_score = -np.inf
        best_strategy = default_strategy
        candidates = 0
        for params in _parameter_product(grid):
            strategy = strategy_type(**params)
            result = run_backtest(data, strategy, fee_rate, slippage_rate)
            metrics = result.metrics(train_start, train_end)
            score = metrics.sharpe - 0.5 * metrics.max_drawdown
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
        statistically_supported = bootstrap["cagr_ci_95"][0] > 0
        verdict = (
            "统计通过"
            if optimized_positive and default_positive and statistically_supported
            else "点估计为正（无统计支持）"
            if optimized_positive
            else "未通过"
        )
        strategies[best_strategy.name] = {
            "source_ids": list(best_strategy.source_ids),
            "candidates_tested": candidates,
            "selection_objective": "train_sharpe - 0.5 * train_max_drawdown",
            "default_parameters": _strategy_parameters(default_strategy),
            "selected_parameters": _strategy_parameters(best_strategy),
            "default_train": default_result.metrics_dict(train_start, train_end),
            "default_test": default_test,
            "optimized_train": optimized_result.metrics_dict(train_start, train_end),
            "optimized_test": optimized_test,
            "optimized_test_bootstrap": bootstrap,
            "cross_market_verdict": verdict,
        }

    benchmark = buy_and_hold(data, fee_rate=fee_rate, slippage_rate=slippage_rate)
    benchmark_test = benchmark.metrics_dict(test_start, test_end)
    for result in strategies.values():
        test = result["optimized_test"]
        test["oos_positive"] = test["cagr"] > 0 and test["sharpe"] > 0
        test["default_oos_positive"] = (
            result["default_test"]["cagr"] > 0 and result["default_test"]["sharpe"] > 0
        )
        test["risk_adjusted_better_than_btc"] = (
            test["sharpe"] > benchmark_test["sharpe"]
            and test["max_drawdown"] < benchmark_test["max_drawdown"]
        )

    return {
        "methodology": {
            "signal_timing": "使用 T-1 及更早收盘数据生成权重，获取 T 日收盘到收盘收益",
            "positioning": "仅现货多头，不加杠杆、不做空；未分配权重为零收益现金",
            "fee_rate_one_way": fee_rate,
            "slippage_rate_one_way": slippage_rate,
            "annualization_days": 365,
            "selection": "仅训练区间网格选参；样本外区间不参与选参",
            "uncertainty": "样本外日收益使用 14 日循环区块 Bootstrap，2,000 次重采样",
            "known_biases": [
                "固定使用当前仍在 OKX 交易且历史较长的币对，存在幸存者偏差",
                "成交量是小市值/基本面因子的代理，不等同于历史流通市值或链上财务",
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """以 UTF-8 和稳定缩进写出研究结果。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown_report(path: Path, results: dict[str, Any], manifest: dict[str, Any]) -> None:
    """生成包含可追溯数据与样本外结论的 Markdown 报告。"""

    split = results["split"]
    benchmark = results["benchmark"]["test"]
    rows = []
    verdict_counts = {"统计通过": 0, "点估计为正（无统计支持）": 0, "未通过": 0}
    for name, item in results["strategies"].items():
        metrics = item["optimized_test"]
        default_metrics = item["default_test"]
        bootstrap = item["optimized_test_bootstrap"]
        verdict = item["cross_market_verdict"]
        verdict_counts[verdict] += 1
        rows.append(
            "| {name} | {sources} | {default_cagr:.1%} | {cagr:.1%} | {sharpe:.2f} | {mdd:.1%} | {lower:.1%}–{upper:.1%} | {verdict} |".format(
                name=name,
                sources=", ".join(item["source_ids"]),
                default_cagr=default_metrics["cagr"],
                cagr=metrics["cagr"],
                sharpe=metrics["sharpe"],
                mdd=metrics["max_drawdown"],
                lower=bootstrap["cagr_ci_95"][0],
                upper=bootstrap["cagr_ci_95"][1],
                verdict=verdict,
            )
        )
    files = manifest.get("files", [])
    lines = [
        "# 聚宽策略迁移 Crypto：真实行情样本外验证",
        "",
        "## 结论",
        "",
        "本报告评价的是受聚宽策略启发、重新设计后的 Crypto 原型，不把合并后的原型收益归因给任一原策略。",
        "原聚宽收益声明没有逐日净值可复现，因此不能据此认定原策略跨市场有效，也不构成实盘收益承诺。",
        "",
        f"- 训练区间：`{split['train_start']}` 至 `{split['train_end']}`",
        f"- 样本外区间：`{split['test_start']}` 至 `{split['test_end']}`",
        f"- BTC 样本外：CAGR `{benchmark['cagr']:.1%}`，Sharpe `{benchmark['sharpe']:.2f}`，最大回撤 `{benchmark['max_drawdown']:.1%}`",
        f"- 成本假设：单边手续费 `{results['methodology']['fee_rate_one_way']:.2%}` + 单边滑点 `{results['methodology']['slippage_rate_one_way']:.2%}`",
        f"- 判定汇总：统计通过 `{verdict_counts['统计通过']}`，点估计为正但无统计支持 `{verdict_counts['点估计为正（无统计支持）']}`，未通过 `{verdict_counts['未通过']}`。",
        "",
        "| Crypto 策略族 | 借鉴来源 | 默认参数 CAGR | 优化参数 CAGR | Sharpe | 最大回撤 | 区块 Bootstrap CAGR 95% CI | 判定 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## 迁移设计",
        "",
        "| 原策略 | Crypto 迁移 | 保留 | 替换 |",
        "|---|---|---|---|",
        "| 01 | 趋势轮动 | MA、动量、Top-K | ETF 池→高流动性币对 |",
        "| 02 | 全天候轮动 | 风险状态切换 | 大小盘指数/外盘 ETF→主流币/山寨币/现金 |",
        "| 03、04、06、10-A | 小盘代理轮动 | 周频、低规模倾斜、趋势门控 | 市值/审计/红利→流动性池内低成交额代理 |",
        "| 05、07、10-C/D、11、12 | 量价多因子 | 因子漏斗、Top-K、指数择时 | PEG/ROE/股息→动量、量增、波动、流动性 |",
        "| 08 | RSI 因子轮动 | 慢筛选、快退出 | 财务质量→量价质量代理 |",
        "| 09 | 滚动机器学习 | 滚动训练、横截面预测 | XGBoost 财务因子→正则化岭回归量价因子 |",
        "",
        "## 数据证据",
        "",
        f"- 来源：`{manifest.get('source', 'N/A')}`",
        f"- 周期：`{manifest.get('bar', 'N/A')}`；币对数：`{len(files)}`",
        f"- 每个 CSV 均记录 UTC 时间戳并在 `data_manifest.json` 保存 SHA-256；行数范围："
        f" `{min((item['rows'] for item in files), default=0)}`–`{max((item['rows'] for item in files), default=0)}`。",
        "",
        "## 方法",
        "",
        "1. 所有信号只读取 T-1 及更早数据，持有 T 日 close-to-close 收益。",
        "2. 前 60% 时间段执行有限网格优化，后 40% 不参与参数选择；固定币池仍含测试期末幸存者偏差。",
        "3. 每次目标权重变化按绝对权重变化计交易成本；现金收益按 0 处理。",
        "4. 对样本外日收益进行 14 日区块 Bootstrap（2,000 次），保留收益自相关后估计 CAGR 置信区间。",
        "5. 以 BTC 买入持有作为统一基准，Crypto 按 365 天年化。",
        "",
        "判定规则：优化参数与默认参数均为正，且区块 Bootstrap CAGR 95% 置信区间下界大于 0，才记为“统计通过”；",
        "仅优化参数点估计为正记为“点估计为正（无统计支持）”；其余为“未通过”。",
        "",
        "## 限制",
        "",
        *[f"- {item}" for item in results["methodology"]["known_biases"]],
        "- 原始策略包只有源码和宣称指标，没有原始 A 股逐日净值，因此不能做同口径统计显著性检验。",
        "- 当前结果是研究回测，不包含交易所 API 下单、密钥管理或自动实盘执行。",
        "",
        "详细参数及训练/测试指标见 `backtest_results.json`。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parameter_product(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """展开有限参数网格。"""

    names = list(grid)
    return [dict(zip(names, values)) for values in itertools.product(*(grid[name] for name in names))]


def _strategy_parameters(strategy: object) -> dict[str, Any]:
    """仅导出可调参数，排除名称和来源元数据。"""

    return {
        field.name: getattr(strategy, field.name)
        for field in fields(strategy)
        if field.name not in {"name", "source_ids"}
    }


def _block_bootstrap_cagr(
    returns: np.ndarray,
    block_size: int = 14,
    samples: int = 2_000,
    seed: int = 20260716,
) -> dict[str, Any]:
    """用循环区块 Bootstrap 估计复合年化收益不确定性。"""

    values = np.asarray(returns, dtype=float)
    if len(values) < block_size * 2 or np.any(values <= -1):
        raise ValueError("Bootstrap 收益样本无效或过短")
    generator = np.random.default_rng(seed)
    block_count = int(np.ceil(len(values) / block_size))
    offsets = np.arange(block_size)
    cagrs = np.empty(samples, dtype=float)
    for sample in range(samples):
        starts = generator.integers(0, len(values), size=block_count)
        indices = ((starts[:, None] + offsets) % len(values)).ravel()[: len(values)]
        annual_log_return = float(np.mean(np.log1p(values[indices])) * 365.0)
        cagrs[sample] = np.expm1(annual_log_return)
    lower, median, upper = np.quantile(cagrs, [0.025, 0.5, 0.975])
    return {
        "method": "circular_block_bootstrap",
        "block_days": block_size,
        "samples": samples,
        "cagr_ci_95": [float(lower), float(upper)],
        "cagr_median": float(median),
        "positive_resample_fraction": float(np.mean(cagrs > 0)),
    }

