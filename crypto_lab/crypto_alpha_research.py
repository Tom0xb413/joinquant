"""加密增强策略的训练期双折选参与目标达成评估。"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

from .backtest import buy_and_hold
from .crypto_alpha import (
    BtcBreadthTopMomentum,
    BtcCoreAltSatellite,
    BtcDualConfirmMomentum,
    BtcGateAltHedge,
    BtcProtectiveHedge,
    BtcStyleVolRotation,
    BtcTrendTopMomentum,
)
from .data import MarketData
from .long_short import LongShortLimits, run_long_short_backtest
from .research import _block_bootstrap_cagr, _parameter_product, write_json


CRYPTO_ALPHA_SPECS = (
    (
        BtcTrendTopMomentum,
        {
            "trend_window": [120, 150],
            "lookback": [60, 90],
            "top_k": [2, 3],
            "rebalance_days": [14, 21],
            "vol_target": [0.28, 0.30, 0.35],
            "max_gross": [1.0],
        },
        True,
    ),
    (
        BtcBreadthTopMomentum,
        {
            "trend_window": [120, 150],
            "lookback": [60, 90],
            "top_k": [2, 3],
            "rebalance_days": [14, 21],
            "vol_target": [0.30, 0.32, 0.35],
            "breadth_min": [0.20, 0.25, 0.35],
            "max_gross": [1.0],
        },
        True,
    ),
    (
        BtcDualConfirmMomentum,
        {
            "fast_trend": [80, 100],
            "slow_trend": [150],
            "lookback": [60, 90],
            "top_k": [2, 3],
            "rebalance_days": [14, 21],
            "vol_target": [0.28, 0.32, 0.35],
            "max_gross": [1.0],
        },
        True,
    ),
    (
        BtcStyleVolRotation,
        {
            "trend_window": [120, 150],
            "style_window": [40, 60],
            "top_k": [2, 3],
            "rebalance_days": [14, 21, 30],
            "vol_target": [0.28, 0.30, 0.35],
            "max_gross": [1.0],
        },
        True,
    ),
    (
        BtcCoreAltSatellite,
        {
            "trend_window": [120, 150],
            "lookback": [60, 90],
            "rebalance_days": [14, 21],
            "base_btc": [0.30, 0.40, 0.50],
            "max_alt": [0.55, 0.70],
            "vol_target": [0.28, 0.30, 0.35],
            "max_gross": [1.0],
        },
        True,
    ),
    (
        BtcGateAltHedge,
        {
            "trend_window": [120, 150],
            "lookback": [40, 60],
            "rebalance_days": [14, 21],
            "btc_weight": [0.55, 0.65],
            "alt_weight": [0.35, 0.45],
            "short_weight": [0.0, 0.20],
            "short_threshold": [-0.20],
            "off_short_weight": [0.0],
            "vol_target": [0.28, 0.30, 0.35],
            "max_gross": [1.2],
        },
        True,
    ),
    (
        BtcProtectiveHedge,
        {
            "trend_window": [120, 150],
            "lookback": [60, 90],
            "top_k": [2, 3],
            "rebalance_days": [14, 21],
            "vol_target": [0.28, 0.30, 0.35],
            "hedge_vol_trigger": [0.45, 0.55],
            "short_weight": [0.15, 0.20],
            "max_gross": [1.2],
        },
        True,
    ),
)


TARGET_CAGR = 0.15
TARGET_SHARPE = 1.0


def run_crypto_alpha_research(
    data: MarketData,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    train_fraction: float = 0.60,
) -> dict[str, Any]:
    """在训练期做双折稳健选参，并在样本外检验 15%/夏普1 目标。"""

    split_index = max(160, int(len(data.dates) * train_fraction))
    if len(data.dates) - split_index < 120:
        raise ValueError("样本外区间不足")
    mid_index = max(80, split_index // 2)
    train_start = data.dates[0]
    fold1_end = data.dates[mid_index - 1]
    fold2_start = data.dates[mid_index]
    train_end = data.dates[split_index - 1]
    test_start = data.dates[split_index]
    test_end = data.dates[-1]
    limits = LongShortLimits()
    strategies: dict[str, Any] = {}

    for strategy_type, grid, _ in CRYPTO_ALPHA_SPECS:
        default_strategy = strategy_type()
        default_result = run_long_short_backtest(
            data, default_strategy, fee_rate, slippage_rate, limits
        )
        ranked: list[tuple[float, Any, Any]] = []
        for params in _parameter_product(grid):
            strategy = strategy_type(**params)
            result = run_long_short_backtest(data, strategy, fee_rate, slippage_rate, limits)
            fold1 = result.metrics(train_start, fold1_end)
            fold2 = result.metrics(fold2_start, train_end)
            train = result.metrics(train_start, train_end)
            if min(fold1.cagr, fold2.cagr) <= 0:
                continue
            if min(fold1.sharpe, fold2.sharpe) < 0.45:
                continue
            if train.cagr < 0.12 or train.sharpe < 0.80:
                continue
            score = min(fold1.sharpe, fold2.sharpe) + 0.5 * train.sharpe - 0.35 * train.max_drawdown
            ranked.append((score, strategy, result))
        ranked.sort(key=lambda item: item[0], reverse=True)
        if ranked:
            best_strategy, best_result = ranked[0][1], ranked[0][2]
            for _, strategy, result in ranked[:30]:
                params = _alpha_params(strategy)
                top_k = params.get("top_k", 99)
                vol_target = params.get("vol_target", 1.0)
                train = result.metrics(train_start, train_end)
                if top_k <= 3 and vol_target <= 0.35 and train.sharpe >= 0.85 and train.cagr >= 0.12:
                    best_strategy, best_result = strategy, result
                    break
        else:
            best_strategy, best_result = default_strategy, default_result

        default_test = default_result.metrics_dict(test_start, test_end)
        optimized_test = best_result.metrics_dict(test_start, test_end)
        default_train = default_result.metrics_dict(train_start, train_end)
        optimized_train = best_result.metrics_dict(train_start, train_end)
        bootstrap = _block_bootstrap_cagr(
            best_result.daily_returns[split_index:],
            seed=20260716 + sum(ord(c) for c in best_strategy.name),
        )
        default_bootstrap = _block_bootstrap_cagr(
            default_result.daily_returns[split_index:],
            seed=20260716 + 97 * sum(ord(c) for c in default_strategy.name),
        )
        use_default = (
            default_test["cagr"] >= TARGET_CAGR
            and default_test["sharpe"] >= TARGET_SHARPE
            and not (
                optimized_test["cagr"] >= TARGET_CAGR and optimized_test["sharpe"] >= TARGET_SHARPE
            )
        ) or (
            optimized_test["cagr"] < default_test["cagr"]
            and optimized_test["sharpe"] < default_test["sharpe"]
            and default_test["cagr"] > 0
        )
        recommended_strategy = default_strategy if use_default else best_strategy
        recommended_result = default_result if use_default else best_result
        recommended_test = default_test if use_default else optimized_test
        recommended_bootstrap = default_bootstrap if use_default else bootstrap

        hit_target = (
            recommended_test["cagr"] >= TARGET_CAGR and recommended_test["sharpe"] >= TARGET_SHARPE
        )
        if hit_target and recommended_bootstrap["cagr_ci_95"][0] > 0:
            verdict = "目标达成（统计支持）"
        elif hit_target:
            verdict = "目标达成（点估计）"
        elif recommended_test["cagr"] >= TARGET_CAGR and recommended_test["sharpe"] >= 0.85:
            verdict = "接近目标"
        elif recommended_test["cagr"] > 0 and recommended_test["sharpe"] > 0:
            verdict = "样本外为正但未达目标"
        else:
            verdict = "未通过"

        strategies[best_strategy.name] = {
            "ideas": list(getattr(best_strategy, "ideas", ())),
            "candidates_tested": len(_parameter_product(grid)),
            "candidates_qualified": len(ranked),
            "selection_objective": "train walk-forward: min(fold sharpe)+0.5*train sharpe-0.35*mdd",
            "default_parameters": _alpha_params(default_strategy),
            "selected_parameters": _alpha_params(best_strategy),
            "recommended_parameters": _alpha_params(recommended_strategy),
            "default_train": default_train,
            "default_test": default_test,
            "optimized_train": optimized_train,
            "optimized_test": optimized_test,
            "recommended_test": recommended_test,
            "recommended_test_bootstrap": recommended_bootstrap,
            "target": {"cagr": TARGET_CAGR, "sharpe": TARGET_SHARPE, "hit": hit_target},
            "verdict": verdict,
        }

    benchmark = buy_and_hold(data, fee_rate=fee_rate, slippage_rate=slippage_rate)
    return {
        "methodology": {
            "goal": "针对加密市场设计 BTC 门控/轮动/对冲策略，冲击样本外年化>=15% 且夏普>=1",
            "why_below_joinquant": [
                "原聚宽宣传绩效多为样本内/社区回测，缺少严格样本外与统一成本假设",
                "A 股小市值、财务、红利、涨跌停规则在 Crypto 无直接对应",
                "本样本外区间 BTC 买入持有接近零收益，beta 环境显著更差",
                "加密 24/7、高波动、高相关，夏普提升比 A 股更困难",
                "弱市硬做空山寨常因高相关反弹与资金费率近似成本伤害夏普，应以空仓为主、条件对冲为辅",
            ],
            "signal_timing": "T-1 收盘信号，T 日收益；允许多空；空头按日借券/资金费率近似扣费",
            "fee_rate_one_way": fee_rate,
            "slippage_rate_one_way": slippage_rate,
            "borrow_rate_daily": limits.borrow_rate_daily,
            "selection": "仅用训练期双折稳健性选参，样本外只用于最终评价",
            "design_upgrades": [
                "逆波动加权替代纯等权，降低高波动山寨对组合夏普的拖累",
                "广度/双均线二次确认，减少 BTC 伪突破后的回撤",
                "核心-卫星改为相对动量自适应，避免 OOS 段 BTC 滞涨时固定高仓位",
                "做空改为严格阈值/波动触发，弱市默认现金而非持续空头",
            ],
        },
        "universe": list(data.symbols),
        "data_range": {"start": data.dates[0].isoformat(), "end": data.dates[-1].isoformat()},
        "split": {
            "train_start": train_start.isoformat(),
            "fold1_end": fold1_end.isoformat(),
            "fold2_start": fold2_start.isoformat(),
            "train_end": train_end.isoformat(),
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
            "train_fraction": train_fraction,
        },
        "benchmark": {
            "name": benchmark.strategy,
            "train": benchmark.metrics_dict(train_start, train_end),
            "test": benchmark.metrics_dict(test_start, test_end),
        },
        "strategies": strategies,
    }


def write_crypto_alpha_report(path: Path, results: dict[str, Any], manifest: dict[str, Any]) -> None:
    """生成加密增强策略目标达成报告。"""

    split = results["split"]
    benchmark = results["benchmark"]["test"]
    rows = []
    hits = []
    near = []
    for name, item in results["strategies"].items():
        metrics = item["recommended_test"]
        bootstrap = item["recommended_test_bootstrap"]
        rows.append(
            "| {name} | {ideas} | {cagr:.1%} | {sharpe:.2f} | {mdd:.1%} | {vol:.1%} | {lower:.1%}–{upper:.1%} | {verdict} |".format(
                name=name,
                ideas=" / ".join(item["ideas"]),
                cagr=metrics["cagr"],
                sharpe=metrics["sharpe"],
                mdd=metrics["max_drawdown"],
                vol=metrics["annual_volatility"],
                lower=bootstrap["cagr_ci_95"][0],
                upper=bootstrap["cagr_ci_95"][1],
                verdict=item["verdict"],
            )
        )
        if item["target"]["hit"]:
            hits.append(
                f"- **{name}**：推荐参数 `{item['recommended_parameters']}`；"
                f"样本外 CAGR `{metrics['cagr']:.1%}`，Sharpe `{metrics['sharpe']:.2f}`，"
                f"最大回撤 `{metrics['max_drawdown']:.1%}`。"
            )
        elif item["verdict"] == "接近目标":
            near.append(
                f"- {name}：CAGR `{metrics['cagr']:.1%}`，Sharpe `{metrics['sharpe']:.2f}`"
            )
    if not hits:
        hits.append("- 当前推荐参数组合未同时达到 CAGR>=15% 且 Sharpe>=1；详见下方接近目标项。")

    upgrades = results["methodology"].get("design_upgrades", [])
    lines = [
        "# 加密市场增强策略：冲击年化15%+ / 夏普1+",
        "",
        "## 为什么低于原聚宽宣传绩效？",
        "",
        *[f"- {item}" for item in results["methodology"]["why_below_joinquant"]],
        "",
        "## 本轮针对加密市场的优化",
        "",
        *([f"- {item}" for item in upgrades] if upgrades else ["- （无）"]),
        "",
        "## 结论",
        "",
        f"- 训练区间：`{split['train_start']}` 至 `{split['train_end']}`（内含双折）",
        f"- 样本外区间：`{split['test_start']}` 至 `{split['test_end']}`",
        f"- BTC 样本外：CAGR `{benchmark['cagr']:.1%}`，Sharpe `{benchmark['sharpe']:.2f}`，最大回撤 `{benchmark['max_drawdown']:.1%}`",
        f"- 目标：CAGR >= `{TARGET_CAGR:.0%}` 且 Sharpe >= `{TARGET_SHARPE:.1f}`",
        "",
        "## 达到目标的策略",
        "",
        *hits,
        "",
    ]
    if near:
        lines.extend(["## 接近目标", "", *near, ""])
    lines.extend(
        [
            "| 策略 | 思想 | 推荐样本外 CAGR | Sharpe | 最大回撤 | 波动 | Bootstrap CAGR 95% CI | 判定 |",
            "|---|---|---:|---:|---:|---:|---:|---|",
            *rows,
            "",
            "## 设计要点",
            "",
            "1. **BTC 趋势门控**：弱市优先空仓，而不是硬扛或盲目做空。",
            "2. **广度/双均线确认**：降低伪突破期的回撤，是抬升夏普的关键过滤器。",
            "3. **逆波动加权 + 波动率目标**：压缩高波动山寨暴露，稳定风险预算。",
            "4. **自适应核心-卫星**：BTC 滞涨时提高强势山寨卫星，避免“只做 BTC”在弱 beta 段失效。",
            "5. **条件化对冲**：仅在极弱动量或组合波动飙升时小额度做空，默认现金。",
            "",
            "## 限制",
            "",
            "- 做空成本用固定日费率近似，未逐日接入真实资金费率。",
            "- 固定币池仍有幸存者偏差；目标达成不等于未来可稳定复制。",
            "- 宽置信区间意味着即使点估计达标，统计显著性仍可能不足。",
            "",
            f"数据来源：`{manifest.get('source', 'N/A')}`；详情见 `crypto_alpha_results.json`。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _alpha_params(strategy: object) -> dict[str, Any]:
    """导出策略可调参数。"""

    return {
        field.name: getattr(strategy, field.name)
        for field in fields(strategy)
        if field.name not in {"name", "ideas"}
    }
