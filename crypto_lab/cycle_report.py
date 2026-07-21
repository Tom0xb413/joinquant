"""2021–2026 全周期与不同 beta 分段的详细回测报告。"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .backtest import BacktestResult, PerformanceMetrics, buy_and_hold
from .crypto_alpha import crypto_alpha_catalog
from .data import MarketData
from .long_short import LongShortLimits, run_long_short_backtest
from .research import write_json


# 加密市场宏观 beta 分段：按 BTC 周期高低点与主流叙事划分（非样本内优化结果）。
BETA_REGIMES: tuple[dict[str, Any], ...] = (
    {
        "id": "full_cycle",
        "name": "全周期 2021-2026",
        "start": date(2021, 1, 1),
        "end": date(2026, 7, 15),
        "beta": "mixed",
        "note": "完整一轮牛熊震荡",
    },
    {
        "id": "bull_2021",
        "name": "2021 牛市",
        "start": date(2021, 1, 1),
        "end": date(2021, 11, 10),
        "beta": "bull",
        "note": "至本轮高点附近",
    },
    {
        "id": "bear_2021_2022",
        "name": "2021-2022 熊市",
        "start": date(2021, 11, 11),
        "end": date(2022, 11, 21),
        "beta": "bear",
        "note": "见顶至周期底部",
    },
    {
        "id": "recovery_2023_2024",
        "name": "2023-2024 复苏/ETF牛",
        "start": date(2022, 11, 22),
        "end": date(2024, 3, 13),
        "beta": "bull",
        "note": "底部反弹至现货 ETF 落地前后",
    },
    {
        "id": "bull_2024_2025",
        "name": "2024-2025 主升浪",
        "start": date(2024, 3, 14),
        "end": date(2025, 10, 6),
        "beta": "bull",
        "note": "ETF 后至数据内 ATH",
    },
    {
        "id": "correction_2025_2026",
        "name": "2025-2026 回调",
        "start": date(2025, 10, 7),
        "end": date(2026, 7, 15),
        "beta": "bear",
        "note": "ATH 后的深度回调段",
    },
    {
        "id": "year_2021",
        "name": "日历年 2021",
        "start": date(2021, 1, 1),
        "end": date(2021, 12, 31),
        "beta": "mixed",
        "note": "年化切片",
    },
    {
        "id": "year_2022",
        "name": "日历年 2022",
        "start": date(2022, 1, 1),
        "end": date(2022, 12, 31),
        "beta": "bear",
        "note": "年化切片",
    },
    {
        "id": "year_2023",
        "name": "日历年 2023",
        "start": date(2023, 1, 1),
        "end": date(2023, 12, 31),
        "beta": "bull",
        "note": "年化切片",
    },
    {
        "id": "year_2024",
        "name": "日历年 2024",
        "start": date(2024, 1, 1),
        "end": date(2024, 12, 31),
        "beta": "bull",
        "note": "年化切片",
    },
    {
        "id": "year_2025",
        "name": "日历年 2025",
        "start": date(2025, 1, 1),
        "end": date(2025, 12, 31),
        "beta": "mixed",
        "note": "年化切片",
    },
    {
        "id": "year_2026ytd",
        "name": "日历年 2026YTD",
        "start": date(2026, 1, 1),
        "end": date(2026, 7, 15),
        "beta": "bear",
        "note": "年化切片（截至数据末日）",
    },
)


# 研究阶段推荐参数（与 reports/crypto_alpha_results.json 对齐）。
RECOMMENDED_PARAMS: dict[str, dict[str, Any]] = {
    "btc_trend_top_momentum": {
        "trend_window": 150,
        "lookback": 90,
        "top_k": 2,
        "rebalance_days": 14,
        "vol_target": 0.3,
        "max_gross": 1.0,
    },
    "btc_breadth_top_momentum": {
        "trend_window": 150,
        "lookback": 90,
        "top_k": 2,
        "rebalance_days": 14,
        "vol_target": 0.32,
        "breadth_min": 0.25,
        "max_gross": 1.0,
    },
    "btc_dual_confirm_momentum": {
        "fast_trend": 100,
        "slow_trend": 150,
        "lookback": 90,
        "top_k": 3,
        "rebalance_days": 14,
        "vol_target": 0.32,
        "max_gross": 1.0,
    },
    "btc_style_vol_rotation": {
        "trend_window": 150,
        "style_window": 60,
        "top_k": 2,
        "rebalance_days": 21,
        "vol_target": 0.3,
        "max_gross": 1.0,
    },
    "btc_core_alt_satellite": {
        "trend_window": 150,
        "lookback": 90,
        "rebalance_days": 14,
        "base_btc": 0.4,
        "max_alt": 0.7,
        "vol_target": 0.3,
        "max_gross": 1.0,
    },
    "btc_gate_alt_hedge": {
        "trend_window": 150,
        "lookback": 60,
        "rebalance_days": 14,
        "btc_weight": 0.55,
        "alt_weight": 0.45,
        "short_weight": 0.2,
        "short_threshold": -0.2,
        "off_short_weight": 0.0,
        "vol_target": 0.3,
        "max_gross": 1.2,
    },
    "btc_protective_hedge": {
        "trend_window": 150,
        "lookback": 90,
        "top_k": 2,
        "rebalance_days": 14,
        "vol_target": 0.3,
        "hedge_vol_trigger": 0.55,
        "short_weight": 0.2,
        "max_gross": 1.2,
    },
}


@dataclass(frozen=True)
class TradeRecord:
    """单笔调仓记录（由相邻交易日目标权重差分得到）。"""

    date: str
    strategy: str
    symbol: str
    side: str
    weight_from: float
    weight_to: float
    weight_delta: float
    price: float
    turnover_contribution: float
    equity_before: float


def extract_trades(
    result: BacktestResult,
    data: MarketData,
    min_delta: float = 0.01,
) -> list[TradeRecord]:
    """从日度权重序列提取调仓日成交；仅在引擎记到换手时输出，忽略纯漂移。"""

    trades: list[TradeRecord] = []
    previous = np.zeros(len(data.symbols), dtype=float)
    for index in range(len(result.dates)):
        current = np.asarray(result.weights[index], dtype=float)
        traded_today = float(result.turnover[index]) >= min_delta
        if traded_today:
            deltas = current - previous
            for column, delta in enumerate(deltas):
                if abs(delta) < min_delta:
                    continue
                if delta > 0:
                    side = "buy" if current[column] >= 0 and previous[column] >= 0 else "cover_or_buy"
                    if previous[column] < 0 < current[column]:
                        side = "cover_and_buy"
                    elif previous[column] < 0 and current[column] <= 0:
                        side = "cover"
                else:
                    side = "sell" if current[column] >= 0 and previous[column] >= 0 else "short_or_sell"
                    if previous[column] > 0 > current[column]:
                        side = "sell_and_short"
                    elif previous[column] <= 0 and current[column] < 0:
                        side = "short"
                trades.append(
                    TradeRecord(
                        date=result.dates[index].isoformat(),
                        strategy=result.strategy,
                        symbol=data.symbols[column],
                        side=side,
                        weight_from=float(previous[column]),
                        weight_to=float(current[column]),
                        weight_delta=float(delta),
                        price=float(data.close[index, column]),
                        turnover_contribution=float(abs(delta)),
                        equity_before=float(result.equity[index - 1]) if index > 0 else 1.0,
                    )
                )
        previous = current
    return trades


def drawdown_series(equity: np.ndarray) -> np.ndarray:
    """计算相对历史峰值的回撤序列（负值）。"""

    peaks = np.maximum.accumulate(np.maximum(equity, 1e-12))
    return equity / peaks - 1.0


def regime_beta_label(data: MarketData, start: date, end: date, window: int = 200) -> str:
    """用区间内 BTC 相对均线占比粗分 bull/bear/mixed。"""

    btc = data.symbol_index("BTC-USDT")
    indices = [i for i, day in enumerate(data.dates) if start <= day <= end]
    if len(indices) < window + 5:
        return "mixed"
    above = 0
    total = 0
    for index in indices:
        if index < window:
            continue
        mean = float(np.mean(data.close[index - window + 1 : index + 1, btc]))
        above += int(data.close[index, btc] > mean)
        total += 1
    if total == 0:
        return "mixed"
    ratio = above / total
    if ratio >= 0.65:
        return "bull"
    if ratio <= 0.35:
        return "bear"
    return "mixed"


def build_strategies() -> dict[str, Any]:
    """按推荐参数实例化策略目录。"""

    catalog = crypto_alpha_catalog()
    strategies = {}
    for name, params in RECOMMENDED_PARAMS.items():
        strategies[name] = catalog[name](**params)
    return strategies


def _metrics_with_excess(
    strategy_metrics: PerformanceMetrics,
    benchmark_metrics: PerformanceMetrics,
) -> dict[str, Any]:
    """附加相对 BTC 的超额收益指标。"""

    payload = asdict(strategy_metrics)
    payload["excess_cagr"] = strategy_metrics.cagr - benchmark_metrics.cagr
    payload["excess_total_return"] = strategy_metrics.total_return - benchmark_metrics.total_return
    payload["mdd_improvement"] = benchmark_metrics.max_drawdown - strategy_metrics.max_drawdown
    payload["beat_btc_cagr"] = strategy_metrics.cagr > benchmark_metrics.cagr
    payload["beat_btc_sharpe"] = strategy_metrics.sharpe > benchmark_metrics.sharpe
    return payload


def run_cycle_report(
    data: MarketData,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    trade_min_delta: float = 0.01,
) -> dict[str, Any]:
    """运行全周期回测，按 beta/年分段汇总，并提取交易记录。"""

    strategies = build_strategies()
    limits = LongShortLimits()
    benchmark = buy_and_hold(data, fee_rate=fee_rate, slippage_rate=slippage_rate)
    results: dict[str, BacktestResult] = {"BTC-USDT_buy_hold": benchmark}
    trades_by_strategy: dict[str, list[dict[str, Any]]] = {
        "BTC-USDT_buy_hold": [asdict(trade) for trade in extract_trades(benchmark, data, trade_min_delta)]
    }

    for name, strategy in strategies.items():
        result = run_long_short_backtest(data, strategy, fee_rate, slippage_rate, limits)
        results[name] = result
        trades_by_strategy[name] = [asdict(trade) for trade in extract_trades(result, data, trade_min_delta)]

    regime_rows: list[dict[str, Any]] = []
    for regime in BETA_REGIMES:
        start, end = regime["start"], regime["end"]
        if end < data.dates[0] or start > data.dates[-1]:
            continue
        clipped_start = max(start, data.dates[0])
        clipped_end = min(end, data.dates[-1])
        if clipped_start >= clipped_end:
            continue
        try:
            bench_metrics = benchmark.metrics(clipped_start, clipped_end)
        except ValueError:
            continue
        measured_beta = regime_beta_label(data, clipped_start, clipped_end)
        row: dict[str, Any] = {
            "id": regime["id"],
            "name": regime["name"],
            "start": clipped_start.isoformat(),
            "end": clipped_end.isoformat(),
            "declared_beta": regime["beta"],
            "measured_beta_200dma": measured_beta,
            "note": regime["note"],
            "benchmark": asdict(bench_metrics),
            "strategies": {},
        }
        for name, result in results.items():
            if name == "BTC-USDT_buy_hold":
                continue
            metrics = result.metrics(clipped_start, clipped_end)
            row["strategies"][name] = _metrics_with_excess(metrics, bench_metrics)
        regime_rows.append(row)

    # 风险开/关聚合：BTC 是否站上 200 日均线
    risk_on_mask, risk_off_mask = _risk_masks(data, window=200)
    beta_state_summary = {
        "risk_on": _mask_segment_summary(results, benchmark, risk_on_mask, "risk_on_btc_above_200dma"),
        "risk_off": _mask_segment_summary(results, benchmark, risk_off_mask, "risk_off_btc_below_200dma"),
    }

    equity_curves = {
        name: {
            "dates": [day.isoformat() for day in result.dates],
            "equity": [float(value) for value in result.equity],
            "drawdown": [float(value) for value in drawdown_series(result.equity)],
        }
        for name, result in results.items()
    }

    full_start, full_end = data.dates[0], data.dates[-1]
    ranking = []
    for name, result in results.items():
        metrics = result.metrics(full_start, full_end)
        ranking.append(
            {
                "strategy": name,
                "metrics": asdict(metrics),
                "trade_count": len(trades_by_strategy[name]),
                "avg_abs_exposure": float(np.mean(np.sum(np.abs(result.weights), axis=1))),
                "time_in_market": float(np.mean(np.sum(np.abs(result.weights), axis=1) > 1e-6)),
            }
        )
    ranking.sort(key=lambda item: item["metrics"]["sharpe"], reverse=True)

    return {
        "methodology": {
            "period": f"{full_start.isoformat()} ~ {full_end.isoformat()}",
            "signal_timing": "T-1 收盘信号，T 日收益；多空引擎；含手续费/滑点/空头借券近似",
            "fee_rate_one_way": fee_rate,
            "slippage_rate_one_way": slippage_rate,
            "borrow_rate_daily": limits.borrow_rate_daily,
            "parameters": "使用 crypto_alpha 研究推荐参数，本报告不再二次选参",
            "regimes": "宏观 beta 分段 + 日历年 + BTC 200DMA 风险开关聚合",
            "trades": f"相邻交易日权重变化绝对值 >= {trade_min_delta} 记为一笔调仓",
        },
        "universe": list(data.symbols),
        "recommended_parameters": RECOMMENDED_PARAMS,
        "ranking_full_cycle": ranking,
        "regimes": regime_rows,
        "beta_state_summary": beta_state_summary,
        "trades": trades_by_strategy,
        "equity_curves": equity_curves,
    }


def _risk_masks(data: MarketData, window: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """生成 BTC 相对 200 日均线的风险开/关布尔掩码。"""

    btc = data.symbol_index("BTC-USDT")
    on = np.zeros(len(data.dates), dtype=bool)
    off = np.zeros(len(data.dates), dtype=bool)
    for index in range(len(data.dates)):
        if index + 1 < window:
            continue
        mean = float(np.mean(data.close[index - window + 1 : index + 1, btc]))
        if data.close[index, btc] > mean:
            on[index] = True
        else:
            off[index] = True
    return on, off


def _mask_segment_summary(
    results: dict[str, BacktestResult],
    benchmark: BacktestResult,
    mask: np.ndarray,
    label: str,
) -> dict[str, Any]:
    """在非连续掩码日期上，用日收益拼接估算该 beta 状态下的表现。"""

    if int(mask.sum()) < 30:
        return {"label": label, "observations": int(mask.sum()), "strategies": {}}
    summary: dict[str, Any] = {
        "label": label,
        "observations": int(mask.sum()),
        "strategies": {},
    }
    bench_returns = benchmark.daily_returns[mask]
    bench_metrics = _metrics_from_returns(bench_returns, label)
    summary["benchmark"] = bench_metrics
    for name, result in results.items():
        if name == "BTC-USDT_buy_hold":
            continue
        metrics = _metrics_from_returns(result.daily_returns[mask], label)
        summary["strategies"][name] = {
            **metrics,
            "excess_cagr": metrics["cagr"] - bench_metrics["cagr"],
            "excess_total_return": metrics["total_return"] - bench_metrics["total_return"],
            "mdd_improvement": bench_metrics["max_drawdown"] - metrics["max_drawdown"],
            "beat_btc_cagr": metrics["cagr"] > bench_metrics["cagr"],
            "beat_btc_sharpe": metrics["sharpe"] > bench_metrics["sharpe"],
        }
    return summary


def _metrics_from_returns(returns: np.ndarray, label: str) -> dict[str, Any]:
    """由日收益序列计算核心指标（用于非连续 beta 状态聚合）。"""

    if len(returns) < 2:
        raise ValueError("收益序列过短")
    equity = np.cumprod(1.0 + returns)
    years = len(returns) / 365.0
    total_return = float(equity[-1] - 1.0)
    cagr = float(equity[-1] ** (1.0 / years) - 1.0) if equity[-1] > 0 else -1.0
    volatility = float(np.std(returns, ddof=1) * np.sqrt(365.0))
    sharpe = float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(365.0)) if volatility else 0.0
    anchored = np.concatenate(([1.0], equity))
    peaks = np.maximum.accumulate(anchored)
    max_drawdown = float(-np.min(anchored / peaks - 1.0))
    calmar = cagr / max_drawdown if max_drawdown > 1e-12 else 0.0
    return {
        "start": label,
        "end": label,
        "observations": int(len(returns)),
        "total_return": total_return,
        "cagr": cagr,
        "annual_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "turnover": None,
        "cost_paid": None,
    }


def save_trade_csvs(trades: dict[str, list[dict[str, Any]]], output_dir: Path) -> list[Path]:
    """将各策略交易记录写为 CSV。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    fieldnames = [
        "date",
        "strategy",
        "symbol",
        "side",
        "weight_from",
        "weight_to",
        "weight_delta",
        "price",
        "turnover_contribution",
        "equity_before",
    ]
    for name, rows in trades.items():
        path = output_dir / f"trades_{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        paths.append(path)
    return paths


def plot_cycle_charts(report: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    """绘制全周期权益曲线、回撤曲线与分段超额柱状图。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    curves = report["equity_curves"]
    # 重点策略：达标项 + BTC
    highlight = [
        "BTC-USDT_buy_hold",
        "btc_breadth_top_momentum",
        "btc_dual_confirm_momentum",
        "btc_trend_top_momentum",
        "btc_protective_hedge",
        "btc_core_alt_satellite",
    ]
    colors = {
        "BTC-USDT_buy_hold": "#111111",
        "btc_breadth_top_momentum": "#0B6E4F",
        "btc_dual_confirm_momentum": "#1B4F72",
        "btc_trend_top_momentum": "#B9770E",
        "btc_protective_hedge": "#6C3483",
        "btc_core_alt_satellite": "#922B21",
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    for name in highlight:
        curve = curves[name]
        dates = np.arange(len(curve["equity"]))
        ax.plot(
            dates,
            curve["equity"],
            label=name.replace("_", " "),
            color=colors.get(name),
            linewidth=2.0 if name != "BTC-USDT_buy_hold" else 2.4,
            alpha=0.95,
        )
    ax.set_yscale("log")
    ax.set_title("2021-2026 Full-Cycle Equity (log scale)")
    ax.set_xlabel("Trading day index")
    ax.set_ylabel("Equity (start=1)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    equity_path = output_dir / "equity_full_cycle.png"
    fig.tight_layout()
    fig.savefig(equity_path, dpi=140)
    plt.close(fig)
    paths["equity"] = equity_path

    fig, ax = plt.subplots(figsize=(12, 5))
    for name in highlight:
        curve = curves[name]
        ax.plot(
            np.arange(len(curve["drawdown"])),
            curve["drawdown"],
            label=name.replace("_", " "),
            color=colors.get(name),
            linewidth=1.6,
        )
    ax.set_title("2021-2026 Drawdown")
    ax.set_xlabel("Trading day index")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=8)
    dd_path = output_dir / "drawdown_full_cycle.png"
    fig.tight_layout()
    fig.savefig(dd_path, dpi=140)
    plt.close(fig)
    paths["drawdown"] = dd_path

    # 分段超额 CAGR：breadth vs BTC（图表用英文标签，避免缺字）
    regime_ids = []
    strategy_excess = []
    btc_cagr = []
    focus = "btc_breadth_top_momentum"
    label_map = {
        "full_cycle": "Full cycle",
        "bull_2021": "Bull 2021",
        "bear_2021_2022": "Bear 21-22",
        "recovery_2023_2024": "Recovery 23-24",
        "bull_2024_2025": "Bull 24-25",
        "correction_2025_2026": "Correction 25-26",
    }
    for regime in report["regimes"]:
        if regime["id"].startswith("year_"):
            continue
        regime_ids.append(label_map.get(regime["id"], regime["id"]))
        btc_cagr.append(regime["benchmark"]["cagr"] * 100)
        strategy_excess.append(regime["strategies"][focus]["cagr"] * 100)
    x = np.arange(len(regime_ids))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, btc_cagr, width, label="BTC buy&hold CAGR%", color="#333333")
    ax.bar(x + width / 2, strategy_excess, width, label="breadth top mom CAGR%", color="#0B6E4F")
    ax.set_xticks(x)
    ax.set_xticklabels(regime_ids, rotation=20, ha="right")
    ax.set_ylabel("CAGR %")
    ax.set_title("Beta-Regime CAGR: Breadth Momentum vs BTC")
    ax.axhline(0, color="#666666", linewidth=1)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    regime_path = output_dir / "regime_cagr_vs_btc.png"
    fig.tight_layout()
    fig.savefig(regime_path, dpi=140)
    plt.close(fig)
    paths["regime_cagr"] = regime_path

    # 全策略全周期雷达式条形：Sharpe / MDD
    ranking = [item for item in report["ranking_full_cycle"] if item["strategy"] != "BTC-USDT_buy_hold"]
    names = [item["strategy"] for item in ranking]
    sharpes = [item["metrics"]["sharpe"] for item in ranking]
    mdds = [item["metrics"]["max_drawdown"] * 100 for item in ranking]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(names, sharpes, color="#1B4F72")
    axes[0].axvline(1.0, color="#C0392B", linestyle="--", linewidth=1)
    axes[0].set_title("Full-cycle Sharpe")
    axes[1].barh(names, mdds, color="#922B21")
    btc_mdd = next(
        item["metrics"]["max_drawdown"] * 100
        for item in report["ranking_full_cycle"]
        if item["strategy"] == "BTC-USDT_buy_hold"
    )
    axes[1].axvline(btc_mdd, color="#111111", linestyle="--", linewidth=1, label="BTC MDD")
    axes[1].set_title("Full-cycle Max Drawdown %")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(True, axis="x", alpha=0.25)
    compare_path = output_dir / "full_cycle_sharpe_mdd.png"
    fig.tight_layout()
    fig.savefig(compare_path, dpi=140)
    plt.close(fig)
    paths["sharpe_mdd"] = compare_path
    return paths


def write_cycle_markdown(
    path: Path,
    report: dict[str, Any],
    chart_paths: dict[str, Path],
    trade_dir: Path,
) -> None:
    """写出含图表引用、分段对比与交易摘要的 Markdown 报告。"""

    ranking = report["ranking_full_cycle"]
    btc = next(item for item in ranking if item["strategy"] == "BTC-USDT_buy_hold")
    lines: list[str] = [
        "# 加密策略全周期回测报告（2021–2026）",
        "",
        "## 范围与方法",
        "",
        f"- 区间：`{report['methodology']['period']}`",
        f"- {report['methodology']['signal_timing']}",
        f"- 单向费率 `{report['methodology']['fee_rate_one_way']}`，滑点 `{report['methodology']['slippage_rate_one_way']}`，"
        f"空头日借券 `{report['methodology']['borrow_rate_daily']}`",
        f"- {report['methodology']['parameters']}",
        f"- 分段：{report['methodology']['regimes']}",
        f"- 交易记录：{report['methodology']['trades']}；CSV 目录 `{trade_dir.as_posix()}`",
        "",
        "## 全周期总览（对比 BTC 买入持有）",
        "",
        (
            f"- BTC 买入持有：CAGR `{btc['metrics']['cagr']:.1%}`，Sharpe `{btc['metrics']['sharpe']:.2f}`，"
            f"最大回撤 `{btc['metrics']['max_drawdown']:.1%}`，总收益 `{btc['metrics']['total_return']:.1%}`"
        ),
        "",
        "| 策略 | 总收益 | CAGR | Sharpe | 最大回撤 | 波动 | 成交次数 | 在市时间 | vs BTC超额CAGR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in ranking:
        metrics = item["metrics"]
        excess = metrics["cagr"] - btc["metrics"]["cagr"]
        lines.append(
            "| {name} | {tr:.1%} | {cagr:.1%} | {sharpe:.2f} | {mdd:.1%} | {vol:.1%} | {trades} | {tim:.0%} | {excess:+.1%} |".format(
                name=item["strategy"],
                tr=metrics["total_return"],
                cagr=metrics["cagr"],
                sharpe=metrics["sharpe"],
                mdd=metrics["max_drawdown"],
                vol=metrics["annual_volatility"],
                trades=item["trade_count"],
                tim=item["time_in_market"],
                excess=excess,
            )
        )

    lines.extend(
        [
            "",
            "## 权益曲线与回撤",
            "",
            f"![全周期权益曲线]({_rel(path, chart_paths['equity'])})",
            "",
            f"![全周期回撤]({_rel(path, chart_paths['drawdown'])})",
            "",
            f"![全周期夏普与回撤对比]({_rel(path, chart_paths['sharpe_mdd'])})",
            "",
            "## 宏观 Beta 分段表现",
            "",
            f"![分段 CAGR 对比]({_rel(path, chart_paths['regime_cagr'])})",
            "",
        ]
    )

    for regime in report["regimes"]:
        if regime["id"].startswith("year_"):
            continue
        bench = regime["benchmark"]
        lines.extend(
            [
                f"### {regime['name']}（{regime['start']} ~ {regime['end']}）",
                "",
                f"- 声明 beta：`{regime['declared_beta']}`；实测 200DMA beta：`{regime['measured_beta_200dma']}`",
                f"- 说明：{regime['note']}",
                (
                    f"- BTC：CAGR `{bench['cagr']:.1%}`，Sharpe `{bench['sharpe']:.2f}`，"
                    f"最大回撤 `{bench['max_drawdown']:.1%}`，总收益 `{bench['total_return']:.1%}`"
                ),
                "",
                "| 策略 | CAGR | Sharpe | 最大回撤 | 总收益 | 超额CAGR | 回撤改善 | 胜BTC收益 | 胜BTC夏普 |",
                "|---|---:|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        ordered = sorted(
            regime["strategies"].items(),
            key=lambda item: item[1]["sharpe"],
            reverse=True,
        )
        for name, metrics in ordered:
            lines.append(
                "| {name} | {cagr:.1%} | {sharpe:.2f} | {mdd:.1%} | {tr:.1%} | {ex:+.1%} | {mddi:+.1%} | {beat_c} | {beat_s} |".format(
                    name=name,
                    cagr=metrics["cagr"],
                    sharpe=metrics["sharpe"],
                    mdd=metrics["max_drawdown"],
                    tr=metrics["total_return"],
                    ex=metrics["excess_cagr"],
                    mddi=metrics["mdd_improvement"],
                    beat_c="是" if metrics["beat_btc_cagr"] else "否",
                    beat_s="是" if metrics["beat_btc_sharpe"] else "否",
                )
            )
        lines.append("")

    lines.extend(["## 日历年切片", ""])
    year_regimes = [regime for regime in report["regimes"] if regime["id"].startswith("year_")]
    lines.extend(
        [
            "| 年份 | BTC CAGR | BTC Sharpe | BTC MDD | 最佳策略 | 策略CAGR | 策略Sharpe | 策略MDD |",
            "|---|---:|---:|---:|---|---:|---:|---:|",
        ]
    )
    for regime in year_regimes:
        best_name, best = max(regime["strategies"].items(), key=lambda item: item[1]["sharpe"])
        bench = regime["benchmark"]
        lines.append(
            "| {year} | {bc:.1%} | {bs:.2f} | {bm:.1%} | {name} | {sc:.1%} | {ss:.2f} | {sm:.1%} |".format(
                year=regime["name"],
                bc=bench["cagr"],
                bs=bench["sharpe"],
                bm=bench["max_drawdown"],
                name=best_name,
                sc=best["cagr"],
                ss=best["sharpe"],
                sm=best["max_drawdown"],
            )
        )

    lines.extend(["", "## BTC 200DMA 风险开关聚合", ""])
    for key in ("risk_on", "risk_off"):
        block = report["beta_state_summary"][key]
        lines.append(f"### {block['label']}（观测 {block['observations']} 天）")
        lines.append("")
        if "benchmark" not in block:
            lines.append("- 样本不足")
            lines.append("")
            continue
        bench = block["benchmark"]
        lines.append(
            f"- BTC：CAGR `{bench['cagr']:.1%}`，Sharpe `{bench['sharpe']:.2f}`，最大回撤 `{bench['max_drawdown']:.1%}`"
        )
        lines.extend(
            [
                "",
                "| 策略 | CAGR | Sharpe | 最大回撤 | 超额CAGR |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        ordered = sorted(block["strategies"].items(), key=lambda item: item[1]["sharpe"], reverse=True)
        for name, metrics in ordered:
            lines.append(
                "| {name} | {cagr:.1%} | {sharpe:.2f} | {mdd:.1%} | {ex:+.1%} |".format(
                    name=name,
                    cagr=metrics["cagr"],
                    sharpe=metrics["sharpe"],
                    mdd=metrics["max_drawdown"],
                    ex=metrics["excess_cagr"],
                )
            )
        lines.append("")

    lines.extend(["## 交易记录摘要", ""])
    for name, rows in report["trades"].items():
        if name == "BTC-USDT_buy_hold":
            continue
        buys = sum(1 for row in rows if "buy" in row["side"] or row["side"] == "cover")
        sells = sum(1 for row in rows if "sell" in row["side"] or "short" in row["side"])
        symbols = sorted({row["symbol"] for row in rows})
        sample = rows[:8]
        lines.append(f"### {name}")
        lines.append("")
        lines.append(
            f"- 调仓笔数 `{len(rows)}`；偏多侧动作约 `{buys}`，偏空/减仓动作约 `{sells}`；涉及标的：{', '.join(symbols)}"
        )
        lines.append(f"- 完整记录：`{(trade_dir / f'trades_{name}.csv').as_posix()}`")
        if sample:
            lines.extend(
                [
                    "",
                    "| 日期 | 标的 | 方向 | 权重前 | 权重后 | 价格 |",
                    "|---|---|---|---:|---:|---:|",
                ]
            )
            for row in sample:
                lines.append(
                    "| {date} | {symbol} | {side} | {wf:.2f} | {wt:.2f} | {price:.2f} |".format(
                        date=row["date"],
                        symbol=row["symbol"],
                        side=row["side"],
                        wf=row["weight_from"],
                        wt=row["weight_to"],
                        price=row["price"],
                    )
                )
            if len(rows) > 8:
                lines.append("")
                lines.append(f"- … 其余 {len(rows) - 8} 笔见 CSV")
        lines.append("")

    lines.extend(
        [
            "## 结论要点",
            "",
            "- 全周期应同时看 CAGR、Sharpe 与最大回撤；单纯追高收益在熊市段会被打回。",
            "- BTC 门控类策略的核心价值通常体现在熊市/风险关阶段的回撤控制，而非每个牛市都跑赢 BTC。",
            "- 交易记录显示多数增强策略以低频调仓 + 大量现金期为主，这是夏普提升的重要来源。",
            "",
            "详情 JSON：`reports/cycle_full_results.json`（含完整权益与交易数组）。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rel(markdown_path: Path, target: Path) -> str:
    """生成 Markdown 相对图片路径。"""

    try:
        return Path(target).resolve().relative_to(markdown_path.resolve().parent).as_posix()
    except ValueError:
        return target.as_posix()


def save_equity_csvs(report: dict[str, Any], output_dir: Path) -> list[Path]:
    """将权益与回撤曲线写为 CSV，避免主 JSON 过大。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, curve in report["equity_curves"].items():
        path = output_dir / f"equity_{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["date", "equity", "drawdown"])
            for day, equity, drawdown in zip(curve["dates"], curve["equity"], curve["drawdown"]):
                writer.writerow([day, f"{equity:.8f}", f"{drawdown:.8f}"])
        paths.append(path)
    return paths


def report_for_json(report: dict[str, Any]) -> dict[str, Any]:
    """导出精简 JSON：保留交易与分段，权益曲线改为文件索引。"""

    payload = dict(report)
    payload["equity_curves"] = {
        name: {
            "points": len(curve["dates"]),
            "csv": f"cycle_curves/equity_{name}.csv",
            "final_equity": curve["equity"][-1] if curve["equity"] else None,
            "max_drawdown": float(-min(curve["drawdown"])) if curve["drawdown"] else None,
        }
        for name, curve in report["equity_curves"].items()
    }
    return payload


def export_cycle_artifacts(report: dict[str, Any], chart_paths: Iterable[Path], artifact_dir: Path) -> None:
    """复制关键图表到 artifacts 目录，便于 PR 展示。"""

    import shutil

    artifact_dir.mkdir(parents=True, exist_ok=True)
    for path in chart_paths:
        shutil.copy2(path, artifact_dir / path.name)
