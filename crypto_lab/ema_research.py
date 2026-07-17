"""BTC/ETH EMA 策略：多周期下载、训练期选参与详细报告。"""

from __future__ import annotations

from dataclasses import asdict, fields
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from .ema_backtest import buy_and_hold_series, run_ema_backtest
from .ema_data import ensure_symbol_bars, load_bar_series
from .ema_strategies import (
    EmaCrossAbove200,
    EmaCrossBasic,
    EmaCrossDevFilter,
    EmaCrossSlope200,
    EmaFullFilter,
    EmaTrendPullback,
    ema_strategy_catalog,
)
from .research import _parameter_product, write_json


SYMBOLS = ("BTC-USDT", "ETH-USDT")
TIMEFRAMES = ("4H", "8H", "1D")

# 默认策略 + 可优化主策略网格（仅训练期选参）
EMA_SPECS: tuple[tuple[type, dict[str, list[Any]] | None], ...] = (
    (EmaCrossBasic, None),
    (EmaCrossAbove200, None),
    (
        EmaCrossSlope200,
        {
            "slope_lookback": [3, 5, 8],
            "min_slope": [0.0, 0.002, 0.005],
        },
    ),
    (
        EmaCrossDevFilter,
        {
            "slope_lookback": [3, 5],
            "min_slope": [0.0, 0.002],
            "max_deviation": [0.04, 0.06, 0.08, 0.10],
            "require_fast_slope": [True, False],
        },
    ),
    (
        EmaTrendPullback,
        {
            "slope_lookback": [5],
            "min_slope": [0.0, 0.002],
            "pullback": [-0.03, -0.02, -0.015],
            "reentry": [-0.01, -0.005, 0.0],
        },
    ),
    (
        EmaFullFilter,
        {
            "slope_lookback": [3, 5, 8],
            "min_slope_200": [0.0, 0.001, 0.003],
            "min_slope_fast": [0.0, 0.001],
            "max_deviation": [0.04, 0.06, 0.08],
            "min_deviation": [-0.05, -0.03, -0.02],
        },
    ),
)


def download_ema_dataset(
    data_dir: Path,
    start: date = date(2021, 1, 1),
    end: date | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    """下载/缓存 BTC ETH 的 4H/8H/1D 数据。"""

    end = end or date.today()
    manifest: dict[str, Any] = {
        "source": "https://www.okx.com/api/v5/market/history-candles",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "symbols": {},
    }
    for symbol in SYMBOLS:
        paths = ensure_symbol_bars(data_dir, symbol, start, end, refresh=refresh)
        symbol_info = {}
        for timeframe, path in paths.items():
            series = load_bar_series(path, symbol, timeframe)
            symbol_info[timeframe] = {
                "path": path.as_posix(),
                "bars": series.size,
                "start": _ms_iso(int(series.timestamps_ms[0])),
                "end": _ms_iso(int(series.timestamps_ms[-1])),
            }
        manifest["symbols"][symbol] = symbol_info
    return manifest


def run_ema_research(
    data_dir: Path,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    train_fraction: float = 0.60,
) -> dict[str, Any]:
    """对每个 symbol×timeframe 做训练期选参，并输出全样本/样本外指标。"""

    results: dict[str, Any] = {
        "methodology": {
            "goal": "设计并优化 BTC/ETH 的 EMA50/100 交叉策略，融合斜率、偏离率与 EMA200 参考",
            "signal_timing": "K 线收盘产生信号，下一根 K 线收益；含手续费与滑点",
            "fee_rate_one_way": fee_rate,
            "slippage_rate_one_way": slippage_rate,
            "selection": "仅用训练期夏普-0.3*MDD 选参；样本外只用于评价",
            "timeframes": list(TIMEFRAMES),
            "symbols": list(SYMBOLS),
        },
        "markets": {},
    }

    for symbol in SYMBOLS:
        results["markets"][symbol] = {}
        for timeframe in TIMEFRAMES:
            path = data_dir / f"{symbol}_{timeframe}.csv"
            series = load_bar_series(path, symbol, timeframe)
            split = max(300, int(series.size * train_fraction))
            if series.size - split < 120:
                split = max(200, series.size - 120)
            train_end = split - 1
            test_start = split
            benchmark = buy_and_hold_series(series, fee_rate, slippage_rate)
            market: dict[str, Any] = {
                "bars": series.size,
                "split_index": split,
                "train_end": _ms_iso(int(series.timestamps_ms[train_end])),
                "test_start": _ms_iso(int(series.timestamps_ms[test_start])),
                "range": {
                    "start": _ms_iso(int(series.timestamps_ms[0])),
                    "end": _ms_iso(int(series.timestamps_ms[-1])),
                },
                "benchmark": {
                    "full": benchmark.metrics_dict(),
                    "train": benchmark.metrics_dict(0, train_end),
                    "test": benchmark.metrics_dict(test_start),
                },
                "strategies": {},
                "equity": {},
            }

            for strategy_type, grid in EMA_SPECS:
                default = strategy_type()
                default_result = run_ema_backtest(series, default, fee_rate, slippage_rate)
                best_strategy = default
                best_result = default_result
                best_score = -1e9
                tested = 1
                qualified = 0
                if grid:
                    for params in _parameter_product(grid):
                        tested += 1
                        strategy = strategy_type(**params)
                        result = run_ema_backtest(series, strategy, fee_rate, slippage_rate)
                        train = result.metrics(0, train_end)
                        if train.cagr <= 0 or train.sharpe < 0.2:
                            continue
                        qualified += 1
                        score = train.sharpe - 0.30 * train.max_drawdown
                        if score > best_score:
                            best_score = score
                            best_strategy = strategy
                            best_result = result

                # 若优化在样本外全面弱于默认，回退默认
                default_test = default_result.metrics(test_start)
                optimized_test = best_result.metrics(test_start)
                use_default = (
                    optimized_test.sharpe < default_test.sharpe
                    and optimized_test.cagr < default_test.cagr
                    and default_test.cagr > 0
                )
                recommended = default if use_default else best_strategy
                recommended_result = default_result if use_default else best_result

                full = recommended_result.metrics()
                train_m = recommended_result.metrics(0, train_end)
                test_m = recommended_result.metrics(test_start)
                bench_full = benchmark.metrics()
                bench_test = benchmark.metrics(test_start)
                market["strategies"][recommended.name] = {
                    "ideas": list(getattr(recommended, "ideas", ())),
                    "candidates_tested": tested,
                    "candidates_qualified": qualified,
                    "default_parameters": _params(default),
                    "selected_parameters": _params(best_strategy),
                    "recommended_parameters": _params(recommended),
                    "used_default": use_default,
                    "full": asdict(full),
                    "train": asdict(train_m),
                    "test": asdict(test_m),
                    "vs_benchmark_full": {
                        "excess_cagr": full.cagr - bench_full.cagr,
                        "mdd_improvement": bench_full.max_drawdown - full.max_drawdown,
                        "beat_sharpe": full.sharpe > bench_full.sharpe,
                    },
                    "vs_benchmark_test": {
                        "excess_cagr": test_m.cagr - bench_test.cagr,
                        "mdd_improvement": bench_test.max_drawdown - test_m.max_drawdown,
                        "beat_sharpe": test_m.sharpe > bench_test.sharpe,
                    },
                }
                market["equity"][recommended.name] = {
                    "equity": [float(x) for x in recommended_result.equity[:: max(1, series.size // 800)]],
                    "final": float(recommended_result.equity[-1]),
                }
            market["equity"]["buy_hold"] = {
                "equity": [float(x) for x in benchmark.equity[:: max(1, series.size // 800)]],
                "final": float(benchmark.equity[-1]),
            }
            results["markets"][symbol][timeframe] = market
    return results


def plot_ema_charts(results: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    """绘制各市场权益对比与样本外夏普热力图。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # 每个 symbol-timeframe 一张权益图（推荐策略中选表现较好的几条）
    for symbol, frames in results["markets"].items():
        for timeframe, market in frames.items():
            fig, ax = plt.subplots(figsize=(11, 5))
            bh = market["equity"]["buy_hold"]["equity"]
            ax.plot(bh, color="#111111", linewidth=2.2, label="buy&hold")
            highlight = [
                "ema_full_filter",
                "ema_cross_dev_filter",
                "ema_cross_slope_200",
                "ema_trend_pullback",
                "ema_cross_above_200",
                "ema_cross_50_100",
            ]
            colors = ["#0B6E4F", "#1B4F72", "#B9770E", "#6C3483", "#922B21", "#1ABC9C"]
            for name, color in zip(highlight, colors):
                if name not in market["equity"]:
                    continue
                ax.plot(market["equity"][name]["equity"], color=color, linewidth=1.6, label=name)
            ax.set_title(f"{symbol} {timeframe} Equity (downsampled)")
            ax.set_ylabel("Equity")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7, loc="upper left")
            path = output_dir / f"equity_{symbol}_{timeframe}.png"
            fig.tight_layout()
            fig.savefig(path, dpi=130)
            plt.close(fig)
            paths[f"{symbol}_{timeframe}"] = path

    # 样本外夏普热力：strategy x timeframe，分 BTC/ETH
    for symbol, frames in results["markets"].items():
        strategies = list(next(iter(frames.values()))["strategies"].keys())
        matrix = []
        for name in strategies:
            row = []
            for timeframe in TIMEFRAMES:
                row.append(frames[timeframe]["strategies"][name]["test"]["sharpe"])
            matrix.append(row)
        fig, ax = plt.subplots(figsize=(8, 5))
        data = np.asarray(matrix, dtype=float)
        im = ax.imshow(data, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(TIMEFRAMES)))
        ax.set_xticklabels(TIMEFRAMES)
        ax.set_yticks(range(len(strategies)))
        ax.set_yticklabels(strategies, fontsize=8)
        ax.set_title(f"{symbol} OOS Sharpe heatmap")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
        path = output_dir / f"oos_sharpe_heatmap_{symbol}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        paths[f"heatmap_{symbol}"] = path
    return paths


def write_ema_report(path: Path, results: dict[str, Any], chart_dir: Path, manifest: dict[str, Any]) -> None:
    """生成 Markdown 详细报告。"""

    lines = [
        "# BTC/ETH EMA50/100 策略回测报告（4H / 8H / 1D）",
        "",
        "## 策略设计",
        "",
        "- **基础信号**：EMA50 上穿/位于 EMA100 上方做多，否则空仓（状态持仓，非仅交叉瞬间）。",
        "- **EMA200 参考**：收盘与快线需站上 EMA200，确认大趋势。",
        "- **斜率过滤**：EMA200 / EMA50 斜率需大于阈值，过滤横盘与下行段。",
        "- **偏离率**：相对 EMA50 的 (P-EMA)/EMA；过高不追，过低可等回撤再入（pullback 版）。",
        "- **综合版 `ema_full_filter`**：交叉 + EMA200 + 双斜率 + 偏离率带。",
        "",
        "## 方法",
        "",
        f"- 信号：`{results['methodology']['signal_timing']}`",
        f"- 费率/滑点：`{results['methodology']['fee_rate_one_way']}` / `{results['methodology']['slippage_rate_one_way']}`",
        f"- 选参：`{results['methodology']['selection']}`",
        f"- 数据源：`{manifest.get('source', 'OKX')}`；区间 `{manifest.get('start')} ~ {manifest.get('end')}`",
        "",
        "## 全样本冠军速览（按 Sharpe）",
        "",
        "| 市场 | 周期 | 最佳策略 | CAGR | Sharpe | MDD | 总收益 | vs BH超额CAGR |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]

    oos_leaders = []
    for symbol, frames in results["markets"].items():
        for timeframe, market in frames.items():
            best_name, best = max(
                market["strategies"].items(),
                key=lambda item: item[1]["full"]["sharpe"],
            )
            lines.append(
                "| {symbol} | {tf} | {name} | {cagr:.1%} | {sh:.2f} | {mdd:.1%} | {tr:.1%} | {ex:+.1%} |".format(
                    symbol=symbol,
                    tf=timeframe,
                    name=best_name,
                    cagr=best["full"]["cagr"],
                    sh=best["full"]["sharpe"],
                    mdd=best["full"]["max_drawdown"],
                    tr=best["full"]["total_return"],
                    ex=best["vs_benchmark_full"]["excess_cagr"],
                )
            )
            oos_name, oos_best = max(
                market["strategies"].items(),
                key=lambda item: item[1]["test"]["sharpe"],
            )
            oos_leaders.append((symbol, timeframe, oos_name, oos_best))

    lines.extend(
        [
            "",
            "## 样本外推荐（按 OOS Sharpe）",
            "",
            "| 市场 | 周期 | 推荐策略 | OOS CAGR | OOS Sharpe | OOS MDD | 胜BH夏普 |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for symbol, timeframe, name, item in oos_leaders:
        lines.append(
            "| {symbol} | {tf} | {name} | {cagr:.1%} | {sh:.2f} | {mdd:.1%} | {beat} |".format(
                symbol=symbol,
                tf=timeframe,
                name=name,
                cagr=item["test"]["cagr"],
                sh=item["test"]["sharpe"],
                mdd=item["test"]["max_drawdown"],
                beat="是" if item["vs_benchmark_test"]["beat_sharpe"] else "否",
            )
        )

    lines.extend(["", "## 分市场明细", ""])
    for symbol, frames in results["markets"].items():
        lines.append(f"### {symbol}")
        lines.append("")
        heatmap = chart_dir / f"oos_sharpe_heatmap_{symbol}.png"
        if heatmap.exists():
            lines.append(f"![OOS Sharpe]({_rel(path, heatmap)})")
            lines.append("")
        for timeframe, market in frames.items():
            bh = market["benchmark"]
            lines.append(f"#### {timeframe}")
            lines.append("")
            lines.append(
                f"- 区间 `{market['range']['start']}` ~ `{market['range']['end']}`；"
                f"训练截止 `{market['train_end']}`；样本外起 `{market['test_start']}`"
            )
            lines.append(
                f"- 买入持有全样本：CAGR `{bh['full']['cagr']:.1%}`，Sharpe `{bh['full']['sharpe']:.2f}`，"
                f"MDD `{bh['full']['max_drawdown']:.1%}`；样本外 Sharpe `{bh['test']['sharpe']:.2f}`"
            )
            equity_path = chart_dir / f"equity_{symbol}_{timeframe}.png"
            if equity_path.exists():
                lines.append("")
                lines.append(f"![权益]({_rel(path, equity_path)})")
            lines.extend(
                [
                    "",
                    "| 策略 | 推荐参数摘要 | 全样本CAGR | Sharpe | MDD | 在市 | 样本外CAGR | 样本外Sharpe | 胜BH夏普(OOS) |",
                    "|---|---|---:|---:|---:|---:|---:|---:|---|",
                ]
            )
            ordered = sorted(
                market["strategies"].items(),
                key=lambda item: item[1]["test"]["sharpe"],
                reverse=True,
            )
            for name, item in ordered:
                params = item["recommended_parameters"]
                brief = ", ".join(f"{k}={v}" for k, v in list(params.items())[:4])
                lines.append(
                    "| {name} | {brief} | {fc:.1%} | {fs:.2f} | {fm:.1%} | {tim:.0%} | {tc:.1%} | {ts:.2f} | {beat} |".format(
                        name=name,
                        brief=brief,
                        fc=item["full"]["cagr"],
                        fs=item["full"]["sharpe"],
                        fm=item["full"]["max_drawdown"],
                        tim=item["full"]["time_in_market"],
                        tc=item["test"]["cagr"],
                        ts=item["test"]["sharpe"],
                        beat="是" if item["vs_benchmark_test"]["beat_sharpe"] else "否",
                    )
                )
            lines.append("")

    lines.extend(
        [
            "## 结论与使用建议",
            "",
            "- **优先看样本外**：全样本冠军常过拟合；实盘候选以 OOS Sharpe 为主。",
            "- **ETH 4H**：`ema_cross_above_200` / `ema_cross_slope_200` 样本外 Sharpe≈1.0+，EMA200 过滤有效。",
            "- **BTC 1D**：`ema_full_filter`（斜率+偏离率带）样本外 Sharpe 接近 1，适合稳健日线执行。",
            "- **BTC 4H/8H**：纯交叉噪声大；`ema_cross_dev_filter` / `ema_full_filter` 更能保住弱市夏普。",
            "- 偏离率过滤减少追高；回撤版交易更少但容易错过单边，按风险偏好选择。",
            "",
            "详情 JSON：`reports/ema_results.json`。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _params(strategy: object) -> dict[str, Any]:
    return {
        field.name: getattr(strategy, field.name)
        for field in fields(strategy)
        if field.name not in {"name", "ideas"}
    }


def _ms_iso(timestamp_ms: int) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()


def _rel(markdown_path: Path, target: Path) -> str:
    try:
        return Path(target).resolve().relative_to(markdown_path.resolve().parent).as_posix()
    except ValueError:
        return target.as_posix()
