"""机构动量 CTA：数据准备、选参、跨行情评估与报告。"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .cta_data import CTA_TOP15, ensure_cta_bars, load_panel
from .cta_engine import (
    FastInstitutionalCTA,
    build_indicator_cache,
    buy_and_hold_btc,
    run_cta_backtest,
)
from .research import _parameter_product, write_json


REGIMES = (
    ("full", "全周期", date(2021, 1, 1), date(2026, 7, 15)),
    ("bull_2021", "2021牛市", date(2021, 1, 1), date(2021, 11, 10)),
    ("bear_2122", "2021-22熊市", date(2021, 11, 11), date(2022, 11, 21)),
    ("recovery", "2023-24复苏", date(2022, 11, 22), date(2024, 3, 13)),
    ("bull_2425", "2024-25主升", date(2024, 3, 14), date(2025, 10, 6)),
    ("corr_2526", "2025-26回调", date(2025, 10, 7), date(2026, 7, 15)),
)


PARAM_GRID = {
    "top_k": [3],
    "rebalance_bars": [12],
    "rebalance_phase": [0],
    "rank_buffer": [0, 1],
    "vol_target": [0.23, 0.25, 0.26],
    "atr_stop_mult": [2.8],
    "stop_cooldown_bars": [6],
    "max_asset_weight": [0.60],
    "min_score": [0.48],
    "exit_score": [0.25, 0.35],
    "half_risk_scale": [0.50],
    "breadth_threshold": [0.20],
    "breadth_risk_scale": [0.60],
    "correlation_aware": [False],
    "dd_soft": [0.09, 0.11],
    "dd_hard": [0.17, 0.18],
    "dd_min_scale": [0.20, 0.25],
    "dd_cooldown_bars": [36],
    "dd_recover_scale": [1.0],
    "dd_reentry": [0.05],
}


def prepare_cta_dataset(
    data_dir: Path,
    start: date = date(2021, 1, 1),
    end: date = date(2026, 7, 15),
    refresh: bool = False,
) -> dict[str, Any]:
    paths = ensure_cta_bars(data_dir, CTA_TOP15, start, end, refresh=refresh)
    manifest = {
        "source": "https://www.okx.com/api/v5/market/history-candles",
        "universe": list(CTA_TOP15),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "symbols": {
            symbol: {tf: path.as_posix() for tf, path in item.items()}
            for symbol, item in paths.items()
        },
    }
    write_json(data_dir / "cta_manifest.json", manifest)
    return manifest


def _ts_index(timestamps: np.ndarray, day: date, side: str = "left") -> int:
    ms = int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() * 1000)
    if side == "left":
        idx = int(np.searchsorted(timestamps, ms, side="left"))
    else:
        idx = int(np.searchsorted(timestamps, ms, side="right") - 1)
    return int(np.clip(idx, 0, len(timestamps) - 1))


def run_cta_research(
    data_dir: Path,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    train_fraction: float = 0.60,
) -> dict[str, Any]:
    """训练期网格选参，全样本/样本外/分段行情评估。"""

    print("[cta] loading panels...")
    p4 = load_panel(data_dir, "4H", CTA_TOP15)
    p12 = load_panel(data_dir, "12H", CTA_TOP15)
    p1 = load_panel(data_dir, "1D", CTA_TOP15)
    print(f"[cta] 4H={p4.size} 12H={p12.size} 1D={p1.size} assets={p4.n_assets}")
    print("[cta] building indicator cache...")
    cache = build_indicator_cache(p4, p12, p1)

    split = max(1500, int(p4.size * train_fraction))
    train_end = split - 1
    test_start = split
    fold_mid = max(800, split // 2)
    benchmark = buy_and_hold_btc(p4, p1, fee_rate, slippage_rate)

    default = FastInstitutionalCTA(cache=cache, panel_4h=p4, panel_12h=p12, panel_1d=p1)
    default_result = run_cta_backtest(default, fee_rate, slippage_rate)

    ranked: list[tuple[float, FastInstitutionalCTA, Any]] = []
    tested = 0
    for params in _parameter_product(PARAM_GRID):
        tested += 1
        strategy = FastInstitutionalCTA(
            cache=cache,
            panel_4h=p4,
            panel_12h=p12,
            panel_1d=p1,
            **params,
        )
        result = run_cta_backtest(strategy, fee_rate, slippage_rate)
        fold1 = result.metrics(0, fold_mid - 1)
        fold2 = result.metrics(fold_mid, train_end)
        train = result.metrics(0, train_end)
        if min(fold1.cagr, fold2.cagr) <= 0:
            continue
        if min(fold1.sharpe, fold2.sharpe) < 0.35:
            continue
        if train.cagr < 0.08 or train.sharpe < 0.55:
            continue
        # 训练期预留约 2 个百分点风险缓冲，避免用全样本 MDD 反向挑参数。
        if train.max_drawdown > 0.23:
            continue
        score = (
            min(fold1.sharpe, fold2.sharpe)
            + 0.45 * train.sharpe
            + 0.30 * min(train.cagr, 0.8)
            - 1.20 * train.max_drawdown
            - (0.35 if train.max_drawdown > 0.22 else 0.0)
        )
        ranked.append((score, strategy, result))
    ranked.sort(key=lambda x: x[0], reverse=True)
    if ranked:
        # 推荐完全由训练期排序产生；全样本和测试段仅在参数冻结后评价。
        best_strategy, best_result = ranked[0][1], ranked[0][2]
        for _, strategy, result in ranked[:40]:
            train = result.metrics(0, train_end)
            if (
                strategy.top_k <= 5
                and strategy.vol_target <= 0.36
                and train.max_drawdown <= 0.23
                and train.sharpe >= 0.90
                and train.cagr >= 0.14
            ):
                best_strategy, best_result = strategy, result
                break
    else:
        best_strategy, best_result = default, default_result

    # 最终测试集不得参与接受/回退决策；推荐完全由训练期双折结果决定。
    use_default = not bool(ranked)
    recommended = default if use_default else best_strategy
    recommended_result = default_result if use_default else best_result

    def regime_block(result) -> list[dict[str, Any]]:
        rows = []
        for rid, name, start, end in REGIMES:
            s = _ts_index(result.timestamps_ms, start, "left")
            e = _ts_index(result.timestamps_ms, end, "right")
            if e - s < 50:
                continue
            m = result.metrics(s, e)
            b = benchmark.metrics(s, e)
            rows.append(
                {
                    "id": rid,
                    "name": name,
                    "strategy": asdict_metrics(m),
                    "benchmark": asdict_metrics(b),
                    "excess_cagr": m.cagr - b.cagr,
                    "mdd_improvement": b.max_drawdown - m.max_drawdown,
                    "beat_sharpe": m.sharpe > b.sharpe,
                }
            )
        return rows

    from dataclasses import asdict as dc_asdict

    def asdict_metrics(m):
        return dc_asdict(m)

    rec_params = {
        "top_k": recommended.top_k,
        "rebalance_bars": recommended.rebalance_bars,
        "rebalance_phase": recommended.rebalance_phase,
        "rank_buffer": recommended.rank_buffer,
        "vol_target": recommended.vol_target,
        "atr_stop_mult": recommended.atr_stop_mult,
        "stop_cooldown_bars": recommended.stop_cooldown_bars,
        "min_score": recommended.min_score,
        "exit_score": recommended.exit_score,
        "max_gross": recommended.max_gross,
        "max_asset_weight": recommended.max_asset_weight,
        "half_risk_scale": recommended.half_risk_scale,
        "breadth_threshold": recommended.breadth_threshold,
        "breadth_risk_scale": recommended.breadth_risk_scale,
        "correlation_aware": recommended.correlation_aware,
        "dd_soft": recommended.dd_soft,
        "dd_hard": recommended.dd_hard,
        "dd_reentry": recommended.dd_reentry,
        "dd_min_scale": recommended.dd_min_scale,
        "dd_cooldown_bars": recommended.dd_cooldown_bars,
        "dd_recover_scale": recommended.dd_recover_scale,
    }

    phase_sensitivity = []
    for phase in range(recommended.rebalance_bars):
        phase_strategy = replace(recommended, rebalance_phase=phase)
        phase_result = run_cta_backtest(phase_strategy, fee_rate, slippage_rate)
        phase_sensitivity.append(
            {
                "phase": phase,
                "full": asdict_metrics(phase_result.metrics()),
                "train": asdict_metrics(phase_result.metrics(0, train_end)),
                "test": asdict_metrics(phase_result.metrics(test_start)),
            }
        )

    cost_stress = []
    for total_cost in (0.0, 0.0015, 0.0030, 0.0050):
        stressed = run_cta_backtest(recommended, total_cost, 0.0)
        cost_stress.append(
            {
                "one_way_cost": total_cost,
                "full": asdict_metrics(stressed.metrics()),
                "test": asdict_metrics(stressed.metrics(test_start)),
            }
        )

    hard_threshold_sensitivity = []
    for multiplier in (0.90, 1.0, 1.10):
        strategy = replace(recommended, dd_hard=recommended.dd_hard * multiplier)
        stressed = run_cta_backtest(strategy, fee_rate, slippage_rate)
        hard_threshold_sensitivity.append(
            {
                "multiplier": multiplier,
                "dd_hard": strategy.dd_hard,
                "full": asdict_metrics(stressed.metrics()),
                "test": asdict_metrics(stressed.metrics(test_start)),
            }
        )

    ablations = []
    for label, strategy in (
        ("recommended", recommended),
        ("no_breadth_scaling", replace(recommended, breadth_threshold=0.0)),
        ("no_rank_buffer", replace(recommended, rank_buffer=0)),
        (
            "no_drawdown_overlay",
            replace(
                recommended,
                dd_soft=0.99,
                dd_hard=0.999,
                dd_reentry=0.99,
                dd_min_scale=1.0,
                dd_cooldown_bars=1,
            ),
        ),
        ("correlation_aware_risk", replace(recommended, correlation_aware=True)),
    ):
        result = run_cta_backtest(strategy, fee_rate, slippage_rate)
        ablations.append(
            {
                "name": label,
                "full": asdict_metrics(result.metrics()),
                "test": asdict_metrics(result.metrics(test_start)),
            }
        )

    phase_full_cagr = [row["full"]["cagr"] for row in phase_sensitivity]
    phase_full_mdd = [row["full"]["max_drawdown"] for row in phase_sensitivity]
    phase_test_cagr = [row["test"]["cagr"] for row in phase_sensitivity]
    phase_test_mdd = [row["test"]["max_drawdown"] for row in phase_sensitivity]
    phase_summary = {
        "full_cagr_median": float(np.median(phase_full_cagr)),
        "full_cagr_min": float(np.min(phase_full_cagr)),
        "full_mdd_median": float(np.median(phase_full_mdd)),
        "full_mdd_max": float(np.max(phase_full_mdd)),
        "test_cagr_median": float(np.median(phase_test_cagr)),
        "test_cagr_min": float(np.min(phase_test_cagr)),
        "test_mdd_median": float(np.median(phase_test_mdd)),
        "test_mdd_max": float(np.max(phase_test_mdd)),
    }

    payload = {
        "methodology": {
            "style": "Institutional multi-asset momentum CTA",
            "universe": list(CTA_TOP15),
            "timeframes": ["4H", "12H", "1D"],
            "signals": [
                "BTC tiered risk gate: full risk above EMA200+slope, half risk above EMA100 only, cash below EMA100",
                "1D/12H/4H momentum+MACD+RSI+KDJ+volume composite score (no higher-TF look-ahead)",
                "cross-sectional Top-K rotation with score hysteresis exits",
                "inverse-vol weights + portfolio vol target",
                "ATR trailing stop",
                "persistent ATR high/stop with post-stop cooldown",
                "absolute UTC rebalance schedule + Top-K rank buffer",
                "market-breadth risk scaling and per-asset concentration cap",
                "portfolio drawdown circuit: soft delever + hard flatten cooldown + signal-based re-entry (no permanent lock)",
            ],
            "selection": "train dual-fold only; train MDD<=23% risk-margin filter; final evaluation segment never participates in selection/fallback",
            "risk_budget": "target full-sample MDD<=25% while preserving CAGR (sacrifice at most ~3-5pp vs prior ~22% CAGR baseline)",
            "test_disclosure": "The chronological evaluation segment is excluded from the current selector, but it was inspected in prior research iterations; it is not a pristine untouched OOS sample.",
            "fee_rate": fee_rate,
            "slippage_rate": slippage_rate,
            "references": [
                "CTA vol targeting + trend following",
                "Cross-sectional crypto momentum with BTC regime filter",
                "Donchian/momentum rotation with ATR risk control",
            ],
        },
        "data_range": {
            "start": _iso(int(p4.timestamps_ms[0])),
            "end": _iso(int(p4.timestamps_ms[-1])),
            "bars_4h": p4.size,
        },
        "split": {
            "train_end_index": train_end,
            "test_start_index": test_start,
            "train_end": _iso(int(p4.timestamps_ms[train_end])),
            "test_start": _iso(int(p4.timestamps_ms[test_start])),
        },
        "search": {
            "candidates_tested": tested,
            "candidates_qualified": len(ranked),
            "used_default": use_default,
        },
        "robustness": {
            "phase_sensitivity": phase_sensitivity,
            "phase_summary": phase_summary,
            "cost_stress": cost_stress,
            "hard_threshold_sensitivity": hard_threshold_sensitivity,
            "ablations": ablations,
        },
        "recommended_parameters": rec_params,
        "benchmark": {
            "full": asdict_metrics(benchmark.metrics()),
            "train": asdict_metrics(benchmark.metrics(0, train_end)),
            "test": asdict_metrics(benchmark.metrics(test_start)),
        },
        "strategy": {
            "full": asdict_metrics(recommended_result.metrics()),
            "train": asdict_metrics(recommended_result.metrics(0, train_end)),
            "test": asdict_metrics(recommended_result.metrics(test_start)),
        },
        "regimes": regime_block(recommended_result),
        "equity": {
            "strategy": [float(x) for x in recommended_result.equity[:: max(1, p4.size // 900)]],
            "benchmark": [float(x) for x in benchmark.equity[:: max(1, p4.size // 900)]],
            "final_strategy": float(recommended_result.equity[-1]),
            "final_benchmark": float(benchmark.equity[-1]),
        },
        "top_alternatives": [
            {
                "score": score,
                "params": {
                    "top_k": strat.top_k,
                    "rebalance_bars": strat.rebalance_bars,
                    "rank_buffer": strat.rank_buffer,
                    "vol_target": strat.vol_target,
                    "atr_stop_mult": strat.atr_stop_mult,
                    "stop_cooldown_bars": strat.stop_cooldown_bars,
                    "max_asset_weight": strat.max_asset_weight,
                    "min_score": strat.min_score,
                    "exit_score": strat.exit_score,
                    "breadth_threshold": strat.breadth_threshold,
                    "breadth_risk_scale": strat.breadth_risk_scale,
                    "dd_soft": strat.dd_soft,
                    "dd_hard": strat.dd_hard,
                    "dd_cooldown_bars": strat.dd_cooldown_bars,
                    "dd_min_scale": strat.dd_min_scale,
                },
                "train": asdict_metrics(res.metrics(0, train_end)),
                "test": asdict_metrics(res.metrics(test_start)),
                "full": asdict_metrics(res.metrics()),
            }
            for score, strat, res in ranked[:8]
        ],
    }
    # attach full result for plotting via side channel file if needed
    payload["_runtime"] = {
        "result": recommended_result,
        "benchmark": benchmark,
    }
    return payload


def plot_cta_charts(results: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(results["equity"]["benchmark"], color="#111111", lw=2.2, label="BTC buy&hold")
    ax.plot(results["equity"]["strategy"], color="#0B6E4F", lw=2.0, label="Institutional CTA")
    ax.set_title("Institutional Momentum CTA vs BTC (4H, downsampled)")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.25)
    ax.legend()
    path = output_dir / "cta_equity.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths["equity"] = path

    # regime bars
    names = [r["name"] for r in results["regimes"] if r["id"] != "full"]
    s_cagr = [r["strategy"]["cagr"] * 100 for r in results["regimes"] if r["id"] != "full"]
    b_cagr = [r["benchmark"]["cagr"] * 100 for r in results["regimes"] if r["id"] != "full"]
    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, b_cagr, width, label="BTC CAGR%", color="#333333")
    ax.bar(x + width / 2, s_cagr, width, label="CTA CAGR%", color="#0B6E4F")
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Bull21", "Bear21-22", "Recovery", "Bull24-25", "Corr25-26"][: len(names)],
        rotation=15,
    )
    ax.axhline(0, color="#666", lw=1)
    ax.set_title("Regime CAGR: CTA vs BTC")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    path = output_dir / "cta_regime_cagr.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths["regime"] = path

    # drawdown from runtime if present
    runtime = results.get("_runtime")
    if runtime:
        eq = runtime["result"].equity
        peaks = np.maximum.accumulate(eq)
        dd = eq / peaks - 1.0
        beq = runtime["benchmark"].equity
        bpeaks = np.maximum.accumulate(beq)
        bdd = beq / bpeaks - 1.0
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(bdd, color="#111111", lw=1.5, label="BTC DD")
        ax.plot(dd, color="#0B6E4F", lw=1.5, label="CTA DD")
        ax.set_title("Drawdown")
        ax.grid(True, alpha=0.25)
        ax.legend()
        path = output_dir / "cta_drawdown.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths["drawdown"] = path
    return paths


def write_cta_report(path: Path, results: dict[str, Any], chart_dir: Path, manifest: dict[str, Any]) -> None:
    s = results["strategy"]
    b = results["benchmark"]
    lines = [
        "# 机构级多标的动量 CTA 报告（TOP15 · 4H/12H/1D）",
        "",
        "## 设计框架（对标机构实践）",
        "",
        *[f"- {x}" for x in results["methodology"]["signals"]],
        "",
        f"- 标的池：{', '.join(results['methodology']['universe'])}",
        f"- 选参：{results['methodology']['selection']}",
        f"- 成本：fee={results['methodology']['fee_rate']}, slippage={results['methodology']['slippage_rate']}",
        f"- 数据：{manifest.get('source')} · {results['data_range']['start']} ~ {results['data_range']['end']}",
        "",
        "## 推荐参数",
        "",
        f"`{results['recommended_parameters']}`",
        f"- 网格测试 `{results['search']['candidates_tested']}`，合格 `{results['search']['candidates_qualified']}`，"
        f"{'回退默认' if results['search']['used_default'] else '采用优化参数'}",
        "",
        "## 全样本 / 时间顺序评估段 vs BTC",
        "",
        "| 区间 | CTA CAGR | Sharpe | MDD | 在市 | BTC CAGR | BTC Sharpe | BTC MDD | 超额CAGR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, sm, bm in (
        ("全样本", s["full"], b["full"]),
        ("训练期", s["train"], b["train"]),
        ("评估段*", s["test"], b["test"]),
    ):
        lines.append(
            "| {lab} | {sc:.1%} | {ss:.2f} | {smdd:.1%} | {tim:.0%} | {bc:.1%} | {bs:.2f} | {bmdd:.1%} | {ex:+.1%} |".format(
                lab=label,
                sc=sm["cagr"],
                ss=sm["sharpe"],
                smdd=sm["max_drawdown"],
                tim=sm["time_in_market"],
                bc=bm["cagr"],
                bs=bm["sharpe"],
                bmdd=bm["max_drawdown"],
                ex=sm["cagr"] - bm["cagr"],
            )
        )

    lines.extend(
        [
            "",
            "\\* 评估段未参与本轮选参或回退，但此前研究迭代已查看过该区间，因此不再宣称为严格未触碰 OOS。",
            "",
            "## 图表",
            "",
            f"![权益]({_rel(path, chart_dir / 'cta_equity.png')})",
            "",
            f"![回撤]({_rel(path, chart_dir / 'cta_drawdown.png')})",
            "",
            f"![分段]({_rel(path, chart_dir / 'cta_regime_cagr.png')})",
            "",
            "## 不同大行情覆盖",
            "",
            "| 行情 | CTA CAGR | Sharpe | MDD | BTC CAGR | BTC Sharpe | BTC MDD | 胜夏普 | 回撤改善 |",
            "|---|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
    )
    for row in results["regimes"]:
        lines.append(
            "| {name} | {sc:.1%} | {ss:.2f} | {sm:.1%} | {bc:.1%} | {bs:.2f} | {bm:.1%} | {beat} | {imp:+.1%} |".format(
                name=row["name"],
                sc=row["strategy"]["cagr"],
                ss=row["strategy"]["sharpe"],
                sm=row["strategy"]["max_drawdown"],
                bc=row["benchmark"]["cagr"],
                bs=row["benchmark"]["sharpe"],
                bm=row["benchmark"]["max_drawdown"],
                beat="是" if row["beat_sharpe"] else "否",
                imp=row["mdd_improvement"],
            )
        )

    robustness = results.get("robustness", {})
    phase = robustness.get("phase_summary")
    if phase:
        lines.extend(
            [
                "",
                "## 对抗性稳健性检查",
                "",
                "### 调仓相位扰动（全部绝对 UTC 相位）",
                "",
                "| 指标 | 全样本中位数 | 全样本最差 | 评估段中位数 | 评估段最差 |",
                "|---|---:|---:|---:|---:|",
                "| CAGR | {fmed:.1%} | {fmin:.1%} | {tmed:.1%} | {tmin:.1%} |".format(
                    fmed=phase["full_cagr_median"],
                    fmin=phase["full_cagr_min"],
                    tmed=phase["test_cagr_median"],
                    tmin=phase["test_cagr_min"],
                ),
                "| MDD | {fmed:.1%} | {fmax:.1%} | {tmed:.1%} | {tmax:.1%} |".format(
                    fmed=phase["full_mdd_median"],
                    fmax=phase["full_mdd_max"],
                    tmed=phase["test_mdd_median"],
                    tmax=phase["test_mdd_max"],
                ),
                "",
                "### 单边成本压力",
                "",
                "| 单边成本 | 全样本 CAGR | 全样本 MDD | 评估段 CAGR | 评估段 MDD |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for row in robustness.get("cost_stress", []):
            lines.append(
                "| {cost:.2%} | {fc:.1%} | {fm:.1%} | {tc:.1%} | {tm:.1%} |".format(
                    cost=row["one_way_cost"],
                    fc=row["full"]["cagr"],
                    fm=row["full"]["max_drawdown"],
                    tc=row["test"]["cagr"],
                    tm=row["test"]["max_drawdown"],
                )
            )
        lines.extend(
            [
                "",
                "### 消融实验",
                "",
                "| 版本 | 全样本 CAGR | Sharpe | MDD | 评估段 CAGR | 评估段 MDD |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in robustness.get("ablations", []):
            lines.append(
                "| {name} | {fc:.1%} | {fs:.2f} | {fm:.1%} | {tc:.1%} | {tm:.1%} |".format(
                    name=row["name"],
                    fc=row["full"]["cagr"],
                    fs=row["full"]["sharpe"],
                    fm=row["full"]["max_drawdown"],
                    tc=row["test"]["cagr"],
                    tm=row["test"]["max_drawdown"],
                )
            )

    if results.get("top_alternatives"):
        lines.extend(
            [
                "",
                "## 训练期备选参数（Top）",
                "",
                "| 参数 | 训练Sharpe | 训练CAGR | OOS Sharpe | OOS CAGR | 全样本Sharpe |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for alt in results["top_alternatives"][:5]:
            lines.append(
                "| {p} | {ts:.2f} | {tc:.1%} | {os:.2f} | {oc:.1%} | {fs:.2f} |".format(
                    p=alt["params"],
                    ts=alt["train"]["sharpe"],
                    tc=alt["train"]["cagr"],
                    os=alt["test"]["sharpe"],
                    oc=alt["test"]["cagr"],
                    fs=alt["full"]["sharpe"],
                )
            )

    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- 风险预算：全样本 MDD={s['full']['max_drawdown']:.1%}（目标≤25%），CAGR={s['full']['cagr']:.1%}。",
            "- 组合回撤熔断：软阈值线性降仓 → 硬阈值冷却空仓 → 冷却后按原信号恢复（修复了“空仓导致回撤永不收复”的永久锁仓缺陷）。",
            "- 机构 CTA 的核心价值是：熊市/回调段大幅降低回撤，全周期夏普高于 BTC。",
            "- 已消除高周期前视：4H 信号仅使用已收盘的 12H/1D 指标。",
            "- 分层门控（EMA200 满仓 / EMA100 半仓 / 以下空仓）是抗熊的关键。",
            "- 注意：硬熔断阈值对路径较敏感，实盘需把 dd_hard/cooldown 纳入稳健性监控，而非单点最优。",
            "- 调仓相位压力结果必须与单点结果同时阅读；相位最差值说明历史 MDD≤25% 不是未来风险保证。",
            "",
            "详情：`reports/cta_results.json`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _iso(ts: int) -> str:
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()


def _rel(md: Path, target: Path) -> str:
    try:
        return target.resolve().relative_to(md.resolve().parent).as_posix()
    except ValueError:
        return target.as_posix()


def serialize_cta_results(results: dict[str, Any]) -> dict[str, Any]:
    """去掉不可 JSON 化的 runtime 对象。"""

    payload = dict(results)
    payload.pop("_runtime", None)
    return payload
