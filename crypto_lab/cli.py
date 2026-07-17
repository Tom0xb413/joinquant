"""命令行入口：下载真实行情并运行可复现研究。"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path

from .crypto_alpha_research import run_crypto_alpha_research, write_crypto_alpha_report
from .cycle_report import (
    export_cycle_artifacts,
    plot_cycle_charts,
    report_for_json,
    run_cycle_report,
    save_equity_csvs,
    save_trade_csvs,
    write_cycle_markdown,
)
from .ema_research import (
    download_ema_dataset,
    plot_ema_charts,
    run_ema_research,
    write_ema_report,
)
from .optimize_research import run_optimized_research, write_optimized_markdown_report
from .research import run_research, write_json, write_markdown_report
from .data import (
    OkxDataClient,
    align_market_data,
    dataset_manifest,
    load_candles,
    save_candles,
)


DEFAULT_SYMBOLS = (
    "BTC-USDT",
    "ETH-USDT",
    "SOL-USDT",
    "XRP-USDT",
    "ADA-USDT",
    "DOGE-USDT",
    "LTC-USDT",
    "BCH-USDT",
    "LINK-USDT",
    "DOT-USDT",
    "ETC-USDT",
    "TRX-USDT",
)


def build_parser() -> argparse.ArgumentParser:
    """构建参数解析器。"""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="从 OKX 下载真实 UTC 日线")
    download.add_argument("--data-dir", type=Path, default=Path("data/okx"))
    download.add_argument("--start", type=date.fromisoformat, default=date(2021, 1, 1))
    download.add_argument("--end", type=date.fromisoformat, default=date.today())
    download.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    download.add_argument("--refresh", action="store_true")

    research = subparsers.add_parser("research", help="优化并运行样本外回测")
    research.add_argument("--data-dir", type=Path, default=Path("data/okx"))
    research.add_argument("--output-dir", type=Path, default=Path("reports"))
    research.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    research.add_argument("--fee-rate", type=float, default=0.001)
    research.add_argument("--slippage-rate", type=float, default=0.0005)
    research.add_argument("--train-fraction", type=float, default=0.60)

    optimize = subparsers.add_parser("optimize", help="运行重新设计的低换手优化策略研究")
    optimize.add_argument("--data-dir", type=Path, default=Path("data/okx"))
    optimize.add_argument("--output-dir", type=Path, default=Path("reports"))
    optimize.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    optimize.add_argument("--fee-rate", type=float, default=0.001)
    optimize.add_argument("--slippage-rate", type=float, default=0.0005)
    optimize.add_argument("--train-fraction", type=float, default=0.60)

    alpha = subparsers.add_parser("crypto-alpha", help="BTC门控/轮动/对冲增强策略，冲击15%+夏普1+")
    alpha.add_argument("--data-dir", type=Path, default=Path("data/okx"))
    alpha.add_argument("--output-dir", type=Path, default=Path("reports"))
    alpha.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    alpha.add_argument("--fee-rate", type=float, default=0.001)
    alpha.add_argument("--slippage-rate", type=float, default=0.0005)
    alpha.add_argument("--train-fraction", type=float, default=0.60)

    cycle = subparsers.add_parser("cycle-report", help="2021-2026全周期与beta分段详细报告")
    cycle.add_argument("--data-dir", type=Path, default=Path("data/okx"))
    cycle.add_argument("--output-dir", type=Path, default=Path("reports"))
    cycle.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    cycle.add_argument("--fee-rate", type=float, default=0.001)
    cycle.add_argument("--slippage-rate", type=float, default=0.0005)
    cycle.add_argument("--trade-min-delta", type=float, default=0.01)
    cycle.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("/opt/cursor/artifacts/cycle-report"),
    )

    ema = subparsers.add_parser("ema-research", help="BTC/ETH EMA50/100 多周期策略研究")
    ema.add_argument("--data-dir", type=Path, default=Path("data/okx_bars"))
    ema.add_argument("--output-dir", type=Path, default=Path("reports"))
    ema.add_argument("--start", type=date.fromisoformat, default=date(2021, 1, 1))
    ema.add_argument("--end", type=date.fromisoformat, default=date.today())
    ema.add_argument("--refresh", action="store_true")
    ema.add_argument("--fee-rate", type=float, default=0.001)
    ema.add_argument("--slippage-rate", type=float, default=0.0005)
    ema.add_argument("--train-fraction", type=float, default=0.60)
    ema.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("/opt/cursor/artifacts/ema-report"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """分派子命令并返回进程退出码。"""

    args = build_parser().parse_args(argv)
    if args.command == "download":
        return _download(args)
    if args.command == "research":
        return _research(args)
    if args.command == "optimize":
        return _optimize(args)
    if args.command == "crypto-alpha":
        return _crypto_alpha(args)
    if args.command == "cycle-report":
        return _cycle_report(args)
    if args.command == "ema-research":
        return _ema_research(args)
    raise AssertionError(f"未知命令：{args.command}")


def _download(args: argparse.Namespace) -> int:
    """下载每个币对并生成数据清单。"""

    client = OkxDataClient()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for symbol in args.symbols:
        path = args.data_dir / f"{symbol}.csv"
        if path.exists() and not args.refresh:
            existing = load_candles(path)
            if existing:
                existing_start = datetime.fromtimestamp(
                    existing[0].timestamp_ms / 1000,
                    tz=timezone.utc,
                ).date()
                existing_end = datetime.fromtimestamp(
                    existing[-1].timestamp_ms / 1000,
                    tz=timezone.utc,
                ).date()
                if existing_start <= args.start and existing_end >= args.end:
                    print(f"[cache] {symbol}: {len(existing)} rows")
                    continue
        candles = client.fetch_daily(symbol, args.start, args.end)
        if not candles:
            raise RuntimeError(f"{symbol} 未下载到行情")
        save_candles(path, candles)
        print(f"[okx] {symbol}: {len(candles)} rows")
    manifest = dataset_manifest(args.data_dir, args.symbols)
    manifest["request"] = {
        "command": (
            "python3 -m crypto_lab.cli download "
            f"--start {args.start.isoformat()} --end {args.end.isoformat()} "
            f"--symbols {' '.join(args.symbols)}"
        ),
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "symbols": list(args.symbols),
    }
    (args.data_dir / "data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


def _research(args: argparse.Namespace) -> int:
    """加载缓存行情并生成 JSON 与 Markdown 研究结果。"""

    series = {
        symbol: load_candles(args.data_dir / f"{symbol}.csv")
        for symbol in args.symbols
    }
    data = align_market_data(series)
    cached_manifest = args.data_dir / "data_manifest.json"
    manifest = (
        json.loads(cached_manifest.read_text(encoding="utf-8"))
        if cached_manifest.exists()
        else dataset_manifest(args.data_dir, args.symbols)
    )
    results = run_research(
        data,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        train_fraction=args.train_fraction,
    )
    write_json(args.output_dir / "backtest_results.json", results)
    write_json(args.output_dir / "data_manifest.json", manifest)
    write_markdown_report(args.output_dir / "cross_market_report.md", results, manifest)
    print(f"研究完成：{args.output_dir / 'cross_market_report.md'}")
    return 0


def _optimize(args: argparse.Namespace) -> int:
    """加载缓存行情并生成优化策略研究结果。"""

    series = {
        symbol: load_candles(args.data_dir / f"{symbol}.csv")
        for symbol in args.symbols
    }
    data = align_market_data(series)
    cached_manifest = args.data_dir / "data_manifest.json"
    manifest = (
        json.loads(cached_manifest.read_text(encoding="utf-8"))
        if cached_manifest.exists()
        else dataset_manifest(args.data_dir, args.symbols)
    )
    results = run_optimized_research(
        data,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        train_fraction=args.train_fraction,
    )
    write_json(args.output_dir / "optimized_backtest_results.json", results)
    write_json(args.output_dir / "data_manifest.json", manifest)
    write_optimized_markdown_report(
        args.output_dir / "optimized_strategies_report.md",
        results,
        manifest,
    )
    print(f"优化研究完成：{args.output_dir / 'optimized_strategies_report.md'}")
    return 0


def _crypto_alpha(args: argparse.Namespace) -> int:
    """运行加密增强策略研究并写出目标达成报告。"""

    series = {
        symbol: load_candles(args.data_dir / f"{symbol}.csv")
        for symbol in args.symbols
    }
    data = align_market_data(series)
    cached_manifest = args.data_dir / "data_manifest.json"
    manifest = (
        json.loads(cached_manifest.read_text(encoding="utf-8"))
        if cached_manifest.exists()
        else dataset_manifest(args.data_dir, args.symbols)
    )
    results = run_crypto_alpha_research(
        data,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        train_fraction=args.train_fraction,
    )
    write_json(args.output_dir / "crypto_alpha_results.json", results)
    write_json(args.output_dir / "data_manifest.json", manifest)
    write_crypto_alpha_report(
        args.output_dir / "crypto_alpha_report.md",
        results,
        manifest,
    )
    print(f"加密增强研究完成：{args.output_dir / 'crypto_alpha_report.md'}")
    return 0


def _cycle_report(args: argparse.Namespace) -> int:
    """生成 2021-2026 全周期与 beta 分段详细报告。"""

    series = {
        symbol: load_candles(args.data_dir / f"{symbol}.csv")
        for symbol in args.symbols
    }
    data = align_market_data(series)
    report = run_cycle_report(
        data,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        trade_min_delta=args.trade_min_delta,
    )
    chart_dir = args.output_dir / "cycle_charts"
    trade_dir = args.output_dir / "cycle_trades"
    curve_dir = args.output_dir / "cycle_curves"
    chart_paths = plot_cycle_charts(report, chart_dir)
    save_trade_csvs(report["trades"], trade_dir)
    save_equity_csvs(report, curve_dir)
    write_json(args.output_dir / "cycle_full_results.json", report_for_json(report))
    write_cycle_markdown(
        args.output_dir / "cycle_full_report.md",
        report,
        chart_paths,
        trade_dir,
    )
    export_cycle_artifacts(report, chart_paths.values(), args.artifact_dir)
    print(f"全周期报告完成：{args.output_dir / 'cycle_full_report.md'}")
    return 0


def _ema_research(args: argparse.Namespace) -> int:
    """下载 BTC/ETH 多周期数据并运行 EMA 策略研究。"""

    import shutil

    manifest = download_ema_dataset(
        args.data_dir,
        start=args.start,
        end=args.end,
        refresh=args.refresh,
    )
    write_json(args.data_dir / "ema_data_manifest.json", manifest)
    results = run_ema_research(
        args.data_dir,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        train_fraction=args.train_fraction,
    )
    chart_dir = args.output_dir / "ema_charts"
    chart_paths = plot_ema_charts(results, chart_dir)
    write_json(args.output_dir / "ema_results.json", results)
    write_ema_report(
        args.output_dir / "ema_report.md",
        results,
        chart_dir,
        manifest,
    )
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    for path in chart_paths.values():
        shutil.copy2(path, args.artifact_dir / path.name)
    print(f"EMA 研究完成：{args.output_dir / 'ema_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

