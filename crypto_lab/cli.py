"""命令行入口：下载真实行情并运行可复现研究。"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from .data import (
    OkxDataClient,
    align_market_data,
    dataset_manifest,
    load_candles,
    save_candles,
)
from .research import run_research, write_json, write_markdown_report


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
    return parser


def main(argv: list[str] | None = None) -> int:
    """分派子命令并返回进程退出码。"""

    args = build_parser().parse_args(argv)
    if args.command == "download":
        return _download(args)
    if args.command == "research":
        return _research(args)
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
                existing_start = date.fromtimestamp(existing[0].timestamp_ms / 1000)
                existing_end = date.fromtimestamp(existing[-1].timestamp_ms / 1000)
                if existing_start <= args.start and existing_end >= args.end:
                    print(f"[cache] {symbol}: {len(existing)} rows")
                    continue
        candles = client.fetch_daily(symbol, args.start, args.end)
        if not candles:
            raise RuntimeError(f"{symbol} 未下载到行情")
        save_candles(path, candles)
        print(f"[okx] {symbol}: {len(candles)} rows")
    manifest = dataset_manifest(args.data_dir, args.symbols)
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
    manifest = dataset_manifest(args.data_dir, args.symbols)
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


if __name__ == "__main__":
    raise SystemExit(main())

