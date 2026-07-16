"""OKX 公开行情下载、缓存与对齐。"""

from __future__ import annotations

import csv
import hashlib
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


DAY_MS = 86_400_000


@dataclass(frozen=True)
class Candle:
    """一根已确认的 UTC 日 K 线。"""

    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume_base: float
    volume_quote: float


@dataclass(frozen=True)
class MarketData:
    """按 UTC 日期对齐后的多资产行情矩阵。"""

    dates: tuple[date, ...]
    symbols: tuple[str, ...]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume_quote: np.ndarray

    def symbol_index(self, symbol: str) -> int:
        """返回标的列位置，不存在时抛出清晰错误。"""

        try:
            return self.symbols.index(symbol)
        except ValueError as exc:
            raise KeyError(f"行情中不存在标的：{symbol}") from exc


class OkxDataClient:
    """使用 OKX 无需鉴权的 REST API 获取真实现货日线。"""

    endpoint = "https://www.okx.com/api/v5/market/history-candles"

    def __init__(self, timeout_seconds: float = 20.0, pause_seconds: float = 0.12):
        self.timeout_seconds = timeout_seconds
        self.pause_seconds = pause_seconds

    def fetch_daily(
        self,
        symbol: str,
        start: date,
        end: date,
        max_retries: int = 4,
    ) -> list[Candle]:
        """倒序分页下载指定日期区间内所有已确认的 UTC 日线。"""

        if start > end:
            raise ValueError("start 不能晚于 end")
        start_ms = _date_to_ms(start)
        end_ms = _date_to_ms(end) + DAY_MS - 1
        cursor = end_ms + 1
        candles: dict[int, Candle] = {}

        while cursor >= start_ms:
            payload = self._request(
                {"instId": symbol, "bar": "1Dutc", "limit": "300", "after": str(cursor)},
                max_retries,
            )
            rows = payload.get("data", [])
            if not rows:
                break
            oldest = cursor
            for row in rows:
                timestamp_ms = int(row[0])
                oldest = min(oldest, timestamp_ms)
                if start_ms <= timestamp_ms <= end_ms and row[8] == "1":
                    candles[timestamp_ms] = Candle(
                        timestamp_ms=timestamp_ms,
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                        volume_base=float(row[5]),
                        volume_quote=float(row[7]),
                    )
            if oldest >= cursor or oldest < start_ms:
                break
            cursor = oldest
            time.sleep(self.pause_seconds)

        return [candles[key] for key in sorted(candles)]

    def _request(self, params: dict[str, str], max_retries: int) -> dict:
        """请求 JSON，并对限流和暂时网络错误执行指数退避。"""

        url = f"{self.endpoint}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "joinquant-crypto-lab/0.1"},
        )
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read())
                if payload.get("code") != "0":
                    raise RuntimeError(f"OKX 返回错误：{payload}")
                return payload
            except Exception as exc:  # 网络失败类型跨 Python 版本不同
                last_error = exc
                if attempt + 1 < max_retries:
                    time.sleep(2**attempt)
        raise RuntimeError(f"下载 OKX 行情失败：{url}") from last_error


def save_candles(path: Path, candles: Iterable[Candle]) -> None:
    """以稳定 CSV 格式保存原始 K 线，便于审计和增量复用。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["timestamp_ms", "date_utc", "open", "high", "low", "close", "volume_base", "volume_quote"]
        )
        for candle in candles:
            day = datetime.fromtimestamp(candle.timestamp_ms / 1000, tz=timezone.utc).date()
            writer.writerow(
                [
                    candle.timestamp_ms,
                    day.isoformat(),
                    candle.open,
                    candle.high,
                    candle.low,
                    candle.close,
                    candle.volume_base,
                    candle.volume_quote,
                ]
            )


def load_candles(path: Path) -> list[Candle]:
    """读取本工具生成的 CSV，并验证价格和成交量非负。"""

    candles: list[Candle] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            candle = Candle(
                timestamp_ms=int(row["timestamp_ms"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume_base=float(row["volume_base"]),
                volume_quote=float(row["volume_quote"]),
            )
            if min(candle.open, candle.high, candle.low, candle.close) <= 0:
                raise ValueError(f"{path} 含非正价格：{candle}")
            if min(candle.volume_base, candle.volume_quote) < 0:
                raise ValueError(f"{path} 含负成交量：{candle}")
            candles.append(candle)
    return candles


def align_market_data(series: dict[str, list[Candle]]) -> MarketData:
    """按所有标的共同日期对齐，防止缺失资产被隐式向前填充。"""

    if not series:
        raise ValueError("至少需要一个行情序列")
    date_maps: dict[str, dict[date, Candle]] = {}
    common_dates: set[date] | None = None
    for symbol, candles in series.items():
        mapping = {
            datetime.fromtimestamp(item.timestamp_ms / 1000, tz=timezone.utc).date(): item
            for item in candles
        }
        if not mapping:
            raise ValueError(f"{symbol} 没有有效行情")
        date_maps[symbol] = mapping
        common_dates = set(mapping) if common_dates is None else common_dates & set(mapping)
    dates = tuple(sorted(common_dates or ()))
    if len(dates) < 120:
        raise ValueError(f"共同交易日期仅 {len(dates)} 天，无法进行稳健回测")

    symbols = tuple(series)
    shape = (len(dates), len(symbols))
    matrices = [np.empty(shape, dtype=float) for _ in range(5)]
    for column, symbol in enumerate(symbols):
        for row, day in enumerate(dates):
            item = date_maps[symbol][day]
            values = (item.open, item.high, item.low, item.close, item.volume_quote)
            for matrix, value in zip(matrices, values):
                matrix[row, column] = value
    return MarketData(dates, symbols, *matrices)


def dataset_manifest(data_dir: Path, symbols: Iterable[str]) -> dict:
    """生成行情文件范围、行数与 SHA-256，证明回测输入可追溯。"""

    files = []
    for symbol in symbols:
        path = data_dir / f"{symbol}.csv"
        candles = load_candles(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append(
            {
                "symbol": symbol,
                "rows": len(candles),
                "start": _ms_to_date(candles[0].timestamp_ms).isoformat(),
                "end": _ms_to_date(candles[-1].timestamp_ms).isoformat(),
                "sha256": digest,
            }
        )
    return {
        "source": OkxDataClient.endpoint,
        "bar": "1Dutc",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }


def _date_to_ms(value: date) -> int:
    """将 UTC 日期转换为毫秒时间戳。"""

    return int(datetime(value.year, value.month, value.day, tzinfo=timezone.utc).timestamp() * 1000)


def _ms_to_date(value: int) -> date:
    """将毫秒时间戳转换为 UTC 日期。"""

    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).date()

