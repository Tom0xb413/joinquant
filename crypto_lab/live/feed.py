"""实盘行情加载：优先本地缓存，可选刷新 OKX 公开接口。"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from ..data import OkxDataClient, align_market_data, load_candles, save_candles
from ..data import MarketData


def load_market_bundle(
    data_dir: Path,
    symbols: tuple[str, ...],
    refresh: bool = False,
    lookback_days: int = 800,
) -> MarketData:
    """加载策略所需的对齐日线行情。

    ``refresh=True`` 时拉取最近窗口的 OKX 公开数据并覆盖缓存；网络失败则
    回退到本地 CSV，保证模拟盘在离线环境仍可启动。lookback 需覆盖策略
    慢趋势窗口（默认 200）并留有余量。

    @author Cursor
    @since 0.3.0
    """

    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=lookback_days)
    data_dir.mkdir(parents=True, exist_ok=True)
    client = OkxDataClient()
    series = {}
    for symbol in symbols:
        path = data_dir / f"{symbol}.csv"
        if refresh:
            try:
                candles = client.fetch_daily(symbol, start, end)
                if candles:
                    save_candles(path, candles)
            except Exception:
                if not path.exists():
                    raise
        if not path.exists():
            raise FileNotFoundError(
                f"缺少行情缓存 {path}；请先运行 download 或开启 refresh_market"
            )
        candles = load_candles(path)
        # 仅保留最近 lookback，降低内存与信号计算成本
        cutoff_ms = int(
            datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp()
            * 1000
        )
        candles = [item for item in candles if item.timestamp_ms >= cutoff_ms]
        if len(candles) < 120:
            raise ValueError(f"{symbol} 有效行情不足 120 天")
        series[symbol] = candles
    return align_market_data(series)


def latest_prices(data: MarketData) -> dict[str, float]:
    """返回各标的最新收盘价映射。"""

    return {
        symbol: float(data.close[-1, index])
        for index, symbol in enumerate(data.symbols)
    }


def latest_signal_date(data: MarketData) -> date:
    """返回可用于生成下一交易日权重的最新信号日。"""

    return data.dates[-1]
