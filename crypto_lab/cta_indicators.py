"""机构动量 CTA 指标库：MACD / KDJ / RSI / ATR / 波动率 / 量能。"""

from __future__ import annotations

import numpy as np

from .ema_data import ema


def sma(values: np.ndarray, window: int) -> np.ndarray:
    """简单移动平均；窗口不足为 NaN。"""

    out = np.full_like(values, np.nan, dtype=float)
    if window < 1:
        raise ValueError("window 必须 >= 1")
    if values.ndim == 1:
        csum = np.cumsum(np.insert(values, 0, 0.0))
        for i in range(window - 1, len(values)):
            out[i] = (csum[i + 1] - csum[i + 1 - window]) / window
        return out
    for col in range(values.shape[1]):
        out[:, col] = sma(values[:, col], window)
    return out


def rsi(close: np.ndarray, window: int = 14) -> np.ndarray:
    """RSI（Wilder 平滑近似：用 EMA 实现）。"""

    if close.ndim == 1:
        delta = np.diff(close, prepend=close[0])
        gain = np.maximum(delta, 0.0)
        loss = np.maximum(-delta, 0.0)
        avg_gain = ema(gain, window)
        avg_loss = ema(loss, window)
        rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.inf), where=avg_loss > 1e-12)
        out = 100.0 - 100.0 / (1.0 + rs)
        out[:window] = np.nan
        return out
    out = np.full_like(close, np.nan, dtype=float)
    for col in range(close.shape[1]):
        out[:, col] = rsi(close[:, col], window)
    return out


def macd(
    close: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 MACD 线、信号线、柱状图。"""

    if close.ndim == 1:
        line = ema(close, fast) - ema(close, slow)
        sig = ema(line, signal)
        hist = line - sig
        return line, sig, hist
    n, m = close.shape
    line = np.zeros((n, m))
    sig = np.zeros((n, m))
    hist = np.zeros((n, m))
    for col in range(m):
        line[:, col], sig[:, col], hist[:, col] = macd(close[:, col], fast, slow, signal)
    return line, sig, hist


def kdj(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int = 9,
    k_smooth: int = 3,
    d_smooth: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """KDJ 指标；RSV 不足窗口为 NaN。"""

    if close.ndim == 1:
        n = len(close)
        rsv = np.full(n, np.nan)
        for i in range(window - 1, n):
            hh = np.max(high[i - window + 1 : i + 1])
            ll = np.min(low[i - window + 1 : i + 1])
            denom = hh - ll
            rsv[i] = 50.0 if denom < 1e-12 else (close[i] - ll) / denom * 100.0
        k = np.full(n, np.nan)
        d = np.full(n, np.nan)
        k_val = 50.0
        d_val = 50.0
        for i in range(n):
            if not np.isfinite(rsv[i]):
                continue
            k_val = (rsv[i] + (k_smooth - 1) * k_val) / k_smooth
            d_val = (k_val + (d_smooth - 1) * d_val) / d_smooth
            k[i] = k_val
            d[i] = d_val
        j = 3 * k - 2 * d
        return k, d, j
    n, m = close.shape
    k = np.full((n, m), np.nan)
    d = np.full((n, m), np.nan)
    j = np.full((n, m), np.nan)
    for col in range(m):
        k[:, col], d[:, col], j[:, col] = kdj(
            high[:, col], low[:, col], close[:, col], window, k_smooth, d_smooth
        )
    return k, d, j


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> np.ndarray:
    """平均真实波幅。"""

    if close.ndim == 1:
        prev = np.roll(close, 1)
        prev[0] = close[0]
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
        return ema(tr, window)
    out = np.full_like(close, np.nan, dtype=float)
    for col in range(close.shape[1]):
        out[:, col] = atr(high[:, col], low[:, col], close[:, col], window)
    return out


def realized_vol(close: np.ndarray, window: int, bars_per_year: float) -> np.ndarray:
    """滚动已实现年化波动率。"""

    if close.ndim == 1:
        rets = np.zeros_like(close)
        rets[1:] = close[1:] / close[:-1] - 1.0
        out = np.full_like(close, np.nan, dtype=float)
        for i in range(window, len(close)):
            chunk = rets[i - window + 1 : i + 1]
            out[i] = float(np.std(chunk, ddof=1) * np.sqrt(bars_per_year))
        return out
    out = np.full_like(close, np.nan, dtype=float)
    for col in range(close.shape[1]):
        out[:, col] = realized_vol(close[:, col], window, bars_per_year)
    return out


def momentum_return(close: np.ndarray, lookback: int) -> np.ndarray:
    """简单动量收益。"""

    out = np.full_like(close, np.nan, dtype=float)
    if close.ndim == 1:
        out[lookback:] = close[lookback:] / close[:-lookback] - 1.0
        return out
    for col in range(close.shape[1]):
        out[:, col] = momentum_return(close[:, col], lookback)
    return out


def volume_ratio(volume: np.ndarray, window: int = 20) -> np.ndarray:
    """量能相对均量比。"""

    base = sma(volume, window)
    out = np.full_like(volume, np.nan, dtype=float)
    valid = np.isfinite(base) & (base > 1e-12)
    out[valid] = volume[valid] / base[valid]
    return out


def cross_sectional_rank(values: np.ndarray) -> np.ndarray:
    """横截面百分位排名（0~1）；NaN 保持 NaN。"""

    out = np.full_like(values, np.nan, dtype=float)
    for i in range(values.shape[0]):
        row = values[i]
        finite = np.isfinite(row)
        if finite.sum() < 2:
            continue
        order = np.argsort(np.argsort(row[finite]))
        out[i, finite] = order / max(finite.sum() - 1, 1)
    return out
