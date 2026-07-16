"""不依赖未来数据的通用量价指标。"""

from __future__ import annotations

import numpy as np


def trailing_mean(values: np.ndarray, end: int, window: int) -> np.ndarray:
    """计算截至 end（含）的窗口均值，历史不足时返回 NaN。"""

    if end + 1 < window:
        return np.full(values.shape[1], np.nan)
    return np.nanmean(values[end - window + 1 : end + 1], axis=0)


def trailing_return(close: np.ndarray, end: int, window: int) -> np.ndarray:
    """计算截至 end 的 window 日简单收益，严格使用已有收盘价。"""

    if end < window:
        return np.full(close.shape[1], np.nan)
    return close[end] / close[end - window] - 1.0


def trailing_volatility(close: np.ndarray, end: int, window: int) -> np.ndarray:
    """计算日收益年化波动率，Crypto 按 365 天年化。"""

    if end < window:
        return np.full(close.shape[1], np.nan)
    prices = close[end - window : end + 1]
    returns = prices[1:] / prices[:-1] - 1.0
    return np.nanstd(returns, axis=0, ddof=1) * np.sqrt(365.0)


def rsi(close: np.ndarray, end: int, window: int = 14) -> np.ndarray:
    """使用简单移动平均涨跌幅计算横截面 RSI。"""

    if end < window:
        return np.full(close.shape[1], np.nan)
    prices = close[end - window : end + 1]
    changes = np.diff(prices, axis=0)
    gains = np.mean(np.maximum(changes, 0.0), axis=0)
    losses = np.mean(np.maximum(-changes, 0.0), axis=0)
    relative_strength = np.divide(
        gains,
        losses,
        out=np.full_like(gains, np.inf),
        where=losses > 0,
    )
    return 100.0 - 100.0 / (1.0 + relative_strength)


def cross_sectional_zscore(values: np.ndarray) -> np.ndarray:
    """对有限值做横截面标准化，常数截面返回零。"""

    result = np.full_like(values, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return result
    subset = values[finite]
    std = float(np.std(subset))
    result[finite] = 0.0 if std < 1e-12 else (subset - float(np.mean(subset))) / std
    return result


def equal_weights(indices: np.ndarray, asset_count: int) -> np.ndarray:
    """为指定资产生成多头等权目标，其余资金保持现金。"""

    weights = np.zeros(asset_count, dtype=float)
    if len(indices):
        weights[indices] = 1.0 / len(indices)
    return weights


def finite_top(values: np.ndarray, count: int, descending: bool = True) -> np.ndarray:
    """从有限值中选前 count 个索引，结果顺序稳定。"""

    indices = np.flatnonzero(np.isfinite(values))
    if not len(indices):
        return indices
    order = np.argsort(values[indices], kind="stable")
    if descending:
        order = order[::-1]
    return indices[order[:count]]

