"""机构级多标的动量 CTA：多周期多因子评分 + BTC 门控 + 波动率目标。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cta_data import PanelData, map_higher_tf_to_base
from .cta_indicators import (
    atr,
    cross_sectional_rank,
    kdj,
    macd,
    momentum_return,
    realized_vol,
    rsi,
    volume_ratio,
)
from .ema_data import ema, ema_slope


@dataclass
class InstitutionalMomentumCTA:
    """机构风格动量 CTA（多头-现金）。

    设计要点（对标主流 CTA / crypto momentum 实践）：
    1. BTC 大周期趋势门控：弱市强制现金
    2. 1D + 12H + 4H 多因子综合评分（动量/MACD/RSI/KDJ/量能）
    3. 横截面 Top-K 轮动
    4. 逆波动加权 + 组合波动率目标
    5. ATR 跟踪止损削弱“趋势回吐”
    """

    top_k: int = 4
    rebalance_bars: int = 6  # 4H 面板上约日频
    vol_target: float = 0.28
    vol_lookback: int = 48  # 4H≈8天
    max_gross: float = 1.0
    atr_stop_mult: float = 2.5
    min_score: float = 0.55
    btc_trend_span: int = 200
    name: str = "institutional_momentum_cta"
    ideas: tuple[str, ...] = (
        "BTC门控",
        "多周期多因子动量",
        "TOP-K轮动",
        "波动率目标",
        "ATR止损",
    )

    def target_weights(
        self,
        panel_4h: PanelData,
        panel_12h: PanelData,
        panel_1d: PanelData,
        index: int,
        previous: np.ndarray,
        stop_state: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """生成 index 时刻的目标权重；stop_state 在回测引擎间传递。"""

        stop_state = {} if stop_state is None else dict(stop_state)
        n = panel_4h.n_assets
        if index < max(220, self.vol_lookback + 5):
            return np.zeros(n), stop_state

        # ---------- ATR 止损：非调仓日也可清仓被打止损的仓位 ----------
        weights = previous.copy()
        atr_4h = stop_state.get("atr")
        peaks = stop_state.get("peaks")
        if atr_4h is not None and peaks is not None and np.any(np.abs(previous) > 1e-9):
            for col in range(n):
                if previous[col] <= 1e-9:
                    continue
                peaks[col] = max(peaks[col], panel_4h.close[index, col])
                stop = peaks[col] - self.atr_stop_mult * atr_4h[col]
                if panel_4h.close[index, col] < stop:
                    weights[col] = 0.0
                    peaks[col] = 0.0
            stop_state["peaks"] = peaks
            if index % self.rebalance_bars != 0:
                return weights, stop_state
        elif index % self.rebalance_bars != 0:
            return previous, stop_state

        # ---------- BTC 门控（用 1D） ----------
        map_1d = map_higher_tf_to_base(panel_4h, panel_1d)["index"]
        map_12h = map_higher_tf_to_base(panel_4h, panel_12h)["index"]
        i1 = int(map_1d[index])
        i12 = int(map_12h[index])
        if i1 < 0 or i12 < 0:
            return np.zeros(n), stop_state
        btc = panel_1d.symbol_index("BTC-USDT")
        btc_close = panel_1d.close[: i1 + 1, btc]
        btc_ema = ema(btc_close, self.btc_trend_span)
        btc_slope = ema_slope(btc_ema, 5)
        if not (
            np.isfinite(btc_ema[-1])
            and panel_1d.close[i1, btc] > btc_ema[-1]
            and np.isfinite(btc_slope[-1])
            and btc_slope[-1] > 0
        ):
            stop_state["peaks"] = np.zeros(n)
            return np.zeros(n), stop_state

        score = self._composite_score(panel_4h, panel_12h, panel_1d, index, i12, i1)
        tradable = np.isfinite(score) & (score >= self.min_score)
        if not tradable.any():
            stop_state["peaks"] = np.zeros(n)
            return np.zeros(n), stop_state

        ranked = np.flatnonzero(tradable)
        order = ranked[np.argsort(score[ranked])[::-1]]
        selected = order[: self.top_k]
        vols = realized_vol(panel_4h.close[: index + 1], self.vol_lookback, 365 * 6)[-1]
        inv = np.zeros(n)
        for col in selected:
            v = vols[col]
            if np.isfinite(v) and v > 1e-6:
                inv[col] = 1.0 / v
            else:
                inv[col] = 1.0
        if inv.sum() <= 0:
            return np.zeros(n), stop_state
        raw = inv / inv.sum()

        # 组合波动率缩放（对角近似）
        port_vol = float(np.sqrt(np.sum((raw * vols) ** 2)))
        if port_vol > 1e-8 and self.vol_target > 0:
            raw *= min(self.max_gross, self.vol_target / port_vol)
        raw = np.minimum(raw, self.max_gross)
        if raw.sum() > self.max_gross:
            raw *= self.max_gross / raw.sum()

        # 初始化/刷新 ATR 峰值
        atr_now = atr(panel_4h.high, panel_4h.low, panel_4h.close, 14)[index]
        peaks = np.zeros(n)
        for col in range(n):
            if raw[col] > 1e-9:
                peaks[col] = panel_4h.close[index, col]
        stop_state["atr"] = atr_now
        stop_state["peaks"] = peaks
        return raw, stop_state

    def _composite_score(
        self,
        p4: PanelData,
        p12: PanelData,
        p1: PanelData,
        i4: int,
        i12: int,
        i1: int,
    ) -> np.ndarray:
        """多周期多因子综合分（约 0~1）。"""

        # ---- 1D 因子（权重最高）----
        mom_1d = momentum_return(p1.close[: i1 + 1], 20)[-1]
        _, _, macd_h_1d = macd(p1.close[: i1 + 1])
        macd_h_1d = macd_h_1d[-1]
        rsi_1d = rsi(p1.close[: i1 + 1], 14)[-1]
        k_1d, d_1d, _ = kdj(p1.high[: i1 + 1], p1.low[: i1 + 1], p1.close[: i1 + 1])
        k_1d, d_1d = k_1d[-1], d_1d[-1]
        volr_1d = volume_ratio(p1.volume_quote[: i1 + 1], 20)[-1]
        ema50 = np.array([ema(p1.close[: i1 + 1, c], 50)[-1] for c in range(p1.n_assets)])
        ema100 = np.array([ema(p1.close[: i1 + 1, c], 100)[-1] for c in range(p1.n_assets)])
        ema200 = np.array([ema(p1.close[: i1 + 1, c], 200)[-1] for c in range(p1.n_assets)])
        trend_1d = (
            (p1.close[i1] > ema200)
            & (ema50 > ema100)
            & (ema50 > ema200)
        ).astype(float)

        # ---- 12H 因子 ----
        mom_12 = momentum_return(p12.close[: i12 + 1], 21)[-1]  # ~10.5天
        _, _, macd_h_12 = macd(p12.close[: i12 + 1])
        macd_h_12 = macd_h_12[-1]
        rsi_12 = rsi(p12.close[: i12 + 1], 14)[-1]
        k_12, d_12, _ = kdj(p12.high[: i12 + 1], p12.low[: i12 + 1], p12.close[: i12 + 1])
        k_12, d_12 = k_12[-1], d_12[-1]

        # ---- 4H 择时因子 ----
        mom_4 = momentum_return(p4.close[: i4 + 1], 18)[-1]  # 3天
        _, _, macd_h_4 = macd(p4.close[: i4 + 1])
        macd_h_4 = macd_h_4[-1]
        rsi_4 = rsi(p4.close[: i4 + 1], 14)[-1]
        volr_4 = volume_ratio(p4.volume_quote[: i4 + 1], 24)[-1]

        # 对齐到 4H 面板列顺序
        score = np.full(p4.n_assets, np.nan)
        for col, symbol in enumerate(p4.symbols):
            c1 = p1.symbol_index(symbol)
            c12 = p12.symbol_index(symbol)
            parts = []
            # 趋势硬门槛：1D 多头结构
            if not (
                np.isfinite(trend_1d[c1])
                and trend_1d[c1] > 0
                and np.isfinite(mom_1d[c1])
                and mom_1d[c1] > 0
            ):
                continue
            # 动量横截面稍后统一 rank；这里先放原始组件
            parts.append(("mom1", mom_1d[c1], 0.22))
            parts.append(("mom12", mom_12[c12], 0.12))
            parts.append(("mom4", mom_4[col], 0.08))
            parts.append(("macd1", 1.0 if macd_h_1d[c1] > 0 else 0.0, 0.12))
            parts.append(("macd12", 1.0 if macd_h_12[c12] > 0 else 0.0, 0.08))
            parts.append(("macd4", 1.0 if macd_h_4[col] > 0 else 0.0, 0.05))
            # RSI 甜区
            r1 = rsi_1d[c1]
            parts.append(("rsi1", _rsi_sweet(r1), 0.10))
            parts.append(("rsi12", _rsi_sweet(rsi_12[c12]), 0.05))
            parts.append(("rsi4", _rsi_sweet(rsi_4[col]), 0.03))
            # KDJ
            parts.append(("kdj1", 1.0 if k_1d[c1] > d_1d[c1] and 20 < k_1d[c1] < 80 else 0.0, 0.07))
            parts.append(("kdj12", 1.0 if k_12[c12] > d_12[c12] else 0.0, 0.04))
            # 量能
            parts.append(("vol1", float(np.clip((volr_1d[c1] - 0.8) / 1.2, 0, 1)) if np.isfinite(volr_1d[c1]) else 0.0, 0.02))
            parts.append(("vol4", float(np.clip((volr_4[col] - 0.8) / 1.2, 0, 1)) if np.isfinite(volr_4[col]) else 0.0, 0.02))
            score[col] = sum(val * w for _, val, w in parts if np.isfinite(val))
        # 动量部分再做一次横截面增强：把 1D 动量 rank 混入
        mom_rank = np.full(p4.n_assets, np.nan)
        raw_mom = np.full(p4.n_assets, np.nan)
        for col, symbol in enumerate(p4.symbols):
            c1 = p1.symbol_index(symbol)
            raw_mom[col] = mom_1d[c1]
        # 单行 rank
        finite = np.isfinite(raw_mom)
        if finite.sum() >= 2:
            order = np.argsort(np.argsort(raw_mom[finite]))
            mom_rank[finite] = order / max(finite.sum() - 1, 1)
            score = np.where(np.isfinite(score), 0.75 * score + 0.25 * mom_rank, np.nan)
        return score


def _rsi_sweet(value: float) -> float:
    """RSI 甜区评分：45~68 最高，极端追高/超卖降分。"""

    if not np.isfinite(value):
        return 0.0
    if 48 <= value <= 68:
        return 1.0
    if 40 <= value < 48:
        return 0.6
    if 68 < value <= 75:
        return 0.4
    return 0.0
