"""机构动量 CTA 回测引擎与预计算指标缓存。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np

from .cta_data import PanelData, map_higher_tf_to_base
from .cta_indicators import (
    atr,
    kdj,
    macd,
    momentum_return,
    realized_vol,
    rsi,
    volume_ratio,
)
from .ema_data import ema, ema_slope


@dataclass(frozen=True)
class CtaMetrics:
    start: str
    end: str
    observations: int
    total_return: float
    cagr: float
    annual_volatility: float
    sharpe: float
    max_drawdown: float
    calmar: float
    turnover: float
    cost_paid: float
    time_in_market: float
    avg_names: float


@dataclass
class IndicatorCache:
    """三周期指标预计算，避免回测中重复扫描。"""

    # 1D
    mom_1d_20: np.ndarray
    macd_hist_1d: np.ndarray
    rsi_1d: np.ndarray
    k_1d: np.ndarray
    d_1d: np.ndarray
    volr_1d: np.ndarray
    ema50_1d: np.ndarray
    ema100_1d: np.ndarray
    ema200_1d: np.ndarray
    # 12H
    mom_12_21: np.ndarray
    macd_hist_12: np.ndarray
    rsi_12: np.ndarray
    k_12: np.ndarray
    d_12: np.ndarray
    # 4H
    mom_4_18: np.ndarray
    macd_hist_4: np.ndarray
    rsi_4: np.ndarray
    volr_4: np.ndarray
    atr_4: np.ndarray
    vol_4: np.ndarray
    # BTC 1D gate helpers
    btc_ema200: np.ndarray
    btc_slope: np.ndarray
    # maps
    map_1d: np.ndarray
    map_12h: np.ndarray


def build_indicator_cache(p4: PanelData, p12: PanelData, p1: PanelData) -> IndicatorCache:
    """一次性计算全部指标。"""

    _, _, macd_h_1d = macd(p1.close)
    k1, d1, _ = kdj(p1.high, p1.low, p1.close)
    ema50 = np.column_stack([ema(p1.close[:, c], 50) for c in range(p1.n_assets)])
    ema100 = np.column_stack([ema(p1.close[:, c], 100) for c in range(p1.n_assets)])
    ema200 = np.column_stack([ema(p1.close[:, c], 200) for c in range(p1.n_assets)])
    _, _, macd_h_12 = macd(p12.close)
    k12, d12, _ = kdj(p12.high, p12.low, p12.close)
    _, _, macd_h_4 = macd(p4.close)
    btc = p1.symbol_index("BTC-USDT")
    btc_ema200 = ema(p1.close[:, btc], 200)
    btc_slope = ema_slope(btc_ema200, 5)
    return IndicatorCache(
        mom_1d_20=momentum_return(p1.close, 20),
        macd_hist_1d=macd_h_1d,
        rsi_1d=rsi(p1.close, 14),
        k_1d=k1,
        d_1d=d1,
        volr_1d=volume_ratio(p1.volume_quote, 20),
        ema50_1d=ema50,
        ema100_1d=ema100,
        ema200_1d=ema200,
        mom_12_21=momentum_return(p12.close, 21),
        macd_hist_12=macd_h_12,
        rsi_12=rsi(p12.close, 14),
        k_12=k12,
        d_12=d12,
        mom_4_18=momentum_return(p4.close, 18),
        macd_hist_4=macd_h_4,
        rsi_4=rsi(p4.close, 14),
        volr_4=volume_ratio(p4.volume_quote, 24),
        atr_4=atr(p4.high, p4.low, p4.close, 14),
        vol_4=realized_vol(p4.close, 48, 365 * 6),
        btc_ema200=btc_ema200,
        btc_slope=btc_slope,
        map_1d=map_higher_tf_to_base(p4, p1)["index"],
        map_12h=map_higher_tf_to_base(p4, p12)["index"],
    )


@dataclass
class FastInstitutionalCTA:
    """基于预计算缓存的快速机构 CTA。"""

    cache: IndicatorCache
    panel_4h: PanelData
    panel_12h: PanelData
    panel_1d: PanelData
    top_k: int = 3
    rebalance_bars: int = 12
    vol_target: float = 0.30
    max_gross: float = 1.0
    atr_stop_mult: float = 3.0
    min_score: float = 0.48
    exit_score: float = 0.35
    # 组合回撤熔断：软降仓 + 硬空仓冷却后按信号全风险再入场（避免永久锁仓）
    # 目标：全样本 MDD≤25%，并尽量保留/提升 CAGR
    dd_soft: float = 0.12
    dd_hard: float = 0.195
    dd_reentry: float = 0.06
    dd_min_scale: float = 0.40
    dd_cooldown_bars: int = 42  # 4H≈1周；硬熔断后强制空仓的最短冷却
    dd_recover_scale: float = 1.0  # 冷却结束后按原信号恢复；勿在恢复期再套 soft→0
    half_risk_scale: float = 0.50
    name: str = "institutional_momentum_cta"

    def target_weights(self, index: int, previous: np.ndarray, stop_state: dict) -> tuple[np.ndarray, dict]:
        n = self.panel_4h.n_assets
        if index < 240:
            return np.zeros(n), stop_state

        weights = previous.copy()
        peaks = stop_state.get("peaks")
        if peaks is not None and np.any(previous > 1e-9):
            atr_row = self.cache.atr_4[index]
            for col in range(n):
                if previous[col] <= 1e-9:
                    continue
                peaks[col] = max(peaks[col], self.panel_4h.close[index, col])
                stop = peaks[col] - self.atr_stop_mult * atr_row[col]
                if self.panel_4h.close[index, col] < stop:
                    weights[col] = 0.0
                    peaks[col] = 0.0
            stop_state["peaks"] = peaks
            if index % self.rebalance_bars != 0:
                return weights, stop_state
        elif index % self.rebalance_bars != 0:
            return previous, stop_state

        i1 = int(self.cache.map_1d[index])
        i12 = int(self.cache.map_12h[index])
        if i1 < 200 or i12 < 50:
            return np.zeros(n), stop_state

        btc = self.panel_1d.symbol_index("BTC-USDT")
        btc_px = self.panel_1d.close[i1, btc]
        ema100 = self.cache.ema100_1d[i1, btc]
        ema200 = self.cache.btc_ema200[i1]
        slope = self.cache.btc_slope[i1]
        above100 = np.isfinite(ema100) and btc_px > ema100
        above200 = np.isfinite(ema200) and btc_px > ema200
        slope_ok = np.isfinite(slope) and slope >= 0
        if above200 and slope_ok:
            risk_scale = 1.0
            active_top_k = self.top_k
        elif above100:
            risk_scale = self.half_risk_scale
            active_top_k = max(1, min(2, self.top_k))
        else:
            stop_state["peaks"] = np.zeros(n)
            return np.zeros(n), stop_state

        score = self._score_at(index, i1, i12)
        held = previous > 1e-9
        tradable = np.isfinite(score) & (
            (score >= self.min_score) | (held & (score >= self.exit_score))
        )
        if not tradable.any():
            stop_state["peaks"] = np.zeros(n)
            return np.zeros(n), stop_state
        ranked = np.flatnonzero(tradable)
        selected = ranked[np.argsort(score[ranked])[::-1][: active_top_k]]
        vols = self.cache.vol_4[index]
        inv = np.zeros(n)
        for col in selected:
            v = vols[col]
            inv[col] = 1.0 / v if np.isfinite(v) and v > 1e-6 else 1.0
        if inv.sum() <= 0:
            return np.zeros(n), stop_state
        raw = inv / inv.sum()
        port_vol = float(np.sqrt(np.nansum((raw * np.nan_to_num(vols, nan=0.0)) ** 2)))
        target_vol = self.vol_target * risk_scale
        if port_vol > 1e-8 and target_vol > 0:
            raw *= min(self.max_gross * risk_scale, target_vol / port_vol)
        gross_cap = self.max_gross * risk_scale
        if raw.sum() > gross_cap:
            raw *= gross_cap / raw.sum()
        peaks = np.zeros(n)
        for col in range(n):
            if raw[col] > 1e-9:
                peaks[col] = self.panel_4h.close[index, col]
        stop_state["peaks"] = peaks
        return raw, stop_state

    def _score_at(self, i4: int, i1: int, i12: int) -> np.ndarray:
        c = self.cache
        p4, p1, p12 = self.panel_4h, self.panel_1d, self.panel_12h
        score = np.full(p4.n_assets, np.nan)
        raw_mom = np.full(p4.n_assets, np.nan)
        for col, symbol in enumerate(p4.symbols):
            a = p1.symbol_index(symbol)
            b = p12.symbol_index(symbol)
            # 资产趋势：收盘站上 EMA100 且 20 日动量为正（比三重均线更不易漏趋势）
            if not (
                np.isfinite(c.ema100_1d[i1, a])
                and p1.close[i1, a] > c.ema100_1d[i1, a]
                and np.isfinite(c.mom_1d_20[i1, a])
                and c.mom_1d_20[i1, a] > 0
            ):
                continue
            # 12H 或 4H 至少一个动量确认，避免纯日线滞后
            if not (
                (np.isfinite(c.mom_12_21[i12, b]) and c.mom_12_21[i12, b] > 0)
                or (np.isfinite(c.macd_hist_4[i4, col]) and c.macd_hist_4[i4, col] > 0)
            ):
                continue
            raw_mom[col] = c.mom_1d_20[i1, a]
            s = 0.0
            s += 0.18 * _clip01((c.mom_1d_20[i1, a] + 0.05) / 0.40)
            s += 0.12 * _clip01((c.mom_12_21[i12, b] + 0.03) / 0.25)
            s += 0.08 * _clip01((c.mom_4_18[i4, col] + 0.02) / 0.15)
            s += 0.12 * (1.0 if c.macd_hist_1d[i1, a] > 0 else 0.0)
            s += 0.08 * (1.0 if c.macd_hist_12[i12, b] > 0 else 0.0)
            s += 0.06 * (1.0 if c.macd_hist_4[i4, col] > 0 else 0.0)
            s += 0.10 * _rsi_sweet(c.rsi_1d[i1, a])
            s += 0.05 * _rsi_sweet(c.rsi_12[i12, b])
            s += 0.04 * _rsi_sweet(c.rsi_4[i4, col])
            s += 0.07 * (
                1.0
                if np.isfinite(c.k_1d[i1, a])
                and np.isfinite(c.d_1d[i1, a])
                and c.k_1d[i1, a] > c.d_1d[i1, a]
                and 20 < c.k_1d[i1, a] < 80
                else 0.0
            )
            s += 0.04 * (
                1.0
                if np.isfinite(c.k_12[i12, b])
                and np.isfinite(c.d_12[i12, b])
                and c.k_12[i12, b] > c.d_12[i12, b]
                else 0.0
            )
            # 站上 EMA200 加分，但不强制
            if np.isfinite(c.ema200_1d[i1, a]) and p1.close[i1, a] > c.ema200_1d[i1, a]:
                s += 0.04
            s += 0.02 * _clip01((c.volr_1d[i1, a] - 0.8) / 1.2) if np.isfinite(c.volr_1d[i1, a]) else 0.0
            score[col] = s
        finite = np.isfinite(raw_mom)
        if finite.sum() >= 2:
            order = np.argsort(np.argsort(raw_mom[finite]))
            ranks = np.full(p4.n_assets, np.nan)
            ranks[finite] = order / max(finite.sum() - 1, 1)
            score = np.where(np.isfinite(score), 0.75 * score + 0.25 * ranks, np.nan)
        return score


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _rsi_sweet(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    if 48 <= value <= 68:
        return 1.0
    if 40 <= value < 48:
        return 0.6
    if 68 < value <= 75:
        return 0.4
    return 0.0


@dataclass
class CtaBacktestResult:
    strategy: str
    timestamps_ms: np.ndarray
    daily_returns: np.ndarray
    equity: np.ndarray
    weights: np.ndarray
    turnover: np.ndarray
    costs: np.ndarray
    bars_per_year: float = 365.0 * 6.0

    def metrics(self, start: int = 0, end: int | None = None) -> CtaMetrics:
        end = len(self.daily_returns) if end is None else end + 1
        returns = self.daily_returns[start:end]
        equity = np.cumprod(1.0 + returns)
        years = len(returns) / self.bars_per_year
        total_return = float(equity[-1] - 1.0)
        cagr = float(equity[-1] ** (1.0 / years) - 1.0) if equity[-1] > 0 and years > 0 else -1.0
        vol = float(np.std(returns, ddof=1) * np.sqrt(self.bars_per_year))
        sharpe = float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(self.bars_per_year)) if vol > 0 else 0.0
        anchored = np.concatenate(([1.0], equity))
        peaks = np.maximum.accumulate(anchored)
        mdd = float(-np.min(anchored / peaks - 1.0))
        calmar = cagr / mdd if mdd > 1e-12 else 0.0
        w = self.weights[start:end]
        return CtaMetrics(
            start=_iso(int(self.timestamps_ms[start])),
            end=_iso(int(self.timestamps_ms[end - 1])),
            observations=len(returns),
            total_return=total_return,
            cagr=cagr,
            annual_volatility=vol,
            sharpe=sharpe,
            max_drawdown=mdd,
            calmar=calmar,
            turnover=float(np.sum(self.turnover[start:end])),
            cost_paid=float(np.sum(self.costs[start:end])),
            time_in_market=float(np.mean(np.sum(np.abs(w), axis=1) > 1e-6)),
            avg_names=float(np.mean(np.sum(np.abs(w) > 1e-6, axis=1))),
        )

    def metrics_dict(self, start: int = 0, end: int | None = None) -> dict:
        return asdict(self.metrics(start, end))


def run_cta_backtest(
    strategy: FastInstitutionalCTA,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
) -> CtaBacktestResult:
    """4H 面板回测：T 收盘信号，持有至 T+1；叠加组合回撤熔断。"""

    panel = strategy.panel_4h
    n_rows, n_cols = panel.size, panel.n_assets
    weights = np.zeros((n_rows, n_cols))
    returns = np.zeros(n_rows)
    turnover = np.zeros(n_rows)
    costs = np.zeros(n_rows)
    previous = np.zeros(n_cols)
    stop_state: dict = {"dd_cooldown": 0, "dd_recovering": False}
    cost_rate = fee_rate + slippage_rate
    asset_rets = np.zeros((n_rows, n_cols))
    asset_rets[1:] = panel.close[1:] / panel.close[:-1] - 1.0
    equity = 1.0
    peak_equity = 1.0

    for index in range(1, n_rows):
        target, stop_state = strategy.target_weights(index - 1, previous, stop_state)
        target = np.asarray(target, dtype=float)
        # 组合级回撤熔断（仅使用截至上一根的净值，无前视）
        # 硬熔断后：冷却期空仓 → 以 recover_scale 再入场，避免“空仓导致回撤永不收复”的永久锁仓。
        current_dd = 0.0 if peak_equity <= 1e-12 else max(0.0, 1.0 - equity / peak_equity)
        cooldown = int(stop_state.get("dd_cooldown", 0))
        recovering = bool(stop_state.get("dd_recovering", False))

        if current_dd <= strategy.dd_reentry:
            recovering = False
            cooldown = 0
            stop_state["lock_dd"] = 0.0

        lock_dd = float(stop_state.get("lock_dd", 0.0))
        # 首次触及硬阈值，或恢复期回撤再加深超过缓冲后，重启冷却
        hard_hit = current_dd >= strategy.dd_hard and cooldown <= 0
        deeper = recovering and current_dd >= lock_dd + 0.03 and cooldown <= 0
        if hard_hit and (not recovering or deeper):
            cooldown = max(1, int(strategy.dd_cooldown_bars))
            recovering = True
            stop_state["lock_dd"] = current_dd

        if cooldown > 0:
            target = np.zeros(n_cols)
            cooldown -= 1
        elif recovering and current_dd > strategy.dd_reentry:
            # 恢复期不再套用 soft→0 公式（否则硬阈值附近会永久空仓）
            target = target * float(np.clip(strategy.dd_recover_scale, 0.0, 1.0))
        elif current_dd >= strategy.dd_soft:
            span = max(strategy.dd_hard - strategy.dd_soft, 1e-6)
            frac = min(1.0, (current_dd - strategy.dd_soft) / span)
            scale = 1.0 - frac * (1.0 - strategy.dd_min_scale)
            target = target * float(np.clip(scale, strategy.dd_min_scale, 1.0))

        stop_state["dd_cooldown"] = cooldown
        stop_state["dd_recovering"] = recovering

        turnover[index] = float(np.sum(np.abs(target - previous)))
        costs[index] = turnover[index] * cost_rate
        gross = float(target @ asset_rets[index])
        returns[index] = max((1.0 + gross) * (1.0 - costs[index]) - 1.0, -1.0)
        weights[index] = target
        equity *= 1.0 + returns[index]
        peak_equity = max(peak_equity, equity)
        denom = 1.0 + gross
        previous = target * (1.0 + asset_rets[index]) / denom if denom > 0 else np.zeros(n_cols)
        previous[np.abs(previous) < 1e-14] = 0.0

    return CtaBacktestResult(
        strategy=strategy.name,
        timestamps_ms=panel.timestamps_ms.copy(),
        daily_returns=returns,
        equity=np.cumprod(1.0 + returns),
        weights=weights,
        turnover=turnover,
        costs=costs,
    )


def buy_and_hold_btc(panel_4h: PanelData, panel_1d: PanelData, fee_rate: float = 0.001, slippage: float = 0.0005) -> CtaBacktestResult:
    """BTC 买入持有基准（映射到 4H 收益）。"""

    btc4 = panel_4h.symbol_index("BTC-USDT")
    returns = np.zeros(panel_4h.size)
    returns[1:] = panel_4h.close[1:, btc4] / panel_4h.close[:-1, btc4] - 1.0
    costs = np.zeros(panel_4h.size)
    costs[1] = fee_rate + slippage
    returns[1] -= costs[1]
    weights = np.zeros((panel_4h.size, panel_4h.n_assets))
    weights[1:, btc4] = 1.0
    return CtaBacktestResult(
        strategy="BTC_buy_hold",
        timestamps_ms=panel_4h.timestamps_ms.copy(),
        daily_returns=returns,
        equity=np.cumprod(1.0 + returns),
        weights=weights,
        turnover=np.array([0.0] + [1.0] + [0.0] * (panel_4h.size - 2)),
        costs=costs,
    )


def _iso(ts: int) -> str:
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
