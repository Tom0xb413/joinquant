# 机构级多标的动量 CTA 报告（TOP15 · 4H/12H/1D）

## 设计框架（对标机构实践）

- BTC tiered risk gate: full risk above EMA200+slope, half risk above EMA100 only, cash below EMA100
- 1D/12H/4H momentum+MACD+RSI+KDJ+volume composite score (no higher-TF look-ahead)
- cross-sectional Top-K rotation with score hysteresis exits
- inverse-vol weights + portfolio vol target
- ATR trailing stop
- persistent ATR high/stop with post-stop cooldown
- absolute UTC rebalance schedule + Top-K rank buffer
- market-breadth risk scaling and per-asset concentration cap
- portfolio drawdown circuit: soft delever + hard flatten cooldown + signal-based re-entry (no permanent lock)

- 标的池：BTC-USDT, ETH-USDT, SOL-USDT, XRP-USDT, ADA-USDT, DOGE-USDT, LTC-USDT, BCH-USDT, LINK-USDT, DOT-USDT, AVAX-USDT, UNI-USDT, ATOM-USDT, NEAR-USDT, AAVE-USDT
- 选参：train dual-fold only; train MDD<=23% risk-margin filter; final evaluation segment never participates in selection/fallback
- 成本：fee=0.001, slippage=0.0005
- 数据：https://www.okx.com/api/v5/market/history-candles · 2021-01-01T00:00:00+00:00 ~ 2026-07-15T20:00:00+00:00

## 推荐参数

`{'top_k': 3, 'rebalance_bars': 12, 'rebalance_phase': 0, 'rank_buffer': 1, 'vol_target': 0.25, 'atr_stop_mult': 2.8, 'stop_cooldown_bars': 6, 'min_score': 0.48, 'exit_score': 0.25, 'max_gross': 1.0, 'max_asset_weight': 0.6, 'half_risk_scale': 0.5, 'breadth_threshold': 0.2, 'breadth_risk_scale': 0.6, 'correlation_aware': False, 'dd_soft': 0.11, 'dd_hard': 0.17, 'dd_reentry': 0.05, 'dd_min_scale': 0.2, 'dd_cooldown_bars': 36, 'dd_recover_scale': 1.0}`
- 网格测试 `96`，合格 `6`，采用优化参数

## 全样本 / 时间顺序评估段 vs BTC

| 区间 | CTA CAGR | Sharpe | MDD | 在市 | BTC CAGR | BTC Sharpe | BTC MDD | 超额CAGR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 全样本 | 22.1% | 1.14 | 22.4% | 37% | 15.4% | 0.54 | 77.0% | +6.8% |
| 训练期 | 21.3% | 1.08 | 22.4% | 35% | 26.5% | 0.69 | 77.0% | -5.2% |
| 评估段* | 23.4% | 1.24 | 19.7% | 41% | 0.5% | 0.24 | 53.4% | +22.9% |

\* 评估段未参与本轮选参或回退，但此前研究迭代已查看过该区间，因此不再宣称为严格未触碰 OOS。

## 图表

![权益](cta_charts/cta_equity.png)

![回撤](cta_charts/cta_drawdown.png)

![分段](cta_charts/cta_regime_cagr.png)

## 不同大行情覆盖

| 行情 | CTA CAGR | Sharpe | MDD | BTC CAGR | BTC Sharpe | BTC MDD | 胜夏普 | 回撤改善 |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| 全周期 | 22.2% | 1.14 | 22.4% | 15.4% | 0.54 | 77.0% | 是 | +54.7% |
| 2021牛市 | 41.7% | 2.19 | 5.9% | 159.7% | 1.51 | 54.1% | 是 | +48.2% |
| 2021-22熊市 | -8.5% | -1.15 | 9.8% | -74.3% | -1.78 | 75.8% | 是 | +66.0% |
| 2023-24复苏 | 49.4% | 1.65 | 16.0% | 219.5% | 2.94 | 21.2% | 否 | +5.2% |
| 2024-25主升 | 32.2% | 1.35 | 20.1% | 40.1% | 0.94 | 30.6% | 是 | +10.5% |
| 2025-26回调 | -6.2% | -1.26 | 7.6% | -57.2% | -1.66 | 53.2% | 是 | +45.6% |

## 对抗性稳健性检查

### 调仓相位扰动（全部绝对 UTC 相位）

| 指标 | 全样本中位数 | 全样本最差 | 评估段中位数 | 评估段最差 |
|---|---:|---:|---:|---:|
| CAGR | 14.0% | 10.1% | 14.7% | 2.1% |
| MDD | 33.2% | 45.4% | 18.6% | 23.7% |

### 单边成本压力

| 单边成本 | 全样本 CAGR | 全样本 MDD | 评估段 CAGR | 评估段 MDD |
|---:|---:|---:|---:|---:|
| 0.00% | 21.5% | 20.7% | 24.9% | 17.1% |
| 0.15% | 22.1% | 22.4% | 23.4% | 19.7% |
| 0.30% | 15.3% | 31.1% | 18.8% | 21.6% |
| 0.50% | 4.8% | 44.8% | 7.2% | 23.6% |

### 消融实验

| 版本 | 全样本 CAGR | Sharpe | MDD | 评估段 CAGR | 评估段 MDD |
|---|---:|---:|---:|---:|---:|
| recommended | 22.1% | 1.14 | 22.4% | 23.4% | 19.7% |
| no_breadth_scaling | 20.7% | 1.06 | 29.2% | 22.2% | 21.3% |
| no_rank_buffer | 19.5% | 1.02 | 27.9% | 20.7% | 19.5% |
| no_drawdown_overlay | 18.1% | 0.93 | 32.4% | 20.8% | 23.4% |
| correlation_aware_risk | 10.9% | 0.80 | 24.0% | 6.7% | 16.5% |

## 训练期备选参数（Top）

| 参数 | 训练Sharpe | 训练CAGR | 评估段Sharpe | 评估段CAGR | 全样本Sharpe |
|---|---:|---:|---:|---:|---:|
| {'top_k': 3, 'rebalance_bars': 12, 'rank_buffer': 1, 'vol_target': 0.25, 'atr_stop_mult': 2.8, 'stop_cooldown_bars': 6, 'max_asset_weight': 0.6, 'min_score': 0.48, 'exit_score': 0.25, 'breadth_threshold': 0.2, 'breadth_risk_scale': 0.6, 'dd_soft': 0.11, 'dd_hard': 0.17, 'dd_cooldown_bars': 36, 'dd_min_scale': 0.2} | 1.08 | 21.3% | 1.24 | 23.4% | 1.14 |
| {'top_k': 3, 'rebalance_bars': 12, 'rank_buffer': 1, 'vol_target': 0.25, 'atr_stop_mult': 2.8, 'stop_cooldown_bars': 6, 'max_asset_weight': 0.6, 'min_score': 0.48, 'exit_score': 0.25, 'breadth_threshold': 0.2, 'breadth_risk_scale': 0.6, 'dd_soft': 0.09, 'dd_hard': 0.17, 'dd_cooldown_bars': 36, 'dd_min_scale': 0.2} | 1.07 | 20.4% | 1.27 | 24.0% | 1.15 |
| {'top_k': 3, 'rebalance_bars': 12, 'rank_buffer': 1, 'vol_target': 0.23, 'atr_stop_mult': 2.8, 'stop_cooldown_bars': 6, 'max_asset_weight': 0.6, 'min_score': 0.48, 'exit_score': 0.25, 'breadth_threshold': 0.2, 'breadth_risk_scale': 0.6, 'dd_soft': 0.11, 'dd_hard': 0.17, 'dd_cooldown_bars': 36, 'dd_min_scale': 0.2} | 1.03 | 18.6% | 1.23 | 21.7% | 1.11 |
| {'top_k': 3, 'rebalance_bars': 12, 'rank_buffer': 0, 'vol_target': 0.23, 'atr_stop_mult': 2.8, 'stop_cooldown_bars': 6, 'max_asset_weight': 0.6, 'min_score': 0.48, 'exit_score': 0.25, 'breadth_threshold': 0.2, 'breadth_risk_scale': 0.6, 'dd_soft': 0.09, 'dd_hard': 0.17, 'dd_cooldown_bars': 36, 'dd_min_scale': 0.2} | 1.04 | 18.7% | 0.72 | 10.3% | 0.92 |
| {'top_k': 3, 'rebalance_bars': 12, 'rank_buffer': 0, 'vol_target': 0.23, 'atr_stop_mult': 2.8, 'stop_cooldown_bars': 6, 'max_asset_weight': 0.6, 'min_score': 0.48, 'exit_score': 0.35, 'breadth_threshold': 0.2, 'breadth_risk_scale': 0.6, 'dd_soft': 0.09, 'dd_hard': 0.17, 'dd_cooldown_bars': 36, 'dd_min_scale': 0.2} | 1.04 | 18.7% | 0.72 | 10.3% | 0.92 |

## 结论

- 风险预算：全样本 MDD=22.4%（目标≤25%），CAGR=22.1%。
- 组合回撤熔断：软阈值线性降仓 → 硬阈值冷却空仓 → 冷却后按原信号恢复（修复了“空仓导致回撤永不收复”的永久锁仓缺陷）。
- 机构 CTA 的核心价值是：熊市/回调段大幅降低回撤，全周期夏普高于 BTC。
- 已消除高周期前视：4H 信号仅使用已收盘的 12H/1D 指标。
- 分层门控（EMA200 满仓 / EMA100 半仓 / 以下空仓）是抗熊的关键。
- 注意：硬熔断阈值对路径较敏感，实盘需把 dd_hard/cooldown 纳入稳健性监控，而非单点最优。
- 调仓相位压力结果必须与单点结果同时阅读；相位最差值说明历史 MDD≤25% 不是未来风险保证。

详情：`reports/cta_results.json`
