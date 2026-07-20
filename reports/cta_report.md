# 机构级多标的动量 CTA 报告（TOP15 · 4H/12H/1D）

## 设计框架（对标机构实践）

- BTC tiered risk gate: full risk above EMA200+slope, half risk above EMA100 only, cash below EMA100
- 1D/12H/4H momentum+MACD+RSI+KDJ+volume composite score (no higher-TF look-ahead)
- cross-sectional Top-K rotation with score hysteresis exits
- inverse-vol weights + portfolio vol target
- ATR trailing stop

- 标的池：BTC-USDT, ETH-USDT, SOL-USDT, XRP-USDT, ADA-USDT, DOGE-USDT, LTC-USDT, BCH-USDT, LINK-USDT, DOT-USDT, AVAX-USDT, UNI-USDT, ATOM-USDT, NEAR-USDT, AAVE-USDT
- 选参：train dual-fold min sharpe + 0.45*train sharpe + 0.25*cagr - 0.50*mdd; OOS evaluation only
- 成本：fee=0.001, slippage=0.0005
- 数据：https://www.okx.com/api/v5/market/history-candles · 2021-01-01T00:00:00+00:00 ~ 2026-07-15T20:00:00+00:00

## 推荐参数

`{'top_k': 3, 'rebalance_bars': 12, 'vol_target': 0.32, 'atr_stop_mult': 3.5, 'min_score': 0.48, 'exit_score': 0.35, 'max_gross': 1.0}`
- 网格测试 `324`，合格 `188`，采用优化参数

## 全样本 / 样本外 vs BTC

| 区间 | CTA CAGR | Sharpe | MDD | 在市 | BTC CAGR | BTC Sharpe | BTC MDD | 超额CAGR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 全样本 | 21.7% | 0.82 | 41.8% | 44% | 15.4% | 0.54 | 77.0% | +6.3% |
| 训练期 | 32.2% | 1.13 | 40.6% | 40% | 26.5% | 0.69 | 77.0% | +5.7% |
| 样本外 | 7.5% | 0.39 | 36.0% | 51% | 0.5% | 0.24 | 53.4% | +7.0% |

## 图表

![权益](cta_charts/cta_equity.png)

![回撤](cta_charts/cta_drawdown.png)

![分段](cta_charts/cta_regime_cagr.png)

## 不同大行情覆盖

| 行情 | CTA CAGR | Sharpe | MDD | BTC CAGR | BTC Sharpe | BTC MDD | 胜夏普 | 回撤改善 |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| 全周期 | 21.7% | 0.82 | 41.8% | 15.4% | 0.54 | 77.0% | 是 | +35.2% |
| 2021牛市 | 57.9% | 2.02 | 11.0% | 159.7% | 1.51 | 54.1% | 是 | +43.1% |
| 2021-22熊市 | -2.6% | -0.13 | 15.9% | -74.3% | -1.78 | 75.8% | 是 | +59.9% |
| 2023-24复苏 | 62.4% | 1.50 | 35.6% | 219.5% | 2.94 | 21.2% | 否 | -14.5% |
| 2024-25主升 | 15.2% | 0.56 | 41.8% | 40.1% | 0.94 | 30.6% | 否 | -11.2% |
| 2025-26回调 | -14.0% | -1.46 | 16.8% | -57.2% | -1.66 | 53.2% | 是 | +36.3% |

## 训练期备选参数（Top）

| 参数 | 训练Sharpe | 训练CAGR | OOS Sharpe | OOS CAGR | 全样本Sharpe |
|---|---:|---:|---:|---:|---:|
| {'top_k': 3, 'rebalance_bars': 12, 'vol_target': 0.38, 'atr_stop_mult': 3.5, 'min_score': 0.48, 'exit_score': 0.35} | 1.14 | 37.8% | 0.32 | 5.1% | 0.79 |
| {'top_k': 3, 'rebalance_bars': 12, 'vol_target': 0.32, 'atr_stop_mult': 3.5, 'min_score': 0.48, 'exit_score': 0.35} | 1.13 | 32.2% | 0.39 | 7.5% | 0.82 |
| {'top_k': 3, 'rebalance_bars': 12, 'vol_target': 0.26, 'atr_stop_mult': 3.5, 'min_score': 0.48, 'exit_score': 0.35} | 1.11 | 26.6% | 0.41 | 7.6% | 0.82 |
| {'top_k': 3, 'rebalance_bars': 12, 'vol_target': 0.38, 'atr_stop_mult': 3.5, 'min_score': 0.48, 'exit_score': 0.28} | 1.13 | 37.3% | 0.33 | 5.5% | 0.79 |
| {'top_k': 3, 'rebalance_bars': 12, 'vol_target': 0.32, 'atr_stop_mult': 3.5, 'min_score': 0.48, 'exit_score': 0.28} | 1.11 | 31.7% | 0.40 | 7.8% | 0.81 |

## 结论

- 机构 CTA 的核心价值是：**熊市/回调段大幅降低回撤**（本回测熊市约 -3% vs BTC -74%），全周期夏普高于 BTC。
- 强牛市会让渡部分收益，这是趋势+波动率目标策略的正常特征，不是单纯调参能消除的。
- 已消除高周期前视：4H 信号仅使用已收盘的 12H/1D 指标。
- 分层门控（EMA200 满仓 / EMA100 半仓 / 以下空仓）是抗熊的关键。

详情：`reports/cta_results.json`
