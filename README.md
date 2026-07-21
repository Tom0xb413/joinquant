# 聚宽策略跨市场 Crypto 研究

本仓库保留 12 份聚宽 A 股原始策略，并提供一个仅依赖 NumPy 的研究框架，将其中可跨市场验证的结构迁移到 Crypto。框架直接下载 OKX 公开现货日 K，使用严格的 T-1 信号、真实交易成本、时间序列切分和样本外评价。

## 策略原型与借鉴来源

| Crypto 策略族 | 聚宽借鉴来源 | 核心迁移 |
|---|---|---|
| `trend_rotation` | 01 | 多标的均线 + 动量 Top-K |
| `all_weather_rotation` | 02 | 主流币/山寨币/现金风险状态轮动 |
| `small_liquidity_rotation` | 03、04、06、10-A | 低规模效应改用流动性池内低成交额代理 |
| `composite_factor_rotation` | 05、07、10-C/D、11、12 | 财务因子改为动量、量增、波动率、流动性 |
| `rsi_factor_rotation` | 08 | 慢频质量筛选 + 日频 RSI 退出 |
| `rolling_ridge_rotation` | 09 | 滚动机器学习，使用正则化量价因子降低过拟合 |

不能直接迁移的内容包括 A 股财报、审计意见、分红、ST、涨跌停、T+1 和指数绝对点位阈值。相关策略只保留调仓与风控结构，不把成交量代理伪装成真正的链上基本面。

## 快速运行

环境要求：Python 3.10+、NumPy 2.4.4+、Flask 3.x（Web 控制台）。

```bash
python3 -m crypto_lab.cli download --start 2021-01-01 --end 2026-07-15
python3 -m crypto_lab.cli research
python3 -m crypto_lab.cli optimize
python3 -m crypto_lab.cli crypto-alpha
python3 -m crypto_lab.cli core-top5
python3 -m crypto_lab.cli live-console
python3 -m unittest discover -s tests -v
```

一键启动模拟盘 + Web（默认 `http://127.0.0.1:8787/`，用户 `admin` / 密码见配置）：

```bash
cp configs/live_console.example.json configs/live_console.json
# 修改 auth.password，可选填写 wecom.webhook_url
python3 -m crypto_lab.cli live-console
```

仅导出历史买卖点：

```bash
python3 -m crypto_lab.cli trade-book --output reports/trade_book.json
```

输出：

- `reports/cross_market_report.md`：首轮迁移原型样本外结论；
- `reports/optimized_strategies_report.md`：低换手优化策略设计与验证；
- `reports/crypto_alpha_report.md`：BTC门控/轮动/对冲增强，冲击年化15%+夏普1+；
- `reports/core_top5_report.md`：TOP5 核心池激进轮动、关键位杠杆及做空降级验证；
- `reports/core_top5_validation.png`：TOP5 方案净值与锁定参数后的状态分段对比；
- `reports/crypto_alpha_results.json`：增强策略参数与目标达成明细；
- `reports/optimized_backtest_results.json`：优化策略全部参数及训练/样本外指标；
- `reports/backtest_results.json`：首轮策略参数及训练/样本外指标；
- `reports/data_manifest.json`：数据来源、日期范围、行数和 SHA-256。

仓库包含本次报告使用的原始 CSV 快照；上述固定起止日期命令可重新下载并核对清单中的 SHA-256。

## 优化策略设计（第二轮）

针对首轮失败原因（高换手、弱市满仓、财务因子失效），新增 4 个原型：

| 策略 | 核心机制 | 目标 |
|---|---|---|
| `btc_dual_momentum` | BTC 均线门控 + 绝对/相对双动量 | 中低频捕捉趋势，避免接飞刀 |
| `breadth_regime_rotation` | 市场广度三档仓位 | 用宽度替代财务择时 |
| `core_satellite_vol_scaled` | BTC/ETH 核心 + 山寨卫星 + 波动缩放 | 低换手控制回撤 |
| `majors_alts_regime` | 主流/山寨相对强弱 + BTC 趋势过滤 | 强化首轮唯一弱正信号 |

判定规则：默认与优化参数样本外均需 CAGR、Sharpe 为正，才记为稳健候选；Bootstrap CAGR 95% CI 下界大于 0 才记为统计通过。

## TOP5 激进牛熊轮动

`core-top5` 固定使用 BTC、ETH、SOL、XRP、DOGE 五个大市值/高流动性核心标的，
不根据全样本事后收益换池。BTC 站上慢趋势且长动量为正时先持有 BTC；快慢均线
和核心池广度进一步确认后，再集中轮动至长短动量最强的 1–2 个标的。只有 BTC
在调仓信号日收盘突破此前 55 日最高收盘价时，名义总敞口才进入至少 1.2 倍、最高
1.3–1.5 倍的候选范围，并持有至下一次调仓。

熊市模块只做空跌破趋势且动量显著为负的最弱核心标的。每次启用前会用已发生
行情按真实调仓周期和 0.3 倍敞口回放最近 90 日影子空头，扣除换手和借券成本；
净收益、连续持仓 episode 数或胜率不达标则当期空仓。研究阶段还会比较两个训练
折中的做空与现金表现，贡献不稳定时把探索方案
方案全局降级为熊市现金。训练截止日固定为 2024-04-27，后续新增行情不会回流
训练集；每个输入 CSV 的 SHA-256 会随结果保存。当前截止 2026-07-15 的区间
已在开发中被查看，只能称开发后验证集；其后的新增行情才可作为 forward OOS。

## 实盘 / 模拟盘 Web 控制台

`live-console` 提供可扩展的多策略部署台：

| 模式 | 作用 |
|---|---|
| `paper` | 用实盘（或本地缓存）行情在本地模拟成交与盈亏 |
| `live` | 生成实盘信号与意向单；默认干跑，`allow_live_orders=true` 才允许发单适配 |
| `backtest` | 只读历史买卖点清单，不进入轮询成交 |

能力：

- 登录密码防护的 Web 界面：持仓、权益、收益/回撤/夏普、目标权重、成交与买卖点分析；
- 企业微信群机器人通知（成交、账户摘要、异常）；
- 策略注册表（`crypto_lab/live/registry.py`）便于后续继续挂接回测/模拟/实盘部署；
- 状态落在 `runtime/live/console.db`，重启可恢复。

安全默认：`allow_live_orders=false`；即使打开，在未完成交易所签名下单联调前仍会拒绝真实发单。

## 回测约束

- T-1 收盘后生成目标权重，获得 T 日 close-to-close 收益，禁止未来数据；
- 仅现货多头，未分配资金视为现金，不加杠杆；
- 默认单边手续费 0.10%，滑点 0.05%；
- 前 60% 样本有限网格选参，后 40% 锁定参数后评价；
- BTC 现货买入持有为统一基准，按 365 天年化。
- `core-top5` 使用多空引擎，名义总敞口硬上限 1.5 倍、空头上限 0.3 倍；
  回测未模拟强平、保证金阶梯和逐币种实时资金费率。

## 重要限制

固定使用当前仍在 OKX 交易的长历史币对，存在测试期末幸存者偏差。成交量并非历史流通市值，也不能替代协议收入、TVL 或链上审计。本项目评价的是受原策略启发而重新设计的 Crypto 原型，不把原型收益归因给任一原策略，也不是对原策略宣传收益的复现或实盘收益保证。
