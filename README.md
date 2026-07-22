# 聚宽策略跨市场 Crypto 研究

本仓库保留 12 份聚宽 A 股原始策略，并提供一个可复现的 Python 研究框架（`crypto_lab`），将其中可跨市场验证的结构迁移到 Crypto。框架使用 OKX 公开现货 K 线、严格的 T-1 信号、真实交易成本、时间序列切分与样本外评价。

## 仓库结构

```text
joinquant/
├── 01–12-*.py              # 聚宽 A 股参考策略（仅 joinquant.com 可运行，勿本地执行）
├── README.txt              # 原始 Top12 策略包说明
├── AGENTS.md               # Cursor Cloud 开发环境约定
├── pyproject.toml          # 包元数据与依赖（numpy / matplotlib / flask）
├── Dockerfile / docker-compose.yml / .env.example
├── docker/entrypoint.sh    # 容器入口：种子配置并启动 live-console
├── configs/
│   ├── live_console.example.json   # 本地示例配置
│   └── live_console.docker.json    # 容器默认配置（0.0.0.0）
├── crypto_lab/             # 可运行研究与部署代码
│   ├── cli.py              # 统一 CLI 入口
│   ├── data.py / indicators.py / backtest.py / long_short.py
│   ├── strategies.py / research.py                 # 首轮跨市场迁移
│   ├── optimized_strategies.py / optimize_research.py
│   ├── crypto_alpha.py / crypto_alpha_research.py
│   ├── core_top5.py / core_top5_research.py
│   ├── cycle_report.py
│   ├── ema_*.py            # BTC/ETH EMA 多周期研究
│   ├── cta_*.py            # TOP15 多周期动量 CTA
│   ├── live/               # paper / live / backtest 运行时
│   └── web/                # 登录保护的 Web 操作台
├── data/
│   ├── okx/                # 日线快照（首轮/优化/alpha/TOP5；已打入镜像）
│   ├── okx_bars/           # EMA 多周期 K 线
│   └── okx_cta/            # CTA 多周期 K 线
├── reports/                # 研究产出（markdown / json / png）
└── tests/                  # unittest 套件
```

`NN-*.py` 风格的顶层 `01–12-*.py` 仅为聚宽平台参考源码；本地可执行入口一律走：

```bash
python3 -m crypto_lab.cli <subcommand>
```

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

## 安装与快速运行

环境要求：Python 3.10+、NumPy 2.4.4+、matplotlib 3.8+、Flask 3.x（Web 控制台）。

```bash
pip install -e .
python3 -m crypto_lab.cli download --start 2021-01-01 --end 2026-07-15
python3 -m crypto_lab.cli research
python3 -m crypto_lab.cli optimize
python3 -m crypto_lab.cli crypto-alpha
python3 -m crypto_lab.cli core-top5
python3 -m crypto_lab.cli cycle-report
python3 -m crypto_lab.cli ema-research
python3 -m crypto_lab.cli cta-research
python3 -m crypto_lab.cli live-console
python3 -m unittest discover -s tests -v
```

离线命令（`research` / `optimize` / `crypto-alpha` / `core-top5` / `cycle-report`，以及不加 `--refresh` 的 `ema-research` / `cta-research`）直接读取 `data/` 下已提交的 CSV 快照，无需外网。只有 `download` 与带 `--refresh` 的行情刷新会访问 OKX 公开 API。

## CLI 子命令一览

| 命令 | 作用 | 主要产出 |
|---|---|---|
| `download` | 下载 OKX UTC 日线 | `data/okx/*.csv`、`data_manifest.json` |
| `research` | 首轮跨市场迁移原型 | `reports/cross_market_report.md` |
| `optimize` | 低换手优化策略 | `reports/optimized_strategies_report.md` |
| `crypto-alpha` | BTC 门控/轮动/对冲增强 | `reports/crypto_alpha_report.md` |
| `core-top5` | TOP5 激进牛熊轮动 | `reports/core_top5_report.md` |
| `cycle-report` | 2021–2026 全周期与 beta 分段 | `reports/cycle_full_report.md` |
| `ema-research` | BTC/ETH EMA50/100 多周期 | `reports/ema_report.md` |
| `cta-research` | TOP15 多周期动量 CTA | `reports/cta_report.md`、对抗审查报告 |
| `live-console` | 模拟/实盘干跑 + Web 操作台 | `runtime/live/console.db` |
| `trade-book` | 导出历史买卖点 JSON | `reports/trade_book.json` |

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
折中的做空与现金表现，贡献不稳定时把探索方案全局降级为熊市现金。训练截止日固定为
2024-04-27，后续新增行情不会回流训练集；每个输入 CSV 的 SHA-256 会随结果保存。
当前截止 2026-07-15 的区间已在开发中被查看，只能称开发后验证集；其后的新增行情才可作为
forward OOS。

## 实盘 / 模拟盘 Web 控制台

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

| 模式 | 作用 |
|---|---|
| `paper` | 用实盘（或本地缓存）行情在本地模拟成交与盈亏 |
| `live` | 生成实盘信号与意向单；默认干跑，`allow_live_orders=true` 才允许发单适配 |
| `backtest` | 只读历史买卖点清单，不进入轮询成交 |

能力：登录保护的 Web 界面（持仓、权益、收益/回撤/夏普、目标权重、成交与买卖点）、
企业微信群机器人通知、策略注册表（`crypto_lab/live/registry.py`）、状态落盘
`runtime/live/console.db`。安全默认：`allow_live_orders=false`；即使打开，在未完成
交易所签名下单联调前仍会拒绝真实发单。

## Docker 一键部署

镜像打包：`crypto_lab` 策略代码、Web 控制台（waitress WSGI）、注册表内策略（TOP5 轮动 /
BTC 门控动量）、以及 `data/okx` 日线快照。容器默认监听 `0.0.0.0:8787`，状态写入卷
`live-runtime`。

```bash
cp .env.example .env
# 务必修改 LIVE_AUTH_PASSWORD；可选填写 WECOM_WEBHOOK_URL
docker compose up -d --build
# 浏览器打开 http://127.0.0.1:8787/  （用户 admin / 密码见 .env）
docker compose logs -f live-console
docker compose down
```

仅构建镜像：

```bash
docker build -t joinquant-live-console:latest .
docker run --rm -p 8787:8787 \
  -e LIVE_AUTH_PASSWORD='your-strong-password' \
  -v joinquant-live-runtime:/app/runtime \
  joinquant-live-console:latest
```

常用环境变量：

| 变量 | 作用 |
|---|---|
| `LIVE_AUTH_USERNAME` / `LIVE_AUTH_PASSWORD` | Web 登录账号 |
| `LIVE_SESSION_SECRET` | Flask 会话密钥（建议固定，避免重启掉登录态） |
| `WECOM_WEBHOOK_URL` | 企业微信群机器人；非空则自动启用通知 |
| `REFRESH_MARKET` | `true` 时启动轮询会尝试刷新 OKX 公开日线 |
| `ALLOW_LIVE_ORDERS` | 默认 `false`；未完成交易所签名联调前请勿打开 |
| `LIVE_CONSOLE_PUBLISH_PORT` | compose 宿主机映射端口，默认 8787 |

容器内也可执行 `docker compose run --rm live-console trade-book --output /tmp/book.json`。

## 报告产物索引

| 文件 | 说明 |
|---|---|
| `reports/cross_market_report.md` | 首轮迁移原型样本外结论 |
| `reports/optimized_strategies_report.md` | 低换手优化策略设计与验证 |
| `reports/crypto_alpha_report.md` | BTC 门控/轮动/对冲增强 |
| `reports/core_top5_report.md` | TOP5 激进轮动与做空降级验证 |
| `reports/cycle_full_report.md` | 2021–2026 全周期与 beta 分段 |
| `reports/ema_report.md` | EMA 多周期研究 |
| `reports/cta_report.md` / `cta_adversarial_review.md` | CTA 回测与对抗审查 |
| `reports/*_results.json` / `data_manifest.json` | 参数、指标与数据清单 |

仓库已包含报告所用 CSV 快照；固定起止日期的 `download` 可重新拉取并核对 SHA-256。

## 回测约束

- T-1 收盘后生成目标权重，获得 T 日 close-to-close 收益，禁止未来数据；
- 多数研究策略仅现货多头，未分配资金视为现金；`core-top5` / CTA 等另有敞口上限；
- 默认单边手续费 0.10%，滑点 0.05%；
- 前 60% 样本有限网格选参，后 40% 锁定参数后评价（TOP5 用固定训练截止日）；
- BTC 现货买入持有为统一基准，按 365 天年化；
- `core-top5` 使用多空引擎，名义总敞口硬上限 1.5 倍、空头上限 0.3 倍；
  回测未模拟强平、保证金阶梯和逐币种实时资金费率。

## Pull / 分支合并状态

| PR | 原分支 | 状态 | 内容 |
|---|---|---|---|
| #1 | `cursor/crypto-cross-market-backtest-7e58` | 已合并 | 跨市场回测与 BTC 门控增强 |
| #2 | `cursor/crypto-cross-market-backtest-7e58` | 已合并 | 2021–2026 全周期详细报告 |
| #3 | `cursor/core-top5-regime-f4f4` | 已合并 | TOP5 核心池激进牛熊轮动 |
| #4 | `cursor/live-paper-console-f4f4` | 已关闭 | 与 #3 squash 后冲突；由 #5 承接 |
| #5 | `cursor/docs-structure-merge-97c6` | 已合并 | 干净合入 live/web 控制台 + README 结构文档 |

上述特性分支已从远程删除；当前默认分支为 `main`。

## 重要限制

固定使用当前仍在 OKX 交易的长历史币对，存在测试期末幸存者偏差。成交量并非历史流通市值，也不能替代协议收入、TVL 或链上审计。本项目评价的是受原策略启发而重新设计的 Crypto 原型，不把原型收益归因给任一原策略，也不是对原策略宣传收益的复现或实盘收益保证。
