"""实盘/模拟引擎、买卖点与 Web 登录测试。"""

from __future__ import annotations

import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from crypto_lab.data import Candle, MarketData, save_candles
from crypto_lab.live.broker import PaperBroker, PortfolioState, performance_metrics
from crypto_lab.live.config import default_live_config, load_live_config, save_live_config
from crypto_lab.live.models import Fill
from crypto_lab.live.notifier import WeComNotifier
from crypto_lab.live.registry import build_strategy, strategy_catalog
from crypto_lab.live.store import RuntimeStore
from crypto_lab.live.trade_points import fills_to_trade_points, summarize_trade_points
from crypto_lab.web import create_app
from crypto_lab.live.engine import TradingEngine


def _write_symbol_csv(path: Path, start: date, days: int, start_price: float, slope: float) -> None:
    """写入合成日线 CSV，供引擎离线测试。"""

    candles = []
    for index in range(days):
        day = start + timedelta(days=index)
        price = start_price * np.exp(slope * index)
        ts = int((day - date(1970, 1, 1)).days * 86_400_000)
        candles.append(
            Candle(
                timestamp_ms=ts,
                open=price,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume_base=1000.0,
                volume_quote=1000.0 * price,
            )
        )
    save_candles(path, candles)


class LiveModuleTests(unittest.TestCase):
    """验证模拟成交、持久化、策略注册与通知静默行为。

    @author Cursor
    @since 0.3.0
    """

    def test_paper_broker_and_portfolio_pnl(self):
        """模拟买入再卖出后，已实现盈亏应为正且现金回升。"""

        symbols = ("BTC-USDT",)
        portfolio = PortfolioState.create(10_000.0, symbols)
        broker = PaperBroker(min_delta=0.001)
        prices = {"BTC-USDT": 100.0}
        fills = broker.rebalance(
            deployment_id="t",
            strategy="s",
            mode="paper",
            symbols=symbols,
            current_weights=np.zeros(1),
            target_weights=np.array([1.0]),
            quantities=portfolio.quantities,
            prices=prices,
            equity=10_000.0,
            fee_rate=0.001,
            slippage_rate=0.0,
        )
        self.assertEqual(len(fills), 1)
        portfolio.apply_fills(fills)
        self.assertGreater(portfolio.quantities["BTC-USDT"], 0)
        equity_after, _, _ = portfolio.valuation({"BTC-USDT": 110.0})
        sell_fills = broker.rebalance(
            deployment_id="t",
            strategy="s",
            mode="paper",
            symbols=symbols,
            current_weights=portfolio.current_weights(symbols, {"BTC-USDT": 110.0}),
            target_weights=np.zeros(1),
            quantities=portfolio.quantities,
            prices={"BTC-USDT": 110.0},
            equity=equity_after,
            fee_rate=0.001,
            slippage_rate=0.0,
        )
        portfolio.apply_fills(sell_fills)
        self.assertAlmostEqual(portfolio.quantities["BTC-USDT"], 0.0, places=8)
        self.assertGreater(portfolio.realized_pnl, 0.0)

    def test_store_roundtrip(self):
        """成交与快照写入 SQLite 后可完整读回。"""

        with tempfile.TemporaryDirectory() as tmp:
            store = RuntimeStore(Path(tmp) / "t.db")
            fill = Fill(
                id="1",
                deployment_id="d1",
                strategy="core_top5_regime_rotation",
                mode="paper",
                timestamp="2026-07-21T00:00:00+00:00",
                symbol="BTC-USDT",
                side="buy",
                quantity=1.0,
                price=100.0,
                notional=100.0,
                fee=0.1,
                weight_from=0.0,
                weight_to=1.0,
                equity_before=1000.0,
            )
            store.save_fill(fill)
            rows = store.list_fills("d1")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["symbol"], "BTC-USDT")
            points = fills_to_trade_points([fill])
            store.replace_trade_points("d1", points)
            self.assertEqual(len(store.list_trade_points("d1")), 1)
            store.close()

    def test_strategy_catalog_builds_core_top5(self):
        """注册表应能构造 TOP5 策略实例。"""

        self.assertIn("core_top5_regime_rotation", strategy_catalog())
        strategy = build_strategy("core_top5_regime_rotation", {"top_k": 1, "short_gross": 0.0})
        self.assertEqual(strategy.name, "core_top5_regime_rotation")

    def test_wecom_disabled_is_silent(self):
        """未启用 webhook 时通知必须静默返回 False。"""

        from crypto_lab.live.config import WeComConfig

        notifier = WeComNotifier(WeComConfig(enabled=False, webhook_url=""))
        self.assertFalse(notifier.send_markdown("hello"))

    def test_performance_metrics_drawdown(self):
        """权益回撤指标应反映峰值到谷底的跌幅。"""

        metrics = performance_metrics([1.0, 1.2, 0.9], [0.2, -0.25])
        self.assertAlmostEqual(metrics["max_drawdown"], 0.25, places=6)

    def test_summarize_trade_points(self):
        """买卖点汇总应统计方向与标的次数。"""

        summary = summarize_trade_points(
            [
                {
                    "symbol": "BTC-USDT",
                    "side": "buy",
                    "mode": "backtest",
                },
                {
                    "symbol": "ETH-USDT",
                    "side": "sell",
                    "mode": "paper",
                },
            ]
        )
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["buy_like"], 1)
        self.assertEqual(summary["sell_like"], 1)


class LiveConsoleWebTests(unittest.TestCase):
    """验证登录门禁与受保护 API。

    @author Cursor
    @since 0.3.0
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        data_dir = root / "data"
        data_dir.mkdir()
        start = date(2022, 1, 1)
        for symbol, slope in [
            ("BTC-USDT", 0.001),
            ("ETH-USDT", 0.0012),
            ("SOL-USDT", 0.0015),
            ("XRP-USDT", 0.0008),
            ("DOGE-USDT", 0.0011),
        ]:
            _write_symbol_csv(data_dir / f"{symbol}.csv", start, 420, 100.0, slope)
        config = default_live_config()
        config.data_dir = str(data_dir)
        config.runtime_dir = str(root / "runtime")
        config.auth.username = "admin"
        config.auth.password = "secret"
        config.auth.session_secret = "test-secret"
        config.refresh_market = False
        # 测试中只启用 paper，避免 backtest 过慢；backtest 仍可单独测
        for deployment in config.deployments:
            if deployment.mode == "backtest":
                deployment.enabled = False
            if deployment.mode == "live":
                deployment.enabled = False
            deployment.poll_seconds = 3600
        self.config = config
        self.engine = TradingEngine(config)
        self.engine.run_once()
        self.app = create_app(self.engine, config)
        self.client = self.app.test_client()

    def tearDown(self):
        self.engine.stop()
        self.tmp.cleanup()

    def test_login_required_for_dashboard(self):
        """未登录访问总览应重定向到登录页。"""

        response = self.client.get("/")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.headers["Location"])

    def test_login_and_api_overview(self):
        """正确密码登录后可读取总览 API。"""

        bad = self.client.post(
            "/login",
            data={"username": "admin", "password": "wrong"},
            follow_redirects=False,
        )
        self.assertEqual(bad.status_code, 401)
        ok = self.client.post(
            "/login",
            data={"username": "admin", "password": "secret"},
            follow_redirects=False,
        )
        self.assertEqual(ok.status_code, 302)
        overview = self.client.get("/api/overview")
        self.assertEqual(overview.status_code, 200)
        payload = overview.get_json()
        self.assertIn("deployments", payload)
        self.assertTrue(any(item["id"] == "core-top5-paper" for item in payload["deployments"]))

    def test_config_roundtrip(self):
        """配置文件应可保存并重新加载。"""

        path = Path(self.tmp.name) / "cfg.json"
        save_live_config(path, self.config)
        loaded = load_live_config(path)
        self.assertEqual(loaded.auth.username, "admin")
        self.assertGreaterEqual(len(loaded.deployments), 1)


if __name__ == "__main__":
    unittest.main()
