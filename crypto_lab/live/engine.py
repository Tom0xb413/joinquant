"""交易引擎：信号生成、调仓执行、快照持久化与通知。"""

from __future__ import annotations

import threading
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from .broker import LiveBroker, PaperBroker, PortfolioState, performance_metrics, utc_now_iso
from .config import DeploymentConfig, LiveConsoleConfig
from .feed import latest_prices, latest_signal_date, load_market_bundle
from .models import AccountSnapshot
from .notifier import WeComNotifier
from .registry import build_strategy, strategy_catalog
from .store import RuntimeStore
from .trade_points import (
    build_backtest_trade_points,
    fills_to_trade_points,
)


class TradingEngine:
    """多部署交易引擎，可在后台线程周期运行。

    每个启用的 paper/live 部署独立持有组合状态；backtest 部署只生成买卖点
    清单供 Web 查阅。引擎对外暴露只读快照接口，供 Flask 页面轮询。

    @author Cursor
    @since 0.3.0
    """

    def __init__(self, config: LiveConsoleConfig):
        self.config = config
        self.store = RuntimeStore(config.database_path())
        self.notifier = WeComNotifier(config.wecom)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._portfolios: dict[str, PortfolioState] = {}
        self._last_signal_index: dict[str, int] = {}
        self._status = "initialized"
        self._bootstrap()

    def _bootstrap(self) -> None:
        """初始化组合、回填 backtest 买卖点，并写入首屏快照。"""

        for deployment in self.config.deployments:
            if not deployment.enabled:
                continue
            if deployment.mode == "backtest":
                self._refresh_backtest_book(deployment)
                continue
            spec = strategy_catalog()[deployment.strategy]
            portfolio = PortfolioState.create(deployment.initial_equity, spec.symbols)
            self._portfolios[deployment.id] = portfolio
            self._last_signal_index[deployment.id] = -1
            self._write_snapshot(
                deployment,
                portfolio,
                prices={symbol: 0.0 for symbol in spec.symbols},
                target_weights={},
                status="ready",
                message="等待首次轮询",
                signal_date="",
            )

    def start_background(self) -> None:
        """启动后台轮询线程。"""

        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="trading-engine", daemon=True)
        self._thread.start()
        self._status = "running"

    def stop(self) -> None:
        """请求停止后台线程。"""

        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._status = "stopped"

    def run_once(self) -> None:
        """同步执行一轮全部部署（测试与手动刷新）。"""

        with self._lock:
            for deployment in self.config.deployments:
                if not deployment.enabled:
                    continue
                try:
                    if deployment.mode == "backtest":
                        self._refresh_backtest_book(deployment)
                    else:
                        self._run_deployment(deployment)
                except Exception as exc:  # 单部署失败不影响其他
                    self.notifier.notify_error(deployment.id, str(exc))
                    self._write_error_snapshot(deployment, str(exc))

    def _loop(self) -> None:
        """按最短 poll 间隔循环执行。"""

        while not self._stop.is_set():
            self.run_once()
            sleep_for = min(
                (
                    max(5, item.poll_seconds)
                    for item in self.config.deployments
                    if item.enabled and item.mode in {"paper", "live"}
                ),
                default=60,
            )
            self._stop.wait(sleep_for)

    def _run_deployment(self, deployment: DeploymentConfig) -> None:
        """对单个 paper/live 部署：拉行情、算信号、调仓、落库与通知。"""

        spec = strategy_catalog()[deployment.strategy]
        data = load_market_bundle(
            Path(self.config.data_dir),
            spec.symbols,
            refresh=self.config.refresh_market,
        )
        strategy = build_strategy(deployment.strategy, deployment.parameters)
        signal_index = len(data.dates) - 1
        prices = latest_prices(data)
        portfolio = self._portfolios[deployment.id]
        current_weights = portfolio.current_weights(spec.symbols, prices)
        previous = current_weights.copy()
        # 首次同步时对齐到最近调仓相位，避免卡在非调仓日的“保持空仓”
        decision_index = signal_index
        if self._last_signal_index.get(deployment.id, -1) < 0:
            rebalance_days = int(getattr(strategy, "rebalance_days", 1) or 1)
            decision_index = signal_index - (signal_index % rebalance_days)
        target = np.asarray(
            strategy.target_weights(data, decision_index, previous),
            dtype=float,
        )
        equity, _, _ = portfolio.valuation(prices)
        broker = self._broker_for(deployment)
        fills = broker.rebalance(
            deployment_id=deployment.id,
            strategy=deployment.strategy,
            mode=deployment.mode,
            symbols=spec.symbols,
            current_weights=current_weights,
            target_weights=target,
            quantities=portfolio.quantities,
            prices=prices,
            equity=max(equity, deployment.initial_equity * 0.01),
            fee_rate=deployment.fee_rate,
            slippage_rate=deployment.slippage_rate,
        )
        if fills:
            if deployment.mode == "paper":
                portfolio.apply_fills(fills)
            for fill in fills:
                self.store.save_fill(fill)
            for point in fills_to_trade_points(fills):
                self.store.append_trade_point(point)
            self.notifier.notify_fills(fills)

        equity, _, _ = portfolio.valuation(prices)
        # 仅在信号日变化时记录权益点，避免同一天高频轮询稀释夏普
        if self._last_signal_index.get(deployment.id) != signal_index:
            portfolio.record_equity(equity)
            self._last_signal_index[deployment.id] = signal_index
        target_map = {
            symbol: float(weight)
            for symbol, weight in zip(spec.symbols, target)
            if abs(weight) > 1e-8
        }
        snapshot = self._write_snapshot(
            deployment,
            portfolio,
            prices=prices,
            target_weights=target_map,
            status="synced",
            message=f"fills={len(fills)}",
            signal_date=latest_signal_date(data).isoformat(),
        )
        if fills:
            self.notifier.notify_snapshot(snapshot)

    def _broker_for(self, deployment: DeploymentConfig) -> PaperBroker | LiveBroker:
        """按部署模式选择经纪商。"""

        if deployment.mode == "live":
            return LiveBroker(allow_orders=self.config.allow_live_orders)
        return PaperBroker()

    def _refresh_backtest_book(self, deployment: DeploymentConfig) -> None:
        """重建 backtest 买卖点清单，并写只读快照。"""

        points = build_backtest_trade_points(
            deployment_id=deployment.id,
            strategy_key=deployment.strategy,
            parameters=deployment.parameters,
            data_dir=Path(self.config.data_dir),
            fee_rate=deployment.fee_rate,
            slippage_rate=deployment.slippage_rate,
        )
        self.store.replace_trade_points(deployment.id, points)
        snapshot = AccountSnapshot(
            deployment_id=deployment.id,
            strategy=deployment.strategy,
            mode="backtest",
            updated_at=utc_now_iso(),
            cash=0.0,
            equity=0.0,
            initial_equity=deployment.initial_equity,
            total_return=0.0,
            day_pnl=0.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            max_drawdown=0.0,
            sharpe=0.0,
            positions=[],
            target_weights={},
            last_signal_date="",
            status="book_ready",
            message=f"trade_points={len(points)}",
        )
        self.store.save_snapshot(snapshot)

    def _write_snapshot(
        self,
        deployment: DeploymentConfig,
        portfolio: PortfolioState,
        *,
        prices: dict[str, float],
        target_weights: dict[str, float],
        status: str,
        message: str,
        signal_date: str,
    ) -> AccountSnapshot:
        """计算绩效并持久化账户快照。"""

        equity, unrealized, positions = portfolio.valuation(prices)
        metrics = performance_metrics(portfolio.equity_curve, portfolio.daily_returns)
        day_pnl = 0.0
        if len(portfolio.equity_curve) >= 2:
            day_pnl = portfolio.equity_curve[-1] - portfolio.equity_curve[-2]
        snapshot = AccountSnapshot(
            deployment_id=deployment.id,
            strategy=deployment.strategy,
            mode=deployment.mode,
            updated_at=utc_now_iso(),
            cash=portfolio.cash,
            equity=equity,
            initial_equity=portfolio.initial_equity,
            total_return=metrics["total_return"],
            day_pnl=day_pnl,
            unrealized_pnl=unrealized,
            realized_pnl=portfolio.realized_pnl,
            max_drawdown=metrics["max_drawdown"],
            sharpe=metrics["sharpe"],
            positions=positions,
            target_weights=target_weights,
            last_signal_date=signal_date,
            status=status,
            message=message,
        )
        self.store.save_snapshot(snapshot)
        return snapshot

    def _write_error_snapshot(self, deployment: DeploymentConfig, message: str) -> None:
        """在部署失败时保留可观察的错误快照。"""

        existing = self.store.get_snapshot(deployment.id) or {}
        snapshot = AccountSnapshot(
            deployment_id=deployment.id,
            strategy=deployment.strategy,
            mode=deployment.mode,
            updated_at=utc_now_iso(),
            cash=float(existing.get("cash", deployment.initial_equity)),
            equity=float(existing.get("equity", deployment.initial_equity)),
            initial_equity=deployment.initial_equity,
            total_return=float(existing.get("total_return", 0.0)),
            day_pnl=0.0,
            unrealized_pnl=float(existing.get("unrealized_pnl", 0.0)),
            realized_pnl=float(existing.get("realized_pnl", 0.0)),
            max_drawdown=float(existing.get("max_drawdown", 0.0)),
            sharpe=float(existing.get("sharpe", 0.0)),
            positions=[],
            target_weights={},
            last_signal_date=str(existing.get("last_signal_date", "")),
            status="error",
            message=message[:500],
        )
        self.store.save_snapshot(snapshot)

    def overview(self) -> dict[str, Any]:
        """供 Web 仪表盘使用的总览数据。"""

        return {
            "status": self._status,
            "allow_live_orders": self.config.allow_live_orders,
            "strategies": [
                {
                    "key": spec.key,
                    "title": spec.title,
                    "description": spec.description,
                }
                for spec in strategy_catalog().values()
            ],
            "deployments": [
                {
                    "id": item.id,
                    "strategy": item.strategy,
                    "mode": item.mode,
                    "enabled": item.enabled,
                    "snapshot": self.store.get_snapshot(item.id),
                }
                for item in self.config.deployments
            ],
            "trade_summary": self.trade_summary(),
        }

    def trade_summary(self) -> dict[str, Any]:
        """汇总买卖点统计。"""

        from .trade_points import summarize_trade_points

        return summarize_trade_points(self.store.list_trade_points(limit=5000))
