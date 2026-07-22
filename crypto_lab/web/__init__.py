"""Flask Web 控制台：登录防护、多策略仪表盘与交易清单。"""

from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any, Callable

from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from ..live.config import LiveConsoleConfig
from ..live.engine import TradingEngine
from ..live.registry import strategy_catalog
from ..live.trade_points import summarize_trade_points


def create_app(engine: TradingEngine, config: LiveConsoleConfig) -> Flask:
    """创建绑定交易引擎的 Flask 应用。

    会话基于服务端 secret；未登录访问业务页一律重定向到登录。API 同样
    需要登录，避免未授权读取持仓与成交。

    @author Cursor
    @since 0.3.0
    """

    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )
    app.secret_key = config.auth.session_secret or "dev-only-secret"

    def login_required(view: Callable[..., Any]) -> Callable[..., Any]:
        """保护页面与 JSON API 的登录装饰器。"""

        @wraps(view)
        def wrapped(*args: Any, **kwargs: Any):
            if not session.get("authenticated"):
                if request.path.startswith("/api/"):
                    return jsonify({"error": "unauthorized"}), 401
                return redirect(url_for("login", next=request.path))
            return view(*args, **kwargs)

        return wrapped

    @app.get("/login")
    def login():
        """渲染登录页。"""

        if session.get("authenticated"):
            return redirect(url_for("dashboard"))
        return render_template("login.html", brand="Crypto Lab Desk")

    @app.post("/login")
    def login_submit():
        """校验用户名密码并建立会话。"""

        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if (
            username == config.auth.username
            and password == config.auth.password
        ):
            session["authenticated"] = True
            session["username"] = username
            return redirect(request.args.get("next") or url_for("dashboard"))
        return render_template(
            "login.html",
            brand="Crypto Lab Desk",
            error="用户名或密码错误",
        ), 401

    @app.post("/logout")
    def logout():
        """注销会话。"""

        session.clear()
        return redirect(url_for("login"))

    @app.get("/")
    @login_required
    def dashboard():
        """多策略部署总览。"""

        overview = engine.overview()
        return render_template(
            "dashboard.html",
            brand="Crypto Lab Desk",
            username=session.get("username"),
            overview=overview,
            catalog=strategy_catalog(),
        )

    @app.get("/deployments/<deployment_id>")
    @login_required
    def deployment_detail(deployment_id: str):
        """单个部署：持仓、绩效、近期成交。"""

        snapshot = engine.store.get_snapshot(deployment_id)
        if snapshot is None:
            return render_template("not_found.html", brand="Crypto Lab Desk"), 404
        fills = engine.store.list_fills(deployment_id, limit=100)
        points = engine.store.list_trade_points(deployment_id, limit=200)
        return render_template(
            "deployment.html",
            brand="Crypto Lab Desk",
            username=session.get("username"),
            snapshot=snapshot,
            fills=fills,
            points=points,
            summary=summarize_trade_points(points),
        )

    @app.get("/trades")
    @login_required
    def trades():
        """全局买卖点清单与分析。"""

        deployment_id = request.args.get("deployment_id") or None
        points = engine.store.list_trade_points(deployment_id, limit=1000)
        return render_template(
            "trades.html",
            brand="Crypto Lab Desk",
            username=session.get("username"),
            points=points,
            summary=summarize_trade_points(points),
            deployments=config.deployments,
            selected=deployment_id or "",
        )

    @app.get("/api/overview")
    @login_required
    def api_overview():
        """仪表盘轮询接口。"""

        return jsonify(engine.overview())

    @app.get("/api/deployments/<deployment_id>")
    @login_required
    def api_deployment(deployment_id: str):
        """部署详情 JSON。"""

        snapshot = engine.store.get_snapshot(deployment_id)
        if snapshot is None:
            return jsonify({"error": "not found"}), 404
        return jsonify(
            {
                "snapshot": snapshot,
                "fills": engine.store.list_fills(deployment_id, limit=100),
                "trade_points": engine.store.list_trade_points(deployment_id, limit=200),
            }
        )

    @app.post("/api/run-once")
    @login_required
    def api_run_once():
        """手动触发一轮引擎同步。"""

        engine.run_once()
        return jsonify({"ok": True, "overview": engine.overview()})

    return app
