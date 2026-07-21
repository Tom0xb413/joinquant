"""一键启动：引擎后台线程 + Web 控制台。"""

from __future__ import annotations

from pathlib import Path

from ..live.config import LiveConsoleConfig, load_live_config, save_live_config
from ..live.engine import TradingEngine
from . import create_app


def run_live_console(
    config_path: Path | None = None,
    host: str | None = None,
    port: int | None = None,
    run_once_before_serve: bool = True,
) -> None:
    """加载配置、启动交易引擎并打开 Web 控制台。

    首次运行若配置不存在会自动写出默认模板。默认先同步一轮（生成
    backtest 买卖点与 paper 首屏快照），再阻塞服务 HTTP。

    @author Cursor
    @since 0.3.0
    """

    path = config_path or Path("configs/live_console.json")
    config = load_live_config(path)
    if host:
        config.host = host
    if port:
        config.port = port
    save_live_config(path, config)

    engine = TradingEngine(config)
    if run_once_before_serve:
        engine.run_once()
    engine.start_background()
    app = create_app(engine, config)
    print(
        f"Live console ready: http://{config.host}:{config.port}/  "
        f"(user={config.auth.username})"
    )
    try:
        app.run(host=config.host, port=config.port, debug=False, use_reloader=False)
    finally:
        engine.stop()
