"""一键启动：引擎后台线程 + Web 控制台。"""

from __future__ import annotations

from pathlib import Path

from ..live.config import (
    apply_env_overrides,
    load_live_config,
    save_live_config,
)
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
    backtest 买卖点与 paper 首屏快照），再阻塞服务 HTTP。CLI 的
    ``host`` / ``port`` 优先于配置文件；环境变量（Docker 密钥注入）
    再覆盖认证与通知等敏感项。配置目录只读时跳过落盘，仅使用内存配置。
    优先使用 waitress 作为生产 WSGI；缺失时回退 Flask 内置服务器。

    @author Cursor
    @since 0.3.0
    """

    path = config_path or Path("configs/live_console.json")
    config = load_live_config(path)
    if host:
        config.host = host
    if port:
        config.port = port
    apply_env_overrides(config)
    try:
        save_live_config(path, config)
    except OSError as exc:
        print(f"[warn] config not writable ({exc}); using in-memory overrides")

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
        try:
            from waitress import serve
        except ImportError:
            app.run(
                host=config.host,
                port=config.port,
                debug=False,
                use_reloader=False,
            )
        else:
            serve(app, host=config.host, port=config.port, threads=8)
    finally:
        engine.stop()
