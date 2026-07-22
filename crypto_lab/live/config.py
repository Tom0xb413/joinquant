"""实盘控制台配置加载与安全默认值。"""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class WeComConfig:
    """企业微信群机器人通知配置。

    webhook 留空时通知器静默跳过，便于本地开发；启用后每次成交、引擎异常
    与日终摘要都会推送 Markdown。密钥应只放在本机配置文件或环境变量中。

    @author Cursor
    @since 0.3.0
    """

    enabled: bool = False
    webhook_url: str = ""
    mention_all: bool = False


@dataclass
class AuthConfig:
    """Web 登录防护配置。

    密码以明文写入示例配置，仅用于本地快速启动；生产环境应改为环境变量
    并定期轮换。会话密钥为空时会在启动时自动生成临时随机值。

    @author Cursor
    @since 0.3.0
    """

    username: str = "admin"
    password: str = "change-me-now"
    session_secret: str = ""


@dataclass
class DeploymentConfig:
    """单个策略部署定义，便于后续扩展多策略回测/模拟/实盘。

    mode 取值：
    - ``paper``：用实盘行情在本地模拟成交与盈亏；
    - ``live``：生成实盘信号；默认只记录意向单，显式开启才允许真实下单；
    - ``backtest``：只读展示历史买卖点清单，不进入运行循环。

    @author Cursor
    @since 0.3.0
    """

    id: str
    strategy: str
    mode: str = "paper"
    enabled: bool = True
    initial_equity: float = 100_000.0
    fee_rate: float = 0.001
    slippage_rate: float = 0.0005
    poll_seconds: int = 60
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class LiveConsoleConfig:
    """一键启动控制台的总配置。

    把行情目录、运行时状态库、Web 端口、登录与企业微信集中到同一对象，
    使 CLI、服务与页面读取同一真相来源，避免多处硬编码分叉。

    @author Cursor
    @since 0.3.0
    """

    data_dir: str = "data/okx"
    runtime_dir: str = "runtime/live"
    host: str = "127.0.0.1"
    port: int = 8787
    refresh_market: bool = False
    allow_live_orders: bool = False
    auth: AuthConfig = field(default_factory=AuthConfig)
    wecom: WeComConfig = field(default_factory=WeComConfig)
    deployments: list[DeploymentConfig] = field(default_factory=list)

    def runtime_path(self) -> Path:
        """返回运行时目录绝对路径并确保存在。"""

        path = Path(self.runtime_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def database_path(self) -> Path:
        """返回 SQLite 状态库路径。"""

        return self.runtime_path() / "console.db"


def default_live_config() -> LiveConsoleConfig:
    """生成带 TOP5 模拟部署的默认配置，便于首次一键启动。

    默认只启用 paper 部署，实盘下单开关关闭，强制用户显式开启才能发单。
    """

    return LiveConsoleConfig(
        auth=AuthConfig(session_secret=secrets.token_hex(16)),
        deployments=[
            DeploymentConfig(
                id="core-top5-paper",
                strategy="core_top5_regime_rotation",
                mode="paper",
                enabled=True,
                parameters={
                    "top_k": 1,
                    "rebalance_days": 14,
                    "vol_target": 0.45,
                    "breakout_min_gross": 1.2,
                    "leveraged_max_gross": 1.3,
                    "short_gross": 0.0,
                },
            ),
            DeploymentConfig(
                id="core-top5-live-dryrun",
                strategy="core_top5_regime_rotation",
                mode="live",
                enabled=False,
                parameters={
                    "top_k": 1,
                    "rebalance_days": 14,
                    "vol_target": 0.45,
                    "breakout_min_gross": 1.2,
                    "leveraged_max_gross": 1.3,
                    "short_gross": 0.0,
                },
            ),
            DeploymentConfig(
                id="core-top5-backtest-book",
                strategy="core_top5_regime_rotation",
                mode="backtest",
                enabled=True,
                parameters={
                    "top_k": 1,
                    "rebalance_days": 14,
                    "vol_target": 0.45,
                    "breakout_min_gross": 1.2,
                    "leveraged_max_gross": 1.3,
                    "short_gross": 0.0,
                },
            ),
        ],
    )


def load_live_config(path: Path | None = None) -> LiveConsoleConfig:
    """从 JSON 加载配置；文件缺失时写出默认模板并返回。

    自动生成可降低首次启动摩擦；已有文件则严格按用户设置覆盖默认值，
    避免静默改写密码或 webhook。
    """

    if path is None:
        path = Path("configs/live_console.json")
    if not path.exists():
        config = default_live_config()
        save_live_config(path, config)
        return config
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _from_dict(payload)


def save_live_config(path: Path, config: LiveConsoleConfig) -> None:
    """将配置写成稳定缩进的 JSON，便于人工审阅与版本管理。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def apply_env_overrides(config: LiveConsoleConfig) -> LiveConsoleConfig:
    """用环境变量覆盖部署敏感项与容器网络绑定。

    Docker / K8s 常用密钥注入方式是环境变量；本函数只改动显式提供的键，
    避免空字符串误清空配置。布尔值接受 ``1/true/yes/on``（大小写不敏感）。

    支持的变量：
    - ``LIVE_CONSOLE_HOST`` / ``LIVE_CONSOLE_PORT``
    - ``LIVE_AUTH_USERNAME`` / ``LIVE_AUTH_PASSWORD`` / ``LIVE_SESSION_SECRET``
    - ``WECOM_ENABLED`` / ``WECOM_WEBHOOK_URL`` / ``WECOM_MENTION_ALL``
    - ``ALLOW_LIVE_ORDERS`` / ``REFRESH_MARKET``
    - ``LIVE_DATA_DIR`` / ``LIVE_RUNTIME_DIR``

    @author Cursor
    @since 0.3.1
    """

    host = os.environ.get("LIVE_CONSOLE_HOST")
    if host:
        config.host = host
    port_raw = os.environ.get("LIVE_CONSOLE_PORT")
    if port_raw:
        config.port = int(port_raw)
    username = os.environ.get("LIVE_AUTH_USERNAME")
    if username:
        config.auth.username = username
    password = os.environ.get("LIVE_AUTH_PASSWORD")
    if password:
        config.auth.password = password
    session_secret = os.environ.get("LIVE_SESSION_SECRET")
    if session_secret:
        config.auth.session_secret = session_secret
    if "WECOM_ENABLED" in os.environ:
        config.wecom.enabled = _env_bool("WECOM_ENABLED")
    webhook = os.environ.get("WECOM_WEBHOOK_URL")
    if webhook:
        config.wecom.webhook_url = webhook
        config.wecom.enabled = True
    if "WECOM_MENTION_ALL" in os.environ:
        config.wecom.mention_all = _env_bool("WECOM_MENTION_ALL")
    if "ALLOW_LIVE_ORDERS" in os.environ:
        config.allow_live_orders = _env_bool("ALLOW_LIVE_ORDERS")
    if "REFRESH_MARKET" in os.environ:
        config.refresh_market = _env_bool("REFRESH_MARKET")
    data_dir = os.environ.get("LIVE_DATA_DIR")
    if data_dir:
        config.data_dir = data_dir
    runtime_dir = os.environ.get("LIVE_RUNTIME_DIR")
    if runtime_dir:
        config.runtime_dir = runtime_dir
    return config


def _env_bool(name: str, default: bool = False) -> bool:
    """解析常见真值字符串；缺失时返回 default。"""

    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _from_dict(payload: dict[str, Any]) -> LiveConsoleConfig:
    """把原始字典安全转换为强类型配置对象。"""

    auth_raw = payload.get("auth", {})
    wecom_raw = payload.get("wecom", {})
    deployments = [
        DeploymentConfig(**item) for item in payload.get("deployments", [])
    ]
    auth = AuthConfig(**auth_raw)
    if not auth.session_secret:
        auth.session_secret = secrets.token_hex(16)
    return LiveConsoleConfig(
        data_dir=payload.get("data_dir", "data/okx"),
        runtime_dir=payload.get("runtime_dir", "runtime/live"),
        host=payload.get("host", "127.0.0.1"),
        port=int(payload.get("port", 8787)),
        refresh_market=bool(payload.get("refresh_market", False)),
        allow_live_orders=bool(payload.get("allow_live_orders", False)),
        auth=auth,
        wecom=WeComConfig(**wecom_raw),
        deployments=deployments,
    )
