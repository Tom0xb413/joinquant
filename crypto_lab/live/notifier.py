"""企业微信群机器人通知。"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .config import WeComConfig
from .models import AccountSnapshot, Fill


class WeComNotifier:
    """通过企业微信群机器人 Webhook 推送 Markdown。

    webhook 为空或 disabled 时所有方法立即返回 False，不影响交易主循环。
    网络失败会被吞掉并返回 False，避免通知故障阻断策略。

    @author Cursor
    @since 0.3.0
    """

    def __init__(self, config: WeComConfig):
        self.config = config

    def send_markdown(self, content: str) -> bool:
        """发送 Markdown 消息；成功返回 True。"""

        if not self.config.enabled or not self.config.webhook_url:
            return False
        payload: dict[str, Any] = {
            "msgtype": "markdown",
            "markdown": {"content": content},
        }
        if self.config.mention_all:
            payload["markdown"]["content"] = content + "\n<@all>"
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.config.webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                body = json.loads(response.read().decode("utf-8"))
            return body.get("errcode", 1) == 0
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return False

    def notify_fills(self, fills: list[Fill]) -> bool:
        """汇总一批成交/意向并推送。"""

        if not fills:
            return False
        lines = [
            f"### 策略成交通知 `{fills[0].deployment_id}`",
            f"> 模式：<font color=\"info\">{fills[0].mode}</font>  策略：{fills[0].strategy}",
            "",
        ]
        for fill in fills[:12]:
            color = "warning" if fill.status != "filled" else "info"
            lines.append(
                f"- <font color=\"{color}\">{fill.side}</font> "
                f"**{fill.symbol}** qty=`{fill.quantity:.6f}` "
                f"price=`{fill.price:.4f}` status=`{fill.status}`"
            )
        if len(fills) > 12:
            lines.append(f"- … 其余 {len(fills) - 12} 笔省略")
        return self.send_markdown("\n".join(lines))

    def notify_snapshot(self, snapshot: AccountSnapshot) -> bool:
        """推送账户权益与持仓摘要。"""

        lines = [
            f"### 账户摘要 `{snapshot.deployment_id}`",
            f"> 权益：**{snapshot.equity:,.2f}**  "
            f"收益：<font color=\"{'info' if snapshot.total_return >= 0 else 'warning'}\">"
            f"{snapshot.total_return:.2%}</font>",
            f"> 回撤：{snapshot.max_drawdown:.2%}  夏普：{snapshot.sharpe:.2f}",
            f"> 状态：{snapshot.status}  信号日：{snapshot.last_signal_date}",
            "",
        ]
        for position in snapshot.positions[:8]:
            lines.append(
                f"- {position.symbol} w=`{position.weight:.2%}` "
                f"qty=`{position.quantity:.6f}` uPnL=`{position.unrealized_pnl:,.2f}`"
            )
        if not snapshot.positions:
            lines.append("- 当前空仓（现金）")
        return self.send_markdown("\n".join(lines))

    def notify_error(self, deployment_id: str, message: str) -> bool:
        """推送引擎异常。"""

        return self.send_markdown(
            f"### 引擎异常 `{deployment_id}`\n> <font color=\"warning\">{message}</font>"
        )
