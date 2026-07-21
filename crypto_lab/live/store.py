"""SQLite 持久化：成交、快照与买卖点清单。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .models import AccountSnapshot, Fill, TradePoint


class RuntimeStore:
    """轻量 SQLite 状态库，支撑控制台重启后恢复。

    使用单文件库避免额外依赖；所有写操作自动提交。表结构保持窄而稳定，
    复杂对象以 JSON 文本存储，方便后续迁移。

    @author Cursor
    @since 0.3.0
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        """创建成交、快照与买卖点表（若不存在）。"""

        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS fills (
                id TEXT PRIMARY KEY,
                deployment_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS snapshots (
                deployment_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS trade_points (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                deployment_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_fills_dep ON fills(deployment_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_points_dep ON trade_points(deployment_id, date);
            """
        )
        self._conn.commit()

    def save_fill(self, fill: Fill) -> None:
        """写入或覆盖一笔成交/意向单。"""

        self._conn.execute(
            "INSERT OR REPLACE INTO fills(id, deployment_id, payload, timestamp) VALUES (?, ?, ?, ?)",
            (fill.id, fill.deployment_id, json.dumps(fill.to_dict(), ensure_ascii=False), fill.timestamp),
        )
        self._conn.commit()

    def list_fills(self, deployment_id: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        """按时间倒序读取成交记录。"""

        if deployment_id:
            rows = self._conn.execute(
                "SELECT payload FROM fills WHERE deployment_id=? ORDER BY timestamp DESC LIMIT ?",
                (deployment_id, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT payload FROM fills ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def save_snapshot(self, snapshot: AccountSnapshot) -> None:
        """保存部署的最新账户快照。"""

        self._conn.execute(
            "INSERT OR REPLACE INTO snapshots(deployment_id, payload, updated_at) VALUES (?, ?, ?)",
            (
                snapshot.deployment_id,
                json.dumps(snapshot.to_dict(), ensure_ascii=False),
                snapshot.updated_at,
            ),
        )
        self._conn.commit()

    def get_snapshot(self, deployment_id: str) -> dict[str, Any] | None:
        """读取单个部署快照。"""

        row = self._conn.execute(
            "SELECT payload FROM snapshots WHERE deployment_id=?",
            (deployment_id,),
        ).fetchone()
        return json.loads(row["payload"]) if row else None

    def list_snapshots(self) -> list[dict[str, Any]]:
        """读取全部部署快照。"""

        rows = self._conn.execute(
            "SELECT payload FROM snapshots ORDER BY deployment_id"
        ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def replace_trade_points(self, deployment_id: str, points: Iterable[TradePoint]) -> None:
        """用新的买卖点清单替换指定部署的历史记录。"""

        self._conn.execute(
            "DELETE FROM trade_points WHERE deployment_id=?",
            (deployment_id,),
        )
        for point in points:
            self._conn.execute(
                """
                INSERT INTO trade_points(deployment_id, mode, date, symbol, side, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    point.deployment_id,
                    point.mode,
                    point.date,
                    point.symbol,
                    point.side,
                    json.dumps(point.to_dict(), ensure_ascii=False),
                ),
            )
        self._conn.commit()

    def append_trade_point(self, point: TradePoint) -> None:
        """追加一条运行时产生的买卖点。"""

        self._conn.execute(
            """
            INSERT INTO trade_points(deployment_id, mode, date, symbol, side, payload)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                point.deployment_id,
                point.mode,
                point.date,
                point.symbol,
                point.side,
                json.dumps(point.to_dict(), ensure_ascii=False),
            ),
        )
        self._conn.commit()

    def list_trade_points(
        self,
        deployment_id: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """读取买卖点清单，默认按日期倒序。"""

        if deployment_id:
            rows = self._conn.execute(
                """
                SELECT payload FROM trade_points
                WHERE deployment_id=?
                ORDER BY date DESC, id DESC
                LIMIT ?
                """,
                (deployment_id, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT payload FROM trade_points
                ORDER BY date DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def close(self) -> None:
        """关闭数据库连接。"""

        self._conn.close()
