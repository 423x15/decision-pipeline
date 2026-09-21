from __future__ import annotations
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

_CREATE_CONVERSATIONS = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    title TEXT
);
"""
_CREATE_MESSAGES = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tool_trace TEXT,
    created_at TEXT NOT NULL
);
"""


class Database:
    def __init__(self, db_path: Path):
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = str(db_path)
        with self._connect() as conn:
            conn.execute(_CREATE_CONVERSATIONS)
            conn.execute(_CREATE_MESSAGES)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def create_conversation(self, title: str) -> str:
        conv_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, created_at, title) VALUES (?, ?, ?)",
                (conv_id, self._now(), title[:80]),
            )
        return conv_id

    def list_conversations(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, created_at FROM conversations ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def save_message(
        self,
        conv_id: str,
        role: str,
        content: str,
        tool_trace: list[dict] | None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, tool_trace, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    conv_id,
                    role,
                    content,
                    json.dumps(tool_trace) if tool_trace is not None else None,
                    self._now(),
                ),
            )

    def load_messages(self, conv_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content, tool_trace, created_at FROM messages "
                "WHERE conversation_id = ? ORDER BY id ASC",
                (conv_id,),
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["tool_trace"] = json.loads(d["tool_trace"]) if d["tool_trace"] else None
            result.append(d)
        return result

    def delete_conversation(self, conv_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
            conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
