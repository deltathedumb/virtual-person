from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(slots=True, frozen=True)
class Memory:
    id: int
    sim_time: float
    kind: str
    summary: str
    importance: float
    metadata: dict[str, Any]


class MemoryStore:
    """Small SQLite-backed episodic memory store."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._db = sqlite3.connect(self.path)
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sim_time REAL NOT NULL,
                kind TEXT NOT NULL,
                summary TEXT NOT NULL,
                importance REAL NOT NULL,
                metadata_json TEXT NOT NULL
            )
            """
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS ix_memories_time ON memories(sim_time DESC)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS ix_memories_kind ON memories(kind)"
        )
        self._db.commit()

    def add(
        self,
        sim_time: float,
        kind: str,
        summary: str,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        cur = self._db.execute(
            """
            INSERT INTO memories(sim_time, kind, summary, importance, metadata_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                float(sim_time),
                kind,
                summary,
                max(0.0, min(1.0, importance)),
                json.dumps(metadata or {}, separators=(",", ":")),
            ),
        )
        self._db.commit()
        return int(cur.lastrowid)

    def recent(self, limit: int = 20, kind: str | None = None) -> list[Memory]:
        if kind is None:
            rows = self._db.execute(
                """
                SELECT id, sim_time, kind, summary, importance, metadata_json
                FROM memories ORDER BY sim_time DESC, id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        else:
            rows = self._db.execute(
                """
                SELECT id, sim_time, kind, summary, importance, metadata_json
                FROM memories WHERE kind = ?
                ORDER BY sim_time DESC, id DESC LIMIT ?
                """,
                (kind, limit),
            ).fetchall()
        return [
            Memory(
                id=row[0],
                sim_time=row[1],
                kind=row[2],
                summary=row[3],
                importance=row[4],
                metadata=json.loads(row[5]),
            )
            for row in rows
        ]

    def search(self, text: str, limit: int = 20) -> list[Memory]:
        pattern = f"%{text}%"
        rows = self._db.execute(
            """
            SELECT id, sim_time, kind, summary, importance, metadata_json
            FROM memories
            WHERE summary LIKE ?
            ORDER BY importance DESC, sim_time DESC
            LIMIT ?
            """,
            (pattern, limit),
        ).fetchall()
        return [
            Memory(
                id=row[0],
                sim_time=row[1],
                kind=row[2],
                summary=row[3],
                importance=row[4],
                metadata=json.loads(row[5]),
            )
            for row in rows
        ]

    def consolidate(self, before_time: float, keep_importance: float = 0.75) -> int:
        """Delete low-importance old memories after a summary has been stored."""
        cur = self._db.execute(
            """
            DELETE FROM memories
            WHERE sim_time < ? AND importance < ? AND kind != 'sleep_summary'
            """,
            (before_time, keep_importance),
        )
        self._db.commit()
        return cur.rowcount

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
