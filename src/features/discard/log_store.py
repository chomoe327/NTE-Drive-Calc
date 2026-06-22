# 盲筛标记执行日志与回滚。
"""Persistent log for marking sessions and rollback support."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

EntryStatus = Literal["marked", "skipped_already_marked", "unmarked"]


@dataclass
class MarkLogEntry:
    scan_index: int
    uid: str
    action: str
    status: EntryStatus
    marked_at: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MarkLogSession:
    id: str
    scan_session_id: str
    created_at: str
    rules: dict
    roles: list[str]
    entries: list[MarkLogEntry]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "scan_session_id": self.scan_session_id,
            "created_at": self.created_at,
            "rules": self.rules,
            "roles": self.roles,
            "entries": [e.to_dict() for e in self.entries],
        }

    @classmethod
    def from_dict(cls, data: dict) -> MarkLogSession:
        entries = [MarkLogEntry(**entry) for entry in data.get("entries", []) if isinstance(entry, dict)]
        return cls(
            id=str(data.get("id") or uuid.uuid4()),
            scan_session_id=str(data.get("scan_session_id") or ""),
            created_at=str(data.get("created_at") or ""),
            rules=dict(data.get("rules") or {}),
            roles=list(data.get("roles") or []),
            entries=entries,
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MarkLogStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    def _load_all(self) -> list[MarkLogSession]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            sessions = raw.get("sessions", []) if isinstance(raw, dict) else []
            return [MarkLogSession.from_dict(item) for item in sessions if isinstance(item, dict)]
        except Exception:
            return []

    def _save_all(self, sessions: list[MarkLogSession]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"sessions": [session.to_dict() for session in sessions]}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def append_session(self, session: MarkLogSession) -> None:
        sessions = self._load_all()
        sessions.append(session)
        self._save_all(sessions)

    def latest_session(self) -> MarkLogSession | None:
        sessions = self._load_all()
        return sessions[-1] if sessions else None

    def update_session(self, session: MarkLogSession) -> None:
        sessions = self._load_all()
        for index, existing in enumerate(sessions):
            if existing.id == session.id:
                sessions[index] = session
                self._save_all(sessions)
                return
        sessions.append(session)
        self._save_all(sessions)

    def rollback_candidates(self, session: MarkLogSession) -> list[MarkLogEntry]:
        marked = [entry for entry in session.entries if entry.status == "marked" and entry.marked_at]
        marked.sort(key=lambda e: e.marked_at or "", reverse=True)
        return marked
