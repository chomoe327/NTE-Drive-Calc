# 全量扫描会话快照与签名校验。
"""Scan session snapshot used for grid mapping and inventory drift detection."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.features.identification.parser import item_signature_from_dict, normalized_signature_data
from src.features.scanning.naming import raw_drive_index_from_name


class InventoryChangedError(RuntimeError):
    """Raised when live inventory no longer matches the scan session snapshot."""


@dataclass
class ScanSessionEntry:
    scan_index: int
    uid: str
    quality: str
    item_type: str
    signature: str
    source_filename: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScanSession:
    session_id: str
    created_at: str
    total_drives: int
    cols: int
    entries: list[ScanSessionEntry]

    def entry_by_index(self, scan_index: int) -> ScanSessionEntry | None:
        return next((e for e in self.entries if e.scan_index == scan_index), None)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "total_drives": self.total_drives,
            "cols": self.cols,
            "entries": [e.to_dict() for e in self.entries],
        }

    @classmethod
    def from_dict(cls, data: dict) -> ScanSession:
        entries = [
            ScanSessionEntry(**entry)
            for entry in data.get("entries", [])
            if isinstance(entry, dict)
        ]
        return cls(
            session_id=str(data.get("session_id") or uuid.uuid4()),
            created_at=str(data.get("created_at") or _now_iso()),
            total_drives=int(data.get("total_drives") or 0),
            cols=int(data.get("cols") or 7),
            entries=entries,
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def signature_from_item_dict(item: dict) -> str:
    return item_signature_from_dict(normalized_signature_data(item))


def build_session_from_inventory(
    inventory: list[dict],
    *,
    source_filenames: dict[int, str] | None = None,
) -> ScanSession:
    items = [
        item
        for item in inventory
        if isinstance(item, dict) and item.get("item_type") in ("drive", "tape")
    ]
    missing = [item.get("uid", "?") for item in items if not item.get("scan_index")]
    if missing:
        raise ValueError("库存中存在缺少 scan_index 的驱动/卡带，请先进行全量扫描。")

    entries: list[ScanSessionEntry] = []
    for item in items:
        scan_index = int(item["scan_index"])
        filenames = source_filenames or {}
        entries.append(
            ScanSessionEntry(
                scan_index=scan_index,
                uid=str(item.get("uid") or ""),
                quality=str(item.get("quality") or "Gold"),
                item_type=str(item.get("item_type") or "drive"),
                signature=signature_from_item_dict(item),
                source_filename=filenames.get(scan_index),
            )
        )
    entries.sort(key=lambda e: e.scan_index)
    total = max((e.scan_index for e in entries), default=0)
    if len(entries) != total:
        raise ValueError("scan_index 不连续或存在重复，请重新全量扫描。")
    return ScanSession(
        session_id=str(uuid.uuid4()),
        created_at=_now_iso(),
        total_drives=total,
        cols=7,
        entries=entries,
    )


def attach_scan_index_from_filename(item_dict: dict, filename: str | None) -> dict:
    idx = raw_drive_index_from_name(filename or "")
    if idx is not None:
        item_dict["scan_index"] = idx
    return item_dict


def load_session(path: Path) -> ScanSession | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return ScanSession.from_dict(data)
    except Exception:
        return None
    return None


def save_session(path: Path, session: ScanSession) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(session.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def verify_signature(expected: str, actual: str) -> None:
    if expected != actual:
        raise InventoryChangedError("当前格子装备与扫描快照不一致，背包可能已变动。")
