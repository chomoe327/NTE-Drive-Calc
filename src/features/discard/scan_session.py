# 全量扫描会话快照与签名校验。
"""Scan session snapshot used for grid mapping and inventory drift detection."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.features.identification.parser import equipment_identity_signature
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
    is_duplicate_signature: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScanSession:
    session_id: str
    created_at: str
    total_drives: int
    cols: int
    entries: list[ScanSessionEntry]
    signature_to_indices: dict[str, list[int]] = field(default_factory=dict)

    def entry_by_index(self, scan_index: int) -> ScanSessionEntry | None:
        return next((e for e in self.entries if e.scan_index == scan_index), None)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "total_drives": self.total_drives,
            "cols": self.cols,
            "entries": [e.to_dict() for e in self.entries],
            "signature_to_indices": self.signature_to_indices,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ScanSession:
        entries = [
            ScanSessionEntry(
                scan_index=int(entry["scan_index"]),
                uid=str(entry.get("uid") or ""),
                quality=str(entry.get("quality") or "Gold"),
                item_type=str(entry.get("item_type") or "drive"),
                signature=str(entry.get("signature") or ""),
                source_filename=entry.get("source_filename"),
                is_duplicate_signature=bool(entry.get("is_duplicate_signature", False)),
            )
            for entry in data.get("entries", [])
            if isinstance(entry, dict)
        ]
        signature_to_indices = data.get("signature_to_indices")
        if not isinstance(signature_to_indices, dict):
            signature_to_indices = build_signature_to_indices(entries)
        _apply_duplicate_flags(entries, signature_to_indices)
        return cls(
            session_id=str(data.get("session_id") or uuid.uuid4()),
            created_at=str(data.get("created_at") or _now_iso()),
            total_drives=int(data.get("total_drives") or 0),
            cols=int(data.get("cols") or 7),
            entries=entries,
            signature_to_indices={
                str(sig): [int(i) for i in indices]
                for sig, indices in signature_to_indices.items()
                if isinstance(indices, list)
            },
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def signature_from_item_dict(item: dict) -> str:
    return equipment_identity_signature(item)


def build_signature_to_indices(entries: list[ScanSessionEntry]) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    for entry in entries:
        index.setdefault(entry.signature, []).append(entry.scan_index)
    for indices in index.values():
        indices.sort()
    return index


def _apply_duplicate_flags(
    entries: list[ScanSessionEntry],
    signature_to_indices: dict[str, list[int]],
) -> None:
    for entry in entries:
        entry.is_duplicate_signature = len(signature_to_indices.get(entry.signature, [])) > 1


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
        signature = signature_from_item_dict(item)
        entries.append(
            ScanSessionEntry(
                scan_index=scan_index,
                uid=str(item.get("uid") or ""),
                quality=str(item.get("quality") or "Gold"),
                item_type=str(item.get("item_type") or "drive"),
                signature=signature,
                source_filename=filenames.get(scan_index),
            )
        )
    entries.sort(key=lambda e: e.scan_index)
    total = max((e.scan_index for e in entries), default=0)
    if len(entries) != total:
        raise ValueError("scan_index 不连续或存在重复，请重新全量扫描。")

    signature_to_indices = build_signature_to_indices(entries)
    _apply_duplicate_flags(entries, signature_to_indices)
    for item in items:
        item["is_duplicate_signature"] = len(signature_to_indices.get(signature_from_item_dict(item), [])) > 1

    return ScanSession(
        session_id=str(uuid.uuid4()),
        created_at=_now_iso(),
        total_drives=total,
        cols=7,
        entries=entries,
        signature_to_indices=signature_to_indices,
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
