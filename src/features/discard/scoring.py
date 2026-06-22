# 盲筛评分与目标列表生成。
"""Offline scoring and target list generation for marking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.features.discard.quality_rules import Action, QualityMarkRule, resolve_action
from src.features.discard.scan_session import ScanSession
from src.models.equipment import Drive
from src.optimizer.scoring import ScoringEngine

ActionType = Literal["discard", "lock"]


@dataclass
class MarkTarget:
    scan_index: int
    uid: str
    quality: str
    max_score: float
    action: ActionType

    def to_dict(self) -> dict:
        return {
            "scan_index": self.scan_index,
            "uid": self.uid,
            "quality": self.quality,
            "max_score": self.max_score,
            "action": self.action,
        }


@dataclass
class MarkPreview:
    targets: list[MarkTarget]
    discard_count: int
    lock_count: int
    blue_skipped: int
    total_scored: int


def build_mark_preview(
    session: ScanSession,
    selected_roles: list[str],
    rules: dict[str, QualityMarkRule],
    engine: ScoringEngine,
    inventory: list[dict],
) -> MarkPreview:
    targets: list[MarkTarget] = []
    blue_skipped = 0
    total_scored = 0
    by_uid = {
        str(item.get("uid")): item
        for item in inventory
        if isinstance(item, dict) and item.get("uid")
    }

    for entry in session.entries:
        if entry.item_type != "drive":
            continue
        total_scored += 1
        item_data = by_uid.get(entry.uid)
        if not item_data:
            raise ValueError(f"库存中找不到 uid={entry.uid} 的驱动数据")
        drive = Drive.model_validate(item_data)

        scores: dict[str, float] = {}
        for role in selected_roles:
            role_data = engine.roles_db.get(role, {})
            weights = role_data.get("weights", {})
            max_weight = engine._get_max_theoretical_weight(weights)
            scores[role] = engine.calculate_drive_score(drive, weights, max_weight)
        max_score = max(scores.values()) if scores else 0.0
        if entry.quality == "Blue":
            blue_skipped += 1
            continue
        action: Action | None = resolve_action(entry.quality, max_score, rules)  # type: ignore[arg-type]
        if action is None:
            continue
        targets.append(
            MarkTarget(
                scan_index=entry.scan_index,
                uid=entry.uid,
                quality=entry.quality,
                max_score=max_score,
                action=action,
            )
        )

    targets.sort(key=lambda t: t.scan_index)
    discard_count = sum(1 for t in targets if t.action == "discard")
    lock_count = sum(1 for t in targets if t.action == "lock")
    return MarkPreview(
        targets=targets,
        discard_count=discard_count,
        lock_count=lock_count,
        blue_skipped=blue_skipped,
        total_scored=total_scored,
    )
