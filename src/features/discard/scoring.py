# 盲筛评分与目标列表生成。
"""Offline scoring and target list generation for marking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from src.domain.grade_scoring import evaluate_item_grades
from src.features.discard.quality_rules import Action, MarkingOptions, QualityMarkRule, resolve_action
from src.features.discard.scan_session import ScanSession
from src.models.equipment import Drive, Tape
from src.optimizer.scoring import ScoringEngine

ActionType = Literal["discard", "lock"]
ITEM_LABELS = {"drive": "驱动", "tape": "卡带"}


@dataclass
class MarkTarget:
    scan_index: int
    uid: str
    item_type: str
    quality: str
    max_grade: str
    best_role: str | None
    action: ActionType

    def to_dict(self) -> dict:
        return {
            "scan_index": self.scan_index,
            "uid": self.uid,
            "item_type": self.item_type,
            "quality": self.quality,
            "max_grade": self.max_grade,
            "best_role": self.best_role,
            "action": self.action,
        }


@dataclass
class MarkPreview:
    targets: list[MarkTarget]
    discard_count: int
    lock_count: int
    blue_skipped: int
    no_usable_role: int
    no_usable_role_skipped: int
    total_scored: int


def _parse_item(item_data: dict) -> Drive | Tape:
    if item_data.get("item_type") == "tape":
        return Tape.model_validate(item_data)
    return Drive.model_validate(item_data)


def build_mark_preview(
    session: ScanSession,
    selected_roles: list[str],
    rules: dict[str, QualityMarkRule],
    engine: ScoringEngine,
    inventory: list[dict],
    orchestrator: Any,
    blueprints: dict,
    options: MarkingOptions | None = None,
) -> MarkPreview:
    options = options or MarkingOptions()
    targets: list[MarkTarget] = []
    blue_skipped = 0
    no_usable_role = 0
    no_usable_role_skipped = 0
    total_scored = 0
    by_uid = {
        str(item.get("uid")): item
        for item in inventory
        if isinstance(item, dict) and item.get("uid")
    }

    for entry in session.entries:
        if entry.item_type not in ("drive", "tape"):
            continue
        total_scored += 1
        item_data = by_uid.get(entry.uid)
        if not item_data:
            raise ValueError(f"库存中找不到 uid={entry.uid} 的装备数据")
        item = _parse_item(item_data)

        max_grade, best_role, had_usable = evaluate_item_grades(
            item, selected_roles, orchestrator, blueprints, engine
        )
        if not had_usable:
            no_usable_role += 1

        if entry.quality == "Blue":
            blue_skipped += 1
            continue

        action: Action | None = resolve_action(entry.quality, max_grade, rules)  # type: ignore[arg-type]
        if action is None:
            continue
        if (
            options.ignore_no_usable_role
            and not had_usable
            and action == "discard"
        ):
            no_usable_role_skipped += 1
            continue
        targets.append(
            MarkTarget(
                scan_index=entry.scan_index,
                uid=entry.uid,
                item_type=entry.item_type,
                quality=entry.quality,
                max_grade=max_grade,
                best_role=best_role,
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
        no_usable_role=no_usable_role,
        no_usable_role_skipped=no_usable_role_skipped,
        total_scored=total_scored,
    )
