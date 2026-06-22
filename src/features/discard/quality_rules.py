# 金/紫双阈值弃置与上锁规则（等级制）。
"""Per-quality discard/lock grade rules for blind marking."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from src.domain.grade_scoring import GRADE_LADDER, UI_GRADE_OPTIONS, grade_rank

Quality = Literal["Gold", "Purple"]
Action = Literal["discard", "lock"]
Grade = str


@dataclass
class QualityMarkRule:
    quality: Quality
    discard_grade: Grade = "B"
    lock_grade: Grade = "SS"
    discard_below_enabled: bool = False
    lock_above_enabled: bool = False

    def validate(self) -> str | None:
        if self.discard_grade not in GRADE_LADDER or self.lock_grade not in GRADE_LADDER:
            return "等级必须在 D~ACE 范围内"
        if grade_rank(self.lock_grade) <= grade_rank(self.discard_grade):
            return "上锁等级必须高于弃置等级"
        if not self.discard_below_enabled and not self.lock_above_enabled:
            return "至少开启弃置或上锁其中一项"
        return None


DEFAULT_RULES: dict[Quality, QualityMarkRule] = {
    "Gold": QualityMarkRule("Gold", "B", "SS", discard_below_enabled=True, lock_above_enabled=False),
    "Purple": QualityMarkRule("Purple", "C", "S", discard_below_enabled=False, lock_above_enabled=True),
}


def resolve_action(quality: str, max_grade: str, rules: dict[Quality, QualityMarkRule]) -> Action | None:
    rule = rules.get(quality)  # type: ignore[arg-type]
    if rule is None:
        return None
    current = grade_rank(max_grade)
    if rule.discard_below_enabled and current < grade_rank(rule.discard_grade):
        return "discard"
    if rule.lock_above_enabled and current >= grade_rank(rule.lock_grade):
        return "lock"
    return None


def validate_rules(rules: dict[Quality, QualityMarkRule]) -> str | None:
    enabled = False
    for rule in rules.values():
        err = rule.validate()
        if err:
            return f"{rule.quality}: {err}"
        if rule.discard_below_enabled or rule.lock_above_enabled:
            enabled = True
    if not enabled:
        return "至少为一个品质开启弃置或上锁规则"
    return None


def _grade_from_entry(entry: dict, key: str, default: str) -> str:
    value = entry.get(key)
    if isinstance(value, str) and value.upper() in GRADE_LADDER:
        return value.upper()
    return default


def load_rules(path: Path) -> dict[Quality, QualityMarkRule]:
    if not path.exists():
        return {k: QualityMarkRule(**asdict(v)) for k, v in DEFAULT_RULES.items()}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {k: QualityMarkRule(**asdict(v)) for k, v in DEFAULT_RULES.items()}
    rules: dict[Quality, QualityMarkRule] = {}
    for quality in ("Gold", "Purple"):
        entry = raw.get(quality, {})
        default = DEFAULT_RULES[quality]
        rules[quality] = QualityMarkRule(
            quality=quality,
            discard_grade=_grade_from_entry(entry, "discard_grade", default.discard_grade),
            lock_grade=_grade_from_entry(entry, "lock_grade", default.lock_grade),
            discard_below_enabled=bool(entry.get("discard_below_enabled", default.discard_below_enabled)),
            lock_above_enabled=bool(entry.get("lock_above_enabled", default.lock_above_enabled)),
        )
    return rules


def save_rules(path: Path, rules: dict[Quality, QualityMarkRule]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {quality: asdict(rule) for quality, rule in rules.items()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
