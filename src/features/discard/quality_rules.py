# 金/紫双阈值弃置与上锁规则。
"""Per-quality discard/lock threshold rules for blind marking."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

Quality = Literal["Gold", "Purple"]
Action = Literal["discard", "lock"]


@dataclass
class QualityMarkRule:
    quality: Quality
    discard_threshold: float = 10.0
    lock_threshold: float = 18.0
    discard_below_enabled: bool = False
    lock_above_enabled: bool = False

    def validate(self) -> str | None:
        if self.lock_threshold <= self.discard_threshold:
            return "上锁阈值必须大于弃置阈值"
        if not self.discard_below_enabled and not self.lock_above_enabled:
            return "至少开启弃置或上锁其中一项"
        return None


DEFAULT_RULES: dict[Quality, QualityMarkRule] = {
    "Gold": QualityMarkRule("Gold", 10.0, 18.0, discard_below_enabled=True, lock_above_enabled=False),
    "Purple": QualityMarkRule("Purple", 8.0, 15.0, discard_below_enabled=False, lock_above_enabled=True),
}


def resolve_action(quality: str, max_score: float, rules: dict[Quality, QualityMarkRule]) -> Action | None:
    rule = rules.get(quality)  # type: ignore[arg-type]
    if rule is None:
        return None
    if rule.discard_below_enabled and max_score < rule.discard_threshold:
        return "discard"
    if rule.lock_above_enabled and max_score >= rule.lock_threshold:
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
            discard_threshold=float(entry.get("discard_threshold", default.discard_threshold)),
            lock_threshold=float(entry.get("lock_threshold", default.lock_threshold)),
            discard_below_enabled=bool(entry.get("discard_below_enabled", default.discard_below_enabled)),
            lock_above_enabled=bool(entry.get("lock_above_enabled", default.lock_above_enabled)),
        )
    return rules


def save_rules(path: Path, rules: dict[Quality, QualityMarkRule]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {quality: asdict(rule) for quality, rule in rules.items()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
