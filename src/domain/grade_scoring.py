# 装备等级评分与鉴定/盲筛共用的可用性判断。
"""Shared grade ladder and per-role scoring aligned with the identify page."""

from __future__ import annotations

from typing import Any

from src.models.equipment import Drive, Tape
from src.optimizer.scoring import ScoringEngine

GRADE_LADDER: tuple[str, ...] = ("D", "C", "B", "A", "S", "SS", "SSS", "ACE")
UI_GRADE_OPTIONS: tuple[str, ...] = ("B", "A", "S", "SS", "SSS", "ACE")
TAPE_AREA = 15


def grade_rank(grade: str) -> int:
    normalized = str(grade or "D").upper()
    try:
        return GRADE_LADDER.index(normalized)
    except ValueError:
        return 0


def best_grade(grades: list[str]) -> str:
    if not grades:
        return "D"
    return max(grades, key=grade_rank)


def drive_usable_for_role(drive: Drive, role_name: str, orchestrator: Any, blueprints: dict) -> bool:
    role_bps = blueprints.get(role_name, [])
    if not role_bps:
        return False
    role_data = orchestrator.roles_db.get(role_name, {})
    target_set = orchestrator._resolve_set_name(role_data.get("default_set", ""))
    set_info = orchestrator.sets_db.get(target_set)
    if not set_info:
        return False
    set_shapes = set_info.get("shapes", [])
    in_set = drive.shape_id in set_shapes
    in_extra = any(drive.shape_id in bp.get("extra_pieces", []) for bp in role_bps)
    return in_set or in_extra


def tape_usable_for_role(tape: Tape, role_name: str, orchestrator: Any) -> bool:
    role_data = orchestrator.roles_db.get(role_name, {})
    target_set = orchestrator._resolve_set_name(role_data.get("default_set", ""))
    tape_set = orchestrator._resolve_set_name(tape.set_name)
    return tape_set == target_set


def score_drive_for_role(
    drive: Drive,
    role_name: str,
    orchestrator: Any,
    blueprints: dict,
    engine: ScoringEngine,
) -> tuple[float, str, str]:
    role_data = orchestrator.roles_db.get(role_name, {})
    target_set = orchestrator._resolve_set_name(role_data.get("default_set", ""))
    role_bps = blueprints.get(role_name, [])
    set_shapes = orchestrator.sets_db[target_set]["shapes"]
    in_set = drive.shape_id in set_shapes
    in_extra = any(drive.shape_id in bp.get("extra_pieces", []) for bp in role_bps)
    match_desc = "套装位" if in_set else "散件位"
    weights = role_data.get("weights", {})
    max_weight = engine._get_max_theoretical_weight(weights)
    score = engine.calculate_drive_score(drive, weights, max_weight)
    grade = engine.get_grade_tag(score, drive.area)
    return score, grade, match_desc


def score_tape_for_role(
    tape: Tape,
    role_name: str,
    orchestrator: Any,
    engine: ScoringEngine,
) -> tuple[float, str]:
    role_data = orchestrator.roles_db.get(role_name, {})
    weights = role_data.get("weights", {})
    max_weight = engine._get_max_theoretical_weight(weights)
    score = engine.calculate_cartridge_score(tape, weights, max_weight)
    grade = engine.get_grade_tag(score, TAPE_AREA)
    return score, grade


def evaluate_item_grades(
    item: Drive | Tape,
    selected_roles: list[str],
    orchestrator: Any,
    blueprints: dict,
    engine: ScoringEngine,
) -> tuple[str, str | None, bool]:
    """Return (max_grade, best_role, had_usable_role). No usable role -> ('D', None, False)."""
    if isinstance(item, Tape):
        resolved_set = orchestrator._resolve_set_name(item.set_name)
        if resolved_set != item.set_name:
            item = item.model_copy(update={"set_name": resolved_set})

    scored: dict[str, tuple[float, str]] = {}
    for role_name in selected_roles:
        if role_name not in orchestrator.roles_db:
            continue
        role_bps = blueprints.get(role_name, [])
        if not role_bps:
            continue
        if isinstance(item, Tape):
            if not tape_usable_for_role(item, role_name, orchestrator):
                continue
            score, grade = score_tape_for_role(item, role_name, orchestrator, engine)
        else:
            if not drive_usable_for_role(item, role_name, orchestrator, blueprints):
                continue
            score, grade, _match = score_drive_for_role(item, role_name, orchestrator, blueprints, engine)
        scored[role_name] = (score, grade)

    if not scored:
        return "D", None, False
    best_role = max(scored.keys(), key=lambda name: scored[name][0])
    return scored[best_role][1], best_role, True


def build_identify_rows(
    item: Drive | Tape,
    orchestrator: Any,
    blueprints: dict,
    engine: ScoringEngine,
) -> list[dict]:
    """Build per-role identify rows for a drive or tape (all roles in roles_db)."""
    if isinstance(item, Tape):
        resolved_set = orchestrator._resolve_set_name(item.set_name)
        if resolved_set != item.set_name:
            item = item.model_copy(update={"set_name": resolved_set})

    rows: list[dict] = []
    for role_name, role_data in orchestrator.roles_db.items():
        role_bps = blueprints.get(role_name, [])
        if not role_bps:
            continue
        target_set = orchestrator._resolve_set_name(role_data.get("default_set", ""))
        weights = role_data.get("weights", {})
        max_weight = engine._get_max_theoretical_weight(weights)
        if isinstance(item, Tape):
            if not tape_usable_for_role(item, role_name, orchestrator):
                continue
            score, grade = score_tape_for_role(item, role_name, orchestrator, engine)
            match_desc = "套装匹配"
            area = TAPE_AREA
        else:
            if not drive_usable_for_role(item, role_name, orchestrator, blueprints):
                continue
            score, grade, match_desc = score_drive_for_role(item, role_name, orchestrator, blueprints, engine)
            area = item.area
        max_score = area * 10.0
        rows.append(
            {
                "role": role_name,
                "set": target_set,
                "score": score,
                "grade": grade,
                "percent": round(score / max_score * 100, 1) if max_score else 0,
                "match": match_desc,
                "weights": weights,
            }
        )
    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows
