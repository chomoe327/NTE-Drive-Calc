# 为自动装配测试计算单块驱动的放置坐标。
"""Placement planning helpers for assembly automation tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

from src.app import runtime
from src.features.role.dao import load_real_inventory
from src.solver.dfs_puzzle import DFSPuzzleSolver
from src.solver.orchestrator import NTEPipelineOrchestrator

ASSEMBLY_BLUEPRINT_ROLE = "九原"
ASSEMBLY_UI_ROLE = "真红"
ASSEMBLY_TEST_ROLE = ASSEMBLY_BLUEPRINT_ROLE
ASSEMBLY_TEST_SHAPE = "H_2"


@dataclass(frozen=True)
class PlacementPlan:
    role_name: str
    piece_id: str
    start_r: int
    start_c: int
    piece_matrix: List[List[int]]
    board_matrix: List[List[int]]
    solved_board: List[List[int]]


@dataclass(frozen=True)
class AssemblyPlan:
    role_name: str
    piece_id: str
    start_r: int
    start_c: int
    blueprint_layout: list[list[str]]
    inventory_drive: dict[str, Any] | None
    equipped_drive: dict[str, Any] | None
    piece_matrix: List[List[int]]


@dataclass(frozen=True)
class AssemblyPiecePlan:
    piece_id: str
    start_r: int
    start_c: int
    piece_matrix: List[List[int]]


@dataclass(frozen=True)
class FullAssemblyPlan:
    blueprint_role: str
    blueprint_layout: list[list[str]]
    pieces: list[AssemblyPiecePlan]


def _first_empty_cell(board: List[List[int]]) -> tuple[int, int]:
    for row_index, row in enumerate(board):
        for col_index, cell in enumerate(row):
            if cell == 0:
                return row_index, col_index
    raise ValueError("底盘上没有可放置的空格。")


def _piece_anchor(piece_matrix: List[List[int]]) -> tuple[int, int]:
    for row_index, row in enumerate(piece_matrix):
        for col_index, cell in enumerate(row):
            if cell == 1:
                return row_index, col_index
    raise ValueError("形状矩阵中没有有效占用格。")


def find_piece_anchor(board: List[List[int]], piece_id: str, piece_matrix: List[List[int]]) -> tuple[int, int]:
    rows = len(board)
    cols = len(board[0]) if board else 0
    piece_rows = len(piece_matrix)
    piece_cols = len(piece_matrix[0]) if piece_matrix else 0

    for start_r in range(rows):
        for start_c in range(cols):
            matched = True
            for pr in range(piece_rows):
                for pc in range(piece_cols):
                    if piece_matrix[pr][pc] != 1:
                        continue
                    if board[start_r + pr][start_c + pc] != piece_id:
                        matched = False
                        break
                if not matched:
                    break
            if matched:
                return start_r, start_c
    raise ValueError(f"在求解结果中未找到 {piece_id} 的放置位置。")


def load_equipped_state(state_path: Path | None = None) -> dict[str, Any]:
    path = state_path or runtime.USER_CONFIG_DIR / "equipped_state.json"
    if not path.exists():
        raise FileNotFoundError(f"找不到配装文件: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"配装文件格式无效: {path}")
    return data


def blueprint_layout_to_board(blueprint_layout: list[list[Any]]) -> list[list[Any]]:
    board: list[list[Any]] = []
    for row in blueprint_layout or []:
        converted: list[Any] = []
        for cell in row:
            if cell in ("XX", -1):
                converted.append(-1)
            elif cell in ("0", 0):
                converted.append(0)
            else:
                converted.append(str(cell))
        board.append(converted)
    return board


def find_shape_anchor_in_blueprint(
    blueprint_layout: list[list[Any]],
    piece_id: str,
    piece_matrix: List[List[int]],
) -> tuple[int, int]:
    board = blueprint_layout_to_board(blueprint_layout)
    if not board:
        raise ValueError("配装图纸为空，无法定位驱动块位置。")
    return find_piece_anchor(board, piece_id, piece_matrix)


def _is_empty_blueprint_cell(cell: Any) -> bool:
    return cell in ("XX", -1, "0", 0, None, "")


def iter_blueprint_piece_placements(
    blueprint_layout: list[list[Any]],
    shapes_db: dict[str, Any],
) -> list[AssemblyPiecePlan]:
    """Enumerate every drive piece instance on a saved blueprint layout."""
    board = blueprint_layout_to_board(blueprint_layout)
    if not board:
        raise ValueError("配装图纸为空，无法解析驱动块位置。")

    rows = len(board)
    cols = len(board[0]) if board else 0
    visited = [[False] * cols for _ in range(rows)]
    placements: list[AssemblyPiecePlan] = []

    for row_index in range(rows):
        for col_index in range(cols):
            cell = board[row_index][col_index]
            if _is_empty_blueprint_cell(cell) or visited[row_index][col_index]:
                continue

            piece_id = str(cell)
            if piece_id not in shapes_db:
                raise ValueError(f"图纸中的形状 {piece_id!r} 不存在于 shapes.json。")

            component_cells: list[tuple[int, int]] = []
            stack = [(row_index, col_index)]
            visited[row_index][col_index] = True
            while stack:
                current_r, current_c = stack.pop()
                component_cells.append((current_r, current_c))
                for delta_r, delta_c in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    next_r = current_r + delta_r
                    next_c = current_c + delta_c
                    if not (0 <= next_r < rows and 0 <= next_c < cols):
                        continue
                    if visited[next_r][next_c]:
                        continue
                    if board[next_r][next_c] != piece_id:
                        continue
                    visited[next_r][next_c] = True
                    stack.append((next_r, next_c))

            piece_matrix = shapes_db[piece_id].matrix
            sub_board = [[0 for _ in range(cols)] for _ in range(rows)]
            for current_r, current_c in component_cells:
                sub_board[current_r][current_c] = piece_id
            start_r, start_c = find_piece_anchor(sub_board, piece_id, piece_matrix)
            placements.append(
                AssemblyPiecePlan(
                    piece_id=piece_id,
                    start_r=start_r,
                    start_c=start_c,
                    piece_matrix=piece_matrix,
                )
            )

    placements.sort(key=lambda item: (item.start_r, item.start_c, item.piece_id))
    return placements


def find_first_inventory_drive(
    inventory: list[dict[str, Any]] | dict[str, Any] | None,
    shape_id: str,
) -> dict[str, Any] | None:
    if not inventory:
        return None
    items = inventory if isinstance(inventory, list) else inventory.get("drives", [])
    for item in items or []:
        if not isinstance(item, dict):
            continue
        if item.get("shape_id") == shape_id:
            return item
    return None


def find_equipped_drive(role_state: dict[str, Any], shape_id: str) -> dict[str, Any] | None:
    drives = role_state.get("equipped_drives") or []
    for drive in drives:
        if isinstance(drive, dict) and drive.get("shape_id") == shape_id:
            return drive
    return None


def plan_assembly(
    role_name: str = ASSEMBLY_TEST_ROLE,
    piece_id: str = ASSEMBLY_TEST_SHAPE,
    *,
    config_dir: str = "config",
    equipped_state: dict[str, Any] | None = None,
    inventory: list[dict[str, Any]] | None = None,
    state_path: Path | None = None,
) -> AssemblyPlan:
    """Build an assembly plan from saved loadout and inventory metadata."""
    state = equipped_state if equipped_state is not None else load_equipped_state(state_path)
    role_state = state.get(role_name)
    if not isinstance(role_state, dict):
        raise ValueError(f"配装文件中没有角色 {role_name!r} 的方案。")

    blueprint_layout = role_state.get("blueprint_layout") or []
    if not blueprint_layout:
        raise ValueError(f"角色 {role_name!r} 的配装图纸为空，请先在配装页保存统筹结果。")

    orchestrator = NTEPipelineOrchestrator(config_dir=config_dir)
    if piece_id not in orchestrator.shapes_db:
        raise ValueError(f"形状 {piece_id!r} 不存在于 shapes.json。")
    piece_matrix = orchestrator.shapes_db[piece_id].matrix

    start_r, start_c = find_shape_anchor_in_blueprint(blueprint_layout, piece_id, piece_matrix)
    inventory_items = inventory if inventory is not None else load_real_inventory()
    inventory_drive = find_first_inventory_drive(inventory_items, piece_id)
    equipped_drive = find_equipped_drive(role_state, piece_id)

    return AssemblyPlan(
        role_name=role_name,
        piece_id=piece_id,
        start_r=start_r,
        start_c=start_c,
        blueprint_layout=[list(row) for row in blueprint_layout],
        inventory_drive=inventory_drive,
        equipped_drive=equipped_drive,
        piece_matrix=piece_matrix,
    )


def plan_full_assembly(
    role_name: str = ASSEMBLY_BLUEPRINT_ROLE,
    *,
    config_dir: str = "config",
    equipped_state: dict[str, Any] | None = None,
    state_path: Path | None = None,
) -> FullAssemblyPlan:
    """Build a multi-piece assembly plan from a saved role blueprint layout."""
    state = equipped_state if equipped_state is not None else load_equipped_state(state_path)
    role_state = state.get(role_name)
    if not isinstance(role_state, dict):
        raise ValueError(f"配装文件中没有角色 {role_name!r} 的方案。")

    blueprint_layout = role_state.get("blueprint_layout") or []
    if not blueprint_layout:
        raise ValueError(f"角色 {role_name!r} 的配装图纸为空，请先在配装页保存统筹结果。")

    orchestrator = NTEPipelineOrchestrator(config_dir=config_dir)
    pieces = iter_blueprint_piece_placements(blueprint_layout, orchestrator.shapes_db)
    if not pieces:
        raise ValueError(f"角色 {role_name!r} 的配装图纸中没有可装配的驱动块。")

    return FullAssemblyPlan(
        blueprint_role=role_name,
        blueprint_layout=[list(row) for row in blueprint_layout],
        pieces=pieces,
    )


def plan_test_placement(
    role_name: str = "真红",
    piece_id: str = "H_2",
    config_dir: str = "config",
) -> PlacementPlan:
    """Solve a single-piece placement on the role board and return anchor coordinates."""
    orchestrator = NTEPipelineOrchestrator(config_dir=config_dir)
    if role_name not in orchestrator.roles_db:
        raise ValueError(f"角色 {role_name!r} 不存在于 roles.json。")
    if piece_id not in orchestrator.shapes_db:
        raise ValueError(f"形状 {piece_id!r} 不存在于 shapes.json。")

    board_matrix = [row[:] for row in orchestrator.roles_db[role_name]["board_matrix"]]
    piece_matrix = orchestrator.shapes_db[piece_id].matrix
    solver = DFSPuzzleSolver(orchestrator.shapes_db)

    results: list[list] = []
    solver.solve(board_matrix, [piece_id], results, max_solutions=1)
    if not results:
        raise ValueError(f"无法在 [{role_name}] 底盘上放置 {piece_id}。")

    solved_board = results[0]
    target_r, target_c = _first_empty_cell(board_matrix)
    anchor_r, anchor_c = _piece_anchor(piece_matrix)
    start_r = target_r - anchor_r
    start_c = target_c - anchor_c

    if not solver.can_place(board_matrix, piece_matrix, start_r, start_c):
        start_r, start_c = find_piece_anchor(solved_board, piece_id, piece_matrix)

    return PlacementPlan(
        role_name=role_name,
        piece_id=piece_id,
        start_r=start_r,
        start_c=start_c,
        piece_matrix=piece_matrix,
        board_matrix=board_matrix,
        solved_board=solved_board,
    )
