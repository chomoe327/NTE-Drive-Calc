# 为自动装配测试计算单块驱动的放置坐标。
"""Placement planning helpers for assembly automation tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.solver.dfs_puzzle import DFSPuzzleSolver
from src.solver.orchestrator import NTEPipelineOrchestrator


@dataclass(frozen=True)
class PlacementPlan:
    role_name: str
    piece_id: str
    start_r: int
    start_c: int
    piece_matrix: List[List[int]]
    board_matrix: List[List[int]]
    solved_board: List[List[int]]


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
