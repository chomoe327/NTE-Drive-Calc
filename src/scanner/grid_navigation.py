# 全量扫描背包格子的 S 形路径生成。
"""Grid navigation helpers for inventory gamepad traversal.

Full scan (`GamepadScanner.start_scan`) uses `generate_path_commands`: one move
list per cell, visiting every slot in S-curve order.

Blind marking (`MarkingExecutor`) uses `moves_between_scan_indices`: direct D/U
then R/L between targets without walking intermediate cells.
"""

from __future__ import annotations

COLS = 7


def generate_scan_order(total_drives: int, cols: int = COLS) -> list[tuple[int, int]]:
    scan_order: list[tuple[int, int]] = []
    for row in range((total_drives + cols - 1) // cols):
        cols_in_row = min(cols, total_drives - row * cols)
        if row % 2 == 0:
            for col in range(cols_in_row):
                scan_order.append((row, col))
        else:
            for col in range(cols_in_row - 1, -1, -1):
                scan_order.append((row, col))
    return scan_order


def generate_path_commands(total_drives: int, cols: int = COLS) -> list[list[str]]:
    """Incremental S-curve steps for full inventory scan (one entry per cell)."""
    commands: list[list[str]] = []
    curr_row, curr_col = 0, 0
    for target_row, target_col in generate_scan_order(total_drives, cols):
        moves: list[str] = []
        while curr_col < target_col:
            moves.append("R")
            curr_col += 1
        while curr_col > target_col:
            moves.append("L")
            curr_col -= 1
        while curr_row < target_row:
            moves.append("D")
            curr_row += 1
        commands.append(moves)
    return commands


def moves_for_scan_index(scan_index: int, total_drives: int, cols: int = COLS) -> list[str]:
    if not 1 <= scan_index <= total_drives:
        raise ValueError(f"scan_index {scan_index} 超出范围 1-{total_drives}")
    return generate_path_commands(total_drives, cols)[scan_index - 1]


_INVERT_MOVE = {"R": "L", "L": "R", "D": "U", "U": "D"}
MOVE_DELTAS = {"R": (0, 1), "L": (0, -1), "D": (1, 0), "U": (-1, 0)}


def cols_in_row(row: int, total_drives: int, cols: int = COLS) -> int:
    if row < 0:
        return 0
    return min(cols, max(0, total_drives - row * cols))


def index_to_position(scan_index: int, total_drives: int, cols: int = COLS) -> tuple[int, int]:
    if not 1 <= scan_index <= total_drives:
        raise ValueError(f"scan_index {scan_index} 超出范围 1-{total_drives}")
    return generate_scan_order(total_drives, cols)[scan_index - 1]


def position_to_index(row: int, col: int, total_drives: int, cols: int = COLS) -> int:
    return scan_index_for_position(row, col, total_drives, cols)


def scan_index_for_position(
    row: int,
    col: int,
    total_drives: int,
    cols: int = COLS,
) -> int:
    for index, position in enumerate(generate_scan_order(total_drives, cols), 1):
        if position == (row, col):
            return index
    raise ValueError(f"坐标 ({row}, {col}) 不在扫描网格中")


def moves_between_scan_indices(
    from_index: int,
    to_index: int,
    total_drives: int,
    cols: int = COLS,
) -> list[str]:
    """Direct grid navigation for blind marking only (D/U first, then R/L)."""
    if from_index == to_index:
        return []
    if not 1 <= from_index <= total_drives or not 1 <= to_index <= total_drives:
        raise ValueError(f"scan_index 超出范围 1-{total_drives}")
    order = generate_scan_order(total_drives, cols)
    row, col = order[from_index - 1]
    target_row, target_col = order[to_index - 1]
    moves: list[str] = []

    if to_index > from_index:
        while row < target_row:
            moves.append("D")
            row += 1
        while col < target_col:
            moves.append("R")
            col += 1
        while col > target_col:
            moves.append("L")
            col -= 1
        return moves

    while row > target_row:
        moves.append("U")
        row -= 1
    while col > target_col:
        moves.append("L")
        col -= 1
    while col < target_col:
        moves.append("R")
        col += 1
    return moves


def index_after_moves(
    from_index: int,
    moves: list[str],
    total_drives: int,
    cols: int = COLS,
) -> int:
    row, col = index_to_position(from_index, total_drives, cols)
    for move in moves:
        dr, dc = MOVE_DELTAS[move]
        row += dr
        col += dc
    return position_to_index(row, col, total_drives, cols)
