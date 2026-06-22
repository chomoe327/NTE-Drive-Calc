# 全量扫描背包格子的 S 形路径生成。
"""Grid navigation helpers for full-inventory gamepad traversal."""

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


def moves_between_scan_indices(
    from_index: int,
    to_index: int,
    total_drives: int,
    cols: int = COLS,
) -> list[str]:
    """Direct grid moves between two slots (horizontal first, then vertical)."""
    if from_index == to_index:
        return []
    if not 1 <= from_index <= total_drives or not 1 <= to_index <= total_drives:
        raise ValueError(f"scan_index 超出范围 1-{total_drives}")
    order = generate_scan_order(total_drives, cols)
    from_row, from_col = order[from_index - 1]
    to_row, to_col = order[to_index - 1]
    moves: list[str] = []
    row, col = from_row, from_col
    if to_index > from_index:
        while col < to_col:
            moves.append("R")
            col += 1
        while col > to_col:
            moves.append("L")
            col -= 1
        while row < to_row:
            moves.append("D")
            row += 1
        return moves
    while row > to_row:
        moves.append("U")
        row -= 1
    while col > to_col:
        moves.append("L")
        col -= 1
    while col < to_col:
        moves.append("R")
        col += 1
    return moves


def moves_between_scan_indices_along_scan_path(
    from_index: int,
    to_index: int,
    total_drives: int,
    cols: int = COLS,
) -> list[str]:
    """Step through every intermediate cell along the full-scan S-curve path."""
    if from_index == to_index:
        return []
    if not 1 <= from_index <= total_drives or not 1 <= to_index <= total_drives:
        raise ValueError(f"scan_index 超出范围 1-{total_drives}")
    paths = generate_path_commands(total_drives, cols)
    if to_index > from_index:
        moves: list[str] = []
        for step in range(from_index, to_index):
            moves.extend(paths[step])
        return moves
    moves: list[str] = []
    for step in range(to_index, from_index):
        moves.extend(paths[step])
    return [_INVERT_MOVE[move] for move in reversed(moves)]
