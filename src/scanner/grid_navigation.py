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
