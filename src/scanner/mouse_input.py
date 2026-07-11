# 通过 SendInput 在 Windows 上模拟鼠标绝对移动与点击。
"""Windows SendInput helpers for absolute mouse clicks."""

from __future__ import annotations

import sys
import time

from src.utils.logger import logger

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000


def _windows_ready() -> bool:
    return sys.platform.startswith("win") and hasattr(__import__("ctypes"), "windll")


def click_screen_absolute(
    screen_x: int,
    screen_y: int,
    *,
    move_settle_seconds: float = 0.05,
    click_hold_seconds: float = 0.02,
) -> None:
    """Move to an absolute screen pixel and left-click once."""
    if not _windows_ready():
        raise RuntimeError("鼠标点击仅支持 Windows（SendInput）。")

    import ctypes
    import ctypes.wintypes

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", ctypes.c_long),
            ("dy", ctypes.c_long),
            ("mouseData", ctypes.wintypes.DWORD),
            ("dwFlags", ctypes.wintypes.DWORD),
            ("time", ctypes.wintypes.DWORD),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class INPUT(ctypes.Structure):
        _fields_ = [
            ("type", ctypes.wintypes.DWORD),
            ("mi", MOUSEINPUT),
        ]

    user32 = ctypes.windll.user32
    screen_w = max(1, int(user32.GetSystemMetrics(0)))
    screen_h = max(1, int(user32.GetSystemMetrics(1)))
    ax = int(int(screen_x) * 65535 / screen_w)
    ay = int(int(screen_y) * 65535 / screen_h)

    def _send(flags: int, dx: int = 0, dy: int = 0) -> None:
        mi = MOUSEINPUT(dx, dy, 0, flags, 0, None)
        inp = INPUT(INPUT_MOUSE, mi)
        user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

    _send(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, ax, ay)
    if move_settle_seconds > 0:
        time.sleep(move_settle_seconds)
    _send(MOUSEEVENTF_LEFTDOWN)
    if click_hold_seconds > 0:
        time.sleep(click_hold_seconds)
    _send(MOUSEEVENTF_LEFTUP)
    logger.debug(f"鼠标点击屏幕坐标: ({screen_x}, {screen_y})")
