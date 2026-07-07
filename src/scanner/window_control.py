# 按窗口标题查找并激活游戏前台窗口。
"""Win32 window discovery and foreground activation for game automation."""

from __future__ import annotations

import ctypes
import ctypes.wintypes
from dataclasses import dataclass

from src.utils.logger import logger


SW_RESTORE = 9


class WindowControlError(RuntimeError):
    """Raised when window operations are unavailable or fail."""


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str


def _require_windows() -> None:
    if not hasattr(ctypes, "windll"):
        raise WindowControlError("窗口控制仅支持 Windows 平台。")


def list_visible_windows() -> list[WindowInfo]:
    """Enumerate visible top-level windows and log their titles."""
    _require_windows()
    user32 = ctypes.windll.user32
    windows: list[WindowInfo] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
    def callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        title = buffer.value.strip()
        if not title:
            return True
        windows.append(WindowInfo(hwnd=int(hwnd), title=title))
        return True

    user32.EnumWindows(callback, 0)
    logger.info("====== 可见窗口标题列表（用于确认游戏窗口名称）======")
    for index, info in enumerate(windows, 1):
        logger.info(f"[窗口 {index:03d}] hwnd={info.hwnd} title={info.title!r}")
    logger.info(f"====== 共 {len(windows)} 个可见窗口 ======")
    return windows


def find_window(title_contains: str | list[str] | None = None) -> WindowInfo | None:
    """Return the first visible window whose title contains any candidate substring."""
    candidates = title_contains if isinstance(title_contains, list) else [title_contains or "异环"]
    candidates = [str(item).strip() for item in candidates if str(item).strip()]
    if not candidates:
        candidates = ["异环"]

    windows = list_visible_windows()
    for info in windows:
        for candidate in candidates:
            if candidate in info.title:
                logger.info(
                    f"已匹配游戏窗口: hwnd={info.hwnd}, title={info.title!r}, "
                    f"matched_by={candidate!r}"
                )
                return info

    logger.warning(f"未找到标题包含 {candidates!r} 的窗口。")
    return None


def activate_window(hwnd: int) -> bool:
    """Restore and bring the target window to the foreground."""
    _require_windows()
    user32 = ctypes.windll.user32
    target = int(hwnd)
    if not target or not user32.IsWindow(ctypes.wintypes.HWND(target)):
        logger.error(f"无效窗口句柄: {target}")
        return False

    if user32.IsIconic(ctypes.wintypes.HWND(target)):
        user32.ShowWindow(ctypes.wintypes.HWND(target), SW_RESTORE)

    user32.SetForegroundWindow(ctypes.wintypes.HWND(target))
    active = int(user32.GetForegroundWindow())
    success = active == target
    if success:
        logger.success(f"已将窗口切换到前台: hwnd={target}")
    else:
        logger.warning(f"窗口激活可能失败: target={target}, foreground={active}")
    return success
