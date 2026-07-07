# 按窗口标题查找并激活游戏前台窗口。
"""Win32 window discovery and foreground activation for game automation."""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import time
from dataclasses import dataclass

from src.utils.logger import logger


SW_RESTORE = 9
SW_SHOW = 5
VK_MENU = 0x12
KEYEVENTF_KEYUP = 0x0002
ASFW_ANY = ctypes.c_uint32(-1).value


class WindowControlError(RuntimeError):
    """Raised when window operations are unavailable or fail."""


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str


def _require_windows() -> None:
    if not hasattr(ctypes, "windll"):
        raise WindowControlError("窗口控制仅支持 Windows 平台。")


def _hwnd(value: int):
    return ctypes.wintypes.HWND(int(value))


def _user32():
    return ctypes.windll.user32


def _kernel32():
    return ctypes.windll.kernel32


def grant_foreground_permission() -> None:
    """Allow subsequent SetForegroundWindow calls while this process still has focus rights."""
    _require_windows()
    user32 = _user32()
    process_id = _kernel32().GetCurrentProcessId()
    user32.AllowSetForegroundWindow(process_id)
    user32.AllowSetForegroundWindow(ASFW_ANY)
    logger.info("已请求前台切换权限: process_id=%s", process_id)


def list_visible_windows() -> list[WindowInfo]:
    """Enumerate visible top-level windows and log their titles."""
    _require_windows()
    user32 = _user32()
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


def find_game_window(title: str = "异环") -> WindowInfo | None:
    """Find the game window by exact title without enumerating all visible windows."""
    _require_windows()
    user32 = _user32()
    exact_title = str(title or "异环").strip() or "异环"
    hwnd = int(user32.FindWindowW(None, exact_title) or 0)
    if hwnd and user32.IsWindowVisible(_hwnd(hwnd)):
        logger.info("已找到游戏窗口: hwnd=%s, title=%r", hwnd, exact_title)
        return WindowInfo(hwnd=hwnd, title=exact_title)

    logger.warning("未找到标题为 %r 的游戏窗口。", exact_title)
    return None


def activate_game_window(title: str = "异环") -> WindowInfo:
    """Find and activate the game window by exact title."""
    window_info = find_game_window(title)
    if window_info is None:
        raise WindowControlError(f"未找到游戏窗口: {title!r}")
    grant_foreground_permission()
    if not activate_window(window_info.hwnd):
        raise WindowControlError(f"无法将游戏窗口切换到前台: {window_info.title!r}")
    return window_info


def find_window(title_contains: str | list[str] | None = None) -> WindowInfo | None:
    """Backward-compatible helper that resolves to an exact game window title."""
    if isinstance(title_contains, list):
        for candidate in title_contains:
            matched = find_game_window(str(candidate).strip())
            if matched is not None:
                return matched
        return None
    return find_game_window(str(title_contains or "异环").strip() or "异环")


def _window_thread_id(hwnd: int) -> int:
    thread_id = ctypes.wintypes.DWORD()
    _user32().GetWindowThreadProcessId(_hwnd(hwnd), ctypes.byref(thread_id))
    return int(thread_id.value)


def _attach_thread_input(from_thread: int, to_thread: int, attach: bool) -> None:
    if from_thread and to_thread and from_thread != to_thread:
        _user32().AttachThreadInput(from_thread, to_thread, attach)


def _simulate_alt_key_press() -> None:
    user32 = _user32()
    user32.keybd_event(VK_MENU, 0, 0, 0)
    user32.keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0)


def _switch_to_this_window(hwnd: int) -> None:
    try:
        switch = _user32().SwitchToThisWindow
        switch.argtypes = [ctypes.wintypes.HWND, ctypes.wintypes.BOOL]
        switch.restype = None
        switch(_hwnd(hwnd), True)
    except Exception as exc:
        logger.debug("SwitchToThisWindow 不可用: %s", exc)


def _activate_window_once(hwnd: int) -> None:
    user32 = _user32()
    kernel32 = _kernel32()
    target = int(hwnd)

    foreground_hwnd = int(user32.GetForegroundWindow() or 0)
    foreground_thread = _window_thread_id(foreground_hwnd) if foreground_hwnd else 0
    target_thread = _window_thread_id(target)
    current_thread = int(kernel32.GetCurrentThreadId())

    attached_pairs: list[tuple[int, int]] = []
    for from_thread, to_thread in (
        (foreground_thread, target_thread),
        (current_thread, target_thread),
    ):
        if from_thread and to_thread and from_thread != to_thread:
            _attach_thread_input(from_thread, to_thread, True)
            attached_pairs.append((from_thread, to_thread))

    try:
        if user32.IsIconic(_hwnd(target)):
            user32.ShowWindow(_hwnd(target), SW_RESTORE)
        else:
            user32.ShowWindow(_hwnd(target), SW_SHOW)

        _simulate_alt_key_press()
        user32.BringWindowToTop(_hwnd(target))
        user32.SetForegroundWindow(_hwnd(target))
        _switch_to_this_window(target)
    finally:
        for from_thread, to_thread in reversed(attached_pairs):
            _attach_thread_input(from_thread, to_thread, False)


def is_window_foreground(hwnd: int) -> bool:
    _require_windows()
    return int(_user32().GetForegroundWindow() or 0) == int(hwnd)


def activate_window(hwnd: int, retries: int = 3, retry_delay_seconds: float = 0.25) -> bool:
    """Restore and bring the target window to the foreground."""
    _require_windows()
    target = int(hwnd)
    user32 = _user32()
    if not target or not user32.IsWindow(_hwnd(target)):
        logger.error(f"无效窗口句柄: {target}")
        return False

    attempts = max(1, int(retries))
    for attempt in range(1, attempts + 1):
        logger.info("尝试激活窗口 (%s/%s): hwnd=%s", attempt, attempts, target)
        _activate_window_once(target)
        time.sleep(max(0.0, float(retry_delay_seconds)))

        active = int(user32.GetForegroundWindow() or 0)
        if active == target:
            logger.success(f"已将窗口切换到前台: hwnd={target}")
            return True

        logger.warning(
            "窗口激活未确认: target=%s, foreground=%s, attempt=%s/%s",
            target,
            active,
            attempt,
            attempts,
        )

    return is_window_foreground(target)
