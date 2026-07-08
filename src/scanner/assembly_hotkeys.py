# 自动装配运行时的全局停止热键。
"""Global hotkey listener to abort assembly automation immediately."""

from __future__ import annotations

import ctypes
import re
import sys
import threading
import time
from typing import Callable

from src.utils.logger import logger

_listener_lock = threading.Lock()
_listener_state: dict | None = None


def _hotkey_to_vk(hotkey: str) -> int | None:
    text = str(hotkey or "").strip().upper()
    match = re.fullmatch(r"F(\d{1,2})", text)
    if match:
        num = int(match.group(1))
        if 1 <= num <= 24:
            return 0x70 + num - 1
    if len(text) == 1 and ("A" <= text <= "Z" or "0" <= text <= "9"):
        return ord(text)
    return None


def _win_hotkey_loop(stop_key: str, on_stop: Callable[[], None]) -> bool:
    if sys.platform != "win32":
        return False

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    WM_HOTKEY = 0x0312
    PM_REMOVE = 0x0001
    MOD_NOREPEAT = 0x4000

    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    class MSG(ctypes.Structure):
        _fields_ = [
            ("hwnd", ctypes.c_void_p),
            ("message", ctypes.c_uint),
            ("wParam", ctypes.c_size_t),
            ("lParam", ctypes.c_size_t),
            ("time", ctypes.c_uint),
            ("pt", POINT),
        ]

    vk = _hotkey_to_vk(stop_key)
    if not vk:
        return False
    if not user32.RegisterHotKey(None, 1, MOD_NOREPEAT, vk):
        return False

    thread_id = kernel32.GetCurrentThreadId()
    with _listener_lock:
        if _listener_state is not None:
            _listener_state["thread_id"] = thread_id

    msg = MSG()
    try:
        while True:
            with _listener_lock:
                active = bool(_listener_state and _listener_state.get("active"))
            if not active:
                break
            while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
                if msg.message == WM_HOTKEY:
                    logger.warning(f"检测到装配停止热键 {stop_key!r}，正在中止...")
                    on_stop()
            time.sleep(0.05)
    finally:
        user32.UnregisterHotKey(None, 1)
    return True


def _keyboard_poll_loop(stop_key: str, on_stop: Callable[[], None]) -> None:
    try:
        import keyboard as kb
    except Exception as exc:
        logger.warning(f"装配停止热键不可用（keyboard 库）: {exc}")
        return

    pressed = False
    while True:
        with _listener_lock:
            active = bool(_listener_state and _listener_state.get("active"))
        if not active:
            break
        try:
            is_pressed = kb.is_pressed(stop_key.lower())
            if is_pressed and not pressed:
                logger.warning(f"检测到装配停止热键 {stop_key!r}，正在中止...")
                on_stop()
                time.sleep(0.35)
            pressed = is_pressed
        except Exception:
            pass
        time.sleep(0.05)


def _hotkey_poll_loop(stop_key: str, on_stop: Callable[[], None]) -> None:
    if not _win_hotkey_loop(stop_key, on_stop):
        _keyboard_poll_loop(stop_key, on_stop)


def start_assembly_stop_hotkey(stop_key: str, on_stop: Callable[[], None]) -> None:
    """Start listening for the configured stop hotkey."""
    global _listener_state
    stop_assembly_stop_hotkey()
    with _listener_lock:
        _listener_state = {"active": True, "thread_id": None}
    thread = threading.Thread(
        target=_hotkey_poll_loop,
        args=(stop_key, on_stop),
        daemon=True,
        name="assembly-stop-hotkey",
    )
    with _listener_lock:
        if _listener_state is not None:
            _listener_state["thread"] = thread
    thread.start()
    logger.info(f"装配停止热键已启用: {stop_key}")


def stop_assembly_stop_hotkey() -> None:
    """Stop the assembly hotkey listener thread."""
    global _listener_state
    with _listener_lock:
        state = _listener_state
        if state is None:
            return
        state["active"] = False
        thread_id = state.get("thread_id")
    if sys.platform == "win32" and thread_id:
        try:
            ctypes.windll.user32.PostThreadMessageW(int(thread_id), 0, 0, 0)
        except Exception:
            pass
    thread = state.get("thread") if state else None
    if thread is not None and thread.is_alive():
        thread.join(timeout=1.0)
    with _listener_lock:
        _listener_state = None
