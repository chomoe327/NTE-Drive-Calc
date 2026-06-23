# 虚拟手柄执行背包标记遍历。
"""Gamepad-driven marking executor for discard/lock toggles."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import mss
import numpy as np

from src.features.discard.mark_state import MarkStateDetector
from src.features.discard.scan_session import InventoryChangedError, ScanSession, signature_from_item_dict, verify_signature
from src.features.discard.scoring import MarkTarget
from src.scanner.batch_processor import BatchProcessor
from src.scanner.gamepad_controller import GamepadScanner
from src.scanner.ocr_grid_navigator import OcrGridNavigator
from src.scanner.window_capture import capture_foreground_window, crop_window_border_from_image
from src.utils.logger import logger

try:
    import vgamepad as vg
except Exception:  # pragma: no cover - optional at import time
    vg = None

BUTTON_ALIASES: dict[str, tuple[str, ...]] = {
    "DPAD_LEFT": ("XUSB_GAMEPAD_DPAD_LEFT", "XUSB_BUTTON_DPAD_LEFT"),
    "DPAD_RIGHT": ("XUSB_GAMEPAD_DPAD_RIGHT", "XUSB_BUTTON_DPAD_RIGHT"),
    "DPAD_UP": ("XUSB_GAMEPAD_DPAD_UP", "XUSB_BUTTON_DPAD_UP"),
    "DPAD_DOWN": ("XUSB_GAMEPAD_DPAD_DOWN", "XUSB_BUTTON_DPAD_DOWN"),
    "A": ("XUSB_GAMEPAD_A", "XUSB_BUTTON_A"),
    "B": ("XUSB_GAMEPAD_B", "XUSB_BUTTON_B"),
    "X": ("XUSB_GAMEPAD_X", "XUSB_BUTTON_X"),
    "Y": ("XUSB_GAMEPAD_Y", "XUSB_BUTTON_Y"),
}


def _resolve_xusb_button(button_name: str):
    if vg is None:
        return None
    for attr in BUTTON_ALIASES.get(button_name.upper().strip(), ()):
        if hasattr(vg.XUSB_BUTTON, attr):
            return getattr(vg.XUSB_BUTTON, attr)
    return None


MARKING_PANEL_SETTLE_SEC = 0.45
MARKING_VERIFY_ATTEMPTS = 4
MARKING_VERIFY_RETRY_DELAY_SEC = 0.4


@dataclass
class MarkStepResult:
    scan_index: int
    action: str
    status: str
    message: str = ""
    was_locked_before: bool = False


class MarkingExecutor:
    def __init__(
        self,
        session: ScanSession,
        template_dir: Path,
        config_dir: Path,
        macros_path: Path | None = None,
        *,
        ignore_locked: bool = False,
    ):
        self.session = session
        self.template_dir = Path(template_dir)
        self.config_dir = Path(config_dir)
        self.macros_path = Path(macros_path) if macros_path else None
        self.ignore_locked = ignore_locked
        self.detector = MarkStateDetector(self.template_dir)
        self._stopped = False
        self.scanner: GamepadScanner | None = None
        self.navigator: OcrGridNavigator | None = None
        self._macros = self._load_macros()
        self._processor = BatchProcessor(
            input_dir=tempfile.gettempdir(),
            output_file=os.path.join(tempfile.gettempdir(), "unused_marking_inventory.json"),
            config_dir=str(self.config_dir),
            replace_output=True,
        )

    def emergency_stop(self) -> None:
        self._stopped = True
        if self.scanner:
            self.scanner.emergency_stop()

    def _load_macros(self) -> dict:
        candidates = []
        if self.macros_path:
            candidates.append(self.macros_path)
        candidates.append(self.config_dir / "marking_macros.json")
        bundled = Path(__file__).resolve().parents[3] / "config" / "marking_macros.json"
        candidates.append(bundled)
        for path in candidates:
            if path.exists():
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
        return {
            "discard": [{"button": "DPAD_LEFT", "hold_ms": 80, "after_ms": 250}],
            "lock": [{"button": "DPAD_RIGHT", "hold_ms": 80, "after_ms": 250}],
            "confirm": [{"button": "A", "hold_ms": 80, "after_ms": 300}],
        }

    def _wait_panel_settle(self, extra: float = 0.0) -> None:
        time.sleep(MARKING_PANEL_SETTLE_SEC + max(0.0, extra))

    def _parse_item_dict(self, image_bgr: np.ndarray) -> dict:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            cv2.imwrite(tmp_path, image_bgr)
            item = self._processor._process_single_image(tmp_path)
            return item.model_dump()
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def _signature_looks_unreadable(self, item_dict: dict) -> bool:
        if item_dict.get("item_type") == "tape":
            return (
                str(item_dict.get("set_name") or "") in ("", "未知套装")
                or str(item_dict.get("main_stats") or "") in ("", "未知主词条")
                or not item_dict.get("sub_stats")
            )
        if item_dict.get("item_type") == "drive":
            return (
                str(item_dict.get("shape_id") or "") in ("", "Unknown")
                or not item_dict.get("main_stats")
            )
        return True

    def _verify_entry_with_retry(self, sct, entry) -> np.ndarray:
        last_error: InventoryChangedError | None = None
        for attempt in range(1, MARKING_VERIFY_ATTEMPTS + 1):
            if attempt > 1:
                time.sleep(MARKING_VERIFY_RETRY_DELAY_SEC)
            image_bgr = self._capture_bgr(sct)
            try:
                item_dict = self._parse_item_dict(image_bgr)
                if self._signature_looks_unreadable(item_dict):
                    raise InventoryChangedError("详情面板尚未加载完成或截图无法识别")
                actual = signature_from_item_dict(item_dict)
                if entry.signature != actual:
                    logger.warning(
                        f"第 {entry.scan_index} 格签名校验失败(尝试 {attempt}/{MARKING_VERIFY_ATTEMPTS}) "
                        f"| 期望={entry.signature} | 实际={actual}"
                    )
                verify_signature(entry.signature, actual)
                return image_bgr
            except InventoryChangedError as exc:
                last_error = exc
                logger.debug(
                    f"第 {entry.scan_index} 格校验重试 {attempt}/{MARKING_VERIFY_ATTEMPTS}: {exc}"
                )
        assert last_error is not None
        raise InventoryChangedError(
            f"第 {entry.scan_index} 格装备与快照不一致（{last_error}）。"
            "请确认背包与扫描会话一致。"
        ) from last_error

    def _detect_mark_states(self, image_bgr: np.ndarray, scan_index: int) -> dict[str, bool | float]:
        states = self.detector.detect_from_bgr(image_bgr)
        logger.info(
            f"第 {scan_index} 格标记状态 "
            f"discard bright={float(states.get('discard_brightness', 0.0)):.1f}/"
            f"{float(states.get('discard_brightness_mid', 0.0)):.1f} "
            f"loc={float(states.get('discard_locate_score', 0.0)):.3f} "
            f"tm={float(states.get('discard_marked_score', 0.0)):.3f}/"
            f"{float(states.get('discard_unmarked_score', 0.0)):.3f} "
            f"lock bright={float(states.get('lock_brightness', 0.0)):.1f}/"
            f"{float(states.get('lock_brightness_mid', 0.0)):.1f} "
            f"loc={float(states.get('lock_locate_score', 0.0)):.3f} "
            f"tm={float(states.get('lock_marked_score', 0.0)):.3f}/"
            f"{float(states.get('lock_unmarked_score', 0.0)):.3f} "
            f"=> discard={bool(states.get('discard'))} lock={bool(states.get('lock'))}"
        )
        return states

    def _ensure_unlocked_for_discard(self, sct, image_bgr: np.ndarray, scan_index: int) -> tuple[np.ndarray, bool]:
        states = self._detect_mark_states(image_bgr, scan_index)
        was_locked = bool(states.get("lock"))
        if not was_locked:
            return image_bgr, False
        logger.info(f"第 {scan_index} 格已上锁，先解锁再弃置")
        self._run_macro("lock")
        self._wait_panel_settle(0.15)
        image_bgr = self._capture_bgr(sct)
        self._detect_mark_states(image_bgr, scan_index)
        return image_bgr, True

    def _execute_mark_action(self, sct, target: MarkTarget, entry) -> MarkStepResult:
        image_bgr = self._verify_entry_with_retry(sct, entry)
        was_locked_before = False
        if target.action == "discard":
            states = self._detect_mark_states(image_bgr, target.scan_index)
            if states.get("discard"):
                return MarkStepResult(target.scan_index, target.action, "skipped_already_marked")
            if self.ignore_locked and states.get("lock"):
                logger.info(f"第 {target.scan_index} 格已上锁，忽略弃置")
                return MarkStepResult(target.scan_index, target.action, "skipped_locked")
            image_bgr, was_locked_before = self._ensure_unlocked_for_discard(sct, image_bgr, target.scan_index)
            states = self._detect_mark_states(image_bgr, target.scan_index)
            if states.get("discard"):
                return MarkStepResult(target.scan_index, target.action, "skipped_already_marked")
            self._run_macro("discard")
        elif target.action == "lock":
            states = self._detect_mark_states(image_bgr, target.scan_index)
            if states.get("lock"):
                return MarkStepResult(target.scan_index, target.action, "skipped_already_marked")
            self._run_macro("lock")
        else:
            return MarkStepResult(target.scan_index, target.action, "error", f"未知动作: {target.action}")
        time.sleep(0.25)
        return MarkStepResult(
            target.scan_index,
            target.action,
            "marked",
            was_locked_before=was_locked_before,
        )

    def _rollback_log_entry(self, sct, log_entry, session_entry) -> MarkStepResult:
        image_bgr = self._verify_entry_with_retry(sct, session_entry)
        restore_lock = bool(getattr(log_entry, "was_locked_before", False))

        if log_entry.action == "discard":
            image_bgr, _ = self._ensure_unlocked_for_discard(sct, image_bgr, log_entry.scan_index)
            states = self._detect_mark_states(image_bgr, log_entry.scan_index)
            if not states.get("discard"):
                return MarkStepResult(log_entry.scan_index, log_entry.action, "skipped_not_marked")
            self._run_macro("discard")
            time.sleep(0.25)
            if restore_lock:
                image_bgr = self._capture_bgr(sct)
                states = self._detect_mark_states(image_bgr, log_entry.scan_index)
                if not states.get("lock"):
                    logger.info(f"第 {log_entry.scan_index} 格回滚弃置后恢复上锁")
                    self._run_macro("lock")
                    time.sleep(0.25)
            return MarkStepResult(
                log_entry.scan_index,
                log_entry.action,
                "unmarked",
                was_locked_before=restore_lock,
            )

        if log_entry.action == "lock":
            states = self._detect_mark_states(image_bgr, log_entry.scan_index)
            if not states.get("lock"):
                return MarkStepResult(log_entry.scan_index, log_entry.action, "skipped_not_marked")
            self._run_macro("lock")
            time.sleep(0.25)
            return MarkStepResult(log_entry.scan_index, log_entry.action, "unmarked")

        return MarkStepResult(
            log_entry.scan_index,
            log_entry.action,
            "error",
            f"未知动作: {log_entry.action}",
        )

    def _connect(self) -> None:
        if self.scanner is None:
            self.scanner = GamepadScanner(output_dir=tempfile.mkdtemp(prefix="marking_scan_"))
        if self.navigator is None:
            assert self.scanner is not None
            self.navigator = OcrGridNavigator(
                self.session,
                self.scanner,
                capture_bgr=lambda sct: self._capture_bgr(sct),
                parse_item_dict=self._parse_item_dict,
                signature_looks_unreadable=self._signature_looks_unreadable,
                is_stopped=lambda: self._stopped,
            )

    def _press_button(self, button_name: str, hold_ms: int = 80) -> None:
        if vg is None or not self.scanner:
            raise RuntimeError("虚拟手柄不可用")
        button = _resolve_xusb_button(button_name)
        if button is None:
            raise ValueError(f"未知按键: {button_name}")
        self.scanner.gamepad.press_button(button)
        self.scanner.gamepad.update()
        time.sleep(max(0.01, hold_ms / 1000.0))
        self.scanner.gamepad.release_button(button)
        self.scanner.gamepad.update()

    def _run_macro(self, action: str) -> None:
        steps = self._macros.get(action, [])
        for step in steps:
            if self._stopped:
                return
            self._press_button(str(step.get("button", "")), int(step.get("hold_ms", 80) or 80))
            time.sleep(max(0.0, int(step.get("after_ms", 200) or 200) / 1000.0))

    def _capture_bgr(self, sct) -> np.ndarray:
        screenshot, _ = capture_foreground_window(sct)
        arr = np.array(screenshot)
        if arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        else:
            arr = arr[:, :, :3]
        return crop_window_border_from_image(arr)

    def execute_targets(
        self,
        targets: list[MarkTarget],
        *,
        on_progress: Callable[[int, int, MarkStepResult], None] | None = None,
    ) -> list[MarkStepResult]:
        self._connect()
        assert self.scanner is not None and self.navigator is not None
        self._stopped = False
        self.scanner.wait_for_handoff()
        results: list[MarkStepResult] = []
        total = len(targets)
        current_index: int | None = None
        with mss.mss() as sct:
            for idx, target in enumerate(targets, 1):
                if self._stopped:
                    break
                entry = self.session.entry_by_index(target.scan_index)
                if entry is None:
                    result = MarkStepResult(target.scan_index, target.action, "error", "缺少快照条目")
                    results.append(result)
                    if on_progress:
                        on_progress(idx, total, result)
                    continue
                self.navigator.navigate_to(sct, target.scan_index, start_index=current_index)
                current_index = target.scan_index
                try:
                    result = self._execute_mark_action(sct, target, entry)
                except InventoryChangedError as exc:
                    result = MarkStepResult(target.scan_index, target.action, "aborted", str(exc))
                    results.append(result)
                    if on_progress:
                        on_progress(idx, total, result)
                    raise
                results.append(result)
                if on_progress:
                    on_progress(idx, total, result)
        return results

    def rollback_entries(
        self,
        entries: list,
        *,
        on_progress: Callable[[int, int, MarkStepResult], None] | None = None,
    ) -> list[MarkStepResult]:
        self._connect()
        assert self.scanner is not None and self.navigator is not None
        self._stopped = False
        self.scanner.wait_for_handoff()
        results: list[MarkStepResult] = []
        total = len(entries)
        current_index: int | None = None
        with mss.mss() as sct:
            for idx, entry in enumerate(entries, 1):
                if self._stopped:
                    break
                session_entry = self.session.entry_by_index(entry.scan_index)
                if session_entry is None:
                    continue
                self.navigator.navigate_to(sct, entry.scan_index, start_index=current_index)
                current_index = entry.scan_index
                try:
                    result = self._rollback_log_entry(sct, entry, session_entry)
                except InventoryChangedError as exc:
                    result = MarkStepResult(entry.scan_index, entry.action, "aborted", str(exc))
                    results.append(result)
                    if on_progress:
                        on_progress(idx, total, result)
                    raise
                results.append(result)
                if on_progress:
                    on_progress(idx, total, result)
        return results
