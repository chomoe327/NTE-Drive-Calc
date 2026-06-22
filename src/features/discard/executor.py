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
import mss.tools
import numpy as np

from src.features.discard.mark_state import MarkStateDetector
from src.features.discard.scan_session import InventoryChangedError, ScanSession, signature_from_item_dict, verify_signature
from src.features.discard.scoring import MarkTarget
from src.features.identification.parser import normalized_signature_data
from src.scanner.batch_processor import BatchProcessor
from src.scanner.gamepad_controller import GamepadScanner, ViGEmDriverNotReadyError
from src.scanner.grid_navigation import moves_for_scan_index
from src.scanner.window_capture import capture_foreground_window, crop_window_border_from_image
from src.utils.image_io import imread_unicode
from src.utils.logger import logger

try:
    import vgamepad as vg
except Exception:  # pragma: no cover - optional at import time
    vg = None

BUTTON_MAP = {
    "DPAD_LEFT": "XUSB_BUTTON_DPAD_LEFT",
    "DPAD_RIGHT": "XUSB_BUTTON_DPAD_RIGHT",
    "DPAD_UP": "XUSB_BUTTON_DPAD_UP",
    "DPAD_DOWN": "XUSB_BUTTON_DPAD_DOWN",
    "A": "XUSB_BUTTON_A",
    "B": "XUSB_BUTTON_B",
    "X": "XUSB_BUTTON_X",
    "Y": "XUSB_BUTTON_Y",
}


@dataclass
class MarkStepResult:
    scan_index: int
    action: str
    status: str
    message: str = ""


class MarkingExecutor:
    def __init__(
        self,
        session: ScanSession,
        template_dir: Path,
        config_dir: Path,
        macros_path: Path | None = None,
    ):
        self.session = session
        self.template_dir = Path(template_dir)
        self.config_dir = Path(config_dir)
        self.macros_path = Path(macros_path) if macros_path else None
        self.detector = MarkStateDetector(self.template_dir)
        self._stopped = False
        self.scanner: GamepadScanner | None = None
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
            "discard": [{"button": "DPAD_LEFT", "hold_ms": 80, "after_ms": 200}],
            "lock": [{"button": "DPAD_RIGHT", "hold_ms": 80, "after_ms": 200}],
        }

    def _connect(self) -> None:
        if self.scanner is None:
            self.scanner = GamepadScanner(output_dir=tempfile.mkdtemp(prefix="marking_scan_"))

    def _press_button(self, button_name: str, hold_ms: int = 80) -> None:
        if vg is None or not self.scanner:
            raise RuntimeError("虚拟手柄不可用")
        attr = BUTTON_MAP.get(button_name.upper())
        if not attr or not hasattr(vg.XUSB_BUTTON, attr):
            raise ValueError(f"未知按键: {button_name}")
        button = getattr(vg.XUSB_BUTTON, attr)
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
            return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        return arr[:, :, :3]

    def _verify_current_drive(self, image_bgr: np.ndarray, entry) -> None:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            cv2.imwrite(tmp_path, image_bgr)
            item = self._processor._process_single_image(tmp_path)
            actual = signature_from_item_dict(normalized_signature_data(item.model_dump()))
            verify_signature(entry.signature, actual)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def _reset_to_start(self) -> None:
        assert self.scanner is not None
        self.scanner.push_left_joystick(-1.0, 0.0)
        time.sleep(0.5)

    def _navigate_to_index(self, scan_index: int) -> None:
        assert self.scanner is not None
        moves = moves_for_scan_index(scan_index, self.session.total_drives, self.session.cols)
        self.scanner._apply_moves(moves)

    def _already_marked(self, image_bgr: np.ndarray, action: str) -> bool:
        states = self.detector.detect_from_bgr(image_bgr)
        return bool(states.get(action))

    def execute_targets(
        self,
        targets: list[MarkTarget],
        *,
        switch_delay: float = 3.0,
        on_progress: Callable[[int, int, MarkStepResult], None] | None = None,
    ) -> list[MarkStepResult]:
        self._connect()
        assert self.scanner is not None
        self._stopped = False
        logger.warning(f"将在 {switch_delay:.0f} 秒后开始标记，请切回游戏并选中第一格驱动。")
        time.sleep(max(0.0, switch_delay))
        self._reset_to_start()
        results: list[MarkStepResult] = []
        total = len(targets)
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
                self._navigate_to_index(target.scan_index)
                time.sleep(0.35)
                image_bgr = self._capture_bgr(sct)
                try:
                    self._verify_current_drive(image_bgr, entry)
                except InventoryChangedError as exc:
                    result = MarkStepResult(target.scan_index, target.action, "aborted", str(exc))
                    results.append(result)
                    if on_progress:
                        on_progress(idx, total, result)
                    raise
                if self._already_marked(image_bgr, target.action):
                    result = MarkStepResult(target.scan_index, target.action, "skipped_already_marked")
                    results.append(result)
                    if on_progress:
                        on_progress(idx, total, result)
                    continue
                self._run_macro(target.action)
                time.sleep(0.25)
                result = MarkStepResult(target.scan_index, target.action, "marked")
                results.append(result)
                if on_progress:
                    on_progress(idx, total, result)
        return results

    def rollback_entries(
        self,
        entries: list,
        *,
        switch_delay: float = 3.0,
        on_progress: Callable[[int, int, MarkStepResult], None] | None = None,
    ) -> list[MarkStepResult]:
        self._connect()
        assert self.scanner is not None
        self._stopped = False
        logger.warning(f"将在 {switch_delay:.0f} 秒后开始回滚标记，请切回游戏。")
        time.sleep(max(0.0, switch_delay))
        self._reset_to_start()
        results: list[MarkStepResult] = []
        total = len(entries)
        with mss.mss() as sct:
            for idx, entry in enumerate(entries, 1):
                if self._stopped:
                    break
                session_entry = self.session.entry_by_index(entry.scan_index)
                if session_entry is None:
                    continue
                self._navigate_to_index(entry.scan_index)
                time.sleep(0.35)
                image_bgr = self._capture_bgr(sct)
                self._verify_current_drive(image_bgr, session_entry)
                if not self._already_marked(image_bgr, entry.action):
                    result = MarkStepResult(entry.scan_index, entry.action, "skipped_not_marked")
                    results.append(result)
                    if on_progress:
                        on_progress(idx, total, result)
                    continue
                self._run_macro(entry.action)
                time.sleep(0.25)
                result = MarkStepResult(entry.scan_index, entry.action, "unmarked")
                results.append(result)
                if on_progress:
                    on_progress(idx, total, result)
        return results
