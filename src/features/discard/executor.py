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
from src.scanner.batch_processor import BatchProcessor
from src.scanner.gamepad_controller import GamepadScanner, ViGEmDriverNotReadyError
from src.scanner.grid_navigation import moves_between_scan_indices
from src.scanner.window_capture import capture_foreground_window, crop_window_border_from_image
from src.utils.image_io import imread_unicode
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

    def _verify_current_drive(self, image_bgr: np.ndarray, entry) -> None:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            cv2.imwrite(tmp_path, image_bgr)
            item = self._processor._process_single_image(tmp_path)
            actual = signature_from_item_dict(item.model_dump())
            if entry.signature != actual:
                logger.warning(
                    f"第 {entry.scan_index} 格签名校验失败 | 期望={entry.signature} | 实际={actual}"
                )
            verify_signature(entry.signature, actual)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def _verify_entry_at_current_cell(self, image_bgr: np.ndarray, entry) -> None:
        try:
            self._verify_current_drive(image_bgr, entry)
        except InventoryChangedError as exc:
            raise InventoryChangedError(
                f"第 {entry.scan_index} 格装备与快照不一致（{exc}）。"
                "请确认已选中第一排第一个格子，且背包未手动变动。"
            ) from exc

    def _navigate_between(self, from_index: int, to_index: int) -> None:
        assert self.scanner is not None
        moves = moves_between_scan_indices(
            from_index,
            to_index,
            self.session.total_drives,
            self.session.cols,
        )
        self.scanner._apply_moves(moves, pace="marking")
        time.sleep(0.2)

    def _matches_first_cell(self, sct, entry) -> bool:
        if entry is None:
            return False
        try:
            self._verify_current_drive(self._capture_bgr(sct), entry)
            return True
        except InventoryChangedError:
            return False

    def _prepare_at_first_cell(self, sct) -> None:
        assert self.scanner is not None
        entry = self.session.entry_by_index(1)
        self.scanner.anchor_to_first_cell(
            self.session.total_drives,
            self.session.cols,
            is_at_first_cell=lambda: self._matches_first_cell(sct, entry),
        )
        if entry is None:
            return
        image_bgr = self._capture_bgr(sct)
        try:
            self._verify_entry_at_current_cell(image_bgr, entry)
        except InventoryChangedError as exc:
            raise InventoryChangedError(
                f"自动归位后第 1 格校验失败（{exc}）。"
                "请确认在仓库页面且背包与扫描会话一致。"
            ) from exc

    def _already_marked(self, image_bgr: np.ndarray, action: str) -> bool:
        states = self.detector.detect_from_bgr(image_bgr)
        marked = bool(states.get(action))
        logger.debug(
            f"标记状态检测 action={action} marked={marked} "
            f"discard={float(states.get('discard_marked_score', 0.0)):.3f}/"
            f"{float(states.get('discard_unmarked_score', 0.0)):.3f} "
            f"lock={float(states.get('lock_marked_score', 0.0)):.3f}/"
            f"{float(states.get('lock_unmarked_score', 0.0)):.3f}"
        )
        return marked

    def execute_targets(
        self,
        targets: list[MarkTarget],
        *,
        on_progress: Callable[[int, int, MarkStepResult], None] | None = None,
    ) -> list[MarkStepResult]:
        self._connect()
        assert self.scanner is not None
        self._stopped = False
        self.scanner.wait_for_handoff(for_marking=True)
        results: list[MarkStepResult] = []
        total = len(targets)
        current_index = 1
        with mss.mss() as sct:
            self._prepare_at_first_cell(sct)
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
                self._navigate_between(current_index, target.scan_index)
                current_index = target.scan_index
                time.sleep(0.35)
                image_bgr = self._capture_bgr(sct)
                try:
                    self._verify_entry_at_current_cell(image_bgr, entry)
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
        on_progress: Callable[[int, int, MarkStepResult], None] | None = None,
    ) -> list[MarkStepResult]:
        self._connect()
        assert self.scanner is not None
        self._stopped = False
        self.scanner.wait_for_handoff(for_marking=True)
        results: list[MarkStepResult] = []
        total = len(entries)
        current_index = 1
        with mss.mss() as sct:
            self._prepare_at_first_cell(sct)
            for idx, entry in enumerate(entries, 1):
                if self._stopped:
                    break
                session_entry = self.session.entry_by_index(entry.scan_index)
                if session_entry is None:
                    continue
                self._navigate_between(current_index, entry.scan_index)
                current_index = entry.scan_index
                time.sleep(0.35)
                image_bgr = self._capture_bgr(sct)
                self._verify_entry_at_current_cell(image_bgr, session_entry)
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
