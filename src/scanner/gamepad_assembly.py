# 使用虚拟手柄执行驱动块拖动装配测试。
"""Virtual gamepad drag-and-drop helpers for assembly automation tests."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mss
import mss.tools
import numpy as np

from src.app import runtime
from src.features.inventory_import.equipment_classifier import (
    build_inventory_grid_layout,
    load_inventory_selection_triangle,
    locate_selected_inventory_shape,
)
from src.scanner.assembly_dialog import detect_equip_transfer_dialog
from src.scanner.assembly_vision import (
    assembly_board_region,
    compute_position_error,
    correction_stick_for_error,
    detect_dragged_piece_anchor,
    estimate_position_error_from_origin,
    expand_board_region,
    grid_dimensions,
    is_detection_outlier,
    needs_position_correction,
    position_error_magnitude,
)
from src.scanner.config import ScannerConfig
from src.scanner.gamepad_controller import ViGEmDriverNotReadyError, _format_vigem_error
from src.scanner.shape_recognizer import GoldShapeRecognizer
from src.scanner.window_capture import capture_foreground_window, get_foreground_client_rect
from src.utils.logger import logger


@dataclass(frozen=True)
class StickMove:
    stick_x: float
    stick_y: float
    duration_seconds: float
    label: str = "move"


@dataclass(frozen=True)
class AssemblyTestResult:
    role_name: str
    piece_id: str
    start_r: int
    start_c: int
    window_title: str
    before_screenshot: str
    after_screenshot: str
    applied_moves: list[StickMove]
    debug_screenshots: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AssemblyPieceResult:
    piece_id: str
    start_r: int
    start_c: int
    applied_moves: list[StickMove]
    success: bool
    error: str | None = None


@dataclass(frozen=True)
class FullAssemblyResult:
    blueprint_role: str
    window_title: str
    before_screenshot: str
    after_screenshot: str
    pieces: list[AssemblyPieceResult]
    debug_screenshots: list[str] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return sum(1 for piece in self.pieces if piece.success)

    @property
    def failed_pieces(self) -> list[AssemblyPieceResult]:
        return [piece for piece in self.pieces if not piece.success]


def load_assembly_calibration(config_path: Path | None = None) -> dict:
    path = config_path or runtime.CONFIG_DIR / "assembly_calibration.json"
    bundled = runtime.BUNDLED_CONFIG_DIR / "assembly_calibration.json"
    for candidate in (path, bundled):
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as handle:
                return json.load(handle)
    raise FileNotFoundError("找不到 assembly_calibration.json 配置文件。")


def assembly_test_dir() -> Path:
    output_dir = runtime.ACCOUNT_DATA_ROOT / "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


class InventoryDriveNotFoundError(RuntimeError):
    """Raised when shape-based inventory search cannot find the target drive."""


class AssemblyStoppedError(RuntimeError):
    """Raised when the user aborts assembly via the stop hotkey."""


_active_assembly_controller: "GamepadAssemblyController | None" = None


def set_active_assembly_controller(controller: "GamepadAssemblyController | None") -> None:
    global _active_assembly_controller
    _active_assembly_controller = controller


def request_assembly_stop() -> None:
    controller = _active_assembly_controller
    if controller is not None:
        controller.stop()


class GamepadAssemblyController:
    """Drive-block drag controller built on top of ViGEm virtual gamepad."""

    def __init__(self, calibration: dict | None = None):
        self.calibration = calibration or load_assembly_calibration()
        self._stick_interval = float(
            self.calibration.get("stick_update_interval_seconds", 0.05) or 0.05
        )
        self._debug_cfg = self.calibration.get("debug_capture", {}) or {}
        self._debug_enabled = bool(self._debug_cfg.get("enabled", True))
        self._debug_screenshots: list[str] = []
        self._debug_counter = 0
        self._shape_recognizer: GoldShapeRecognizer | None = None
        self._selection_triangle_template = None
        self._ocr_engine = None
        logger.info("正在连接虚拟 Xbox 360 手柄（装配测试）...")
        try:
            import vgamepad as vg

            self.gamepad = vg.VX360Gamepad()
            self._buttons = vg.XUSB_BUTTON
        except Exception as exc:
            text = str(exc).upper()
            if "VIGEM" in text or "BUS_NOT_FOUND" in text or "VI_GEM" in text:
                raise ViGEmDriverNotReadyError(_format_vigem_error(exc)) from exc
            raise
        time.sleep(2)
        logger.success("虚拟手柄连接完成（装配测试）")
        self._drag_active = False
        self._current_stick = (0.0, 0.0)
        self._stopped = False

    def stop(self) -> None:
        self._stopped = True

    def _check_stopped(self) -> None:
        if self._stopped:
            raise AssemblyStoppedError("用户已中止自动装配。")

    def _abort_drag_hold(self) -> None:
        if not self._drag_active:
            return
        logger.warning("中止拖动：松开 A 并复位摇杆")
        self.gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
        self.gamepad.release_button(button=self._buttons.XUSB_GAMEPAD_A)
        self.gamepad.update()
        self._drag_active = False
        self._current_stick = (0.0, 0.0)

    def _debug_capture_enabled(self, stage: str) -> bool:
        if not self._debug_enabled:
            return False
        return bool(self._debug_cfg.get(stage, True))

    def capture_screenshot(self, filename: str) -> str:
        output_dir = assembly_test_dir()
        output_path = str(output_dir / filename)
        with mss.MSS() as sct:
            screenshot, _ = capture_foreground_window(sct)
        mss.tools.to_png(screenshot.rgb, screenshot.size, output=output_path)
        logger.info(f"装配测试截图已保存: {output_path}")
        return output_path

    def _capture_debug(self, stage: str) -> None:
        if not self._debug_capture_enabled(stage):
            return
        self._debug_counter += 1
        filename = f"assembly_test_{self._debug_counter:02d}_{stage}.png"
        path = self.capture_screenshot(filename)
        self._debug_screenshots.append(path)

    def _reset_stick(self) -> None:
        self.gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
        self.gamepad.update()

    def _inventory_nav_cfg(self) -> dict:
        return self.calibration.get("inventory_nav", {}) or {}

    def _inventory_search_cfg(self) -> dict:
        return self.calibration.get("inventory_search", {}) or {}

    def _inventory_grid_cfg(self) -> dict:
        return self.calibration.get("inventory_grid", {}) or {}

    def _get_selection_triangle_template(self):
        if self._selection_triangle_template is None:
            search_cfg = self._inventory_search_cfg()
            template_name = str(
                search_cfg.get("selection_triangle_template", "inventory_selection_triangle.png")
                or "inventory_selection_triangle.png"
            )
            template_dir = runtime.TEMPLATE_DIR or runtime.CONFIG_DIR / "templates"
            template_path = Path(template_dir) / template_name
            self._selection_triangle_template = load_inventory_selection_triangle(template_path)
        return self._selection_triangle_template

    def _inventory_content_rect(self, image: np.ndarray) -> tuple[int, int, int, int]:
        image_h, image_w = image.shape[:2]
        return ScannerConfig.get_content_rect(image_w, image_h)

    def _inventory_grid_layout(self, image: np.ndarray) -> dict:
        image_h, image_w = image.shape[:2]
        base_width = int(self.calibration.get("base_width", 2560) or 2560)
        base_height = int(self.calibration.get("base_height", 1440) or 1440)
        return build_inventory_grid_layout(
            image_w,
            image_h,
            self._inventory_grid_cfg(),
            base_width=base_width,
            base_height=base_height,
            content_rect=self._inventory_content_rect(image),
        )

    def _wait_after_inventory_move(self) -> None:
        search_cfg = self._inventory_search_cfg()
        delay = float(search_cfg.get("post_move_delay_seconds", 0.40) or 0.40)
        if delay > 0:
            time.sleep(delay)

    def _format_selected_slot_log(self, recognition: dict) -> str:
        slot_index = recognition.get("slot_index")
        slot_row = recognition.get("slot_row")
        slot_col = recognition.get("slot_col")
        if slot_index is None or slot_row is None or slot_col is None:
            return "当前未能定位选中格子"
        return f"当前选中第 {slot_index} 格 (行{slot_row}, 列{slot_col})"

    def _tap_left_stick(
        self,
        stick_x: float,
        stick_y: float,
        *,
        vertical: bool = False,
    ) -> None:
        nav = self._inventory_nav_cfg()
        if vertical:
            tap_seconds = float(nav.get("row_tap_seconds", 0.15) or 0.15)
            settle_seconds = float(nav.get("row_settle_seconds", 0.30) or 0.30)
        else:
            tap_seconds = float(nav.get("tap_seconds", 0.10) or 0.10)
            settle_seconds = float(nav.get("settle_seconds", 0.25) or 0.25)
        self.gamepad.left_joystick_float(x_value_float=float(stick_x), y_value_float=float(stick_y))
        self.gamepad.update()
        time.sleep(tap_seconds)
        self._reset_stick()
        time.sleep(settle_seconds)

    def _tap_inventory_right(self) -> None:
        nav = self._inventory_nav_cfg()
        right_stick_x = float(nav.get("right_stick_x", 1.0) or 1.0)
        self._tap_left_stick(right_stick_x, 0.0, vertical=False)

    def _tap_inventory_down(self) -> None:
        nav = self._inventory_nav_cfg()
        down_stick_y = float(nav.get("down_stick_y", -1.0) or -1.0)
        self._tap_left_stick(0.0, down_stick_y, vertical=True)

    def _inventory_slot_position(self, recognition: dict) -> tuple[int | None, int | None]:
        row = recognition.get("slot_row")
        col = recognition.get("slot_col")
        if row is None or col is None:
            return None, None
        return int(row), int(col)

    def _advance_inventory_selection(
        self,
        *,
        row: int | None,
        col: int | None,
        grid_columns: int,
        grid_rows: int,
        step: int,
        max_steps: int,
    ) -> str:
        if row is None or col is None:
            logger.warning(
                f"  [库存搜索 {step}/{max_steps}] 未能定位当前格，尝试右移一格"
            )
            self._tap_inventory_right()
            return "right"

        if col < grid_columns - 1:
            logger.info(
                f"  [库存搜索 {step}/{max_steps}] 右移一格 "
                f"(当前 行{row} 列{col} → 目标列{col + 1})"
            )
            self._tap_inventory_right()
            return "right"

        if row < grid_rows - 1:
            logger.info(
                f"  [库存搜索 {step}/{max_steps}] 下移一行 "
                f"(当前 行{row} 列{col} → 目标行{row + 1})"
            )
            self._tap_inventory_down()
            return "down"

        raise InventoryDriveNotFoundError(
            f"已遍历库存网格末尾（行{row} 列{col}），仍未找到目标驱动。"
        )

    def _log_inventory_move_anomaly(
        self,
        move: str,
        prev_row: int | None,
        prev_col: int | None,
        new_row: int | None,
        new_col: int | None,
    ) -> None:
        if prev_row is None or prev_col is None or new_row is None or new_col is None:
            return
        if move == "right" and (new_row != prev_row or new_col != prev_col + 1):
            logger.warning(
                f"右移结果异常: ({prev_row}, {prev_col}) → ({new_row}, {new_col})，"
                "将以视觉定位为准继续搜索。"
            )
        elif move == "down" and (new_row != prev_row + 1 or new_col != 0):
            logger.warning(
                f"下移结果异常: ({prev_row}, {prev_col}) → ({new_row}, {new_col})，"
                "将以视觉定位为准继续搜索。"
            )

    def _position_correction_cfg(self) -> dict:
        return self.calibration.get("position_correction", {}) or {}

    def _equip_transfer_dialog_cfg(self) -> dict:
        return self.calibration.get("equip_transfer_dialog", {}) or {}

    def _get_ocr_engine(self):
        if self._ocr_engine is None:
            from src.scanner.ocr_engine import OCREngine

            self._ocr_engine = OCREngine()
        return self._ocr_engine

    def _press_a_tap(self, hold_seconds: float | None = None, settle_seconds: float | None = None) -> None:
        dialog_cfg = self._equip_transfer_dialog_cfg()
        hold = float(
            hold_seconds
            if hold_seconds is not None
            else dialog_cfg.get("confirm_a_hold_seconds", 0.08) or 0.08
        )
        settle = float(
            settle_seconds
            if settle_seconds is not None
            else dialog_cfg.get("confirm_a_settle_seconds", 0.35) or 0.35
        )
        self.gamepad.press_button(button=self._buttons.XUSB_GAMEPAD_A)
        self.gamepad.update()
        time.sleep(hold)
        self.gamepad.release_button(button=self._buttons.XUSB_GAMEPAD_A)
        self.gamepad.update()
        time.sleep(settle)

    def _detect_equip_transfer_dialog(self) -> bool:
        dialog_cfg = self._equip_transfer_dialog_cfg()
        if not dialog_cfg.get("enabled", True):
            return False
        image = self._capture_foreground_bgr()
        keywords = dialog_cfg.get("keywords") or None
        return detect_equip_transfer_dialog(
            image,
            ocr_engine=self._get_ocr_engine(),
            keywords=keywords,
            width_ratio=float(dialog_cfg.get("crop_width_ratio", 0.72) or 0.72),
            height_ratio=float(dialog_cfg.get("crop_height_ratio", 0.55) or 0.55),
        )

    def _confirm_equip_transfer_dialog(self) -> bool:
        dialog_cfg = self._equip_transfer_dialog_cfg()
        if not dialog_cfg.get("enabled", True):
            return False

        settle_seconds = float(dialog_cfg.get("detect_settle_seconds", 0.45) or 0.45)
        if settle_seconds > 0:
            time.sleep(settle_seconds)

        if not self._detect_equip_transfer_dialog():
            return False

        logger.warning("检测到装备转移确认框，准备点击「确认」。")
        self._capture_debug("equip_transfer_dialog")

        nav_moves = dialog_cfg.get("confirm_nav") or [{"stick_x": 1.0, "stick_y": 0.0}]
        tap_seconds = float(dialog_cfg.get("nav_tap_seconds", 0.10) or 0.10)
        nav_settle_seconds = float(dialog_cfg.get("nav_settle_seconds", 0.20) or 0.20)
        for index, item in enumerate(nav_moves, 1):
            if not isinstance(item, dict):
                continue
            stick_x = float(item.get("stick_x", 0.0) or 0.0)
            stick_y = float(item.get("stick_y", 0.0) or 0.0)
            logger.info(
                f"  [确认框导航 {index}/{len(nav_moves)}] stick=({stick_x:.3f}, {stick_y:.3f})"
            )
            self.gamepad.left_joystick_float(x_value_float=stick_x, y_value_float=stick_y)
            self.gamepad.update()
            time.sleep(tap_seconds)
            self._reset_stick()
            time.sleep(nav_settle_seconds)

        self._press_a_tap()
        post_confirm_seconds = float(dialog_cfg.get("post_confirm_seconds", 0.50) or 0.50)
        if post_confirm_seconds > 0:
            time.sleep(post_confirm_seconds)
        self._capture_debug("after_equip_transfer_confirm")
        logger.success("已确认装备转移。")
        return True

    def _handle_post_place_dialogs(self) -> None:
        try:
            self._confirm_equip_transfer_dialog()
        except Exception as exc:
            logger.warning(f"装备转移确认框处理失败，请人工检查: {exc}")

    def _get_shape_recognizer(self) -> GoldShapeRecognizer:
        if self._shape_recognizer is None:
            template_dir = str(runtime.TEMPLATE_DIR or runtime.CONFIG_DIR / "templates")
            search_cfg = self._inventory_search_cfg()
            template_suffix = str(search_cfg.get("template_suffix", "_Inv.png") or "_Inv.png")
            purple_suffix = str(
                search_cfg.get("purple_template_suffix", "_Inv_Purple.png") or "_Inv_Purple.png"
            )
            quality_suffixes = {"Purple": purple_suffix} if purple_suffix else {}
            self._shape_recognizer = GoldShapeRecognizer(
                template_dir=template_dir,
                template_suffix=template_suffix,
                quality_template_suffixes=quality_suffixes,
            )
        return self._shape_recognizer

    def _inventory_match_qualities(self, quality: str | None = None) -> list[str]:
        if quality:
            return [quality]
        search_cfg = self._inventory_search_cfg()
        configured = search_cfg.get("match_qualities")
        if isinstance(configured, list) and configured:
            return [str(item) for item in configured if str(item).strip()]
        return ["Gold", "Purple"]

    def _inventory_templates_for_quality(self, quality: str | None = None) -> dict:
        search_cfg = self._inventory_search_cfg()
        default_quality = str(search_cfg.get("default_quality", "Gold") or "Gold")
        resolved_quality = quality or default_quality
        return self._get_shape_recognizer().templates_for_quality(resolved_quality)

    def _inventory_template_variants_for_shape_match(
        self,
        quality: str | None = None,
    ) -> dict[str, list[np.ndarray]]:
        qualities = self._inventory_match_qualities(quality)
        return self._get_shape_recognizer().templates_for_shape_match(qualities)

    def _capture_foreground_bgr(self) -> np.ndarray:
        with mss.MSS() as sct:
            screenshot, _ = capture_foreground_window(sct)
        image = np.array(screenshot)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def _inventory_panel_region(self) -> tuple[int, int, int, int]:
        rect = get_foreground_client_rect()
        profiles = ScannerConfig.get_region_profiles(rect.width, rect.height)
        _, regions = profiles[0]
        return regions["inventory_panel"]

    def _recognize_selected_drive_shape(
        self,
        *,
        quality: str | None = None,
    ) -> dict:
        image = self._capture_foreground_bgr()
        search_cfg = self._inventory_search_cfg()
        min_margin = float(search_cfg.get("min_margin", 0.05) or 0.05)
        min_confidence = float(search_cfg.get("min_confidence", 0.65) or 0.65)
        high_confidence = float(
            search_cfg.get("high_confidence", 0.95) or 0.95
        )
        min_triangle_confidence = float(
            search_cfg.get("selection_min_triangle_confidence", 0.80) or 0.80
        )
        triangle_size_cfg = search_cfg.get("selection_triangle_size_2k") or [70, 20]
        triangle_size_2k = (int(triangle_size_cfg[0]), int(triangle_size_cfg[1]))
        base_width = int(self.calibration.get("base_width", 2560) or 2560)
        base_height = int(self.calibration.get("base_height", 1440) or 1440)
        panel_region = self._inventory_panel_region()
        grid_layout = self._inventory_grid_layout(image)
        content_rect = self._inventory_content_rect(image)
        triangle_template = self._get_selection_triangle_template()
        templates = self._inventory_template_variants_for_shape_match(quality)
        result = locate_selected_inventory_shape(
            templates,
            image,
            panel_region,
            grid_layout=grid_layout,
            triangle_template=triangle_template,
            min_confidence=min_confidence,
            high_confidence=high_confidence,
            min_margin=min_margin,
            min_triangle_confidence=min_triangle_confidence,
            triangle_size_2k=triangle_size_2k,
            base_width=base_width,
            base_height=base_height,
            content_rect=content_rect,
        )
        if result.get("selection_box"):
            x1, y1, x2, y2 = result["selection_box"]
            triangle_top_left = result.get("triangle_top_left")
            logger.debug(
                f"库存选中框: ({x1}, {y1})-({x2}, {y2}) "
                f"triangle={triangle_top_left} "
                f"triangle_conf={result.get('triangle_confidence')} "
                f"shape_margin={result.get('margin')} "
                f"second={result.get('second_best_confidence')} "
                f"high_conf={result.get('high_confidence_accepted')}"
            )
        return result

    def _selection_matches_target(self, shape_id: str, recognition: dict) -> bool:
        search_cfg = self._inventory_search_cfg()
        min_confidence = float(
            search_cfg.get("target_shape_min_confidence", 0.55) or 0.55
        )
        min_margin = float(search_cfg.get("min_margin", 0.05) or 0.05)
        high_confidence = float(
            search_cfg.get("high_confidence", 0.95) or 0.95
        )
        detected = str(recognition.get("shape_id") or "Unknown")
        confidence = float(recognition.get("confidence") or -1.0)
        margin = float(recognition.get("margin") or 0.0)
        if detected != shape_id or detected == "Unknown":
            return False
        if confidence >= high_confidence or recognition.get("high_confidence_accepted"):
            return True
        return confidence >= min_confidence and margin >= min_margin

    def _wake_gamepad(self) -> None:
        """Send a disposable stick tap so the game binds the virtual gamepad input."""
        wake = self.calibration.get("gamepad_wake", {}) or {}
        if not wake.get("enabled", True):
            return

        stick_x = float(wake.get("stick_x", -1.0) or -1.0)
        stick_y = float(wake.get("stick_y", 0.0) or 0.0)
        tap_seconds = float(wake.get("tap_seconds", 0.12) or 0.12)
        settle_seconds = float(wake.get("settle_seconds", 0.50) or 0.50)
        logger.info(
            f"发送虚拟手柄唤醒信号（左摇杆轻推，避免右移库存焦点）: "
            f"stick=({stick_x:.3f}, {stick_y:.3f})"
        )
        self.gamepad.left_joystick_float(x_value_float=stick_x, y_value_float=stick_y)
        self.gamepad.update()
        time.sleep(tap_seconds)
        self._reset_stick()
        time.sleep(settle_seconds)
        self._capture_debug("after_gamepad_wake")

    def select_inventory_drive_by_shape(
        self,
        shape_id: str,
        *,
        quality: str | None = None,
    ) -> dict:
        """Search the inventory by recognizing the currently selected drive shape."""
        search_cfg = self._inventory_search_cfg()
        max_steps = max(1, int(search_cfg.get("max_steps", 20) or 20))
        grid_columns = max(1, int(search_cfg.get("grid_columns", 4) or 4))
        grid_rows = max(1, int(self._inventory_grid_cfg().get("grid_rows", 5) or 5))

        self._wake_gamepad()
        self._wait_after_inventory_move()
        recognition = self._recognize_selected_drive_shape(quality=quality)
        logger.info(
            f"{self._format_selected_slot_log(recognition)} | "
            f"triangle_conf={recognition.get('triangle_confidence')} "
            f"shape={recognition.get('shape_id')} "
            f"confidence={recognition.get('confidence')} "
            f"margin={recognition.get('margin')}"
        )
        if self._selection_matches_target(shape_id, recognition):
            logger.info(f"当前选中驱动已是目标形状 {shape_id}，无需继续搜索。")
            self._capture_debug("after_inventory_search")
            return recognition

        for step in range(1, max_steps + 1):
            self._check_stopped()
            prev_row, prev_col = self._inventory_slot_position(recognition)
            move = self._advance_inventory_selection(
                row=prev_row,
                col=prev_col,
                grid_columns=grid_columns,
                grid_rows=grid_rows,
                step=step,
                max_steps=max_steps,
            )

            self._wait_after_inventory_move()
            recognition = self._recognize_selected_drive_shape(quality=quality)
            new_row, new_col = self._inventory_slot_position(recognition)
            self._log_inventory_move_anomaly(move, prev_row, prev_col, new_row, new_col)
            logger.info(
                f"  {self._format_selected_slot_log(recognition)} | "
                f"triangle_conf={recognition.get('triangle_confidence')} "
                f"shape={recognition.get('shape_id')} "
                f"confidence={recognition.get('confidence')} "
                f"margin={recognition.get('margin')}"
            )
            if self._selection_matches_target(shape_id, recognition):
                logger.success(f"已找到目标形状 {shape_id}。")
                self._capture_debug("after_inventory_search")
                return recognition

        raise InventoryDriveNotFoundError(
            f"在库存中未找到形状为 {shape_id} 的驱动块（已搜索 {max_steps} 步）。"
        )

    def _send_drag_input(self, stick_x: float, stick_y: float) -> None:
        if not self._drag_active:
            raise RuntimeError("拖动尚未开始，不能发送拖动输入。")
        self._current_stick = (float(stick_x), float(stick_y))
        self.gamepad.left_joystick_float(
            x_value_float=self._current_stick[0],
            y_value_float=self._current_stick[1],
        )
        self.gamepad.press_button(button=self._buttons.XUSB_GAMEPAD_A)
        self.gamepad.update()

    def _hold_for(self, duration_seconds: float, stick_x: float = 0.0, stick_y: float = 0.0) -> None:
        duration_seconds = max(0.0, float(duration_seconds))
        interval_capture = float(self._debug_cfg.get("interval_seconds", 0.0) or 0.0)
        next_capture_at = time.perf_counter() + interval_capture if interval_capture > 0 else None
        end_time = time.perf_counter() + duration_seconds
        tick = 0
        while time.perf_counter() < end_time:
            self._check_stopped()
            self._send_drag_input(stick_x, stick_y)
            time.sleep(self._stick_interval)
            tick += 1
            if (
                self._drag_active
                and next_capture_at is not None
                and time.perf_counter() >= next_capture_at
            ):
                self._capture_debug(f"drag_tick_{tick:03d}")
                next_capture_at = time.perf_counter() + interval_capture
        self._send_drag_input(stick_x, stick_y)

    def _begin_drag_hold(self) -> None:
        if self._drag_active:
            raise RuntimeError("拖动已在进行中。")

        hold_seconds = float(self.calibration.get("hold_a_seconds", 0.55) or 0.55)
        post_hold_seconds = float(self.calibration.get("post_hold_a_seconds", 0.20) or 0.20)
        self._drag_active = True
        self._current_stick = (0.0, 0.0)

        logger.info(
            f"开始长按 A 抓起驱动块，全程保持按住 "
            f"(hold={hold_seconds:.2f}s, settle={post_hold_seconds:.2f}s)"
        )
        self._hold_for(hold_seconds, 0.0, 0.0)
        if post_hold_seconds > 0:
            self._hold_for(post_hold_seconds, 0.0, 0.0)
        self._capture_debug("after_pickup")

    def _move_stick_while_holding(self, move: StickMove) -> None:
        if not self._drag_active:
            raise RuntimeError("必须先长按 A 抓起驱动块，才能在保持按住时移动。")
        logger.debug(
            f"保持 A 按住并移动摇杆: label={move.label} "
            f"stick=({move.stick_x:.3f}, {move.stick_y:.3f}) "
            f"duration={move.duration_seconds:.2f}s"
        )
        self._hold_for(move.duration_seconds, move.stick_x, move.stick_y)
        self._capture_debug(f"after_move_{move.label}")

    def _finish_drag_hold(self) -> None:
        if not self._drag_active:
            return

        self._capture_debug("before_release")
        logger.info("已到达目标位置，松开 A 放置驱动块")
        self.gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
        self.gamepad.release_button(button=self._buttons.XUSB_GAMEPAD_A)
        self.gamepad.update()
        self._drag_active = False
        self._current_stick = (0.0, 0.0)

        settle_seconds = float(self.calibration.get("settle_seconds", 0.25) or 0.25)
        time.sleep(settle_seconds)
        self._handle_post_place_dialogs()

    def assemble_piece(
        self,
        start_r: int,
        start_c: int,
        piece_id: str,
        *,
        quality: str | None = None,
    ) -> list[StickMove]:
        return self.drag_to_grid_cell(start_r, start_c, piece_id=piece_id, quality=quality)

    def assemble_all_pieces(
        self,
        pieces: list[tuple[str, int, int]],
        *,
        stop_on_error: bool = True,
    ) -> list[AssemblyPieceResult]:
        results: list[AssemblyPieceResult] = []
        total = len(pieces)
        for index, (piece_id, start_r, start_c) in enumerate(pieces, 1):
            self._check_stopped()
            logger.warning(
                f"开始装配第 {index}/{total} 块: {piece_id} → ({start_r}, {start_c})"
            )
            try:
                applied_moves = self.assemble_piece(start_r, start_c, piece_id)
                results.append(
                    AssemblyPieceResult(
                        piece_id=piece_id,
                        start_r=start_r,
                        start_c=start_c,
                        applied_moves=applied_moves,
                        success=True,
                    )
                )
                logger.success(
                    f"第 {index}/{total} 块装配完成: {piece_id} → ({start_r}, {start_c})"
                )
            except AssemblyStoppedError:
                self._abort_drag_hold()
                logger.warning(f"第 {index}/{total} 块装配被用户中止: {piece_id}")
                results.append(
                    AssemblyPieceResult(
                        piece_id=piece_id,
                        start_r=start_r,
                        start_c=start_c,
                        applied_moves=[],
                        success=False,
                        error="用户中止装配",
                    )
                )
                break
            except Exception as exc:
                logger.error(
                    f"第 {index}/{total} 块装配失败: {piece_id} → ({start_r}, {start_c}) | {exc}"
                )
                results.append(
                    AssemblyPieceResult(
                        piece_id=piece_id,
                        start_r=start_r,
                        start_c=start_c,
                        applied_moves=[],
                        success=False,
                        error=str(exc),
                    )
                )
                if stop_on_error:
                    break
        return results

    def _nudge_moves_to_target(self, start_r: int, start_c: int) -> list[StickMove]:
        origin_r = int(self.calibration.get("grid_origin_r", 0) or 0)
        origin_c = int(self.calibration.get("grid_origin_c", 0) or 0)
        delta_r = int(start_r) - origin_r
        delta_c = int(start_c) - origin_c
        if delta_r == 0 and delta_c == 0:
            return []

        nudge = self.calibration.get("grid_nudge", {}) or {}
        per_row = float(nudge.get("stick_y_per_row", 0.0) or 0.0)
        per_col = float(nudge.get("stick_x_per_col", 0.0) or 0.0)
        per_step = float(nudge.get("duration_per_step", 0.12) or 0.12)
        moves: list[StickMove] = []
        for step in range(abs(delta_r)):
            moves.append(
                StickMove(
                    stick_x=0.0,
                    stick_y=per_row if delta_r > 0 else -per_row,
                    duration_seconds=per_step,
                    label=f"target_r_{step + 1}",
                )
            )
        for step in range(abs(delta_c)):
            moves.append(
                StickMove(
                    stick_x=per_col if delta_c > 0 else -per_col,
                    stick_y=0.0,
                    duration_seconds=per_step,
                    label=f"target_c_{step + 1}",
                )
            )
        return moves

    def _moves_from_config_list(self, start_r: int, start_c: int) -> list[StickMove]:
        configured = self.calibration.get("drag_moves")
        if isinstance(configured, list) and configured:
            moves: list[StickMove] = []
            for index, item in enumerate(configured, 1):
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label") or f"move_{index:02d}")
                moves.append(
                    StickMove(
                        stick_x=float(item.get("stick_x", 0.0) or 0.0),
                        stick_y=float(item.get("stick_y", 0.0) or 0.0),
                        duration_seconds=float(item.get("duration_seconds", 0.0) or 0.0),
                        label=label,
                    )
                )
            if moves:
                moves.extend(self._nudge_moves_to_target(start_r, start_c))
                return moves

        base = self.calibration.get("drag_from_inventory_focus", {}) or {}
        nudge = self.calibration.get("grid_nudge", {}) or {}
        origin_r = int(self.calibration.get("grid_origin_r", 0) or 0)
        origin_c = int(self.calibration.get("grid_origin_c", 0) or 0)

        moves = [
            StickMove(
                stick_x=float(base.get("stick_x", 0.0) or 0.0),
                stick_y=float(base.get("stick_y", 0.0) or 0.0),
                duration_seconds=float(base.get("duration_seconds", 1.0) or 1.0),
                label="base",
            )
        ]

        delta_r = int(start_r) - origin_r
        delta_c = int(start_c) - origin_c
        per_row = float(nudge.get("stick_y_per_row", 0.0) or 0.0)
        per_col = float(nudge.get("stick_x_per_col", 0.0) or 0.0)
        per_step = float(nudge.get("duration_per_step", 0.12) or 0.12)

        for step in range(abs(delta_r)):
            moves.append(
                StickMove(
                    stick_x=0.0,
                    stick_y=per_row if delta_r > 0 else -per_row,
                    duration_seconds=per_step,
                    label=f"nudge_r_{step + 1}",
                )
            )
        for step in range(abs(delta_c)):
            moves.append(
                StickMove(
                    stick_x=per_col if delta_c > 0 else -per_col,
                    stick_y=0.0,
                    duration_seconds=per_step,
                    label=f"nudge_c_{step + 1}",
                )
            )
        return moves

    def _detect_piece_position(
        self,
        piece_id: str,
        *,
        quality: str | None = None,
        target_r: int | None = None,
        target_c: int | None = None,
    ) -> dict:
        image = self._capture_foreground_bgr()
        image_h, image_w = image.shape[:2]
        board_region = assembly_board_region(
            self.calibration,
            image_width=image_w,
            image_height=image_h,
        )
        grid_rows, grid_cols = grid_dimensions(self.calibration)
        correction_cfg = self._position_correction_cfg()
        min_confidence = float(correction_cfg.get("min_match_confidence", 0.50) or 0.50)
        margin_ratio = float(correction_cfg.get("detection_expand_margin_ratio", 0.35) or 0.35)
        left_margin_ratio = correction_cfg.get("detection_expand_left_margin_ratio")
        right_margin_ratio = correction_cfg.get("detection_expand_right_margin_ratio")
        detection_region = expand_board_region(
            board_region,
            image_w,
            image_h,
            margin_ratio=margin_ratio,
            left_margin_ratio=float(left_margin_ratio) if left_margin_ratio is not None else None,
            right_margin_ratio=float(right_margin_ratio) if right_margin_ratio is not None else None,
        )
        min_orange_solidity = float(correction_cfg.get("min_orange_solidity", 0.42) or 0.42)
        return detect_dragged_piece_anchor(
            image,
            piece_id,
            self._inventory_template_variants_for_shape_match(quality),
            board_region,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            min_confidence=min_confidence,
            target_r=float(target_r) if target_r is not None else None,
            target_c=float(target_c) if target_c is not None else None,
            detection_region=detection_region,
            min_orange_solidity=min_orange_solidity,
        )

    def _correction_settle_seconds(self) -> float:
        correction_cfg = self._position_correction_cfg()
        return float(correction_cfg.get("post_drag_settle_seconds", 0.35) or 0.35)

    def _correct_drag_position(
        self,
        piece_id: str,
        target_r: int,
        target_c: int,
        *,
        quality: str | None = None,
    ) -> list[StickMove]:
        correction_cfg = self._position_correction_cfg()
        if not correction_cfg.get("enabled", True):
            return []

        max_iterations = max(1, int(correction_cfg.get("max_iterations", 10) or 10))
        max_cell_error = float(correction_cfg.get("max_cell_error", 0.45) or 0.45)
        settle_seconds = self._correction_settle_seconds()
        origin_r = int(self.calibration.get("grid_origin_r", 0) or 0)
        origin_c = int(self.calibration.get("grid_origin_c", 0) or 0)
        outlier_max_cell_jump = float(correction_cfg.get("outlier_max_cell_jump", 1.20) or 1.20)
        stop_on_worse_streak = max(1, int(correction_cfg.get("stop_on_worse_streak", 2) or 2))
        corrections: list[StickMove] = []
        used_origin_fallback = False
        last_accepted: tuple[float, float] | None = None
        best_error = float("inf")
        worse_streak = 0

        for iteration in range(1, max_iterations + 1):
            self._check_stopped()
            detection = self._detect_piece_position(
                piece_id,
                quality=quality,
                target_r=target_r,
                target_c=target_c,
            )
            detected_r = detection.get("anchor_r")
            detected_c = detection.get("anchor_c")
            if (
                detected_r is not None
                and detected_c is not None
                and last_accepted is not None
                and is_detection_outlier(
                    float(detected_r),
                    float(detected_c),
                    last_accepted[0],
                    last_accepted[1],
                    max_cell_jump=outlier_max_cell_jump,
                )
            ):
                worse_streak += 1
                logger.warning(
                    f"  [位置校正 {iteration}/{max_iterations}] "
                    f"检测跳变过大，忽略本次结果 "
                    f"({detected_r}, {detected_c}) vs 上次 {last_accepted} "
                    f"(streak={worse_streak}/{stop_on_worse_streak})"
                )
                if worse_streak >= stop_on_worse_streak:
                    logger.warning("检测持续跳变，停止位置校正。")
                    break
                continue

            error = compute_position_error(detected_r, detected_c, target_r, target_c)
            if error is None:
                if used_origin_fallback:
                    logger.warning(
                        f"  [位置校正 {iteration}/{max_iterations}] "
                        "连续检测失败，停止校正。"
                    )
                    break
                used_origin_fallback = True
                delta_r, delta_c = estimate_position_error_from_origin(
                    target_r,
                    target_c,
                    origin_r=origin_r,
                    origin_c=origin_c,
                )
                logger.warning(
                    f"  [位置校正 {iteration}/{max_iterations}] "
                    f"未能检测拖动块 (conf={detection.get('confidence')})，"
                    f"改从估计起点 ({origin_r}, {origin_c}) 计算误差 "
                    f"(Δr={delta_r:.2f}, Δc={delta_c:.2f})"
                )
            else:
                delta_r, delta_c = error
                current_error = position_error_magnitude(delta_r, delta_c)
                if iteration > 1 and current_error > best_error + max_cell_error:
                    worse_streak += 1
                    logger.warning(
                        f"  [位置校正 {iteration}/{max_iterations}] "
                        f"误差变大 ({current_error:.2f} > {best_error:.2f})，"
                        f"停止继续校正 (streak={worse_streak}/{stop_on_worse_streak})"
                    )
                    if worse_streak >= stop_on_worse_streak:
                        break
                    continue

                worse_streak = 0
                best_error = min(best_error, current_error)
                last_accepted = (float(detected_r), float(detected_c))
                logger.info(
                    f"  [位置校正 {iteration}/{max_iterations}] "
                    f"检测=({detected_r}, {detected_c}) 目标=({target_r}, {target_c}) "
                    f"误差=(Δr={delta_r:.2f}, Δc={delta_c:.2f}) "
                    f"conf={detection.get('confidence')} method={detection.get('method')}"
                )

            if not needs_position_correction(delta_r, delta_c, max_cell_error=max_cell_error):
                if error is not None:
                    logger.success("拖动位置已接近目标格，停止校正。")
                break

            stick_x, stick_y, duration = correction_stick_for_error(delta_r, delta_c, correction_cfg)
            if abs(stick_x) < 0.01 and abs(stick_y) < 0.01:
                logger.info("  当前轴误差已在容差内，停止校正。")
                break
            move = StickMove(
                stick_x=stick_x,
                stick_y=stick_y,
                duration_seconds=duration,
                label=f"correct_{iteration:02d}",
            )
            corrections.append(move)
            logger.info(
                f"  发送校正摇杆: stick=({stick_x:.3f}, {stick_y:.3f}) duration={duration:.2f}s"
            )
            self._move_stick_while_holding(move)
            if settle_seconds > 0:
                time.sleep(settle_seconds)
        else:
            logger.warning("位置校正达到最大次数，仍将尝试放置。")

        return corrections

    def drag_to_grid_cell(
        self,
        start_r: int,
        start_c: int,
        piece_id: str = "H_2",
        *,
        quality: str | None = None,
    ) -> list[StickMove]:
        moves = self._moves_from_config_list(start_r, start_c)
        logger.info(
            f"开始拖动到网格 ({start_r}, {start_c})，共 {len(moves)} 段初始摇杆动作。"
        )

        self.select_inventory_drive_by_shape(piece_id, quality=quality)
        self._begin_drag_hold()
        applied_moves: list[StickMove] = []
        try:
            for index, move in enumerate(moves, 1):
                self._check_stopped()
                logger.info(
                    f"  [拖动 {index}/{len(moves)} | A 保持按住 | {move.label}] "
                    f"stick=({move.stick_x:.3f}, {move.stick_y:.3f}) "
                    f"duration={move.duration_seconds:.2f}s"
                )
                self._move_stick_while_holding(move)
                applied_moves.append(move)

                correction_cfg = self._position_correction_cfg()
                if correction_cfg.get("correct_after_each_move", False):
                    applied_moves.extend(
                        self._correct_drag_position(
                            piece_id, start_r, start_c, quality=quality
                        )
                    )

            settle_seconds = self._correction_settle_seconds()
            if settle_seconds > 0:
                time.sleep(settle_seconds)
            applied_moves.extend(
                self._correct_drag_position(piece_id, start_r, start_c, quality=quality)
            )
        finally:
            if self._stopped and self._drag_active:
                self._abort_drag_hold()
            elif self._drag_active:
                self._finish_drag_hold()

        return applied_moves


def run_assembly_drag_test(
    *,
    role_name: str,
    piece_id: str,
    start_r: int,
    start_c: int,
    window_title: str,
    delay_seconds: float = 3.0,
    calibration: dict | None = None,
) -> AssemblyTestResult:
    merged_calibration = dict(calibration or load_assembly_calibration())
    controller = GamepadAssemblyController(calibration=merged_calibration)
    before_path = controller.capture_screenshot("assembly_test_before.png")

    logger.warning(f"装配测试将在 {delay_seconds:.1f} 秒后接管控制，请保持游戏界面不动。")
    time.sleep(max(0.0, float(delay_seconds)))

    applied_moves = controller.drag_to_grid_cell(start_r, start_c, piece_id=piece_id)
    after_path = controller.capture_screenshot("assembly_test_after.png")
    return AssemblyTestResult(
        role_name=role_name,
        piece_id=piece_id,
        start_r=start_r,
        start_c=start_c,
        window_title=window_title,
        before_screenshot=before_path,
        after_screenshot=after_path,
        applied_moves=applied_moves,
        debug_screenshots=list(controller._debug_screenshots),
    )


def run_full_assembly_test(
    *,
    blueprint_role: str,
    pieces: list[tuple[str, int, int]],
    window_title: str,
    delay_seconds: float = 3.0,
    calibration: dict | None = None,
    stop_on_error: bool = True,
) -> FullAssemblyResult:
    merged_calibration = dict(calibration or load_assembly_calibration())
    controller = GamepadAssemblyController(calibration=merged_calibration)
    set_active_assembly_controller(controller)
    before_path = controller.capture_screenshot("assembly_full_before.png")

    logger.warning(
        f"完整装配测试将在 {delay_seconds:.1f} 秒后接管控制，共 {len(pieces)} 块驱动。"
    )
    try:
        end_at = time.perf_counter() + max(0.0, float(delay_seconds))
        while time.perf_counter() < end_at:
            controller._check_stopped()
            time.sleep(min(0.1, max(0.0, end_at - time.perf_counter())))

        piece_results = controller.assemble_all_pieces(pieces, stop_on_error=stop_on_error)
    finally:
        set_active_assembly_controller(None)

    after_path = controller.capture_screenshot("assembly_full_after.png")
    return FullAssemblyResult(
        blueprint_role=blueprint_role,
        window_title=window_title,
        before_screenshot=before_path,
        after_screenshot=after_path,
        pieces=piece_results,
        debug_screenshots=list(controller._debug_screenshots),
    )
