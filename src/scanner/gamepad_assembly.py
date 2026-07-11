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
from src.scanner.mouse_input import click_screen_absolute
from src.scanner.shape_recognizer import GoldShapeRecognizer
from src.scanner.window_capture import (
    capture_foreground_window,
    get_foreground_client_rect,
    scale_point,
)
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
    """Raised when shape-filter inventory selection cannot prepare the target drive."""


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
        self._ocr_engine = None
        self._filtered_shape_id: str | None = None
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

    def _inventory_filter_cfg(self) -> dict:
        return self.calibration.get("inventory_filter", {}) or {}

    def _inventory_search_cfg(self) -> dict:
        return self.calibration.get("inventory_search", {}) or {}

    def _base_size(self) -> tuple[int, int]:
        return (
            int(self.calibration.get("base_width", 2560) or 2560),
            int(self.calibration.get("base_height", 1440) or 1440),
        )

    def _point_2k_to_client(self, point_2k: list[int] | tuple[int, int]) -> tuple[int, int]:
        rect = get_foreground_client_rect()
        content_rect = ScannerConfig.get_content_rect(rect.width, rect.height)
        return scale_point(
            (int(point_2k[0]), int(point_2k[1])),
            rect.width,
            rect.height,
            self._base_size(),
            content_rect=content_rect,
        )

    def _click_point_2k(self, point_2k: list[int] | tuple[int, int], *, label: str = "") -> None:
        self._check_stopped()
        filter_cfg = self._inventory_filter_cfg()
        move_settle = float(filter_cfg.get("click_move_settle_seconds", 0.05) or 0.05)
        click_hold = float(filter_cfg.get("click_hold_seconds", 0.02) or 0.02)
        client_x, client_y = self._point_2k_to_client(point_2k)
        rect = get_foreground_client_rect()
        screen_x = rect.left + client_x
        screen_y = rect.top + client_y
        suffix = f" ({label})" if label else ""
        logger.info(
            f"点击 UI{suffix}: 2k=({int(point_2k[0])}, {int(point_2k[1])}) "
            f"client=({client_x}, {client_y}) screen=({screen_x}, {screen_y})"
        )
        click_screen_absolute(
            screen_x,
            screen_y,
            move_settle_seconds=move_settle,
            click_hold_seconds=click_hold,
        )

    def _sleep_cfg(self, key: str, default: float) -> None:
        filter_cfg = self._inventory_filter_cfg()
        delay = float(filter_cfg.get(key, default) or default)
        if delay > 0:
            time.sleep(delay)

    def _shape_center_2k(self, shape_id: str) -> tuple[int, int]:
        filter_cfg = self._inventory_filter_cfg()
        centers = filter_cfg.get("shape_centers_2k") or {}
        point = centers.get(shape_id)
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise InventoryDriveNotFoundError(
                f"装配标定缺少形状 {shape_id} 的筛选坐标（inventory_filter.shape_centers_2k）。"
            )
        return int(point[0]), int(point[1])

    def _apply_shape_filter(self, shape_id: str) -> None:
        filter_cfg = self._inventory_filter_cfg()
        if not filter_cfg.get("enabled", True):
            raise InventoryDriveNotFoundError("inventory_filter.enabled=false，无法通过筛选选择形状。")

        filter_button = filter_cfg.get("filter_button_2k")
        shape_select = filter_cfg.get("shape_click_select_2k")
        modal_reset = filter_cfg.get("shape_modal_reset_2k")
        modal_confirm = filter_cfg.get("shape_modal_confirm_2k")
        panel_confirm = filter_cfg.get("filter_panel_confirm_2k")
        for name, point in (
            ("filter_button_2k", filter_button),
            ("shape_click_select_2k", shape_select),
            ("shape_modal_confirm_2k", modal_confirm),
            ("filter_panel_confirm_2k", panel_confirm),
        ):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise InventoryDriveNotFoundError(f"装配标定缺少有效坐标: inventory_filter.{name}")

        shape_point = self._shape_center_2k(shape_id)
        logger.warning(f"通过形状筛选定位库存驱动: {shape_id}")

        self._click_point_2k(filter_button, label="筛选按钮")
        self._sleep_cfg("after_open_filter_seconds", 0.45)

        self._click_point_2k(shape_select, label="外形-点击选择")
        self._sleep_cfg("after_open_shape_modal_seconds", 0.45)

        if filter_cfg.get("reset_before_select", True):
            if isinstance(modal_reset, (list, tuple)) and len(modal_reset) == 2:
                self._click_point_2k(modal_reset, label="重置筛选")
                self._sleep_cfg("after_reset_seconds", 0.25)

        self._click_point_2k(shape_point, label=f"形状 {shape_id}")
        self._sleep_cfg("after_click_shape_seconds", 0.25)

        self._click_point_2k(modal_confirm, label="确认筛选")
        self._sleep_cfg("after_confirm_modal_seconds", 0.40)

        self._click_point_2k(panel_confirm, label="筛选面板确认")
        self._sleep_cfg("after_confirm_panel_seconds", 0.55)
        self._capture_debug("after_inventory_filter")
        logger.success(f"已筛选形状 {shape_id}，准备抓取库存第一格。")

    def _ensure_shape_filter(self, shape_id: str) -> None:
        if self._filtered_shape_id == shape_id:
            logger.info(f"当前筛选已是 {shape_id}，跳过重复筛选。")
            return
        self._apply_shape_filter(shape_id)
        self._filtered_shape_id = shape_id

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

        logger.warning("检测到装备转移确认框（取消+确认），直接按 A 确认。")
        self._capture_debug("equip_transfer_dialog")

        nav_moves = dialog_cfg.get("confirm_nav") or []
        if isinstance(nav_moves, list) and nav_moves:
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
        """Filter inventory to the target shape, then focus the first matching slot."""
        del quality  # board correction still uses quality; inventory pick is shape-only
        self._ensure_shape_filter(shape_id)
        self._wake_gamepad()
        self._capture_debug("after_inventory_search")
        return {
            "shape_id": shape_id,
            "method": "shape_filter",
            "confidence": 1.0,
            "margin": 1.0,
        }

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
