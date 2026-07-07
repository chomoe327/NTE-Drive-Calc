# 使用虚拟手柄执行驱动块拖动装配测试。
"""Virtual gamepad drag-and-drop helpers for assembly automation tests."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import mss
import mss.tools

from src.app import runtime
from src.scanner.gamepad_controller import ViGEmDriverNotReadyError, _format_vigem_error
from src.scanner.window_capture import capture_foreground_window
from src.utils.logger import logger


@dataclass(frozen=True)
class StickMove:
    stick_x: float
    stick_y: float
    duration_seconds: float


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


def load_assembly_calibration(config_path: Path | None = None) -> dict:
    path = config_path or runtime.CONFIG_DIR / "assembly_calibration.json"
    bundled = runtime.BUNDLED_CONFIG_DIR / "assembly_calibration.json"
    for candidate in (path, bundled):
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as handle:
                return json.load(handle)
    raise FileNotFoundError("找不到 assembly_calibration.json 配置文件。")


class GamepadAssemblyController:
    """Drive-block drag controller built on top of ViGEm virtual gamepad."""

    def __init__(self, calibration: dict | None = None):
        self.calibration = calibration or load_assembly_calibration()
        self._stick_interval = float(
            self.calibration.get("stick_update_interval_seconds", 0.05) or 0.05
        )
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

    def hold_button_a(self) -> None:
        hold_seconds = float(self.calibration.get("hold_a_seconds", 0.35) or 0.35)
        self.gamepad.press_button(button=self._buttons.XUSB_GAMEPAD_A)
        self.gamepad.update()
        time.sleep(hold_seconds)

    def move_left_stick(self, stick_x: float, stick_y: float, duration_seconds: float) -> None:
        duration_seconds = max(0.0, float(duration_seconds))
        end_time = time.perf_counter() + duration_seconds
        self.gamepad.left_joystick_float(x_value_float=float(stick_x), y_value_float=float(stick_y))
        self.gamepad.update()
        while time.perf_counter() < end_time:
            time.sleep(self._stick_interval)
            self.gamepad.left_joystick_float(x_value_float=float(stick_x), y_value_float=float(stick_y))
            self.gamepad.update()

    def release_button_a(self) -> None:
        self.gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
        self.gamepad.release_button(button=self._buttons.XUSB_GAMEPAD_A)
        self.gamepad.update()
        settle_seconds = float(self.calibration.get("settle_seconds", 0.20) or 0.20)
        time.sleep(settle_seconds)

    def _build_drag_moves(self, start_r: int, start_c: int) -> list[StickMove]:
        base = self.calibration.get("drag_from_inventory_focus", {}) or {}
        nudge = self.calibration.get("grid_nudge", {}) or {}
        origin_r = int(self.calibration.get("grid_origin_r", 0) or 0)
        origin_c = int(self.calibration.get("grid_origin_c", 0) or 0)

        moves = [
            StickMove(
                stick_x=float(base.get("stick_x", 0.0) or 0.0),
                stick_y=float(base.get("stick_y", 0.0) or 0.0),
                duration_seconds=float(base.get("duration_seconds", 1.0) or 1.0),
            )
        ]

        delta_r = int(start_r) - origin_r
        delta_c = int(start_c) - origin_c
        per_row = float(nudge.get("stick_y_per_row", 0.0) or 0.0)
        per_col = float(nudge.get("stick_x_per_col", 0.0) or 0.0)
        per_step = float(nudge.get("duration_per_step", 0.12) or 0.12)

        for _ in range(abs(delta_r)):
            moves.append(
                StickMove(
                    stick_x=0.0,
                    stick_y=per_row if delta_r > 0 else -per_row,
                    duration_seconds=per_step,
                )
            )
        for _ in range(abs(delta_c)):
            moves.append(
                StickMove(
                    stick_x=per_col if delta_c > 0 else -per_col,
                    stick_y=0.0,
                    duration_seconds=per_step,
                )
            )
        return moves

    def drag_to_grid_cell(self, start_r: int, start_c: int) -> list[StickMove]:
        moves = self._build_drag_moves(start_r, start_c)
        logger.info(
            f"开始拖动到网格 ({start_r}, {start_c})，共 {len(moves)} 段摇杆动作。"
        )
        for index, move in enumerate(moves, 1):
            logger.info(
                f"  [拖动 {index}/{len(moves)}] stick=({move.stick_x:.3f}, {move.stick_y:.3f}) "
                f"duration={move.duration_seconds:.2f}s"
            )
            self.move_left_stick(move.stick_x, move.stick_y, move.duration_seconds)
        return moves

    @staticmethod
    def capture_screenshot(filename: str) -> str:
        output_dir = runtime.SCREENSHOT_DIR
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        with mss.MSS() as sct:
            screenshot, _ = capture_foreground_window(sct)
        mss.tools.to_png(screenshot.rgb, screenshot.size, output=output_path)
        logger.info(f"装配测试截图已保存: {output_path}")
        return output_path


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
    """Capture before/after screenshots while dragging one drive block to the target cell."""
    controller = GamepadAssemblyController(calibration=calibration)
    before_path = controller.capture_screenshot("assembly_test_before.png")

    logger.warning("装配测试将在 %.1f 秒后接管控制，请保持游戏界面不动。", delay_seconds)
    time.sleep(max(0.0, float(delay_seconds)))

    controller.hold_button_a()
    applied_moves = controller.drag_to_grid_cell(start_r, start_c)
    controller.release_button_a()

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
    )
