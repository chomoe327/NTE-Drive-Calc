# 使用虚拟手柄执行驱动块拖动装配测试。
"""Virtual gamepad drag-and-drop helpers for assembly automation tests."""

from __future__ import annotations

import json
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


def assembly_test_dir() -> Path:
    output_dir = runtime.ACCOUNT_DATA_ROOT / "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


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
        self._drag_active = False
        self._current_stick = (0.0, 0.0)

    def _send_drag_input(self, stick_x: float, stick_y: float) -> None:
        """Send stick + A-pressed in one frame. A is never released while dragging."""
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
        end_time = time.perf_counter() + max(0.0, float(duration_seconds))
        while time.perf_counter() < end_time:
            self._send_drag_input(stick_x, stick_y)
            time.sleep(self._stick_interval)
        self._send_drag_input(stick_x, stick_y)

    def _begin_drag_hold(self) -> None:
        """Press and continuously hold A until _finish_drag_hold is called."""
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

    def _move_stick_while_holding(self, stick_x: float, stick_y: float, duration_seconds: float) -> None:
        if not self._drag_active:
            raise RuntimeError("必须先长按 A 抓起驱动块，才能在保持按住时移动。")
        logger.debug(
            f"保持 A 按住并移动摇杆: stick=({stick_x:.3f}, {stick_y:.3f}) "
            f"duration={duration_seconds:.2f}s"
        )
        self._hold_for(duration_seconds, stick_x, stick_y)

    def _finish_drag_hold(self) -> None:
        """Release A only after reaching the target grid cell."""
        if not self._drag_active:
            return

        logger.info("已到达目标位置，松开 A 放置驱动块")
        self.gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
        self.gamepad.release_button(button=self._buttons.XUSB_GAMEPAD_A)
        self.gamepad.update()
        self._drag_active = False
        self._current_stick = (0.0, 0.0)

        settle_seconds = float(self.calibration.get("settle_seconds", 0.25) or 0.25)
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
        """Pick up with A held, move to target cell while still holding, then release A."""
        moves = self._build_drag_moves(start_r, start_c)
        logger.info(
            f"开始拖动到网格 ({start_r}, {start_c})，共 {len(moves)} 段摇杆动作。"
        )

        self._begin_drag_hold()
        try:
            for index, move in enumerate(moves, 1):
                logger.info(
                    f"  [拖动 {index}/{len(moves)} | A 保持按住] "
                    f"stick=({move.stick_x:.3f}, {move.stick_y:.3f}) "
                    f"duration={move.duration_seconds:.2f}s"
                )
                self._move_stick_while_holding(
                    move.stick_x,
                    move.stick_y,
                    move.duration_seconds,
                )
        finally:
            if self._drag_active:
                self._finish_drag_hold()

        return moves

    @staticmethod
    def capture_screenshot(filename: str) -> str:
        output_dir = assembly_test_dir()
        output_path = str(output_dir / filename)
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

    logger.warning(f"装配测试将在 {delay_seconds:.1f} 秒后接管控制，请保持游戏界面不动。")
    time.sleep(max(0.0, float(delay_seconds)))

    applied_moves = controller.drag_to_grid_cell(start_r, start_c)

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
