# 使用虚拟手柄执行驱动块拖动装配测试。
"""Virtual gamepad drag-and-drop helpers for assembly automation tests."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
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
        self._debug_cfg = self.calibration.get("debug_capture", {}) or {}
        self._debug_enabled = bool(self._debug_cfg.get("enabled", True))
        self._debug_screenshots: list[str] = []
        self._debug_counter = 0
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

    def _tap_left_stick(self, stick_x: float, stick_y: float) -> None:
        nav = self.calibration.get("inventory_nav", {}) or {}
        tap_seconds = float(nav.get("tap_seconds", 0.10) or 0.10)
        settle_seconds = float(nav.get("settle_seconds", 0.25) or 0.25)
        self.gamepad.left_joystick_float(x_value_float=float(stick_x), y_value_float=float(stick_y))
        self.gamepad.update()
        time.sleep(tap_seconds)
        self._reset_stick()
        time.sleep(settle_seconds)

    def _wake_gamepad(self) -> None:
        """Send a disposable stick tap so the game binds the virtual gamepad input."""
        wake = self.calibration.get("gamepad_wake", {}) or {}
        if not wake.get("enabled", True):
            return

        stick_x = float(wake.get("stick_x", 1.0) or 1.0)
        stick_y = float(wake.get("stick_y", 0.0) or 0.0)
        tap_seconds = float(wake.get("tap_seconds", 0.12) or 0.12)
        settle_seconds = float(wake.get("settle_seconds", 0.50) or 0.50)
        logger.info(
            f"发送虚拟手柄唤醒信号（此输入可能被游戏消耗，不会移动库存焦点）: "
            f"stick=({stick_x:.3f}, {stick_y:.3f})"
        )
        self.gamepad.left_joystick_float(x_value_float=stick_x, y_value_float=stick_y)
        self.gamepad.update()
        time.sleep(tap_seconds)
        self._reset_stick()
        time.sleep(settle_seconds)
        self._capture_debug("after_gamepad_wake")

    def focus_inventory_item(self) -> None:
        """Navigate from the first inventory slot to the configured focus index."""
        target_index = max(1, int(self.calibration.get("inventory_focus_index", 1) or 1))
        self._wake_gamepad()
        if target_index <= 1:
            logger.info("库存焦点已在第 1 个驱动，无需导航。")
            return

        nav = self.calibration.get("inventory_nav", {}) or {}
        right_stick_x = float(nav.get("right_stick_x", 1.0) or 1.0)
        steps = target_index - 1
        logger.info(
            f"库存导航: 唤醒后从第 1 个驱动右移 {steps} 格，选中第 {target_index} 个驱动。"
        )
        for step in range(steps):
            logger.info(f"  [库存导航 {step + 1}/{steps}] stick_x={right_stick_x:.3f}")
            self._tap_left_stick(right_stick_x, 0.0)
        self._capture_debug("after_inventory_focus")

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

    def drag_to_grid_cell(self, start_r: int, start_c: int) -> list[StickMove]:
        moves = self._moves_from_config_list(start_r, start_c)
        logger.info(
            f"开始拖动到网格 ({start_r}, {start_c})，共 {len(moves)} 段摇杆动作。"
        )

        self.focus_inventory_item()
        self._begin_drag_hold()
        try:
            for index, move in enumerate(moves, 1):
                logger.info(
                    f"  [拖动 {index}/{len(moves)} | A 保持按住 | {move.label}] "
                    f"stick=({move.stick_x:.3f}, {move.stick_y:.3f}) "
                    f"duration={move.duration_seconds:.2f}s"
                )
                self._move_stick_while_holding(move)
        finally:
            if self._drag_active:
                self._finish_drag_hold()

        return moves


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
        debug_screenshots=list(controller._debug_screenshots),
    )
