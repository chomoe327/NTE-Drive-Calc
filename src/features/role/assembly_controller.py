# 配装页自动装配测试入口。
"""UI controller for the assembly automation prototype test."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QMessageBox

from src.app import runtime
from src.app.workers import WorkerThread
from src.scanner.assembly_planner import plan_test_placement
from src.scanner.gamepad_assembly import load_assembly_calibration, run_assembly_drag_test
from src.scanner.window_control import (
    WindowControlError,
    activate_game_window,
    activate_window,
    grant_foreground_permission,
    is_window_foreground,
)
from src.utils.logger import logger

__all__ = [
    "install_methods",
    "_start_assembly_test",
    "_on_assembly_test_done",
    "_on_assembly_test_error",
]


def install_methods(app_module, window_cls):
    window_cls._start_assembly_test = _start_assembly_test
    window_cls._on_assembly_test_done = _on_assembly_test_done
    window_cls._on_assembly_test_error = _on_assembly_test_error


def _resolve_config_dir() -> str:
    for candidate in (runtime.CONFIG_DIR, runtime.BUNDLED_CONFIG_DIR):
        if (candidate / "roles.json").exists():
            return str(candidate)
    return "config"


def _game_window_title(calibration: dict) -> str:
    title = calibration.get("game_window_title")
    if title:
        return str(title).strip()
    candidates = calibration.get("title_candidates") or ["异环"]
    if isinstance(candidates, list) and candidates:
        return str(candidates[0]).strip()
    return "异环"


def _start_assembly_test(self, _role_name: str | None = None):
    if not sys.platform.startswith("win"):
        QMessageBox.warning(self, "不支持", "自动装配测试仅支持 Windows 平台。")
        return

    confirmed = QMessageBox.question(
        self,
        "自动装配测试",
        (
            "本阶段为原型测试，固定执行：\n"
            "  角色：真红\n"
            "  驱动块：H_2\n"
            "  目标格：(1, 0)\n\n"
            "请先确保：\n"
            "  1. 游戏已在真红驱动装配页\n"
            "  2. 中间 5×5 网格为空\n"
            "  3. 左侧库存第一个驱动已被选中\n\n"
            "点击确定后程序将最小化，并切换到「异环」窗口，3 秒后接管虚拟手柄。"
        ),
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No,
    )
    if confirmed != QMessageBox.Yes:
        return

    if hasattr(self, "_assembly_worker") and self._assembly_worker.isRunning():
        QMessageBox.information(self, "提示", "自动装配测试正在运行，请稍候。")
        return

    calibration = load_assembly_calibration()
    game_title = _game_window_title(calibration)
    try:
        window_info = activate_game_window(game_title)
    except WindowControlError as exc:
        QMessageBox.critical(self, "无法激活游戏窗口", str(exc))
        return

    target_hwnd = window_info.hwnd
    target_title = window_info.title
    self.showMinimized()

    def _run_test():
        plan = plan_test_placement(
            role_name="真红",
            piece_id="H_2",
            config_dir=_resolve_config_dir(),
        )
        logger.info(
            "装配测试规划结果: role=%s piece=%s start=(%s, %s)",
            plan.role_name,
            plan.piece_id,
            plan.start_r,
            plan.start_c,
        )

        if not is_window_foreground(target_hwnd):
            grant_foreground_permission()
            if not activate_window(target_hwnd):
                raise WindowControlError(f"无法将游戏窗口切换到前台: {target_title!r}")

        return run_assembly_drag_test(
            role_name=plan.role_name,
            piece_id=plan.piece_id,
            start_r=plan.start_r,
            start_c=plan.start_c,
            window_title=target_title,
            delay_seconds=3.0,
            calibration=calibration,
        )

    self._assembly_worker = WorkerThread(target=_run_test, parent=self)
    self._assembly_worker.result_ready.connect(self._on_assembly_test_done)
    self._assembly_worker.error.connect(self._on_assembly_test_error)
    self._assembly_worker.start()


def _on_assembly_test_done(self, result):
    self.showNormal()
    self.activateWindow()
    QMessageBox.information(
        self,
        "自动装配测试完成",
        (
            f"窗口: {result.window_title}\n"
            f"规划: {result.role_name} / {result.piece_id} → ({result.start_r}, {result.start_c})\n"
            f"拖动段数: {len(result.applied_moves)}\n\n"
            f"拖动前截图:\n{result.before_screenshot}\n\n"
            f"拖动后截图:\n{result.after_screenshot}\n\n"
            "请对比两张截图确认是否落到目标格；若偏差，请调整 config/assembly_calibration.json。"
        ),
    )


def _on_assembly_test_error(self, message: str):
    self.showNormal()
    self.activateWindow()
    if isinstance(message, str) and "ViGEmBus" in message:
        title = "虚拟手柄驱动未就绪"
    elif isinstance(message, str) and "未找到游戏窗口" in message:
        title = "未找到游戏窗口"
    else:
        title = "自动装配测试失败"
    logger.error("自动装配测试失败: %s", message)
    QMessageBox.critical(self, title, str(message))
