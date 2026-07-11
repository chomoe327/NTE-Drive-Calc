# 配装页自动装配测试入口。
"""UI controller for the assembly automation prototype test."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QMessageBox

from src.app import runtime
from src.app.workers import WorkerThread
from src.features.settings.hotkeys import load_hotkey_config
from src.scanner.assembly_hotkeys import (
    start_assembly_stop_hotkey,
    stop_assembly_stop_hotkey,
)
from src.scanner.gamepad_assembly import (
    load_assembly_calibration,
    request_assembly_stop,
    run_full_assembly_test,
)
from src.scanner.assembly_planner import (
    ASSEMBLY_BLUEPRINT_ROLE,
    ASSEMBLY_UI_ROLE,
    plan_full_assembly,
)
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


def _format_piece_plan(plan) -> str:
    lines = []
    for index, piece in enumerate(plan.pieces, 1):
        lines.append(f"  {index}. {piece.piece_id} → ({piece.start_r}, {piece.start_c})")
    return "\n".join(lines)


def _assembly_stop_hotkey(window) -> str:
    hotkeys = load_hotkey_config(runtime.USER_CONFIG_DIR)
    return str(hotkeys.get("stop") or "F12")


def _start_assembly_test(self, _role_name: str | None = None):
    if not sys.platform.startswith("win"):
        QMessageBox.warning(self, "不支持", "自动装配测试仅支持 Windows 平台。")
        return

    try:
        preview_plan = plan_full_assembly(
            role_name=ASSEMBLY_BLUEPRINT_ROLE,
            config_dir=_resolve_config_dir(),
        )
    except Exception as exc:
        QMessageBox.critical(self, "无法读取配装图纸", str(exc))
        return

    stop_hotkey = _assembly_stop_hotkey(self)
    piece_summary = _format_piece_plan(preview_plan)
    confirmed = QMessageBox.question(
        self,
        "自动装配测试",
        (
            "本阶段将按已保存图纸完整装配驱动块：\n"
            f"  图纸角色：{ASSEMBLY_BLUEPRINT_ROLE}\n"
            f"  装配界面：{ASSEMBLY_UI_ROLE}（底盘格子相同）\n"
            f"  驱动块数：{len(preview_plan.pieces)}（仅匹配形状，暂不管副词条）\n"
            f"{piece_summary}\n\n"
            "请先确保：\n"
            f"  1. 游戏已在 {ASSEMBLY_UI_ROLE} 的驱动装配页\n"
            "  2. 中间 5×5 网格为空\n"
            "  3. 左侧库存焦点在第一个驱动\n"
            f"  4. 配装页已保存 {ASSEMBLY_BLUEPRINT_ROLE} 的统筹方案\n\n"
            "将通过左下角「筛选 → 外形 → 形状」固定坐标选中目标形状（不再用库存模板搜索）。\n"
            "若目标驱动块已被其他角色装备，OCR 同时识别到「取消」「确认」后直接按 A。\n"
            f"运行中按 {stop_hotkey} 可立即停止装配。\n"
            "点击确定后程序将最小化，并切换到「异环」窗口，3 秒后接管虚拟手柄。\n"
            "调试截图将保存到 accounts/default/test/ 目录。"
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
    start_assembly_stop_hotkey(stop_hotkey, request_assembly_stop)

    def _run_test():
        plan = plan_full_assembly(
            role_name=ASSEMBLY_BLUEPRINT_ROLE,
            config_dir=_resolve_config_dir(),
        )
        piece_specs = [
            (piece.piece_id, piece.start_r, piece.start_c)
            for piece in plan.pieces
        ]
        logger.info(
            f"完整装配规划: blueprint={plan.blueprint_role} pieces={len(piece_specs)} "
            f"specs={piece_specs}"
        )

        if not is_window_foreground(target_hwnd):
            grant_foreground_permission()
            if not activate_window(target_hwnd):
                raise WindowControlError(f"无法将游戏窗口切换到前台: {target_title!r}")

        return run_full_assembly_test(
            blueprint_role=plan.blueprint_role,
            pieces=piece_specs,
            window_title=target_title,
            delay_seconds=3.0,
            calibration=calibration,
            stop_on_error=True,
        )

    self._assembly_worker = WorkerThread(target=_run_test, parent=self)
    self._assembly_worker.result_ready.connect(self._on_assembly_test_done)
    self._assembly_worker.error.connect(self._on_assembly_test_error)
    self._assembly_worker.start()


def _on_assembly_test_done(self, result):
    stop_assembly_stop_hotkey()
    self.showNormal()
    self.activateWindow()
    debug_lines = "\n".join(result.debug_screenshots) if result.debug_screenshots else "（无）"
    piece_lines = []
    for index, piece in enumerate(result.pieces, 1):
        status = "成功" if piece.success else f"失败: {piece.error}"
        piece_lines.append(
            f"  {index}. {piece.piece_id} → ({piece.start_r}, {piece.start_c}) | {status}"
        )
    piece_summary = "\n".join(piece_lines) if piece_lines else "（无）"
    failed = result.failed_pieces
    stopped = any((piece.error or "") == "用户中止装配" for piece in result.pieces)
    if stopped:
        title = "自动装配已中止"
    elif not failed:
        title = "自动装配测试完成"
    else:
        title = "自动装配测试部分失败"
    QMessageBox.information(
        self,
        title,
        (
            f"窗口: {result.window_title}\n"
            f"图纸: {result.blueprint_role}\n"
            f"成功: {result.success_count}/{len(result.pieces)}\n\n"
            f"装配明细:\n{piece_summary}\n\n"
            f"拖动前截图:\n{result.before_screenshot}\n\n"
            f"调试过程截图:\n{debug_lines}\n\n"
            f"拖动后截图:\n{result.after_screenshot}\n\n"
            "请对比截图确认位置；参数请调整 config/assembly_calibration.json。"
        ),
    )


def _on_assembly_test_error(self, message: str):
    stop_assembly_stop_hotkey()
    self.showNormal()
    self.activateWindow()
    if isinstance(message, str) and "ViGEmBus" in message:
        title = "虚拟手柄驱动未就绪"
    elif isinstance(message, str) and "未找到游戏窗口" in message:
        title = "未找到游戏窗口"
    else:
        title = "自动装配测试失败"
    logger.error(f"自动装配测试失败: {message}")
    QMessageBox.critical(self, title, str(message))
