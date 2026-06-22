# 盲筛标记页业务逻辑。
"""MainWindow methods for blind marking."""

from __future__ import annotations

from dataclasses import asdict

from PySide6.QtWidgets import QMessageBox, QTableWidgetItem

from src.app import runtime
from src.app.workers import MarkingWorkerThread, WorkerThread
from src.features.discard.executor import InventoryChangedError, MarkingExecutor, MarkStepResult
from src.features.discard.log_store import MarkLogEntry, MarkLogSession, MarkLogStore
from src.features.discard.page import build_marking_page
from src.features.discard.quality_rules import (
    QualityMarkRule,
    load_rules,
    save_rules,
    validate_rules,
)
from src.features.discard.scan_session import build_session_from_inventory, load_session, save_session
from src.features.discard.scoring import MarkPreview, build_mark_preview
from src.storage.json_store import read_json
from src.ui.main_window_method_install import install_methods as _install_main_window_methods

__all__ = [
    "_page_marking",
    "_refresh_marking",
    "_marking_paths",
    "_marking_invalidate_preview",
    "_marking_on_rules_changed",
    "_marking_get_rules_from_ui",
    "_marking_set_rules_to_ui",
    "_marking_load_inventory",
    "_marking_start_full_scan",
    "_marking_on_pipeline_scan_done",
    "_marking_on_pipeline_vision_done",
    "_marking_on_pipeline_error",
    "_marking_calculate",
    "_marking_on_calculate_done",
    "_marking_on_calculate_error",
    "_marking_execute",
    "_marking_on_execute_done",
    "_marking_on_execute_error",
    "_marking_stop",
    "_marking_rollback",
    "_marking_on_rollback_done",
    "_marking_on_rollback_error",
    "_marking_update_session_status",
    "_marking_render_preview",
    "_marking_get_blueprints",
]


def install_methods(app_module, window_cls):
    _install_main_window_methods(app_module, window_cls, __all__, globals())


def _marking_get_blueprints(self):
    if getattr(self, "_marking_blueprint_cache", None):
        return self._marking_blueprint_cache
    from src.solver.orchestrator import NTEPipelineOrchestrator

    orchestrator = NTEPipelineOrchestrator(config_dir=str(runtime.CONFIG_DIR))
    roles = list(orchestrator.roles_db.keys())
    blueprints = orchestrator.solve_blueprints(roles)
    self._marking_blueprint_cache = (orchestrator, blueprints)
    return self._marking_blueprint_cache


def _marking_paths(self):
    user_dir = runtime.USER_CONFIG_DIR
    return {
        "session": user_dir / "marking_scan_session.json",
        "log": user_dir / "marking_log.json",
        "rules": user_dir / "marking_rules.json",
        "macros": user_dir / "marking_macros.json",
        "templates": runtime.CONFIG_DIR / "templates" / "marking",
    }


def _page_marking(self):
    return build_marking_page(self)


def _refresh_marking(self):
    if not hasattr(self, "marking_role_selector"):
        return
    self._marking_blueprint_cache = None
    self.marking_role_selector.load_roles(getattr(self, "roles_db", {}) or {})
    self._marking_set_rules_to_ui(load_rules(_marking_paths(self)["rules"]))
    session = load_session(_marking_paths(self)["session"])
    self._marking_session = session
    self._marking_update_session_status()
    self._marking_invalidate_preview()


def _marking_set_rules_to_ui(self, rules: dict[str, QualityMarkRule]):
    widgets = getattr(self, "_marking_rule_widgets", {}) or {}
    for quality, rule in rules.items():
        row = widgets.get(quality)
        if not row:
            continue
        discard_combo = row["discard_grade"]
        lock_combo = row["lock_grade"]
        discard_idx = discard_combo.findText(rule.discard_grade)
        lock_idx = lock_combo.findText(rule.lock_grade)
        if discard_idx >= 0:
            discard_combo.setCurrentIndex(discard_idx)
        if lock_idx >= 0:
            lock_combo.setCurrentIndex(lock_idx)
        row["discard_below_enabled"].setChecked(rule.discard_below_enabled)
        row["lock_above_enabled"].setChecked(rule.lock_above_enabled)
    self._marking_on_rules_changed()


def _marking_get_rules_from_ui(self) -> dict[str, QualityMarkRule]:
    widgets = getattr(self, "_marking_rule_widgets", {}) or {}
    rules: dict[str, QualityMarkRule] = {}
    for quality in ("Gold", "Purple"):
        row = widgets.get(quality, {})
        rules[quality] = QualityMarkRule(
            quality=quality,  # type: ignore[arg-type]
            discard_grade=str(row["discard_grade"].currentText() or "B"),
            lock_grade=str(row["lock_grade"].currentText() or "SS"),
            discard_below_enabled=bool(row["discard_below_enabled"].isChecked()),
            lock_above_enabled=bool(row["lock_above_enabled"].isChecked()),
        )
    return rules


def _marking_on_rules_changed(self):
    rules = self._marking_get_rules_from_ui()
    err = validate_rules(rules)
    if hasattr(self, "marking_rule_hint"):
        self.marking_rule_hint.setText(err or "")
    enabled = err is None
    if hasattr(self, "marking_calc_btn"):
        self.marking_calc_btn.setEnabled(enabled and self._marking_session is not None)
    self._marking_invalidate_preview()


def _marking_invalidate_preview(self):
    self._marking_preview = None
    self._marking_cached_targets = []
    if hasattr(self, "marking_execute_btn"):
        self.marking_execute_btn.setEnabled(False)
    if hasattr(self, "marking_preview_table"):
        self.marking_preview_table.setVisible(False)
        self.marking_preview_table.setRowCount(0)
    if hasattr(self, "marking_preview_summary"):
        if self._marking_session is None:
            self.marking_preview_summary.setText("请先加载数据并点击「计算」。")
        else:
            self.marking_preview_summary.setText("规则或角色已变更，请重新点击「计算」。")


def _marking_update_session_status(self):
    session = getattr(self, "_marking_session", None)
    if not hasattr(self, "marking_session_status"):
        return
    if session is None:
        self.marking_session_status.setText("尚未加载扫描会话")
        self.marking_session_status.setStyleSheet("color:#d2991d;font-size:12px;border:none")
        return
    self.marking_session_status.setText(
        f"会话就绪：共 {session.total_drives} 格（驱动/卡带），快照 ID {session.session_id[:8]}"
    )
    self.marking_session_status.setStyleSheet("color:#3fb950;font-size:12px;border:none")
    self._marking_on_rules_changed()


def _marking_load_inventory(self):
    paths = _marking_paths(self)
    if not runtime.OUTPUT_FILE.exists():
        QMessageBox.warning(self, "无法读取", "库存文件不存在，请先完成全量扫描。")
        return
    try:
        inventory = read_json(runtime.OUTPUT_FILE, default=[]) or []
        if not isinstance(inventory, list):
            raise ValueError("库存文件格式异常")
        session = build_session_from_inventory(inventory)
        save_session(paths["session"], session)
        save_rules(paths["rules"], self._marking_get_rules_from_ui())
        self._marking_session = session
        self._marking_inventory = inventory
        self._marking_update_session_status()
        self._marking_invalidate_preview()
        QMessageBox.information(self, "读取完成", f"已加载 {session.total_drives} 格装备快照（驱动/卡带）。")
    except Exception as exc:
        QMessageBox.critical(self, "读取失败", str(exc))


def _marking_start_full_scan(self):
    raw_count = self.marking_total_count_edit.text().strip()
    if not raw_count:
        QMessageBox.warning(self, "提示", "全量扫描前请先填写库存数量。")
        return
    total_drives = int(raw_count)
    if not 0 < total_drives <= 2000:
        QMessageBox.warning(self, "提示", "库存数量必须在 1-2000 之间。")
        return
    save_rules(_marking_paths(self)["rules"], self._marking_get_rules_from_ui())
    self._marking_pipeline = True
    self._replace_inventory_on_next_parse = True
    self._pending_scan_mode = "gamepad"
    QMessageBox.information(
        self,
        "全量扫描准备",
        "点击 OK 后程序会最小化并开始全量扫描。\n\n"
        "请切换至游戏的驱动/卡带仓库页面，并确保当前选中第一排第一个格子。",
    )
    self.showMinimized()
    from src.app.workers import GamepadScanWorkerThread

    self._gamepad_worker = GamepadScanWorkerThread(total_drives=total_drives, parent=self)
    self._gamepad_worker.scan_done.connect(self._marking_on_pipeline_scan_done)
    self._gamepad_worker.error.connect(self._marking_on_pipeline_error)
    self._register_scan_hotkeys("gamepad")
    self.marking_start_scan_btn.setEnabled(False)
    self.marking_start_scan_btn.setText("扫描中... (F12 停止)")
    self._gamepad_worker.start()


def _marking_on_pipeline_scan_done(self, count):
    if not getattr(self, "_marking_pipeline", False):
        return
    if count <= 0:
        self._marking_pipeline = False
        self._unregister_scan_hotkeys()
        self.showNormal()
        self.activateWindow()
        self.marking_start_scan_btn.setEnabled(True)
        self.marking_start_scan_btn.setText("开始全量扫描")
        QMessageBox.information(self, "扫描完成", "未捕获到新装备，无需解析。")
        return
    self._on_scan_done(count)


def _marking_on_pipeline_vision_done(self, stats):
    if not getattr(self, "_marking_pipeline", False):
        return False
    stats = stats or {}
    self._marking_pipeline = False
    if hasattr(self, "_progress_dlg") and self._progress_dlg:
        self._progress_dlg.close()
    post = self._postprocess_vision_files(stats)
    success_count = int(stats.get("success_count", 0) or 0)
    failed_count = int(stats.get("failed_count", 0) or 0)
    duplicate_count = int(stats.get("duplicate_count", 0) or 0) + int(post.get("probe_duplicates", 0) or 0)
    summary = f"解析成功 {success_count} 张，解析失败 {failed_count} 张，过滤重复 {duplicate_count} 张。"
    try:
        inventory = read_json(runtime.OUTPUT_FILE, default=[]) or []
        session = build_session_from_inventory(inventory)
        save_session(_marking_paths(self)["session"], session)
        self._marking_session = session
        self._marking_inventory = inventory
        self._marking_update_session_status()
        self._marking_invalidate_preview()
    except Exception as exc:
        QMessageBox.critical(self, "快照建立失败", str(exc))
        return True
    self.marking_start_scan_btn.setEnabled(True)
    self.marking_start_scan_btn.setText("开始全量扫描")
    QMessageBox.information(self, "全量扫描完成", summary + "\n\n扫描会话已建立，可继续配置规则并计算。")
    return True


def _marking_on_pipeline_error(self, err):
    if not getattr(self, "_marking_pipeline", False):
        return
    self._marking_pipeline = False
    self._unregister_scan_hotkeys()
    self.showNormal()
    self.activateWindow()
    self.marking_start_scan_btn.setEnabled(True)
    self.marking_start_scan_btn.setText("开始全量扫描")
    QMessageBox.critical(self, "扫描失败", str(err))


def _marking_calculate(self):
    if self._marking_session is None:
        QMessageBox.warning(self, "提示", "请先读取库存或完成全量扫描。")
        return
    rules = self._marking_get_rules_from_ui()
    err = validate_rules(rules)
    if err:
        QMessageBox.warning(self, "规则无效", err)
        return
    selected = self.marking_role_selector.get_selected()
    if not selected:
        QMessageBox.warning(self, "提示", "请至少选择一个评分角色。")
        return
    save_rules(_marking_paths(self)["rules"], rules)
    self.marking_calc_btn.setEnabled(False)
    self.marking_calc_btn.setText("计算中...")

    def _run():
        inventory = self._marking_inventory
        if not inventory:
            inventory = read_json(runtime.OUTPUT_FILE, default=[]) or []
        orchestrator, blueprints = self._marking_get_blueprints()
        return build_mark_preview(
            self._marking_session,
            selected,
            rules,
            self.scoring_engine,
            inventory,
            orchestrator,
            blueprints,
        )

    self._marking_calc_worker = WorkerThread(target=_run, parent=self)
    self._marking_calc_worker.result_ready.connect(self._marking_on_calculate_done)
    self._marking_calc_worker.error.connect(self._marking_on_calculate_error)
    self._marking_calc_worker.start()


def _marking_render_preview(self, preview: MarkPreview):
    from src.features.discard.scoring import ITEM_LABELS

    self.marking_preview_summary.setText(
        f"将标记弃置 {preview.discard_count} 个，将标记上锁 {preview.lock_count} 个，"
        f"蓝色跳过 {preview.blue_skipped} 个，无可用角色 {preview.no_usable_role} 个。\n"
        "执行时将逐格校验背包是否与快照一致。"
    )
    table = self.marking_preview_table
    table.setRowCount(len(preview.targets))
    action_labels = {"discard": "弃置", "lock": "上锁"}
    for row, target in enumerate(preview.targets):
        table.setItem(row, 0, QTableWidgetItem(str(target.scan_index)))
        table.setItem(row, 1, QTableWidgetItem(ITEM_LABELS.get(target.item_type, target.item_type)))
        table.setItem(row, 2, QTableWidgetItem(target.quality))
        table.setItem(row, 3, QTableWidgetItem(target.max_grade))
        table.setItem(row, 4, QTableWidgetItem(target.best_role or "-"))
        table.setItem(row, 5, QTableWidgetItem(action_labels.get(target.action, target.action)))
    table.setVisible(bool(preview.targets))
    self.marking_execute_btn.setEnabled(bool(preview.targets))


def _marking_on_calculate_done(self, preview: MarkPreview):
    self.marking_calc_btn.setEnabled(True)
    self.marking_calc_btn.setText("计算")
    self._marking_preview = preview
    self._marking_cached_targets = list(preview.targets)
    self._marking_render_preview(preview)


def _marking_on_calculate_error(self, err):
    self.marking_calc_btn.setEnabled(True)
    self.marking_calc_btn.setText("计算")
    QMessageBox.critical(self, "计算失败", str(err))


def _marking_execute(self):
    if not self._marking_cached_targets:
        QMessageBox.warning(self, "提示", "请先计算并确认存在待标记项。")
        return
    delay = float(self.marking_delay_edit.text() or "3")
    paths = _marking_paths(self)
    self.marking_execute_btn.setEnabled(False)
    self.marking_calc_btn.setEnabled(False)
    self.marking_stop_btn.setEnabled(True)

    def _run(worker: MarkingWorkerThread):
        executor = MarkingExecutor(
            self._marking_session,
            template_dir=paths["templates"],
            config_dir=str(runtime.CONFIG_DIR),
            macros_path=paths["macros"],
        )
        worker.executor = executor

        def on_progress(current, total, result: MarkStepResult):
            self.marking_preview_summary.setText(
                f"执行中 ({current}/{total})：#{result.scan_index} {result.action} -> {result.status}"
            )

        return executor.execute_targets(
            self._marking_cached_targets,
            switch_delay=delay,
            on_progress=on_progress,
        )

    self._marking_exec_worker = MarkingWorkerThread(target=_run, parent=self)
    self._marking_exec_worker.finished_ok.connect(self._marking_on_execute_done)
    self._marking_exec_worker.error.connect(self._marking_on_execute_error)
    self.showMinimized()
    self._register_scan_hotkeys("marking")
    self._marking_exec_worker.start()


def _marking_on_execute_done(self, results):
    self._unregister_scan_hotkeys()
    self.showNormal()
    self.activateWindow()
    self.marking_stop_btn.setEnabled(False)
    self.marking_calc_btn.setEnabled(True)
    self.marking_execute_btn.setEnabled(bool(self._marking_cached_targets))
    paths = _marking_paths(self)
    session = MarkLogSession(
        id=str(__import__("uuid").uuid4()),
        scan_session_id=self._marking_session.session_id,
        created_at=self._marking_session.created_at,
        rules={q: asdict(r) for q, r in self._marking_get_rules_from_ui().items()},
        roles=self.marking_role_selector.get_selected(),
        entries=[],
    )
    for result in results:
        if result.status in ("marked", "skipped_already_marked"):
            session.entries.append(
                MarkLogEntry(
                    scan_index=result.scan_index,
                    uid=next((t.uid for t in self._marking_cached_targets if t.scan_index == result.scan_index), ""),
                    action=result.action,
                    status=result.status,  # type: ignore[arg-type]
                    marked_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
                    if result.status == "marked"
                    else None,
                )
            )
    MarkLogStore(paths["log"]).append_session(session)
    marked = sum(1 for r in results if r.status == "marked")
    skipped = sum(1 for r in results if r.status == "skipped_already_marked")
    aborted = next((r for r in results if r.status == "aborted"), None)
    if aborted:
        QMessageBox.critical(self, "执行终止", aborted.message or "背包已变动，任务已终止。")
        return
    QMessageBox.information(self, "执行完成", f"成功标记 {marked} 个，跳过已标记 {skipped} 个。")


def _marking_on_execute_error(self, err):
    self._unregister_scan_hotkeys()
    self.showNormal()
    self.activateWindow()
    self.marking_stop_btn.setEnabled(False)
    self.marking_calc_btn.setEnabled(True)
    self.marking_execute_btn.setEnabled(bool(self._marking_cached_targets))
    if isinstance(err, InventoryChangedError) or "背包" in str(err):
        QMessageBox.critical(self, "执行终止", str(err))
    else:
        QMessageBox.critical(self, "执行失败", str(err))


def _marking_stop(self):
    for attr in ("_marking_exec_worker", "_marking_rollback_worker"):
        worker = getattr(self, attr, None)
        if worker:
            worker.request_stop()
    w = getattr(self, "_gamepad_worker", None)
    if w and w.scanner:
        w.scanner._stopped = True


def _marking_rollback(self):
    paths = _marking_paths(self)
    store = MarkLogStore(paths["log"])
    session = store.latest_session()
    if session is None:
        QMessageBox.information(self, "回滚", "没有可回滚的标记记录。")
        return
    candidates = store.rollback_candidates(session)
    if not candidates:
        QMessageBox.information(self, "回滚", "最近一次执行没有成功标记的条目。")
        return
    if self._marking_session is None:
        QMessageBox.warning(self, "回滚", "请先加载与当时一致的扫描会话。")
        return
    delay = float(self.marking_delay_edit.text() or "3")

    def _run(worker: MarkingWorkerThread):
        executor = MarkingExecutor(
            self._marking_session,
            template_dir=paths["templates"],
            config_dir=str(runtime.CONFIG_DIR),
            macros_path=paths["macros"],
        )
        worker.executor = executor
        return executor.rollback_entries(candidates, switch_delay=delay)

    self.marking_rollback_btn.setEnabled(False)
    self.marking_stop_btn.setEnabled(True)
    self._marking_rollback_worker = MarkingWorkerThread(target=_run, parent=self)
    self._marking_rollback_worker.finished_ok.connect(lambda results: self._marking_on_rollback_done(session, store, results))
    self._marking_rollback_worker.error.connect(self._marking_on_rollback_error)
    self.showMinimized()
    self._register_scan_hotkeys("marking")
    self._marking_rollback_worker.start()


def _marking_on_rollback_done(self, session, store, results):
    self._unregister_scan_hotkeys()
    self.showNormal()
    self.activateWindow()
    self.marking_rollback_btn.setEnabled(True)
    self.marking_stop_btn.setEnabled(False)
    for result in results:
        for entry in session.entries:
            if entry.scan_index == result.scan_index and entry.action == result.action and result.status == "unmarked":
                entry.status = "unmarked"
    store.update_session(session)
    count = sum(1 for r in results if r.status == "unmarked")
    QMessageBox.information(self, "回滚完成", f"已取消标记 {count} 个。")


def _marking_on_rollback_error(self, err):
    self._unregister_scan_hotkeys()
    self.showNormal()
    self.activateWindow()
    self.marking_rollback_btn.setEnabled(True)
    self.marking_stop_btn.setEnabled(False)
    QMessageBox.critical(self, "回滚失败", str(err))
