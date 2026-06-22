# 盲筛标记页 UI 构建。
"""Marking page UI builder."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.domain.grade_scoring import UI_GRADE_OPTIONS
from src.features.discard.role_multi_selector import RoleMultiSelector


def _quality_row(window, quality: str, label: str):
    row = QWidget()
    layout = QVBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)

    title = QLabel(label)
    title.setStyleSheet("font-weight:600;color:#c9d1d9;border:none")
    layout.addWidget(title)

    controls = QHBoxLayout()
    controls.setSpacing(10)
    discard_combo = QComboBox()
    discard_combo.addItems(list(UI_GRADE_OPTIONS))
    discard_combo.setMaximumWidth(80)
    lock_combo = QComboBox()
    lock_combo.addItems(list(UI_GRADE_OPTIONS))
    lock_combo.setMaximumWidth(80)
    discard_cb = QCheckBox("低于弃置等级 → 标记弃置")
    lock_cb = QCheckBox("高于等于上锁等级 → 标记上锁")
    controls.addWidget(QLabel("弃置等级"))
    controls.addWidget(discard_combo)
    controls.addWidget(discard_cb)
    controls.addSpacing(12)
    controls.addWidget(QLabel("上锁等级"))
    controls.addWidget(lock_combo)
    controls.addWidget(lock_cb)
    controls.addStretch()
    layout.addLayout(controls)

    window._marking_rule_widgets[quality] = {
        "discard_grade": discard_combo,
        "lock_grade": lock_combo,
        "discard_below_enabled": discard_cb,
        "lock_above_enabled": lock_cb,
    }
    return row


def build_marking_page(window):
    page = QWidget()
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(page)
    layout = QVBoxLayout(page)
    layout.setContentsMargins(20, 16, 20, 16)
    layout.setSpacing(12)

    source_card = window._card("第一步 · 数据来源")
    window.marking_source_group = QButtonGroup(window)
    source_layout = QVBoxLayout()
    source_layout.setSpacing(8)
    for value, text in [("inventory", "直接读库存"), ("full_scan", "全量手柄扫描")]:
        rb = QRadioButton(text)
        rb.setChecked(value == "inventory")
        rb.setProperty("source_key", value)
        window.marking_source_group.addButton(rb)
        source_layout.addWidget(rb)
    window.marking_full_scan_frame = QWidget()
    full_scan_layout = QHBoxLayout(window.marking_full_scan_frame)
    full_scan_layout.setContentsMargins(24, 0, 0, 0)
    full_scan_layout.addWidget(QLabel("仓库总格数:"))
    window.marking_total_count_edit = QLineEdit()
    window.marking_total_count_edit.setPlaceholderText("驱动+卡带合计格数")
    window.marking_total_count_edit.setValidator(QIntValidator(1, 2000, window.marking_total_count_edit))
    window.marking_total_count_edit.setMaximumWidth(180)
    full_scan_layout.addWidget(window.marking_total_count_edit)
    window.marking_start_scan_btn = QPushButton("开始全量扫描")
    window.marking_start_scan_btn.setObjectName("btnAction")
    window.marking_start_scan_btn.clicked.connect(window._marking_start_full_scan)
    full_scan_layout.addWidget(window.marking_start_scan_btn)
    full_scan_layout.addStretch()
    source_layout.addWidget(window.marking_full_scan_frame)
    window.marking_load_inventory_btn = QPushButton("读取库存并建立快照")
    window.marking_load_inventory_btn.setObjectName("btnAction")
    window.marking_load_inventory_btn.clicked.connect(window._marking_load_inventory)
    source_layout.addWidget(window.marking_load_inventory_btn)
    window.marking_session_status = QLabel("尚未加载扫描会话")
    window.marking_session_status.setStyleSheet("color:#8b949e;font-size:12px;border:none")
    source_layout.addWidget(window.marking_session_status)
    source_card.layout().addLayout(source_layout)
    layout.addWidget(source_card)

    role_card = window._card("第二步 · 评分角色")
    window.marking_role_selector = RoleMultiSelector()
    window.marking_role_selector.selectionChanged.connect(window._marking_invalidate_preview)
    role_card.layout().addWidget(window.marking_role_selector)
    layout.addWidget(role_card)

    rule_card = window._card("第三步 · 规则与执行")
    window._marking_rule_widgets = {}
    rule_card.layout().addWidget(_quality_row(window, "Gold", "金色"))
    rule_card.layout().addWidget(_quality_row(window, "Purple", "紫色"))
    window.marking_rule_hint = QLabel("")
    window.marking_rule_hint.setStyleSheet("color:#f85149;font-size:12px;border:none")
    rule_card.layout().addWidget(window.marking_rule_hint)

    delay_row = QHBoxLayout()
    delay_row.addWidget(QLabel("切回游戏延迟(秒):"))
    window.marking_delay_edit = QLineEdit("3")
    window.marking_delay_edit.setValidator(QIntValidator(0, 120, window.marking_delay_edit))
    window.marking_delay_edit.setMaximumWidth(80)
    delay_row.addWidget(window.marking_delay_edit)
    delay_row.addStretch()
    rule_card.layout().addLayout(delay_row)

    action_row = QHBoxLayout()
    window.marking_calc_btn = QPushButton("计算")
    window.marking_calc_btn.setObjectName("btnAction")
    window.marking_calc_btn.clicked.connect(window._marking_calculate)
    window.marking_execute_btn = QPushButton("执行标记")
    window.marking_execute_btn.setObjectName("btnPrimary")
    window.marking_execute_btn.setEnabled(False)
    window.marking_execute_btn.clicked.connect(window._marking_execute)
    window.marking_stop_btn = QPushButton("停止")
    window.marking_stop_btn.setObjectName("btnDanger")
    window.marking_stop_btn.setEnabled(False)
    window.marking_stop_btn.clicked.connect(window._marking_stop)
    window.marking_rollback_btn = QPushButton("回滚上次")
    window.marking_rollback_btn.setObjectName("btnAction")
    window.marking_rollback_btn.clicked.connect(window._marking_rollback)
    action_row.addWidget(window.marking_calc_btn)
    action_row.addWidget(window.marking_execute_btn)
    action_row.addWidget(window.marking_stop_btn)
    action_row.addWidget(window.marking_rollback_btn)
    action_row.addStretch()
    rule_card.layout().addLayout(action_row)

    window.marking_preview_summary = QLabel("请先加载数据并点击「计算」。")
    window.marking_preview_summary.setStyleSheet("color:#8b949e;font-size:12px;border:none")
    window.marking_preview_summary.setWordWrap(True)
    rule_card.layout().addWidget(window.marking_preview_summary)

    window.marking_preview_table = QTableWidget(0, 6)
    window.marking_preview_table.setHorizontalHeaderLabels(
        ["格子", "类型", "品质", "最高等级", "最佳角色", "动作"]
    )
    window.marking_preview_table.horizontalHeader().setStretchLastSection(True)
    window.marking_preview_table.setEditTriggers(QTableWidget.NoEditTriggers)
    window.marking_preview_table.setSelectionBehavior(QTableWidget.SelectRows)
    window.marking_preview_table.setVisible(False)
    rule_card.layout().addWidget(window.marking_preview_table)
    layout.addWidget(rule_card)

    def _on_source_changed():
        checked = window.marking_source_group.checkedButton()
        key = checked.property("source_key") if checked else "inventory"
        window.marking_full_scan_frame.setVisible(key == "full_scan")
        window.marking_load_inventory_btn.setVisible(key == "inventory")
        window._marking_invalidate_preview()

    for btn in window.marking_source_group.buttons():
        btn.toggled.connect(lambda _checked=False: _on_source_changed())
    _on_source_changed()

    for widgets in window._marking_rule_widgets.values():
        for key, widget in widgets.items():
            if key in ("discard_grade", "lock_grade"):
                widget.currentIndexChanged.connect(window._marking_on_rules_changed)
            elif hasattr(widget, "toggled"):
                widget.toggled.connect(window._marking_on_rules_changed)

    return scroll
