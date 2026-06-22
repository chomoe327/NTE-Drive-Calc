# 盲筛页多选角色组件。
"""Simple multi-select role grid without priority ordering."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.ui.widgets import match_pinyin


class RoleMultiSelector(QWidget):
    selectionChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.all_roles: dict = {}
        self.selected: list[str] = []
        self._cards: dict[str, QFrame] = {}
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        search_row = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("搜索角色（支持拼音）...")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._render_grid)
        search_row.addWidget(self.search, 1)
        select_all_btn = QPushButton("全选")
        select_all_btn.setObjectName("btnAction")
        select_all_btn.clicked.connect(self.select_all)
        search_row.addWidget(select_all_btn)
        reset_btn = QPushButton("清空选择")
        reset_btn.setObjectName("btnDanger")
        reset_btn.clicked.connect(self.clear_selection)
        search_row.addWidget(reset_btn)
        layout.addLayout(search_row)

        tip = QLabel("点击角色即可多选，参与盲筛评分；无需设置优先级。")
        tip.setStyleSheet("color:#8b949e;font-size:11px;border:none")
        layout.addWidget(tip)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(220)
        self.grid_host = QWidget()
        self.grid_layout = QGridLayout(self.grid_host)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(6)
        self.grid_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        scroll.setWidget(self.grid_host)
        layout.addWidget(scroll, 1)

        self.summary_label = QLabel("已选 0 个角色")
        self.summary_label.setStyleSheet("color:#8b949e;font-size:12px;border:none")
        layout.addWidget(self.summary_label)

    _CARD_SEL = "QFrame{background:#1f6feb22;border:2px solid #58a6ff;border-radius:8px}QFrame:hover{border-color:#79c0ff}"
    _CARD_OFF = "QFrame{background:#161b22;border:1px solid #21262d;border-radius:8px}QFrame:hover{border-color:#30363d}"

    def load_roles(self, roles_db: dict):
        self.all_roles = roles_db or {}
        self._render_grid()

    def _render_grid(self):
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._cards.clear()

        query = self.search.text().strip() if hasattr(self, "search") else ""
        names = sorted(self.all_roles.keys())
        if query:
            names = [name for name in names if match_pinyin(name, query)]

        row = col = 0
        for name in names:
            self.grid_layout.addWidget(self._make_card(name), row, col)
            col += 1
            if col >= 8:
                col = 0
                row += 1
        self.summary_label.setText(f"已选 {len(self.selected)} 个角色")

    def _make_card(self, name: str) -> QFrame:
        card = QFrame()
        card.setCursor(Qt.PointingHandCursor)
        card.setFixedSize(96, 34)
        selected = name in self.selected
        card.setStyleSheet(self._CARD_SEL if selected else self._CARD_OFF)
        label = QLabel(name)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("border:none;color:#c9d1d9;font-size:12px;font-weight:600")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(label)
        card.mousePressEvent = lambda _event, role=name: self._toggle(role)
        self._cards[name] = card
        return card

    def _toggle(self, role: str):
        if role in self.selected:
            self.selected.remove(role)
        else:
            self.selected.append(role)
        self._render_grid()
        self.selectionChanged.emit()

    def clear_selection(self):
        self.selected = []
        self._render_grid()
        self.selectionChanged.emit()

    def select_all(self):
        query = self.search.text().strip() if hasattr(self, "search") else ""
        names = sorted(self.all_roles.keys())
        if query:
            names = [name for name in names if match_pinyin(name, query)]
        for name in names:
            if name not in self.selected:
                self.selected.append(name)
        self._render_grid()
        self.selectionChanged.emit()

    def get_selected(self) -> list[str]:
        return list(self.selected)
