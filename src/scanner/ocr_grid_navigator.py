# OCR 定位与网格导航。
"""OCR-driven grid navigation for blind marking."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.features.discard.scan_session import InventoryChangedError, ScanSession, signature_from_item_dict, verify_signature
from src.scanner.gamepad_controller import GamepadScanner
from src.scanner.grid_navigation import (
    _INVERT_MOVE,
    index_to_position,
    moves_between_scan_indices,
)
from src.utils.logger import logger


class AmbiguousLocationError(RuntimeError):
    """Raised when OCR cannot uniquely resolve the current grid cell."""


@dataclass
class GridPosition:
    scan_index: int
    row: int
    col: int


class OcrGridNavigator:
    MAX_PROBE_STEPS = 3
    MAX_NAV_RETRIES = 3
    PANEL_SETTLE_SEC = 0.45
    PROBE_DIRECTIONS = ("R", "L", "U", "D")

    def __init__(
        self,
        session: ScanSession,
        scanner: GamepadScanner,
        *,
        capture_bgr: Callable,
        parse_item_dict: Callable[[np.ndarray], dict],
        signature_looks_unreadable: Callable[[dict], bool],
        is_stopped: Callable[[], bool] | None = None,
    ):
        self.session = session
        self.scanner = scanner
        self._capture_bgr = capture_bgr
        self._parse_item_dict = parse_item_dict
        self._signature_looks_unreadable = signature_looks_unreadable
        self._is_stopped = is_stopped or (lambda: False)

    def _wait_panel_settle(self) -> None:
        time.sleep(self.PANEL_SETTLE_SEC)

    def _position_for_index(self, scan_index: int) -> GridPosition:
        row, col = index_to_position(scan_index, self.session.total_drives, self.session.cols)
        return GridPosition(scan_index=scan_index, row=row, col=col)

    def _ocr_signature(self, image_bgr: np.ndarray) -> str | None:
        try:
            item_dict = self._parse_item_dict(image_bgr)
            if self._signature_looks_unreadable(item_dict):
                return None
            return signature_from_item_dict(item_dict)
        except Exception:
            return None

    def _indices_for_signature(self, signature: str) -> list[int]:
        return list(self.session.signature_to_indices.get(signature, []))

    def _locate_from_image(self, image_bgr: np.ndarray) -> tuple[str | None, list[int]]:
        signature = self._ocr_signature(image_bgr)
        if signature is None:
            return None, []
        return signature, self._indices_for_signature(signature)

    def _apply_probe_moves(self, moves: list[str]) -> None:
        for move in moves:
            if self._is_stopped():
                return
            self.scanner.apply_moves_batch([move])

    def _probe_to_unique_anchor(self, sct) -> GridPosition:
        for direction in self.PROBE_DIRECTIONS:
            probe_moves: list[str] = []
            for _ in range(self.MAX_PROBE_STEPS):
                if self._is_stopped():
                    raise AmbiguousLocationError("导航已停止")
                probe_moves.append(direction)
                self._apply_probe_moves([direction])
                self._wait_panel_settle()
                image_bgr = self._capture_bgr(sct)
                signature, indices = self._locate_from_image(image_bgr)
                if signature is None:
                    continue
                if len(indices) == 1:
                    anchor_index = indices[0]
                    logger.info(
                        f"重复属性探针：沿 {direction} 移动 {len(probe_moves)} 步后定位到唯一锚点第 {anchor_index} 格"
                    )
                    return self._position_for_index(anchor_index)
            if probe_moves:
                rewind = [_INVERT_MOVE[move] for move in reversed(probe_moves)]
                logger.debug(f"探针方向 {direction} 未找到唯一锚点，退回 {len(rewind)} 步")
                self._apply_probe_moves(rewind)
                self._wait_panel_settle()
        raise AmbiguousLocationError("无法在邻近格找到唯一属性锚点，请确认背包与扫描会话一致。")

    def locate_resolved(self, sct) -> GridPosition:
        image_bgr = self._capture_bgr(sct)
        signature, indices = self._locate_from_image(image_bgr)
        if signature is None:
            raise AmbiguousLocationError("当前面板无法识别，请确认已选中装备详情。")
        if not indices:
            raise InventoryChangedError("当前格子装备不在扫描会话中，背包可能已变动。")
        if len(indices) == 1:
            return self._position_for_index(indices[0])
        logger.info(f"当前格属性重复（候选 {indices}），启动探针消歧")
        return self._probe_to_unique_anchor(sct)

    def _verify_target_signature(self, sct, target_index: int) -> bool:
        entry = self.session.entry_by_index(target_index)
        if entry is None:
            return False
        image_bgr = self._capture_bgr(sct)
        signature = self._ocr_signature(image_bgr)
        if signature is None:
            return False
        try:
            verify_signature(entry.signature, signature)
        except InventoryChangedError:
            return False
        return True

    def navigate_to(
        self,
        sct,
        target_index: int,
        *,
        start_index: int | None = None,
    ) -> GridPosition:
        for attempt in range(1, self.MAX_NAV_RETRIES + 1):
            if self._is_stopped():
                raise AmbiguousLocationError("导航已停止")

            if attempt == 1 and start_index is not None:
                current = self._position_for_index(start_index)
            else:
                current = self.locate_resolved(sct)

            if current.scan_index != target_index:
                moves = moves_between_scan_indices(
                    current.scan_index,
                    target_index,
                    self.session.total_drives,
                    self.session.cols,
                )
                logger.info(
                    f"导航 {current.scan_index} -> {target_index}：{len(moves)} 步"
                    + (f"（重试 {attempt}/{self.MAX_NAV_RETRIES}）" if attempt > 1 else "")
                )
                self.scanner.apply_moves_batch(moves)
                self._wait_panel_settle()

            if self._verify_target_signature(sct, target_index):
                return self._position_for_index(target_index)

            logger.warning(
                f"第 {target_index} 格终点校验失败，整段重试 {attempt}/{self.MAX_NAV_RETRIES}"
            )

        raise InventoryChangedError(
            f"第 {target_index} 格终点校验连续 {self.MAX_NAV_RETRIES} 次失败，背包可能已变动。"
        )
