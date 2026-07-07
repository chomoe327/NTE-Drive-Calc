# 装配拖动过程中的视觉定位与校正。
"""Vision helpers for dragged piece localization on the assembly board."""

from __future__ import annotations

import cv2
import numpy as np

from src.features.inventory_import.equipment_classifier import match_shape_templates_in_crop
from src.scanner.config import ScannerConfig


def assembly_board_region(
    calibration: dict | None = None,
    *,
    image_width: int | None = None,
    image_height: int | None = None,
) -> tuple[int, int, int, int]:
    if image_width is None or image_height is None:
        from src.scanner.window_capture import get_foreground_client_rect

        rect = get_foreground_client_rect()
        image_width, image_height = rect.width, rect.height

    profiles = ScannerConfig.get_region_profiles(image_width, image_height)
    _, regions = profiles[0]
    return regions["assembly_board"]


def grid_dimensions(calibration: dict | None = None) -> tuple[int, int]:
    board_cfg = (calibration or {}).get("assembly_board", {}) or {}
    rows = max(1, int(board_cfg.get("grid_rows", 5) or 5))
    cols = max(1, int(board_cfg.get("grid_cols", 5) or 5))
    return rows, cols


def _detect_orange_piece_anchor(
    board_crop: np.ndarray,
    *,
    grid_rows: int,
    grid_cols: int,
) -> dict | None:
    """Fallback anchor detection using the dragged piece's orange glow."""
    if board_crop is None or board_crop.size == 0:
        return None

    board_h, board_w = board_crop.shape[:2]
    hsv = cv2.cvtColor(board_crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([5, 90, 110]), np.array([28, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(120, int(board_w * board_h * 0.004))
    max_area = int(board_w * board_h * 0.30)
    edge_margin = max(6, int(min(board_w, board_h) * 0.04))
    best = None
    best_score = -1.0

    for contour in contours:
        x, y, box_w, box_h = cv2.boundingRect(contour)
        area = box_w * box_h
        if area < min_area or area > max_area:
            continue
        if x <= edge_margin or y <= edge_margin:
            continue
        if x + box_w >= board_w - edge_margin or y + box_h >= board_h - edge_margin:
            continue
        aspect = box_w / max(box_h, 1)
        if aspect > 4.5 or aspect < 0.20:
            continue
        shape_bias = 1.0
        if 1.2 <= aspect <= 3.5:
            shape_bias = 1.25
        center_x = x + box_w / 2
        center_bias = 1.0 - abs(center_x - board_w / 2) / max(1, board_w / 2)
        score = area * shape_bias * (0.70 + 0.30 * center_bias)
        if score > best_score:
            best_score = score
            best = (x, y, box_w, box_h)

    if best is None:
        return None

    x, y, box_w, box_h = best
    cell_h = board_h / max(1, grid_rows)
    cell_w = board_w / max(1, grid_cols)
    return {
        "anchor_r": round(y / cell_h, 2),
        "anchor_c": round(x / cell_w, 2),
        "confidence": 0.75,
        "match_box": (x, y, x + box_w, y + box_h),
        "method": "orange_blob",
    }


def detect_dragged_piece_anchor(
    image_bgr: np.ndarray,
    piece_id: str,
    gold_templates: dict[str, np.ndarray],
    board_region: tuple[int, int, int, int],
    *,
    grid_rows: int = 5,
    grid_cols: int = 5,
    min_confidence: float = 0.50,
) -> dict:
    x1, y1, x2, y2 = board_region
    board_crop = image_bgr[y1:y2, x1:x2]
    if board_crop is None or board_crop.size == 0:
        return {
            "anchor_r": None,
            "anchor_c": None,
            "confidence": -1.0,
            "match_box": None,
            "method": None,
        }

    template = gold_templates.get(piece_id)
    if template is not None:
        match = match_shape_templates_in_crop(
            board_crop,
            {piece_id: template},
            min_confidence=min_confidence,
            min_margin=0.0,
        )
        if match["shape_id"] == piece_id:
            board_h, board_w = board_crop.shape[:2]
            cell_h = board_h / max(1, grid_rows)
            cell_w = board_w / max(1, grid_cols)
            top_left = match.get("match_top_left") or (0, 0)
            match_w, match_h = match.get("match_size") or (0, 0)
            return {
                "anchor_r": round(float(top_left[1]) / cell_h, 2),
                "anchor_c": round(float(top_left[0]) / cell_w, 2),
                "confidence": float(match["confidence"]),
                "match_box": (
                    x1 + top_left[0],
                    y1 + top_left[1],
                    x1 + top_left[0] + match_w,
                    y1 + top_left[1] + match_h,
                ),
                "method": "gold_template",
            }

    orange = _detect_orange_piece_anchor(
        board_crop,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
    )
    if orange is None:
        template_conf = -1.0
        if template is not None:
            fallback = match_shape_templates_in_crop(
                board_crop,
                {piece_id: template},
                min_confidence=0.0,
                min_margin=0.0,
            )
            template_conf = float(fallback.get("confidence") or -1.0)
        return {
            "anchor_r": None,
            "anchor_c": None,
            "confidence": template_conf,
            "match_box": None,
            "method": None,
        }

    match_x1, match_y1, match_x2, match_y2 = orange["match_box"]
    return {
        "anchor_r": orange["anchor_r"],
        "anchor_c": orange["anchor_c"],
        "confidence": orange["confidence"],
        "match_box": (
            x1 + match_x1,
            y1 + match_y1,
            x1 + match_x2,
            y1 + match_y2,
        ),
        "method": orange["method"],
    }


def compute_position_error(
    detected_r: float | None,
    detected_c: float | None,
    target_r: int,
    target_c: int,
) -> tuple[float, float]:
    if detected_r is None or detected_c is None:
        return 0.0, 0.0
    return float(target_r) - float(detected_r), float(target_c) - float(detected_c)


def needs_position_correction(
    delta_r: float,
    delta_c: float,
    *,
    max_cell_error: float,
) -> bool:
    return abs(delta_r) > max_cell_error or abs(delta_c) > max_cell_error


def correction_stick_for_error(
    delta_r: float,
    delta_c: float,
    correction_cfg: dict,
) -> tuple[float, float, float]:
    """Return stick vector and duration for one corrective nudge."""
    stick_x_per_col = float(correction_cfg.get("stick_x_per_col", 0.10) or 0.10)
    stick_y_per_row = float(correction_cfg.get("stick_y_per_row", 0.12) or 0.12)
    max_stick = float(correction_cfg.get("max_stick", 0.45) or 0.45)
    duration = float(correction_cfg.get("nudge_seconds", 0.10) or 0.10)

    stick_x = max(-max_stick, min(max_stick, delta_c * stick_x_per_col))
    # Gamepad Y is inverted in existing calibration: negative stick_y moves down.
    stick_y = max(-max_stick, min(max_stick, -delta_r * stick_y_per_row))
    return stick_x, stick_y, duration
