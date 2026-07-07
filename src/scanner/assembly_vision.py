# 装配拖动过程中的视觉定位与校正。
"""Vision helpers for dragged piece localization on the assembly board."""

from __future__ import annotations

import cv2
import numpy as np

from src.features.inventory_import.equipment_classifier import match_shape_templates_in_crop
from src.scanner.config import ScannerConfig
from src.scanner.window_capture import get_foreground_client_rect


def assembly_board_region(
    calibration: dict | None = None,
    *,
    image_width: int | None = None,
    image_height: int | None = None,
) -> tuple[int, int, int, int]:
    if image_width is None or image_height is None:
        rect = get_foreground_client_rect()
        image_width, image_height = rect.width, rect.height

    override = (calibration or {}).get("assembly_board", {}) or {}
    region_2k = override.get("region_2k")
    if isinstance(region_2k, (list, tuple)) and len(region_2k) == 4:
        base = ScannerConfig.BASE_WIDTH, ScannerConfig.BASE_HEIGHT
        scale_x = image_width / base[0]
        scale_y = image_height / base[1]
        x1, y1, x2, y2 = region_2k
        return (
            int(x1 * scale_x),
            int(y1 * scale_y),
            int(x2 * scale_x),
            int(y2 * scale_y),
        )

    profiles = ScannerConfig.get_region_profiles(image_width, image_height)
    _, regions = profiles[0]
    return regions["assembly_board"]


def grid_dimensions(calibration: dict | None = None) -> tuple[int, int]:
    board_cfg = (calibration or {}).get("assembly_board", {}) or {}
    rows = max(1, int(board_cfg.get("grid_rows", 5) or 5))
    cols = max(1, int(board_cfg.get("grid_cols", 5) or 5))
    return rows, cols


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
        }

    template = gold_templates.get(piece_id)
    if template is None:
        return {
            "anchor_r": None,
            "anchor_c": None,
            "confidence": -1.0,
            "match_box": None,
        }

    match = match_shape_templates_in_crop(
        board_crop,
        {piece_id: template},
        min_confidence=min_confidence,
        min_margin=0.0,
    )
    if match["shape_id"] != piece_id:
        return {
            "anchor_r": None,
            "anchor_c": None,
            "confidence": float(match.get("confidence") or -1.0),
            "match_box": None,
        }

    board_h, board_w = board_crop.shape[:2]
    cell_h = board_h / max(1, grid_rows)
    cell_w = board_w / max(1, grid_cols)
    top_left = match.get("match_top_left") or (0, 0)
    anchor_r = float(top_left[1]) / cell_h
    anchor_c = float(top_left[0]) / cell_w
    match_w, match_h = match.get("match_size") or (0, 0)
    return {
        "anchor_r": round(anchor_r, 2),
        "anchor_c": round(anchor_c, 2),
        "confidence": float(match["confidence"]),
        "match_box": (
            x1 + top_left[0],
            y1 + top_left[1],
            x1 + top_left[0] + match_w,
            y1 + top_left[1] + match_h,
        ),
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
