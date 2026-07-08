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

    content_rect = ScannerConfig.get_content_rect(image_width, image_height)
    override = (calibration or {}).get("assembly_board", {}) or {}
    region_2k = override.get("region_2k")
    if isinstance(region_2k, (list, tuple)) and len(region_2k) == 4:
        from src.scanner.window_capture import scale_region

        return scale_region(
            tuple(int(v) for v in region_2k),
            image_width,
            image_height,
            (ScannerConfig.BASE_WIDTH, ScannerConfig.BASE_HEIGHT),
            content_rect=content_rect,
        )

    profiles = ScannerConfig.get_region_profiles(image_width, image_height)
    _, regions = profiles[0]
    return regions["assembly_board"]


def grid_dimensions(calibration: dict | None = None) -> tuple[int, int]:
    board_cfg = (calibration or {}).get("assembly_board", {}) or {}
    rows = max(1, int(board_cfg.get("grid_rows", 5) or 5))
    cols = max(1, int(board_cfg.get("grid_cols", 5) or 5))
    return rows, cols


def expand_board_region(
    board_region: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    *,
    margin_ratio: float = 0.35,
    left_margin_ratio: float | None = None,
    right_margin_ratio: float | None = None,
) -> tuple[int, int, int, int]:
    """Expand the board crop so dragged pieces slightly outside the grid remain visible."""
    x1, y1, x2, y2 = board_region
    board_w = max(1, x2 - x1)
    board_h = max(1, y2 - y1)
    left_ratio = float(left_margin_ratio if left_margin_ratio is not None else margin_ratio)
    right_ratio = float(right_margin_ratio if right_margin_ratio is not None else margin_ratio * 0.35)
    pad_top = int(board_h * margin_ratio)
    pad_bottom = int(board_h * margin_ratio)
    pad_left = int(board_w * left_ratio)
    pad_right = int(board_w * right_ratio)
    return (
        max(0, x1 - pad_left),
        max(0, y1 - pad_top),
        min(image_width, x2 + pad_right),
        min(image_height, y2 + pad_bottom),
    )


def position_error_magnitude(delta_r: float, delta_c: float) -> float:
    return max(abs(float(delta_r)), abs(float(delta_c)))


def is_detection_outlier(
    detected_r: float,
    detected_c: float,
    reference_r: float,
    reference_c: float,
    *,
    max_cell_jump: float,
) -> bool:
    return (
        abs(float(detected_r) - float(reference_r)) > max_cell_jump
        or abs(float(detected_c) - float(reference_c)) > max_cell_jump
    )


def _score_orange_contour(
    contour,
    *,
    board_w: int,
    board_h: int,
    min_solidity: float,
    target_r: float | None,
    target_c: float | None,
    cell_w: float,
    cell_h: float,
    grid_offset_x: float = 0.0,
    grid_offset_y: float = 0.0,
) -> float | None:
    x, y, box_w, box_h = cv2.boundingRect(contour)
    area = box_w * box_h
    min_area = max(120, int(board_w * board_h * 0.004))
    max_area = int(board_w * board_h * 0.28)
    if area < min_area or area > max_area:
        return None

    contour_area = float(cv2.contourArea(contour))
    if contour_area <= 0:
        return None
    solidity = contour_area / max(float(area), 1.0)
    if solidity < min_solidity:
        return None

    aspect = box_w / max(box_h, 1)
    if aspect > 4.5 or aspect < 0.18:
        return None

    score = contour_area * (0.75 + 0.25 * min(1.0, solidity))
    if 1.0 <= aspect <= 3.5:
        score *= 1.15

    if target_r is not None and target_c is not None and cell_w > 0 and cell_h > 0:
        anchor_r = (y - grid_offset_y) / cell_h
        anchor_c = (x - grid_offset_x) / cell_w
        distance = ((anchor_r - float(target_r)) ** 2 + (anchor_c - float(target_c)) ** 2) ** 0.5
        score *= max(0.35, 1.8 - distance * 0.55)
        if float(target_c) <= 1.0 and anchor_c <= float(target_c) + 0.75:
            score *= 1.35
    else:
        center_x = x + box_w / 2
        center_bias = 1.0 - abs(center_x - board_w / 2) / max(1, board_w / 2)
        score *= 0.70 + 0.30 * center_bias

    return score


def _detect_orange_piece_anchor(
    board_crop: np.ndarray,
    *,
    grid_rows: int,
    grid_cols: int,
    target_r: float | None = None,
    target_c: float | None = None,
    min_solidity: float = 0.42,
    cell_w: float | None = None,
    cell_h: float | None = None,
    grid_offset_x: float = 0.0,
    grid_offset_y: float = 0.0,
) -> dict | None:
    """Fallback anchor detection using the dragged piece's orange glow."""
    if board_crop is None or board_crop.size == 0:
        return None

    board_h, board_w = board_crop.shape[:2]
    resolved_cell_h = float(cell_h or (board_h / max(1, grid_rows)))
    resolved_cell_w = float(cell_w or (board_w / max(1, grid_cols)))
    hsv = cv2.cvtColor(board_crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([8, 120, 140]), np.array([24, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = -1.0

    for contour in contours:
        score = _score_orange_contour(
            contour,
            board_w=board_w,
            board_h=board_h,
            min_solidity=min_solidity,
            target_r=target_r,
            target_c=target_c,
            cell_w=resolved_cell_w,
            cell_h=resolved_cell_h,
            grid_offset_x=grid_offset_x,
            grid_offset_y=grid_offset_y,
        )
        if score is None or score <= best_score:
            continue
        x, y, box_w, box_h = cv2.boundingRect(contour)
        best_score = score
        best = (x, y, box_w, box_h)

    if best is None:
        relaxed_mask = cv2.inRange(hsv, np.array([6, 90, 120]), np.array([28, 255, 255]))
        relaxed_mask = cv2.morphologyEx(relaxed_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(relaxed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            score = _score_orange_contour(
                contour,
                board_w=board_w,
                board_h=board_h,
                min_solidity=max(0.30, min_solidity - 0.08),
                target_r=target_r,
                target_c=target_c,
                cell_w=resolved_cell_w,
                cell_h=resolved_cell_h,
                grid_offset_x=grid_offset_x,
                grid_offset_y=grid_offset_y,
            )
            if score is None or score <= best_score:
                continue
            x, y, box_w, box_h = cv2.boundingRect(contour)
            best_score = score
            best = (x, y, box_w, box_h)

    if best is None:
        return None

    x, y, box_w, box_h = best
    return {
        "anchor_r": round((y - grid_offset_y) / resolved_cell_h, 2),
        "anchor_c": round((x - grid_offset_x) / resolved_cell_w, 2),
        "confidence": min(0.95, 0.55 + best_score / max(board_w * board_h, 1) * 8.0),
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
    target_r: float | None = None,
    target_c: float | None = None,
    detection_region: tuple[int, int, int, int] | None = None,
    min_orange_solidity: float = 0.42,
) -> dict:
    bx1, by1, bx2, by2 = board_region
    board_w = max(1, bx2 - bx1)
    board_h = max(1, by2 - by1)
    cell_h = board_h / max(1, grid_rows)
    cell_w = board_w / max(1, grid_cols)

    detect_x1, detect_y1, detect_x2, detect_y2 = detection_region or board_region
    detect_crop = image_bgr[detect_y1:detect_y2, detect_x1:detect_x2]
    if detect_crop is None or detect_crop.size == 0:
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
            detect_crop,
            {piece_id: template},
            min_confidence=min_confidence,
            min_margin=0.0,
        )
        if match["shape_id"] == piece_id:
            top_left = match.get("match_top_left") or (0, 0)
            match_w, match_h = match.get("match_size") or (0, 0)
            abs_x = detect_x1 + top_left[0]
            abs_y = detect_y1 + top_left[1]
            return {
                "anchor_r": round((abs_y - by1) / cell_h, 2),
                "anchor_c": round((abs_x - bx1) / cell_w, 2),
                "confidence": float(match["confidence"]),
                "match_box": (
                    abs_x,
                    abs_y,
                    abs_x + match_w,
                    abs_y + match_h,
                ),
                "method": "gold_template",
            }

    orange = _detect_orange_piece_anchor(
        detect_crop,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        target_r=target_r,
        target_c=target_c,
        min_solidity=min_orange_solidity,
        cell_w=cell_w,
        cell_h=cell_h,
        grid_offset_x=float(bx1 - detect_x1),
        grid_offset_y=float(by1 - detect_y1),
    )
    if orange is None:
        template_conf = -1.0
        if template is not None:
            fallback = match_shape_templates_in_crop(
                detect_crop,
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
    abs_x1 = detect_x1 + match_x1
    abs_y1 = detect_y1 + match_y1
    abs_x2 = detect_x1 + match_x2
    abs_y2 = detect_y1 + match_y2
    return {
        "anchor_r": round((abs_y1 - by1) / cell_h, 2),
        "anchor_c": round((abs_x1 - bx1) / cell_w, 2),
        "confidence": orange["confidence"],
        "match_box": (abs_x1, abs_y1, abs_x2, abs_y2),
        "method": orange["method"],
    }


def compute_position_error(
    detected_r: float | None,
    detected_c: float | None,
    target_r: int,
    target_c: int,
) -> tuple[float, float] | None:
    if detected_r is None or detected_c is None:
        return None
    return float(target_r) - float(detected_r), float(target_c) - float(detected_c)


def estimate_position_error_from_origin(
    target_r: int,
    target_c: int,
    *,
    origin_r: int,
    origin_c: int,
) -> tuple[float, float]:
    """Fallback error estimate when vision cannot detect the dragged piece."""
    return float(target_r) - float(origin_r), float(target_c) - float(origin_c)


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
    stick_scale = float(correction_cfg.get("correction_stick_scale", 1.0) or 1.0)
    max_cell_error = float(correction_cfg.get("max_cell_error", 0.45) or 0.45)

    effective_delta_r = 0.0 if abs(delta_r) <= max_cell_error else delta_r
    effective_delta_c = 0.0 if abs(delta_c) <= max_cell_error else delta_c

    stick_x = max(-max_stick, min(max_stick, effective_delta_c * stick_x_per_col * stick_scale))
    # Gamepad Y is inverted in existing calibration: negative stick_y moves down.
    stick_y = max(-max_stick, min(max_stick, -effective_delta_r * stick_y_per_row * stick_scale))
    return stick_x, stick_y, duration
