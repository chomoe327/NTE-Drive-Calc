# 判断截图对应的装备类型。
"""Helpers for classifying screenshots as drive or tape."""

from __future__ import annotations

import json
import os
import time

import cv2
import numpy as np

from src.utils.logger import logger
from src.utils.perf import log_perf


HIGH_CONFIDENCE_DRIVE_SHAPE = 0.95
INVENTORY_SLOT_MIN_CONFIDENCE = 0.65
INVENTORY_SLOT_MIN_MARGIN = 0.05


def match_shape_templates_in_crop(
    crop_bgr: np.ndarray,
    templates: dict[str, np.ndarray],
    *,
    min_confidence: float = INVENTORY_SLOT_MIN_CONFIDENCE,
    high_confidence: float = HIGH_CONFIDENCE_DRIVE_SHAPE,
    min_margin: float = INVENTORY_SLOT_MIN_MARGIN,
    scale_hint: tuple[int, int] | None = None,
) -> dict:
    if crop_bgr is None or crop_bgr.size == 0 or not templates:
        return {
            "shape_id": "Unknown",
            "confidence": -1.0,
            "second_best_confidence": -1.0,
            "margin": 0.0,
            "match_top_left": None,
            "match_size": None,
        }

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(crop_bgr.shape) == 3 else crop_bgr
    crop_h, crop_w = gray.shape[:2]
    if scale_hint is None:
        sample = next(iter(templates.values()))
        scale_hint = sample.shape[1], sample.shape[0]
    base_scale = max(min(crop_w / max(1, scale_hint[0]), crop_h / max(1, scale_hint[1])), 0.12)
    scales = sorted(
        {
            round(max(0.18, base_scale + delta), 2)
            for delta in (-0.15, -0.10, -0.06, -0.03, 0.0, 0.03, 0.06, 0.10, 0.15)
        }
    )

    scores: list[tuple[float, str, tuple[int, int, int, int]]] = []
    for shape_id, template in templates.items():
        template_h, template_w = template.shape[:2]
        best_for_shape = -1.0
        best_loc = (0, 0)
        best_size = (template_w, template_h)
        for scale in scales:
            resized_w = int(template_w * scale)
            resized_h = int(template_h * scale)
            if resized_w < 8 or resized_h < 8 or resized_w > crop_w or resized_h > crop_h:
                continue
            resized = cv2.resize(template, (resized_w, resized_h))
            result = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best_for_shape:
                best_for_shape = float(max_val)
                best_loc = max_loc
                best_size = (resized_w, resized_h)
        scores.append((best_for_shape, shape_id, (best_loc[0], best_loc[1], best_size[0], best_size[1])))

    scores.sort(reverse=True, key=lambda item: item[0])
    if not scores or scores[0][0] < min_confidence:
        best_score = scores[0][0] if scores else -1.0
        return {
            "shape_id": "Unknown",
            "confidence": round(best_score, 2),
            "second_best_confidence": round(scores[1][0], 2) if len(scores) > 1 else -1.0,
            "margin": 0.0,
            "match_top_left": None,
            "match_size": None,
        }

    best_score, best_shape, (match_x, match_y, match_w, match_h) = scores[0]
    second_score = scores[1][0] if len(scores) > 1 else -1.0
    margin = best_score - second_score if second_score >= 0 else best_score
    if margin < min_margin:
        return {
            "shape_id": "Unknown",
            "confidence": round(best_score, 2),
            "second_best_confidence": round(second_score, 2),
            "margin": round(margin, 2),
            "match_top_left": (match_x, match_y),
            "match_size": (match_w, match_h),
        }

    return {
        "shape_id": best_shape,
        "confidence": round(best_score, 2),
        "second_best_confidence": round(second_score, 2),
        "margin": round(margin, 2),
        "match_top_left": (match_x, match_y),
        "match_size": (match_w, match_h),
    }


def _match_scales_for_crop(
    crop_h: int,
    crop_w: int,
    *,
    scale_hint: tuple[int, int] | None = None,
    sample_template: np.ndarray | None = None,
) -> list[float]:
    if scale_hint is None and sample_template is not None:
        scale_hint = sample_template.shape[1], sample_template.shape[0]
    if scale_hint is None:
        scale_hint = (crop_w, crop_h)
    base_scale = max(min(crop_w / max(1, scale_hint[0]), crop_h / max(1, scale_hint[1])), 0.12)
    return sorted(
        {
            round(max(0.18, base_scale + delta), 2)
            for delta in (-0.15, -0.10, -0.06, -0.03, 0.0, 0.03, 0.06, 0.10, 0.15)
        }
    )


def _best_template_match_in_crop(
    gray: np.ndarray,
    template: np.ndarray,
    scales: list[float],
) -> tuple[float, tuple[int, int], tuple[int, int]]:
    crop_h, crop_w = gray.shape[:2]
    template_h, template_w = template.shape[:2]
    best_score = -1.0
    best_loc = (0, 0)
    best_size = (template_w, template_h)
    for scale in scales:
        resized_w = int(template_w * scale)
        resized_h = int(template_h * scale)
        if resized_w < 8 or resized_h < 8 or resized_w > crop_w or resized_h > crop_h:
            continue
        resized = cv2.resize(template, (resized_w, resized_h))
        result = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > best_score:
            best_score = float(max_val)
            best_loc = max_loc
            best_size = (resized_w, resized_h)
    return best_score, best_loc, best_size


def match_shape_template_variants_in_crop(
    crop_bgr: np.ndarray,
    template_variants: dict[str, list[np.ndarray]],
    *,
    min_confidence: float = INVENTORY_SLOT_MIN_CONFIDENCE,
    high_confidence: float = HIGH_CONFIDENCE_DRIVE_SHAPE,
    min_margin: float = INVENTORY_SLOT_MIN_MARGIN,
    scale_hint: tuple[int, int] | None = None,
) -> dict:
    """Match crop against multiple quality variants per shape_id, returning the base shape."""
    if crop_bgr is None or crop_bgr.size == 0 or not template_variants:
        return {
            "shape_id": "Unknown",
            "confidence": -1.0,
            "second_best_confidence": -1.0,
            "margin": 0.0,
            "match_top_left": None,
            "match_size": None,
        }

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(crop_bgr.shape) == 3 else crop_bgr
    crop_h, crop_w = gray.shape[:2]
    sample_template = next(
        (variant for variants in template_variants.values() for variant in variants),
        None,
    )
    scales = _match_scales_for_crop(
        crop_h,
        crop_w,
        scale_hint=scale_hint,
        sample_template=sample_template,
    )

    scores: list[tuple[float, str, tuple[int, int, int, int]]] = []
    for shape_id, variants in template_variants.items():
        best_for_shape = -1.0
        best_loc = (0, 0)
        best_size = (0, 0)
        for template in variants:
            score, match_loc, match_size = _best_template_match_in_crop(gray, template, scales)
            if score > best_for_shape:
                best_for_shape = score
                best_loc = match_loc
                best_size = match_size
        if best_for_shape >= 0:
            scores.append(
                (
                    best_for_shape,
                    shape_id,
                    (best_loc[0], best_loc[1], best_size[0], best_size[1]),
                )
            )

    scores.sort(reverse=True, key=lambda item: item[0])
    if not scores or scores[0][0] < min_confidence:
        best_score = scores[0][0] if scores else -1.0
        return {
            "shape_id": "Unknown",
            "confidence": round(best_score, 2),
            "second_best_confidence": round(scores[1][0], 2) if len(scores) > 1 else -1.0,
            "margin": 0.0,
            "match_top_left": None,
            "match_size": None,
        }

    best_score, best_shape, (match_x, match_y, match_w, match_h) = scores[0]
    second_score = scores[1][0] if len(scores) > 1 else -1.0
    margin = best_score - second_score if second_score >= 0 else best_score
    if margin < min_margin:
        return {
            "shape_id": "Unknown",
            "confidence": round(best_score, 2),
            "second_best_confidence": round(second_score, 2),
            "margin": round(margin, 2),
            "match_top_left": (match_x, match_y),
            "match_size": (match_w, match_h),
        }

    return {
        "shape_id": best_shape,
        "confidence": round(best_score, 2),
        "second_best_confidence": round(second_score, 2),
        "margin": round(margin, 2),
        "match_top_left": (match_x, match_y),
        "match_size": (match_w, match_h),
    }


def match_inventory_slot_templates_execute_style(
    crop_bgr: np.ndarray,
    template_variants: dict[str, list[np.ndarray]],
    *,
    min_confidence: float = INVENTORY_SLOT_MIN_CONFIDENCE,
    high_confidence: float = HIGH_CONFIDENCE_DRIVE_SHAPE,
    min_margin: float = INVENTORY_SLOT_MIN_MARGIN,
) -> dict:
    """Match an inventory cell crop against all _Inv variants; highest score wins.

    Competes across every Gold/Purple inventory template at once, using the same
    multi-scale local matching as inventory import parsing.
    """
    empty_result = {
        "shape_id": "Unknown",
        "confidence": -1.0,
        "second_best_confidence": -1.0,
        "margin": 0.0,
        "high_confidence_accepted": False,
        "match_top_left": None,
        "match_size": None,
    }
    if crop_bgr is None or crop_bgr.size == 0 or not template_variants:
        return dict(empty_result)

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(crop_bgr.shape) == 3 else crop_bgr
    crop_h, crop_w = gray.shape[:2]
    if crop_h < 8 or crop_w < 8:
        return dict(empty_result)

    sample_template = next(
        (variant for variants in template_variants.values() for variant in variants),
        None,
    )
    scales = _match_scales_for_crop(
        crop_h,
        crop_w,
        sample_template=sample_template,
    )

    scores: list[tuple[float, str]] = []
    for shape_id, variants in template_variants.items():
        for template in variants:
            if template is None or template.size == 0:
                continue
            score, _, _ = _best_template_match_in_crop(gray, template, scales)
            scores.append((score, shape_id))

    if not scores:
        return dict(empty_result)

    scores.sort(reverse=True, key=lambda item: item[0])
    best_score, best_shape = scores[0]
    second_score = scores[1][0] if len(scores) > 1 else -1.0
    margin = best_score - second_score if second_score >= 0 else best_score
    accepted = best_score >= high_confidence or (
        best_score >= min_confidence and margin >= min_margin
    )
    if not accepted and best_score < min_confidence:
        return {
            **empty_result,
            "confidence": round(best_score, 2),
            "second_best_confidence": round(second_score, 2),
            "margin": round(margin, 2),
        }

    return {
        "shape_id": best_shape,
        "confidence": round(best_score, 2),
        "second_best_confidence": round(second_score, 2),
        "margin": round(margin, 2),
        "high_confidence_accepted": best_score >= high_confidence,
        "match_top_left": None,
        "match_size": (crop_w, crop_h),
    }


DEFAULT_INVENTORY_GRID = {
    "origin_2k": [53, 358],
    "cell_size_2k": [169, 140],
    "grid_columns": 4,
    "grid_rows": 5,
}
DEFAULT_SELECTION_TRIANGLE_SIZE_2K = (70, 20)


def build_inventory_grid_layout(
    target_width: int,
    target_height: int,
    grid_cfg: dict | None = None,
    *,
    base_width: int = 2560,
    base_height: int = 1440,
    content_rect: tuple[int, int, int, int] | None = None,
) -> dict:
    """Scale the inventory slot grid from 2K calibration coordinates."""
    from src.scanner.window_capture import scale_region

    cfg = dict(DEFAULT_INVENTORY_GRID)
    if grid_cfg:
        cfg.update(grid_cfg)

    origin_x, origin_y = cfg["origin_2k"]
    cell_w, cell_h = cfg["cell_size_2k"]
    columns = max(1, int(cfg.get("grid_columns", 4) or 4))
    rows = max(1, int(cfg.get("grid_rows", 5) or 5))
    base_size = (base_width, base_height)

    scaled_origin = scale_region(
        (origin_x, origin_y, origin_x, origin_y),
        target_width,
        target_height,
        base_size,
        content_rect=content_rect,
    )
    scaled_cell = scale_region(
        (0, 0, cell_w, cell_h),
        target_width,
        target_height,
        base_size,
        content_rect=content_rect,
    )
    cell_width = max(1, scaled_cell[2] - scaled_cell[0])
    cell_height = max(1, scaled_cell[3] - scaled_cell[1])

    return {
        "origin_x": scaled_origin[0],
        "origin_y": scaled_origin[1],
        "cell_width": cell_width,
        "cell_height": cell_height,
        "grid_columns": columns,
        "grid_rows": rows,
    }


def _inventory_cell_box(layout: dict, row: int, col: int) -> tuple[int, int, int, int]:
    x1 = int(layout["origin_x"] + col * layout["cell_width"])
    y1 = int(layout["origin_y"] + row * layout["cell_height"])
    x2 = int(x1 + layout["cell_width"])
    y2 = int(y1 + layout["cell_height"])
    return x1, y1, x2, y2


def load_inventory_selection_triangle(template_path: str | os.PathLike[str]) -> np.ndarray | None:
    path = os.fspath(template_path)
    if not os.path.exists(path):
        logger.warning(f"库存选中三角模板不存在: {path}")
        return None
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        logger.warning(f"无法读取库存选中三角模板: {path}")
        return None
    return image


def _grid_triangle_search_region(
    grid_layout: dict,
    panel_region: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Limit triangle matching to the inventory grid band inside the panel."""
    origin_x = int(grid_layout["origin_x"])
    origin_y = int(grid_layout["origin_y"])
    cell_width = int(grid_layout["cell_width"])
    cell_height = int(grid_layout["cell_height"])
    columns = int(grid_layout["grid_columns"])
    rows = int(grid_layout["grid_rows"])

    pad_x = max(4, int(cell_width * 0.15))
    pad_right = max(pad_x, int(cell_width * 0.40))
    pad_top = max(4, int(cell_height * 0.35))
    pad_bottom = max(4, int(cell_height * 0.10))
    x1 = max(0, origin_x - pad_x)
    y1 = max(0, origin_y - pad_top)
    x2 = min(image_width, origin_x + columns * cell_width + pad_right)
    y2 = min(image_height, origin_y + rows * cell_height + pad_bottom)

    px1, py1, px2, py2 = panel_region
    return (
        max(x1, px1),
        max(y1, py1),
        min(x2, px2),
        min(y2, py2),
    )


def _selection_box_from_triangle(
    triangle_match: dict,
    grid_layout: dict,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Build the full slot crop directly below the matched selection triangle."""
    cell_width = int(grid_layout["cell_width"])
    cell_height = int(grid_layout["cell_height"])
    center_x = int(triangle_match["center_x"])
    y1 = int(triangle_match["bottom_y"])
    x1 = int(round(center_x - cell_width / 2))
    y2 = y1 + cell_height
    x2 = x1 + cell_width

    if x1 < 0:
        x1 = 0
        x2 = min(image_width, cell_width)
    if x2 > image_width:
        x2 = image_width
        x1 = max(0, x2 - cell_width)
    if y1 < 0:
        y1 = 0
        y2 = min(image_height, cell_height)
    if y2 > image_height:
        y2 = image_height
        y1 = max(0, y2 - cell_height)

    return x1, y1, x2, y2


def _triangle_match_is_plausible(triangle_match: dict, grid_layout: dict) -> bool:
    origin_x = float(grid_layout["origin_x"])
    origin_y = float(grid_layout["origin_y"])
    cell_width = float(grid_layout["cell_width"])
    cell_height = float(grid_layout["cell_height"])
    columns = int(grid_layout["grid_columns"])
    rows = int(grid_layout["grid_rows"])

    center_x = float(triangle_match["center_x"])
    bottom_y = float(triangle_match["bottom_y"])
    grid_right = origin_x + columns * cell_width
    grid_bottom = origin_y + rows * cell_height
    return (
        origin_x - cell_width * 0.35 <= center_x <= grid_right + cell_width * 0.35
        and origin_y - cell_height * 0.35 <= bottom_y <= grid_bottom + cell_height * 0.20
    )


def _match_selection_triangle(
    img: np.ndarray,
    search_region: tuple[int, int, int, int],
    triangle_template: np.ndarray,
    *,
    min_confidence: float = 0.80,
    triangle_size_2k: tuple[int, int] = DEFAULT_SELECTION_TRIANGLE_SIZE_2K,
    base_width: int = 2560,
    base_height: int = 1440,
    content_rect: tuple[int, int, int, int] | None = None,
) -> dict | None:
    """Locate the fixed orange selection triangle inside the inventory panel."""
    from src.scanner.window_capture import scale_region

    x1, y1, x2, y2 = search_region
    image_h, image_w = img.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_w, x2)
    y2 = min(image_h, y2)
    search = img[y1:y2, x1:x2]
    if search is None or search.size == 0 or triangle_template is None or triangle_template.size == 0:
        return None

    gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
    template = (
        cv2.cvtColor(triangle_template, cv2.COLOR_BGR2GRAY)
        if len(triangle_template.shape) == 3
        else triangle_template
    )
    template_h, template_w = template.shape[:2]
    expected_w, expected_h = triangle_size_2k
    scaled_size = scale_region(
        (0, 0, expected_w, expected_h),
        image_w,
        image_h,
        (base_width, base_height),
        content_rect=content_rect,
    )
    target_w = max(4, scaled_size[2] - scaled_size[0])
    target_h = max(3, scaled_size[3] - scaled_size[1])
    base_scale = min(target_w / max(1, template_w), target_h / max(1, template_h))
    scales = sorted(
        {
            round(base_scale * factor, 2)
            for factor in (0.85, 0.92, 1.0, 1.08, 1.15)
            if base_scale * factor > 0.05
        }
    )

    best_score = -1.0
    best_loc = (0, 0)
    best_size = (template_w, template_h)
    search_h, search_w = gray.shape[:2]
    for scale in scales:
        resized_w = max(4, int(template_w * scale))
        resized_h = max(3, int(template_h * scale))
        if resized_w > search_w or resized_h > search_h:
            continue
        resized = cv2.resize(template, (resized_w, resized_h))
        result = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > best_score:
            best_score = float(max_val)
            best_loc = max_loc
            best_size = (resized_w, resized_h)

    if best_score < min_confidence:
        return None

    abs_x = x1 + best_loc[0]
    abs_y = y1 + best_loc[1]
    resized_w, resized_h = best_size
    center_x = abs_x + resized_w // 2
    bottom_y = abs_y + resized_h
    return {
        "confidence": round(best_score, 3),
        "top_left": (abs_x, abs_y),
        "size": best_size,
        "center_x": center_x,
        "bottom_y": bottom_y,
    }


def _row_index_from_triangle_bottom(bottom_y: float, grid_layout: dict) -> int:
    """Pick the grid row whose cell top is closest to the triangle bottom edge."""
    origin_y = float(grid_layout["origin_y"])
    cell_height = float(grid_layout["cell_height"])
    rows = int(grid_layout["grid_rows"])
    best_row = 0
    best_dist = float("inf")
    for candidate in range(rows):
        expected_top = origin_y + candidate * cell_height
        dist = abs(bottom_y - expected_top)
        if dist < best_dist:
            best_dist = dist
            best_row = candidate
    return best_row


def _cell_index_from_triangle(
    triangle_match: dict,
    grid_layout: dict,
    selection_box: tuple[int, int, int, int] | None = None,
) -> tuple[int, int]:
    origin_x = float(grid_layout["origin_x"])
    cell_width = float(grid_layout["cell_width"])
    cell_height = float(grid_layout["cell_height"])
    columns = int(grid_layout["grid_columns"])
    rows = int(grid_layout["grid_rows"])

    center_x = float(triangle_match["center_x"])
    bottom_y = float(triangle_match["bottom_y"])
    col = int(round((center_x - origin_x - cell_width / 2) / cell_width))
    row = _row_index_from_triangle_bottom(bottom_y, grid_layout)
    col = max(0, min(columns - 1, col))
    row = max(0, min(rows - 1, row))
    return row, col


def _triangle_aligns_with_grid_row(
    triangle_match: dict,
    grid_layout: dict,
    row: int,
) -> bool:
    origin_y = float(grid_layout["origin_y"])
    cell_height = float(grid_layout["cell_height"])
    expected_top = origin_y + row * cell_height
    return abs(float(triangle_match["bottom_y"]) - expected_top) <= cell_height * 0.45


def find_selected_inventory_cell(
    img: np.ndarray,
    grid_layout: dict,
    panel_region: tuple[int, int, int, int],
    triangle_template: np.ndarray,
    *,
    min_triangle_confidence: float = 0.80,
    triangle_size_2k: tuple[int, int] = DEFAULT_SELECTION_TRIANGLE_SIZE_2K,
    base_width: int = 2560,
    base_height: int = 1440,
    content_rect: tuple[int, int, int, int] | None = None,
) -> dict | None:
    """Find the selected inventory slot via the orange triangle indicator above it."""
    image_h, image_w = img.shape[:2]
    search_region = _grid_triangle_search_region(
        grid_layout,
        panel_region,
        image_w,
        image_h,
    )
    triangle_match = _match_selection_triangle(
        img,
        search_region,
        triangle_template,
        min_confidence=min_triangle_confidence,
        triangle_size_2k=triangle_size_2k,
        base_width=base_width,
        base_height=base_height,
        content_rect=content_rect,
    )
    if triangle_match is None or not _triangle_match_is_plausible(triangle_match, grid_layout):
        return None

    provisional_box = _selection_box_from_triangle(triangle_match, grid_layout, image_w, image_h)
    row, col = _cell_index_from_triangle(triangle_match, grid_layout, provisional_box)
    if not _triangle_aligns_with_grid_row(triangle_match, grid_layout, row):
        return None

    selection_box = _inventory_cell_box(grid_layout, row, col)
    columns = int(grid_layout["grid_columns"])
    slot_index = row * columns + col + 1
    return {
        "slot_index": slot_index,
        "slot_row": row,
        "slot_col": col,
        "selection_box": selection_box,
        "triangle_confidence": triangle_match["confidence"],
        "triangle_top_left": triangle_match["top_left"],
        "triangle_size": triangle_match["size"],
    }


def locate_shape_in_slot_crop(
    gold_templates: dict[str, np.ndarray] | dict[str, list[np.ndarray]],
    slot_crop: np.ndarray,
    *,
    candidate_shape_ids: list[str] | None = None,
    min_confidence: float = INVENTORY_SLOT_MIN_CONFIDENCE,
    high_confidence: float = HIGH_CONFIDENCE_DRIVE_SHAPE,
    min_margin: float = INVENTORY_SLOT_MIN_MARGIN,
) -> dict:
    if gold_templates and isinstance(next(iter(gold_templates.values())), list):
        template_variants = gold_templates
        if candidate_shape_ids is not None:
            template_variants = {
                shape_id: template_variants[shape_id]
                for shape_id in candidate_shape_ids
                if shape_id in template_variants
            }
        return match_inventory_slot_templates_execute_style(
            slot_crop,
            template_variants,
            min_confidence=min_confidence,
            high_confidence=high_confidence,
            min_margin=min_margin,
        )

    templates = gold_templates
    if candidate_shape_ids is not None:
        templates = {
            shape_id: templates[shape_id]
            for shape_id in candidate_shape_ids
            if shape_id in templates
        }
    return match_shape_templates_in_crop(
        slot_crop,
        templates,
        min_confidence=min_confidence,
        min_margin=min_margin,
    )


def locate_selected_inventory_shape(
    gold_templates: dict[str, np.ndarray],
    img: np.ndarray,
    panel_region: tuple[int, int, int, int] | None = None,
    *,
    grid_layout: dict | None = None,
    grid_cfg: dict | None = None,
    triangle_template: np.ndarray | None = None,
    triangle_template_path: str | os.PathLike[str] | None = None,
    base_width: int = 2560,
    base_height: int = 1440,
    min_confidence: float = INVENTORY_SLOT_MIN_CONFIDENCE,
    high_confidence: float = HIGH_CONFIDENCE_DRIVE_SHAPE,
    min_margin: float = INVENTORY_SLOT_MIN_MARGIN,
    min_triangle_confidence: float = 0.80,
    triangle_size_2k: tuple[int, int] = DEFAULT_SELECTION_TRIANGLE_SIZE_2K,
    content_rect: tuple[int, int, int, int] | None = None,
) -> dict:
    """Detect the highlighted inventory slot and recognize the drive shape inside it."""
    if panel_region is None:
        return {"shape_id": "Unknown", "confidence": -1.0}

    image_h, image_w = img.shape[:2]
    if grid_layout is None:
        grid_layout = build_inventory_grid_layout(
            image_w,
            image_h,
            grid_cfg,
            base_width=base_width,
            base_height=base_height,
            content_rect=content_rect,
        )

    if triangle_template is None and triangle_template_path is not None:
        triangle_template = load_inventory_selection_triangle(triangle_template_path)
    if triangle_template is None:
        return {"shape_id": "Unknown", "confidence": -1.0}

    selected = find_selected_inventory_cell(
        img,
        grid_layout,
        panel_region,
        triangle_template,
        min_triangle_confidence=min_triangle_confidence,
        triangle_size_2k=triangle_size_2k,
        base_width=base_width,
        base_height=base_height,
        content_rect=content_rect,
    )
    if selected is None:
        return {"shape_id": "Unknown", "confidence": -1.0}

    x1, y1, x2, y2 = selected["selection_box"]
    slot_crop = img[y1:y2, x1:x2]
    result = locate_shape_in_slot_crop(
        gold_templates,
        slot_crop,
        min_confidence=min_confidence,
        high_confidence=high_confidence,
        min_margin=min_margin,
    )
    result.update(selected)
    return result


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def locate_shape_in_image(shape_recognizer, img, region=None) -> dict:
    if region is not None:
        x1, y1, x2, y2 = region
        search = img[y1:y2, x1:x2]
    else:
        search = img
    if search is None or search.size == 0:
        return {"shape_id": "Unknown", "confidence": -1.0}
    gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY) if len(search.shape) == 3 else search
    search_h, search_w = gray.shape[:2]
    best_shape = "Unknown"
    best_score = -1.0
    base_scale = max(0.35, min(1.6, search_w / 900))
    scales = sorted({0.35, 0.45, 0.55, 0.65, 0.75, 0.9, 1.0, 1.15, 1.3, round(base_scale, 2)})
    for shape_id, template in shape_recognizer.templates.items():
        th, tw = template.shape[:2]
        for scale in scales:
            rw, rh = int(tw * scale), int(th * scale)
            if rw < 32 or rh < 32 or rw > search_w or rh > search_h:
                continue
            resized = cv2.resize(template, (rw, rh))
            res = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > best_score:
                best_score = max_val
                best_shape = shape_id
    if best_score < 0.58:
        best_shape = "Unknown"
    return {"shape_id": best_shape, "confidence": round(float(best_score), 2)}


def locate_selected_reward_shape(shape_recognizer, img) -> dict:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if len(img.shape) != 3:
        hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([140, 80, 120]), np.array([179, 255, 255]))
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    image_h, image_w = img.shape[:2]
    min_area = max(300, int(image_w * image_h * 0.0001))
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h >= min_area:
            boxes.append((x, y, x + w, y + h))
    if boxes:
        x1 = max(0, min(box[0] for box in boxes) - int(image_w * 0.012))
        y1 = max(0, min(box[1] for box in boxes) - int(image_h * 0.02))
        x2 = min(image_w, max(box[2] for box in boxes) + int(image_w * 0.012))
        y2 = min(image_h, max(box[3] for box in boxes) + int(image_h * 0.02))
        selected = locate_shape_in_image(shape_recognizer, img, (x1, y1, x2, y2))
        if selected["shape_id"] != "Unknown":
            return selected
    return locate_shape_in_image(shape_recognizer, img, None)


def looks_like_drive_identity(text: str) -> bool:
    clean = (text or "").replace(" ", "")
    if "卡带" in clean:
        return False
    if "驱动块" in clean or "驱动" in clean:
        return True
    return "驱" in clean and "动" in clean


def looks_like_tape_identity(parser, text: str) -> bool:
    clean = (text or "").replace(" ", "")
    if not clean:
        return False
    if "卡带" in clean:
        return True

    only_cn = "".join(ch for ch in clean if "\u4e00" <= ch <= "\u9fff")
    for set_name in parser.REAL_SETS_WHITE_LIST:
        set_cn = "".join(ch for ch in set_name if "\u4e00" <= ch <= "\u9fff")
        if set_cn and (set_cn in only_cn or only_cn in set_cn and len(only_cn) >= 3):
            return True
    return False


def classify_item(processor, img, region_profiles):
    candidates = []

    for profile_name, regions in region_profiles:
        profile_start = time.perf_counter()
        shape_box = regions["drive_shape_icon"]
        crop = img[shape_box[1]:shape_box[3], shape_box[0]:shape_box[2]]
        if crop.size == 0:
            continue
        shape_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        shape_start = time.perf_counter()
        shape_res = processor.shape_recognizer.recognize(shape_crop)
        shape_ms = _elapsed_ms(shape_start)
        if (
            shape_res["shape_id"] != "Unknown"
            and shape_res["confidence"] >= HIGH_CONFIDENCE_DRIVE_SHAPE
        ):
            shape_res = dict(shape_res)
            shape_res["identity_skipped"] = True
            log_perf(
                logger,
                "parse.classify_profile",
                elapsed_ms=_elapsed_ms(profile_start),
                profile=profile_name,
                shape_ms=shape_ms,
                identity_ocr_ms=0.0,
                shape_id=shape_res["shape_id"],
                shape_conf=shape_res["confidence"],
                identity_len=0,
                drive_text=1,
                tape_text=0,
                identity_skipped=1,
            )
            logger.debug(
                f"候选坐标[{profile_name}] shape={shape_box} "
                f"高置信形状={shape_res['shape_id']}({shape_res['confidence']})，跳过身份 OCR"
            )
            return "drive", profile_name, regions, shape_res, ""

        id_box = regions["identity_check"]
        id_crop = img[id_box[1]:id_box[3], id_box[0]:id_box[2]]
        identity_start = time.perf_counter()
        raw_hub_texts = processor.ocr_engine.extract_text(id_crop)
        identity_ocr_ms = _elapsed_ms(identity_start)
        hub_text = "".join(raw_hub_texts)
        is_drive_text = looks_like_drive_identity(hub_text)
        is_tape_text = looks_like_tape_identity(processor.parser, hub_text)

        candidates.append(
            {
                "profile": profile_name,
                "regions": regions,
                "shape": shape_res,
                "hub_text": hub_text,
                "is_drive_text": is_drive_text,
                "is_tape_text": is_tape_text,
            }
        )
        log_perf(
            logger,
            "parse.classify_profile",
            elapsed_ms=_elapsed_ms(profile_start),
            profile=profile_name,
            shape_ms=shape_ms,
            identity_ocr_ms=identity_ocr_ms,
            shape_id=shape_res["shape_id"],
            shape_conf=shape_res["confidence"],
            identity_len=len(hub_text),
            drive_text=int(is_drive_text),
            tape_text=int(is_tape_text),
        )
        logger.debug(
            f"候选坐标[{profile_name}] identity={id_box} shape={shape_box} "
            f"text={hub_text} shape={shape_res['shape_id']}({shape_res['confidence']})"
        )

    if not candidates:
        profile_name, regions = region_profiles[0]
        return "tape", profile_name, regions, {"shape_id": "Unknown", "confidence": -1.0}, ""

    tape_candidates = [c for c in candidates if c["is_tape_text"] and not c["is_drive_text"]]
    if tape_candidates:
        chosen = max(tape_candidates, key=lambda c: len(c["hub_text"]))
        return "tape", chosen["profile"], chosen["regions"], chosen["shape"], chosen["hub_text"]

    drive_text_candidates = [c for c in candidates if c["is_drive_text"]]
    if drive_text_candidates:
        chosen = max(drive_text_candidates, key=lambda c: c["shape"]["confidence"])
        return "drive", chosen["profile"], chosen["regions"], chosen["shape"], chosen["hub_text"]

    best_shape_candidate = max(candidates, key=lambda c: c["shape"]["confidence"])
    if (
        best_shape_candidate["shape"]["shape_id"] != "Unknown"
        and best_shape_candidate["shape"]["confidence"] >= processor.DRIVE_TYPE_CONFIDENCE
    ):
        return (
            "drive",
            best_shape_candidate["profile"],
            best_shape_candidate["regions"],
            best_shape_candidate["shape"],
            best_shape_candidate["hub_text"],
        )

    chosen = max(candidates, key=lambda c: len(c["hub_text"]))
    return "tape", chosen["profile"], chosen["regions"], chosen["shape"], chosen["hub_text"]
