# 判断截图对应的装备类型。
"""Helpers for classifying screenshots as drive or tape."""

from __future__ import annotations

import json
import os
import time
from functools import lru_cache

import cv2
import numpy as np

from src.utils.logger import logger
from src.utils.perf import log_perf


HIGH_CONFIDENCE_DRIVE_SHAPE = 0.95
INVENTORY_SLOT_MIN_CONFIDENCE = 0.58


@lru_cache(maxsize=4)
def _load_shape_matrices(config_dir: str) -> dict[str, np.ndarray]:
    shapes_path = os.path.join(config_dir, "shapes.json")
    if not os.path.exists(shapes_path):
        return {}
    with open(shapes_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    matrices: dict[str, np.ndarray] = {}
    for item in data.get("shapes", []):
        shape_id = item.get("shape_id")
        matrix = item.get("matrix")
        if shape_id and shape_id != "TAPE_15" and matrix:
            matrices[str(shape_id)] = np.array(matrix, dtype=int)
    return matrices


def _selection_highlight_mask(hsv_img: np.ndarray) -> np.ndarray:
    mask_pink = cv2.inRange(hsv_img, np.array([140, 80, 120]), np.array([179, 255, 255]))
    mask_red = cv2.inRange(hsv_img, np.array([0, 80, 120]), np.array([10, 255, 255]))
    return cv2.bitwise_or(mask_pink, mask_red)


def _find_inventory_selection_box(
    img: np.ndarray,
    panel_region: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int] | None:
    image_h, image_w = img.shape[:2]
    search = img
    offset_x = offset_y = 0
    if panel_region is not None:
        x1, y1, x2, y2 = panel_region
        pad_x = max(8, int((x2 - x1) * 0.08))
        pad_y = max(8, int((y2 - y1) * 0.05))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(image_w, x2 + pad_x)
        y2 = min(image_h, y2 + pad_y)
        search = img[y1:y2, x1:x2]
        offset_x, offset_y = x1, y1
    if search is None or search.size == 0:
        return None

    hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV) if len(search.shape) == 3 else cv2.cvtColor(search, cv2.COLOR_GRAY2BGR)
    mask = _selection_highlight_mask(hsv)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    search_h, search_w = search.shape[:2]
    min_area = max(500, int(search_w * search_h * 0.0010))
    max_area = int(search_w * search_h * 0.12)
    best_box = None
    best_area = 0

    for contour in contours:
        x, y, box_w, box_h = cv2.boundingRect(contour)
        area = box_w * box_h
        if area < min_area or area > max_area:
            continue
        aspect = box_w / max(box_h, 1)
        if aspect < 0.65 or aspect > 1.55:
            continue
        center_x = offset_x + x + box_w / 2
        center_y = offset_y + y + box_h / 2
        if center_x > image_w * 0.30 or center_y < image_h * 0.10 or center_y > image_h * 0.82:
            continue
        if area > best_area:
            best_area = area
            best_box = (
                offset_x + x,
                offset_y + y,
                offset_x + x + box_w,
                offset_y + y + box_h,
            )
    return best_box


def _estimate_shape_matrix_size(aspect: float) -> tuple[int, int]:
    if aspect >= 1.05:
        return 1, max(2, round(aspect * 1.05))
    if aspect <= 0.95:
        return max(2, round((1 / max(aspect, 1e-6)) * 1.05)), 1
    return 2, 2


def _slot_orange_bbox(slot_crop: np.ndarray) -> tuple[float, tuple[int, int, int, int] | None]:
    if slot_crop is None or slot_crop.size == 0:
        return 1.0, None
    h, w = slot_crop.shape[:2]
    inner = slot_crop[int(h * 0.12) : int(h * 0.88), int(w * 0.05) : int(w * 0.95)]
    if inner.size == 0:
        return 1.0, None
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([10, 120, 140]), np.array([35, 255, 255]))
    coords = np.column_stack(np.where(mask > 0))
    if coords.size == 0:
        return 1.0, None
    y1, x1 = coords.min(axis=0)
    y2, x2 = coords.max(axis=0)
    box_w = int(x2 - x1 + 1)
    box_h = int(y2 - y1 + 1)
    return box_w / max(box_h, 1), (int(x1), int(y1), int(x2 + 1), int(y2 + 1))


def locate_shape_in_slot_crop(
    shape_recognizer,
    slot_crop: np.ndarray,
    *,
    candidate_shape_ids: list[str] | None = None,
    min_confidence: float = INVENTORY_SLOT_MIN_CONFIDENCE,
) -> dict:
    if slot_crop is None or slot_crop.size == 0:
        return {"shape_id": "Unknown", "confidence": -1.0}

    aspect, _orange_box = _slot_orange_bbox(slot_crop)
    est_rows, est_cols = _estimate_shape_matrix_size(aspect)
    config_dir = os.path.dirname(getattr(shape_recognizer, "template_dir", "config/templates"))
    shape_matrices = _load_shape_matrices(config_dir)

    if candidate_shape_ids is None:
        candidate_shape_ids = [
            shape_id
            for shape_id, matrix in shape_matrices.items()
            if matrix.shape == (est_rows, est_cols)
        ]
    else:
        candidate_shape_ids = [
            shape_id
            for shape_id in candidate_shape_ids
            if shape_id in shape_matrices and shape_matrices[shape_id].shape == (est_rows, est_cols)
        ]

    if not candidate_shape_ids:
        return {"shape_id": "Unknown", "confidence": -1.0, "matrix_size": (est_rows, est_cols)}

    gray = cv2.cvtColor(slot_crop, cv2.COLOR_BGR2GRAY) if len(slot_crop.shape) == 3 else slot_crop
    slot_h, slot_w = gray.shape[:2]
    base_scale = max(min(slot_w / 250.0, slot_h / 240.0) * 0.95, 0.08)
    scales = sorted({round(base_scale + delta, 2) for delta in (-0.08, -0.05, -0.03, 0.0, 0.03, 0.05, 0.08)})

    best_shape = "Unknown"
    best_score = -1.0
    for shape_id in candidate_shape_ids:
        template = shape_recognizer.templates.get(shape_id)
        if template is None:
            continue
        template_h, template_w = template.shape[:2]
        for scale in scales:
            resized_w = int(template_w * scale)
            resized_h = int(template_h * scale)
            if resized_w < 6 or resized_h < 6 or resized_w > slot_w or resized_h > slot_h:
                continue
            resized = cv2.resize(template, (resized_w, resized_h))
            result = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > best_score:
                best_score = float(max_val)
                best_shape = shape_id

    if best_score < min_confidence:
        best_shape = "Unknown"
    return {
        "shape_id": best_shape,
        "confidence": round(best_score, 2),
        "matrix_size": (est_rows, est_cols),
    }


def locate_selected_inventory_shape(
    shape_recognizer,
    img: np.ndarray,
    panel_region: tuple[int, int, int, int] | None = None,
    *,
    min_confidence: float = INVENTORY_SLOT_MIN_CONFIDENCE,
) -> dict:
    """Detect the highlighted inventory slot and recognize the drive shape inside it."""
    selection_box = _find_inventory_selection_box(img, panel_region)
    if selection_box is None:
        return {"shape_id": "Unknown", "confidence": -1.0}

    x1, y1, x2, y2 = selection_box
    slot_crop = img[y1:y2, x1:x2]
    result = locate_shape_in_slot_crop(
        shape_recognizer,
        slot_crop,
        min_confidence=min_confidence,
    )
    result["selection_box"] = selection_box
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
