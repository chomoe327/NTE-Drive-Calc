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
    base_scale = max(min(crop_w / max(1, scale_hint[0]), crop_h / max(1, scale_hint[1])) * 0.92, 0.12)
    scales = sorted({round(base_scale + delta, 2) for delta in (-0.10, -0.06, -0.03, 0.0, 0.03, 0.06, 0.10)})

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


def locate_shape_in_slot_crop(
    gold_templates: dict[str, np.ndarray],
    slot_crop: np.ndarray,
    *,
    candidate_shape_ids: list[str] | None = None,
    min_confidence: float = INVENTORY_SLOT_MIN_CONFIDENCE,
    min_margin: float = INVENTORY_SLOT_MIN_MARGIN,
) -> dict:
    templates = gold_templates
    if candidate_shape_ids is not None:
        templates = {
            shape_id: gold_templates[shape_id]
            for shape_id in candidate_shape_ids
            if shape_id in gold_templates
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
    min_confidence: float = INVENTORY_SLOT_MIN_CONFIDENCE,
    min_margin: float = INVENTORY_SLOT_MIN_MARGIN,
) -> dict:
    """Detect the highlighted inventory slot and recognize the drive shape inside it."""
    selection_box = _find_inventory_selection_box(img, panel_region)
    if selection_box is None:
        return {"shape_id": "Unknown", "confidence": -1.0}

    x1, y1, x2, y2 = selection_box
    slot_crop = img[y1:y2, x1:x2]
    result = locate_shape_in_slot_crop(
        gold_templates,
        slot_crop,
        min_confidence=min_confidence,
        min_margin=min_margin,
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
