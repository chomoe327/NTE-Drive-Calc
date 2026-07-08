# 检测驱动装配时的装备转移确认弹窗。
"""OCR helpers for the equip-transfer confirmation dialog during assembly."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from src.utils.logger import logger


DEFAULT_EQUIP_TRANSFER_KEYWORDS = (
    "是否镶嵌于此处",
    "该装备已镶嵌",
    "已镶嵌在",
)


def _normalize_dialog_text(text: str) -> str:
    return str(text or "").replace(" ", "").replace("\n", "")


def dialog_text_matches(texts: Iterable[str], keywords: Iterable[str]) -> bool:
    combined = _normalize_dialog_text("".join(texts))
    if not combined:
        return False
    for keyword in keywords:
        normalized_keyword = _normalize_dialog_text(keyword)
        if normalized_keyword and normalized_keyword in combined:
            return True
    return False


def center_dialog_crop(image: np.ndarray, width_ratio: float = 0.72, height_ratio: float = 0.55) -> np.ndarray:
    """Crop the central region where the modal dialog usually appears."""
    image_h, image_w = image.shape[:2]
    crop_w = max(1, int(image_w * width_ratio))
    crop_h = max(1, int(image_h * height_ratio))
    x1 = max(0, (image_w - crop_w) // 2)
    y1 = max(0, (image_h - crop_h) // 2)
    x2 = min(image_w, x1 + crop_w)
    y2 = min(image_h, y1 + crop_h)
    return image[y1:y2, x1:x2]


def detect_equip_transfer_dialog(
    image: np.ndarray,
    *,
    ocr_engine,
    keywords: Iterable[str] | None = None,
    width_ratio: float = 0.72,
    height_ratio: float = 0.55,
) -> bool:
    """Return True when the screenshot center contains equip-transfer dialog text."""
    crop = center_dialog_crop(image, width_ratio=width_ratio, height_ratio=height_ratio)
    texts = ocr_engine.extract_text(crop)
    matched = dialog_text_matches(texts, keywords or DEFAULT_EQUIP_TRANSFER_KEYWORDS)
    if matched:
        logger.info(f"检测到装备转移确认框 OCR: {texts}")
    return matched
