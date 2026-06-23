# 详情面板弃置/上锁状态模板匹配。
"""Template-based detection for discard/lock button states in the detail panel."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.scanner.config import ScannerConfig
from src.scanner.window_capture import crop_window_border_from_image, fit_content_rect, scale_region
from src.utils.image_io import imread_unicode
from src.utils.logger import logger

BASE_DISCARD_ROI = (2055, 330, 2145, 420)
BASE_LOCK_ROI = (2265, 330, 2355, 420)
DEFAULT_THRESHOLD = 0.72
RELATIVE_MARGIN = 0.02
RELATIVE_RATIO = 0.10
CONTRAST_HYSTERESIS = 4.0


@dataclass
class _PairCalibration:
    icon_mask: np.ndarray
    ring_mask: np.ndarray
    contrast_mid: float
    template_shape: tuple[int, int]


class MarkStateDetector:
    def __init__(
        self,
        template_dir: Path,
        confidence_threshold: float = DEFAULT_THRESHOLD,
    ):
        self.template_dir = Path(template_dir)
        self.confidence_threshold = confidence_threshold
        self._templates: dict[str, np.ndarray] = {}
        self._calibrations: dict[str, _PairCalibration] = {}
        self._load_templates()
        self._calibrate_pairs()

    def _load_templates(self) -> None:
        names = (
            "discard_marked",
            "discard_unmarked",
            "lock_marked",
            "lock_unmarked",
        )
        for name in names:
            path = self.template_dir / f"{name}.png"
            img = imread_unicode(str(path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self._templates[name] = img
            else:
                logger.warning(f"标记状态模板缺失: {path}")

    def _calibrate_pairs(self) -> None:
        for prefix in ("discard", "lock"):
            marked = self._templates.get(f"{prefix}_marked")
            unmarked = self._templates.get(f"{prefix}_unmarked")
            if marked is None or unmarked is None:
                continue
            if unmarked.shape != marked.shape:
                unmarked = cv2.resize(
                    unmarked,
                    (marked.shape[1], marked.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
            icon_mask, ring_mask = self._build_masks(marked)
            marked_contrast = self._template_contrast(marked, icon_mask, ring_mask)
            unmarked_contrast = self._template_contrast(unmarked, icon_mask, ring_mask)
            self._calibrations[prefix] = _PairCalibration(
                icon_mask=icon_mask,
                ring_mask=ring_mask,
                contrast_mid=(marked_contrast + unmarked_contrast) / 2.0,
                template_shape=marked.shape[:2],
            )

    def _build_masks(self, template: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = template.shape[:2]
        cy, cx = h / 2.0, w / 2.0
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_r = float(min(cx, cy, w - 1 - cx, h - 1 - cy))
        icon_mask = dist <= max_r * 0.42
        ring_mask = (dist >= max_r * 0.58) & (dist <= max_r * 0.92)
        return icon_mask, ring_mask

    def _template_contrast(
        self,
        template: np.ndarray,
        icon_mask: np.ndarray,
        ring_mask: np.ndarray,
    ) -> float:
        if not np.any(icon_mask) or not np.any(ring_mask):
            return 0.0
        return float(np.mean(template[icon_mask]) - np.mean(template[ring_mask]))

    def _roi_contrast(self, region_gray: np.ndarray, cal: _PairCalibration) -> float:
        th, tw = cal.template_shape
        rh, rw = region_gray.shape[:2]
        if rh == 0 or rw == 0:
            return 0.0
        if rh < th or rw < tw:
            scaled = cv2.resize(region_gray, (tw, th), interpolation=cv2.INTER_AREA)
            patch = scaled
        else:
            y0 = max(0, (rh - th) // 2)
            x0 = max(0, (rw - tw) // 2)
            patch = region_gray[y0 : y0 + th, x0 : x0 + tw]
            if patch.shape[:2] != (th, tw):
                patch = cv2.resize(patch, (tw, th), interpolation=cv2.INTER_AREA)
        return self._template_contrast(patch, cal.icon_mask, cal.ring_mask)

    def _scale_roi(self, roi: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
        content_rect = fit_content_rect(
            width,
            height,
            (ScannerConfig.BASE_WIDTH, ScannerConfig.BASE_HEIGHT),
        )
        return scale_region(
            roi,
            width,
            height,
            (ScannerConfig.BASE_WIDTH, ScannerConfig.BASE_HEIGHT),
            preserve_aspect=True,
            content_rect=content_rect,
        )

    def _best_score(self, region_gray: np.ndarray, template: np.ndarray) -> float:
        if region_gray.size == 0 or template.size == 0:
            return 0.0
        th, tw = template.shape[:2]
        rh, rw = region_gray.shape[:2]
        if rh < th or rw < tw:
            scaled = cv2.resize(template, (rw, rh), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(region_gray, scaled, cv2.TM_CCOEFF_NORMED)
        else:
            res = cv2.matchTemplate(region_gray, template, cv2.TM_CCOEFF_NORMED)
        return float(res.max()) if res.size else 0.0

    def _template_fallback(self, marked_score: float, unmarked_score: float) -> bool:
        margin = marked_score - unmarked_score
        return marked_score > unmarked_score and (
            marked_score >= self.confidence_threshold
            or margin >= RELATIVE_MARGIN
            or margin >= unmarked_score * RELATIVE_RATIO
        )

    def _pair_state(
        self,
        region_gray: np.ndarray,
        marked_key: str,
        unmarked_key: str,
    ) -> tuple[bool, float, float, float, float]:
        marked = self._templates.get(marked_key)
        unmarked = self._templates.get(unmarked_key)
        if marked is None or unmarked is None:
            return False, 0.0, 0.0, 0.0, 0.0

        marked_score = self._best_score(region_gray, marked)
        unmarked_score = self._best_score(region_gray, unmarked)

        prefix = marked_key[: -len("_marked")]
        cal = self._calibrations.get(prefix)
        if cal is None:
            is_marked = self._template_fallback(marked_score, unmarked_score)
            return is_marked, marked_score, unmarked_score, 0.0, 0.0

        roi_contrast = self._roi_contrast(region_gray, cal)
        contrast_mid = cal.contrast_mid
        if roi_contrast >= contrast_mid:
            is_marked = True
        elif roi_contrast <= contrast_mid - CONTRAST_HYSTERESIS:
            is_marked = False
        else:
            is_marked = self._template_fallback(marked_score, unmarked_score)

        return is_marked, marked_score, unmarked_score, roi_contrast, contrast_mid

    def detect_from_bgr(self, image_bgr: np.ndarray) -> dict[str, bool | float]:
        image_bgr = crop_window_border_from_image(image_bgr)
        height, width = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        discard_roi = self._scale_roi(BASE_DISCARD_ROI, width, height)
        lock_roi = self._scale_roi(BASE_LOCK_ROI, width, height)
        x1, y1, x2, y2 = discard_roi
        discard_region = gray[y1:y2, x1:x2]
        x1, y1, x2, y2 = lock_roi
        lock_region = gray[y1:y2, x1:x2]
        (
            discard_marked,
            discard_marked_score,
            discard_unmarked_score,
            discard_contrast,
            discard_contrast_mid,
        ) = self._pair_state(discard_region, "discard_marked", "discard_unmarked")
        (
            lock_marked,
            lock_marked_score,
            lock_unmarked_score,
            lock_contrast,
            lock_contrast_mid,
        ) = self._pair_state(lock_region, "lock_marked", "lock_unmarked")
        return {
            "discard": discard_marked,
            "lock": lock_marked,
            "discard_marked_score": discard_marked_score,
            "discard_unmarked_score": discard_unmarked_score,
            "discard_contrast": discard_contrast,
            "discard_contrast_mid": discard_contrast_mid,
            "lock_marked_score": lock_marked_score,
            "lock_unmarked_score": lock_unmarked_score,
            "lock_contrast": lock_contrast,
            "lock_contrast_mid": lock_contrast_mid,
        }

    def is_discard_marked(self, image_bgr: np.ndarray) -> bool:
        return bool(self.detect_from_bgr(image_bgr)["discard"])

    def is_locked(self, image_bgr: np.ndarray) -> bool:
        return bool(self.detect_from_bgr(image_bgr)["lock"])
