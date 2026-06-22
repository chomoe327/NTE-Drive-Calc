# 详情面板弃置/上锁状态模板匹配。
"""Template-based detection for discard/lock button states in the detail panel."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.scanner.config import ScannerConfig
from src.scanner.window_capture import crop_window_border_from_image, fit_content_rect, scale_region
from src.utils.image_io import imread_unicode
from src.utils.logger import logger

BASE_DISCARD_ROI = (2055, 330, 2145, 420)
BASE_LOCK_ROI = (2265, 330, 2355, 420)
DEFAULT_THRESHOLD = 0.78
RELATIVE_MARGIN = 0.05


class MarkStateDetector:
    def __init__(
        self,
        template_dir: Path,
        confidence_threshold: float = DEFAULT_THRESHOLD,
    ):
        self.template_dir = Path(template_dir)
        self.confidence_threshold = confidence_threshold
        self._templates: dict[str, np.ndarray] = {}
        self._load_templates()

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

    def _pair_state(
        self,
        region_gray: np.ndarray,
        marked_key: str,
        unmarked_key: str,
    ) -> tuple[bool, float, float]:
        marked = self._templates.get(marked_key)
        unmarked = self._templates.get(unmarked_key)
        if marked is None or unmarked is None:
            return False, 0.0, 0.0
        marked_score = self._best_score(region_gray, marked)
        unmarked_score = self._best_score(region_gray, unmarked)
        margin = marked_score - unmarked_score
        is_marked = marked_score > unmarked_score and (
            marked_score >= self.confidence_threshold
            or margin >= RELATIVE_MARGIN
        )
        return is_marked, marked_score, unmarked_score

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
        discard_marked, discard_marked_score, discard_unmarked_score = self._pair_state(
            discard_region,
            "discard_marked",
            "discard_unmarked",
        )
        lock_marked, lock_marked_score, lock_unmarked_score = self._pair_state(
            lock_region,
            "lock_marked",
            "lock_unmarked",
        )
        return {
            "discard": discard_marked,
            "lock": lock_marked,
            "discard_marked_score": discard_marked_score,
            "discard_unmarked_score": discard_unmarked_score,
            "lock_marked_score": lock_marked_score,
            "lock_unmarked_score": lock_unmarked_score,
        }

    def is_discard_marked(self, image_bgr: np.ndarray) -> bool:
        return bool(self.detect_from_bgr(image_bgr)["discard"])

    def is_locked(self, image_bgr: np.ndarray) -> bool:
        return bool(self.detect_from_bgr(image_bgr)["lock"])
