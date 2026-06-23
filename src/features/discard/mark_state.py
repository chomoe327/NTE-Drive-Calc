# 详情面板弃置/上锁状态：先模板定位，再图标亮度判态。
"""Locate discard/lock icons via template matching, then judge state by icon brightness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.scanner.config import ScannerConfig
from src.scanner.window_capture import crop_window_border_from_image, fit_content_rect, scale_region
from src.utils.image_io import imread_unicode
from src.utils.logger import logger

# 中心不变，向四周外扩，给 matchTemplate 留出定位余量。
BASE_DISCARD_ROI = (2025, 300, 2175, 450)
BASE_LOCK_ROI = (2235, 300, 2385, 450)
DEFAULT_THRESHOLD = 0.72
RELATIVE_MARGIN = 0.02
RELATIVE_RATIO = 0.10
LOCATE_MIN_SCORE = 0.10
BRIGHTNESS_HYSTERESIS = 3.0
# 实机截图中模板分普遍偏低（~0.15–0.20）；亮起时两模板分差小，变暗时 marked 模板明显更高。
LOW_SCORE_MARKED_CEILING = 0.35
CLOSE_RELATIVE_MARGIN = 0.25
OPEN_ABSOLUTE_MARGIN = 0.045


@dataclass
class _PairCalibration:
    icon_mask: np.ndarray
    template_shape: tuple[int, int]
    marked_brightness: float
    unmarked_brightness: float

    @property
    def brightness_mid(self) -> float:
        return (self.marked_brightness + self.unmarked_brightness) / 2.0


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
            icon_mask = self._build_icon_mask(marked)
            self._calibrations[prefix] = _PairCalibration(
                icon_mask=icon_mask,
                template_shape=marked.shape[:2],
                marked_brightness=self._icon_brightness(marked, icon_mask),
                unmarked_brightness=self._icon_brightness(unmarked, icon_mask),
            )

    def _build_icon_mask(self, template: np.ndarray) -> np.ndarray:
        h, w = template.shape[:2]
        cy, cx = h / 2.0, w / 2.0
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_r = float(min(cx, cy, w - 1 - cx, h - 1 - cy))
        return dist <= max_r * 0.42

    def _icon_mask_for_patch(self, cal: _PairCalibration, height: int, width: int) -> np.ndarray:
        th, tw = cal.template_shape
        if (height, width) == (th, tw):
            return cal.icon_mask
        return cv2.resize(cal.icon_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0

    def _icon_brightness(self, patch: np.ndarray, icon_mask: np.ndarray) -> float:
        if patch.size == 0:
            return 0.0
        mask = icon_mask
        if mask.shape[:2] != patch.shape[:2]:
            mask = cv2.resize(
                icon_mask.astype(np.uint8),
                (patch.shape[1], patch.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        if not np.any(mask):
            return float(np.mean(patch))
        return float(np.mean(patch[mask]))

    def _locate_patch(
        self,
        region_gray: np.ndarray,
        template: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        if region_gray.size == 0 or template.size == 0:
            return region_gray, 0.0
        th, tw = template.shape[:2]
        rh, rw = region_gray.shape[:2]
        search_template = template
        if rh < th or rw < tw:
            search_template = cv2.resize(template, (rw, rh), interpolation=cv2.INTER_AREA)
            th, tw = rh, rw
        res = cv2.matchTemplate(region_gray, search_template, cv2.TM_CCOEFF_NORMED)
        if res.size == 0:
            return region_gray, 0.0
        locate_score = float(res.max())
        _, _, _, (x, y) = cv2.minMaxLoc(res)
        patch = region_gray[y : y + th, x : x + tw]
        if patch.shape[:2] != (th, tw):
            patch = cv2.resize(patch, (tw, th), interpolation=cv2.INTER_AREA)
        return patch, locate_score

    def _content_scale_factor(self, width: int, height: int) -> float:
        content_rect = fit_content_rect(
            width,
            height,
            (ScannerConfig.BASE_WIDTH, ScannerConfig.BASE_HEIGHT),
        )
        content_h = content_rect[3] - content_rect[1]
        return content_h / ScannerConfig.BASE_HEIGHT

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

    def _scale_templates(
        self,
        marked: np.ndarray,
        unmarked: np.ndarray,
        scale_factor: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if abs(scale_factor - 1.0) <= 0.01:
            return marked, unmarked
        new_w = max(1, int(marked.shape[1] * scale_factor))
        new_h = max(1, int(marked.shape[0] * scale_factor))
        marked_scaled = cv2.resize(marked, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if unmarked.shape != marked.shape:
            unmarked = cv2.resize(
                unmarked,
                (marked.shape[1], marked.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        unmarked_scaled = cv2.resize(unmarked, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return marked_scaled, unmarked_scaled

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

    def _patch_pair_scores(
        self,
        patch: np.ndarray,
        marked: np.ndarray,
        unmarked: np.ndarray,
    ) -> tuple[float, float]:
        if patch.size == 0:
            return 0.0, 0.0
        if unmarked.shape != marked.shape:
            unmarked = cv2.resize(
                unmarked,
                (marked.shape[1], marked.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        return self._best_score(patch, marked), self._best_score(patch, unmarked)

    def _template_says_marked(self, marked_score: float, unmarked_score: float) -> bool:
        margin = marked_score - unmarked_score
        return marked_score > unmarked_score and (
            marked_score >= self.confidence_threshold
            or margin >= RELATIVE_MARGIN
            or margin >= unmarked_score * RELATIVE_RATIO
        )

    def _template_says_unmarked(self, marked_score: float, unmarked_score: float) -> bool:
        margin = unmarked_score - marked_score
        return unmarked_score > marked_score and (
            unmarked_score >= self.confidence_threshold
            or margin >= RELATIVE_MARGIN
            or margin >= marked_score * RELATIVE_RATIO
        )

    def _brightness_says_marked(self, brightness: float, cal: _PairCalibration) -> bool | None:
        mid = cal.brightness_mid
        if brightness >= mid + BRIGHTNESS_HYSTERESIS:
            return True
        if brightness <= mid - BRIGHTNESS_HYSTERESIS:
            return False
        return None

    def _brightness_trustworthy(self, brightness: float, cal: _PairCalibration) -> bool:
        """实机 ROI 亮度常低于模板校准值，此时不应让亮度一票否决。"""
        floor = min(cal.marked_brightness, cal.unmarked_brightness) - BRIGHTNESS_HYSTERESIS
        ceiling = max(cal.marked_brightness, cal.unmarked_brightness) + BRIGHTNESS_HYSTERESIS
        return floor <= brightness <= ceiling

    def _resolve_marked_state(
        self,
        marked_score: float,
        unmarked_score: float,
        brightness: float,
        cal: _PairCalibration | None,
        *,
        locate_ok: bool,
    ) -> bool:
        margin = marked_score - unmarked_score
        rel_margin = margin / max(marked_score, 1e-6)

        if marked_score >= self.confidence_threshold and margin > 0:
            return True
        if unmarked_score >= self.confidence_threshold and margin < 0:
            return False

        if locate_ok:
            if (
                marked_score <= LOW_SCORE_MARKED_CEILING
                and margin > 0
                and rel_margin < CLOSE_RELATIVE_MARGIN
            ):
                return True
            if margin >= OPEN_ABSOLUTE_MARGIN:
                return False

        if cal is not None and self._brightness_trustworthy(brightness, cal):
            brightness_state = self._brightness_says_marked(brightness, cal)
            if brightness_state is not None:
                return brightness_state

        if self._template_says_marked(marked_score, unmarked_score):
            return True
        if self._template_says_unmarked(marked_score, unmarked_score):
            return False
        if cal is not None:
            return brightness >= cal.brightness_mid
        return marked_score >= unmarked_score

    def _pair_state(
        self,
        region_gray: np.ndarray,
        marked_key: str,
        unmarked_key: str,
        scale_factor: float = 1.0,
    ) -> tuple[bool, float, float, float, float, float]:
        marked = self._templates.get(marked_key)
        unmarked = self._templates.get(unmarked_key)
        if marked is None or unmarked is None:
            return False, 0.0, 0.0, 0.0, 0.0, 0.0

        marked, unmarked = self._scale_templates(marked, unmarked, scale_factor)

        region_marked_score = self._best_score(region_gray, marked)
        region_unmarked_score = self._best_score(region_gray, unmarked)
        marked_score = region_marked_score
        unmarked_score = region_unmarked_score

        prefix = marked_key[: -len("_marked")]
        cal = self._calibrations.get(prefix)
        locate_score = 0.0
        brightness = 0.0
        brightness_mid = cal.brightness_mid if cal is not None else 0.0
        locate_ok = False

        if cal is not None:
            patch, locate_score = self._locate_patch(region_gray, marked)
            icon_mask = self._icon_mask_for_patch(cal, patch.shape[0], patch.shape[1])
            brightness = self._icon_brightness(patch, icon_mask)
            if locate_score >= LOCATE_MIN_SCORE:
                locate_ok = True
                marked_score, unmarked_score = self._patch_pair_scores(patch, marked, unmarked)

        is_marked = self._resolve_marked_state(
            marked_score,
            unmarked_score,
            brightness,
            cal,
            locate_ok=locate_ok,
        )

        return (
            is_marked,
            marked_score,
            unmarked_score,
            brightness,
            brightness_mid,
            locate_score,
        )

    def detect_from_bgr(self, image_bgr: np.ndarray) -> dict[str, bool | float]:
        image_bgr = crop_window_border_from_image(image_bgr)
        height, width = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        scale_factor = self._content_scale_factor(width, height)
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
            discard_brightness,
            discard_brightness_mid,
            discard_locate_score,
        ) = self._pair_state(
            discard_region,
            "discard_marked",
            "discard_unmarked",
            scale_factor,
        )
        (
            lock_marked,
            lock_marked_score,
            lock_unmarked_score,
            lock_brightness,
            lock_brightness_mid,
            lock_locate_score,
        ) = self._pair_state(
            lock_region,
            "lock_marked",
            "lock_unmarked",
            scale_factor,
        )
        return {
            "discard": discard_marked,
            "lock": lock_marked,
            "discard_marked_score": discard_marked_score,
            "discard_unmarked_score": discard_unmarked_score,
            "discard_brightness": discard_brightness,
            "discard_brightness_mid": discard_brightness_mid,
            "discard_locate_score": discard_locate_score,
            "lock_marked_score": lock_marked_score,
            "lock_unmarked_score": lock_unmarked_score,
            "lock_brightness": lock_brightness,
            "lock_brightness_mid": lock_brightness_mid,
            "lock_locate_score": lock_locate_score,
            # 兼容旧日志字段名
            "discard_contrast": discard_brightness,
            "discard_contrast_mid": discard_brightness_mid,
            "lock_contrast": lock_brightness,
            "lock_contrast_mid": lock_brightness_mid,
        }

    def is_discard_marked(self, image_bgr: np.ndarray) -> bool:
        return bool(self.detect_from_bgr(image_bgr)["discard"])

    def is_locked(self, image_bgr: np.ndarray) -> bool:
        return bool(self.detect_from_bgr(image_bgr)["lock"])
