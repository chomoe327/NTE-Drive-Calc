# 详情面板弃置/上锁状态：Canny 边缘定位 + 图标亮像素占比判态。
"""Locate discard/lock icons via edge matching, then judge state by bright-pixel ratio."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.scanner.config import ScannerConfig
from src.scanner.window_capture import crop_window_border_from_image, fit_content_rect, scale_region
from src.utils.image_io import imread_unicode
from src.utils.logger import logger

# 1920x1080 实机标定（2560x1440 基准），向四周外扩供边缘定位搜索。
BASE_DISCARD_ROI = (2214, 245, 2364, 394)
BASE_LOCK_ROI = (2304, 238, 2453, 388)

LOCATE_MIN_SCORE_EDGE = 0.30
BRIGHT_PIXEL_PERCENT_THRESHOLD = 0.50
BRIGHT_PIXEL_VALUE = 100
MASK_THRESHOLD = 150
CANNY_LOW = 50
CANNY_HIGH = 150
SCALE_PYRAMID_SPREAD = 0.12
SCALE_PYRAMID_STEP = 0.02


@dataclass
class _MarkCalibration:
    edge_locator: np.ndarray
    icon_binary_mask: np.ndarray
    base_shape: tuple[int, int]


class MarkStateDetector:
    def __init__(
        self,
        template_dir: Path,
        confidence_threshold: float = 0.72,
    ):
        self.template_dir = Path(template_dir)
        self.confidence_threshold = confidence_threshold
        self._templates: dict[str, np.ndarray] = {}
        self._calibrations: dict[str, _MarkCalibration] = {}
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

    def _make_binary_mask(self, template: np.ndarray, thresh_val: int = MASK_THRESHOLD) -> np.ndarray:
        _, binary = cv2.threshold(template, thresh_val, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        return mask > 0

    def _combine_edge_locator(self, *templates: np.ndarray | None) -> np.ndarray | None:
        edges = []
        for template in templates:
            if template is None:
                continue
            edges.append(cv2.Canny(template, CANNY_LOW, CANNY_HIGH))
        if not edges:
            return None
        combined = edges[0]
        for edge in edges[1:]:
            if edge.shape != combined.shape:
                edge = cv2.resize(
                    edge,
                    (combined.shape[1], combined.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            combined = cv2.max(combined, edge)
        return combined

    def _calibrate_pairs(self) -> None:
        discard_marked = self._templates.get("discard_marked")
        discard_unmarked = self._templates.get("discard_unmarked")
        if discard_marked is not None and discard_unmarked is not None:
            edge_loc = self._combine_edge_locator(discard_marked, discard_unmarked)
            assert edge_loc is not None
            self._calibrations["discard"] = _MarkCalibration(
                edge_locator=edge_loc,
                icon_binary_mask=self._make_binary_mask(discard_marked),
                base_shape=discard_unmarked.shape[:2],
            )

        lock_marked = self._templates.get("lock_marked")
        lock_unmarked = self._templates.get("lock_unmarked")
        if lock_marked is not None and lock_unmarked is not None:
            edge_loc = self._combine_edge_locator(lock_marked, lock_unmarked)
            assert edge_loc is not None
            self._calibrations["lock"] = _MarkCalibration(
                edge_locator=edge_loc,
                icon_binary_mask=self._make_binary_mask(lock_marked),
                base_shape=lock_unmarked.shape[:2],
            )

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

    def _scale_pyramid(self, scale_factor: float) -> list[float]:
        if abs(scale_factor - 1.0) <= 0.01:
            center = 1.0
        else:
            center = max(0.1, scale_factor)
        spread = SCALE_PYRAMID_SPREAD
        step = SCALE_PYRAMID_STEP
        scales: set[float] = set()
        value = center - spread
        while value <= center + spread + step * 0.5:
            scales.add(round(max(0.1, value), 4))
            value += step
        return sorted(scales)

    def _locate_patch_multiscale(
        self,
        region_gray: np.ndarray,
        edge_template: np.ndarray,
        scale_factor: float,
    ) -> tuple[np.ndarray | None, float]:
        if region_gray.size == 0 or edge_template.size == 0:
            return None, 0.0

        region_edge = cv2.Canny(region_gray, CANNY_LOW, CANNY_HIGH)
        base_h, base_w = edge_template.shape[:2]

        best_score = 0.0
        best_loc = (0, 0)
        best_patch_shape = (base_h, base_w)

        for scale in self._scale_pyramid(scale_factor):
            new_w = max(1, int(round(base_w * scale)))
            new_h = max(1, int(round(base_h * scale)))
            if new_h > region_edge.shape[0] or new_w > region_edge.shape[1]:
                continue
            scaled_template = cv2.resize(edge_template, (new_w, new_h), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(region_edge, scaled_template, cv2.TM_CCOEFF_NORMED)
            if res.size == 0:
                continue
            loc_score = float(res.max())
            if loc_score > best_score:
                best_score = loc_score
                best_patch_shape = (new_h, new_w)
                _, _, _, best_loc = cv2.minMaxLoc(res)

        if best_score < LOCATE_MIN_SCORE_EDGE:
            return None, best_score

        h, w = best_patch_shape
        x, y = best_loc
        patch = region_gray[y : y + h, x : x + w]
        if patch.size == 0:
            return None, best_score
        return patch, best_score

    def _bright_pixel_percent(self, patch: np.ndarray, icon_binary_mask: np.ndarray) -> float:
        if patch.size == 0:
            return 0.0
        mask = cv2.resize(
            icon_binary_mask.astype(np.uint8),
            (patch.shape[1], patch.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
        if not np.any(mask):
            return 0.0
        pixels = patch[mask]
        return float(np.mean(pixels > BRIGHT_PIXEL_VALUE))

    def _patch_template_scores(
        self,
        patch: np.ndarray,
        pair_prefix: str,
        scale_factor: float,
    ) -> tuple[float, float]:
        marked = self._templates.get(f"{pair_prefix}_marked")
        unmarked = self._templates.get(f"{pair_prefix}_unmarked")
        if patch is None or patch.size == 0 or marked is None or unmarked is None:
            return 0.0, 0.0
        if abs(scale_factor - 1.0) > 0.01:
            new_w = max(1, int(round(marked.shape[1] * scale_factor)))
            new_h = max(1, int(round(marked.shape[0] * scale_factor)))
            marked = cv2.resize(marked, (new_w, new_h), interpolation=cv2.INTER_AREA)
            if unmarked.shape != marked.shape:
                unmarked = cv2.resize(unmarked, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return self._match_score(patch, marked), self._match_score(patch, unmarked)

    def _match_score(self, region_gray: np.ndarray, template: np.ndarray) -> float:
        if region_gray.size == 0 or template.size == 0:
            return 0.0
        th, tw = template.shape[:2]
        rh, rw = region_gray.shape[:2]
        if th > rh or tw > rw:
            scale = min(rw / tw, rh / th)
            new_w = max(1, int(round(tw * scale)))
            new_h = max(1, int(round(th * scale)))
            template = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if template.shape[0] > region_gray.shape[0] or template.shape[1] > region_gray.shape[1]:
            return 0.0
        res = cv2.matchTemplate(region_gray, template, cv2.TM_CCOEFF_NORMED)
        return float(res.max()) if res.size else 0.0

    def _is_patch_marked(self, patch: np.ndarray, cal: _MarkCalibration) -> tuple[bool, float]:
        bright_percent = self._bright_pixel_percent(patch, cal.icon_binary_mask)
        return bright_percent > BRIGHT_PIXEL_PERCENT_THRESHOLD, bright_percent

    def _pair_state(
        self,
        region_gray: np.ndarray,
        pair_prefix: str,
        scale_factor: float = 1.0,
    ) -> tuple[bool, float, float, float, float, float]:
        cal = self._calibrations.get(pair_prefix)
        if cal is None:
            return False, 0.0, 0.0, 0.0, 0.0, 0.0

        patch, locate_score = self._locate_patch_multiscale(
            region_gray,
            cal.edge_locator,
            scale_factor,
        )
        if patch is None:
            return False, 0.0, 0.0, 0.0, BRIGHT_PIXEL_PERCENT_THRESHOLD * 100.0, locate_score

        is_marked, bright_percent = self._is_patch_marked(patch, cal)
        marked_score, unmarked_score = self._patch_template_scores(patch, pair_prefix, scale_factor)
        brightness = bright_percent * 100.0
        brightness_mid = BRIGHT_PIXEL_PERCENT_THRESHOLD * 100.0

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
        ) = self._pair_state(discard_region, "discard", scale_factor)
        (
            lock_marked,
            lock_marked_score,
            lock_unmarked_score,
            lock_brightness,
            lock_brightness_mid,
            lock_locate_score,
        ) = self._pair_state(lock_region, "lock", scale_factor)
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
            "discard_contrast": discard_brightness,
            "discard_contrast_mid": discard_brightness_mid,
            "lock_contrast": lock_brightness,
            "lock_contrast_mid": lock_brightness_mid,
        }

    def is_discard_marked(self, image_bgr: np.ndarray) -> bool:
        return bool(self.detect_from_bgr(image_bgr)["discard"])

    def is_locked(self, image_bgr: np.ndarray) -> bool:
        return bool(self.detect_from_bgr(image_bgr)["lock"])
