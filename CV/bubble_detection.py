from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np


@dataclass
class ManualROIConfig:
    # enabled=True полностью отключает автоматический поиск капилляра.
    enabled: bool = False
    center_xy: tuple[float, float] = (320.0, 240.0)
    angle_deg: float = 0.0       # Угол оси капилляра в исходном изображении.
    length_px: float = 400.0     # Длина рабочей области вдоль оси.
    width_px: float = 60.0       # Ширина области между/вокруг стенок.


@dataclass
class AutoROIConfig:
    canny_low: int = 40
    canny_high: int = 120
    hough_threshold: int = 50
    min_line_length_px: int = 120
    max_line_gap_px: int = 30
    angle_tolerance_deg: float = 8.0
    min_width_px: float = 12.0
    max_width_px: float = 140.0


@dataclass
class DetectorConfig:
    polarity: str = "dark"       # "dark": капля темнее фона; "bright": светлее.
    reference: str = "center"   # "left", "right" или "center" — характерная точка.
    profile_band_ratio: float = 0.5  # Центральная часть сечения для профиля.
    gaussian_sigma: float = 2.0
    noise_k: float = 4.0         # Порог относительно робастной оценки шума.
    min_contrast: float = 12.0   # Минимальный перепад яркости, 0..255.
    close_gap_px: int = 7        # Склеивание небольших разрывов маски капли.
    min_drop_length_px: int = 8
    max_drop_length_px: int = 180
    min_quality: float = 0.25


@dataclass
class CalibrationConfig:
    # x_mm = direction * (x_px - zero_px) * mm_per_px.
    mm_per_px: float = 0.02
    zero_px: float = 0.0
    direction: int = 1           # -1 позволяет развернуть физическую ось.
    ruler_step_mm: float = 5.0


@dataclass
class FilterConfig:
    median_window: int = 5       # 1 отключает медианное сглаживание.
    ema_alpha: float = 0.35      # 1.0 отключает экспоненциальное сглаживание.
    reset_after_invalid: int = 10


@dataclass
class AppConfig:
    manual_roi: ManualROIConfig = field(default_factory=ManualROIConfig)
    auto_roi: AutoROIConfig = field(default_factory=AutoROIConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    filtering: FilterConfig = field(default_factory=FilterConfig)


def load_config(path: Optional[str]) -> AppConfig:
    """Загружает JSON поверх значений по умолчанию, не требуя полного файла."""
    cfg = AppConfig()
    if path is None:
        return cfg

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sections = {
        "manual_roi": cfg.manual_roi,
        "auto_roi": cfg.auto_roi,
        "detector": cfg.detector,
        "calibration": cfg.calibration,
        "filtering": cfg.filtering,
    }
    for section_name, values in data.items():
        if section_name not in sections or not isinstance(values, dict):
            raise ValueError(f"Неизвестная секция конфигурации: {section_name}")
        section = sections[section_name]
        for key, value in values.items():
            if not hasattr(section, key):
                raise ValueError(f"Неизвестный параметр: {section_name}.{key}")
            setattr(section, key, value)

    validate_config(cfg)
    return cfg


def validate_config(cfg: AppConfig) -> None:
    if cfg.detector.polarity not in {"dark", "bright"}:
        raise ValueError("detector.polarity должен быть 'dark' или 'bright'")
    if cfg.detector.reference not in {"left", "right", "center"}:
        raise ValueError("detector.reference должен быть 'left', 'right' или 'center'")
    if cfg.calibration.mm_per_px <= 0:
        raise ValueError("calibration.mm_per_px должен быть положительным")
    if cfg.calibration.direction not in {-1, 1}:
        raise ValueError("calibration.direction должен быть -1 или 1")
    if not 0 < cfg.filtering.ema_alpha <= 1:
        raise ValueError("filtering.ema_alpha должен быть в интервале (0; 1]")
    if cfg.manual_roi.length_px <= 0 or cfg.manual_roi.width_px <= 0:
        raise ValueError("Размеры ручной области должны быть положительными")

#Геометрия капилляра и автоматическое определение рабочей области.

@dataclass
class CapillaryGeometry:
    center: np.ndarray           # Центр рабочей области в координатах кадра.
    axis: np.ndarray             # Единичный вектор вдоль капилляра.
    length_px: float
    width_px: float

    @classmethod
    def from_manual(cls, cfg: ManualROIConfig) -> "CapillaryGeometry":
        angle = math.radians(cfg.angle_deg)
        return cls(
            center=np.asarray(cfg.center_xy, dtype=np.float64),
            axis=np.array([math.cos(angle), math.sin(angle)], dtype=np.float64),
            length_px=float(cfg.length_px),
            width_px=float(cfg.width_px),
        )

    @property
    def normal(self) -> np.ndarray:
        return np.array([-self.axis[1], self.axis[0]], dtype=np.float64)

    def point_at(self, x_px: float, offset_normal: float = 0.0) -> np.ndarray:
        """Переводит локальную координату вдоль оси обратно в координаты кадра."""
        return (
            self.center
            + self.axis * (float(x_px) - self.length_px / 2.0)
            + self.normal * offset_normal
        )

    def warp_matrix(self) -> tuple[np.ndarray, tuple[int, int]]:
        """Матрица поворота: после warpAffine капилляр расположен горизонтально."""
        out_w = max(2, int(math.ceil(self.length_px)))
        out_h = max(2, int(math.ceil(self.width_px)))
        matrix = np.zeros((2, 3), dtype=np.float32)
        matrix[0, :2] = self.axis
        matrix[1, :2] = self.normal
        matrix[0, 2] = out_w / 2.0 - float(self.axis @ self.center)
        matrix[1, 2] = out_h / 2.0 - float(self.normal @ self.center)
        return matrix, (out_w, out_h)


def angle_difference_pi(a: np.ndarray, b: np.ndarray) -> float:
    """Угол между неориентированными прямыми, результат в диапазоне [0; pi/2]."""
    cross = abs(float(a[0] * b[1] - a[1] * b[0]))
    return math.asin(min(1.0, cross))


class AutoCapillaryLocator:
    """Ищет две длинные параллельные линии — предполагаемые стенки капилляра."""

    def __init__(self, cfg: AutoROIConfig):
        self.cfg = cfg

    def locate(self, frame_bgr: np.ndarray) -> CapillaryGeometry:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, self.cfg.canny_low, self.cfg.canny_high)
        raw_lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 720.0,
            threshold=self.cfg.hough_threshold,
            minLineLength=self.cfg.min_line_length_px,
            maxLineGap=self.cfg.max_line_gap_px,
        )
        if raw_lines is None:
            raise RuntimeError("Капилляр не найден: линии Хафа отсутствуют")

        segments = []
        for x1, y1, x2, y2 in raw_lines[:, 0, :]:
            p1 = np.array([x1, y1], dtype=np.float64)
            p2 = np.array([x2, y2], dtype=np.float64)
            delta = p2 - p1
            length = float(np.linalg.norm(delta))
            if length < self.cfg.min_line_length_px:
                continue
            segments.append((p1, p2, delta / length, length, (p1 + p2) / 2.0))

        if len(segments) < 2:
            raise RuntimeError("Капилляр не найден: недостаточно длинных линий")

        # Ограничение числа кандидатов предотвращает квадратичный рост вычислений
        # на изображениях с большим количеством посторонних линий.
        segments = sorted(segments, key=lambda item: item[3], reverse=True)[:80]
        reference_axis = segments[0][2]
        tolerance = math.radians(self.cfg.angle_tolerance_deg)
        parallel = [s for s in segments if angle_difference_pi(reference_axis, s[2]) <= tolerance]

        best_score = -float("inf")
        best_geometry: Optional[CapillaryGeometry] = None
        for i, first in enumerate(parallel):
            for second in parallel[i + 1:]:
                p1, p2, axis1, len1, mid1 = first
                q1, q2, axis2, len2, mid2 = second
                axis2 = axis2 if float(axis1 @ axis2) >= 0 else -axis2
                axis = axis1 * len1 + axis2 * len2
                axis /= np.linalg.norm(axis)
                normal = np.array([-axis[1], axis[0]])

                separation = abs(float((mid2 - mid1) @ normal))
                if not self.cfg.min_width_px <= separation <= self.cfg.max_width_px:
                    continue

                # Стенки должны не только быть параллельны, но и перекрываться
                # по длине. Это отсекает случайные линии из разных частей кадра.
                all_points = (p1, p2, q1, q2)
                projections = [float((p - mid1) @ axis) for p in all_points]
                first_interval = sorted(projections[:2])
                second_interval = sorted(projections[2:])
                overlap = min(first_interval[1], second_interval[1]) - max(
                    first_interval[0], second_interval[0]
                )
                if overlap < 0.4 * min(len1, len2):
                    continue

                # Середина объединения проекций точнее простого среднего центров
                # отрезков, если Хаф нашел стенки разной длины или со сдвигом.
                length = max(projections) - min(projections)
                center_t = (max(projections) + min(projections)) / 2.0
                center_n = float((mid2 - mid1) @ normal) / 2.0
                center = mid1 + axis * center_t + normal * center_n
                score = overlap + 0.25 * (len1 + len2) - 0.2 * separation
                if score > best_score:
                    best_score = score
                    best_geometry = CapillaryGeometry(center, axis, length, separation)

        if best_geometry is None:
            raise RuntimeError(
                "Капилляр не найден: нет пары подходящих параллельных стенок. "
                "Задайте manual_roi или настройте параметры auto_roi."
            )
        return best_geometry


def rectify_to_capillary(gray: np.ndarray, geometry: CapillaryGeometry) -> np.ndarray:
    matrix, size = geometry.warp_matrix()
    return cv2.warpAffine(
        gray,
        matrix,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


#Детектор капли

@dataclass
class DropDetection:
    valid: bool
    position_px: Optional[float]
    quality: float
    reason: str
    bounds_px: Optional[tuple[float, float]] = None


class ProfileDropDetector:
    """Сегментирует каплю по усредненному профилю яркости вдоль капилляра."""

    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg

    @staticmethod
    def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
        """Возвращает непрерывные участки True как полуинтервалы [start; end)."""
        padded = np.pad(mask.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        return list(zip(starts.tolist(), ends.tolist()))

    def _profile(self, roi_gray: np.ndarray) -> np.ndarray:
        height, width = roi_gray.shape
        band = int(round(height * self.cfg.profile_band_ratio))
        band = max(1, min(height, band))
        y0 = (height - band) // 2
        central = roi_gray[y0:y0 + band, :].astype(np.float32)
        # Медиана по сечению менее чувствительна к бликам и стенкам, чем среднее.
        profile = np.median(central, axis=0)
        sigma = max(0.0, float(self.cfg.gaussian_sigma))
        if sigma > 0:
            profile = cv2.GaussianBlur(profile.reshape(1, -1), (0, 0), sigma).ravel()
        return profile

    def detect(
        self,
        roi_gray: np.ndarray,
        background_roi_gray: Optional[np.ndarray] = None,
    ) -> DropDetection:
        if roi_gray.ndim != 2 or min(roi_gray.shape) < 2:
            return DropDetection(False, None, 0.0, "Некорректная область капилляра")

        profile = self._profile(roi_gray)
        if background_roi_gray is not None:
            if background_roi_gray.shape != roi_gray.shape:
                return DropDetection(False, None, 0.0, "Фон имеет другой размер")
            baseline_profile = self._profile(background_roi_gray)
        else:
            baseline_profile = np.full_like(profile, np.median(profile))

        # В signal капля всегда должна быть положительным участком.
        if self.cfg.polarity == "dark":
            signal = baseline_profile - profile
        else:
            signal = profile - baseline_profile

        offset = float(np.median(signal))
        centered = signal - offset
        mad = float(np.median(np.abs(centered - np.median(centered))))
        noise = max(1.0, 1.4826 * mad)
        threshold = max(float(self.cfg.min_contrast), self.cfg.noise_k * noise)

        mask = (centered >= threshold).astype(np.uint8)[None, :]
        if self.cfg.close_gap_px > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (int(self.cfg.close_gap_px), 1)
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)),
        )
        runs = self._runs(mask.ravel() > 0)
        if not runs:
            return DropDetection(False, None, 0.0, "Капля не обнаружена")

        candidates = []
        for start, end in runs:
            length = end - start
            if not self.cfg.min_drop_length_px <= length <= self.cfg.max_drop_length_px:
                continue
            segment = centered[start:end]
            peak = float(np.max(segment))
            mean = float(np.mean(segment))
            # Интегральный критерий предпочитает выраженную каплю слабым шумам.
            score = mean * math.sqrt(length)
            candidates.append((score, start, end, peak))
        if not candidates:
            return DropDetection(False, None, 0.0, "Нет участка допустимой длины")

        _, start, end, peak = max(candidates, key=lambda item: item[0])
        if start == 0 or end == len(profile):
            return DropDetection(
                False, None, 0.0, "Капля касается границы области", (start, end - 1)
            )

        gradient = np.gradient(centered)
        edge_strength = 0.5 * (
            abs(float(gradient[start - 1])) + abs(float(gradient[end - 1]))
        )
        contrast_quality = min(1.0, peak / max(3.0 * threshold, 1e-6))
        edge_quality = min(1.0, edge_strength / max(3.0 * noise, 1e-6))
        quality = float(np.clip(0.7 * contrast_quality + 0.3 * edge_quality, 0.0, 1.0))

        if self.cfg.reference == "left":
            position = float(start)
        elif self.cfg.reference == "right":
            position = float(end - 1)
        else:
            position = (start + end - 1) / 2.0

        valid = quality >= self.cfg.min_quality
        reason = "OK" if valid else "Низкая достоверность детекции"
        return DropDetection(valid, position, quality, reason, (start, end - 1))


# Временной фильтр, калибровка и результат измерения

class PositionFilter:
    """Сглаживает только достоверные значения"""

    def __init__(self, cfg: FilterConfig):
        self.cfg = cfg
        self.window: deque[float] = deque(maxlen=max(1, cfg.median_window))
        self.state: Optional[float] = None
        self.invalid_count = 0

    def update(self, detection: DropDetection) -> Optional[float]:
        if not detection.valid or detection.position_px is None:
            self.invalid_count += 1
            if self.invalid_count >= self.cfg.reset_after_invalid:
                self.reset()
            return None

        self.invalid_count = 0
        value = float(detection.position_px)
        self.window.append(value)
        median_value = float(np.median(self.window))
        if self.state is None:
            self.state = median_value
        else:
            alpha = self.cfg.ema_alpha
            self.state = alpha * median_value + (1.0 - alpha) * self.state
        return self.state

    def reset(self) -> None:
        self.window.clear()
        self.state = None
        self.invalid_count = 0


@dataclass
class Measurement:
    frame_index: int
    timestamp_s: float
    valid: bool
    position_px: Optional[float]
    raw_position_mm: Optional[float]
    position_mm: Optional[float]
    quality: float
    reason: str


@dataclass
class VisionResult:
    measurement: Measurement
    geometry: Optional[CapillaryGeometry]
    bounds_px: Optional[tuple[float, float]] = None


class VisionPipeline:
    """Единая точка входа для будущего десктопного интерфейса."""

    def __init__(self, cfg: AppConfig, background_bgr: Optional[np.ndarray] = None):
        self.cfg = cfg
        self.locator = AutoCapillaryLocator(cfg.auto_roi)
        self.detector = ProfileDropDetector(cfg.detector)
        self.position_filter = PositionFilter(cfg.filtering)
        self.background_bgr = background_bgr
        self.geometry: Optional[CapillaryGeometry] = None
        self.last_error = ""

    def _get_geometry(self, frame_bgr: np.ndarray) -> CapillaryGeometry:
        if self.cfg.manual_roi.enabled:
            # Ручная коррекция всегда имеет приоритет над автоматикой.
            self.geometry = CapillaryGeometry.from_manual(self.cfg.manual_roi)
        elif self.geometry is None:
            # Капилляр предполагается неподвижным относительно камеры, поэтому
            # автоматический поиск выполняется один раз, а не на каждом кадре.
            self.geometry = self.locator.locate(frame_bgr)
        return self.geometry

    def process(
        self, frame_bgr: np.ndarray, frame_index: int, timestamp_s: float
    ) -> VisionResult:
        try:
            geometry = self._get_geometry(frame_bgr)
        except RuntimeError as exc:
            self.last_error = str(exc)
            detection = DropDetection(False, None, 0.0, self.last_error)
            self.position_filter.update(detection)
            return VisionResult(
                Measurement(
                    frame_index, timestamp_s, False, None, None, None, 0.0, self.last_error
                ),
                None,
            )

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        roi = rectify_to_capillary(gray, geometry)
        background_roi = None
        if self.background_bgr is not None:
            if self.background_bgr.shape[:2] != frame_bgr.shape[:2]:
                raise ValueError("Размер эталонного фона не совпадает с размером кадра")
            background_gray = cv2.cvtColor(self.background_bgr, cv2.COLOR_BGR2GRAY)
            background_roi = rectify_to_capillary(background_gray, geometry)

        detection = self.detector.detect(roi, background_roi)
        filtered_px = self.position_filter.update(detection)
        calibration = self.cfg.calibration

        raw_mm = None
        filtered_mm = None
        if detection.valid and detection.position_px is not None and filtered_px is not None:
            raw_mm = (
                calibration.direction
                * (detection.position_px - calibration.zero_px)
                * calibration.mm_per_px
            )
            filtered_mm = (
                calibration.direction
                * (filtered_px - calibration.zero_px)
                * calibration.mm_per_px
            )

        measurement = Measurement(
            frame_index=frame_index,
            timestamp_s=float(timestamp_s),
            valid=bool(detection.valid and filtered_mm is not None),
            position_px=filtered_px,
            raw_position_mm=raw_mm,
            position_mm=filtered_mm,
            quality=detection.quality,
            reason=detection.reason,
        )
        return VisionResult(measurement, geometry, detection.bounds_px)


# Журнал и визуализация. Недостоверные кадры тоже записываются, но имеют
#    valid=0 и пустые координаты, поэтому не попадут в графики как измерения.


class MeasurementLogger:
    def __init__(self, path: str):
        self.file = open(path, "w", newline="", encoding="utf-8-sig")
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=[
                "frame_index",
                "timestamp_s",
                "valid",
                "position_px",
                "raw_position_mm",
                "position_mm",
                "quality",
                "reason",
            ],
        )
        self.writer.writeheader()

    def write(self, item: Measurement) -> None:
        self.writer.writerow(
            {
                "frame_index": item.frame_index,
                "timestamp_s": f"{item.timestamp_s:.6f}",
                "valid": int(item.valid),
                "position_px": "" if item.position_px is None else f"{item.position_px:.4f}",
                "raw_position_mm": "" if item.raw_position_mm is None else f"{item.raw_position_mm:.6f}",
                "position_mm": "" if item.position_mm is None else f"{item.position_mm:.6f}",
                "quality": f"{item.quality:.4f}",
                "reason": item.reason,
            }
        )
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def as_int_point(point: np.ndarray) -> tuple[int, int]:
    return int(round(float(point[0]))), int(round(float(point[1])))


def draw_overlay(frame: np.ndarray, result: VisionResult) -> np.ndarray:
    output = frame.copy()
    geometry = result.geometry
    measurement = result.measurement
    if geometry is None:
        # Стандартные шрифты OpenCV не выводят кириллицу; подробная русская
        # причина сохраняется в CSV, а на кадр выводится короткий ASCII-статус.
        cv2.putText(
            output, "INVALID: capillary ROI not found", (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA,
        )
        return output

    axis, normal = geometry.axis, geometry.normal
    half_l, half_w = geometry.length_px / 2.0, geometry.width_px / 2.0
    corners = [
        geometry.center - axis * half_l - normal * half_w,
        geometry.center + axis * half_l - normal * half_w,
        geometry.center + axis * half_l + normal * half_w,
        geometry.center - axis * half_l + normal * half_w,
    ]
    cv2.polylines(
        output, [np.asarray([as_int_point(p) for p in corners], dtype=np.int32)],
        True, (255, 180, 0), 1, cv2.LINE_AA,
    )
    cv2.line(
        output, as_int_point(geometry.point_at(0)),
        as_int_point(geometry.point_at(geometry.length_px)), (255, 100, 0), 1, cv2.LINE_AA,
    )

    # Мениски обозначаются даже при низкой достоверности, чтобы было видно,
    # какой именно участок алгоритм посчитал каплей.
    if result.bounds_px is not None:
        for bound in result.bounds_px:
            p1 = geometry.point_at(bound, -0.45 * geometry.width_px)
            p2 = geometry.point_at(bound, 0.45 * geometry.width_px)
            cv2.line(output, as_int_point(p1), as_int_point(p2), (0, 165, 255), 2)

    if measurement.valid and measurement.position_px is not None:
        marker = geometry.point_at(measurement.position_px)
        cv2.circle(output, as_int_point(marker), 7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.line(
            output, as_int_point(marker - normal * 12), as_int_point(marker + normal * 12),
            (0, 0, 255), 2, cv2.LINE_AA,
        )

    color = (0, 220, 0) if measurement.valid else (0, 0, 255)
    if measurement.valid and measurement.position_mm is not None:
        text = f"x={measurement.position_mm:.3f} mm  q={measurement.quality:.2f}"
    else:
        text = f"INVALID  q={measurement.quality:.2f}"
    cv2.putText(output, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return output


def draw_ruler(output: np.ndarray, geometry: CapillaryGeometry, cfg: CalibrationConfig) -> None:
    """Рисует линейку отдельно, чтобы не скрывать калибровку внутри VisionResult."""
    step = cfg.ruler_step_mm
    if step <= 0:
        return
    x_mm_0 = cfg.direction * (0.0 - cfg.zero_px) * cfg.mm_per_px
    x_mm_1 = cfg.direction * (geometry.length_px - cfg.zero_px) * cfg.mm_per_px
    low, high = sorted((x_mm_0, x_mm_1))
    value = math.ceil(low / step) * step
    while value <= high + 1e-9:
        x_px = cfg.zero_px + cfg.direction * value / cfg.mm_per_px
        if 0 <= x_px <= geometry.length_px:
            p = geometry.point_at(x_px)
            cv2.line(
                output, as_int_point(p - geometry.normal * 5),
                as_int_point(p + geometry.normal * 5), (180, 180, 180), 1, cv2.LINE_AA,
            )
            label_p = p + geometry.normal * 18
            cv2.putText(
                output, f"{value:g}", as_int_point(label_p), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (220, 220, 220), 1, cv2.LINE_AA,
            )
        value += step


# Источники кадров и цикл обработки. Этот слой можно не использовать при
#    интеграции в GUI: достаточно вызывать VisionPipeline.process из своего окна.


def iter_video_source(source: str) -> Iterator[tuple[int, float, np.ndarray, float]]:
    numeric_source = int(source) if source.isdigit() else source
    capture = cv2.VideoCapture(numeric_source)
    if not capture.isOpened():
        raise RuntimeError(f"Не удалось открыть источник видео: {source}")

    is_camera = isinstance(numeric_source, int)
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    started = time.monotonic()
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if is_camera:
                timestamp = time.monotonic() - started
            else:
                timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if not math.isfinite(timestamp) or timestamp < 0:
                    timestamp = index / fps
            yield index, timestamp, frame, fps
            index += 1
    finally:
        capture.release()


def iter_input_source(source: str) -> Iterator[tuple[int, float, np.ndarray, float]]:
    """Выбирает обработку одиночного изображения или видеопотока по расширению."""
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    if Path(source).suffix.lower() in image_extensions:
        image = cv2.imread(source, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Не удалось прочитать изображение: {source}")
        yield 0, 0.0, image, 30.0
        return
    yield from iter_video_source(source)


def synthetic_frames(count: int = 180, fps: float = 30.0) -> Iterator[tuple[int, float, np.ndarray, float]]:
    """Синтетический капилляр для проверки каркаса без реальных фото и видео."""
    rng = np.random.default_rng(7)
    for index in range(count):
        frame = np.full((180, 640, 3), 45, dtype=np.uint8)
        cv2.rectangle(frame, (70, 70), (570, 130), (190, 190, 190), -1)
        cv2.line(frame, (70, 70), (570, 70), (45, 45, 45), 4)
        cv2.line(frame, (70, 130), (570, 130), (45, 45, 45), 4)

        # Капля движется вперед и назад; вертикальные границы имитируют мениски.
        center_x = int(round(320 + 150 * math.sin(2 * math.pi * index / count)))
        half_length = 24
        cv2.rectangle(
            frame, (center_x - half_length, 75), (center_x + half_length, 125),
            (65, 65, 65), -1,
        )
        cv2.line(frame, (center_x - half_length, 76), (center_x - half_length, 124),
                 (30, 30, 30), 2)
        cv2.line(frame, (center_x + half_length, 76), (center_x + half_length, 124),
                 (30, 30, 30), 2)

        noise = rng.normal(0, 1.2, frame.shape).astype(np.float32)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        yield index, index / fps, frame, fps


def run(
    cfg: AppConfig,
    frames: Iterator[tuple[int, float, np.ndarray, float]],
    csv_path: str,
    video_path: Optional[str],
    show: bool,
    background_bgr: Optional[np.ndarray],
) -> tuple[int, int]:
    pipeline = VisionPipeline(cfg, background_bgr)
    logger = MeasurementLogger(csv_path)
    writer: Optional[cv2.VideoWriter] = None
    total = 0
    valid = 0
    paused = False

    try:
        for index, timestamp, frame, fps in frames:
            result = pipeline.process(frame, index, timestamp)
            logger.write(result.measurement)
            total += 1
            valid += int(result.measurement.valid)

            annotated = draw_overlay(frame, result)
            if result.geometry is not None:
                draw_ruler(annotated, result.geometry, cfg.calibration)

            if video_path is not None:
                if writer is None:
                    height, width = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Не удалось создать видеофайл: {video_path}")
                writer.write(annotated)

            if show:
                cv2.imshow("Capillary drop tracker", annotated)
                key = cv2.waitKey(0 if paused else 1) & 0xFF
                if key in {27, ord("q")}:
                    break
                if key == ord(" "):
                    paused = not paused
    finally:
        logger.close()
        if writer is not None:
            writer.release()
        if show:
            # В headless-сборках OpenCV оконный модуль может отсутствовать;
            # на обработку и запись результатов это не влияет.
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    return total, valid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Измерение положения капли в капилляре средствами OpenCV"
    )
    parser.add_argument("--source", default="0", help="Фото, видео или номер USB-камеры")
    parser.add_argument("--config", help="JSON с параметрами; необязателен")
    parser.add_argument("--background", help="Кадр пустого капилляра; необязателен")
    parser.add_argument("--output", default="measurements.csv", help="CSV-журнал измерений")
    parser.add_argument("--video-output", help="Видео с наложенными маркерами; необязательно")
    parser.add_argument("--no-display", action="store_true", help="Не открывать окно предпросмотра")
    parser.add_argument("--demo", action="store_true", help="Запуск на синтетическом капилляре")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.demo:
        # Демонстрация использует известную геометрию, чтобы проверять именно
        # детектор, калибровку, фильтр и журнал независимо от поиска линий Хафа.
        cfg.manual_roi = ManualROIConfig(
            enabled=True, center_xy=(320.0, 100.0), angle_deg=0.0,
            length_px=500.0, width_px=60.0,
        )
        cfg.detector.max_drop_length_px = 100
        frames = synthetic_frames()
    else:
        frames = iter_input_source(args.source)

    background = None
    if args.background:
        background = cv2.imread(args.background, cv2.IMREAD_COLOR)
        if background is None:
            raise RuntimeError(f"Не удалось прочитать фон: {args.background}")

    total, valid = run(
        cfg=cfg,
        frames=frames,
        csv_path=args.output,
        video_path=args.video_output,
        show=not args.no_display,
        background_bgr=background,
    )
    print(f"Обработано кадров: {total}; достоверных измерений: {valid}")
    print(f"Журнал сохранен: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
