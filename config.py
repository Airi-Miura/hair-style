from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CaptureGuideConfig:
    """撮影ガイド、判定、正規化で共有する設定値。"""

    canvas_width: int = 1024
    canvas_height: int = 1024
    nose_x_ratio: float = 0.50
    nose_y_ratio: float = 0.45
    eye_line_y_ratio: float = 0.38
    chin_y_ratio: float = 0.68
    head_top_y_ratio: float = 0.16
    target_face_width_ratio: float = 0.30
    min_face_width_ratio: float = 0.25
    max_face_width_ratio: float = 0.35
    nose_tolerance_ratio: float = 0.05
    chin_tolerance_ratio: float = 0.08
    max_eye_tilt_deg: float = 5.0
    max_yaw_score: float = 0.08
    min_face_margin_ratio: float = 0.03


CAPTURE_GUIDE = CaptureGuideConfig()
