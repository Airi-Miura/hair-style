from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw

from config import CAPTURE_GUIDE, CaptureGuideConfig
from face_detection import FaceInfo, detect_face


@dataclass(frozen=True)
class CaptureAssessment:
    """撮影条件判定の結果。"""

    face_info: FaceInfo
    checks: dict[str, tuple[bool, str]]
    ready: bool
    guide_overlay: Image.Image
    debug_overlay: Image.Image
    nose_offset: tuple[float, float]
    face_width_ratio: float
    eye_tilt_deg: float
    yaw_score: float


@dataclass(frozen=True)
class NormalizationResult:
    """顔位置を基準キャンバスへ正規化した結果。"""

    original_image: Image.Image
    normalized_image: Image.Image
    assessment: CaptureAssessment
    normalized_face_info: FaceInfo
    transform_matrix: np.ndarray


def _point(face_info: FaceInfo, attr: str, fallback: tuple[float, float]) -> tuple[float, float]:
    """FaceInfoの任意点を取得し、なければfallbackを返す。"""
    value = getattr(face_info, attr, None)
    return value if value is not None else fallback


def _draw_guide_base(size: tuple[int, int], config: CaptureGuideConfig = CAPTURE_GUIDE) -> Image.Image:
    """透明背景の撮影ガイド画像を作成する。"""
    width, height = size
    guide = Image.new("RGBA", size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(guide)

    nose = (width * config.nose_x_ratio, height * config.nose_y_ratio)
    face_w = width * config.target_face_width_ratio
    ellipse_h = height * (config.chin_y_ratio - config.head_top_y_ratio)
    ellipse_box = (
        nose[0] - face_w * 0.68,
        height * config.head_top_y_ratio,
        nose[0] + face_w * 0.68,
        height * config.head_top_y_ratio + ellipse_h,
    )

    draw.ellipse(ellipse_box, outline=(0, 180, 255, 210), width=max(2, width // 180))
    draw.line((width * 0.18, height * config.eye_line_y_ratio, width * 0.82, height * config.eye_line_y_ratio), fill=(0, 180, 255, 160), width=max(2, width // 220))
    draw.line((width * 0.28, height * config.chin_y_ratio, width * 0.72, height * config.chin_y_ratio), fill=(255, 170, 0, 170), width=max(2, width // 220))
    draw.line((width * 0.34, height * config.head_top_y_ratio, width * 0.66, height * config.head_top_y_ratio), fill=(120, 220, 80, 170), width=max(2, width // 220))
    draw.ellipse((nose[0] - 7, nose[1] - 7, nose[0] + 7, nose[1] + 7), fill=(255, 70, 70, 230))
    draw.line((nose[0], nose[1] - 18, nose[0], nose[1] + 18), fill=(255, 70, 70, 210), width=2)
    draw.line((nose[0] - 18, nose[1], nose[0] + 18, nose[1]), fill=(255, 70, 70, 210), width=2)
    return guide


def create_capture_guide_image(size: tuple[int, int] = (1024, 768), config: CaptureGuideConfig = CAPTURE_GUIDE) -> Image.Image:
    """画面に表示する撮影ガイド画像を作成する。"""
    base = Image.new("RGB", size, (245, 248, 250))
    overlay = _draw_guide_base(size, config)
    image = Image.alpha_composite(base.convert("RGBA"), overlay)
    draw = ImageDraw.Draw(image)
    lines = [
        "鼻を中央の点に合わせてください",
        "顔の輪郭をガイド内に合わせてください",
        "正面を向いてください",
        "頭頂部から肩まで写してください",
    ]
    y = 24
    for line in lines:
        draw.text((24, y), line, fill=(20, 30, 40, 255))
        y += 34
    return image.convert("RGB")


def overlay_capture_guide(image: Image.Image, config: CaptureGuideConfig = CAPTURE_GUIDE) -> Image.Image:
    """入力画像へ半透明の撮影ガイドを重ねる。"""
    rgba = image.convert("RGBA")
    guide = _draw_guide_base(rgba.size, config)
    return Image.alpha_composite(rgba, guide).convert("RGB")


def _face_edges_inside(face_info: FaceInfo, size: tuple[int, int], config: CaptureGuideConfig) -> bool:
    """顔が画像端で大きく切れていないかを確認する。"""
    width, height = size
    margin_x = width * config.min_face_margin_ratio
    margin_y = height * config.min_face_margin_ratio
    points = [face_info.left, face_info.right, face_info.top, face_info.chin]
    return all(margin_x <= x <= width - margin_x and margin_y <= y <= height - margin_y for x, y in points)


def assess_capture(image: Image.Image, config: CaptureGuideConfig = CAPTURE_GUIDE) -> CaptureAssessment:
    """入力画像の顔位置、傾き、サイズ、正面度を判定する。"""
    rgb = np.array(image.convert("RGB"))
    face_info = detect_face(rgb)
    width, height = image.size

    nose = _point(face_info, "nose_tip", face_info.center)
    left_eye = _point(face_info, "left_eye_center", (face_info.center[0] - face_info.width * 0.2, face_info.center[1]))
    right_eye = _point(face_info, "right_eye_center", (face_info.center[0] + face_info.width * 0.2, face_info.center[1]))
    target_nose = (width * config.nose_x_ratio, height * config.nose_y_ratio)
    nose_offset = ((nose[0] - target_nose[0]) / width, (nose[1] - target_nose[1]) / height)
    face_width_ratio = face_info.width / max(1, width)
    eye_tilt_deg = face_info.rotation_deg
    yaw_score = float(getattr(face_info, "yaw_score", 0.0))

    nose_ok = abs(nose_offset[0]) <= config.nose_tolerance_ratio and abs(nose_offset[1]) <= config.nose_tolerance_ratio
    if not nose_ok:
        if abs(nose_offset[0]) > abs(nose_offset[1]):
            nose_message = "もう少し左" if nose_offset[0] > 0 else "もう少し右"
        else:
            nose_message = "もう少し上" if nose_offset[1] > 0 else "もう少し下"
    else:
        nose_message = "OK"

    if face_width_ratio < config.min_face_width_ratio:
        size_message = "もう少し近づいてください"
    elif face_width_ratio > config.max_face_width_ratio:
        size_message = "もう少し離れてください"
    else:
        size_message = "OK"
    face_size_ok = config.min_face_width_ratio <= face_width_ratio <= config.max_face_width_ratio

    chin_offset = (face_info.chin[1] - height * config.chin_y_ratio) / height
    chin_ok = abs(chin_offset) <= config.chin_tolerance_ratio
    chin_message = "OK" if chin_ok else ("顔を少し上へ" if chin_offset > 0 else "顔を少し下へ")

    tilt_ok = abs(eye_tilt_deg) <= config.max_eye_tilt_deg
    yaw_ok = abs(yaw_score) <= config.max_yaw_score
    edge_ok = _face_edges_inside(face_info, image.size, config)

    checks = {
        "顔検出": (face_info.face_count == 1, "OK" if face_info.face_count == 1 else "1人だけ写してください"),
        "鼻位置": (nose_ok, nose_message),
        "顔サイズ": (face_size_ok, size_message),
        "顎位置": (chin_ok, chin_message),
        "顔の傾き": (tilt_ok, "OK" if tilt_ok else "顔をまっすぐにしてください"),
        "正面判定": (yaw_ok, "OK" if yaw_ok else "正面を向いてください"),
        "顔の見切れ": (edge_ok, "OK" if edge_ok else "顔全体が入るようにしてください"),
    }
    ready = all(ok for ok, _ in checks.values())

    guide_overlay = overlay_capture_guide(image, config)
    debug_overlay = _draw_assessment_debug(image, face_info, nose, left_eye, right_eye, config)
    return CaptureAssessment(
        face_info=face_info,
        checks=checks,
        ready=ready,
        guide_overlay=guide_overlay,
        debug_overlay=debug_overlay,
        nose_offset=nose_offset,
        face_width_ratio=face_width_ratio,
        eye_tilt_deg=eye_tilt_deg,
        yaw_score=yaw_score,
    )


def _draw_assessment_debug(
    image: Image.Image,
    face_info: FaceInfo,
    nose: tuple[float, float],
    left_eye: tuple[float, float],
    right_eye: tuple[float, float],
    config: CaptureGuideConfig,
) -> Image.Image:
    """判定内容を確認するためのデバッグ画像を作成する。"""
    debug = overlay_capture_guide(image, config).convert("RGBA")
    draw = ImageDraw.Draw(debug)
    width, height = image.size
    x1 = face_info.center[0] - face_info.width * 0.62
    y1 = face_info.top[1]
    x2 = face_info.center[0] + face_info.width * 0.62
    y2 = face_info.chin[1]
    draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 120, 255), width=max(2, width // 240))
    draw.line((left_eye[0], left_eye[1], right_eye[0], right_eye[1]), fill=(255, 80, 220, 255), width=max(2, width // 260))
    for label, point, color in [
        ("nose", nose, (255, 40, 40, 255)),
        ("left eye", left_eye, (0, 170, 255, 255)),
        ("right eye", right_eye, (0, 170, 255, 255)),
        ("chin", face_info.chin, (255, 180, 0, 255)),
    ]:
        x, y = point
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color)
        draw.text((x + 8, y - 8), label, fill=color)
    draw.text((18, height - 92), f"face width: {face_info.width / max(1, width):.1%}", fill=(20, 30, 40, 255))
    draw.text((18, height - 62), f"tilt: {face_info.rotation_deg:.1f} deg", fill=(20, 30, 40, 255))
    draw.text((18, height - 32), f"yaw score: {getattr(face_info, 'yaw_score', 0.0):.3f}", fill=(20, 30, 40, 255))
    return debug.convert("RGB")


def normalize_face_image(image: Image.Image, config: CaptureGuideConfig = CAPTURE_GUIDE) -> NormalizationResult:
    """顔の鼻位置、目の傾き、顔幅を基準キャンバスへ揃える。"""
    assessment = assess_capture(image, config)
    face_info = assessment.face_info
    source = np.array(image.convert("RGB"))
    src_h, src_w = source.shape[:2]

    nose = _point(face_info, "nose_tip", face_info.center)
    target_nose = (config.canvas_width * config.nose_x_ratio, config.canvas_height * config.nose_y_ratio)
    target_width = config.canvas_width * config.target_face_width_ratio
    scale = target_width / max(face_info.width, 1.0)
    angle = -face_info.rotation_deg

    rotate = cv2.getRotationMatrix2D(nose, angle, scale)
    nose_after = np.array(
        [
            rotate[0, 0] * nose[0] + rotate[0, 1] * nose[1] + rotate[0, 2],
            rotate[1, 0] * nose[0] + rotate[1, 1] * nose[1] + rotate[1, 2],
        ]
    )
    rotate[0, 2] += target_nose[0] - nose_after[0]
    rotate[1, 2] += target_nose[1] - nose_after[1]

    normalized = cv2.warpAffine(
        source,
        rotate,
        (config.canvas_width, config.canvas_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    normalized_image = Image.fromarray(normalized, mode="RGB")
    normalized_face_info = detect_face(np.array(normalized_image))
    return NormalizationResult(
        original_image=image.convert("RGB"),
        normalized_image=normalized_image,
        assessment=assessment,
        normalized_face_info=normalized_face_info,
        transform_matrix=rotate.astype(np.float32),
    )
