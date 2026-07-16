from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


def _get_face_mesh_module() -> object:
    """MediaPipeのバージョン差を吸収してFace Meshモジュールを取得する。"""
    solutions = getattr(mp, "solutions", None)
    if solutions is not None and hasattr(solutions, "face_mesh"):
        return solutions.face_mesh

    try:
        from mediapipe.python.solutions import face_mesh

        return face_mesh
    except Exception as exc:
        raise RuntimeError(
            "MediaPipe Face Meshを読み込めませんでした。mediapipeを再インストールしてください。"
        ) from exc


@dataclass(frozen=True)
class FaceInfo:
    """MediaPipe Face Meshから計算した顔位置情報。"""

    center: tuple[float, float]
    width: float
    height: float
    top: tuple[float, float]
    chin: tuple[float, float]
    left: tuple[float, float]
    right: tuple[float, float]
    rotation_deg: float
    face_count: int
    left_temple: Optional[tuple[float, float]] = None
    right_temple: Optional[tuple[float, float]] = None
    brow_center: Optional[tuple[float, float]] = None
    left_brow: Optional[tuple[float, float]] = None
    right_brow: Optional[tuple[float, float]] = None
    left_cheek: Optional[tuple[float, float]] = None
    right_cheek: Optional[tuple[float, float]] = None
    left_jaw: Optional[tuple[float, float]] = None
    right_jaw: Optional[tuple[float, float]] = None
    nose_tip: Optional[tuple[float, float]] = None
    left_eye_center: Optional[tuple[float, float]] = None
    right_eye_center: Optional[tuple[float, float]] = None
    yaw_score: float = 0.0


def _landmark_to_point(landmark: object, width: int, height: int) -> tuple[float, float]:
    """正規化ランドマークを画像座標へ変換する。"""
    return float(landmark.x * width), float(landmark.y * height)


def _face_bbox(landmarks: list[object], width: int, height: int) -> tuple[float, float, float, float]:
    """顔ランドマーク全体を囲む矩形を返す。"""
    xs = [landmark.x * width for landmark in landmarks]
    ys = [landmark.y * height for landmark in landmarks]
    return min(xs), min(ys), max(xs), max(ys)


def detect_face(image_rgb: np.ndarray) -> FaceInfo:
    """RGB画像から顔を検出し、最も大きい顔の位置情報を返す。

    Args:
        image_rgb: OpenCVではなくPillow基準のRGB配列。

    Returns:
        顔中心、顔幅、顔高さ、頭頂部推定位置、顎位置などを含むFaceInfo。

    Raises:
        ValueError: 顔が検出できない場合。
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("顔検出にはRGB画像が必要です。")

    height, width = image_rgb.shape[:2]
    mp_face_mesh = _get_face_mesh_module()

    # 静止画向けに最大2人まで確認し、複数人の案内に使う。
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        result = face_mesh.process(image_rgb)

    if not result.multi_face_landmarks:
        raise ValueError("顔が検出できませんでした。正面に近い、明るい肩から上の写真を使ってください。")

    faces = result.multi_face_landmarks
    face_count = len(faces)

    # 複数検出時は最も大きく写っている顔を採用する。
    def area(face_landmarks: object) -> float:
        x1, y1, x2, y2 = _face_bbox(face_landmarks.landmark, width, height)
        return (x2 - x1) * (y2 - y1)

    main_face = max(faces, key=area)
    landmarks = main_face.landmark

    # Face Meshの代表点。左右は頬付近、上下は額中央と顎を利用する。
    left = _landmark_to_point(landmarks[234], width, height)
    right = _landmark_to_point(landmarks[454], width, height)
    forehead = _landmark_to_point(landmarks[10], width, height)
    chin = _landmark_to_point(landmarks[152], width, height)
    left_eye = _landmark_to_point(landmarks[33], width, height)
    right_eye = _landmark_to_point(landmarks[263], width, height)
    nose_tip = _landmark_to_point(landmarks[1], width, height)
    left_eye_inner = _landmark_to_point(landmarks[133], width, height)
    right_eye_inner = _landmark_to_point(landmarks[362], width, height)
    left_eye_center = ((left_eye[0] + left_eye_inner[0]) / 2.0, (left_eye[1] + left_eye_inner[1]) / 2.0)
    right_eye_center = ((right_eye[0] + right_eye_inner[0]) / 2.0, (right_eye[1] + right_eye_inner[1]) / 2.0)
    left_temple = _landmark_to_point(landmarks[127], width, height)
    right_temple = _landmark_to_point(landmarks[356], width, height)
    left_brow = _landmark_to_point(landmarks[105], width, height)
    right_brow = _landmark_to_point(landmarks[334], width, height)
    brow_center = _landmark_to_point(landmarks[168], width, height)
    left_cheek = _landmark_to_point(landmarks[50], width, height)
    right_cheek = _landmark_to_point(landmarks[280], width, height)
    left_jaw = _landmark_to_point(landmarks[172], width, height)
    right_jaw = _landmark_to_point(landmarks[397], width, height)

    face_width = float(np.linalg.norm(np.array(right) - np.array(left)))
    face_height = float(np.linalg.norm(np.array(chin) - np.array(forehead)))
    center = ((left[0] + right[0]) / 2.0, (forehead[1] + chin[1]) / 2.0)

    # Face Meshの額点は実際の頭頂より低いため、顔高さから簡易推定する。
    top = (center[0], max(0.0, forehead[1] - face_height * 0.28))

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    rotation_deg = float(np.degrees(np.arctan2(dy, dx)))
    eye_mid_x = (left_eye_center[0] + right_eye_center[0]) / 2.0
    yaw_score = float((nose_tip[0] - eye_mid_x) / max(face_width, 1.0))

    return FaceInfo(
        center=center,
        width=max(face_width, 1.0),
        height=max(face_height, 1.0),
        top=top,
        chin=chin,
        left=left,
        right=right,
        rotation_deg=rotation_deg,
        face_count=face_count,
        left_temple=left_temple,
        right_temple=right_temple,
        brow_center=brow_center,
        left_brow=left_brow,
        right_brow=right_brow,
        left_cheek=left_cheek,
        right_cheek=right_cheek,
        left_jaw=left_jaw,
        right_jaw=right_jaw,
        nose_tip=nose_tip,
        left_eye_center=left_eye_center,
        right_eye_center=right_eye_center,
        yaw_score=yaw_score,
    )


def draw_face_debug(image_rgb: np.ndarray, face_info: FaceInfo) -> np.ndarray:
    """確認用に顔の主要点を描画したRGB画像を返す。通常画面では使わない補助関数。"""
    debug = image_rgb.copy()
    points = [face_info.center, face_info.top, face_info.chin, face_info.left, face_info.right]
    for x, y in points:
        cv2.circle(debug, (int(x), int(y)), 5, (255, 80, 80), -1)
    return debug
