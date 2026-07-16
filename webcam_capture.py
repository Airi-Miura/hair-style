from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from config import CAPTURE_GUIDE
from face_normalization import CaptureAssessment, assess_capture

try:
    import av
    from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

    WEBRTC_AVAILABLE = True
except Exception:
    av = None
    RTCConfiguration = None
    VideoProcessorBase = object
    webrtc_streamer = None
    WEBRTC_AVAILABLE = False


@dataclass(frozen=True)
class CameraSnapshot:
    """撮影ボタンを押した時点のカメラフレームと判定結果。"""

    image: Image.Image
    guided_image: Image.Image
    assessment: CaptureAssessment | None
    ready: bool


def _draw_text_with_background(frame: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    """OpenCVフレームへ読みやすい背景付きテキストを描画する。"""
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.55, frame.shape[1] / 1280)
    thickness = max(1, int(frame.shape[1] / 640))
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(frame, (x - 8, y - th - 10), (x + tw + 8, y + baseline + 8), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _guidance_message(assessment: CaptureAssessment | None, error_message: str | None = None) -> tuple[str, bool]:
    """判定結果から画面上へ表示する代表メッセージを作る。"""
    if error_message:
        return error_message, False
    if assessment is None:
        return "顔をガイドに合わせてください", False
    if assessment.ready:
        return "撮影できます", True
    for _, (ok, message) in assessment.checks.items():
        if not ok:
            return message, False
    return "顔をガイドに合わせてください", False


def draw_guided_camera_frame(frame_rgb: np.ndarray, assessment: CaptureAssessment | None, error_message: str | None = None) -> np.ndarray:
    """カメラフレーム上へ顔ガイドと判定メッセージを直接描画する。"""
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w = frame.shape[:2]
    message, ready = _guidance_message(assessment, error_message)
    color = (60, 210, 80) if ready else (0, 210, 255)
    alert_color = (60, 210, 80) if ready else (0, 80, 255)

    nose = (int(w * CAPTURE_GUIDE.nose_x_ratio), int(h * CAPTURE_GUIDE.nose_y_ratio))
    ellipse_center = (int(w * 0.50), int(h * 0.43))
    ellipse_axes = (int(w * 0.16), int(h * 0.28))
    eye_y = int(h * 0.36)
    head_y = int(h * 0.14)
    chin_y = int(h * 0.71)

    overlay = frame.copy()
    cv2.ellipse(overlay, ellipse_center, ellipse_axes, 0, 0, 360, color, max(2, w // 220))
    cv2.line(overlay, (int(w * 0.22), eye_y), (int(w * 0.78), eye_y), color, max(2, w // 280))
    cv2.line(overlay, (int(w * 0.34), head_y), (int(w * 0.66), head_y), color, max(2, w // 280))
    cv2.line(overlay, (int(w * 0.30), chin_y), (int(w * 0.70), chin_y), color, max(2, w // 280))
    cv2.circle(overlay, nose, max(6, w // 90), alert_color, -1)
    cv2.line(overlay, (nose[0] - 22, nose[1]), (nose[0] + 22, nose[1]), (255, 255, 255), 2)
    cv2.line(overlay, (nose[0], nose[1] - 22), (nose[0], nose[1] + 22), (255, 255, 255), 2)
    frame = cv2.addWeighted(overlay, 0.82, frame, 0.18, 0)

    _draw_text_with_background(frame, message, (18, 38), color if ready else alert_color)
    _draw_text_with_background(frame, "鼻を中央の点に合わせてください", (18, h - 92), (255, 255, 255))
    _draw_text_with_background(frame, "正面を向き、頭頂部から肩まで写してください", (18, h - 50), (255, 255, 255))

    if assessment is not None:
        nose_tip = getattr(assessment.face_info, "nose_tip", None)
        if nose_tip is not None:
            cv2.circle(frame, (int(nose_tip[0]), int(nose_tip[1])), max(4, w // 160), (255, 0, 255), -1)
        left_eye = getattr(assessment.face_info, "left_eye_center", None)
        right_eye = getattr(assessment.face_info, "right_eye_center", None)
        if left_eye is not None and right_eye is not None:
            cv2.line(
                frame,
                (int(left_eye[0]), int(left_eye[1])),
                (int(right_eye[0]), int(right_eye[1])),
                (255, 0, 255),
                max(1, w // 360),
            )

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class GuidedCameraProcessor(VideoProcessorBase):
    """streamlit-webrtcの各フレームへガイドを描画し、最新フレームを保持する。"""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.latest_frame_rgb: np.ndarray | None = None
        self.latest_guided_rgb: np.ndarray | None = None
        self.latest_assessment: CaptureAssessment | None = None
        self.latest_error: str | None = None

    def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
        bgr = frame.to_ndarray(format="bgr24")
        mirrored_bgr = cv2.flip(bgr, 1)
        mirrored_rgb = cv2.cvtColor(mirrored_bgr, cv2.COLOR_BGR2RGB)

        assessment: CaptureAssessment | None = None
        error_message: str | None = None
        try:
            assessment = assess_capture(Image.fromarray(mirrored_rgb))
        except Exception:
            error_message = "顔を検出できません"

        guided_rgb = draw_guided_camera_frame(mirrored_rgb, assessment, error_message)
        with self.lock:
            self.latest_frame_rgb = mirrored_rgb.copy()
            self.latest_guided_rgb = guided_rgb.copy()
            self.latest_assessment = assessment
            self.latest_error = error_message

        return av.VideoFrame.from_ndarray(cv2.cvtColor(guided_rgb, cv2.COLOR_RGB2BGR), format="bgr24")

    def snapshot(self) -> CameraSnapshot | None:
        """最新フレームをスレッド安全にコピーして返す。"""
        with self.lock:
            if self.latest_frame_rgb is None:
                return None
            image = Image.fromarray(self.latest_frame_rgb.copy())
            guided = Image.fromarray(self.latest_guided_rgb.copy()) if self.latest_guided_rgb is not None else image.copy()
            assessment = self.latest_assessment
            ready = bool(assessment and assessment.ready)
            return CameraSnapshot(image=image, guided_image=guided, assessment=assessment, ready=ready)


def render_guided_camera() -> CameraSnapshot | None:
    """ガイド付きリアルタイムカメラを表示し、撮影成功時のスナップショットを返す。"""
    if not WEBRTC_AVAILABLE:
        st.error("streamlit-webrtc が利用できないため、リアルタイムガイド付きカメラを表示できません。画像アップロードを使用してください。")
        return None

    st.subheader("ガイド付きリアルタイムカメラ")
    st.caption("プレビューは鏡像表示です。撮影画像もこの表示と同じ向きで保存し、後続処理へ渡します。")
    ctx = webrtc_streamer(
        key="guided-camera",
        video_processor_factory=GuidedCameraProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    processor = ctx.video_processor if ctx else None
    snapshot = processor.snapshot() if processor else None
    if snapshot and snapshot.assessment:
        if snapshot.assessment.ready:
            st.success("撮影できます")
        else:
            st.warning("顔をガイドに合わせてください")
        cols = st.columns(4)
        for index, (label, (ok, message)) in enumerate(snapshot.assessment.checks.items()):
            with cols[index % 4]:
                if ok:
                    st.success(f"{label}: OK")
                else:
                    st.warning(f"{label}: {message}")
    elif snapshot:
        st.warning("顔を検出できません。顔をガイド内に入れてください。")
    else:
        st.info("カメラの開始後、顔をガイドへ合わせてください。")

    if st.button("撮影", type="primary"):
        snapshot = processor.snapshot() if processor else None
        if snapshot is None:
            st.warning("まだカメラフレームを取得できていません。少し待ってから撮影してください。")
            return None
        if not snapshot.ready:
            st.warning("顔をガイドに合わせてから撮影してください")
            return None
        st.session_state["captured_camera_snapshot"] = snapshot
        st.success("撮影しました")

    stored = st.session_state.get("captured_camera_snapshot")
    if isinstance(stored, CameraSnapshot):
        st.image(stored.image, caption="撮影済み画像（ガイドなし）", width="stretch")
        return stored
    return None
