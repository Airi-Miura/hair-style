from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from face_detection import FaceInfo
from face_normalization import NormalizationResult, normalize_face_image
from face_parsing import (
    FaceParsingModelNotFoundError,
    FaceParsingResult,
    FaceParsingRuntimeError,
    run_face_parsing,
)
from hair_overlay import (
    HairCompositionResult,
    HairstyleAsset,
    HairstyleAssetError,
    compose_hair_layers,
    discover_hairstyles,
    estimate_uncovered_hair_mask_ratio,
)
from image_processing import compose_on_white, load_image_from_upload, pil_to_png_bytes, remove_background, resize_if_large
from webcam_capture import CameraSnapshot, render_guided_camera


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "face_parsing_bisenet.pth"


ProcessedData = tuple[
    Image.Image,
    Image.Image,
    Image.Image,
    Image.Image,
    FaceParsingResult,
    Image.Image,
    FaceInfo,
    NormalizationResult,
]


def _show_error(message: str) -> None:
    """日本語のエラーメッセージを画面へ表示する。"""
    st.error(message)


def _select_input_image() -> tuple[Image.Image | None, CameraSnapshot | None]:
    """ガイド付きカメラまたは画像アップロードから入力画像を取得する。"""
    input_method = st.radio("画像入力", ["カメラで撮影", "画像をアップロード"], horizontal=True)

    if input_method == "カメラで撮影":
        snapshot = render_guided_camera()
        if snapshot is None:
            return None, None
        return snapshot.image, snapshot

    uploaded_file = st.file_uploader("PNG、JPG、JPEG形式の画像をアップロードしてください", type=["png", "jpg", "jpeg"])
    if uploaded_file is None:
        return None, None
    return load_image_from_upload(uploaded_file), None


def _render_capture_assessment(normalization: NormalizationResult) -> None:
    """撮影条件の判定結果を条件ごとに表示する。"""
    assessment = normalization.assessment
    if assessment.ready:
        st.success("撮影できます")
    else:
        st.warning("顔をガイドに合わせてください")
        st.info("顔が正面を向き、頭頂部から肩まで写っている画像を使用してください。")

    cols = st.columns(4)
    for index, (label, (ok, message)) in enumerate(assessment.checks.items()):
        with cols[index % 4]:
            if ok:
                st.success(f"{label}: OK")
            else:
                st.warning(f"{label}: {message}")

    st.caption(
        f"鼻のずれ: x={assessment.nose_offset[0]:+.1%}, y={assessment.nose_offset[1]:+.1%} / "
        f"顔幅: {assessment.face_width_ratio:.1%} / "
        f"顔の傾き: {assessment.eye_tilt_deg:.1f}度 / "
        f"左右向きスコア: {assessment.yaw_score:.3f}"
    )


def _render_adjust_sliders(face_info: FaceInfo, hairstyle: HairstyleAsset) -> tuple[float, float, float, float]:
    """髪型の自動配置に対する微調整スライダーを表示する。"""
    st.subheader("髪型の微調整")
    st.caption("元髪透明化済み人物画像は固定し、ここでは選択髪型の配置だけを追加調整します。")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        size_scale = st.slider("髪型サイズ", 0.70, 1.45, 1.00, 0.01, key=f"{hairstyle.name}_size")
    with col2:
        offset_x = st.slider("左右位置", -180, 180, 0, 1, key=f"{hairstyle.name}_offset_x")
    with col3:
        offset_y = st.slider("上下位置", -210, 210, 0, 1, key=f"{hairstyle.name}_offset_y")
    with col4:
        rotation_adjust = st.slider("回転角度", -20, 20, 0, 1, key=f"{hairstyle.name}_rotation")

    return size_scale, float(offset_x), float(offset_y), float(rotation_adjust)


def _source_hair_mask_area(mask: Image.Image) -> float:
    """元画像から作成した髪マスクの面積割合を返す。"""
    mask_array = np.array(mask.convert("L"))
    return float(np.count_nonzero(mask_array > 80) / mask_array.size)


def _mask_bbox(mask: Image.Image, threshold: int = 80) -> tuple[int, int, int, int] | None:
    """デバッグ用にマスクの非透明領域bboxを返す。"""
    mask_array = np.array(mask.convert("L"))
    ys, xs = np.where(mask_array > threshold)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _process_image(image: Image.Image, input_hash: str) -> ProcessedData:
    """背景削除、Face Parsing、元髪透明化を入力画像単位で一度だけ実行する。"""
    resized = resize_if_large(image)
    normalization = normalize_face_image(resized)
    normalized = normalization.normalized_image
    face_info = normalization.normalized_face_info

    cutout_rgba = remove_background(normalized)
    cutout_on_white = compose_on_white(cutout_rgba)
    parsing_result = run_face_parsing(normalized, cutout_rgba, MODEL_PATH)
    person_without_hair_rgba = parsing_result.hair_removed_rgba.copy()

    st.session_state["source_preprocess_meta"] = {
        "input_image_hash": input_hash,
        "source_hair_mask_recomputed": True,
        "source_hair_mask_area": _source_hair_mask_area(parsing_result.hair_mask),
        "source_hair_mask_bbox": parsing_result.hair_mask_bbox,
        "source_hair_mask_area_pixels": parsing_result.hair_mask_area_pixels,
        "person_without_hair_process_id": datetime.now().strftime("%Y%m%d-%H%M%S-%f"),
    }
    return resized, normalized, cutout_rgba, cutout_on_white, parsing_result, person_without_hair_rgba, face_info, normalization


def _get_processed_image(image: Image.Image) -> ProcessedData:
    """同じ入力画像では元髪検出と元髪透明化済み人物画像を再利用する。"""
    image_bytes = pil_to_png_bytes(image)
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    cached = st.session_state.get("processed_image")
    if cached and cached.get("hash") == image_hash:
        meta = dict(st.session_state.get("source_preprocess_meta", {}))
        meta["source_hair_mask_recomputed"] = False
        st.session_state["source_preprocess_meta"] = meta
        return cached["data"]

    data = _process_image(image, image_hash)
    st.session_state["processed_image"] = {"hash": image_hash, "data": data}
    return data


def _combined_layer_alpha_image(layers: dict[str, Image.Image]) -> Image.Image:
    """選択髪型レイヤーの配置後アルファマスクを1枚にまとめる。"""
    alpha: np.ndarray | None = None
    for layer in layers.values():
        layer_alpha = np.array(layer.convert("RGBA"))[:, :, 3]
        alpha = layer_alpha if alpha is None else np.maximum(alpha, layer_alpha)
    if alpha is None:
        alpha = np.zeros((1, 1), dtype=np.uint8)
    return Image.fromarray(alpha.astype(np.uint8), mode="L")


def _composed_hair_layers_preview(layers: dict[str, Image.Image]) -> Image.Image:
    """選択髪型だけを配置したプレビューを作る。"""
    first = next(iter(layers.values()))
    canvas = Image.new("RGBA", first.size, (255, 255, 255, 255))
    for name in ("hair_full", "side_left", "side_right", "front", "strands"):
        canvas.alpha_composite(layers[name].convert("RGBA"))
    return canvas.convert("RGB")


def _hair_mask_debug_overlay(
    base_image: Image.Image,
    final_mask: Image.Image,
    dilation_added: Image.Image,
    color_assist: Image.Image,
) -> Image.Image:
    """膨張追加と色補助追加を色分けして確認するプレビューを作る。"""
    base = base_image.convert("RGBA")
    transparent = Image.new("RGBA", base.size, (0, 0, 0, 0))
    final_layer = Image.composite(Image.new("RGBA", base.size, (255, 0, 0, 75)), transparent, final_mask.convert("L"))
    dilation_layer = Image.composite(Image.new("RGBA", base.size, (255, 220, 0, 150)), transparent, dilation_added.convert("L"))
    color_layer = Image.composite(Image.new("RGBA", base.size, (0, 210, 255, 150)), transparent, color_assist.convert("L"))
    preview = Image.alpha_composite(base, final_layer)
    preview = Image.alpha_composite(preview, dilation_layer)
    preview = Image.alpha_composite(preview, color_layer)
    return preview.convert("RGB")


def _render_camera_debug(snapshot: CameraSnapshot | None) -> None:
    """Webカメラの生フレーム、ガイド付きフレーム、判定値をデバッグ表示する。"""
    if snapshot is None:
        return
    st.subheader("カメラデバッグ")
    col1, col2 = st.columns(2)
    with col1:
        st.image(snapshot.image, caption="生のカメラフレーム（保存用・ガイドなし）", width="stretch")
    with col2:
        st.image(snapshot.guided_image, caption="ガイド付きカメラフレーム", width="stretch")

    assessment = snapshot.assessment
    if assessment is None:
        st.warning("MediaPipeランドマークを取得できませんでした。")
        return
    nose = getattr(assessment.face_info, "nose_tip", None)
    target = (snapshot.image.size[0] * 0.50, snapshot.image.size[1] * 0.45)
    st.write(
        {
            "鼻の現在座標": nose,
            "鼻の目標座標": target,
            "顔幅の現在値": assessment.face_width_ratio,
            "顔幅の許容範囲": "25%〜35%",
            "目の傾き角度": assessment.eye_tilt_deg,
            "撮影条件OK": assessment.ready,
        }
    )


def _render_preprocess_meta(selected_name: str) -> None:
    """元髪処理が髪型選択から独立していることを確認するメタ情報を表示する。"""
    meta = st.session_state.get("source_preprocess_meta", {})
    st.subheader("元髪処理の固定情報")
    st.write(
        {
            "入力画像ハッシュ": meta.get("input_image_hash"),
            "元髪マスクを再計算したか": meta.get("source_hair_mask_recomputed"),
            "使用中の元髪マスク面積": meta.get("source_hair_mask_area"),
            "選択髪型名": selected_name,
            "元髪透明化画像の処理ID": meta.get("person_without_hair_process_id"),
        }
    )


def _render_debug_view(
    original: Image.Image,
    normalized: Image.Image,
    cutout_rgba: Image.Image,
    cutout_on_white: Image.Image,
    parsing_result: FaceParsingResult,
    person_without_hair_rgba: Image.Image,
    normalization: NormalizationResult,
    composition: HairCompositionResult,
    camera_snapshot: CameraSnapshot | None,
    selected_name: str,
) -> None:
    """正規化、元髪マスク、髪型レイヤー、最終合成のデバッグ表示を行う。"""
    _render_camera_debug(camera_snapshot)
    _render_preprocess_meta(selected_name)
    st.subheader("元髪マスク後処理デバッグ")
    st.write(
        {
            "髪マスク面積px": parsing_result.hair_mask_area_pixels,
            "髪マスク面積率": parsing_result.hair_area_ratio,
            "髪マスクbbox": parsing_result.hair_mask_bbox or _mask_bbox(parsing_result.hair_mask),
            "選択髪型名": selected_name,
        }
    )
    st.subheader("髪素材座標系デバッグ")
    st.write(
        {
            "各レイヤーの画像サイズ": composition.layer_sizes,
            "各レイヤーの非透明領域bbox": composition.layer_alpha_bboxes,
            "original.png側ランドマーク": composition.source_landmarks,
            "撮影画像側ランドマーク": composition.target_landmarks,
            "計算した変換行列": composition.transform_matrix,
        }
    )

    st.subheader("処理デバッグ")
    selected_hair_alpha = _combined_layer_alpha_image(composition.transformed_layers)
    selected_hair_preview = _composed_hair_layers_preview(composition.transformed_layers)
    person_without_hair_on_white = compose_on_white(person_without_hair_rgba)
    hair_mask_addition_preview = _hair_mask_debug_overlay(
        normalized,
        parsing_result.hair_mask,
        parsing_result.dilation_added_mask,
        parsing_result.color_assist_mask,
    )
    debug_items = [
        ("正規化前画像", original),
        ("撮影ガイドとのずれ", normalization.assessment.guide_overlay),
        ("鼻位置・目のライン・顔バウンディングボックス", normalization.assessment.debug_overlay),
        ("正規化後画像", normalized),
        ("背景削除後人物画像", cutout_on_white),
        ("背景削除後人物画像（RGBA確認）", compose_on_white(cutout_rgba)),
        ("Face Parsingのクラス分類画像", parsing_result.class_map),
        ("Face Parsingの生髪マスク", parsing_result.raw_hair_mask),
        ("後処理後の髪マスク", parsing_result.postprocessed_hair_mask),
        ("膨張で追加された領域", parsing_result.dilation_added_mask),
        ("色補助で追加された領域", parsing_result.color_assist_mask),
        ("追加領域の色分けプレビュー", hair_mask_addition_preview),
        ("元画像から作成した髪領域マスク", parsing_result.hair_mask),
        ("髪領域の赤色プレビュー", parsing_result.hair_overlay_preview),
        ("元髪透明化済み人物画像", person_without_hair_on_white),
        ("選択髪型のアルファマスク", selected_hair_alpha),
        ("選択髪型を配置した画像", selected_hair_preview),
        ("顔保護マスク", composition.face_protection_mask),
        ("hair_fullだけを配置した画像", composition.hair_full_only),
        ("人物を重ねた画像", composition.person_on_full),
        ("frontとsideを重ねた画像", composition.front_side_only),
        ("位置合わせデバッグ", composition.debug_overlay),
        ("最終合成結果", composition.final_image),
    ]

    for row_start in range(0, len(debug_items), 3):
        cols = st.columns(3)
        for col, (caption, image) in zip(cols, debug_items[row_start : row_start + 3]):
            with col:
                st.image(image, caption=caption, width="stretch")

    if composition.layer_step_images:
        st.subheader("レイヤー追加順デバッグ")
        step_items = [
            ("hair_fullだけ", composition.layer_step_images.get("hair_full")),
            ("人物を重ねた状態", composition.layer_step_images.get("person")),
            ("side_left追加後", composition.layer_step_images.get("side_left")),
            ("side_right追加後", composition.layer_step_images.get("side_right")),
            ("front追加後", composition.layer_step_images.get("front")),
            ("strands追加後", composition.layer_step_images.get("strands")),
        ]
        step_items = [(caption, image) for caption, image in step_items if image is not None]
        for row_start in range(0, len(step_items), 3):
            cols = st.columns(3)
            for col, (caption, image) in zip(cols, step_items[row_start : row_start + 3]):
                with col:
                    st.image(image, caption=caption, width="stretch")


def main() -> None:
    """Streamlitアプリのエントリーポイント。"""
    st.set_page_config(page_title="AI髪型シミュレーター", layout="wide")

    st.title("AI髪型シミュレーター")
    st.write("写真を撮影して、ボブヘアを試してみよう！")

    debug_enabled = st.checkbox("デバッグ表示", value=False)
    alignment_debug_enabled = st.checkbox("位置合わせデバッグ", value=False)

    try:
        hairstyles = discover_hairstyles()
    except HairstyleAssetError as exc:
        _show_error(str(exc))
        return
    if not hairstyles:
        _show_error(r"C:\Users\airic\Desktop\output に metadata.json を持つ髪型素材フォルダが見つかりません。")
        return

    hairstyle_names = [hairstyle.name for hairstyle in hairstyles]
    selected_name = st.sidebar.selectbox("髪型を選択", hairstyle_names)
    selected_hairstyle = next(hairstyle for hairstyle in hairstyles if hairstyle.name == selected_name)
    layer_test_mode = st.sidebar.selectbox(
        "合成テストモード",
        ["全レイヤー", "hair_fullのみ", "hair_full + side", "hair_full + side + front"],
    )

    try:
        input_image, camera_snapshot = _select_input_image()
    except Exception as exc:
        _show_error(str(exc))
        return

    if input_image is None:
        st.info("ガイド付きカメラで撮影するか、画像をアップロードしてください。")
        return

    try:
        (
            original,
            normalized,
            cutout_rgba,
            cutout_on_white,
            parsing_result,
            person_without_hair_rgba,
            face_info,
            normalization,
        ) = _get_processed_image(input_image)
    except FaceParsingModelNotFoundError as exc:
        _show_error(str(exc))
        st.info("モデルファイルを配置したあと、画面を再読み込みしてもう一度処理してください。")
        return
    except FaceParsingRuntimeError as exc:
        _show_error(str(exc))
        return
    except ValueError as exc:
        _show_error(str(exc))
        return
    except RuntimeError as exc:
        _show_error(str(exc))
        return
    except Exception as exc:
        _show_error(f"処理中に予期しないエラーが発生しました: {exc}")
        return

    _render_capture_assessment(normalization)
    _render_preprocess_meta(selected_name)

    if face_info.face_count > 1:
        st.warning("複数人の顔が検出されました。今回は最も大きく写っている顔を使用します。できれば1人だけ写った画像を使用してください。")

    st.caption(f"髪領域の検出面積: {parsing_result.hair_area_ratio:.3%}")
    size_scale, offset_x, offset_y, rotation_adjust = _render_adjust_sliders(face_info, selected_hairstyle)

    try:
        base_person = person_without_hair_rgba.copy()
        composition = compose_hair_layers(
            base_person,
            face_info,
            selected_hairstyle,
            size_scale=size_scale,
            offset_x=offset_x,
            offset_y=offset_y,
            rotation_adjust=rotation_adjust,
            enabled_layers={
                "全レイヤー": ("hair_full", "side_left", "side_right", "front", "strands"),
                "hair_fullのみ": ("hair_full",),
                "hair_full + side": ("hair_full", "side_left", "side_right"),
                "hair_full + side + front": ("hair_full", "side_left", "side_right", "front"),
            }[layer_test_mode],
        )
    except RuntimeError as exc:
        _show_error(str(exc))
        return
    except Exception as exc:
        _show_error(f"髪型画像の合成に失敗しました: {exc}")
        return

    result = composition.final_image
    if composition.fallback_used:
        st.warning(f"髪型の基準点変換が不安定だったため、顔幅ベース配置へフォールバックしました: {composition.fallback_reason}")

    layer_canvases = [composition.transformed_layers[name] for name in ("hair_full", "side_left", "side_right", "front", "strands")]
    uncovered_ratio = estimate_uncovered_hair_mask_ratio(parsing_result.hair_mask, layer_canvases)
    if uncovered_ratio > 0.08:
        st.warning("元髪透明化済み領域の一部を推薦髪型で覆いきれていない可能性があります。髪型サイズや上下左右位置を調整してください。")

    st.subheader("完成画像")
    st.image(result, caption="髪型合成後の画像", width="stretch")

    st.download_button(
        "完成画像をPNGでダウンロード",
        data=pil_to_png_bytes(result),
        file_name="bob_hairstyle_result.png",
        mime="image/png",
    )

    st.subheader("撮影後・処理結果")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(original, caption="撮影画像（ガイドなし）", width="stretch")
    with col2:
        st.image(normalized, caption="正規化後画像", width="stretch")
    with col3:
        st.image(cutout_on_white, caption="背景削除後の画像", width="stretch")
    with col4:
        st.image(compose_on_white(person_without_hair_rgba), caption="元髪透明化後の画像", width="stretch")

    if debug_enabled or alignment_debug_enabled:
        _render_debug_view(
            original,
            normalized,
            cutout_rgba,
            cutout_on_white,
            parsing_result,
            person_without_hair_rgba,
            normalization,
            composition,
            camera_snapshot,
            selected_name,
        )


if __name__ == "__main__":
    main()
