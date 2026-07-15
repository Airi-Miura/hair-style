from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st
from PIL import Image

from face_detection import FaceInfo, detect_face
from face_parsing import (
    FaceParsingModelNotFoundError,
    FaceParsingResult,
    FaceParsingRuntimeError,
    run_face_parsing,
)
from hair_overlay import (
    ensure_bob_hair_assets,
    estimate_uncovered_hair_mask_ratio,
    overlay_bob_hair_layers,
    render_hair_layer_canvas,
)
from image_processing import (
    compose_on_white,
    load_image_from_upload,
    pil_to_png_bytes,
    pil_to_rgb_array,
    remove_background,
    resize_if_large,
)


APP_DIR = Path(__file__).resolve().parent
BACK_HAIR_PATH = APP_DIR / "assets" / "bob_hair_back.png"
FRONT_HAIR_PATH = APP_DIR / "assets" / "bob_hair_front.png"
MODEL_PATH = APP_DIR / "models" / "face_parsing_bisenet.pth"


def _show_error(message: str) -> None:
    """日本語のエラーメッセージを画面へ表示する。"""
    st.error(message)


def _select_input_image() -> Image.Image | None:
    """カメラ撮影またはアップロードから入力画像を1枚取得する。"""
    input_method = st.radio(
        "画像入力",
        ["カメラで撮影", "画像をアップロード"],
        horizontal=True,
    )

    if input_method == "カメラで撮影":
        camera_file = st.camera_input("正面を向いて、肩から上が入るように撮影してください")
        if camera_file is None:
            return None
        return load_image_from_upload(camera_file)

    uploaded_file = st.file_uploader(
        "PNG、JPG、JPEG形式の画像をアップロードしてください",
        type=["png", "jpg", "jpeg"],
    )
    if uploaded_file is None:
        return None
    return load_image_from_upload(uploaded_file)


def _render_adjust_sliders(face_info: FaceInfo) -> tuple[float, float, float, float]:
    """髪型の自動配置に対する微調整スライダーを表示する。"""
    st.subheader("髪型の微調整")
    st.caption("まずは自動調整のままで試し、必要な時だけ少し動かしてください。")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        size_scale = st.slider("髪型サイズ", 0.70, 1.45, 1.00, 0.01)
    with col2:
        offset_x = st.slider("左右位置", -180, 180, 0, 1)
    with col3:
        offset_y = st.slider("上下位置", -210, 210, 0, 1)
    with col4:
        rotation_adjust = st.slider("回転角度", -20, 20, 0, 1)

    return size_scale, float(offset_x), float(offset_y), float(rotation_adjust)


def _process_image(image: Image.Image) -> tuple[Image.Image, Image.Image, Image.Image, FaceParsingResult, FaceInfo]:
    """入力画像から背景削除、顔検出、髪検出、髪透明化までの基本処理を行う。"""
    resized = resize_if_large(image)
    face_info = detect_face(pil_to_rgb_array(resized))
    cutout = remove_background(resized)
    cutout_on_white = compose_on_white(cutout)
    parsing_result = run_face_parsing(resized, cutout, MODEL_PATH)
    return resized, cutout, cutout_on_white, parsing_result, face_info


def _get_processed_image(image: Image.Image) -> tuple[Image.Image, Image.Image, Image.Image, FaceParsingResult, FaceInfo]:
    """同じ画像では背景削除とFace Parsing結果を再利用し、スライダー操作を軽くする。"""
    image_bytes = pil_to_png_bytes(image)
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    cached = st.session_state.get("processed_image")
    if cached and cached.get("hash") == image_hash:
        return cached["data"]

    data = _process_image(image)
    st.session_state["processed_image"] = {"hash": image_hash, "data": data}
    return data


def _render_debug_view(
    original: Image.Image,
    cutout_on_white: Image.Image,
    parsing_result: FaceParsingResult,
    face_info: FaceInfo,
    final_result: Image.Image,
    size_scale: float,
    offset_x: float,
    offset_y: float,
    rotation_adjust: float,
) -> None:
    """髪マスクや前後レイヤーを確認するためのデバッグ表示を行う。"""
    st.subheader("デバッグ表示")
    back_canvas = render_hair_layer_canvas(
        original.size,
        face_info,
        BACK_HAIR_PATH,
        size_scale=size_scale,
        offset_x=offset_x,
        offset_y=offset_y,
        rotation_adjust=rotation_adjust,
    )
    front_canvas = render_hair_layer_canvas(
        original.size,
        face_info,
        FRONT_HAIR_PATH,
        size_scale=size_scale,
        offset_x=offset_x,
        offset_y=offset_y,
        rotation_adjust=rotation_adjust,
    )
    hair_removed_on_white = compose_on_white(parsing_result.hair_removed_rgba)

    debug_items = [
        ("入力画像", original),
        ("背景削除後の人物画像", cutout_on_white),
        ("Face Parsingのクラス分類画像", parsing_result.class_map),
        ("髪領域マスク", parsing_result.hair_mask),
        ("髪領域の赤色プレビュー", parsing_result.hair_overlay_preview),
        ("元の髪を透明化した人物画像", hair_removed_on_white),
        ("推薦髪型の後ろ側レイヤー", compose_on_white(back_canvas)),
        ("推薦髪型の前側レイヤー", compose_on_white(front_canvas)),
        ("最終合成結果", final_result),
    ]

    for row_start in range(0, len(debug_items), 3):
        cols = st.columns(3)
        for col, (caption, image) in zip(cols, debug_items[row_start : row_start + 3]):
            with col:
                st.image(image, caption=caption, width="stretch")


def main() -> None:
    """Streamlitアプリのエントリーポイント。"""
    st.set_page_config(page_title="AI髪型シミュレーター", layout="wide")

    st.title("AI髪型シミュレーター")
    st.write("写真を撮影して、ボブヘアを試してみよう！")

    debug_enabled = st.checkbox("デバッグ表示", value=False)

    try:
        ensure_bob_hair_assets(BACK_HAIR_PATH, FRONT_HAIR_PATH)
    except Exception as exc:
        _show_error(f"仮のボブヘア画像の準備に失敗しました: {exc}")
        return

    try:
        input_image = _select_input_image()
    except Exception as exc:
        _show_error(str(exc))
        return

    if input_image is None:
        st.info("カメラで撮影するか、画像をアップロードしてください。")
        return

    try:
        original, _, cutout_on_white, parsing_result, face_info = _get_processed_image(input_image)
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

    if face_info.face_count > 1:
        st.warning("複数人の顔が検出されました。今回は最も大きく写っている顔を使用します。できれば1人だけ写った画像を使ってください。")

    st.caption(f"髪領域の検出面積: {parsing_result.hair_area_ratio:.3%}")
    size_scale, offset_x, offset_y, rotation_adjust = _render_adjust_sliders(face_info)

    try:
        result = overlay_bob_hair_layers(
            parsing_result.hair_removed_rgba,
            face_info,
            BACK_HAIR_PATH,
            FRONT_HAIR_PATH,
            size_scale=size_scale,
            offset_x=offset_x,
            offset_y=offset_y,
            rotation_adjust=rotation_adjust,
        )
    except RuntimeError as exc:
        _show_error(str(exc))
        return
    except Exception as exc:
        _show_error(f"髪型画像の合成に失敗しました: {exc}")
        return

    back_canvas = render_hair_layer_canvas(
        original.size,
        face_info,
        BACK_HAIR_PATH,
        size_scale=size_scale,
        offset_x=offset_x,
        offset_y=offset_y,
        rotation_adjust=rotation_adjust,
    )
    front_canvas = render_hair_layer_canvas(
        original.size,
        face_info,
        FRONT_HAIR_PATH,
        size_scale=size_scale,
        offset_x=offset_x,
        offset_y=offset_y,
        rotation_adjust=rotation_adjust,
    )
    uncovered_ratio = estimate_uncovered_hair_mask_ratio(parsing_result.hair_mask, back_canvas, front_canvas)
    if uncovered_ratio > 0.08:
        st.warning(
            "元の髪を透明化した領域の一部を推薦髪型で覆いきれていない可能性があります。"
            "髪型サイズを大きくするか、上下左右位置を調整してください。"
        )

    st.subheader("完成画像")
    st.image(result, caption="ボブヘア合成後の画像", width="stretch")

    st.download_button(
        "完成画像をPNGでダウンロード",
        data=pil_to_png_bytes(result),
        file_name="bob_hairstyle_result.png",
        mime="image/png",
    )

    st.subheader("処理の確認")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(original, caption="元画像", width="stretch")
    with col2:
        st.image(cutout_on_white, caption="背景削除後の画像", width="stretch")
    with col3:
        st.image(result, caption="ボブヘア合成後", width="stretch")

    if debug_enabled:
        _render_debug_view(
            original,
            cutout_on_white,
            parsing_result,
            face_info,
            result,
            size_scale,
            offset_x,
            offset_y,
            rotation_adjust,
        )


if __name__ == "__main__":
    main()
