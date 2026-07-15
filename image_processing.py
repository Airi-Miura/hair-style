from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageOps
from rembg import remove


MAX_IMAGE_SIDE = 1400


def load_image_from_upload(uploaded_file: object) -> Image.Image:
    """StreamlitのアップロードファイルをRGBのPillow画像として読み込む。"""
    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
    except Exception as exc:
        raise ValueError("画像の読み込みに失敗しました。PNG、JPG、JPEG形式の画像を使ってください。") from exc


def resize_if_large(image: Image.Image, max_side: int = MAX_IMAGE_SIDE) -> Image.Image:
    """処理が重くなりすぎないよう、大きい画像だけ縦横比を保って縮小する。"""
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image

    scale = max_side / float(longest)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def remove_background(image_rgb: Image.Image) -> Image.Image:
    """rembgで人物の背景を削除し、透明背景のRGBA画像を返す。"""
    try:
        buffer = BytesIO()
        image_rgb.save(buffer, format="PNG")
        output_bytes = remove(buffer.getvalue())
        cutout = Image.open(BytesIO(output_bytes))
        return cutout.convert("RGBA")
    except Exception as exc:
        raise RuntimeError("背景削除に失敗しました。別の画像で試すか、rembg/onnxruntimeの導入を確認してください。") from exc


def compose_on_white(image_rgba: Image.Image) -> Image.Image:
    """透明背景の人物画像を真っ白な背景へ合成してRGB画像にする。"""
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255, 255))
    background.alpha_composite(image_rgba)
    return background.convert("RGB")


def make_hair_area_transparent(person_rgba: Image.Image, hair_mask: Image.Image, threshold: int = 80) -> Image.Image:
    """髪マスクに該当する人物RGBAのアルファ値を0にして透明化する。"""
    rgba = person_rgba.convert("RGBA")
    mask = hair_mask.convert("L").resize(rgba.size, Image.Resampling.BILINEAR)
    rgba_array = np.array(rgba)
    mask_array = np.array(mask)
    rgba_array[:, :, 3][mask_array > threshold] = 0
    return Image.fromarray(rgba_array, mode="RGBA")


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    """Pillow画像をMediaPipeへ渡せるRGB配列へ変換する。"""
    return np.array(image.convert("RGB"))


def pil_to_png_bytes(image: Image.Image) -> bytes:
    """Pillow画像をダウンロード用PNGバイト列へ変換する。"""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
