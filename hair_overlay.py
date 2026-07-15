from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

if TYPE_CHECKING:
    from face_detection import FaceInfo


HAIR_ASSET_SIZE = (900, 1100)
HAIR_FACE_CENTER_RATIO = (0.5, 0.55)
HAIR_BASE_WIDTH_RATIO = 2.08


def ensure_bob_hair_assets(back_path: str | Path, front_path: str | Path) -> tuple[Path, Path]:
    """前後2レイヤーの仮ボブヘア素材がなければ生成して保存する。"""
    back = Path(back_path)
    front = Path(front_path)
    if back.exists() and front.exists():
        return back, front

    back.parent.mkdir(parents=True, exist_ok=True)
    front.parent.mkdir(parents=True, exist_ok=True)
    back_image, front_image = generate_bob_hair_layers()
    if not back.exists():
        back_image.save(back, format="PNG")
    if not front.exists():
        front_image.save(front, format="PNG")
    return back, front


def generate_bob_hair_layers() -> tuple[Image.Image, Image.Image]:
    """Pillowで後ろ髪レイヤーと前髪レイヤーを生成する。"""
    scale = 3
    width, height = HAIR_ASSET_SIZE
    back = Image.new("RGBA", (width * scale, height * scale), (0, 0, 0, 0))
    front = Image.new("RGBA", (width * scale, height * scale), (0, 0, 0, 0))
    back_draw = ImageDraw.Draw(back)
    front_draw = ImageDraw.Draw(front)

    def sbox(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        return tuple(v * scale for v in box)

    dark = (38, 24, 18, 255)
    shadow = (18, 12, 10, 235)
    line = (18, 12, 10, 145)
    shine = (86, 58, 40, 90)

    # 後ろ側レイヤー: 後頭部と首・肩の後ろに見える外側の髪。
    back_draw.ellipse(sbox((100, 45, 800, 1030)), fill=shadow)
    back_draw.rounded_rectangle(sbox((112, 320, 788, 965)), radius=155 * scale, fill=dark)
    back_draw.ellipse(sbox((115, 65, 785, 760)), fill=dark)
    back_draw.ellipse(sbox((75, 280, 365, 1010)), fill=dark)
    back_draw.ellipse(sbox((535, 280, 825, 1010)), fill=dark)

    # 顔中央は透明にし、人物の顔を後ろレイヤーより前へ出す。
    cut = Image.new("L", back.size, 0)
    cut_draw = ImageDraw.Draw(cut)
    cut_draw.ellipse(sbox((245, 250, 655, 875)), fill=255)
    cut_draw.rounded_rectangle(sbox((285, 520, 615, 935)), radius=120 * scale, fill=255)
    back = Image.composite(Image.new("RGBA", back.size, (0, 0, 0, 0)), back, cut)
    back = back.filter(ImageFilter.GaussianBlur(radius=0.25 * scale))

    # 前側レイヤー: 前髪、顔横の髪、顎付近の毛先。
    bang_points = [
        (160, 252),
        (238, 142),
        (382, 118),
        (515, 126),
        (664, 155),
        (742, 260),
        (665, 402),
        (565, 330),
        (496, 426),
        (426, 323),
        (350, 418),
        (286, 332),
        (218, 405),
    ]
    front_draw.polygon([(x * scale, y * scale) for x, y in bang_points], fill=dark)
    front_draw.ellipse(sbox((82, 310, 322, 1000)), fill=dark)
    front_draw.ellipse(sbox((578, 310, 818, 1000)), fill=dark)
    front_draw.rounded_rectangle(sbox((132, 675, 302, 1000)), radius=80 * scale, fill=dark)
    front_draw.rounded_rectangle(sbox((598, 675, 768, 1000)), radius=80 * scale, fill=dark)
    for offset in (0, 34, 68):
        front_draw.arc(sbox((150 + offset, 120, 750 - offset, 940)), 195, 345, fill=line, width=5 * scale)
    front_draw.arc(sbox((250, 170, 500, 900)), 205, 295, fill=shine, width=7 * scale)
    front_draw.arc(sbox((400, 165, 650, 900)), 245, 335, fill=shine, width=7 * scale)
    front = front.filter(ImageFilter.GaussianBlur(radius=0.25 * scale))

    return (
        back.resize(HAIR_ASSET_SIZE, Image.Resampling.LANCZOS),
        front.resize(HAIR_ASSET_SIZE, Image.Resampling.LANCZOS),
    )


def calculate_hair_transform(
    face_info: FaceInfo,
    size_scale: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    rotation_adjust: float = 0.0,
) -> dict[str, float]:
    """顔位置に合わせた髪型素材の拡大率、配置、回転角を計算する。"""
    target_width = face_info.width * HAIR_BASE_WIDTH_RATIO * size_scale
    hair_width, hair_height = HAIR_ASSET_SIZE
    target_height = target_width * hair_height / hair_width
    x = face_info.center[0] - target_width * HAIR_FACE_CENTER_RATIO[0] + offset_x
    y = face_info.center[1] - target_height * HAIR_FACE_CENTER_RATIO[1] + offset_y
    return {
        "x": float(x),
        "y": float(y),
        "width": float(target_width),
        "height": float(target_height),
        "rotation": float(face_info.rotation_deg + rotation_adjust),
    }


def _prepare_layer(layer_path: str | Path, transform: dict[str, float]) -> tuple[Image.Image, int, int]:
    """髪レイヤーをリサイズ・回転し、合成座標を返す。"""
    target_size = (max(1, int(transform["width"])), max(1, int(transform["height"])))
    layer = Image.open(layer_path).convert("RGBA")
    layer = layer.resize(target_size, Image.Resampling.LANCZOS)
    layer = layer.rotate(transform["rotation"], resample=Image.Resampling.BICUBIC, expand=True)
    x = int(transform["x"] - (layer.size[0] - target_size[0]) / 2.0)
    y = int(transform["y"] - (layer.size[1] - target_size[1]) / 2.0)
    return layer, x, y


def _alpha_composite_safe(base: Image.Image, overlay: Image.Image, x: int, y: int) -> Image.Image:
    """画像外にはみ出す場合も安全にRGBA合成する。"""
    result = base.copy()
    base_width, base_height = result.size
    overlay_width, overlay_height = overlay.size
    left = max(0, x)
    top = max(0, y)
    right = min(base_width, x + overlay_width)
    bottom = min(base_height, y + overlay_height)
    if left >= right or top >= bottom:
        return result
    crop_box = (left - x, top - y, right - x, bottom - y)
    result.alpha_composite(overlay.crop(crop_box), (left, top))
    return result


def render_hair_layer_canvas(
    image_size: tuple[int, int],
    face_info: FaceInfo,
    layer_path: str | Path,
    size_scale: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    rotation_adjust: float = 0.0,
) -> Image.Image:
    """デバッグ表示用に、指定レイヤーだけを元画像サイズの透明キャンバスへ描画する。"""
    transform = calculate_hair_transform(face_info, size_scale, offset_x, offset_y, rotation_adjust)
    layer, x, y = _prepare_layer(layer_path, transform)
    canvas = Image.new("RGBA", image_size, (0, 0, 0, 0))
    return _alpha_composite_safe(canvas, layer, x, y)


def overlay_bob_hair_layers(
    person_without_hair_rgba: Image.Image,
    face_info: FaceInfo,
    back_path: str | Path,
    front_path: str | Path,
    size_scale: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    rotation_adjust: float = 0.0,
) -> Image.Image:
    """白背景、後ろ髪、髪透明化済み人物、前髪の順で合成してRGB画像を返す。"""
    try:
        transform = calculate_hair_transform(face_info, size_scale, offset_x, offset_y, rotation_adjust)
        base = Image.new("RGBA", person_without_hair_rgba.size, (255, 255, 255, 255))

        back_layer, back_x, back_y = _prepare_layer(back_path, transform)
        base = _alpha_composite_safe(base, back_layer, back_x, back_y)
        base.alpha_composite(person_without_hair_rgba.convert("RGBA"))

        front_layer, front_x, front_y = _prepare_layer(front_path, transform)
        base = _alpha_composite_safe(base, front_layer, front_x, front_y)
        return base.convert("RGB")
    except Exception as exc:
        raise RuntimeError("髪型画像の合成に失敗しました。位置やサイズを調整して再度お試しください。") from exc


def estimate_uncovered_hair_mask_ratio(
    hair_mask: Image.Image,
    back_layer_canvas: Image.Image,
    front_layer_canvas: Image.Image,
    hair_threshold: int = 80,
    cover_threshold: int = 30,
) -> float:
    """髪として削除した領域のうち、推薦髪型で覆えていない割合を返す。"""
    mask = np.array(hair_mask.convert("L"))
    back_alpha = np.array(back_layer_canvas.convert("RGBA"))[:, :, 3]
    front_alpha = np.array(front_layer_canvas.convert("RGBA"))[:, :, 3]
    hair_pixels = mask > hair_threshold
    if not np.any(hair_pixels):
        return 1.0
    covered = (back_alpha > cover_threshold) | (front_alpha > cover_threshold)
    uncovered = hair_pixels & ~covered
    return float(np.count_nonzero(uncovered) / np.count_nonzero(hair_pixels))


def rgba_alpha_coverage(image: Image.Image) -> float:
    """素材生成確認用に、透明でないピクセルの割合を返す。"""
    alpha = np.array(image.convert("RGBA"))[:, :, 3]
    return float(np.count_nonzero(alpha) / alpha.size)
