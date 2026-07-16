from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from face_detection import detect_face

if TYPE_CHECKING:
    from face_detection import FaceInfo


DESKTOP_OUTPUT_DIR = Path.home() / "Desktop" / "output"
REQUIRED_LAYER_NAMES = ("side_left", "side_right", "front", "strands")
ALL_LAYER_NAMES = ("hair_full", "side_left", "side_right", "front", "strands")


class HairstyleAssetError(RuntimeError):
    """髪型素材の読み込みに失敗した場合のエラー。"""


@dataclass(frozen=True)
class HairstyleAdjust:
    """metadata.jsonに任意で書ける推奨調整値。"""

    size_scale: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    rotation: float = 0.0


@dataclass(frozen=True)
class HairstyleAsset:
    """Desktop/output配下の1つの髪型素材セット。"""

    name: str
    directory: Path
    metadata_path: Path
    metadata: dict[str, Any]
    side_left: Path
    side_right: Path
    front: Path
    strands: Path
    hair_full: Path | None
    default_adjust: HairstyleAdjust
    original: Path | None = None


@dataclass(frozen=True)
class HairTransform:
    """髪型素材全レイヤーへ共通適用する変換。"""

    matrix: np.ndarray
    source_points: dict[str, tuple[float, float]]
    target_points: dict[str, tuple[float, float]]
    fallback_used: bool = False
    fallback_reason: str = ""


@dataclass(frozen=True)
class HairCompositionResult:
    """髪型合成結果とデバッグ用の中間画像。"""

    final_image: Image.Image
    fallback_used: bool
    fallback_reason: str
    debug_overlay: Image.Image
    hair_removed_person: Image.Image
    hair_full_only: Image.Image
    person_on_full: Image.Image
    front_side_only: Image.Image
    transformed_layers: dict[str, Image.Image]
    face_protection_mask: Image.Image
    actual_hair_removal_mask: Image.Image
    layer_transforms: dict[str, HairTransform]
    layer_sizes: dict[str, tuple[int, int]] = field(default_factory=dict)
    layer_alpha_bboxes: dict[str, tuple[int, int, int, int] | None] = field(default_factory=dict)
    source_landmarks: dict[str, tuple[float, float]] = field(default_factory=dict)
    target_landmarks: dict[str, tuple[float, float]] = field(default_factory=dict)
    transform_matrix: list[list[float]] = field(default_factory=list)
    layer_step_images: dict[str, Image.Image] = field(default_factory=dict)


def _read_json(path: Path) -> dict[str, Any]:
    """metadata.jsonをUTF-8で読み込む。"""
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        raise HairstyleAssetError(f"metadata.jsonを読み込めませんでした: {path}") from exc


def _first_number(metadata: dict[str, Any], keys: tuple[str, ...], default: float) -> float:
    """複数候補キーから数値を取得する。"""
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return default


def _metadata_adjust(metadata: dict[str, Any]) -> HairstyleAdjust:
    """metadata.jsonから推奨倍率、オフセット、回転を取得する。"""
    scale = _first_number(metadata, ("recommended_scale", "recommend_scale", "scale", "size_scale", "default_scale"), 1.0)
    rotation = _first_number(metadata, ("recommended_rotation", "recommend_rotation", "rotation", "rotation_deg", "default_rotation"), 0.0)
    offset_x = _first_number(metadata, ("recommended_offset_x", "offset_x", "default_offset_x"), 0.0)
    offset_y = _first_number(metadata, ("recommended_offset_y", "offset_y", "default_offset_y"), 0.0)
    offset = metadata.get("recommended_offset") or metadata.get("offset") or metadata.get("default_offset")
    if isinstance(offset, dict):
        if isinstance(offset.get("x"), (int, float)):
            offset_x = float(offset["x"])
        if isinstance(offset.get("y"), (int, float)):
            offset_y = float(offset["y"])
    return HairstyleAdjust(size_scale=scale, offset_x=offset_x, offset_y=offset_y, rotation=rotation)


def _validate_rgba_png(path: Path) -> None:
    """透過PNGとして読み込めることを確認する。"""
    if not path.exists():
        raise HairstyleAssetError(f"必要な髪型レイヤーが見つかりません: {path}")
    try:
        Image.open(path).convert("RGBA")
    except Exception as exc:
        raise HairstyleAssetError(f"髪型PNGを読み込めませんでした: {path}") from exc


def discover_hairstyles(output_dir: str | Path = DESKTOP_OUTPUT_DIR) -> list[HairstyleAsset]:
    """Desktop/outputからmetadata.jsonを持つ髪型フォルダ一覧を取得する。"""
    root = Path(output_dir)
    if not root.exists():
        return []

    hairstyles: list[HairstyleAsset] = []
    for directory in sorted([path for path in root.iterdir() if path.is_dir()], key=lambda p: p.name.lower()):
        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            continue

        paths = {name: directory / f"{name}.png" for name in REQUIRED_LAYER_NAMES}
        for path in paths.values():
            _validate_rgba_png(path)

        hair_full = directory / "hair_full.png"
        if hair_full.exists():
            _validate_rgba_png(hair_full)

        original = directory / "original.png"
        if not original.exists():
            raise HairstyleAssetError(f"original.png が見つかりません: {directory}")

        metadata = _read_json(metadata_path)
        hairstyles.append(
            HairstyleAsset(
                name=directory.name,
                directory=directory,
                metadata_path=metadata_path,
                metadata=metadata,
                side_left=paths["side_left"],
                side_right=paths["side_right"],
                front=paths["front"],
                strands=paths["strands"],
                hair_full=hair_full if hair_full.exists() else None,
                default_adjust=_metadata_adjust(metadata),
                original=original,
            )
        )
    return hairstyles


def _layer_paths(asset: HairstyleAsset) -> dict[str, Path]:
    """髪型レイヤー名からパスへの対応を返す。"""
    paths = {
        "side_left": asset.side_left,
        "side_right": asset.side_right,
        "front": asset.front,
        "strands": asset.strands,
    }
    if asset.hair_full:
        paths["hair_full"] = asset.hair_full
    return paths


def _point_from_mapping(value: Any) -> tuple[float, float] | None:
    """配列または{x,y}形式の座標を取り出す。"""
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        x, y = value[0], value[1]
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return float(x), float(y)
    if isinstance(value, dict):
        x, y = value.get("x"), value.get("y")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return float(x), float(y)
    return None


def _crop_bbox_for_layer(metadata: dict[str, Any], layer_name: str) -> tuple[int, int, int, int] | None:
    """metadataからレイヤーのcrop_bboxを読む。"""
    for key in ("crop_bboxes", "layer_crop_bboxes"):
        values = metadata.get(key)
        if isinstance(values, dict):
            bbox = _point_bbox(values.get(layer_name))
            if bbox:
                return bbox
    crop_bbox = metadata.get("crop_bbox")
    return _point_bbox(crop_bbox)


def _point_bbox(value: Any) -> tuple[int, int, int, int] | None:
    """bbox配列またはdictを(x1,y1,x2,y2)として読む。"""
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        vals = value[:4]
        if all(isinstance(v, (int, float)) for v in vals):
            return tuple(int(v) for v in vals)  # type: ignore[return-value]
    if isinstance(value, dict):
        keys = ("x1", "y1", "x2", "y2")
        if all(isinstance(value.get(k), (int, float)) for k in keys):
            return tuple(int(value[k]) for k in keys)  # type: ignore[return-value]
        if all(isinstance(value.get(k), (int, float)) for k in ("x", "y", "w", "h")):
            x, y, w, h = int(value["x"]), int(value["y"]), int(value["w"]), int(value["h"])
            return x, y, x + w, y + h
    return None


def _load_layer_on_original_canvas(asset: HairstyleAsset, layer_name: str) -> Image.Image:
    """レイヤーをoriginal.pngと同じキャンバス座標へ戻して読み込む。"""
    if asset.original is None:
        raise HairstyleAssetError(f"{asset.name}: original.png が必要です。")
    original_size = Image.open(asset.original).size
    path = _layer_paths(asset)[layer_name]
    layer = Image.open(path).convert("RGBA")
    if layer.size == original_size:
        return layer

    bbox = _crop_bbox_for_layer(asset.metadata, layer_name)
    if bbox is None:
        raise HairstyleAssetError(
            f"{asset.name}/{layer_name}.png は original.png とサイズが違います。metadata.json に crop_bbox を追加してください。"
        )
    x1, y1, x2, y2 = bbox
    expected_size = (x2 - x1, y2 - y1)
    if layer.size != expected_size:
        raise HairstyleAssetError(
            f"{asset.name}/{layer_name}.png のサイズ {layer.size} と crop_bbox {bbox} が一致しません。"
        )
    canvas = Image.new("RGBA", original_size, (0, 0, 0, 0))
    canvas.alpha_composite(layer, (x1, y1))
    return canvas


def _alpha_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    """RGBA画像の非透明領域bboxを返す。"""
    alpha = np.array(image.convert("RGBA"))[:, :, 3]
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _landmarks_from_face_info(face_info: FaceInfo) -> dict[str, tuple[float, float]]:
    """FaceInfoから変換に使うランドマークを取り出す。"""
    points = {
        "left_eye": getattr(face_info, "left_eye_center", None),
        "right_eye": getattr(face_info, "right_eye_center", None),
        "nose": getattr(face_info, "nose_tip", None),
        "chin": face_info.chin,
        "left_temple": getattr(face_info, "left_temple", None) or face_info.left,
        "right_temple": getattr(face_info, "right_temple", None) or face_info.right,
    }
    return {k: v for k, v in points.items() if v is not None}


def _original_landmarks(asset: HairstyleAsset) -> dict[str, tuple[float, float]]:
    """original.pngをMediaPipeで解析し、変換元ランドマークを取得する。"""
    if asset.original is None:
        raise HairstyleAssetError(f"{asset.name}: original.png が必要です。")
    original_rgb = np.array(Image.open(asset.original).convert("RGB"))
    return _landmarks_from_face_info(detect_face(original_rgb))


def _validate_transform(matrix: np.ndarray) -> None:
    """極端な変換を検出する。"""
    if matrix is None or matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError("変換行列を推定できませんでした。")
    a, b = float(matrix[0, 0]), float(matrix[0, 1])
    c, d = float(matrix[1, 0]), float(matrix[1, 1])
    scale = (math.hypot(a, c) + math.hypot(b, d)) / 2.0
    rotation = math.degrees(math.atan2(c, a))
    if scale < 0.2 or scale > 5.0:
        raise ValueError(f"拡大率が異常です: {scale:.2f}")
    if abs(rotation) > 40:
        raise ValueError(f"回転角度が大きすぎます: {rotation:.1f}")


def _fallback_matrix(face_info: FaceInfo, asset: HairstyleAsset) -> np.ndarray:
    """original.png全体を顔幅に合わせるフォールバック行列を作る。"""
    if asset.original is None:
        raise HairstyleAssetError(f"{asset.name}: original.png が必要です。")
    original_width, original_height = Image.open(asset.original).size
    target_width = face_info.width * 2.1
    scale = target_width / max(1.0, original_width)
    x = face_info.center[0] - original_width * scale * 0.5
    y = face_info.center[1] - original_height * scale * 0.5
    return np.array([[scale, 0.0, x], [0.0, scale, y]], dtype=np.float32)


def estimate_global_transform(
    face_info: FaceInfo,
    asset: HairstyleAsset,
    size_scale: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    rotation_adjust: float = 0.0,
) -> HairTransform:
    """original.png側と撮影画像側の顔ランドマークから、全レイヤー共通変換を推定する。"""
    source = _original_landmarks(asset)
    target = _landmarks_from_face_info(face_info)
    keys = [key for key in ("left_eye", "right_eye", "nose", "chin", "left_temple", "right_temple") if key in source and key in target]
    fallback_used = False
    fallback_reason = ""
    try:
        if len(keys) < 3:
            raise ValueError("変換に必要なランドマークが不足しています。")
        src = np.array([source[key] for key in keys], dtype=np.float32)
        dst = np.array([target[key] for key in keys], dtype=np.float32)
        matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        _validate_transform(matrix)
        matrix = matrix.astype(np.float32)
    except Exception as exc:
        matrix = _fallback_matrix(face_info, asset)
        fallback_used = True
        fallback_reason = str(exc)

    matrix = apply_manual_adjustment(matrix, face_info.center, size_scale, offset_x, offset_y, rotation_adjust)
    return HairTransform(matrix=matrix, source_points=source, target_points=target, fallback_used=fallback_used, fallback_reason=fallback_reason)


def apply_manual_adjustment(
    matrix: np.ndarray,
    center: tuple[float, float],
    size_scale: float,
    offset_x: float,
    offset_y: float,
    rotation_adjust: float,
) -> np.ndarray:
    """自動変換結果へ手動スライダーの追加補正を適用する。"""
    rotate = cv2.getRotationMatrix2D(center, rotation_adjust, size_scale)
    rotate_3 = np.vstack([rotate, [0.0, 0.0, 1.0]])
    translate_3 = np.array([[1.0, 0.0, offset_x], [0.0, 1.0, offset_y], [0.0, 0.0, 1.0]], dtype=np.float64)
    matrix_3 = np.vstack([matrix, [0.0, 0.0, 1.0]])
    adjusted = translate_3 @ rotate_3 @ matrix_3
    return adjusted[:2].astype(np.float32)


def _warp_layer(layer: Image.Image, matrix: np.ndarray, image_size: tuple[int, int]) -> Image.Image:
    """RGBAレイヤーを共通行列で出力キャンバスへ変換する。"""
    width, height = image_size
    rgba = np.array(layer.convert("RGBA"))
    warped = cv2.warpAffine(
        rgba,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return Image.fromarray(warped.astype(np.uint8), mode="RGBA")


def transform_hair_layers(asset: HairstyleAsset, transform: HairTransform | dict[str, HairTransform] | np.ndarray, image_size: tuple[int, int]) -> dict[str, Image.Image]:
    """全髪レイヤーへ同じ変換行列を適用する。"""
    if isinstance(transform, dict):
        matrix = next(iter(transform.values())).matrix
    elif isinstance(transform, HairTransform):
        matrix = transform.matrix
    else:
        matrix = transform

    layers: dict[str, Image.Image] = {}
    for layer_name in ALL_LAYER_NAMES:
        if layer_name == "hair_full" and asset.hair_full is None:
            layers[layer_name] = Image.new("RGBA", image_size, (0, 0, 0, 0))
            continue
        source_layer = _load_layer_on_original_canvas(asset, layer_name)
        layers[layer_name] = _warp_layer(source_layer, matrix, image_size)
    return layers


def _create_face_protection_mask(image_size: tuple[int, int], face_info: FaceInfo) -> Image.Image:
    """顔中央を人物優先にする保護マスクを作る。"""
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    cx, cy = face_info.center
    w = face_info.width
    h = max(1.0, face_info.height)
    brow_y = (getattr(face_info, "brow_center", None) or (cx, cy - h * 0.22))[1]
    chin_y = face_info.chin[1]
    draw.ellipse((int(cx - w * 0.32), int(brow_y + h * 0.16), int(cx + w * 0.32), int(chin_y - h * 0.05)), fill=255)
    eye_y = brow_y + h * 0.16
    draw.rounded_rectangle((int(cx - w * 0.39), int(eye_y - h * 0.055), int(cx + w * 0.39), int(eye_y + h * 0.085)), radius=max(1, int(w * 0.08)), fill=255)
    return mask.filter(ImageFilter.GaussianBlur(radius=max(1, int(w * 0.015))))


def _apply_protection(layer: Image.Image, protection_mask: Image.Image, strength: float = 1.0) -> Image.Image:
    """顔保護マスク部分の髪レイヤーアルファを弱める。"""
    rgba = np.array(layer.convert("RGBA")).astype(np.float32)
    mask = np.array(protection_mask.convert("L")).astype(np.float32) / 255.0
    rgba[:, :, 3] *= 1.0 - np.clip(mask * strength, 0.0, 1.0)
    return Image.fromarray(np.clip(rgba, 0, 255).astype(np.uint8), mode="RGBA")


def _alpha_composite(base: Image.Image, overlay: Image.Image) -> Image.Image:
    """同じサイズのRGBA画像をアルファ合成する。"""
    result = base.copy()
    result.alpha_composite(overlay.convert("RGBA"))
    return result


def _draw_debug_overlay(base: Image.Image, transform: HairTransform, layers: dict[str, Image.Image]) -> Image.Image:
    """位置合わせ確認用にランドマークと髪アルファ領域を描画する。"""
    debug = base.convert("RGBA")
    alpha = np.zeros((debug.size[1], debug.size[0]), dtype=np.uint8)
    for layer in layers.values():
        alpha = np.maximum(alpha, np.array(layer.convert("RGBA"))[:, :, 3])
    red = Image.new("RGBA", debug.size, (255, 0, 0, 65))
    mask = Image.fromarray((alpha > 15).astype(np.uint8) * 255, mode="L")
    debug = Image.alpha_composite(debug, Image.composite(red, Image.new("RGBA", debug.size, (0, 0, 0, 0)), mask))
    draw = ImageDraw.Draw(debug)
    for label, point in transform.target_points.items():
        x, y = point
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(0, 180, 255, 255))
        draw.text((x + 6, y - 6), f"T-{label}", fill=(0, 120, 255, 255))
    for label, point in transform.source_points.items():
        x, y = _transform_points(transform.matrix, [point])[0]
        draw.rectangle((x - 5, y - 5, x + 5, y + 5), fill=(255, 230, 0, 255))
        draw.text((x + 6, y + 4), f"S-{label}", fill=(210, 160, 0, 255))
    return debug.convert("RGB")


def _transform_points(matrix: np.ndarray, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """アフィン変換後の点を返す。"""
    transformed: list[tuple[float, float]] = []
    for x, y in points:
        tx = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
        ty = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
        transformed.append((float(tx), float(ty)))
    return transformed


def _layer_debug(asset: HairstyleAsset) -> tuple[dict[str, tuple[int, int]], dict[str, tuple[int, int, int, int] | None]]:
    """レイヤーサイズと非透明bboxを返す。"""
    sizes: dict[str, tuple[int, int]] = {}
    bboxes: dict[str, tuple[int, int, int, int] | None] = {}
    if asset.original is not None:
        original = Image.open(asset.original)
        sizes["original"] = original.size
        bboxes["original"] = _alpha_bbox(original.convert("RGBA"))
    for name in ALL_LAYER_NAMES:
        if name == "hair_full" and asset.hair_full is None:
            continue
        layer = _load_layer_on_original_canvas(asset, name)
        sizes[name] = layer.size
        bboxes[name] = _alpha_bbox(layer)
    return sizes, bboxes


def compose_hair_layers(
    person_without_hair_rgba: Image.Image,
    face_info: FaceInfo,
    asset: HairstyleAsset,
    size_scale: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    rotation_adjust: float = 0.0,
    enabled_layers: tuple[str, ...] | None = None,
) -> HairCompositionResult:
    """元髪透明化済み人物画像へ、選択髪型レイヤーを同一行列で合成する。"""
    image_size = person_without_hair_rgba.size
    metadata_adjust = asset.default_adjust
    transform = estimate_global_transform(
        face_info,
        asset,
        metadata_adjust.size_scale * size_scale,
        metadata_adjust.offset_x + offset_x,
        metadata_adjust.offset_y + offset_y,
        metadata_adjust.rotation + rotation_adjust,
    )
    layers = transform_hair_layers(asset, transform, image_size)
    sizes, bboxes = _layer_debug(asset)
    protection = _create_face_protection_mask(image_size, face_info)

    active = set(enabled_layers or ALL_LAYER_NAMES)
    protected_layers: dict[str, Image.Image] = {}
    for name in ALL_LAYER_NAMES:
        layer = layers[name] if name in active else Image.new("RGBA", image_size, (0, 0, 0, 0))
        if name == "hair_full":
            protected_layers[name] = _apply_protection(layer, protection, strength=1.0)
        elif name in ("side_left", "side_right"):
            protected_layers[name] = _apply_protection(layer, protection, strength=0.95)
        elif name == "front":
            protected_layers[name] = _apply_protection(layer, protection, strength=0.70)
        else:
            protected_layers[name] = _apply_protection(layer, protection, strength=0.55)

    base_person = person_without_hair_rgba.convert("RGBA").copy()
    base = Image.new("RGBA", image_size, (255, 255, 255, 255))
    hair_full_only = _alpha_composite(base, protected_layers["hair_full"])
    person_on_full = _alpha_composite(hair_full_only, base_person)

    step_images: dict[str, Image.Image] = {"hair_full": hair_full_only.convert("RGB"), "person": person_on_full.convert("RGB")}
    current = person_on_full
    for name in ("side_left", "side_right", "front", "strands"):
        current = _alpha_composite(current, protected_layers[name])
        step_images[name] = current.convert("RGB")

    final = current
    debug = _draw_debug_overlay(final, transform, protected_layers)
    return HairCompositionResult(
        final_image=final.convert("RGB"),
        fallback_used=transform.fallback_used,
        fallback_reason=transform.fallback_reason,
        debug_overlay=debug,
        hair_removed_person=Image.alpha_composite(Image.new("RGBA", image_size, (255, 255, 255, 255)), base_person).convert("RGB"),
        hair_full_only=hair_full_only.convert("RGB"),
        person_on_full=person_on_full.convert("RGB"),
        front_side_only=step_images.get("front", person_on_full),
        transformed_layers=protected_layers,
        face_protection_mask=protection,
        actual_hair_removal_mask=Image.new("L", image_size, 0),
        layer_transforms={name: transform for name in ALL_LAYER_NAMES},
        layer_sizes=sizes,
        layer_alpha_bboxes=bboxes,
        source_landmarks=transform.source_points,
        target_landmarks=transform.target_points,
        transform_matrix=transform.matrix.astype(float).tolist(),
        layer_step_images=step_images,
    )


def render_hairstyle_layer_canvas(
    image_size: tuple[int, int],
    face_info: FaceInfo,
    asset: HairstyleAsset,
    layer_name: str,
    size_scale: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    rotation_adjust: float = 0.0,
) -> Image.Image:
    """デバッグ表示用に指定レイヤーだけを透過キャンバスへ描画する。"""
    metadata_adjust = asset.default_adjust
    transform = estimate_global_transform(
        face_info,
        asset,
        metadata_adjust.size_scale * size_scale,
        metadata_adjust.offset_x + offset_x,
        metadata_adjust.offset_y + offset_y,
        metadata_adjust.rotation + rotation_adjust,
    )
    return transform_hair_layers(asset, transform, image_size)[layer_name]


def estimate_uncovered_hair_mask_ratio(
    hair_mask: Image.Image,
    layer_canvases: list[Image.Image],
    hair_threshold: int = 80,
    cover_threshold: int = 30,
) -> float:
    """髪として検出された領域のうち、髪素材で覆えていない割合を返す。"""
    mask = np.array(hair_mask.convert("L"))
    hair_pixels = mask > hair_threshold
    if not np.any(hair_pixels):
        return 1.0
    covered = np.zeros(mask.shape, dtype=bool)
    for canvas in layer_canvases:
        alpha = np.array(canvas.convert("RGBA"))[:, :, 3]
        covered |= alpha > cover_threshold
    uncovered = hair_pixels & ~covered
    return float(np.count_nonzero(uncovered) / np.count_nonzero(hair_pixels))


def rgba_alpha_coverage(image: Image.Image) -> float:
    """透明ではないピクセルの割合を返す。"""
    alpha = np.array(image.convert("RGBA"))[:, :, 3]
    return float(np.count_nonzero(alpha) / alpha.size)
