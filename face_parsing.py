from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import OrderedDict as OrderedDictType

import cv2
import numpy as np
from PIL import Image


# CelebAMask-HQの一般的なFace Parsing定義:
# 0 background, 1 skin, 2 left brow, 3 right brow, 4 left eye, 5 right eye,
# 6 eye glasses, 7 left ear, 8 right ear, 9 ear ring, 10 nose, 11 mouth,
# 12 upper lip, 13 lower lip, 14 neck, 15 neck lace, 16 cloth, 17 hair, 18 hat.
# zllrunning/face-parsing.PyTorch 系のBiSeNet 19クラスモデルでは hair は 17。
HAIR_CLASS_ID = 17
NUM_CLASSES = 19
INPUT_SIZE = 512


class FaceParsingModelNotFoundError(FileNotFoundError):
    """Face Parsingモデルが見つからない場合のエラー。"""


class FaceParsingRuntimeError(RuntimeError):
    """Face Parsing推論に失敗した場合のエラー。"""


@dataclass(frozen=True)
class FaceParsingResult:
    """Face Parsingの結果をまとめたデータ。"""

    class_map: Image.Image
    hair_mask: Image.Image
    hair_overlay_preview: Image.Image
    hair_removed_rgba: Image.Image
    hair_area_ratio: float
    raw_hair_mask: Image.Image
    postprocessed_hair_mask: Image.Image
    dilation_added_mask: Image.Image
    color_assist_mask: Image.Image
    final_hair_mask: Image.Image
    hair_mask_bbox: tuple[int, int, int, int] | None
    hair_mask_area_pixels: int


def _lazy_import_torch() -> tuple[object, object]:
    """PyTorchを必要なタイミングで読み込む。"""
    try:
        import torch
        import torch.nn as nn

        return torch, nn
    except Exception as exc:
        raise FaceParsingRuntimeError(
            "Face ParsingにはPyTorchが必要です。requirements.txtを使ってライブラリをインストールしてください。"
        ) from exc


def _build_bisenet_model() -> object:
    """CelebAMask-HQ向けBiSeNetモデルを構築する。"""
    torch, nn = _lazy_import_torch()

    class ConvBNReLU(nn.Module):
        def __init__(self, in_chan: int, out_chan: int, ks: int = 3, stride: int = 1, padding: int = 1) -> None:
            super().__init__()
            self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False)
            self.bn = nn.BatchNorm2d(out_chan)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x: object) -> object:
            return self.relu(self.bn(self.conv(x)))

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_chan: int, out_chan: int, stride: int = 1) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(in_chan, out_chan, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_chan)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_chan, out_chan, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_chan)
            self.downsample = None
            if stride != 1 or in_chan != out_chan:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_chan, out_chan, 1, stride, bias=False),
                    nn.BatchNorm2d(out_chan),
                )

        def forward(self, x: object) -> object:
            identity = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            if self.downsample is not None:
                identity = self.downsample(x)
            return self.relu(out + identity)

    class Resnet18(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_chan = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(64, 2, 1)
            self.layer2 = self._make_layer(128, 2, 2)
            self.layer3 = self._make_layer(256, 2, 2)
            self.layer4 = self._make_layer(512, 2, 2)

        def _make_layer(self, out_chan: int, blocks: int, stride: int) -> object:
            layers = [BasicBlock(self.in_chan, out_chan, stride)]
            self.in_chan = out_chan
            for _ in range(1, blocks):
                layers.append(BasicBlock(self.in_chan, out_chan, 1))
            return nn.Sequential(*layers)

        def forward(self, x: object) -> tuple[object, object, object]:
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            x = self.layer1(x)
            feat8 = self.layer2(x)
            feat16 = self.layer3(feat8)
            feat32 = self.layer4(feat16)
            return feat8, feat16, feat32

    class AttentionRefinementModule(nn.Module):
        def __init__(self, in_chan: int, out_chan: int) -> None:
            super().__init__()
            self.conv = ConvBNReLU(in_chan, out_chan, 3, 1, 1)
            self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
            self.bn_atten = nn.BatchNorm2d(out_chan)
            self.sigmoid_atten = nn.Sigmoid()

        def forward(self, x: object) -> object:
            feat = self.conv(x)
            atten = torch.mean(feat, dim=(2, 3), keepdim=True)
            atten = self.sigmoid_atten(self.bn_atten(self.conv_atten(atten)))
            return feat * atten

    class ContextPath(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.resnet = Resnet18()
            self.arm16 = AttentionRefinementModule(256, 128)
            self.arm32 = AttentionRefinementModule(512, 128)
            self.conv_head32 = ConvBNReLU(128, 128, 3, 1, 1)
            self.conv_head16 = ConvBNReLU(128, 128, 3, 1, 1)
            self.conv_avg = ConvBNReLU(512, 128, 1, 1, 0)

        def forward(self, x: object) -> tuple[object, object, object]:
            feat8, feat16, feat32 = self.resnet(x)
            avg = torch.mean(feat32, dim=(2, 3), keepdim=True)
            avg = self.conv_avg(avg)
            avg_up = torch.nn.functional.interpolate(avg, size=feat32.shape[2:], mode="nearest")
            feat32_arm = self.arm32(feat32)
            feat32_sum = feat32_arm + avg_up
            feat32_up = torch.nn.functional.interpolate(feat32_sum, size=feat16.shape[2:], mode="nearest")
            feat32_up = self.conv_head32(feat32_up)
            feat16_arm = self.arm16(feat16)
            feat16_sum = feat16_arm + feat32_up
            feat16_up = torch.nn.functional.interpolate(feat16_sum, size=feat8.shape[2:], mode="nearest")
            feat16_up = self.conv_head16(feat16_up)
            return feat8, feat16_up, feat32_up

    class FeatureFusionModule(nn.Module):
        def __init__(self, in_chan: int, out_chan: int) -> None:
            super().__init__()
            self.convblk = ConvBNReLU(in_chan, out_chan, 1, 1, 0)
            self.conv1 = nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, bias=False)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, bias=False)
            self.sigmoid = nn.Sigmoid()

        def forward(self, fsp: object, fcp: object) -> object:
            feat = self.convblk(torch.cat([fsp, fcp], dim=1))
            atten = torch.mean(feat, dim=(2, 3), keepdim=True)
            atten = self.conv2(self.relu(self.conv1(atten)))
            atten = self.sigmoid(atten)
            return feat * atten + feat

    class BiSeNetOutput(nn.Module):
        def __init__(self, in_chan: int, mid_chan: int, n_classes: int) -> None:
            super().__init__()
            self.conv = ConvBNReLU(in_chan, mid_chan, 3, 1, 1)
            self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

        def forward(self, x: object) -> object:
            return self.conv_out(self.conv(x))

    class BiSeNet(nn.Module):
        def __init__(self, n_classes: int) -> None:
            super().__init__()
            self.cp = ContextPath()
            self.ffm = FeatureFusionModule(256, 256)
            self.conv_out = BiSeNetOutput(256, 256, n_classes)
            self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
            self.conv_out32 = BiSeNetOutput(128, 64, n_classes)

        def forward(self, x: object) -> tuple[object, object, object]:
            height, width = x.shape[2:]
            feat8, feat16, feat32 = self.cp(x)
            feat_fuse = self.ffm(feat8, feat16)
            out = torch.nn.functional.interpolate(self.conv_out(feat_fuse), (height, width), mode="bilinear", align_corners=True)
            out16 = torch.nn.functional.interpolate(self.conv_out16(feat16), (height, width), mode="bilinear", align_corners=True)
            out32 = torch.nn.functional.interpolate(self.conv_out32(feat32), (height, width), mode="bilinear", align_corners=True)
            return out, out16, out32

    return BiSeNet(NUM_CLASSES)


def _normalize_state_dict(state: object) -> OrderedDictType[str, object]:
    """DataParallelやcheckpoint形式の差を吸収してstate_dictを返す。"""
    from collections import OrderedDict

    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break

    if not isinstance(state, dict):
        raise FaceParsingRuntimeError("Face Parsingモデルの形式を読み取れませんでした。")

    normalized = OrderedDict()
    for key, value in state.items():
        new_key = key[7:] if key.startswith("module.") else key
        normalized[new_key] = value
    return normalized


def load_bisenet_model(model_path: str | Path, device: str = "cpu") -> object:
    """BiSeNetの重みを読み込み、推論可能なモデルを返す。"""
    path = Path(model_path)
    if not path.exists():
        raise FaceParsingModelNotFoundError(
            f"Face Parsingモデルが見つかりません: {path}\n"
            "models/face_parsing_bisenet.pth にCelebAMask-HQ用BiSeNetの重みを配置してください。"
        )

    torch, _ = _lazy_import_torch()
    try:
        model = _build_bisenet_model()
        state = torch.load(path, map_location=device)
        model.load_state_dict(_normalize_state_dict(state), strict=False)
        model.to(device)
        model.eval()
        return model
    except Exception as exc:
        raise FaceParsingRuntimeError(
            "Face Parsingモデルの読み込みに失敗しました。CelebAMask-HQ用BiSeNetの重みか確認してください。"
        ) from exc


def _preprocess(image_rgb: Image.Image) -> object:
    """BiSeNetへ入力するためにRGB画像を512x512の正規化テンソルへ変換する。"""
    torch, _ = _lazy_import_torch()
    resized = image_rgb.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
    array = np.asarray(resized).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    array = (array - mean) / std
    array = array.transpose(2, 0, 1)
    return torch.from_numpy(array).unsqueeze(0)


def _postprocess_hair_mask(raw_mask: np.ndarray) -> np.ndarray:
    """髪領域マスクのノイズ除去、穴埋め、平滑化を行う。"""
    mask = (raw_mask > 0).astype(np.uint8) * 255
    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5, iterations=2)
    mask = cv2.dilate(mask, kernel3, iterations=1)
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    _, mask = cv2.threshold(mask, 40, 255, cv2.THRESH_BINARY)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    return mask.astype(np.uint8)


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """髪の細い毛先を残すため、かなり小さい孤立ノイズだけを除去する。"""
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary, dtype=np.uint8)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def _hair_mask_bbox(mask: np.ndarray, threshold: int = 80) -> tuple[int, int, int, int] | None:
    """髪マスクの非ゼロ領域bboxを返す。"""
    ys, xs = np.where(mask > threshold)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _safe_point(point: object, fallback: tuple[float, float]) -> tuple[float, float]:
    """MediaPipe由来の点がない場合に安全な代替点を使う。"""
    if isinstance(point, tuple) and len(point) == 2:
        return float(point[0]), float(point[1])
    return fallback


def _estimate_face_roi(image_rgb: Image.Image) -> dict[str, float]:
    """MediaPipeの顔位置から、髪補助検出に使う顔周辺ROIを作る。"""
    width, height = image_rgb.size
    default = {
        "x1": width * 0.18,
        "y1": height * 0.02,
        "x2": width * 0.82,
        "y2": height * 0.78,
        "face_left": width * 0.35,
        "face_right": width * 0.65,
        "brow_y": height * 0.34,
        "chin_y": height * 0.68,
        "face_width": width * 0.30,
        "face_height": height * 0.42,
    }
    try:
        from face_detection import detect_face

        face = detect_face(np.asarray(image_rgb.convert("RGB")))
        left = _safe_point(face.left, (default["face_left"], height * 0.45))
        right = _safe_point(face.right, (default["face_right"], height * 0.45))
        chin = _safe_point(face.chin, (width * 0.5, default["chin_y"]))
        top = _safe_point(face.top, (width * 0.5, height * 0.10))
        brow = _safe_point(face.brow_center, (width * 0.5, default["brow_y"]))
        face_width = max(1.0, abs(right[0] - left[0]))
        face_height = max(1.0, abs(chin[1] - top[1]))
        return {
            "x1": max(0.0, left[0] - face_width * 0.55),
            "y1": max(0.0, top[1] - face_height * 0.22),
            "x2": min(float(width), right[0] + face_width * 0.55),
            "y2": min(float(height), chin[1] + face_height * 0.18),
            "face_left": left[0],
            "face_right": right[0],
            "brow_y": brow[1],
            "chin_y": chin[1],
            "face_width": face_width,
            "face_height": face_height,
        }
    except Exception:
        return default


def _make_auxiliary_allowed_area(image_rgb: Image.Image) -> np.ndarray:
    """顔中央を避け、左右外側と頭頂部だけを髪補助検出の対象にする。"""
    width, height = image_rgb.size
    roi = _estimate_face_roi(image_rgb)
    yy, xx = np.indices((height, width))
    in_roi = (xx >= roi["x1"]) & (xx <= roi["x2"]) & (yy >= roi["y1"]) & (yy <= roi["y2"])
    side_margin = roi["face_width"] * 0.12
    left_side = xx < (roi["face_left"] + side_margin)
    right_side = xx > (roi["face_right"] - side_margin)
    top_area = yy < (roi["brow_y"] - roi["face_height"] * 0.03)
    below_chin_limit = yy <= (roi["chin_y"] + roi["face_height"] * 0.14)
    allowed = in_roi & below_chin_limit & (left_side | right_side | top_area)
    return allowed.astype(np.uint8) * 255


def _dark_connected_hair_candidates(image_rgb: Image.Image, base_mask: np.ndarray) -> np.ndarray:
    """既存髪マスクに接している暗色領域だけを髪候補として追加する。"""
    rgb = np.asarray(image_rgb.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    allowed = _make_auxiliary_allowed_area(image_rgb)
    near_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    near_hair = cv2.dilate((base_mask > 80).astype(np.uint8) * 255, near_kernel, iterations=3)
    dark = ((gray < 105) & ((saturation > 18) | (value < 75))).astype(np.uint8) * 255
    candidates = cv2.bitwise_and(dark, allowed)
    candidates = cv2.bitwise_and(candidates, near_hair)
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    count, labels, stats, _ = cv2.connectedComponentsWithStats((candidates > 0).astype(np.uint8), connectivity=8)
    kept = np.zeros_like(candidates, dtype=np.uint8)
    base_touch = cv2.dilate((base_mask > 80).astype(np.uint8) * 255, np.ones((3, 3), np.uint8), iterations=1)
    min_area = max(8, int(candidates.size * 0.000006))
    max_area = max(300, int(candidates.size * 0.075))
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        component = labels == label
        if np.any(base_touch[component] > 0) or np.any(near_hair[component] > 0):
            kept[component] = 255
    return kept


def _postprocess_hair_mask(raw_mask: np.ndarray, image_rgb: Image.Image) -> dict[str, object]:
    """髪マスクのノイズ除去、軽い膨張、暗色連結補助、境界フェザーを行う。"""
    raw = (raw_mask > 0).astype(np.uint8) * 255
    min_component_area = max(8, int(raw.size * 0.000006))
    cleaned = _remove_small_components(raw, min_component_area)

    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel5, iterations=1)
    dilated_binary = cv2.dilate(closed, kernel3, iterations=1)
    dilation_added = cv2.subtract(dilated_binary, closed)

    color_assist = _dark_connected_hair_candidates(image_rgb, dilated_binary)
    combined_binary = np.maximum(dilated_binary, color_assist)
    combined_binary = cv2.morphologyEx(combined_binary, cv2.MORPH_CLOSE, kernel3, iterations=1)

    feathered = cv2.GaussianBlur(combined_binary, (5, 5), 0)
    final = np.maximum(combined_binary, feathered).astype(np.uint8)
    return {
        "raw": raw,
        "postprocessed": closed,
        "dilation_added": dilation_added,
        "color_assist": color_assist,
        "final": final,
        "bbox": _hair_mask_bbox(final),
        "area_pixels": int(np.count_nonzero(final > 80)),
    }


def _class_map_to_color(class_map: np.ndarray) -> Image.Image:
    """Face Parsingのクラス分類結果をデバッグ用カラー画像へ変換する。"""
    palette = np.array(
        [
            [0, 0, 0],
            [255, 220, 185],
            [255, 180, 120],
            [255, 180, 120],
            [80, 160, 255],
            [80, 160, 255],
            [130, 130, 130],
            [255, 200, 160],
            [255, 200, 160],
            [255, 210, 80],
            [255, 120, 120],
            [220, 70, 120],
            [210, 60, 100],
            [190, 40, 90],
            [190, 150, 120],
            [180, 180, 200],
            [80, 200, 120],
            [255, 40, 40],
            [120, 80, 200],
        ],
        dtype=np.uint8,
    )
    safe_map = np.clip(class_map, 0, len(palette) - 1)
    return Image.fromarray(palette[safe_map], mode="RGB")


def create_hair_overlay_preview(image_rgb: Image.Image, hair_mask: Image.Image) -> Image.Image:
    """元画像の髪領域を半透明赤で重ねた確認用画像を作る。"""
    base = image_rgb.convert("RGBA")
    mask = hair_mask.convert("L")
    red = Image.new("RGBA", base.size, (255, 0, 0, 110))
    transparent = Image.new("RGBA", base.size, (0, 0, 0, 0))
    overlay = Image.composite(red, transparent, mask)
    return Image.alpha_composite(base, overlay).convert("RGB")


def run_face_parsing(
    image_rgb: Image.Image,
    person_rgba: Image.Image,
    model_path: str | Path,
    device: str = "cpu",
    min_hair_ratio: float = 0.004,
) -> FaceParsingResult:
    """Face Parsingで髪領域を検出し、人物RGBAの髪部分を透明化する。"""
    torch, _ = _lazy_import_torch()
    model = load_bisenet_model(model_path, device=device)

    try:
        width, height = image_rgb.size
        tensor = _preprocess(image_rgb).to(device)
        with torch.no_grad():
            logits = model(tensor)[0]
            parsing = logits.squeeze(0).argmax(0).detach().cpu().numpy().astype(np.uint8)

        parsing = cv2.resize(parsing, (width, height), interpolation=cv2.INTER_NEAREST)
        raw_hair = (parsing == HAIR_CLASS_ID).astype(np.uint8)
        mask_debug = _postprocess_hair_mask(raw_hair, image_rgb)
        hair_mask_array = mask_debug["final"]
        hair_area_ratio = float(np.count_nonzero(hair_mask_array > 80) / hair_mask_array.size)

        if hair_area_ratio < min_hair_ratio:
            raise FaceParsingRuntimeError(
                "髪領域を正しく検出できませんでした。\n"
                "正面を向き、頭頂部まで写っている明るい画像を使用してください。\n"
                "デバッグ表示で髪マスクを確認してください。"
            )

        hair_mask = Image.fromarray(hair_mask_array, mode="L")
        class_map = _class_map_to_color(parsing)
        overlay_preview = create_hair_overlay_preview(image_rgb, hair_mask)

        from image_processing import create_person_without_hair

        hair_removed = create_person_without_hair(person_rgba, hair_mask)
        return FaceParsingResult(
            class_map=class_map,
            hair_mask=hair_mask,
            hair_overlay_preview=overlay_preview,
            hair_removed_rgba=hair_removed,
            hair_area_ratio=hair_area_ratio,
            raw_hair_mask=Image.fromarray(mask_debug["raw"], mode="L"),
            postprocessed_hair_mask=Image.fromarray(mask_debug["postprocessed"], mode="L"),
            dilation_added_mask=Image.fromarray(mask_debug["dilation_added"], mode="L"),
            color_assist_mask=Image.fromarray(mask_debug["color_assist"], mode="L"),
            final_hair_mask=hair_mask,
            hair_mask_bbox=mask_debug["bbox"],
            hair_mask_area_pixels=int(mask_debug["area_pixels"]),
        )
    except FaceParsingRuntimeError:
        raise
    except Exception as exc:
        raise FaceParsingRuntimeError("Face Parsingの推論に失敗しました。モデルと入力画像を確認してください。") from exc
