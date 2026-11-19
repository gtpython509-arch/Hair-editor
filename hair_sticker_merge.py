"""Production module and CLI for auto-fitting hair sticker merging.

Uses MediaPipe ImageSegmenter + FaceLandmarker and OpenCV/NumPy geometry to:
1. Detect the subject's face landmarks (with head-zoom fallback).
2. Extract clean BGRA alpha from clipart, transparent PNGs, or checkerboard screenshots.
3. Locate the face opening in the hair sticker and auto-fit scale/rotation/position.
4. Optionally inpaint original hair, tint the sticker, match lighting, and protect the face.

CLI usage:
    python hair_sticker_merge.py --user photo.jpg --hair wig.png --out result.png
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

MODEL_URLS = {
    "selfie_multiclass_256x256.tflite": (
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
        "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"
    ),
}

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
    365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
    132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

ImageSource = Union[str, Path, bytes, bytearray]


@dataclass(frozen=True)
class MergeConfig:
    """Configuration options for `smart_merge`."""

    remove_hair: bool = True
    protect_face: bool = True
    face_fill: float = 1.00
    hair_thickness: float = 1.20
    scale_adjust: float = 1.00
    y_adjust: float = 0.00
    x_adjust: float = 0.00
    extra_rot: float = 0.00
    tint: Optional[str] = None
    tint_strength: float = 0.90
    follow_yaw: bool = True
    crown_lift: float = 0.25
    tone: float = 0.00
    inpaint_grow: int = 9
    inpaint_radius: int = 6
    debug: bool = False


class ModelBundle:
    """Lazy-loaded MediaPipe segmenter and face landmarkers."""

    def __init__(self, model_dir: Optional[Union[str, Path]] = None) -> None:
        default_dir = Path(os.environ.get("HAIR_EDITOR_MODEL_DIR", Path(__file__).resolve().parent / "models"))
        self.model_dir = Path(model_dir) if model_dir is not None else default_dir
        self._segmenter: Optional[vision.ImageSegmenter] = None
        self._face_hi: Optional[vision.FaceLandmarker] = None
        self._face_lo: Optional[vision.FaceLandmarker] = None

    def ensure_models(self) -> Tuple[Path, Path]:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for filename, url in MODEL_URLS.items():
            path = self.model_dir / filename
            if not path.exists():
                logger.info("Downloading MediaPipe model %s -> %s", filename, path)
                urllib.request.urlretrieve(url, str(path))
            paths[filename] = path
        return paths["selfie_multiclass_256x256.tflite"], paths["face_landmarker.task"]

    def _load(self) -> None:
        if self._segmenter is not None:
            return
        seg_path, face_path = self.ensure_models()
        self._segmenter = vision.ImageSegmenter.create_from_options(
            vision.ImageSegmenterOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(seg_path)),
                running_mode=vision.RunningMode.IMAGE,
                output_confidence_masks=True,
                output_category_mask=True,
            )
        )
        self._face_hi = self._make_landmarker(face_path, 0.5)
        self._face_lo = self._make_landmarker(face_path, 0.2)

    @staticmethod
    def _make_landmarker(model_path: Path, conf: float) -> vision.FaceLandmarker:
        return vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=conf,
                min_face_presence_confidence=conf,
            )
        )

    @property
    def segmenter(self) -> vision.ImageSegmenter:
        self._load()
        assert self._segmenter is not None
        return self._segmenter

    @property
    def face_hi(self) -> vision.FaceLandmarker:
        self._load()
        assert self._face_hi is not None
        return self._face_hi

    @property
    def face_lo(self) -> vision.FaceLandmarker:
        self._load()
        assert self._face_lo is not None
        return self._face_lo


_DEFAULT_MODELS: Optional[ModelBundle] = None


def get_models(model_dir: Optional[Union[str, Path]] = None) -> ModelBundle:
    """Return a shared `ModelBundle` instance (or a custom one if `model_dir` is set)."""
    global _DEFAULT_MODELS
    if model_dir is not None:
        return ModelBundle(model_dir=model_dir)
    if _DEFAULT_MODELS is None:
        _DEFAULT_MODELS = ModelBundle()
    return _DEFAULT_MODELS


# ------------------------------------------------------------------ Image I/O & Detection


def load_image(source: ImageSource, max_side: int = 1280) -> np.ndarray:
    """Load an image as EXIF-corrected BGR numpy array, optionally downscaled."""
    src: Union[io.BytesIO, str, Path] = (
        io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
    )
    pil = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    if max_side:
        pil.thumbnail((max_side, max_side), Image.LANCZOS)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _detect_landmarks(bgr: np.ndarray, detector: vision.FaceLandmarker) -> Optional[np.ndarray]:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.face_landmarks:
        return None
    h, w = bgr.shape[:2]
    return np.array([[p.x * w, p.y * h] for p in result.face_landmarks[0]], dtype=np.float32)


def person_mask(bgr: np.ndarray, models: Optional[ModelBundle] = None) -> np.ndarray:
    """Return foreground person probability mask in [0, 1]."""
    bundle = models or get_models()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = bundle.segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    h, w = bgr.shape[:2]
    bg = res.confidence_masks[0].numpy_view().astype(np.float32)
    if bg.shape != (h, w):
        bg = cv2.resize(bg, (w, h))
    return 1.0 - bg


def hair_mask(
    bgr: np.ndarray,
    box: Optional[Tuple[int, int, int, int]] = None,
    crop_size: int = 512,
    models: Optional[ModelBundle] = None,
) -> np.ndarray:
    """Return hair confidence mask in [0, 1]; refines detail inside `box` when provided."""
    bundle = models or get_models()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = bundle.segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    h, w = bgr.shape[:2]
    full = res.confidence_masks[1].numpy_view().astype(np.float32)
    if full.shape != (h, w):
        full = cv2.resize(full, (w, h))
    if box is None:
        return full

    x0, y0, x1, y1 = box
    if x1 - x0 < 16 or y1 - y0 < 16:
        return full

    crop = cv2.resize(bgr[y0:y1, x0:x1], (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rc = bundle.segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb))
    hm = rc.confidence_masks[1].numpy_view().astype(np.float32)
    back = np.zeros((h, w), dtype=np.float32)
    back[y0:y1, x0:x1] = cv2.resize(hm, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    return np.maximum(full, back)


def detect_face(bgr: np.ndarray, models: Optional[ModelBundle] = None) -> Optional[np.ndarray]:
    """Detect 478 face landmarks with low-confidence and head-zoom fallbacks."""
    bundle = models or get_models()
    for det, label in ((bundle.face_hi, "pass 1"), (bundle.face_lo, "pass 2 (low conf)")):
        lm = _detect_landmarks(bgr, det)
        if lm is not None:
            logger.debug("Face found (%s)", label)
            return lm

    pm = person_mask(bgr, models=bundle)
    ys, xs = np.where(pm > 0.5)
    if len(ys):
        h, w = bgr.shape[:2]
        top = ys.min()
        band = ys <= top + (ys.max() - top) * 0.45
        cx = int(xs[band].mean())
        cy = int(top + (ys.max() - top) * 0.16)
        side = int(max(80, (xs[band].max() - xs[band].min()) * 1.4))
        x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
        x1, y1 = min(w, cx + side // 2), min(h, cy + side // 2)
        if x1 - x0 > 32 and y1 - y0 > 32:
            crop = bgr[y0:y1, x0:x1]
            scale = 640.0 / max(crop.shape[:2])
            big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            for det, label in ((bundle.face_hi, "pass 3 (head zoom)"), (bundle.face_lo, "pass 4 (head zoom low)")):
                lm = _detect_landmarks(big, det)
                if lm is not None:
                    logger.debug("Face found (%s)", label)
                    return lm / scale + np.array([x0, y0], dtype=np.float32)

    logger.warning("No face found after 4 detection attempts")
    return None


# ------------------------------------------------------------------ Sticker Alpha & Cleanup


def _despill(rgba: np.ndarray) -> np.ndarray:
    """Remove white fringing from semi-transparent edge pixels."""
    out = rgba.copy()
    a = out[:, :, 3:4].astype(np.float32) / 255.0
    edge = ((a > 0.05) & (a < 0.95)).squeeze()
    if edge.any():
        bgr = out[:, :, :3].astype(np.float32)
        core = a.squeeze() >= 0.95
        if core.sum() > 20:
            mean_core = bgr[core].mean(0)
            bgr[edge] = bgr[edge] * 0.45 + mean_core * 0.55
            out[:, :, :3] = np.clip(bgr, 0, 255).astype(np.uint8)
    return out


def prepare_hair_rgba(
    style_raw: Optional[np.ndarray],
    white_thresh: int = 238,
    soft: bool = True,
    tol: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Convert clipart, transparent PNG, or checkerboard screenshot into clean BGRA."""
    if style_raw is None:
        return None
    if style_raw.ndim == 2:
        style_raw = cv2.cvtColor(style_raw, cv2.COLOR_GRAY2BGR)
    if style_raw.shape[2] == 4 and style_raw[:, :, 3].min() < 240:
        return _despill(style_raw)

    bgr = style_raw[:, :, :3]
    h, w = bgr.shape[:2]
    rim = np.concatenate([bgr[0], bgr[-1], bgr[:, 0], bgr[:, -1]], axis=0)
    rim_hsv = cv2.cvtColor(rim.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    light = (rim_hsv[:, 1] < 40) & (rim_hsv[:, 2] > 215)
    if light.mean() < 0.55:
        return None

    rim_v = rim_hsv[light, 2].astype(np.float32)
    spread = float(rim_v.max() - rim_v.min())

    if soft and spread <= 10:
        ffmask = np.zeros((h + 2, w + 2), np.uint8)
        tmp = bgr.copy()
        diff = (tol or 18,) * 3
        for seed in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
            cv2.floodFill(tmp, ffmask, seed, (0, 0, 0), diff, diff, flags=4 | (255 << 8))
        outside = ffmask[1:-1, 1:-1].astype(bool)
        mx = bgr.max(axis=2).astype(np.float32)
        mn = bgr.min(axis=2).astype(np.float32)
        whiteness = np.clip((mx - 200) / max(1.0, white_thresh - 200), 0, 1)
        whiteness *= np.clip(1.0 - (mx - mn) / 40.0, 0, 1)
        alpha = np.where(outside, 1.0 - whiteness, 1.0)
        alpha = np.where(outside & (whiteness >= 1.0), 0.0, alpha)
        alpha = cv2.GaussianBlur((alpha * 255).astype(np.uint8), (3, 3), 0)
    else:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        v_lo = float(np.percentile(rim_v, 5)) - 14
        cand = ((hsv[:, :, 1] < 40) & (hsv[:, :, 2] > v_lo)).astype(np.uint8)
        n, lab, cstats, _ = cv2.connectedComponentsWithStats(cand, 4)
        edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        outside = np.isin(lab, edge[edge != 0])

        v_chan = hsv[:, :, 2].astype(np.float32)
        texture = np.abs(v_chan - cv2.medianBlur(hsv[:, :, 2], 5).astype(np.float32))
        if outside.any():
            ref_v, ref_t = float(v_chan[outside].mean()), float(texture[outside].mean())
            for i in range(1, n):
                if cstats[i, cv2.CC_STAT_AREA] < 0.003 * h * w or np.isin(i, edge):
                    continue
                m = lab == i
                if abs(float(v_chan[m].mean()) - ref_v) < 12 and abs(float(texture[m].mean()) - ref_t) < 6:
                    outside |= m

        outside = cv2.morphologyEx(
            outside.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
        ).astype(bool)
        keep = cv2.erode((~outside).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
        alpha = cv2.GaussianBlur(keep * 255, (0, 0), 0.8)

    return _despill(np.dstack([bgr, alpha]))


def keep_main_object(rgba: np.ndarray, min_frac: float = 0.004, pad_frac: float = 0.02) -> np.ndarray:
    """Keep the largest hair component and overlapping strands; drop unrelated artifacts."""
    a = (rgba[:, :, 3] > 100).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(a, 8)
    if n <= 2:
        return rgba
    areas = stats[1:, cv2.CC_STAT_AREA]
    main = 1 + int(np.argmax(areas))
    h, w = a.shape
    pad = int(max(h, w) * pad_frac)
    mx, my = stats[main, cv2.CC_STAT_LEFT] - pad, stats[main, cv2.CC_STAT_TOP] - pad
    mx2 = stats[main, cv2.CC_STAT_LEFT] + stats[main, cv2.CC_STAT_WIDTH] + pad
    my2 = stats[main, cv2.CC_STAT_TOP] + stats[main, cv2.CC_STAT_HEIGHT] + pad

    keep = [main]
    for i in range(1, n):
        if i == main or stats[i, cv2.CC_STAT_AREA] < min_frac * areas.max():
            continue
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        x2, y2 = x + stats[i, cv2.CC_STAT_WIDTH], y + stats[i, cv2.CC_STAT_HEIGHT]
        if x < mx2 and x2 > mx and y < my2 and y2 > my:
            keep.append(i)
    out = rgba.copy()
    out[:, :, 3] = np.where(np.isin(lab, keep), out[:, :, 3], 0)
    return out


def tight_crop(rgba: np.ndarray, thr: int = 12) -> Optional[np.ndarray]:
    """Crop BGRA image tightly around non-transparent pixels."""
    ys, xs = np.where(rgba[:, :, 3] > thr)
    if len(xs) == 0:
        return None
    return rgba[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def load_sticker(source: ImageSource) -> np.ndarray:
    """Load a hair sticker from path or bytes and return tightly cropped BGRA."""
    if isinstance(source, (bytes, bytearray)):
        raw = cv2.imdecode(np.frombuffer(source, np.uint8), cv2.IMREAD_UNCHANGED)
    else:
        raw = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Could not read sticker image: {source}")
    st = prepare_hair_rgba(raw)
    if st is None:
        raise ValueError(
            "Sticker has no transparency and no light background; provide a cut-out PNG or art on white."
        )
    cropped = tight_crop(keep_main_object(st))
    if cropped is None:
        raise ValueError("Sticker is empty after background removal.")
    return cropped


# ------------------------------------------------------------------ Geometry & Fitting


def analyze_sticker(rgba: np.ndarray, alpha_thr: float = 0.4) -> Dict[str, Any]:
    """Locate the face opening and crown geometry inside a hair sticker."""
    a = rgba[:, :, 3] > int(alpha_thr * 255)
    h, w = a.shape
    left = np.full(h, -1, np.int32)
    right = np.full(h, -1, np.int32)
    gap = np.zeros(h, np.int32)
    gap_cx = np.zeros(h, np.float32)

    for y in range(h):
        xs = np.flatnonzero(a[y])
        if xs.size < 2:
            continue
        left[y], right[y] = xs[0], xs[-1]
        d = np.diff(xs)
        i = int(np.argmax(d))
        if d[i] > 1:
            gap[y] = int(d[i]) - 1
            gap_cx[y] = (xs[i] + xs[i + 1]) / 2.0

    width = np.where(left >= 0, right - left, 0)
    if width.max() <= 0:
        raise ValueError("Sticker is empty.")

    cand = np.flatnonzero((gap > 0.20 * w) & (np.arange(h) > 0.08 * h))
    has_hole = cand.size >= max(4, int(0.04 * h))

    if has_hole:
        hairline_y = int(cand[0])
        cap0 = float(width[: max(1, hairline_y)].max()) if hairline_y > 2 else float(width.max())
        span = max(24.0, 0.75 * (cap0 if cap0 > 0 else w))
        near = cand[(cand >= hairline_y) & (cand <= hairline_y + span)]
        if near.size < 3:
            near = cand[cand <= hairline_y + 0.35 * h]
        hole_width = float(np.percentile(gap[near], 85))
        anchor_x = float(np.median(gap_cx[near]))
    else:
        hairline_y = int(0.38 * h)
        hole_width = 0.0
        anchor_x = w / 2.0

    cap_end = hairline_y if (has_hole and hairline_y >= 0.18 * h) else int(0.50 * h)
    band = width[: max(1, cap_end)]
    cap_width = float(band.max()) if band.size and band.max() > 0 else float(width.max())
    if cap_width < hole_width:
        cap_width = float(width.max())

    rows = np.flatnonzero(band >= 0.97 * band.max()) if band.size and band.max() > 0 else np.array([0])
    cap_y = float(rows.mean())
    if not has_hole and left[int(cap_y)] >= 0:
        anchor_x = float((left[int(cap_y)] + right[int(cap_y)]) / 2.0)

    skull_w = float(width[int(hairline_y)]) if has_hole and width[int(hairline_y)] > 0 else cap_width
    band_top = width[: max(1, int(hairline_y))]
    rows_wide = np.flatnonzero(band_top >= 0.62 * skull_w)
    crown_y = float(rows_wide[0]) if rows_wide.size else 0.0

    return {
        "hairline_y": float(hairline_y),
        "cap_y": cap_y,
        "crown_y": crown_y,
        "skull_w": skull_w,
        "anchor_x": anchor_x,
        "cap_width": cap_width,
        "hole_width": hole_width,
        "has_hole": bool(has_hole),
        "size": (w, h),
    }


def head_metrics(lm: np.ndarray) -> Dict[str, Any]:
    """Compute skull width, height, roll, yaw, and key anchor points from landmarks."""
    left_temple, right_temple = lm[234], lm[454]
    top, chin, nose = lm[10], lm[152], lm[1]
    width = float(np.linalg.norm(right_temple - left_temple))
    height = float(np.linalg.norm(chin - top))
    cx = float((left_temple[0] + right_temple[0]) / 2.0)
    return {
        "width": width,
        "height": height,
        "cx": cx,
        "hairline": (cx, float(top[1])),
        "temple": (cx, float((left_temple[1] + right_temple[1]) / 2.0)),
        "angle": float(-np.degrees(np.arctan2(right_temple[1] - left_temple[1], right_temple[0] - left_temple[0]))),
        "yaw": float((nose[0] - cx) / max(1.0, width / 2.0)),
        "brow_y": float(min(lm[105][1], lm[334][1])),
        "eye_y": float(min(lm[159][1], lm[386][1])),
        "top": top,
        "chin": chin,
        "left": left_temple,
        "right": right_temple,
    }


def fit_params(
    hm: Dict[str, Any],
    sa: Dict[str, Any],
    cfg: MergeConfig,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """Compute `((scale_x, scale_y), anchor_in_sticker, target_in_photo)` from user head geometry."""
    if sa["has_hole"] and sa["hole_width"] > 0.25 * sa["cap_width"]:
        scale_x = (hm["width"] * cfg.face_fill) / max(1.0, sa["hole_width"])
        lo = (hm["width"] * 1.05) / max(1.0, sa["skull_w"])
        hi = (hm["width"] * 2.40) / max(1.0, sa["skull_w"])
        scale_x = float(np.clip(scale_x, lo, hi))
    else:
        scale_x = (hm["width"] * cfg.hair_thickness) / max(1.0, sa["cap_width"])

    # Adapt vertical scale to the user's face height/width aspect ratio (normalized around 1.25)
    user_aspect = hm["height"] / max(1.0, hm["width"])
    scale_y = scale_x * float(np.clip(user_aspect / 1.25, 0.85, 1.20))

    scale_x *= cfg.scale_adjust
    scale_y *= cfg.scale_adjust

    anchor = (sa["anchor_x"], sa["crown_y"])
    ty = hm["hairline"][1] - cfg.crown_lift * hm["height"]
    drop = (sa["hairline_y"] - sa["crown_y"]) * scale_y
    ty = max(ty, hm["hairline"][1] - drop)

    tx = hm["cx"] + hm["width"] * cfg.x_adjust
    if cfg.follow_yaw:
        tx += hm["yaw"] * hm["width"] * 0.12
    return (float(scale_x), float(scale_y)), anchor, (float(tx), float(ty + hm["height"] * cfg.y_adjust))


def place_rgba(
    rgba: np.ndarray,
    anchor: Tuple[float, float],
    target: Tuple[float, float],
    scale: Union[float, Tuple[float, float]],
    angle_deg: float,
    out_hw: Tuple[int, int],
) -> np.ndarray:
    """Warp BGRA sticker onto the target canvas around `anchor -> target`."""
    sx, sy = scale if isinstance(scale, tuple) else (scale, scale)
    r = np.deg2rad(angle_deg)
    c, s = np.cos(r), np.sin(r)
    ax, ay = anchor
    tx, ty = target
    m = np.float32([
        [c * sx, s * sy, tx - (c * sx * ax + s * sy * ay)],
        [-s * sx, c * sy, ty - (-s * sx * ax + c * sy * ay)],
    ])
    return cv2.warpAffine(
        rgba,
        m,
        (out_hw[1], out_hw[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def alpha_over(base_bgr: np.ndarray, warped_rgba: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = warped_rgba[:, :, 3:4].astype(np.float32) / 255.0
    out = base_bgr.astype(np.float32) * (1.0 - a) + warped_rgba[:, :, :3].astype(np.float32) * a
    return out.astype(np.uint8), a[:, :, 0]


def face_front_mask(
    lm: np.ndarray,
    shape: Tuple[int, ...],
    cut_y: Optional[float] = None,
    shrink: float = 0.97,
    feather: int = 3,
) -> np.ndarray:
    pts = lm[FACE_OVAL].astype(np.float32)
    c = pts.mean(0)
    pts = (c + (pts - c) * shrink).astype(np.int32)
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(pts), 255)
    if cut_y is not None:
        m[: max(0, int(cut_y)), :] = 0
    if feather > 0:
        m = cv2.GaussianBlur(m, (0, 0), feather)
    return m.astype(np.float32) / 255.0


def head_box(
    hm: Dict[str, Any],
    shape: Tuple[int, ...],
    grow: float = 1.9,
    up: float = 0.9,
) -> Tuple[int, int, int, int]:
    h, w = shape[:2]
    cx, cy = hm["cx"], (hm["hairline"][1] + hm["chin"][1]) / 2.0
    half = hm["width"] * grow / 2.0
    return (
        int(max(0, cx - half)),
        int(max(0, cy - hm["height"] * up)),
        int(min(w, cx + half)),
        int(min(h, cy + hm["height"] * 1.1)),
    )


# ------------------------------------------------------------------ Effects & Merge


def tint_sticker(rgba: np.ndarray, hex_color: str, strength: float = 0.9) -> np.ndarray:
    hx = hex_color.lstrip("#")
    if len(hx) != 6:
        raise ValueError(f"Expected 6-digit hex color, got {hex_color!r}")
    tgt_bgr = np.uint8([[[int(hx[4:6], 16), int(hx[2:4], 16), int(hx[0:2], 16)]]])
    tgt = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    lab = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_BGR2LAB).astype(np.float32)
    w = (rgba[:, :, 3].astype(np.float32) / 255.0) * strength
    lab[..., 1] = lab[..., 1] * (1.0 - w) + tgt[1] * w
    lab[..., 2] = lab[..., 2] * (1.0 - w) + tgt[2] * w
    out = rgba.copy()
    out[:, :, :3] = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out


def remove_original_hair(
    bgr: np.ndarray,
    hair01: np.ndarray,
    keep01: Optional[np.ndarray] = None,
    box: Optional[Tuple[int, int, int, int]] = None,
    grow: int = 9,
    radius: int = 6,
) -> np.ndarray:
    m = (hair01 > 0.45).astype(np.uint8)
    if keep01 is not None:
        m[keep01 > 0.35] = 0
    if box is not None:
        keep = np.zeros_like(m)
        keep[box[1] : box[3], box[0] : box[2]] = 1
        m *= keep
    if m.sum() == 0:
        return bgr
    k = np.ones((max(1, grow) | 1,) * 2, np.uint8)
    return cv2.inpaint(bgr, cv2.dilate(m, k, 1), radius, cv2.INPAINT_TELEA)


def match_lighting(
    rgba: np.ndarray,
    photo_bgr: np.ndarray,
    box: Tuple[int, int, int, int],
    strength: float = 0.5,
) -> np.ndarray:
    opaque = rgba[:, :, 3] > 200
    if opaque.sum() < 50 or strength <= 0:
        return rgba
    ref = photo_bgr[box[1] : box[3], box[0] : box[2]]
    if ref.size == 0:
        return rgba
    lab = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    s_hi = float(np.percentile(lab[:, :, 0][opaque], 85))
    r_hi = float(np.percentile(ref_lab[:, :, 0], 85))
    gain = 1.0 + strength * ((r_hi / max(1.0, s_hi)) - 1.0)
    lab[:, :, 0] = np.clip(lab[:, :, 0] * gain, 0, 255)
    out = rgba.copy()
    out[:, :, :3] = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out


def debug_view(bgr: np.ndarray, hm: Dict[str, Any], box: Tuple[int, int, int, int]) -> np.ndarray:
    d = bgr.copy()
    cv2.rectangle(d, box[:2], box[2:], (90, 90, 90), 1)
    cv2.line(d, tuple(np.int32(hm["left"])), tuple(np.int32(hm["right"])), (0, 200, 255), 2)
    cv2.circle(d, (int(hm["temple"][0]), int(hm["temple"][1])), 4, (0, 0, 255), -1)
    cv2.circle(d, (int(hm["hairline"][0]), int(hm["hairline"][1])), 4, (0, 255, 0), -1)
    return d


def smart_merge(
    user_bgr: np.ndarray,
    style_rgba: np.ndarray,
    config: Optional[MergeConfig] = None,
    models: Optional[ModelBundle] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Auto-fit and composite `style_rgba` onto `user_bgr`."""
    cfg = config or MergeConfig()
    bundle = models or get_models()
    info: Dict[str, Any] = {}

    lm = detect_face(user_bgr, models=bundle)
    if lm is None:
        return user_bgr, {"error": "no face detected"}

    hm = head_metrics(lm)
    st = tight_crop(keep_main_object(style_rgba))
    if st is None:
        return user_bgr, {"error": "sticker is empty"}
    if cfg.tint:
        st = tint_sticker(st, cfg.tint, strength=cfg.tint_strength)

    box = head_box(hm, user_bgr.shape)
    if cfg.tone:
        st = match_lighting(st, user_bgr, box, strength=cfg.tone)

    sa = analyze_sticker(st)
    (scale_x, scale_y), anchor, target = fit_params(hm, sa, cfg)
    warped = place_rgba(
        st, anchor, target, (scale_x, scale_y), hm["angle"] + cfg.extra_rot, user_bgr.shape[:2]
    )
    cover = warped[:, :, 3].astype(np.float32) / 255.0

    base = user_bgr
    if cfg.remove_hair:
        hmask = hair_mask(user_bgr, box=box, models=bundle)
        base = remove_original_hair(
            user_bgr,
            hmask,
            keep01=cover,
            box=box,
            grow=cfg.inpaint_grow,
            radius=cfg.inpaint_radius,
        )
        info["own_hair_px"] = int((hmask > 0.45).sum())

    out, _ = alpha_over(base, warped)

    if cfg.protect_face:
        fm = face_front_mask(lm, out.shape, cut_y=hm["eye_y"] - 0.03 * hm["height"])[:, :, None]
        out = (out.astype(np.float32) * (1.0 - fm) + user_bgr.astype(np.float32) * fm).astype(np.uint8)

    info.update(
        head_width=round(hm["width"]),
        head_height=round(hm["height"]),
        roll=round(hm["angle"], 1),
        yaw=round(hm["yaw"], 2),
        scale_x=round(scale_x, 3),
        scale_y=round(scale_y, 3),
        opening_px=round(sa["hole_width"] * scale_x),
        outer_px=round(sa["cap_width"] * scale_x),
        found_opening=sa["has_hole"],
        covered_px=int((cover > 0.5).sum()),
    )
    if not sa["has_hole"]:
        info["warn"] = "no face opening found — fitted from proportions"
    if abs(hm["yaw"]) > 0.35:
        info["warn"] = "head turned too far for a flat sticker"
    if cfg.debug:
        info["debug"] = debug_view(user_bgr, hm, box)
    return out, info


def merge_files(
    user_path: Union[str, Path],
    hair_path: Union[str, Path],
    out_path: Union[str, Path],
    *,
    config: Optional[MergeConfig] = None,
    debug_out: Optional[Union[str, Path]] = None,
    model_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """High-level file-to-file entry point."""
    bundle = get_models(model_dir=model_dir)
    user_bgr = load_image(user_path)
    style_rgba = load_sticker(hair_path)
    result_bgr, info = smart_merge(user_bgr, style_rgba, config=config, models=bundle)
    if "error" in info:
        raise RuntimeError(info["error"])

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_file), result_bgr):
        raise OSError(f"Failed to write output image: {out_file}")

    debug_img = info.pop("debug", None)
    if debug_out is not None and debug_img is not None:
        dbg_file = Path(debug_out)
        dbg_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dbg_file), debug_img)

    return info


# ------------------------------------------------------------------ Localhost Test UI

_HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Hair Sticker Merge — Local Test UI</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 1100px; margin: 24px auto; padding: 0 16px; background: #111827; color: #f3f4f6; }
    .card { background: #1f2937; border-radius: 10px; padding: 16px; margin-bottom: 16px; }
    .row { display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-end; }
    label { display: flex; flex-direction: column; gap: 6px; font-size: 13px; color: #d1d5db; }
    input, button { padding: 8px 10px; border-radius: 6px; border: 1px solid #374151; background: #111827; color: #f9fafb; }
    button { background: #2563eb; border-color: #2563eb; cursor: pointer; font-weight: 600; padding: 10px 18px; }
    button:disabled { opacity: 0.6; cursor: wait; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
    img { width: 100%; border-radius: 8px; background: #030712; }
    pre { background: #030712; padding: 12px; border-radius: 8px; overflow: auto; font-size: 12px; }
  </style>
</head>
<body>
  <h2>Hair Sticker Merge — Localhost Test UI</h2>
  <div class="card">
    <div class="row">
      <label>User Photo <input id="userFile" type="file" accept="image/*" /></label>
      <label>Hair Sticker <input id="hairFile" type="file" accept="image/*" /></label>
      <label>Tint Hex (optional) <input id="tint" type="text" placeholder="#1E3FCF" style="width:110px" /></label>
      <label>Tone (0-1) <input id="tone" type="number" min="0" max="1" step="0.1" value="0" style="width:80px" /></label>
      <label>Scale <input id="scale" type="number" min="0.5" max="2" step="0.05" value="1" style="width:80px" /></label>
      <label style="flex-direction:row;align-items:center"><input id="removeHair" type="checkbox" checked /> Remove Hair</label>
      <label style="flex-direction:row;align-items:center"><input id="protectFace" type="checkbox" checked /> Protect Face</label>
      <button id="runBtn">Merge Hair</button>
    </div>
    <p id="status" style="margin:10px 0 0;font-size:13px;color:#9ca3af"></p>
  </div>
  <div class="grid">
    <div class="card"><h4>Result</h4><img id="resImg" /></div>
    <div class="card"><h4>Debug Geometry</h4><img id="dbgImg" /></div>
  </div>
  <div class="card"><h4>Metrics</h4><pre id="metrics">{}</pre></div>
  <script>
    const toB64 = f => new Promise((res, rej) => {
      const r = new FileReader();
      r.onload = () => res(r.result.split(',')[1]);
      r.onerror = rej;
      r.readAsDataURL(f);
    });
    document.getElementById('runBtn').onclick = async () => {
      const uf = document.getElementById('userFile').files[0];
      const hf = document.getElementById('hairFile').files[0];
      const st = document.getElementById('status');
      if (!uf || !hf) { st.textContent = 'Select both User Photo and Hair Sticker.'; return; }
      const btn = document.getElementById('runBtn');
      btn.disabled = true; st.textContent = 'Processing...';
      try {
        const payload = {
          user_b64: await toB64(uf),
          hair_b64: await toB64(hf),
          tint: document.getElementById('tint').value.trim() || null,
          tone: parseFloat(document.getElementById('tone').value) || 0,
          scale_adjust: parseFloat(document.getElementById('scale').value) || 1,
          remove_hair: document.getElementById('removeHair').checked,
          protect_face: document.getElementById('protectFace').checked,
        };
        const resp = await fetch('/merge', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || 'Merge failed');
        document.getElementById('resImg').src = 'data:image/png;base64,' + data.result_b64;
        document.getElementById('dbgImg').src = 'data:image/png;base64,' + data.debug_b64;
        document.getElementById('metrics').textContent = JSON.stringify(data.info, null, 2);
        st.textContent = 'Done.';
      } catch (e) {
        st.textContent = 'Error: ' + e.message;
      } finally { btn.disabled = false; }
    };
  </script>
</body>
</html>
"""


def serve_ui(
    host: str = "127.0.0.1",
    port: int = 8000,
    model_dir: Optional[Union[str, Path]] = None,
) -> None:
    """Start a lightweight localhost HTTP UI for testing `smart_merge`."""
    bundle = get_models(model_dir=model_dir)
    bundle.ensure_models()

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, code: int, obj: Dict[str, Any]) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            body = _HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/merge":
                self._send_json(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(length).decode("utf-8"))
                user_bgr = load_image(base64.b64decode(req["user_b64"]))
                style_rgba = load_sticker(base64.b64decode(req["hair_b64"]))
                cfg = MergeConfig(
                    remove_hair=bool(req.get("remove_hair", True)),
                    protect_face=bool(req.get("protect_face", True)),
                    scale_adjust=float(req.get("scale_adjust", 1.0)),
                    tint=req.get("tint") or None,
                    tone=float(req.get("tone", 0.0)),
                    debug=True,
                )
                result_bgr, info = smart_merge(user_bgr, style_rgba, config=cfg, models=bundle)
                if "error" in info:
                    self._send_json(400, {"error": info["error"]})
                    return
                dbg = info.pop("debug", result_bgr)
                _, res_png = cv2.imencode(".png", result_bgr)
                _, dbg_png = cv2.imencode(".png", dbg)
                self._send_json(
                    200,
                    {
                        "result_b64": base64.b64encode(res_png.tobytes()).decode("ascii"),
                        "debug_b64": base64.b64encode(dbg_png.tobytes()).decode("ascii"),
                        "info": info,
                    },
                )
            except Exception as exc:
                logger.exception("Merge request failed")
                self._send_json(500, {"error": str(exc)})

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Hair Sticker Merge UI running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


# ------------------------------------------------------------------ CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Auto-fit and merge a hair sticker onto a portrait photo.")
    ap.add_argument("--user", help="Path to user portrait photo")
    ap.add_argument("--hair", help="Path to hair sticker PNG/image")
    ap.add_argument("--out", default="result.png", help="Output image path")
    ap.add_argument("--serve", action="store_true", help="Launch localhost web UI for testing")
    ap.add_argument("--host", default="127.0.0.1", help="Host for --serve (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8000, help="Port for --serve (default: 8000)")
    ap.add_argument("--debug-out", default=None, help="Optional path to save debug overlay")
    ap.add_argument("--model-dir", default=None, help="Directory for MediaPipe model files")
    ap.add_argument("--tint", default=None, help="Optional hex tint colour, e.g. '#1E3FCF'")
    ap.add_argument("--tone", type=float, default=0.0, help="Lighting match strength [0..1]")
    ap.add_argument("--face-fill", type=float, default=1.0, help="Face opening fill multiplier")
    ap.add_argument("--scale-adjust", type=float, default=1.0, help="Extra scale multiplier")
    ap.add_argument("--x-adjust", type=float, default=0.0, help="Horizontal offset (fraction of head width)")
    ap.add_argument("--y-adjust", type=float, default=0.0, help="Vertical offset (fraction of head height)")
    ap.add_argument("--extra-rot", type=float, default=0.0, help="Extra rotation in degrees")
    ap.add_argument("--no-remove-hair", action="store_true", help="Do not inpaint original hair")
    ap.add_argument("--no-protect-face", action="store_true", help="Do not paste face mask on top")
    ap.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.serve or not (args.user and args.hair):
        serve_ui(host=args.host, port=args.port, model_dir=args.model_dir)
        return 0
    cfg = MergeConfig(
        remove_hair=not args.no_remove_hair,
        protect_face=not args.no_protect_face,
        face_fill=args.face_fill,
        scale_adjust=args.scale_adjust,
        x_adjust=args.x_adjust,
        y_adjust=args.y_adjust,
        extra_rot=args.extra_rot,
        tint=args.tint,
        tone=args.tone,
        debug=args.debug_out is not None,
    )
    info = merge_files(
        args.user,
        args.hair,
        args.out,
        config=cfg,
        debug_out=args.debug_out,
        model_dir=args.model_dir,
    )
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
