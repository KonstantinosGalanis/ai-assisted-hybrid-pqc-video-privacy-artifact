# Public paper reference: replace file placeholders before running.
# See README.md for dummy commands and required inputs.
# Algorithm constants are retained; placeholders are not experimental settings.

# chaos.py
# Standalone merged Chaos ROI encryption pipeline.
# Generated from framework_faster_optimized.py + framework_proxy_payload_matched_optimized.py.
# Preview mode logic was removed: public video/audio output is always chaos-ciphered.

import json
import base64
import hashlib
import hmac
import os
from fractions import Fraction
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from numba import njit

import time
import zlib
import av
import cv2
import numpy as np
from tqdm import tqdm
import torch
from ultralytics import YOLO

# -----------------------------
# Config
# -----------------------------

# If too slow: yolov8l-seg.pt or yolov8m-seg.pt
COCO_MODEL = os.environ.get("COCO_MODEL", "models/REPLACE_WITH_SEGMENTATION_MODEL.pt")

# COCO ids: person=0, car=2, motorbike=3, bus=5, truck=7
COCO_CLASSES_OF_INTEREST = {0, 2, 3, 5, 7}
COCO_CLASSES_LIST = sorted(COCO_CLASSES_OF_INTEREST)

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)

def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}

CONF_THRES = 0.10
IOU_THRES = 0.50

ROI_PAD = 6

# Keep small to avoid "fat" silhouettes
MASK_DILATE_PX = 1  # try 0..2

DETERMINISTIC_ROI_SORT = True

# Tracking (Ultralytics)
TRACKER_CFG = os.environ.get("TRACKER_CFG", "config/REPLACE_WITH_TRACKER_CONFIG.yaml")  # botsort.yaml / bytetrack.yaml
TRACK_PERSIST = True
RETINA_MASKS = _env_bool("RETINA_MASKS", True)

# Policy keying
POLICY_SCOPE = "class"  # "class" or "object"


# Extra debug prints
ENABLE_DEBUG_PRINTS = False
DEBUG_EVERY_N_FRAMES = 1            # print every frame; change to 10/30 if too verbose
DEBUG_ONLY_CLASSES = None           # e.g. {2,3,5,7} for vehicles only, or None for all
DEBUG_ONLY_TRACK_IDS = None         # e.g. {5,7}, or None
DEBUG_PRINT_RAW_DETECTIONS = False
DEBUG_PRINT_CANDIDATE_FILTERS = False
DEBUG_PRINT_MASK_STATS = False
DEBUG_PRINT_SUMMARY = False
DEBUG_PRINT_SMALL_VEHICLES = False

# Second debug mode: compare track() vs predict() on same frame
DEBUG_COMPARE_TRACK_VS_PREDICT = False
DEBUG_COMPARE_CLASSES = list(COCO_CLASSES_OF_INTEREST)
DEBUG_COMPARE_CONF = CONF_THRES
DEBUG_COMPARE_IOU = IOU_THRES

# -----------------------------
# Device selection
# -----------------------------
CUDA_OK = torch.cuda.is_available() and torch.cuda.device_count() > 0
DEVICE = "cuda" if CUDA_OK else "cpu"
YOLO_DEVICE = 0 if CUDA_OK else "cpu"
YOLO_HALF = True if CUDA_OK else False
TORCH_NONBLOCKING = True
if CUDA_OK:
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# -----------------------------
# Precomputed kernels (speed, no behavior change)
# -----------------------------
KERNEL_3 = np.ones((3, 3), np.uint8)
_KERNEL_CACHE: Dict[int, np.ndarray] = {}

def _get_kernel(k: int) -> np.ndarray:
    kk = int(k)
    ker = _KERNEL_CACHE.get(kk)
    if ker is None:
        ker = np.ones((kk, kk), np.uint8)
        _KERNEL_CACHE[kk] = ker
    return ker

# -----------------------------
# Debug helpers
# -----------------------------

def _dbg_enabled_for_frame(frame_idx: int) -> bool:
    if not ENABLE_DEBUG_PRINTS:
        return False
    if DEBUG_EVERY_N_FRAMES <= 1:
        return True
    return (frame_idx % DEBUG_EVERY_N_FRAMES) == 0

def _dbg_match_roi(cls_id: Optional[int] = None, track_id: Optional[int] = None) -> bool:
    if DEBUG_ONLY_CLASSES is not None and cls_id is not None and cls_id not in DEBUG_ONLY_CLASSES:
        return False
    if DEBUG_ONLY_TRACK_IDS is not None and track_id is not None and track_id not in DEBUG_ONLY_TRACK_IDS:
        return False
    return True

def _dbg(frame_idx: int, msg: str, *, cls_id: Optional[int] = None, track_id: Optional[int] = None) -> None:
    if not _dbg_enabled_for_frame(frame_idx):
        return
    if not _dbg_match_roi(cls_id=cls_id, track_id=track_id):
        return
    print(msg)

# -----------------------------
# Utilities
# -----------------------------

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def pad_bbox_exclusive(x1, y1, x2, y2, pad, w, h):
    """
    bbox is [x1,y1,x2,y2] where x2,y2 are EXCLUSIVE slice ends.
    """
    x1 = clamp(int(x1) - pad, 0, w - 1)
    y1 = clamp(int(y1) - pad, 0, h - 1)
    x2 = clamp(int(x2) + pad, 1, w)
    y2 = clamp(int(y2) + pad, 1, h)
    return x1, y1, x2, y2

def sha256_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

def policy_tag_for_class(cls_id: int) -> str:
    if cls_id == 0:
        return "person_sensitive"
    if cls_id in (2, 3, 5, 7):
        return "vehicle_sensitive"
    return "default_sensitive"

def derive_subkey(master_key: str, policy_tag: str, cls_id: int, track_id: int = -1, scope: str = "class") -> bytes:
    """
    scope:
      - 'class': same subkey for all ROIs of a class/policy
      - 'object': different subkey per tracked object
    """
    if scope == "class":
        payload = f"subkey|policy={policy_tag}|cls={int(cls_id)}".encode("utf-8")
    elif scope == "object":
        payload = f"subkey|policy={policy_tag}|cls={int(cls_id)}|tid={int(track_id)}".encode("utf-8")
    else:
        raise ValueError(scope)

    return hmac.new(master_key.encode("utf-8"), payload, hashlib.sha256).digest()

def derive_policy_seed(
    master_key: str,
    frame_idx: int,
    track_id: int,
    cls_id: int,
    map_name: str,
    policy_tag: str,
    scope: str = "class",
    bbox: Optional[Tuple[int, int, int, int]] = None,
    mask_hash: Optional[str] = None,
    roi_index: int = -1,
) -> bytes:
    subkey = derive_subkey(master_key, policy_tag, cls_id, track_id, scope=scope)

    payload_parts = [
        f"seed|frame={int(frame_idx)}",
        f"tid={int(track_id)}",
        f"cls={int(cls_id)}",
        f"map={map_name}",
        f"policy={policy_tag}",
        f"roi={int(roi_index)}",
    ]
    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        payload_parts.append(f"bbox={x1},{y1},{x2},{y2}")
    if mask_hash:
        payload_parts.append(f"mask={mask_hash}")

    payload = "|".join(payload_parts).encode("utf-8")
    return hmac.new(subkey, payload, hashlib.sha256).digest()

# Backward-compat wrapper if needed elsewhere
def derive_seed(
    master_key: str,
    frame_idx: int,
    track_id: int,
    cls_id: int,
    map_name: str,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    mask_hash: Optional[str] = None,
    roi_index: int = -1,
) -> bytes:
    policy_tag = policy_tag_for_class(cls_id)
    return derive_policy_seed(
        master_key=master_key,
        frame_idx=frame_idx,
        track_id=track_id,
        cls_id=cls_id,
        map_name=map_name,
        policy_tag=policy_tag,
        scope=POLICY_SCOPE,
        bbox=bbox,
        mask_hash=mask_hash,
        roi_index=roi_index,
    )

def _seed_to_unit(seed: bytes, k: int) -> float:
    chunk = seed[k * 8:(k + 1) * 8]
    v = int.from_bytes(chunk, "big", signed=False)
    return (v % (10**12) + 1) / (10**12 + 2)

# -----------------------------
# Chaotic keystream
# -----------------------------

@njit(cache=False, fastmath=True)
def _chen_keystream_u8_jit(n: int, burn: int, x: float, y: float, z: float,
                           A: float, B: float, C: float, M: float) -> np.ndarray:
    out = np.empty(n, dtype=np.uint8)
    t = 0
    total = burn + n
    for i in range(total):
        x = (A * (y - x)) % M
        y = (((C - A) * x) - (x * z) + (C * y)) % M
        z = ((x * y) - (B * z)) % M
        if i >= burn:
            v = int((x / M) * 256.0) & 255
            out[t] = v
            t += 1
    return out

def chen_keystream_u8(seed: bytes, n: int, burn: int = 64,
                      A: float = 35.0, B: float = 3.0, C: float = 21.0, M: float = 0.99) -> np.ndarray:
    x = _seed_to_unit(seed, 0) * M
    y = _seed_to_unit(seed, 1) * M
    z = _seed_to_unit(seed, 2) * M
    return _chen_keystream_u8_jit(n, burn, x, y, z, A, B, C, M)

@njit(cache=False, fastmath=True)
def _cubic_keystream_u8_jit(n: int, burn: int, x: float, u: float) -> np.ndarray:
    out = np.empty(n, dtype=np.uint8)
    t = 0
    total = burn + n
    for i in range(total):
        x = (u * x * (1.0 - x * x)) % 1.0
        if i >= burn:
            out[t] = int(x * 256.0) & 255
            t += 1
    return out

def cubic_keystream_u8(seed: bytes, n: int, burn: int = 64, u: float = 2.54) -> np.ndarray:
    x = _seed_to_unit(seed, 0)
    return _cubic_keystream_u8_jit(n, burn, x, u)

@njit(cache=False, fastmath=True)
def _skew_tent_keystream_u8_jit(n: int, burn: int, x: float, p: float) -> np.ndarray:
    out = np.empty(n, dtype=np.uint8)
    t = 0
    total = burn + n
    for i in range(total):
        if x < p:
            x = x / p
        else:
            x = (1.0 - x) / (1.0 - p)
        x = x % 1.0
        if i >= burn:
            out[t] = int(x * 256.0) & 255
            t += 1
    return out

def skew_tent_keystream_u8(seed: bytes, n: int, burn: int = 64, p: float = 0.426) -> np.ndarray:
    x = _seed_to_unit(seed, 0)
    return _skew_tent_keystream_u8_jit(n, burn, x, p)

def keystream_u8(seed: bytes, map_name: str, n: int) -> np.ndarray:
    if map_name == "chen":
        return chen_keystream_u8(seed, n)
    if map_name == "cubic":
        return cubic_keystream_u8(seed, n)
    if map_name == "skew_tent":
        return skew_tent_keystream_u8(seed, n)
    raise ValueError(map_name)


def _compute_mask_hash(mask_bool: np.ndarray) -> str:
    packed = np.packbits(np.asarray(mask_bool, dtype=np.uint8).reshape(-1))
    return hashlib.blake2b(packed.tobytes(), digest_size=8).hexdigest()

def derive_stream_seed(seed: bytes, label: str) -> bytes:
    return hmac.new(seed, f"stream|{label}".encode("utf-8"), hashlib.sha256).digest()

def _bind_roi_seed(
    seed: bytes,
    frame_idx: int = 0,
    track_id: int = -1,
    cls_id: int = -1,
    policy_tag: str = "default_sensitive",
    bbox: Optional[Tuple[int, int, int, int]] = None,
    mask_hash: str = "",
    roi_index: int = 0,
) -> bytes:
    parts = [
        "chaos_psd_v1",
        f"frame={int(frame_idx)}",
        f"track={int(track_id)}",
        f"cls={int(cls_id)}",
        f"policy={policy_tag}",
        f"roi={int(roi_index)}",
    ]
    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        parts.append(f"bbox={x1},{y1},{x2},{y2}")
    if mask_hash:
        parts.append(f"mask={mask_hash}")
    return hmac.new(seed, "|".join(parts).encode("utf-8"), hashlib.sha256).digest()

_CHAOS_NONCE_COUNTER = 0
_CHAOS_NONCE_SESSION = os.urandom(32)

def _next_chaos_nonce(seed: bytes, map_name: str, label: str, n: int = 16) -> bytes:
    global _CHAOS_NONCE_COUNTER
    _CHAOS_NONCE_COUNTER = (_CHAOS_NONCE_COUNTER + 1) & ((1 << 64) - 1)
    nonce_material = b"|".join([
        f"nonce|{label}|ctr={_CHAOS_NONCE_COUNTER}".encode("utf-8"),
        _CHAOS_NONCE_SESSION,
    ])
    nseed = hmac.new(seed, nonce_material, hashlib.sha256).digest()
    return np.asarray(keystream_u8(nseed, map_name, n), dtype=np.uint8).tobytes()

def _b64e(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")

def _b64d(text_b64: str) -> bytes:
    return base64.b64decode(text_b64.encode("ascii"))

def _prepare_chaos_crypto_meta(
    seed: bytes,
    map_name: str,
    chunk_size: int = 1024,
    block_size: int = 1024,
    frame_idx: int = 0,
    track_id: int = -1,
    cls_id: int = -1,
    policy_tag: str = "default_sensitive",
    bbox: Optional[Tuple[int, int, int, int]] = None,
    mask_hash: str = "",
    roi_index: int = 0,
) -> dict:
    core_seed = _bind_roi_seed(
        seed,
        frame_idx=frame_idx,
        track_id=track_id,
        cls_id=cls_id,
        policy_tag=policy_tag,
        bbox=bbox,
        mask_hash=mask_hash,
        roi_index=roi_index,
    )
    nonce_a = _next_chaos_nonce(core_seed, map_name, "stream", n=16)
    nonce_b = _next_chaos_nonce(core_seed, map_name, "aux", n=16)
    return {
        "scheme": "chaos_stream_v1",
        "stream_nonce_b64": _b64e(nonce_a),
        "tags_b64": _b64e(nonce_b),
        "chunk_size": int(max(256, chunk_size)),
        "block_size": int(max(128, block_size)),
    }


def _chaos_stream_transform(
    data_u8: np.ndarray,
    seed: bytes,
    map_name: str,
    nonce_a: bytes,
    nonce_b: bytes,
) -> np.ndarray:
    n = int(data_u8.size)
    if n <= 0:
        return np.asarray(data_u8, dtype=np.uint8).copy()
    base_arr = np.asarray(data_u8, dtype=np.uint8)
    stream_seed = hmac.new(seed, bytes(nonce_a) + bytes(nonce_b), hashlib.sha256).digest()
    ks_seed = derive_stream_seed(stream_seed, "masked-xor")
    ks = np.asarray(keystream_u8(ks_seed, map_name, n), dtype=np.uint8)
    return np.bitwise_xor(base_arr, ks).astype(np.uint8, copy=False)


def _chaos_stream_bytes(
    seed: bytes,
    map_name: str,
    nonce_a: bytes,
    nonce_b: bytes,
    n: int,
) -> np.ndarray:
    n = int(n)
    if n <= 0:
        return np.empty(0, dtype=np.uint8)
    stream_seed = hmac.new(seed, bytes(nonce_a) + bytes(nonce_b), hashlib.sha256).digest()
    ks_seed = derive_stream_seed(stream_seed, "masked-xor")
    return np.asarray(keystream_u8(ks_seed, map_name, n), dtype=np.uint8)


def _encrypt_masked_bytes_cpu_chaos(
    roi_bgr: np.ndarray,
    mask: np.ndarray,
    seed: bytes,
    map_name: str,
    frame_idx: int = 0,
    track_id: int = -1,
    cls_id: int = -1,
    policy_tag: str = "default_sensitive",
    bbox: Optional[Tuple[int, int, int, int]] = None,
    mask_hash: str = "",
    roi_index: int = 0,
    crypto_meta: Optional[dict] = None,
):
    if roi_bgr.size == 0 or mask is None or mask.size == 0 or not np.any(mask):
        return roi_bgr, crypto_meta if crypto_meta is not None else {
            "scheme": "chaos_stream_v1",
            "stream_nonce_b64": "",
            "tags_b64": "",
            "chunk_size": 1024,
            "block_size": 1024,
        }

    if roi_bgr.dtype != np.uint8:
        roi_bgr = roi_bgr.astype(np.uint8, copy=False)
    if mask.dtype != np.bool_:
        mask = mask.astype(np.bool_, copy=False)

    core_seed = _bind_roi_seed(
        seed,
        frame_idx=frame_idx,
        track_id=track_id,
        cls_id=cls_id,
        policy_tag=policy_tag,
        bbox=bbox,
        mask_hash=mask_hash,
        roi_index=roi_index,
    )
    if crypto_meta is None:
        crypto_meta = _prepare_chaos_crypto_meta(
            seed,
            map_name,
            frame_idx=frame_idx,
            track_id=track_id,
            cls_id=cls_id,
            policy_tag=policy_tag,
            bbox=bbox,
            mask_hash=mask_hash,
            roi_index=roi_index,
        )
    nonce_a = _b64d(str(crypto_meta.get("stream_nonce_b64", "")))
    nonce_b = _b64d(str(crypto_meta.get("tags_b64", "")))

    pix = np.ascontiguousarray(roi_bgr[mask], dtype=np.uint8).reshape(-1)
    enc = _chaos_stream_transform(pix, core_seed, map_name, nonce_a, nonce_b)
    roi_bgr[mask] = enc.reshape(-1, 3)
    return roi_bgr, crypto_meta


def _decrypt_masked_bytes_cpu_chaos(
    roi_bgr: np.ndarray,
    mask: np.ndarray,
    seed: bytes,
    map_name: str,
    crypto_meta: Optional[dict] = None,
    frame_idx: int = 0,
    track_id: int = -1,
    cls_id: int = -1,
    policy_tag: str = "default_sensitive",
    bbox: Optional[Tuple[int, int, int, int]] = None,
    mask_hash: str = "",
    roi_index: int = 0,
):
    if roi_bgr.size == 0 or mask is None or mask.size == 0 or not np.any(mask) or not crypto_meta:
        return roi_bgr

    if roi_bgr.dtype != np.uint8:
        roi_bgr = roi_bgr.astype(np.uint8, copy=False)
    if mask.dtype != np.bool_:
        mask = mask.astype(np.bool_, copy=False)

    core_seed = _bind_roi_seed(
        seed,
        frame_idx=frame_idx,
        track_id=track_id,
        cls_id=cls_id,
        policy_tag=policy_tag,
        bbox=bbox,
        mask_hash=mask_hash,
        roi_index=roi_index,
    )
    nonce_a = _b64d(str(crypto_meta.get("stream_nonce_b64", "")))
    nonce_b = _b64d(str(crypto_meta.get("tags_b64", "")))

    pix = np.ascontiguousarray(roi_bgr[mask], dtype=np.uint8).reshape(-1)
    dec = _chaos_stream_transform(pix, core_seed, map_name, nonce_a, nonce_b)
    roi_bgr[mask] = dec.reshape(-1, 3)
    return roi_bgr


def encrypt_masked_bytes_cpu(roi_bgr: np.ndarray, mask: np.ndarray, seed: bytes, map_name: str) -> np.ndarray:
    roi_out, _ = _encrypt_masked_bytes_cpu_chaos(
        roi_bgr, mask, seed, map_name,
        frame_idx=0, track_id=-1, cls_id=-1,
        policy_tag="default_sensitive",
        bbox=None, mask_hash="", roi_index=0,
        crypto_meta={
            "scheme": "chaos_stream_v1",
            "stream_nonce_b64": _b64e(bytes(16)),
            "tags_b64": _b64e(bytes(16)),
            "chunk_size": 4096,
            "block_size": 1024,
        },
    )
    return roi_out


def decrypt_masked_bytes_cpu(roi_bgr: np.ndarray, mask: np.ndarray, seed: bytes, map_name: str) -> np.ndarray:
    return _decrypt_masked_bytes_cpu_chaos(
        roi_bgr, mask, seed, map_name,
        crypto_meta={
            "scheme": "chaos_stream_v1",
            "stream_nonce_b64": _b64e(bytes(16)),
            "tags_b64": _b64e(bytes(16)),
            "chunk_size": 4096,
            "block_size": 1024,
        },
        frame_idx=0, track_id=-1, cls_id=-1,
        policy_tag="default_sensitive",
        bbox=None, mask_hash="", roi_index=0,
    )

# -----------------------------
# Mask packing for sidecar
# -----------------------------

def pack_mask(mask_bool: np.ndarray) -> dict:
    h, w = mask_bool.shape
    packed = np.packbits(mask_bool.astype(np.uint8).reshape(-1))
    b64 = base64.b64encode(packed.tobytes()).decode("ascii")
    return {"h": h, "w": w, "b64": b64}


def unpack_mask(obj: dict) -> np.ndarray:
    h, w = int(obj["h"]), int(obj["w"])
    raw = base64.b64decode(obj["b64"].encode("ascii"))
    packed = np.frombuffer(raw, dtype=np.uint8)
    bits = np.unpackbits(packed)[: h * w]
    return bits.reshape(h, w).astype(bool)


# -----------------------------
# Detection (SEGMENTATION masks) + TRACKING IDs
# -----------------------------

@dataclass
class DetectorBundle:
    coco: YOLO

def load_detectors() -> DetectorBundle:
    coco = YOLO(COCO_MODEL)
    try:
        coco.fuse()
    except Exception:
        pass
    try:
        coco.model.eval()
    except Exception:
        pass
    return DetectorBundle(coco=coco)

def _maybe_resize_for_detect(frame_bgr: np.ndarray, detect_width: int) -> Tuple[np.ndarray, float, float]:
    H, W = frame_bgr.shape[:2]
    if detect_width is None or detect_width <= 0 or W <= detect_width:
        return frame_bgr, 1.0, 1.0
    scale = detect_width / float(W)
    new_w = detect_width
    new_h = int(round(H * scale))
    small = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    sx = W / float(new_w)
    sy = H / float(new_h)
    return small, sx, sy

def _mask_params_for_class(cls_id: int) -> Tuple[float, int]:
    if cls_id == 0:              # person
        return 0.33, MASK_DILATE_PX
    if cls_id in (2, 3, 5, 7):   # vehicles
        return 0.38, MASK_DILATE_PX
    return 0.45, MASK_DILATE_PX

def _close_only(mask_bool: np.ndarray) -> np.ndarray:
    m = (mask_bool.astype(np.uint8) * 255)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, KERNEL_3, iterations=1)
    return (m > 0)

def _dilate(mask_bool: np.ndarray, px: int) -> np.ndarray:
    if px is None or px <= 0:
        return mask_bool
    k = 2 * int(px) + 1
    m = (mask_bool.astype(np.uint8) * 255)
    m = cv2.dilate(m, _get_kernel(k), iterations=1)
    return (m > 0)

def _bbox_iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = map(int, a)
    bx1, by1, bx2, by2 = map(int, b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / float(union)

def _mask_support_bbox(mask_small: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask_small > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

def _debug_compare_track_vs_predict(
    det: DetectorBundle,
    frame_infer: np.ndarray,
    frame_idx: int,
) -> None:
    if not DEBUG_COMPARE_TRACK_VS_PREDICT or not _dbg_enabled_for_frame(frame_idx):
        return

    try:
        with torch.inference_mode():
            pred_res = det.coco.predict(
                frame_infer,
                conf=DEBUG_COMPARE_CONF,
                iou=DEBUG_COMPARE_IOU,
                device=YOLO_DEVICE,
                half=YOLO_HALF,
                retina_masks=RETINA_MASKS,
                verbose=False,
                classes=DEBUG_COMPARE_CLASSES,
            )[0]
    except Exception as e:
        print(f"[DBG][frame={frame_idx}] predict() failed: {e}")
        return
    
    print(f"[DBG][frame={frame_idx}] ===== TRACK vs PREDICT COMPARISON =====")

    pred_num_boxes = 0 if pred_res.boxes is None else len(pred_res.boxes)
    pred_masks_present = (
        getattr(pred_res, "masks", None) is not None and
        pred_res.masks is not None and
        getattr(pred_res.masks, "data", None) is not None
    )

    print(f"[DBG][frame={frame_idx}] predict(): boxes_present={pred_res.boxes is not None} num_boxes={pred_num_boxes}")
    print(f"[DBG][frame={frame_idx}] predict(): masks_present={pred_masks_present}")

    if pred_res.boxes is not None and len(pred_res.boxes) > 0:
        p_cls = pred_res.boxes.cls.detach().cpu().numpy().astype(int)
        p_conf = pred_res.boxes.conf.detach().cpu().numpy().astype(float)
        p_xyxy = pred_res.boxes.xyxy.detach().cpu().numpy()

        num_p_person = int(np.sum(p_cls == 0))
        num_p_vehicle = int(np.sum(np.isin(p_cls, [2, 3, 5, 7])))
        print(
            f"[DBG][frame={frame_idx}] predict() summary: "
            f"persons={num_p_person} vehicles={num_p_vehicle}"
        )

        p_masks = None
        if pred_masks_present:
            p_masks = pred_res.masks.data.detach().cpu().numpy()

        for j in range(len(p_cls)):
            if p_masks is not None and j < len(p_masks):
                support = _mask_support_bbox(p_masks[j])
            else:
                support = None

            mb = support if support is not None else "EMPTY"

            print(
                f"  PRED det[{j}] cls={p_cls[j]} conf={p_conf[j]:.3f} "
                f"xyxy={[round(v, 1) for v in p_xyxy[j].tolist()]} "
                f"mask_bbox={mb}"
            )

    print(f"[DBG][frame={frame_idx}] =====================================")

def _max_iou_with_class(
    box_xyxy: Tuple[int, int, int, int],
    boxes_xyxy: np.ndarray,
    cls_arr: np.ndarray,
    target_cls: int
) -> float:
    best = 0.0
    for i in range(len(cls_arr)):
        if int(cls_arr[i]) != int(target_cls):
            continue
        bx1, by1, bx2, by2 = boxes_xyxy[i]
        iou = _bbox_iou_xyxy(
            box_xyxy,
            (int(round(bx1)), int(round(by1)), int(round(bx2)), int(round(by2)))
        )
        if iou > best:
            best = iou
    return best

def detect_rois_with_masks(
    det: DetectorBundle,
    frame_bgr: np.ndarray,
    frame_idx: int,
    detect_width: int = 1280,
    pack_for_sidecar: bool = False,
) -> List[dict]:
    """
    Returns list of dicts:
      {"bbox":[x1,y1,x2,y2], "mask": bool[h,w], "mask_pack": optional, "cls": int,
       "conf": float, "track_id": int, "policy_tag": str}

    Fix:
      - Do NOT assume masks[i] belongs to boxes[i]
      - Match each detection box to the best mask by IoU of support bbox
      - If no mask matches, fall back to box encryption for sensitive classes
      - Compare track() vs predict() in debug mode
    """
    H, W = frame_bgr.shape[:2]
    out: List[dict] = []

    frame_infer, sx, sy = _maybe_resize_for_detect(frame_bgr, detect_width)
    inf_H, inf_W = frame_infer.shape[:2]

    _debug_compare_track_vs_predict(det, frame_infer, frame_idx)

    with torch.inference_mode():
        res = det.coco.track(
            frame_infer,
            conf=CONF_THRES,
            iou=IOU_THRES,
            device=YOLO_DEVICE,
            half=YOLO_HALF,
            retina_masks=RETINA_MASKS,
            verbose=False,
            persist=TRACK_PERSIST,
            tracker=TRACKER_CFG,
            classes=COCO_CLASSES_LIST,
        )[0]

    if DEBUG_PRINT_RAW_DETECTIONS:
        _dbg(frame_idx, f"\n[DBG][frame={frame_idx}] YOLO raw result")
        _dbg(
            frame_idx,
            f"  boxes_present={res.boxes is not None} "
            f"num_boxes={0 if res.boxes is None else len(res.boxes)}"
        )
        _dbg(
            frame_idx,
            f"  masks_present={getattr(res, 'masks', None) is not None and getattr(res.masks, 'data', None) is not None}"
        )

    if res.boxes is None or len(res.boxes) == 0:
        _dbg(frame_idx, f"[DBG][frame={frame_idx}] RETURN early reason=no_boxes")
        return out

    cls_arr = res.boxes.cls.detach().cpu().numpy().astype(int)
    conf_arr = res.boxes.conf.detach().cpu().numpy().astype(float)
    boxes_xyxy = res.boxes.xyxy.detach().cpu().numpy()

    if getattr(res.boxes, "id", None) is not None and res.boxes.id is not None:
        id_arr = res.boxes.id.detach().cpu().numpy().astype(int)
    else:
        _dbg(frame_idx, f"[DBG][frame={frame_idx}] no_track_ids -> using deterministic synthetic IDs for this frame")
        # Tracker IDs are often missing on the very first frame even though boxes/masks exist.
        # Do not leave the frame unprotected; assign deterministic per-frame synthetic IDs.
        id_arr = -1 - np.arange(len(cls_arr), dtype=int)

    masks = None
    if getattr(res, "masks", None) is not None and res.masks is not None and getattr(res.masks, "data", None) is not None:
        masks = res.masks.data.detach().cpu().numpy()

    if DEBUG_PRINT_RAW_DETECTIONS:
        for j in range(len(cls_arr)):
            _dbg(
                frame_idx,
                f"  RAW det[{j}] cls={cls_arr[j]} conf={conf_arr[j]:.3f} "
                f"tid={id_arr[j]} xyxy={[round(v, 1) for v in boxes_xyxy[j].tolist()]}",
                cls_id=int(cls_arr[j]),
                track_id=int(id_arr[j]),
            )

    mask_infos = []
    if masks is not None:
        _dbg(frame_idx, f"[DBG][frame={frame_idx}] len(boxes)={len(boxes_xyxy)} len(masks)={len(masks)}")
        for j in range(len(masks)):
            support = _mask_support_bbox(masks[j])
            if support is None:
                mb = "EMPTY"
            else:
                mb = support
            if j < len(boxes_xyxy):
                cls_dbg = int(cls_arr[j])
                tid_dbg = int(id_arr[j])
                box_dbg = [round(v, 1) for v in boxes_xyxy[j].tolist()]
            else:
                cls_dbg = -1
                tid_dbg = -1
                box_dbg = "n/a"

            _dbg(
                frame_idx,
                f"  ALIGN j={j} cls={cls_dbg} tid={tid_dbg} box={box_dbg} mask_bbox={mb}",
                cls_id=cls_dbg,
                track_id=tid_dbg,
            )

            if support is not None:
                mask_infos.append({
                    "mask_index": j,
                    "mask_small": masks[j],
                    "support_bbox": support,
                    "used": False,
                })

    candidates = []
    for i, c in enumerate(cls_arr):
        c = int(c)

        if DEBUG_PRINT_CANDIDATE_FILTERS:
            _dbg(
                frame_idx,
                f"[DBG][frame={frame_idx}] checking det[{i}] cls={c} conf={float(conf_arr[i]):.3f}",
                cls_id=c,
                track_id=int(id_arr[i]),
            )

        if c not in COCO_CLASSES_OF_INTEREST:
            if DEBUG_PRINT_CANDIDATE_FILTERS:
                _dbg(
                    frame_idx,
                    f"  -> REJECT det[{i}] reason=class_not_of_interest",
                    cls_id=c,
                    track_id=int(id_arr[i]),
                )
            continue

        thr, grow_px = _mask_params_for_class(c)

        bx1, by1, bx2, by2 = boxes_xyxy[i]
        bx1, by1, bx2, by2 = map(float, [bx1, by1, bx2, by2])

        x1 = int(round(bx1 * sx))
        y1 = int(round(by1 * sy))
        x2 = int(round(bx2 * sx))
        y2 = int(round(by2 * sy))

        x1 = clamp(x1, 0, W - 1)
        y1 = clamp(y1, 0, H - 1)
        x2 = clamp(x2, 1, W)
        y2 = clamp(y2, 1, H)
        if x2 <= x1 or y2 <= y1:
            if DEBUG_PRINT_CANDIDATE_FILTERS:
                _dbg(
                    frame_idx,
                    f"  -> REJECT det[{i}] reason=invalid_scaled_box "
                    f"scaled_box=({x1},{y1},{x2},{y2})",
                    cls_id=c,
                    track_id=int(id_arr[i]),
                )
            continue

        if DEBUG_PRINT_CANDIDATE_FILTERS:
            _dbg(
                frame_idx,
                f"  -> KEEP candidate det[{i}] cls={c} conf={float(conf_arr[i]):.3f} "
                f"tid={int(id_arr[i])} bbox_full=({x1},{y1},{x2},{y2})",
                cls_id=c,
                track_id=int(id_arr[i]),
            )

        candidates.append({
            "cls": c,
            "conf": float(conf_arr[i]),
            "track_id": int(id_arr[i]),
            "thr": float(thr),
            "grow": int(grow_px),
            "bbox_full": (x1, y1, x2, y2),
            "bbox_inf": (int(round(bx1)), int(round(by1)), int(round(bx2)), int(round(by2))),
        })

    if not candidates:
        _dbg(frame_idx, f"[DBG][frame={frame_idx}] RETURN early reason=no_candidates")
        return out

    candidates.sort(key=lambda it: (-it["conf"], it["cls"], it["track_id"]))

    for c in candidates:
        thr = c["thr"]
        grow_px = c["grow"]

        x1, y1, x2, y2 = c["bbox_full"]
        ibx1, iby1, ibx2, iby2 = c["bbox_inf"]

        det_box_inf = (ibx1, iby1, ibx2, iby2)

        best_mask_info = None
        best_iou = 0.0
        for mi in mask_infos:
            if mi.get("used", False):
                continue
            iou = _bbox_iou_xyxy(det_box_inf, mi["support_bbox"])
            if iou > best_iou:
                best_iou = iou
                best_mask_info = mi

        if best_mask_info is not None:
            mask_small = best_mask_info["mask_small"]
            support_bbox = best_mask_info["support_bbox"]
        else:
            mask_small = None
            support_bbox = None

        _dbg(
            frame_idx,
            f"[DBG][frame={frame_idx}] match tid={c['track_id']} cls={c['cls']} "
            f"det_box_inf={det_box_inf} best_iou={best_iou:.4f} "
            f"support_bbox={support_bbox}",
            cls_id=int(c["cls"]),
            track_id=int(c["track_id"]),
        )

        roi_mask = None

        use_mask = False
        if mask_small is not None and best_iou >= 0.75:
            use_mask = True

            # Extra guard for vehicle detections:
            # if this vehicle box overlaps a person box too much, distrust the mask
            if int(c["cls"]) in (2, 3, 5, 7):
                person_iou = _max_iou_with_class(det_box_inf, boxes_xyxy, cls_arr, 0)
                _dbg(
                    frame_idx,
                    f"  vehicle-person overlap check: person_iou={person_iou:.4f}",
                    cls_id=int(c["cls"]),
                    track_id=int(c["track_id"]),
                )
                if person_iou >= 0.25:
                    _dbg(
                        frame_idx,
                        f"  -> REJECT vehicle mask due to person overlap "
                        f"(person_iou={person_iou:.4f})",
                        cls_id=int(c["cls"]),
                        track_id=int(c["track_id"]),
                    )
                    use_mask = False

        if use_mask:
            if best_mask_info is not None:
                best_mask_info["used"] = True
            mh, mw = mask_small.shape

            scale_mx = mw / float(inf_W)
            scale_my = mh / float(inf_H)

            mx1 = int(round(ibx1 * scale_mx))
            my1 = int(round(iby1 * scale_my))
            mx2 = int(round(ibx2 * scale_mx))
            my2 = int(round(iby2 * scale_my))

            mx1 = clamp(mx1, 0, mw - 1)
            my1 = clamp(my1, 0, mh - 1)
            mx2 = clamp(mx2, mx1 + 1, mw)
            my2 = clamp(my2, my1 + 1, mh)

            mask_crop_small = mask_small[my1:my2, mx1:mx2]

            if DEBUG_PRINT_MASK_STATS:
                full_mask_nonzero = int((mask_small > 0).sum())
                _dbg(
                    frame_idx,
                    f"  full mask check: mask_small_nonzero={full_mask_nonzero} "
                    f"mask_small_min={float(mask_small.min()):.4f} "
                    f"mask_small_max={float(mask_small.max()):.4f} "
                    f"mx1,my1,mx2,my2=({mx1},{my1},{mx2},{my2})",
                    cls_id=int(c["cls"]),
                    track_id=int(c["track_id"]),
                )

                crop_nonzero = int((mask_crop_small > 0).sum())
                _dbg(
                    frame_idx,
                    f"  crop mask check: crop_nonzero={crop_nonzero} "
                    f"crop_min={float(mask_crop_small.min()):.4f} "
                    f"crop_max={float(mask_crop_small.max()):.4f}",
                    cls_id=int(c["cls"]),
                    track_id=int(c["track_id"]),
                )

                ys, xs = np.where(mask_small > 0)
                if len(xs) > 0 and len(ys) > 0:
                    _dbg(
                        frame_idx,
                        f"  mask support bbox: ({xs.min()},{ys.min()},{xs.max()+1},{ys.max()+1})",
                        cls_id=int(c["cls"]),
                        track_id=int(c["track_id"]),
                    )
                else:
                    _dbg(
                        frame_idx,
                        "  mask support bbox: EMPTY",
                        cls_id=int(c["cls"]),
                        track_id=int(c["track_id"]),
                    )

                _dbg(
                    frame_idx,
                    f"[DBG][frame={frame_idx}] cand tid={c['track_id']} cls={c['cls']} conf={c['conf']:.3f} "
                    f"bbox_full={c['bbox_full']} bbox_inf={c['bbox_inf']} "
                    f"mask_small_shape={mask_small.shape} crop_small_shape={mask_crop_small.shape} "
                    f"thr={thr:.2f}",
                    cls_id=int(c["cls"]),
                    track_id=int(c["track_id"]),
                )

            if DEBUG_PRINT_SMALL_VEHICLES and int(c["cls"]) in (2, 3, 5, 7):
                box_w = x2 - x1
                box_h = y2 - y1
                _dbg(
                    frame_idx,
                    f"[DBG][frame={frame_idx}] VEH cand tid={c['track_id']} conf={c['conf']:.3f} "
                    f"box=({x1},{y1},{x2},{y2}) size=({box_w}x{box_h})",
                    cls_id=int(c["cls"]),
                    track_id=int(c["track_id"]),
                )

            mask_crop_fullprob = cv2.resize(
                mask_crop_small.astype(np.float32),
                (x2 - x1, y2 - y1),
                interpolation=cv2.INTER_LINEAR
            )

            mb = (mask_crop_fullprob >= thr)

            if DEBUG_PRINT_MASK_STATS:
                _dbg(
                    frame_idx,
                    f"  mask stats: prob_min={float(mask_crop_fullprob.min()):.4f} "
                    f"prob_max={float(mask_crop_fullprob.max()):.4f} "
                    f"prob_mean={float(mask_crop_fullprob.mean()):.4f} "
                    f"pixels_above_thr={int(mb.sum())}",
                    cls_id=int(c["cls"]),
                    track_id=int(c["track_id"]),
                )

            if mb.sum() > 0:
                mb = _close_only(mb)
                if grow_px and grow_px > 0:
                    mb = _dilate(mb, grow_px)

                if DEBUG_PRINT_MASK_STATS:
                    _dbg(
                        frame_idx,
                        f"  post-morph pixels={int(mb.sum())}",
                        cls_id=int(c["cls"]),
                        track_id=int(c["track_id"]),
                    )

                x1p, y1p, x2p, y2p = pad_bbox_exclusive(x1, y1, x2, y2, ROI_PAD, W, H)
                if x2p > x1p and y2p > y1p:
                    roi_h = y2p - y1p
                    roi_w = x2p - x1p
                    roi_mask_tmp = np.zeros((roi_h, roi_w), dtype=bool)

                    oy = y1 - y1p
                    ox = x1 - x1p
                    roi_mask_tmp[oy:oy + (y2 - y1), ox:ox + (x2 - x1)] = mb

                    if DEBUG_PRINT_MASK_STATS:
                        _dbg(
                            frame_idx,
                            f"  roi placement: padded_bbox=({x1p},{y1p},{x2p},{y2p}) "
                            f"roi_shape=({roi_h},{roi_w}) offset=(oy={oy},ox={ox}) "
                            f"roi_mask_pixels={int(roi_mask_tmp.sum())}",
                            cls_id=int(c["cls"]),
                            track_id=int(c["track_id"]),
                        )

                    if roi_mask_tmp.sum() > 0:
                        roi_mask = roi_mask_tmp

        if roi_mask is None:
            _dbg(
                frame_idx,
                f"  -> NO TRUSTED MASK for tid={c['track_id']} cls={c['cls']} "
                f"(best_iou={best_iou:.4f}); using full-box fallback",
                cls_id=int(c["cls"]),
                track_id=int(c["track_id"]),
            )

            x1p, y1p, x2p, y2p = pad_bbox_exclusive(x1, y1, x2, y2, ROI_PAD, W, H)
            if x2p <= x1p or y2p <= y1p:
                _dbg(
                    frame_idx,
                    "  -> REJECT candidate reason=invalid_padded_box",
                    cls_id=int(c["cls"]),
                    track_id=int(c["track_id"]),
                )
                continue

            roi_h = y2p - y1p
            roi_w = x2p - x1p
            roi_mask = np.ones((roi_h, roi_w), dtype=bool)

            _dbg(
                frame_idx,
                f"  -> FALLBACK full-box ROI tid={c['track_id']} cls={c['cls']} "
                f"bbox={[int(x1p), int(y1p), int(x2p), int(y2p)]}",
                cls_id=int(c["cls"]),
                track_id=int(c["track_id"]),
            )
        else:
            x1p, y1p, x2p, y2p = pad_bbox_exclusive(x1, y1, x2, y2, ROI_PAD, W, H)

        policy_tag = policy_tag_for_class(int(c["cls"]))

        rec = {
            "bbox": [int(x1p), int(y1p), int(x2p), int(y2p)],
            "mask": roi_mask,
            "cls": int(c["cls"]),
            "conf": float(c["conf"]),
            "track_id": int(c["track_id"]),
            "policy_tag": policy_tag,
        }
        if pack_for_sidecar:
            rec["mask_pack"] = pack_mask(roi_mask)
        out.append(rec)

        if DEBUG_PRINT_MASK_STATS:
            _dbg(
                frame_idx,
                f"  -> ACCEPT ROI tid={c['track_id']} cls={c['cls']} conf={c['conf']:.3f} "
                f"bbox={[int(x1p), int(y1p), int(x2p), int(y2p)]} mask_pixels={int(roi_mask.sum())}",
                cls_id=int(c["cls"]),
                track_id=int(c["track_id"]),
            )

    if DETERMINISTIC_ROI_SORT and len(out) > 1:
        def _key2(r):
            x1, y1, x2, y2 = r["bbox"]
            area = max(0, (x2 - x1)) * max(0, (y2 - y1))
            return (-r.get("conf", 0.0), r.get("cls", 999), r.get("track_id", -1), area, x1, y1, x2, y2)
        out.sort(key=_key2)

    if DEBUG_PRINT_SUMMARY:
        num_person = sum(1 for r in out if int(r["cls"]) == 0)
        num_vehicle = sum(1 for r in out if int(r["cls"]) in (2, 3, 5, 7))
        _dbg(
            frame_idx,
            f"[DBG][frame={frame_idx}] summary: final_rois={len(out)} "
            f"persons={num_person} vehicles={num_vehicle}"
        )

    return out

# -----------------------------
# Frame-type mapping (ROBUST)
# -----------------------------

def map_for_pict_type(pict_type) -> str:
    name = getattr(pict_type, "name", None)
    if name in ("I", "P", "B"):
        pict = name
    else:
        pict = str(pict_type)
        if "PictureType" in pict:
            if ".I" in pict or " I" in pict:
                pict = "I"
            elif ".P" in pict or " P" in pict:
                pict = "P"
            elif ".B" in pict or " B" in pict:
                pict = "B"

    if pict in ("1", "I"):
        return "chen"
    if pict in ("2", "P"):
        return "cubic"
    if pict in ("3", "B"):
        return "skew_tent"
    return "cubic"


# -----------------------------
# Audio helpers (frame-synchronous)
# -----------------------------

def _decode_audio_pcm_s16(in_container: av.container.InputContainer) -> Optional[Tuple[np.ndarray, int, str]]:
    """Decode first audio stream to packed s16 PCM.
    Returns (pcm_i16[samples,channels], sample_rate, layout_name) or None if no audio.
    """
    if not in_container.streams.audio:
        return None

    astream = in_container.streams.audio[0]
    layout_name = astream.layout.name if astream.layout else "stereo"
    sr = int(astream.rate) if getattr(astream, "rate", None) else None

    def _channels_from_layout(layout) -> Optional[int]:
        if not layout:
            return None
        ch = getattr(layout, "channels", None)
        if ch is None:
            return None
        if isinstance(ch, int):
            return ch
        if isinstance(ch, (tuple, list)):
            return len(ch)
        return None

    stream_ch = getattr(astream, "channels", None)
    if isinstance(stream_ch, int) and stream_ch > 0:
        channels = stream_ch
    else:
        channels = _channels_from_layout(astream.layout) or 1

    chunks = []
    resampler = None

    for af in in_container.decode(astream):
        if sr is None and getattr(af, "sample_rate", None):
            sr = int(af.sample_rate)
            resampler = av.audio.resampler.AudioResampler(format="s16", layout=layout_name, rate=sr)

        if resampler is None and sr is not None:
            resampler = av.audio.resampler.AudioResampler(format="s16", layout=layout_name, rate=sr)

        if resampler is None:
            continue

        raf = resampler.resample(af)
        if raf is None:
            continue

        frames = raf if isinstance(raf, list) else [raf]
        for fr in frames:
            arr = fr.to_ndarray()

            fr_ch = _channels_from_layout(getattr(fr, "layout", None))
            ch = fr_ch or channels or 1

            if arr.ndim == 1:
                arr_sc = arr.reshape(-1, ch)
            elif arr.ndim == 2:
                if arr.shape[0] == 1:
                    arr_sc = arr.reshape(-1, ch)
                elif arr.shape[1] == ch:
                    arr_sc = arr
                elif arr.shape[0] == ch:
                    arr_sc = arr.T
                else:
                    arr_sc = arr.reshape(-1, ch)
            else:
                arr_sc = arr.reshape(-1, ch)

            chunks.append(np.ascontiguousarray(arr_sc, dtype=np.int16))

    if sr is None or not chunks:
        return None

    pcm = np.concatenate(chunks, axis=0)
    return pcm, sr, layout_name


def _mux_pcm_segment(
    out_container: av.container.OutputContainer,
    aout,
    seg_sc_i16: np.ndarray,
    sr: int,
    layout_name: str,
    pts_samples: int
) -> int:
    if seg_sc_i16.size == 0:
        return pts_samples
    seg_sc_i16 = np.ascontiguousarray(seg_sc_i16, dtype=np.int16)
    packed = seg_sc_i16.reshape(1, -1)
    af = av.AudioFrame.from_ndarray(packed, format="s16", layout=layout_name)
    af.sample_rate = sr
    af.time_base = Fraction(1, sr)
    af.pts = pts_samples

    for pkt in aout.encode(af):
        out_container.mux(pkt)

    return pts_samples + int(seg_sc_i16.shape[0])

# -----------------------------
# Audio chaos transform (non-XOR): deterministic permutation + substitution + diffusion
# -----------------------------

def _audio_chaos_meta(seed: bytes, map_name: str, total_n: int) -> Tuple[bytes, bytes, int]:
    total_n = max(0, int(total_n))
    chunk_size = 8192 if total_n >= 8192 else max(1024, total_n or 1024)
    nonce_a = derive_stream_seed(seed, f"audio|nonce_a|n={total_n}")[:16]
    nonce_b = derive_stream_seed(seed, f"audio|nonce_b|n={total_n}")[:16]
    return nonce_a, nonce_b, chunk_size


def _chaos_audio_segment(seg_sc_i16: np.ndarray, seed: bytes, map_name: str, decrypt: bool = False) -> np.ndarray:
    """Encrypt/decrypt an audio segment using the pure-chaos PSD core."""
    if seg_sc_i16.size == 0:
        return seg_sc_i16

    seg = seg_sc_i16
    if seg.dtype != np.int16:
        seg = seg.astype(np.int16, copy=False)
    if not seg.flags['C_CONTIGUOUS']:
        seg = np.ascontiguousarray(seg)

    raw = seg.view(np.uint8).reshape(-1)
    nonce_a, nonce_b, chunk_size = _audio_chaos_meta(seed, map_name, raw.size)
    transformed = _chaos_stream_transform(
        raw.copy(),
        derive_stream_seed(seed, "audio-psd"),
        map_name,
        nonce_a,
        nonce_b,
    )
    raw[:] = transformed
    return seg


# -----------------------------
# Matched public-output settings
# -----------------------------
MATCHED_VIDEO_CRF = os.environ.get("MATCHED_VIDEO_CRF", "18")
MATCHED_VIDEO_PRESET = os.environ.get("MATCHED_VIDEO_PRESET", "medium")
MATCHED_AUDIO_BITRATE = int(os.environ.get("MATCHED_AUDIO_BITRATE", "192000"))

# Preview mode logic removed: output ROIs/audio are always chaos-encrypted.
PAYLOAD_ZLIB_LEVEL = int(os.environ.get("PAYLOAD_ZLIB_LEVEL", "6"))


# -----------------------------
# Helpers
# -----------------------------


def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _video_identity(path: str) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return {"width": width, "height": height, "fps": fps, "frames": frames}


def _session_id(master_key: str, width: int, height: int, fps: float, frames: int, has_audio: bool) -> str:
    msg = f"session|w={int(width)}|h={int(height)}|fps={float(fps):.6f}|frames={int(frames)}|audio={1 if has_audio else 0}".encode("utf-8")
    return hmac.new(master_key.encode("utf-8"), msg, hashlib.sha256).hexdigest()[:32]


def _manifest_default_path(payload_path: str) -> str:
    return str(payload_path) + ".manifest.json"


def _write_manifest(path: str, rec: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, sort_keys=True)


def _load_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _count_payload_frame_records(payload_path: str) -> int:
    """
    Return the actual number of video frame records written to payload.jsonl.
    This is safer than trusting container metadata from cv2/ffprobe, because
    encoded MKV/MP4 outputs can report a slightly different frame count.
    """
    max_idx = -1
    with open(payload_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            kind = rec.get("type", "frame")
            if kind in {"meta", "tail_audio"}:
                continue
            if "frame_idx" in rec:
                max_idx = max(max_idx, int(rec["frame_idx"]))
    return max_idx + 1 if max_idx >= 0 else 0


def _patch_payload_header_frames(payload_path: str, frame_count: int) -> None:
    """
    Keep the payload header consistent with the actual payload frame records.
    The header is written before the frame loop, but the true processed count is
    only known after encoding finishes.
    """
    lines = []
    changed = False
    with open(payload_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                lines.append(line)
                continue
            rec = json.loads(line)
            if rec.get("type") == "meta":
                rec["frames"] = int(frame_count)
                line = json.dumps(rec, separators=(",", ":")) + "\n"
                changed = True
            lines.append(line)
    if changed:
        with open(payload_path, "w", encoding="utf-8") as f:
            f.writelines(lines)


def _verify_decrypt_inputs(manifest_path: str, encrypted_video_path: str, roi_sidecar_path: str, payload_path: str, payload_header: dict, current_session_id: str, current_identity: dict) -> None:
    if not manifest_path or not os.path.isfile(manifest_path):
        raise ValueError(f"Decrypt verification failed: manifest not found: {manifest_path}")
    manifest = _load_manifest(manifest_path)

    checks = [
        ("encrypted video", _sha256_file(encrypted_video_path), str(manifest.get("encrypted_video_sha256", ""))),
        ("ROI sidecar", _sha256_file(roi_sidecar_path), str(manifest.get("roi_sidecar_sha256", ""))),
        ("payload", _sha256_file(payload_path), str(manifest.get("payload_sha256", ""))),
    ]
    for label, got, expect in checks:
        if got != expect:
            raise ValueError(f"Decrypt verification failed: {label} does not match manifest.")

    manifest_session_id = str(manifest.get("session_id", ""))
    if not manifest_session_id:
        raise ValueError("Decrypt verification failed: manifest session_id is missing.")

    # IMPORTANT: during decrypt, current_identity/current_session_id describe the encrypted
    # input video. The session_id in the payload/manifest describes the original source
    # video used during encryption, so do not compare it to current_session_id.
    # File binding is already enforced above via SHA-256 hashes.

    # If the manifest contains encrypted_* fields, validate the encrypted video identity
    # against those. Otherwise, keep backwards-compatible validation against width/height/fps
    # only and avoid strict frame-count checks because container frame counts can differ.
    if "encrypted_width" in manifest or "encrypted_height" in manifest:
        for mk, ck in (("encrypted_width", "width"), ("encrypted_height", "height")):
            if mk in manifest and int(manifest.get(mk, -1)) != int(current_identity.get(ck, -2)):
                raise ValueError(f"Decrypt verification failed: manifest encrypted-video {ck} mismatch.")
        if "encrypted_fps" in manifest and abs(float(manifest.get("encrypted_fps", 0.0)) - float(current_identity.get("fps", 0.0))) > 0.05:
            raise ValueError("Decrypt verification failed: manifest encrypted-video fps mismatch.")
    else:
        for k in ("width", "height"):
            if int(manifest.get(k, -1)) != int(current_identity.get(k, -2)):
                raise ValueError(f"Decrypt verification failed: manifest {k} mismatch.")
        if abs(float(manifest.get("fps", 0.0)) - float(current_identity.get("fps", 0.0))) > 0.05:
            raise ValueError("Decrypt verification failed: manifest fps mismatch.")

    if payload_header:
        if str(payload_header.get("session_id", "")) != manifest_session_id:
            raise ValueError("Decrypt verification failed: payload header session_id mismatch.")
        for k in ("width", "height", "frames"):
            if int(payload_header.get(k, -1)) != int(manifest.get(k, -2)):
                raise ValueError(f"Decrypt verification failed: payload header {k} mismatch.")
        if abs(float(payload_header.get("fps", 0.0)) - float(manifest.get("fps", 0.0))) > 0.05:
            raise ValueError("Decrypt verification failed: payload header fps mismatch.")


def _guess_container_format_from_path(path: str) -> Optional[str]:
    ext = os.path.splitext(str(path))[1].lower()
    mapping = {
        ".mkv": "matroska",
        ".webm": "webm",
        ".mp4": "mp4",
        ".m4v": "ipod",
        ".mov": "mov",
        ".avi": "avi",
    }
    return mapping.get(ext)


def _extract_pix_fmt_name(vstream) -> Optional[str]:
    try:
        cc = getattr(vstream, "codec_context", None)
        fmt = getattr(cc, "pix_fmt", None)
        if isinstance(fmt, str) and fmt:
            return fmt
        fmt_obj = getattr(cc, "format", None)
        name = getattr(fmt_obj, "name", None)
        if isinstance(name, str) and name:
            return name
    except Exception:
        pass
    return None


def _probe_input_profile(in_path: str) -> Dict[str, Optional[str]]:
    info = {
        "container": None,
        "video_codec": None,
        "video_pix_fmt": None,
        "audio_codec": None,
    }
    try:
        probe = av.open(in_path)
        try:
            fmt = getattr(probe, "format", None)
            info["container"] = getattr(fmt, "name", None)
        except Exception:
            pass
        if probe.streams.video:
            vstream = probe.streams.video[0]
            cc = getattr(vstream, "codec_context", None)
            info["video_codec"] = getattr(cc, "name", None) or getattr(vstream.codec_context.codec, "name", None)
            info["video_pix_fmt"] = _extract_pix_fmt_name(vstream)
        if probe.streams.audio:
            astream = probe.streams.audio[0]
            acc = getattr(astream, "codec_context", None)
            info["audio_codec"] = getattr(acc, "name", None) or getattr(astream.codec_context.codec, "name", None)
        probe.close()
    except Exception:
        pass
    return info


def _matched_yuv_pix_fmt(source_pix_fmt: Optional[str]) -> str:
    spf = (source_pix_fmt or "").lower()
    if "444" in spf:
        return "yuv444p"
    if "422" in spf:
        return "yuv422p"
    if "420" in spf:
        return "yuv420p"
    return "yuv420p"


def _select_video_encoder(source_video_codec: Optional[str]):
    svc = (source_video_codec or "").lower().strip()
    if svc in {"h264", "avc1"}:
        return "libx264", {"crf": str(MATCHED_VIDEO_CRF), "preset": str(MATCHED_VIDEO_PRESET)}
    if svc in {"hevc", "h265", "hev1", "hvc1"}:
        return "libx265", {"crf": str(MATCHED_VIDEO_CRF), "preset": str(MATCHED_VIDEO_PRESET)}
    if svc == "vp9":
        return "libvpx-vp9", {"crf": "33", "b:v": "0"}
    if svc == "vp8":
        return "libvpx", {"crf": "10", "b:v": "0"}
    if svc in {"av1", "av01"}:
        return "libaom-av1", {"crf": "30", "b:v": "0"}
    return "libx264", {"crf": str(MATCHED_VIDEO_CRF), "preset": str(MATCHED_VIDEO_PRESET)}


def _select_video_pix_fmt(source_video_codec: Optional[str], source_pix_fmt: Optional[str]) -> str:
    svc = (source_video_codec or "").lower().strip()
    spf = (source_pix_fmt or "").lower().strip()

    # Normal delivery codecs should stay on broadly supported YUV formats.
    if svc in {"h264", "avc1", "hevc", "h265", "hev1", "hvc1", "vp9", "vp8", "av1", "av01"}:
        return _matched_yuv_pix_fmt(spf)

    # For generated/attacked/helper files, the probe can report formats like bgra/bgr0/rgba
    # or other formats that libx264/libx265 cannot accept directly. Do not preserve those
    # literally when we later fall back to a delivery codec.
    if spf in {
        "bgra", "bgr0", "rgba", "argb", "abgr", "rgb0",
        "bgr24", "rgb24",
        "gbrp", "gbrp10le", "gbrp12le",
        "pal8",
    }:
        return "yuv420p"

    # Gray inputs can stay gray only if you intentionally choose a compatible codec path.
    # For our matched public-output path, yuv420p is the safest default.
    if spf.startswith("gray"):
        return "yuv420p"

    # If the probed format is already a sane YUV/NV format, keep the closest supported family.
    if spf.startswith("yuv") or spf.startswith("nv"):
        return _matched_yuv_pix_fmt(spf)

    return "yuv420p"


def _select_audio_encoder(source_audio_codec: Optional[str]):
    sac = (source_audio_codec or "").lower().strip()
    if sac in {"aac", ""}:
        return "aac", MATCHED_AUDIO_BITRATE
    if sac == "flac":
        return "flac", None
    if sac in {"mp3", "mp2"}:
        return "libmp3lame", MATCHED_AUDIO_BITRATE
    if sac == "opus":
        return "libopus", MATCHED_AUDIO_BITRATE
    if sac == "vorbis":
        return "libvorbis", MATCHED_AUDIO_BITRATE
    if sac in {"alac", "pcm_s16le", "pcm_s24le", "pcm_s32le", "pcm_f32le", "pcm_f64le"}:
        return sac, None
    return "aac", MATCHED_AUDIO_BITRATE


@dataclass
class OutputProfile:
    container_format: Optional[str]
    video_codec: str
    video_pix_fmt: str
    video_options: Dict[str, str]
    audio_codec: str
    audio_bitrate: Optional[int]
    source_container: Optional[str] = None
    source_video_codec: Optional[str] = None
    source_video_pix_fmt: Optional[str] = None
    source_audio_codec: Optional[str] = None


def _sanitize_output_profile(profile: OutputProfile) -> OutputProfile:
    vcodec = (profile.video_codec or "").lower().strip()
    vpf = (profile.video_pix_fmt or "").lower().strip()

    # Delivery codecs in this framework should not be paired with raw RGB/BGRA style formats.
    if vcodec in {"libx264", "libx265", "libvpx", "libvpx-vp9", "libaom-av1"}:
        if (
            not vpf
            or vpf in {"bgra", "bgr0", "rgba", "argb", "abgr", "rgb0", "bgr24", "rgb24", "pal8"}
            or vpf.startswith("gray")
            or vpf.startswith("gbr")
            or (not (vpf.startswith("yuv") or vpf.startswith("nv")))
        ):
            profile.video_pix_fmt = "yuv420p"

    return profile


def choose_output_profile(in_path: str, out_path: str) -> OutputProfile:
    src = _probe_input_profile(in_path)
    container_format = _guess_container_format_from_path(out_path)
    if container_format is None:
        src_container = (src.get("container") or "").split(",")[0].strip() or None
        if src_container in {"matroska", "webm", "mp4", "mov", "avi"}:
            container_format = src_container
    video_codec, video_options = _select_video_encoder(src.get("video_codec"))
    video_pix_fmt = _select_video_pix_fmt(src.get("video_codec"), src.get("video_pix_fmt"))
    audio_codec, audio_bitrate = _select_audio_encoder(src.get("audio_codec"))
    profile = OutputProfile(
        container_format=container_format,
        video_codec=video_codec,
        video_pix_fmt=video_pix_fmt,
        video_options=video_options,
        audio_codec=audio_codec,
        audio_bitrate=audio_bitrate,
        source_container=src.get("container"),
        source_video_codec=src.get("video_codec"),
        source_video_pix_fmt=src.get("video_pix_fmt"),
        source_audio_codec=src.get("audio_codec"),
    )
    return _sanitize_output_profile(profile)

def _safe_rate(fps: float) -> Fraction:
    if fps is None or not np.isfinite(fps) or fps <= 0:
        return Fraction(30, 1)
    return Fraction(fps).limit_denominator(100000)

def open_video_writer(out_path: str, width: int, height: int, fps: float, profile: OutputProfile):
    profile = _sanitize_output_profile(profile)
    if profile.container_format:
        out = av.open(out_path, mode="w", format=profile.container_format)
    else:
        out = av.open(out_path, mode="w")
    rate = _safe_rate(fps)
    st = out.add_stream(profile.video_codec, rate=rate)
    st.width = width
    st.height = height
    st.pix_fmt = profile.video_pix_fmt
    st.options = dict(profile.video_options)
    try:
        st.codec_context.time_base = Fraction(1, int(rate))
    except Exception:
        pass
    return out, st


def add_audio_stream(out_container, sr: int, layout_name: str, profile: OutputProfile):
    aout = out_container.add_stream(profile.audio_codec, rate=sr)
    aout.layout = layout_name
    if profile.audio_bitrate is not None:
        try:
            aout.bit_rate = int(profile.audio_bitrate)
        except Exception:
            pass
    return aout


# -----------------------------
# Payload crypto
# -----------------------------

def _payload_common_kwargs(frame_idx: int, track_id: int, cls_id: int, policy_tag: str, bbox, mask_hash: str, roi_index: int):
    return dict(
        frame_idx=int(frame_idx),
        track_id=int(track_id),
        cls_id=int(cls_id),
        policy_tag=str(policy_tag),
        bbox=None if bbox is None else tuple(int(v) for v in bbox),
        mask_hash=str(mask_hash or ""),
        roi_index=int(roi_index),
    )


def _payload_encrypt_bytes(base, plain: bytes, seed: bytes, map_name: str, domain: str, common: dict):
    work_seed = base.derive_stream_seed(seed, f"payload|{domain}")
    meta = base._prepare_chaos_crypto_meta(work_seed, map_name, **common)
    nonce_a = base._b64d(str(meta.get("stream_nonce_b64", "")))
    nonce_b = base._b64d(str(meta.get("tags_b64", "")))
    core_seed = base._bind_roi_seed(work_seed, **common)
    stream_seed = hmac.new(core_seed, nonce_a + nonce_b, hashlib.sha256).digest()
    arr = np.frombuffer(plain, dtype=np.uint8).copy()
    ks = np.asarray(base.keystream_u8(base.derive_stream_seed(stream_seed, f"payload-xor|{domain}"), map_name, int(arr.size)), dtype=np.uint8)
    out = np.bitwise_xor(arr, ks).astype(np.uint8, copy=False)
    return out.tobytes(), meta


def _payload_decrypt_bytes(base, cipher: bytes, seed: bytes, map_name: str, meta: dict, domain: str, common: dict):
    work_seed = base.derive_stream_seed(seed, f"payload|{domain}")
    nonce_a = base._b64d(str(meta.get("stream_nonce_b64", "")))
    nonce_b = base._b64d(str(meta.get("tags_b64", "")))
    core_seed = base._bind_roi_seed(work_seed, **common)
    stream_seed = hmac.new(core_seed, nonce_a + nonce_b, hashlib.sha256).digest()
    arr = np.frombuffer(cipher, dtype=np.uint8).copy()
    ks = np.asarray(base.keystream_u8(base.derive_stream_seed(stream_seed, f"payload-xor|{domain}"), map_name, int(arr.size)), dtype=np.uint8)
    out = np.bitwise_xor(arr, ks).astype(np.uint8, copy=False)
    return out.tobytes()


def _payload_keystream_bytes(base, plain_len: int, seed: bytes, map_name: str, meta: dict, domain: str, common: dict) -> bytes:
    work_seed = base.derive_stream_seed(seed, f"payload|{domain}")
    nonce_a = base._b64d(str(meta.get("stream_nonce_b64", "")))
    nonce_b = base._b64d(str(meta.get("tags_b64", "")))
    core_seed = base._bind_roi_seed(work_seed, **common)
    stream_seed = hmac.new(core_seed, nonce_a + nonce_b, hashlib.sha256).digest()
    n = max(0, int(plain_len))
    if n == 0:
        return b""
    ks = np.asarray(
        base.keystream_u8(base.derive_stream_seed(stream_seed, f"payload-xor|{domain}"), map_name, n),
        dtype=np.uint8,
    )
    return ks.tobytes()


def _payload_tag(master_key: str, rec_wo_tag: dict, cipher: bytes) -> str:
    aad = json.dumps(rec_wo_tag, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hmac.new(master_key.encode("utf-8"), aad + cipher, hashlib.sha256).hexdigest()


def _store_payload_record(payload_f, rec: dict):
    payload_f.write(json.dumps(rec, separators=(",", ":")) + "\n")


def _load_payload_records(path: str) -> Tuple[dict, Dict[int, dict], Optional[dict]]:
    header = {}
    frames: Dict[int, dict] = {}
    tail = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            kind = rec.get("type", "frame")
            if kind == "meta":
                header = rec
            elif kind == "tail_audio":
                tail = rec
            else:
                frames[int(rec["frame_idx"])] = rec
    return header, frames, tail


# -----------------------------
# Main processing
# -----------------------------

def _process_payload_video(
    base,
    in_path: str,
    out_path: str,
    master_key: str,
    mode: str,
    roi_sidecar_path: Optional[str],
    payload_path: Optional[str],
    manifest_path: Optional[str],
    detect_width: int,
    keystream_dump_path: Optional[str],
    reuse_rois: bool = False,
):
    assert mode in {"encrypt", "decrypt"}
    if mode == "decrypt" and not payload_path:
        raise ValueError("Decrypt requires --payload")
    if payload_path and not manifest_path:
        manifest_path = _manifest_default_path(payload_path)

    roi_records: Dict[int, dict] = {}
    reuse_rois_for_encrypt = bool(mode == "encrypt" and reuse_rois and roi_sidecar_path and os.path.exists(roi_sidecar_path))
    if reuse_rois_for_encrypt:
        with open(roi_sidecar_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                roi_records[int(rec["frame_idx"])] = rec

    det = base.load_detectors() if (mode == "encrypt" and not reuse_rois_for_encrypt) else None

    if mode == "encrypt" and det is not None:
        print("[INFO] Warmup YOLO + Numba...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        with torch.inference_mode():
            try:
                _ = det.coco.track(
                    dummy,
                    device=base.YOLO_DEVICE,
                    half=base.YOLO_HALF,
                    verbose=False,
                    persist=base.TRACK_PERSIST,
                    tracker=base.TRACKER_CFG,
                    classes=getattr(base, "COCO_CLASSES_LIST", sorted(base.COCO_CLASSES_OF_INTEREST)),
                )
            except Exception:
                _ = det.coco.predict(dummy, device=base.YOLO_DEVICE, half=base.YOLO_HALF, verbose=False)
        _ = base.cubic_keystream_u8(b"warmup", 64)
        _ = base.skew_tent_keystream_u8(b"warmup", 64)
        _ = base.chen_keystream_u8(b"warmup", 64)

    in_container = av.open(in_path)
    in_stream = in_container.streams.video[0]
    in_stream.thread_type = "AUTO"

    audio_container = av.open(in_path)
    audio_info = base._decode_audio_pcm_s16(audio_container)
    audio_container.close()
    has_audio = audio_info is not None
    if has_audio:
        pcm_all, audio_sr, audio_layout = audio_info
    else:
        pcm_all, audio_sr, audio_layout = None, None, None

    fps = float(in_stream.average_rate) if in_stream.average_rate else 30.0
    width = in_stream.codec_context.width
    height = in_stream.codec_context.height
    input_identity = _video_identity(in_path)
    session_id = _session_id(master_key, input_identity["width"], input_identity["height"], input_identity["fps"], input_identity["frames"], has_audio)

    output_profile = choose_output_profile(in_path, out_path)
    if mode == "decrypt" and has_audio:
        container_name = (output_profile.container_format or "").lower()
        if container_name in {"mp4", "ipod", "mov"}:
            output_profile.audio_codec = "alac"
        else:
            output_profile.audio_codec = "flac"
        output_profile.audio_bitrate = None
    print(
        f"[INFO] PUBLIC_OUTPUT_MATCH: container={output_profile.container_format or 'auto'} "
        f"video={output_profile.video_codec}/{output_profile.video_pix_fmt} "
        f"audio={output_profile.audio_codec}"
    )

    out_container, out_stream = open_video_writer(out_path, width, height, fps, output_profile)
    aout = add_audio_stream(out_container, audio_sr, audio_layout, output_profile) if has_audio else None

    sidecar_f = None
    payload_f = None
    if mode == "encrypt":
        if roi_sidecar_path:
            sidecar_f = open(roi_sidecar_path, "w", encoding="utf-8", buffering=1024 * 1024)
        if payload_path:
            payload_f = open(payload_path, "w", encoding="utf-8", buffering=1024 * 1024)
            _store_payload_record(payload_f, {
                "type": "meta",
                "version": 1,
                "session_id": session_id,
                "width": int(input_identity["width"]),
                "height": int(input_identity["height"]),
                "fps": float(input_identity["fps"]),
                "frames": int(input_identity["frames"]),
                "video_output_mode": "chaos_cipher",
                "audio_output_mode": "chaos_cipher",
                "audio_sr": audio_sr,
                "audio_layout": audio_layout,
                "has_audio": bool(has_audio),
            })

    payload_header: dict = {}
    payload_records: Dict[int, dict] = {}
    payload_tail: Optional[dict] = None
    if mode == "decrypt":
        if not roi_sidecar_path:
            raise ValueError("Decrypt requires --roi_sidecar")
        with open(roi_sidecar_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                roi_records[int(rec["frame_idx"])] = rec
        payload_header, payload_records, payload_tail = _load_payload_records(payload_path)
        _verify_decrypt_inputs(
            manifest_path=manifest_path,
            encrypted_video_path=in_path,
            roi_sidecar_path=roi_sidecar_path,
            payload_path=payload_path,
            payload_header=payload_header,
            current_session_id=session_id,
            current_identity=input_identity,
        )
        has_audio = bool(payload_header.get("has_audio", has_audio))
        audio_sr = int(payload_header.get("audio_sr", audio_sr or 48000)) if has_audio else None
        audio_layout = payload_header.get("audio_layout", audio_layout or "stereo") if has_audio else None

    total_frames = in_stream.frames if in_stream.frames else None
    pbar = tqdm(total=total_frames, desc=f"{mode} frames", unit="frame")

    ks_f = None
    if mode == "encrypt" and keystream_dump_path:
        ks_f = open(keystream_dump_path, "wb", buffering=1024 * 1024)
        print(f"[INFO] Payload-cipher dump enabled: {keystream_dump_path}")

    audio_pts = 0
    audio_prev_end = 0
    frame_idx = 0
    last_map_name = "cubic"

    for av_frame in in_container.decode(in_stream):
        frame = av_frame.to_ndarray(format="bgr24")
        pict_raw = av_frame.pict_type
        map_name = base.map_for_pict_type(pict_raw)
        last_map_name = map_name

        if mode == "encrypt":
            if reuse_rois_for_encrypt:
                rec = roi_records.get(frame_idx, {"rois": [], "map_name": map_name})
                map_name = str(rec.get("map_name", map_name))
                rois = rec.get("rois", [])
            else:
                rois = base.detect_rois_with_masks(
                    det,
                    frame,
                    frame_idx=frame_idx,
                    detect_width=detect_width,
                    pack_for_sidecar=True,
                )
            for roi_index, r in enumerate(rois):
                r["roi_index"] = int(r.get("roi_index", roi_index))
                cls_id = int(r.get("cls", -1))
                track_id = int(r.get("track_id", -1))
                policy_tag = str(r.get("policy_tag", base.policy_tag_for_class(cls_id)))
                r["policy_tag"] = policy_tag
                roi_mask0 = r.get("mask", None)
                if roi_mask0 is None:
                    roi_mask0 = base.unpack_mask(r["mask_pack"])
                r["mask_hash"] = str(r.get("mask_hash", base._compute_mask_hash(roi_mask0)))
        else:
            rec = roi_records.get(frame_idx, {"rois": [], "map_name": map_name})
            map_name = str(rec.get("map_name", map_name))
            rois = rec.get("rois", [])

        # sidecar write
        if mode == "encrypt" and sidecar_f is not None:
            rois_for_json = []
            for roi_index, r in enumerate(rois):
                mask_obj = r.get("mask_pack") if r.get("mask_pack") is not None else base.pack_mask(r["mask"])
                rois_for_json.append({
                    "bbox": r["bbox"],
                    "mask_pack": mask_obj,
                    "cls": int(r.get("cls", -1)),
                    "conf": float(r.get("conf", 0.0)),
                    "track_id": int(r.get("track_id", -1)),
                    "policy_tag": str(r.get("policy_tag", base.policy_tag_for_class(int(r.get("cls", -1))))),
                    "roi_index": int(r.get("roi_index", roi_index)),
                    "mask_hash": str(r.get("mask_hash", "")),
                })
            sidecar_f.write(json.dumps({
                "frame_idx": frame_idx,
                "map_name": map_name,
                "rois": rois_for_json,
            }, separators=(",", ":")) + "\n")

        payload_frame = {"type": "frame", "frame_idx": frame_idx, "map_name": map_name, "rois": []}

        # process ROI video
        rois_to_apply = rois if mode == "encrypt" else list(reversed(rois))
        for r in rois_to_apply:
            x1, y1, x2, y2 = map(int, r["bbox"])
            x1 = base.clamp(x1, 0, width - 1)
            y1 = base.clamp(y1, 0, height - 1)
            x2 = base.clamp(x2, 1, width)
            y2 = base.clamp(y2, 1, height)
            if x2 <= x1 or y2 <= y1:
                continue
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            roi_mask = r.get("mask")
            if roi_mask is None:
                roi_mask = base.unpack_mask(r["mask_pack"])
            if roi_mask.shape[:2] != (roi.shape[0], roi.shape[1]):
                roi_mask = cv2.resize(roi_mask.astype(np.uint8), (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            if not np.any(roi_mask):
                continue

            track_id = int(r.get("track_id", -1))
            cls_id = int(r.get("cls", -1))
            policy_tag = str(r.get("policy_tag", base.policy_tag_for_class(cls_id)))
            roi_index = int(r.get("roi_index", 0))
            mask_hash = str(r.get("mask_hash", base._compute_mask_hash(roi_mask)))
            bbox = (x1, y1, x2, y2)
            seed = base.derive_policy_seed(
                master_key=master_key,
                frame_idx=frame_idx,
                track_id=track_id,
                cls_id=cls_id,
                map_name=map_name,
                policy_tag=policy_tag,
                scope=base.POLICY_SCOPE,
                bbox=bbox,
                mask_hash=mask_hash,
                roi_index=roi_index,
            )
            common = _payload_common_kwargs(frame_idx, track_id, cls_id, policy_tag, bbox, mask_hash, roi_index)

            if mode == "encrypt":
                # store recoverable secret in payload
                orig_pixels = np.ascontiguousarray(roi[roi_mask], dtype=np.uint8).reshape(-1)
                comp = zlib.compress(orig_pixels.tobytes(), level=PAYLOAD_ZLIB_LEVEL)
                cipher, meta = _payload_encrypt_bytes(base, comp, seed, map_name, "roi", common)
                rec_wo_tag = {
                    "roi_index": roi_index,
                    "bbox": [x1, y1, x2, y2],
                    "cls": cls_id,
                    "track_id": track_id,
                    "policy_tag": policy_tag,
                    "mask_hash": mask_hash,
                    "plain_len": int(orig_pixels.size),
                    "comp_len": int(len(comp)),
                    "crypto_meta": meta,
                }
                tag = _payload_tag(master_key, rec_wo_tag, cipher)
                payload_frame["rois"].append({
                    **rec_wo_tag,
                    "cipher_b64": _b64e(cipher),
                    "tag": tag,
                })
                if ks_f is not None and cipher:
                    ks_f.write(cipher)

                # Write the actual chaos-encrypted ROI to the public/cipher video.
                # The original ROI is still recoverable from the authenticated payload.
                cipher_roi, _visual_meta = base._encrypt_masked_bytes_cpu_chaos(
                    roi.copy(),
                    roi_mask,
                    seed,
                    map_name,
                    frame_idx=frame_idx,
                    track_id=track_id,
                    cls_id=cls_id,
                    policy_tag=policy_tag,
                    bbox=bbox,
                    mask_hash=mask_hash,
                    roi_index=roi_index,
                    crypto_meta=None,
                )
                frame[y1:y2, x1:x2] = cipher_roi
            else:
                frame_payload = payload_records.get(frame_idx, {"rois": []})
                match = None
                for pr in frame_payload.get("rois", []):
                    if int(pr.get("roi_index", -1)) == roi_index:
                        match = pr
                        break
                if match is None:
                    continue
                cipher = _b64d(match["cipher_b64"])
                rec_wo_tag = {
                    "roi_index": int(match["roi_index"]),
                    "bbox": match["bbox"],
                    "cls": int(match["cls"]),
                    "track_id": int(match["track_id"]),
                    "policy_tag": str(match["policy_tag"]),
                    "mask_hash": str(match["mask_hash"]),
                    "plain_len": int(match["plain_len"]),
                    "comp_len": int(match["comp_len"]),
                    "crypto_meta": match["crypto_meta"],
                }
                expect = _payload_tag(master_key, rec_wo_tag, cipher)
                if not hmac.compare_digest(expect, str(match.get("tag", ""))):
                    raise ValueError(f"Payload authentication failed for frame {frame_idx} roi {roi_index}")
                plain_comp = _payload_decrypt_bytes(base, cipher, seed, map_name, match["crypto_meta"], "roi", common)
                plain = zlib.decompress(plain_comp)
                pix = np.frombuffer(plain, dtype=np.uint8)
                if pix.size != int(match["plain_len"]):
                    raise ValueError(f"Payload ROI length mismatch for frame {frame_idx} roi {roi_index}")
                roi_out = roi.copy()
                roi_out[roi_mask] = pix.reshape(-1, 3)
                frame[y1:y2, x1:x2] = roi_out

        # mux video frame
        out_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        try:
            out_frame.pts = frame_idx
            out_frame.time_base = Fraction(1, int(round(fps)) if fps and fps > 0 else 30)
        except Exception:
            pass
        for packet in out_stream.encode(out_frame):
            out_container.mux(packet)

        # audio handling
        if has_audio and aout is not None:
            start = int(round(frame_idx * audio_sr / fps))
            end = int(round((frame_idx + 1) * audio_sr / fps))
            if mode == "encrypt":
                start = max(start, audio_prev_end)
                end = max(end, start)
                end = min(end, int(pcm_all.shape[0]))
                if end > start:
                    seg = np.ascontiguousarray(pcm_all[start:end, :], dtype=np.int16)
                    aseed = base.derive_policy_seed(
                        master_key=master_key,
                        frame_idx=frame_idx,
                        track_id=0,
                        cls_id=-1,
                        map_name=map_name,
                        policy_tag="audio_default",
                        scope="class",
                    )
                    # store recoverable audio in payload
                    audio_plain = zlib.compress(seg.tobytes(), level=PAYLOAD_ZLIB_LEVEL)
                    common_audio = _payload_common_kwargs(frame_idx, 0, -1, "audio_default", None, "", 0)
                    acipher, ameta = _payload_encrypt_bytes(base, audio_plain, aseed, map_name, "audio", common_audio)
                    arec_wo_tag = {
                        "shape": [int(seg.shape[0]), int(seg.shape[1])],
                        "dtype": "int16",
                        "plain_len": int(seg.size * seg.dtype.itemsize),
                        "comp_len": int(len(audio_plain)),
                        "crypto_meta": ameta,
                    }
                    atag = _payload_tag(master_key, {"frame_idx": frame_idx, **arec_wo_tag}, acipher)
                    payload_frame["audio"] = {
                        **arec_wo_tag,
                        "cipher_b64": _b64e(acipher),
                        "tag": atag,
                    }
                    # Do not mix audio payload bytes into the NIST keystream artifact.

                    # Write the actual chaos-encrypted audio segment to the cipher video.
                    seg_out = base._chaos_audio_segment(seg.copy(), aseed, map_name, decrypt=False)
                    audio_pts = base._mux_pcm_segment(out_container, aout, seg_out, audio_sr, audio_layout, audio_pts)
                audio_prev_end = end
            else:
                frame_payload = payload_records.get(frame_idx, {})
                arec = frame_payload.get("audio")
                if arec is not None:
                    common_audio = _payload_common_kwargs(frame_idx, 0, -1, "audio_default", None, "", 0)
                    aseed = base.derive_policy_seed(
                        master_key=master_key,
                        frame_idx=frame_idx,
                        track_id=0,
                        cls_id=-1,
                        map_name=map_name,
                        policy_tag="audio_default",
                        scope="class",
                    )
                    acipher = _b64d(arec["cipher_b64"])
                    arec_wo_tag = {
                        "frame_idx": frame_idx,
                        "shape": arec["shape"],
                        "dtype": arec["dtype"],
                        "plain_len": int(arec["plain_len"]),
                        "comp_len": int(arec["comp_len"]),
                        "crypto_meta": arec["crypto_meta"],
                    }
                    expect = _payload_tag(master_key, arec_wo_tag, acipher)
                    if not hmac.compare_digest(expect, str(arec.get("tag", ""))):
                        raise ValueError(f"Payload authentication failed for frame {frame_idx} audio")
                    plain_comp = _payload_decrypt_bytes(base, acipher, aseed, map_name, arec["crypto_meta"], "audio", common_audio)
                    plain = zlib.decompress(plain_comp)
                    seg = np.frombuffer(plain, dtype=np.int16).reshape(int(arec["shape"][0]), int(arec["shape"][1]))
                    audio_pts = base._mux_pcm_segment(out_container, aout, seg, audio_sr, audio_layout, audio_pts)

        if mode == "encrypt" and payload_f is not None:
            _store_payload_record(payload_f, payload_frame)

        if frame_idx % 30 == 0:
            print(f"frame={frame_idx} rois={len(rois)} map={map_name}")

        frame_idx += 1
        pbar.update(1)

    # tail audio after last frame
    if mode == "encrypt" and has_audio and payload_f is not None and aout is not None and audio_prev_end < int(pcm_all.shape[0]):
        seg = np.ascontiguousarray(pcm_all[audio_prev_end:, :], dtype=np.int16)
        last_fi = max(frame_idx - 1, 0)
        last_map = last_map_name
        aseed = base.derive_policy_seed(
            master_key=master_key,
            frame_idx=last_fi,
            track_id=0,
            cls_id=-1,
            map_name=last_map,
            policy_tag="audio_default",
            scope="class",
        )
        common_audio = _payload_common_kwargs(last_fi, 0, -1, "audio_default", None, "", 0)
        audio_plain = zlib.compress(seg.tobytes(), level=PAYLOAD_ZLIB_LEVEL)
        acipher, ameta = _payload_encrypt_bytes(base, audio_plain, aseed, last_map, "audio", common_audio)
        arec_wo_tag = {
            "type": "tail_audio",
            "frame_idx": last_fi,
            "shape": [int(seg.shape[0]), int(seg.shape[1])],
            "dtype": "int16",
            "plain_len": int(seg.size * seg.dtype.itemsize),
            "comp_len": int(len(audio_plain)),
            "crypto_meta": ameta,
        }
        atag = _payload_tag(master_key, arec_wo_tag, acipher)
        _store_payload_record(payload_f, {**arec_wo_tag, "cipher_b64": _b64e(acipher), "tag": atag})
        # Do not mix tail-audio payload bytes into the NIST keystream artifact.
        seg_out = base._chaos_audio_segment(seg.copy(), aseed, last_map, decrypt=False)
        audio_pts = base._mux_pcm_segment(out_container, aout, seg_out, audio_sr, audio_layout, audio_pts)

    if mode == "decrypt" and has_audio and aout is not None and payload_tail is not None:
        last_fi = int(payload_tail.get("frame_idx", max(frame_idx - 1, 0)))
        last_map = last_map_name
        aseed = base.derive_policy_seed(
            master_key=master_key,
            frame_idx=last_fi,
            track_id=0,
            cls_id=-1,
            map_name=last_map,
            policy_tag="audio_default",
            scope="class",
        )
        common_audio = _payload_common_kwargs(last_fi, 0, -1, "audio_default", None, "", 0)
        acipher = _b64d(payload_tail["cipher_b64"])
        arec_wo_tag = {
            "type": "tail_audio",
            "frame_idx": int(payload_tail["frame_idx"]),
            "shape": payload_tail["shape"],
            "dtype": payload_tail["dtype"],
            "plain_len": int(payload_tail["plain_len"]),
            "comp_len": int(payload_tail["comp_len"]),
            "crypto_meta": payload_tail["crypto_meta"],
        }
        expect = _payload_tag(master_key, arec_wo_tag, acipher)
        if not hmac.compare_digest(expect, str(payload_tail.get("tag", ""))):
            raise ValueError("Payload authentication failed for tail audio")
        plain_comp = _payload_decrypt_bytes(base, acipher, aseed, last_map, payload_tail["crypto_meta"], "audio", common_audio)
        plain = zlib.decompress(plain_comp)
        seg = np.frombuffer(plain, dtype=np.int16).reshape(int(payload_tail["shape"][0]), int(payload_tail["shape"][1]))
        audio_pts = base._mux_pcm_segment(out_container, aout, seg, audio_sr, audio_layout, audio_pts)

    if aout is not None:
        for pkt in aout.encode():
            out_container.mux(pkt)
    for packet in out_stream.encode():
        out_container.mux(packet)

    pbar.close()
    if sidecar_f is not None:
        sidecar_f.close()
    if payload_f is not None:
        payload_f.close()
    if ks_f is not None:
        ks_f.close()
    out_container.close()
    in_container.close()

    if mode == "encrypt" and payload_path and manifest_path:
        out_identity = _video_identity(out_path)

        # The encrypted container can report a wrong/unstable frame count
        # after re-encoding. The payload records are the real processed-frame
        # source of truth for decrypt/recovery, so use them for manifest frames.
        payload_frame_count = _count_payload_frame_records(payload_path)
        if payload_frame_count <= 0:
            payload_frame_count = int(frame_idx)

        # Keep payload header and manifest consistent before hashing payload.
        _patch_payload_header_frames(payload_path, payload_frame_count)
        payload_header, _, _ = _load_payload_records(payload_path)

        manifest = {
            "version": 1,
            "session_id": session_id,
            "width": int(payload_header.get("width", input_identity["width"])),
            "height": int(payload_header.get("height", input_identity["height"])),
            "fps": float(payload_header.get("fps", input_identity["fps"])),
            "frames": int(payload_frame_count),
            "has_audio": bool(has_audio),
            "encrypted_width": int(out_identity["width"]),
            "encrypted_height": int(out_identity["height"]),
            "encrypted_fps": float(out_identity["fps"]),
            "encrypted_frames": int(out_identity["frames"]),
            "encrypted_video_path": os.path.basename(out_path),
            "roi_sidecar_path": os.path.basename(roi_sidecar_path) if roi_sidecar_path else None,
            "payload_path": os.path.basename(payload_path),
            "encrypted_video_sha256": _sha256_file(out_path),
            "roi_sidecar_sha256": _sha256_file(roi_sidecar_path) if roi_sidecar_path else None,
            "payload_sha256": _sha256_file(payload_path),
        }
        _write_manifest(manifest_path, manifest)


# -----------------------------
# Standalone payload-backed chaos runner
# -----------------------------
def process_video(
    in_path: str,
    out_path: str,
    master_key: str,
    mode: str,
    roi_sidecar_path: Optional[str] = "rois.jsonl",
    payload_path: Optional[str] = "payload.jsonl",
    manifest_path: Optional[str] = None,
    detect_width: int = 1280,
    keystream_dump_path: Optional[str] = None,
    reuse_rois: bool = False,
):
    """Standalone wrapper around the payload-backed chaos pipeline.

    Encryption:
      1) detects/tracks ROIs,
      2) stores original ROI/audio in authenticated encrypted payload records,
      3) writes chaos-encrypted ROI/audio into the public cipher video,
      4) writes sidecar + manifest for authorized decryption.

    Decryption:
      1) verifies the manifest/payload/sidecar binding,
      2) reads ROI/audio payload records,
      3) restores original ROI/audio content into the output video.
    """
    import sys as _sys
    return _process_payload_video(
        base=_sys.modules[__name__],
        in_path=in_path,
        out_path=out_path,
        master_key=master_key,
        mode=mode,
        roi_sidecar_path=roi_sidecar_path,
        payload_path=payload_path,
        manifest_path=manifest_path,
        detect_width=detect_width,
        keystream_dump_path=keystream_dump_path,
        reuse_rois=bool(reuse_rois),
    )


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Standalone Chaos ROI video encryption with sidecar, encrypted payload, manifest, and recovery"
    )
    ap.add_argument("--mode", choices=["encrypt", "decrypt"], required=True,
                    help="encrypt creates cipher video + ROI sidecar + encrypted payload; decrypt restores using sidecar/payload/manifest")
    ap.add_argument("--in", dest="inp", required=True, help="input video path")
    ap.add_argument("--out", required=True, help="output video path")
    ap.add_argument("--key", required=True, help="master key string used for per-ROI and per-audio key derivation")
    ap.add_argument("--roi_sidecar", default="rois.jsonl",
                    help="ROI sidecar path: written during encrypt, required during decrypt")
    ap.add_argument("--payload", default="payload.jsonl",
                    help="encrypted payload path: written during encrypt, required during decrypt")
    ap.add_argument("--manifest", default=None,
                    help="manifest path; defaults to <payload>.manifest.json")
    ap.add_argument("--detect_width", type=int, default=_env_int("DETECT_WIDTH", 1280),
                    help="resize video width before YOLO; 0 keeps full resolution; env DETECT_WIDTH sets default")
    ap.add_argument("--reuse_rois", action="store_true",
                    help="encrypt only: reuse an existing ROI sidecar and skip YOLO detection/tracking")
    ap.add_argument("--keystream_dump", default=None,
                    help="encrypt only: dump encrypted payload bytes for NIST/ciphertext testing")
    ap.add_argument("--policy_scope", choices=["class", "object"], default=POLICY_SCOPE,
                    help="policy keying scope: per-class or per-object")

    args = ap.parse_args()
    POLICY_SCOPE = args.policy_scope

    cap = cv2.VideoCapture(args.inp)
    _w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    _h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    _fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    print(f"Resolution: {_w} x {_h}")
    print(f"FPS: {_fps}")
    print(f"[INFO] CUDA_OK: {CUDA_OK}")
    print(f"[INFO] YOLO_DEVICE: {YOLO_DEVICE}, YOLO_HALF: {YOLO_HALF}")
    print(f"[INFO] COCO_MODEL: {COCO_MODEL}")
    print(f"[INFO] TRACKER_CFG: {TRACKER_CFG}, TRACK_PERSIST: {TRACK_PERSIST}, RETINA_MASKS: {RETINA_MASKS}")
    print(f"[INFO] POLICY_SCOPE: {POLICY_SCOPE}")
    print("[INFO] Preview logic: removed; public output ROIs/audio are chaos-ciphered")
    if CUDA_OK:
        print(f"[INFO] CUDA device: {torch.cuda.get_device_name(0)}")

    process_video(
        in_path=args.inp,
        out_path=args.out,
        master_key=args.key,
        mode=args.mode,
        roi_sidecar_path=args.roi_sidecar,
        payload_path=args.payload,
        manifest_path=args.manifest,
        detect_width=args.detect_width,
        keystream_dump_path=args.keystream_dump,
        reuse_rois=bool(args.reuse_rois),
    )

    print(f"[INFO] Done. Wrote: {args.out}")
