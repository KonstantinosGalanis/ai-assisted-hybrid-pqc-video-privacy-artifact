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
COCO_MODEL = os.environ.get("COCO_MODEL", "yolo26x-seg.pt")

# COCO ids: person=0, car=2, motorbike=3, bus=5, truck=7
COCO_CLASSES_OF_INTEREST = {0, 2, 3, 5, 7}

CONF_THRES = 0.10
IOU_THRES = 0.50

ROI_PAD = 6

# Keep small to avoid "fat" silhouettes
MASK_DILATE_PX = 1  # try 0..2

DETERMINISTIC_ROI_SORT = True

# Tracking (Ultralytics)
TRACKER_CFG = "custom_botsort_privacy_v2.yaml"  # botsort.yaml / bytetrack.yaml
TRACK_PERSIST = True
RETINA_MASKS = True

# Policy keying
POLICY_SCOPE = "class"  # "class" or "object"

# Debug overlay (draw bboxes/text into output video)
DRAW_DEBUG_OVERLAY = False
DEBUG_FONT_SCALE = 0.8
DEBUG_THICKNESS = 2
DEBUG_COLOR = (0, 255, 255)  # yellow-ish (BGR)

# Extra debug prints
ENABLE_DEBUG_PRINTS = False
DEBUG_EVERY_N_FRAMES = 1            # print every frame; change to 10/30 if too verbose
DEBUG_ONLY_CLASSES = None           # e.g. {2,3,5,7} for vehicles only, or None for all
DEBUG_ONLY_TRACK_IDS = None         # e.g. {5,7}, or None
DEBUG_PRINT_RAW_DETECTIONS = False
DEBUG_PRINT_CANDIDATE_FILTERS = False
DEBUG_PRINT_MASK_STATS = False
DEBUG_PRINT_FINAL_ROIS = False
DEBUG_PRINT_APPLY_STAGE = False
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

def _chunked_chaos_stream(seed: bytes, map_name: str, total_n: int, label: str, chunk_size: int) -> np.ndarray:
    total_n = int(total_n)
    chunk_size = max(1, int(chunk_size))
    out = np.empty(total_n, dtype=np.uint8)
    off = 0
    chunk_idx = 0
    while off < total_n:
        m = min(chunk_size, total_n - off)
        chunk_seed = derive_stream_seed(seed, f"{label}|chunk={chunk_idx}")
        out[off:off + m] = keystream_u8(chunk_seed, map_name, m)
        off += m
        chunk_idx += 1
    return out

def _chunked_perm_scores(seed: bytes, map_name: str, total_n: int, chunk_size: int) -> np.ndarray:
    total_n = int(total_n)
    chunk_size = max(1, int(chunk_size))
    out = np.empty(total_n, dtype=np.uint32)
    off = 0
    chunk_idx = 0
    while off < total_n:
        m = min(chunk_size, total_n - off)
        chunk_seed = derive_stream_seed(seed, f"perm|chunk={chunk_idx}")
        raw = np.asarray(keystream_u8(chunk_seed, map_name, m * 4), dtype=np.uint8)
        out[off:off + m] = raw.view(np.uint32)
        off += m
        chunk_idx += 1
    return out

def _chaos_sbox(seed: bytes, map_name: str) -> Tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(keystream_u8(derive_stream_seed(seed, "sbox"), map_name, 256 * 4), dtype=np.uint8)
    scores = raw.view(np.uint32)
    sbox = np.argsort(scores, kind="mergesort").astype(np.uint8)
    inv = np.empty(256, dtype=np.uint8)
    inv[sbox] = np.arange(256, dtype=np.uint8)
    return sbox, inv

def _masked_byte_positions(mask: np.ndarray) -> np.ndarray:
    mask3 = np.repeat(np.asarray(mask, dtype=bool).reshape(-1), 3)
    return np.flatnonzero(mask3)

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
        "scheme": "chaos_psd_v2_local",
        "stream_nonce_b64": _b64e(nonce_a),
        "tags_b64": _b64e(nonce_b),
        "chunk_size": int(max(256, chunk_size)),
        "block_size": int(max(128, block_size)),
    }

@njit(cache=False, fastmath=True)
def _psd_encrypt_core(v: np.ndarray, sbox: np.ndarray, k1: np.ndarray, k2: np.ndarray, iv0: int, iv1: int) -> np.ndarray:
    n = int(v.size)
    u = np.empty(n, dtype=np.uint8)
    prev_u = int(iv0)
    for i in range(n):
        vv = (int(v[i]) + int(k1[i]) + prev_u) & 255
        uu = int(sbox[vv])
        u[i] = np.uint8(uu)
        prev_u = uu
    c = np.empty(n, dtype=np.uint8)
    next_c = int(iv1)
    for i in range(n - 1, -1, -1):
        cc = (int(u[i]) + int(k2[i]) + next_c) & 255
        c[i] = np.uint8(cc)
        next_c = cc
    return c

@njit(cache=False, fastmath=True)
def _psd_decrypt_core(c: np.ndarray, inv_sbox: np.ndarray, perm: np.ndarray, k1: np.ndarray, k2: np.ndarray, iv0: int, iv1: int) -> np.ndarray:
    n = int(c.size)
    u = np.empty(n, dtype=np.uint8)
    next_c = int(iv1)
    for i in range(n - 1, -1, -1):
        uu = (int(c[i]) - int(k2[i]) - next_c) & 255
        u[i] = np.uint8(uu)
        next_c = int(c[i])
    v = np.empty(n, dtype=np.uint8)
    prev_u = int(iv0)
    for i in range(n):
        vv = (int(inv_sbox[int(u[i])]) - int(k1[i]) - prev_u) & 255
        v[i] = np.uint8(vv)
        prev_u = int(u[i])
    p = np.empty(n, dtype=np.uint8)
    for i in range(n):
        p[int(perm[i])] = v[i]
    return p

def _chaos_psd_transform(
    data_u8: np.ndarray,
    seed: bytes,
    map_name: str,
    nonce_a: bytes,
    nonce_b: bytes,
    chunk_size: int,
    decrypt: bool,
    block_size: int = 1024,
) -> np.ndarray:
    n = int(data_u8.size)
    if n <= 0:
        return np.asarray(data_u8, dtype=np.uint8).copy()

    base = np.asarray(data_u8, dtype=np.uint8)
    out = np.empty_like(base)
    core_seed = hmac.new(seed, nonce_a + nonce_b, hashlib.sha256).digest()
    block_size = int(max(64, block_size))
    chunk_size = int(max(128, chunk_size))

    off = 0
    blk = 0
    while off < n:
        m = min(block_size, n - off)
        block_seed = derive_stream_seed(core_seed, f"block={blk}")
        perm_scores = _chunked_perm_scores(derive_stream_seed(block_seed, "perm"), map_name, m, min(chunk_size, m))
        perm = np.argsort(perm_scores, kind="mergesort").astype(np.int64, copy=False)
        sbox, inv_sbox = _chaos_sbox(derive_stream_seed(block_seed, "sub"), map_name)
        k1 = _chunked_chaos_stream(derive_stream_seed(block_seed, "diff1"), map_name, m, "d1", min(chunk_size, m))
        k2 = _chunked_chaos_stream(derive_stream_seed(block_seed, "diff2"), map_name, m, "d2", min(chunk_size, m))
        ivs = np.asarray(keystream_u8(derive_stream_seed(block_seed, "ivs"), map_name, 2), dtype=np.uint8)
        if not decrypt:
            v = base[off:off + m][perm]
            out[off:off + m] = _psd_encrypt_core(v, sbox, k1, k2, int(ivs[0]), int(ivs[1]))
        else:
            c = base[off:off + m]
            out[off:off + m] = _psd_decrypt_core(c, inv_sbox, perm, k1, k2, int(ivs[0]), int(ivs[1]))
        off += m
        blk += 1

    return out

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
            "scheme": "chaos_psd_v2_local",
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
    chunk_size = int(crypto_meta.get("chunk_size", 1024) or 1024)
    block_size = int(crypto_meta.get("block_size", 1024) or 1024)

    pix = np.ascontiguousarray(roi_bgr[mask], dtype=np.uint8)
    data = pix.reshape(-1)
    enc = _chaos_psd_transform(data, core_seed, map_name, nonce_a, nonce_b, chunk_size, decrypt=False, block_size=block_size)
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
    chunk_size = int(crypto_meta.get("chunk_size", 256) or 256)
    block_size = int(crypto_meta.get("block_size", 1024) or 1024)

    pix = np.ascontiguousarray(roi_bgr[mask], dtype=np.uint8)
    data = pix.reshape(-1)
    dec = _chaos_psd_transform(data, core_seed, map_name, nonce_a, nonce_b, chunk_size, decrypt=True, block_size=block_size)
    roi_bgr[mask] = dec.reshape(-1, 3)
    return roi_bgr

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
# ROI crypto on masked pixels (FAST PATH)
# -----------------------------

@njit(cache=False, fastmath=True)
def _burn_chen(burn: int, x: float, y: float, z: float, A: float, B: float, C: float, M: float):
    for _ in range(burn):
        x = (A * (y - x)) % M
        y = (((C - A) * x) - (x * z) + (C * y)) % M
        z = ((x * y) - (B * z)) % M
    return x, y, z

@njit(cache=False, fastmath=True)
def _step_chen_u8(x: float, y: float, z: float, A: float, B: float, C: float, M: float):
    x = (A * (y - x)) % M
    y = (((C - A) * x) - (x * z) + (C * y)) % M
    z = ((x * y) - (B * z)) % M
    v = int((x / M) * 256.0) & 255
    return x, y, z, v

@njit(cache=False, fastmath=True)
def _burn_cubic(burn: int, x: float, u: float):
    for _ in range(burn):
        x = (u * x * (1.0 - x * x)) % 1.0
    return x

@njit(cache=False, fastmath=True)
def _step_cubic_u8(x: float, u: float):
    x = (u * x * (1.0 - x * x)) % 1.0
    v = int(x * 256.0) & 255
    return x, v

@njit(cache=False, fastmath=True)
def _burn_skew_tent(burn: int, x: float, p: float):
    for _ in range(burn):
        if x < p:
            x = x / p
        else:
            x = (1.0 - x) / (1.0 - p)
        x = x % 1.0
    return x

@njit(cache=False, fastmath=True)
def _step_skew_tent_u8(x: float, p: float):
    if x < p:
        x = x / p
    else:
        x = (1.0 - x) / (1.0 - p)
    x = x % 1.0
    v = int(x * 256.0) & 255
    return x, v

@njit(cache=False, fastmath=True)
def _xor_masked_roi_chen_inplace(roi_bgr: np.ndarray, mask: np.ndarray,
                                 x: float, y: float, z: float,
                                 A: float, B: float, C: float, M: float,
                                 burn: int):
    x, y, z = _burn_chen(burn, x, y, z, A, B, C, M)
    h, w = mask.shape
    for yy in range(h):
        for xx in range(w):
            if mask[yy, xx]:
                x, y, z, k0 = _step_chen_u8(x, y, z, A, B, C, M)
                x, y, z, k1 = _step_chen_u8(x, y, z, A, B, C, M)
                x, y, z, k2 = _step_chen_u8(x, y, z, A, B, C, M)
                roi_bgr[yy, xx, 0] ^= np.uint8(k0)
                roi_bgr[yy, xx, 1] ^= np.uint8(k1)
                roi_bgr[yy, xx, 2] ^= np.uint8(k2)

@njit(cache=False, fastmath=True)
def _xor_masked_roi_cubic_inplace(roi_bgr: np.ndarray, mask: np.ndarray,
                                  x: float, u: float,
                                  burn: int):
    x = _burn_cubic(burn, x, u)
    h, w = mask.shape
    for yy in range(h):
        for xx in range(w):
            if mask[yy, xx]:
                x, k0 = _step_cubic_u8(x, u)
                x, k1 = _step_cubic_u8(x, u)
                x, k2 = _step_cubic_u8(x, u)
                roi_bgr[yy, xx, 0] ^= np.uint8(k0)
                roi_bgr[yy, xx, 1] ^= np.uint8(k1)
                roi_bgr[yy, xx, 2] ^= np.uint8(k2)

@njit(cache=False, fastmath=True)
def _xor_masked_roi_skew_tent_inplace(roi_bgr: np.ndarray, mask: np.ndarray,
                                      x: float, p: float,
                                      burn: int):
    x = _burn_skew_tent(burn, x, p)
    h, w = mask.shape
    for yy in range(h):
        for xx in range(w):
            if mask[yy, xx]:
                x, k0 = _step_skew_tent_u8(x, p)
                x, k1 = _step_skew_tent_u8(x, p)
                x, k2 = _step_skew_tent_u8(x, p)
                roi_bgr[yy, xx, 0] ^= np.uint8(k0)
                roi_bgr[yy, xx, 1] ^= np.uint8(k1)
                roi_bgr[yy, xx, 2] ^= np.uint8(k2)

def _seed_init_for_map(seed: bytes, map_name: str):
    if map_name == "chen":
        M = 0.99
        x = _seed_to_unit(seed, 0) * M
        y = _seed_to_unit(seed, 1) * M
        z = _seed_to_unit(seed, 2) * M
        return ("chen", x, y, z)
    if map_name == "cubic":
        x = _seed_to_unit(seed, 0)
        return ("cubic", x)
    if map_name == "skew_tent":
        x = _seed_to_unit(seed, 0)
        return ("skew_tent", x)
    raise ValueError(map_name)

def encrypt_masked_bytes_cpu(roi_bgr: np.ndarray, mask: np.ndarray, seed: bytes, map_name: str) -> np.ndarray:
    # Deterministic fallback path kept for legacy callers.
    roi_out, _ = _encrypt_masked_bytes_cpu_chaos(
        roi_bgr, mask, seed, map_name,
        frame_idx=0, track_id=-1, cls_id=-1,
        policy_tag="default_sensitive",
        bbox=None, mask_hash="", roi_index=0,
        crypto_meta={
            "scheme": "chaos_psd_v1",
            "stream_nonce_b64": _b64e(bytes(16)),
            "tags_b64": _b64e(bytes(16)),
            "chunk_size": 4096,
        },
    )
    return roi_out

def decrypt_masked_bytes_cpu(roi_bgr: np.ndarray, mask: np.ndarray, seed: bytes, map_name: str) -> np.ndarray:
    return _decrypt_masked_bytes_cpu_chaos(
        roi_bgr, mask, seed, map_name,
        crypto_meta={
            "scheme": "chaos_psd_v1",
            "stream_nonce_b64": _b64e(bytes(16)),
            "tags_b64": _b64e(bytes(16)),
            "chunk_size": 4096,
        },
        frame_idx=0, track_id=-1, cls_id=-1,
        policy_tag="default_sensitive",
        bbox=None, mask_hash="", roi_index=0,
    )

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
            classes=list(COCO_CLASSES_OF_INTEREST),
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
# Lossless writer using PyAV (FFV1)
# -----------------------------

def _safe_rate(fps: float) -> Fraction:
    if fps is None or not np.isfinite(fps) or fps <= 0:
        return Fraction(30, 1)
    return Fraction(fps).limit_denominator(100000)

def open_lossless_writer(out_path: str, width: int, height: int, fps: float):
    out = av.open(out_path, mode="w", format="mp4")   # use .mkv or .mp4 in the filename as you prefer
    rate = _safe_rate(fps)

    st = out.add_stream("libx264rgb", rate=rate)
    st.width = width
    st.height = height
    st.pix_fmt = "bgr24"

    st.options = {
        "preset": "slower", #medium, slowe. slower
        "qp": "0",
        "x264-params": "keyint=10000:min-keyint=10000:scenecut=0:bframes=0"
    }

    return out, st

# -----------------------------
# Debug drawing
# -----------------------------

def _draw_debug_overlay(frame_bgr: np.ndarray, frame_idx: int, pict_str: str, map_name: str, rois: List[dict]) -> None:
    if not DRAW_DEBUG_OVERLAY:
        return

    H, W = frame_bgr.shape[:2]
    cv2.rectangle(frame_bgr, (1, 1), (W - 2, H - 2), DEBUG_COLOR, 2)

    header = f"frame={frame_idx} pict={pict_str} map={map_name} rois={len(rois)}"
    cv2.putText(frame_bgr, header, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                DEBUG_FONT_SCALE, DEBUG_COLOR, DEBUG_THICKNESS, cv2.LINE_AA)

    y = 56
    for i, r in enumerate(rois[:10]):
        x1, y1, x2, y2 = map(int, r["bbox"])
        cls_id = int(r.get("cls", -1))
        tid = int(r.get("track_id", -1))
        conf = float(r.get("conf", 0.0))
        policy_tag = str(r.get("policy_tag", "n/a"))

        cv2.rectangle(frame_bgr, (x1, y1), (x2 - 1, y2 - 1), DEBUG_COLOR, 2)

        label = f"{i}:cls={cls_id} tid={tid} conf={conf:.2f} pol={policy_tag} [{x1},{y1},{x2},{y2}]"
        cv2.putText(frame_bgr, label, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, DEBUG_COLOR, 2, cv2.LINE_AA)
        y += 22

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



def _add_pcm_audio_stream(out_container: av.container.OutputContainer, sr: int, layout_name: str):
    aout = out_container.add_stream("flac", rate=sr)
    aout.layout = layout_name
    return aout

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
    transformed = _chaos_psd_transform(
        raw.copy(),
        derive_stream_seed(seed, "audio-psd"),
        map_name,
        nonce_a,
        nonce_b,
        chunk_size,
        decrypt=decrypt,
    )
    raw[:] = transformed
    return seg

# -----------------------------
# Core video transform
# -----------------------------

def process_video(
    in_path: str,
    out_path: str,
    master_key: str,
    mode: str,  # "encrypt" or "decrypt"
    roi_sidecar_path: Optional[str] = None,
    reuse_rois: bool = False,
    detect_every: int = 1,
    detect_width: int = 1280,
    keystream_dump_path: Optional[str] = None,
):
    assert mode in {"encrypt", "decrypt"}

    if mode == "encrypt" and detect_every > 1:
        print("[WARN] detect_every > 1 is ignored in encrypt mode when tracking is enabled.")
        print("[WARN] Tracking requires per-frame updates; encryption will track every frame.")

    det = load_detectors() if mode == "encrypt" else None

    if mode == "encrypt" and det is not None:
        print("[INFO] Warmup YOLO + Numba...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)

        with torch.inference_mode():
            try:
                _ = det.coco.track(
                    dummy,
                    device=YOLO_DEVICE,
                    half=YOLO_HALF,
                    verbose=False,
                    persist=TRACK_PERSIST,
                    tracker=TRACKER_CFG,
                    classes=list(COCO_CLASSES_OF_INTEREST),
                )
            except Exception:
                _ = det.coco.predict(
                    dummy,
                    device=YOLO_DEVICE,
                    half=YOLO_HALF,
                    verbose=False
                )

        _ = cubic_keystream_u8(b"warmup", 64)
        _ = skew_tent_keystream_u8(b"warmup", 64)
        _ = chen_keystream_u8(b"warmup", 64)

    in_container = av.open(in_path)
    in_stream = in_container.streams.video[0]
    in_stream.thread_type = "AUTO"

    audio_container = av.open(in_path)
    audio_info = _decode_audio_pcm_s16(audio_container)
    audio_container.close()

    has_audio = audio_info is not None
    if has_audio:
        pcm_all, audio_sr, audio_layout = audio_info
    else:
        pcm_all, audio_sr, audio_layout = None, None, None


    fps = float(in_stream.average_rate) if in_stream.average_rate else 30.0
    width = in_stream.codec_context.width
    height = in_stream.codec_context.height

    out_container, out_stream = open_lossless_writer(out_path, width, height, fps)
    aout = _add_pcm_audio_stream(out_container, audio_sr, audio_layout) if has_audio else None

    audio_pts = 0
    audio_prev_end = 0

    sidecar_f = None
    if roi_sidecar_path and (mode == "encrypt"):
        sidecar_f = open(roi_sidecar_path, "w", encoding="utf-8")

    roi_records: Dict[int, dict] = {}
    if mode == "decrypt" and reuse_rois:
        if not roi_sidecar_path:
            raise ValueError("Decrypt requires roi_sidecar_path")
        with open(roi_sidecar_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                roi_records[int(rec["frame_idx"])] = rec

    total_frames = in_stream.frames if in_stream.frames else None
    pbar = tqdm(total=total_frames, desc=f"{mode} frames", unit="frame")

    frame_idx = 0
    last_map_name = "cubic"

    pack_for_sidecar = (sidecar_f is not None)

    ks_f = None
    if mode == "encrypt" and keystream_dump_path:
        ks_f = open(keystream_dump_path, "wb")
        print(f"[INFO] Cipher-payload dump enabled (for NIST/ciphertext testing): {keystream_dump_path}")

    for av_frame in in_container.decode(in_stream):
        t0 = time.perf_counter()

        frame = av_frame.to_ndarray(format="bgr24")
        H, W = frame.shape[:2]

        pict_raw = av_frame.pict_type
        pict_str = getattr(pict_raw, "name", str(pict_raw))
        map_name = map_for_pict_type(pict_raw)
        last_map_name = map_name

        if mode == "decrypt" and reuse_rois:
            rec = roi_records.get(frame_idx, {"rois": [], "map_name": map_name})
            map_name = str(rec.get("map_name", map_name))
            rois = rec.get("rois", [])
            t1 = t0
        else:
            rois = detect_rois_with_masks(
                det,
                frame,
                frame_idx=frame_idx,
                detect_width=detect_width,
                pack_for_sidecar=pack_for_sidecar
            )
            t1 = time.perf_counter()

        if DEBUG_PRINT_FINAL_ROIS:
            _dbg(frame_idx, f"\n[DBG][frame={frame_idx}] final rois returned: {len(rois)}")
            for k, r in enumerate(rois):
                roi_mask_dbg = r.get("mask", None)
                mask_pixels_dbg = int(roi_mask_dbg.sum()) if roi_mask_dbg is not None else -1
                _dbg(
                    frame_idx,
                    f"  ROI[{k}] cls={int(r.get('cls', -1))} "
                    f"tid={int(r.get('track_id', -1))} "
                    f"conf={float(r.get('conf', 0.0)):.3f} "
                    f"bbox={r['bbox']} "
                    f"mask_pixels={mask_pixels_dbg}",
                    cls_id=int(r.get("cls", -1)),
                    track_id=int(r.get("track_id", -1)),
                )

        if mode == "encrypt":
            for roi_index, r in enumerate(rois):
                r["roi_index"] = int(r.get("roi_index", roi_index))
                cls_id = int(r.get("cls", -1))
                track_id = int(r.get("track_id", -1))
                policy_tag = str(r.get("policy_tag", policy_tag_for_class(cls_id)))
                r["policy_tag"] = policy_tag
                bbox0 = tuple(int(v) for v in r["bbox"])
                roi_mask0 = r.get("mask", None)
                if roi_mask0 is None:
                    roi_mask0 = unpack_mask(r["mask_pack"])
                mask_hash0 = str(r.get("mask_hash", _compute_mask_hash(roi_mask0)))
                r["mask_hash"] = mask_hash0
                seed0 = derive_policy_seed(
                    master_key=master_key,
                    frame_idx=frame_idx,
                    track_id=track_id,
                    cls_id=cls_id,
                    map_name=map_name,
                    policy_tag=policy_tag,
                    scope=POLICY_SCOPE,
                    bbox=bbox0,
                    mask_hash=mask_hash0,
                    roi_index=int(r["roi_index"]),
                )
                if r.get("crypto_meta") is None:
                    r["crypto_meta"] = _prepare_chaos_crypto_meta(
                        seed0,
                        map_name,
                        frame_idx=frame_idx,
                        track_id=track_id,
                        cls_id=cls_id,
                        policy_tag=policy_tag,
                        bbox=bbox0,
                        mask_hash=mask_hash0,
                        roi_index=int(r["roi_index"]),
                    )

        if sidecar_f is not None:
            rois_for_json = []
            for roi_index, r in enumerate(rois):
                rp = {
                    "bbox": r["bbox"],
                    "mask_pack": r.get("mask_pack") if r.get("mask_pack") is not None else pack_mask(r["mask"]),
                    "cls": int(r.get("cls", -1)),
                    "conf": float(r.get("conf", 0.0)),
                    "track_id": int(r.get("track_id", -1)),
                    "policy_tag": str(r.get("policy_tag", policy_tag_for_class(int(r.get("cls", -1))))),
                    "roi_index": int(r.get("roi_index", roi_index)),
                    "mask_hash": str(r.get("mask_hash", "")),
                    "crypto_meta": r.get("crypto_meta"),
                }
                rois_for_json.append(rp)

            sidecar_f.write(json.dumps({
                "frame_idx": frame_idx,
                "pict": pict_str,
                "map_name": map_name,
                "rois": rois_for_json
            }) + "\n")

        rois_to_apply = rois if mode == "encrypt" else list(reversed(rois))

        t_enc0 = time.perf_counter()
        for r in rois_to_apply:
            x1, y1, x2, y2 = map(int, r["bbox"])

            if DEBUG_PRINT_APPLY_STAGE:
                _dbg(
                    frame_idx,
                    f"[DBG][frame={frame_idx}] APPLY ROI "
                    f"cls={int(r.get('cls', -1))} tid={int(r.get('track_id', -1))} "
                    f"bbox={r['bbox']}",
                    cls_id=int(r.get("cls", -1)),
                    track_id=int(r.get("track_id", -1)),
                )

            x1 = clamp(x1, 0, W - 1)
            y1 = clamp(y1, 0, H - 1)
            x2 = clamp(x2, 1, W)
            y2 = clamp(y2, 1, H)

            if x2 <= x1 or y2 <= y1:
                if DEBUG_PRINT_APPLY_STAGE:
                    _dbg(
                        frame_idx,
                        "  -> SKIP apply reason=invalid_bbox_after_clamp",
                        cls_id=int(r.get("cls", -1)),
                        track_id=int(r.get("track_id", -1)),
                    )
                continue

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                if DEBUG_PRINT_APPLY_STAGE:
                    _dbg(
                        frame_idx,
                        "  -> SKIP apply reason=roi_empty",
                        cls_id=int(r.get("cls", -1)),
                        track_id=int(r.get("track_id", -1)),
                    )
                continue

            if DEBUG_PRINT_APPLY_STAGE:
                _dbg(
                    frame_idx,
                    f"  roi_shape={roi.shape}",
                    cls_id=int(r.get("cls", -1)),
                    track_id=int(r.get("track_id", -1)),
                )

            roi_mask = r.get("mask", None)
            if roi_mask is None:
                roi_mask = unpack_mask(r["mask_pack"])

            if roi_mask.shape[:2] != (roi.shape[0], roi.shape[1]):
                old_shape = roi_mask.shape[:2]
                roi_mask = cv2.resize(
                    roi_mask.astype(np.uint8),
                    (roi.shape[1], roi.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                if DEBUG_PRINT_APPLY_STAGE:
                    _dbg(
                        frame_idx,
                        f"  resized mask_shape from={old_shape} to={roi_mask.shape}",
                        cls_id=int(r.get("cls", -1)),
                        track_id=int(r.get("track_id", -1)),
                    )

            if DEBUG_PRINT_APPLY_STAGE:
                _dbg(
                    frame_idx,
                    f"  mask_shape={roi_mask.shape} mask_pixels={int(roi_mask.sum())}",
                    cls_id=int(r.get("cls", -1)),
                    track_id=int(r.get("track_id", -1)),
                )

            if roi_mask.sum() == 0:
                if DEBUG_PRINT_APPLY_STAGE:
                    _dbg(
                        frame_idx,
                        "  -> SKIP apply reason=mask_empty",
                        cls_id=int(r.get("cls", -1)),
                        track_id=int(r.get("track_id", -1)),
                    )
                continue

            track_id = int(r.get("track_id", -1))
            cls_id = int(r.get("cls", -1))
            policy_tag = str(r.get("policy_tag", policy_tag_for_class(cls_id)))
            roi_index = int(r.get("roi_index", 0))
            mask_hash = str(r.get("mask_hash", _compute_mask_hash(roi_mask)))

            seed = derive_policy_seed(
                master_key=master_key,
                frame_idx=frame_idx,
                track_id=track_id,
                cls_id=cls_id,
                map_name=map_name,
                policy_tag=policy_tag,
                scope=POLICY_SCOPE,
                bbox=(x1, y1, x2, y2),
                mask_hash=mask_hash,
                roi_index=roi_index,
            )

            common_kwargs = dict(
                frame_idx=frame_idx,
                track_id=track_id,
                cls_id=cls_id,
                policy_tag=policy_tag,
                bbox=(x1, y1, x2, y2),
                mask_hash=mask_hash,
                roi_index=roi_index,
            )

            if mode == "encrypt":
                crypto_meta = r.get("crypto_meta")
                roi_out, crypto_meta = _encrypt_masked_bytes_cpu_chaos(
                    roi, roi_mask, seed, map_name, crypto_meta=crypto_meta, **common_kwargs
                )
                r["crypto_meta"] = crypto_meta
                r["mask_hash"] = mask_hash
                if ks_f is not None:
                    cipher_payload = np.ascontiguousarray(roi_out[roi_mask], dtype=np.uint8).reshape(-1)
                    if cipher_payload.size > 0:
                        ks_f.write(cipher_payload.tobytes())
            else:
                if r.get("crypto_meta") is not None:
                    roi_out = _decrypt_masked_bytes_cpu_chaos(
                        roi, roi_mask, seed, map_name, crypto_meta=r.get("crypto_meta"), **common_kwargs
                    )
                else:
                    roi_out = decrypt_masked_bytes_cpu(roi, roi_mask, seed, map_name)

            if roi_out is not roi:
                frame[y1:y2, x1:x2] = roi_out

            if DEBUG_PRINT_APPLY_STAGE:
                _dbg(
                    frame_idx,
                    "  -> APPLIED encryption",
                    cls_id=cls_id,
                    track_id=track_id,
                )

        if DRAW_DEBUG_OVERLAY:
            _draw_debug_overlay(frame, frame_idx, pict_str, map_name, rois)
        t_enc1 = time.perf_counter()

        t_write0 = time.perf_counter()

        out_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        for packet in out_stream.encode(out_frame):
            out_container.mux(packet)

        if has_audio:
            start = int(round(frame_idx * audio_sr / fps))
            end = int(round((frame_idx + 1) * audio_sr / fps))

            start = max(start, audio_prev_end)
            end = max(end, start)
            end = min(end, int(pcm_all.shape[0]))

            if end > start:
                seg = pcm_all[start:end, :]

                aseed = derive_policy_seed(
                    master_key=master_key,
                    frame_idx=frame_idx,
                    track_id=0,
                    cls_id=-1,
                    map_name=map_name,
                    policy_tag="audio_default",
                    scope="class",
                )

                seg_out = _chaos_audio_segment(seg, aseed, map_name, decrypt=(mode == "decrypt"))

                audio_pts = _mux_pcm_segment(
                    out_container,
                    aout,
                    seg_out,
                    audio_sr,
                    audio_layout,
                    audio_pts
                )

            audio_prev_end = end

        t_write1 = time.perf_counter()

        if frame_idx % 30 == 0:
            print(
                f"frame={frame_idx} "
                f"yolo={(t1 - t0):.3f}s "
                f"crypt={(t_enc1 - t_enc0):.3f}s "
                f"write={(t_write1 - t_write0):.3f}s "
                f"rois={len(rois)} map={map_name}"
            )

        frame_idx += 1
        pbar.update(1)

    if has_audio:
        if audio_prev_end < int(pcm_all.shape[0]):
            seg = pcm_all[audio_prev_end:, :]
            last_fi = max(frame_idx - 1, 0)
            last_map = last_map_name
            aseed = derive_policy_seed(
                master_key=master_key,
                frame_idx=last_fi,
                track_id=0,
                cls_id=-1,
                map_name=last_map,
                policy_tag="audio_default",
                scope="class",
            )

            seg_out = _chaos_audio_segment(seg, aseed, last_map, decrypt=(mode == "decrypt"))
            audio_pts = _mux_pcm_segment(out_container, aout, seg_out, audio_sr, audio_layout, audio_pts)

        for pkt in aout.encode():
            out_container.mux(pkt)

    for packet in out_stream.encode():
        out_container.mux(packet)

    pbar.close()
    if sidecar_f is not None:
        sidecar_f.close()

    if ks_f is not None:
        ks_f.close()


    out_container.close()
    in_container.close()

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["encrypt", "decrypt"], required=True)
    ap.add_argument("--in", dest="inp", required=True, help="input video path")
    ap.add_argument("--out", required=True, help="output video path (recommend .mkv for ffv1)")
    ap.add_argument("--key", required=True, help="master key (string)")
    ap.add_argument("--roi_sidecar", default="rois.jsonl", help="path to save/load ROI masks (.jsonl)")
    ap.add_argument("--detect_every", type=int, default=1, help="kept for compat; ignored in encrypt with tracking")
    ap.add_argument("--detect_width", type=int, default=0)
    ap.add_argument("--keystream_dump", default=None, help="(encrypt only) compatibility name: path to append encrypted ROI payload bytes for NIST/ciphertext testing, e.g. cipher_payload.bin")
    ap.add_argument("--policy_scope", choices=["class", "object"], default=POLICY_SCOPE,
                    help="policy keying scope: per-class or per-object")

    args = ap.parse_args()

    POLICY_SCOPE = args.policy_scope

    cap = cv2.VideoCapture(args.inp)
    _w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    _h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    _fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"Resolution: {_w} x {_h}")
    print(f"FPS: {_fps}")

    print(f"[INFO] CUDA_OK: {CUDA_OK}")
    print(f"[INFO] YOLO_DEVICE: {YOLO_DEVICE}, YOLO_HALF: {YOLO_HALF}")
    print(f"[INFO] ROI crypto DEVICE: {DEVICE}")
    print(f"[INFO] COCO_MODEL: {COCO_MODEL}")
    print(f"[INFO] TRACKER_CFG: {TRACKER_CFG}, TRACK_PERSIST: {TRACK_PERSIST}, RETINA_MASKS: {RETINA_MASKS}")
    print(f"[INFO] POLICY_SCOPE: {POLICY_SCOPE}")
    print(f"[INFO] ENABLE_DEBUG_PRINTS: {ENABLE_DEBUG_PRINTS}")
    print(f"[INFO] DEBUG_EVERY_N_FRAMES: {DEBUG_EVERY_N_FRAMES}")
    print(f"[INFO] DEBUG_ONLY_CLASSES: {DEBUG_ONLY_CLASSES}")
    print(f"[INFO] DEBUG_ONLY_TRACK_IDS: {DEBUG_ONLY_TRACK_IDS}")
    print(f"[INFO] DEBUG_COMPARE_TRACK_VS_PREDICT: {DEBUG_COMPARE_TRACK_VS_PREDICT}")
    print(f"[INFO] DEBUG_COMPARE_CLASSES: {DEBUG_COMPARE_CLASSES}")
    print(f"[INFO] DEBUG_COMPARE_CONF: {DEBUG_COMPARE_CONF}")
    print(f"[INFO] DEBUG_COMPARE_IOU: {DEBUG_COMPARE_IOU}")
    if CUDA_OK:
        print(f"[INFO] CUDA device: {torch.cuda.get_device_name(0)}")

    process_video(
        in_path=args.inp,
        out_path=args.out,
        master_key=args.key,
        mode=args.mode,
        roi_sidecar_path=args.roi_sidecar,
        reuse_rois=(args.mode == "decrypt"),
        detect_every=args.detect_every,
        detect_width=args.detect_width,
        keystream_dump_path=args.keystream_dump,
    )

    print(f"[INFO] Done. Wrote: {args.out}")