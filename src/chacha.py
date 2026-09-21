# Public paper reference: replace file placeholders before running.
# See README.md for dummy commands and required inputs.
# Algorithm constants are retained; placeholders are not experimental settings.

# chacha.py
# Standalone merged ChaCha20-Poly1305 ROI encryption pipeline.
# Generated from framework_faster_chachapoly_optimized.py + framework_proxy_payload_matched_chachapoly_optimized.py.
# Preview-mode selection was removed: public video/audio output is always ChaCha-ciphered.


from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import time
import zlib
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import av
import cv2
import numpy as np
import torch
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from tqdm import tqdm
from ultralytics import YOLO


# -----------------------------
# Config
# -----------------------------
COCO_MODEL = os.environ.get("COCO_MODEL", "models/REPLACE_WITH_SEGMENTATION_MODEL.pt")
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
MASK_DILATE_PX = 1
DETERMINISTIC_ROI_SORT = True
TRACKER_CFG = os.environ.get("TRACKER_CFG", "config/REPLACE_WITH_TRACKER_CONFIG.yaml")
TRACK_PERSIST = True
RETINA_MASKS = _env_bool("RETINA_MASKS", True)
POLICY_SCOPE = "class"

DRAW_DEBUG_OVERLAY = False
DEBUG_FONT_SCALE = 0.8
DEBUG_THICKNESS = 2
DEBUG_COLOR = (0, 255, 255)
ENABLE_DEBUG_PRINTS = False
DEBUG_EVERY_N_FRAMES = 1
DEBUG_ONLY_CLASSES = None
DEBUG_ONLY_TRACK_IDS = None
DEBUG_PRINT_RAW_DETECTIONS = False
DEBUG_PRINT_CANDIDATE_FILTERS = False
DEBUG_PRINT_MASK_STATS = False
DEBUG_PRINT_SUMMARY = False
DEBUG_COMPARE_TRACK_VS_PREDICT = False
DEBUG_COMPARE_CLASSES = list(COCO_CLASSES_OF_INTEREST)
DEBUG_COMPARE_CONF = CONF_THRES
DEBUG_COMPARE_IOU = IOU_THRES

CUDA_OK = torch.cuda.is_available() and torch.cuda.device_count() > 0
DEVICE = "cuda" if CUDA_OK else "cpu"
YOLO_DEVICE = 0 if CUDA_OK else "cpu"
YOLO_HALF = True if CUDA_OK else False
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

KERNEL_3 = np.ones((3, 3), np.uint8)
_KERNEL_CACHE: Dict[int, np.ndarray] = {}


def _get_kernel(k: int) -> np.ndarray:
    kk = int(k)
    ker = _KERNEL_CACHE.get(kk)
    if ker is None:
        ker = np.ones((kk, kk), np.uint8)
        _KERNEL_CACHE[kk] = ker
    return ker


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


def clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(val)))


def pad_bbox_exclusive(x1: int, y1: int, x2: int, y2: int, pad: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = clamp(int(x1) - pad, 0, w - 1)
    y1 = clamp(int(y1) - pad, 0, h - 1)
    x2 = clamp(int(x2) + pad, 1, w)
    y2 = clamp(int(y2) + pad, 1, h)
    return x1, y1, x2, y2


def sha256_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _b64e(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _b64d(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


def policy_tag_for_class(cls_id: int) -> str:
    if cls_id == 0:
        return "person_sensitive"
    if cls_id in (2, 3, 5, 7):
        return "vehicle_sensitive"
    return "default_sensitive"


def derive_subkey(master_key: str, policy_tag: str, cls_id: int, track_id: int = -1, scope: str = "class") -> bytes:
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
    frame_class: str,
    policy_tag: str,
    scope: str = "class",
    bbox: Optional[Tuple[int, int, int, int]] = None,
    mask_hash: Optional[str] = None,
    roi_index: int = -1,
) -> bytes:
    subkey = derive_subkey(master_key, policy_tag, cls_id, track_id, scope=scope)
    parts = [
        f"seed|frame={int(frame_idx)}",
        f"track={int(track_id)}",
        f"cls={int(cls_id)}",
        f"frame_class={str(frame_class)}",
        f"policy={policy_tag}",
        f"roi={int(roi_index)}",
    ]
    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        parts.append(f"bbox={x1},{y1},{x2},{y2}")
    if mask_hash:
        parts.append(f"mask={mask_hash}")
    return hmac.new(subkey, "|".join(parts).encode("utf-8"), hashlib.sha256).digest()


def derive_seed(
    master_key: str,
    frame_idx: int,
    track_id: int,
    cls_id: int,
    frame_class: str,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    mask_hash: Optional[str] = None,
    roi_index: int = -1,
) -> bytes:
    return derive_policy_seed(
        master_key=master_key,
        frame_idx=frame_idx,
        track_id=track_id,
        cls_id=cls_id,
        frame_class=frame_class,
        policy_tag=policy_tag_for_class(cls_id),
        scope=POLICY_SCOPE,
        bbox=bbox,
        mask_hash=mask_hash,
        roi_index=roi_index,
    )


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
        "aead_v1",
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


def _compute_mask_hash(mask_bool: np.ndarray) -> str:
    packed = np.packbits(np.asarray(mask_bool, dtype=np.uint8).reshape(-1))
    return hashlib.blake2b(packed.tobytes(), digest_size=8).hexdigest()


def _kdf_expand(seed: bytes, label: str, length: int) -> bytes:
    out = b""
    ctr = 1
    while len(out) < int(length):
        out += hmac.new(seed, f"{label}|{ctr}".encode("utf-8"), hashlib.sha256).digest()
        ctr += 1
    return out[: int(length)]


def normalize_aead_nonce(seed: bytes, frame_class: str, nonce: bytes, *, label: str = "roi") -> bytes:
    nb = bytes(nonce or b"")
    if len(nb) == 12:
        return nb
    if len(nb) >= 12:
        return nb[:12]
    return (nb + _kdf_expand(seed, f"nonce-pad|{frame_class}|{label}", 12))[:12]


def stream_key_and_nonce(seed: bytes, frame_class: str, label: str) -> Tuple[bytes, bytes]:
    key = _kdf_expand(seed, f"stream-key|{frame_class}|{label}", 32)
    nonce = _kdf_expand(seed, f"stream-nonce|{frame_class}|{label}", 16)
    return key, nonce


def chacha20_stream_bytes(seed: bytes, frame_class: str, label: str, n: int) -> np.ndarray:
    n = int(n)
    if n <= 0:
        return np.empty(0, dtype=np.uint8)
    key, nonce = stream_key_and_nonce(seed, frame_class, label)
    enc = Cipher(algorithms.ChaCha20(key, nonce), mode=None).encryptor()
    stream = enc.update(b"\x00" * n)
    return np.frombuffer(stream, dtype=np.uint8).copy()


_SESSION_ID = os.urandom(16)
_NONCE_COUNTER = 0

def prepare_aead_meta(
    seed: bytes,
    frame_class: str,
    chunk_size: int = 1024,
    block_size: int = 1024,
    **_: object,
) -> dict:
    global _NONCE_COUNTER
    _NONCE_COUNTER += 1

    nonce = hmac.new(
        seed,
        b"aead-nonce|" +
        frame_class.encode("utf-8") + b"|" +
        _SESSION_ID + b"|" +
        str(_NONCE_COUNTER).encode("utf-8"),
        hashlib.sha256,
    ).digest()[:12]

    return {
        "scheme": "chacha20poly1305_v1",
        "stream_nonce_b64": _b64e(nonce),
        "tags_b64": "",
        "chunk_size": int(max(256, chunk_size)),
        "block_size": int(max(128, block_size)),
    }


def aead_aad_bytes(
    frame_class: str,
    frame_idx: int,
    track_id: int,
    cls_id: int,
    policy_tag: str,
    bbox: Optional[Tuple[int, int, int, int]],
    mask_hash: str,
    roi_index: int,
    plain_len: int,
    domain: str,
) -> bytes:
    rec = {
        "scheme": "chacha20poly1305_v1",
        "domain": str(domain),
        "frame_class": str(frame_class),
        "frame_idx": int(frame_idx),
        "track_id": int(track_id),
        "cls_id": int(cls_id),
        "policy_tag": str(policy_tag),
        "bbox": None if bbox is None else [int(v) for v in bbox],
        "mask_hash": str(mask_hash or ""),
        "roi_index": int(roi_index),
        "plain_len": int(plain_len),
    }
    return json.dumps(rec, sort_keys=True, separators=(",", ":")).encode("utf-8")


def aead_key(seed: bytes, frame_class: str, domain: str) -> bytes:
    return _kdf_expand(seed, f"aead-key|{frame_class}|{domain}", 32)


def encrypt_masked_bytes_aead(
    roi_bgr: np.ndarray,
    mask: np.ndarray,
    seed: bytes,
    frame_class: str,
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
        return roi_bgr, crypto_meta if crypto_meta is not None else prepare_aead_meta(seed, frame_class)
    if roi_bgr.dtype != np.uint8:
        roi_bgr = roi_bgr.astype(np.uint8, copy=False)
    if mask.dtype != np.bool_:
        mask = mask.astype(np.bool_, copy=False)
    core_seed = _bind_roi_seed(seed, frame_idx, track_id, cls_id, policy_tag, bbox, mask_hash, roi_index)
    meta = dict(crypto_meta or {})
    if not meta.get("stream_nonce_b64"):
        meta.update(prepare_aead_meta(seed, frame_class))
    nonce = normalize_aead_nonce(core_seed, frame_class, _b64d(str(meta.get("stream_nonce_b64", ""))), label="roi")
    meta["stream_nonce_b64"] = _b64e(nonce)
    pix = np.ascontiguousarray(roi_bgr[mask], dtype=np.uint8).reshape(-1)
    aad = aead_aad_bytes(frame_class, frame_idx, track_id, cls_id, policy_tag, bbox, mask_hash, roi_index, int(pix.size), "roi")
    key = aead_key(core_seed, frame_class, "roi")
    ct_with_tag = ChaCha20Poly1305(key).encrypt(nonce, pix.tobytes(), aad)
    meta["scheme"] = "chacha20poly1305_v1"
    meta["tags_b64"] = _b64e(ct_with_tag[-16:])
    roi_bgr[mask] = np.frombuffer(ct_with_tag[:-16], dtype=np.uint8).reshape(-1, 3)
    return roi_bgr, meta


def try_decrypt_masked_bytes_aead(
    roi_bgr: np.ndarray,
    mask: np.ndarray,
    seed: bytes,
    frame_class: str,
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
        return roi_bgr, False
    if roi_bgr.dtype != np.uint8:
        roi_bgr = roi_bgr.astype(np.uint8, copy=False)
    if mask.dtype != np.bool_:
        mask = mask.astype(np.bool_, copy=False)
    core_seed = _bind_roi_seed(seed, frame_idx, track_id, cls_id, policy_tag, bbox, mask_hash, roi_index)
    try:
        nonce = normalize_aead_nonce(core_seed, frame_class, _b64d(str(crypto_meta.get("stream_nonce_b64", ""))), label="roi")
        tag = _b64d(str(crypto_meta.get("tags_b64", "")))
        if len(tag) != 16:
            return roi_bgr, False
        pix = np.ascontiguousarray(roi_bgr[mask], dtype=np.uint8).reshape(-1)
        aad = aead_aad_bytes(frame_class, frame_idx, track_id, cls_id, policy_tag, bbox, mask_hash, roi_index, int(pix.size), "roi")
        key = aead_key(core_seed, frame_class, "roi")
        pt = ChaCha20Poly1305(key).decrypt(nonce, pix.tobytes() + tag, aad)
        roi_bgr[mask] = np.frombuffer(pt, dtype=np.uint8).reshape(-1, 3)
        return roi_bgr, True
    except Exception:
        return roi_bgr, False


def decrypt_masked_bytes_aead(
    roi_bgr: np.ndarray,
    mask: np.ndarray,
    seed: bytes,
    frame_class: str,
    crypto_meta: Optional[dict] = None,
    frame_idx: int = 0,
    track_id: int = -1,
    cls_id: int = -1,
    policy_tag: str = "default_sensitive",
    bbox: Optional[Tuple[int, int, int, int]] = None,
    mask_hash: str = "",
    roi_index: int = 0,
):
    roi_out, _ok = try_decrypt_masked_bytes_aead(
        roi_bgr, mask, seed, frame_class,
        crypto_meta=crypto_meta,
        frame_idx=frame_idx,
        track_id=track_id,
        cls_id=cls_id,
        policy_tag=policy_tag,
        bbox=bbox,
        mask_hash=mask_hash,
        roi_index=roi_index,
    )
    return roi_out


def encrypt_masked_bytes_cpu(roi_bgr: np.ndarray, mask: np.ndarray, seed: bytes, frame_class: str) -> np.ndarray:
    preview_meta = {
        "scheme": "chacha20poly1305_v1",
        "stream_nonce_b64": _b64e(normalize_aead_nonce(seed, frame_class, bytes(12), label="preview")),
        "tags_b64": "",
        "chunk_size": 1024,
        "block_size": 1024,
    }
    roi_out, _ = encrypt_masked_bytes_aead(roi_bgr, mask, seed, frame_class, crypto_meta=preview_meta)
    return roi_out


def decrypt_masked_bytes_cpu(roi_bgr: np.ndarray, mask: np.ndarray, seed: bytes, frame_class: str) -> np.ndarray:
    return roi_bgr


def frame_class_for_pict_type(pict_type) -> str:
    name = getattr(pict_type, "name", None)
    if name in ("I", "P", "B"):
        return name

    txt = str(pict_type).strip()

    if txt in ("1", "I"):
        return "I"
    if txt in ("2", "P"):
        return "P"
    if txt in ("3", "B"):
        return "B"

    if "PictureType" in txt:
        if ".I" in txt or " I" in txt:
            return "I"
        if ".P" in txt or " P" in txt:
            return "P"
        if ".B" in txt or " B" in txt:
            return "B"

    return "UNK"


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


@dataclass
class DetectorBundle:
    def __init__(self, coco: YOLO):
        self.coco = coco


def load_detectors() -> DetectorBundle:
    model_path = str(COCO_MODEL)
    print(f"[INFO] Loading YOLO model: {model_path}")
    coco = YOLO(model_path)
    try:
        if not model_path.lower().endswith((".engine", ".onnx")):
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
    if cls_id == 0:
        return 0.33, MASK_DILATE_PX
    if cls_id in (2, 3, 5, 7):
        return 0.38, MASK_DILATE_PX
    return 0.45, MASK_DILATE_PX


def _close_only(mask_bool: np.ndarray) -> np.ndarray:
    m = (mask_bool.astype(np.uint8) * 255)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, KERNEL_3, iterations=1)
    return m > 0


def _dilate(mask_bool: np.ndarray, px: int) -> np.ndarray:
    if px is None or px <= 0:
        return mask_bool
    k = 2 * int(px) + 1
    m = (mask_bool.astype(np.uint8) * 255)
    m = cv2.dilate(m, _get_kernel(k), iterations=1)
    return m > 0


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
    return 0.0 if union <= 0 else inter / float(union)


def _mask_support_bbox(mask_small: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask_small > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _debug_compare_track_vs_predict(det: DetectorBundle, frame_infer: np.ndarray, frame_idx: int) -> None:
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
    print(f"[DBG][frame={frame_idx}] boxes={0 if pred_res.boxes is None else len(pred_res.boxes)}")


def _max_iou_with_class(box_xyxy: Tuple[int, int, int, int], boxes_xyxy: np.ndarray, cls_arr: np.ndarray, target_cls: int) -> float:
    best = 0.0
    for i in range(len(cls_arr)):
        if int(cls_arr[i]) != int(target_cls):
            continue
        bx1, by1, bx2, by2 = boxes_xyxy[i]
        iou = _bbox_iou_xyxy(box_xyxy, (int(round(bx1)), int(round(by1)), int(round(bx2)), int(round(by2))))
        if iou > best:
            best = iou
    return best


def detect_rois_with_masks(det: DetectorBundle, frame_bgr: np.ndarray, frame_idx: int, detect_width: int = 1280, pack_for_sidecar: bool = False) -> List[dict]:
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
    if res.boxes is None or len(res.boxes) == 0:
        return out
    cls_arr = res.boxes.cls.detach().cpu().numpy().astype(int)
    conf_arr = res.boxes.conf.detach().cpu().numpy().astype(float)
    boxes_xyxy = res.boxes.xyxy.detach().cpu().numpy()
    if getattr(res.boxes, "id", None) is not None and res.boxes.id is not None:
        id_arr = res.boxes.id.detach().cpu().numpy().astype(int)
    else:
        id_arr = -1 - np.arange(len(cls_arr), dtype=int)
    masks = None
    if getattr(res, "masks", None) is not None and getattr(res.masks, "data", None) is not None:
        masks = res.masks.data.detach().cpu().numpy()
    mask_infos = []
    if masks is not None:
        for j in range(len(masks)):
            support = _mask_support_bbox(masks[j])
            if support is not None:
                mask_infos.append({"mask_small": masks[j], "support_bbox": support})
    candidates = []
    for i, c in enumerate(cls_arr):
        c = int(c)
        if c not in COCO_CLASSES_OF_INTEREST:
            continue
        thr, grow_px = _mask_params_for_class(c)
        bx1, by1, bx2, by2 = map(float, boxes_xyxy[i])
        x1 = clamp(round(bx1 * sx), 0, W - 1)
        y1 = clamp(round(by1 * sy), 0, H - 1)
        x2 = clamp(round(bx2 * sx), 1, W)
        y2 = clamp(round(by2 * sy), 1, H)
        if x2 <= x1 or y2 <= y1:
            continue
        candidates.append({
            "cls": c,
            "conf": float(conf_arr[i]),
            "track_id": int(id_arr[i]),
            "thr": float(thr),
            "grow": int(grow_px),
            "bbox_full": (x1, y1, x2, y2),
            "bbox_inf": (int(round(bx1)), int(round(by1)), int(round(bx2)), int(round(by2))),
        })
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
        roi_mask = None
        if best_mask_info is not None and best_iou >= 0.75:
            mask_small = best_mask_info["mask_small"]
            if int(c["cls"]) in (2, 3, 5, 7):
                person_iou = _max_iou_with_class(det_box_inf, boxes_xyxy, cls_arr, 0)
                if person_iou >= 0.25:
                    mask_small = None
            if mask_small is not None:
                mh, mw = mask_small.shape
                mx1 = clamp(round(ibx1 * (mw / float(inf_W))), 0, mw - 1)
                my1 = clamp(round(iby1 * (mh / float(inf_H))), 0, mh - 1)
                mx2 = clamp(round(ibx2 * (mw / float(inf_W))), mx1 + 1, mw)
                my2 = clamp(round(iby2 * (mh / float(inf_H))), my1 + 1, mh)
                mask_crop_small = mask_small[my1:my2, mx1:mx2]
                mask_crop_fullprob = cv2.resize(mask_crop_small.astype(np.float32), (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
                mb = mask_crop_fullprob >= thr
                if mb.sum() > 0:
                    mb = _close_only(mb)
                    if grow_px > 0:
                        mb = _dilate(mb, grow_px)
                    x1p, y1p, x2p, y2p = pad_bbox_exclusive(x1, y1, x2, y2, ROI_PAD, W, H)
                    roi_h = y2p - y1p
                    roi_w = x2p - x1p
                    tmp = np.zeros((roi_h, roi_w), dtype=bool)
                    oy = y1 - y1p
                    ox = x1 - x1p
                    tmp[oy:oy + (y2 - y1), ox:ox + (x2 - x1)] = mb
                    if tmp.sum() > 0:
                        roi_mask = tmp
        if roi_mask is None:
            x1p, y1p, x2p, y2p = pad_bbox_exclusive(x1, y1, x2, y2, ROI_PAD, W, H)
            if x2p <= x1p or y2p <= y1p:
                continue
            roi_mask = np.ones((y2p - y1p, x2p - x1p), dtype=bool)
        else:
            x1p, y1p, x2p, y2p = pad_bbox_exclusive(x1, y1, x2, y2, ROI_PAD, W, H)
        rec = {
            "bbox": [int(x1p), int(y1p), int(x2p), int(y2p)],
            "mask": roi_mask,
            "cls": int(c["cls"]),
            "conf": float(c["conf"]),
            "track_id": int(c["track_id"]),
            "policy_tag": policy_tag_for_class(int(c["cls"])),
        }
        if pack_for_sidecar:
            rec["mask_pack"] = pack_mask(roi_mask)
        out.append(rec)
    if DETERMINISTIC_ROI_SORT and len(out) > 1:
        def _key2(r):
            x1, y1, x2, y2 = r["bbox"]
            area = max(0, x2 - x1) * max(0, y2 - y1)
            return (-r.get("conf", 0.0), r.get("cls", 999), r.get("track_id", -1), area, x1, y1, x2, y2)
        out.sort(key=_key2)
    return out


def _safe_rate(fps: float) -> Fraction:
    if fps is None or not np.isfinite(fps) or fps <= 0:
        return Fraction(30, 1)
    return Fraction(fps).limit_denominator(100000)


def open_lossless_writer(out_path: str, width: int, height: int, fps: float):
    out = av.open(out_path, mode="w", format="mp4")
    rate = _safe_rate(fps)
    st = out.add_stream("libx264rgb", rate=rate)
    st.width = width
    st.height = height
    st.pix_fmt = "bgr24"
    st.options = {"preset": "slower", "qp": "0", "x264-params": "keyint=10000:min-keyint=10000:scenecut=0:bframes=0"}
    return out, st


def _draw_debug_overlay(frame_bgr: np.ndarray, frame_idx: int, pict_str: str, frame_class: str, rois: List[dict]) -> None:
    if not DRAW_DEBUG_OVERLAY:
        return
    H, W = frame_bgr.shape[:2]
    cv2.rectangle(frame_bgr, (1, 1), (W - 2, H - 2), DEBUG_COLOR, 2)
    header = f"frame={frame_idx} pict={pict_str} class={frame_class} rois={len(rois)}"
    cv2.putText(frame_bgr, header, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, DEBUG_FONT_SCALE, DEBUG_COLOR, DEBUG_THICKNESS, cv2.LINE_AA)


def _decode_audio_pcm_s16(in_container: av.container.InputContainer) -> Optional[Tuple[np.ndarray, int, str]]:
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
    channels = stream_ch if isinstance(stream_ch, int) and stream_ch > 0 else (_channels_from_layout(astream.layout) or 1)
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
    return np.concatenate(chunks, axis=0), sr, layout_name


def _add_pcm_audio_stream(out_container: av.container.OutputContainer, sr: int, layout_name: str):
    aout = out_container.add_stream("flac", rate=sr)
    aout.layout = layout_name
    return aout


def _mux_pcm_segment(out_container: av.container.OutputContainer, aout, seg_sc_i16: np.ndarray, sr: int, layout_name: str, pts_samples: int) -> int:
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


def audio_preview_segment_chacha(seg_sc_i16: np.ndarray, seed: bytes, frame_class: str, decrypt: bool = False) -> np.ndarray:
    if seg_sc_i16.size == 0:
        return seg_sc_i16
    seg = np.ascontiguousarray(seg_sc_i16.astype(np.int16, copy=False))
    raw = seg.view(np.uint8).reshape(-1)
    stream_seed = derive_stream_seed(seed, "audio-preview")
    ks = chacha20_stream_bytes(stream_seed, str(frame_class or "UNK"), "audio-preview", raw.size)
    transformed = np.bitwise_xor(raw, ks).astype(np.uint8, copy=False)
    return transformed.view(np.int16).reshape(seg.shape)


def warmup_detectors(det=None):
    if det is None:
        det = load_detectors()
    try:
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        with torch.inference_mode():
            try:
                _ = det.coco.track(dummy, device=YOLO_DEVICE, half=YOLO_HALF, verbose=False, persist=TRACK_PERSIST, tracker=TRACKER_CFG, classes=COCO_CLASSES_LIST)
            except Exception:
                _ = det.coco.predict(dummy, device=YOLO_DEVICE, half=YOLO_HALF, verbose=False)
    except Exception:
        pass
    return det



# ============================================================
# Matched public output + encrypted payload recovery (standalone)
# ============================================================

# -----------------------------
# Matched public-output settings
# -----------------------------
MATCHED_VIDEO_CRF = os.environ.get("MATCHED_VIDEO_CRF", "18")
MATCHED_VIDEO_PRESET = os.environ.get("MATCHED_VIDEO_PRESET", "medium")
MATCHED_AUDIO_BITRATE = int(os.environ.get("MATCHED_AUDIO_BITRATE", "192000"))

PAYLOAD_ZLIB_LEVEL = int(os.environ.get("PAYLOAD_ZLIB_LEVEL", "6"))

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)


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

    # IMPORTANT:
    # The session_id and source width/height/fps/frames describe the original/plain input
    # used during encryption. During decrypt, current_identity describes the encrypted
    # public video. Bind the files by SHA-256 above, and validate encrypted-video identity
    # only through the explicit encrypted_* manifest fields when present.
    if "encrypted_width" in manifest or "encrypted_height" in manifest:
        for mk, ck in (("encrypted_width", "width"), ("encrypted_height", "height")):
            if mk in manifest and int(manifest.get(mk, -1)) != int(current_identity.get(ck, -2)):
                raise ValueError(f"Decrypt verification failed: manifest encrypted-video {ck} mismatch.")
        if "encrypted_fps" in manifest and abs(float(manifest.get("encrypted_fps", 0.0)) - float(current_identity.get("fps", 0.0))) > 0.05:
            raise ValueError("Decrypt verification failed: manifest encrypted-video fps mismatch.")
    else:
        # Backwards-compatible manifests do not have encrypted_* fields. Check dimensions
        # and fps, but deliberately do not check container frame count because MKV/MP4
        # frame counts can be off by one after encoding.
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

def _b64e(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _b64d(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


def _safe_rate(fps: float) -> Fraction:
    if fps is None or not np.isfinite(fps) or fps <= 0:
        return Fraction(30, 1)
    return Fraction(fps).limit_denominator(100000)


def _guess_container_format_from_path(path: str) -> Optional[str]:
    ext = os.path.splitext(str(path))[1].lower()
    mapping = {".mkv": "matroska", ".webm": "webm", ".mp4": "mp4", ".m4v": "ipod", ".mov": "mov", ".avi": "avi"}
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
    info = {"container": None, "video_codec": None, "video_pix_fmt": None, "audio_codec": None}
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


def _select_video_pix_fmt(video_codec: str, source_video_codec: Optional[str], source_pix_fmt: Optional[str]) -> str:
    """
    Choose a pixel format that is valid for the selected encoder while staying as close
    as possible to the input profile. This prevents cases like falling back to libx264
    while accidentally carrying over an RGB/BGRA source pixel format from an attacked
    intermediate video written by OpenCV/FFV1.
    """
    vcodec = (video_codec or "").lower().strip()
    svc = (source_video_codec or "").lower().strip()
    spf = (source_pix_fmt or "").lower().strip()

    if vcodec in {"libx264", "libx265", "libvpx-vp9", "libvpx", "libaom-av1"}:
        if "444" in spf:
            return "yuv444p"
        if "422" in spf:
            return "yuv422p"
        if "gray" in spf:
            return "gray"
        return "yuv420p"

    if svc in {"h264", "avc1", "hevc", "h265", "hev1", "hvc1", "vp9", "vp8", "av1", "av01"}:
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


def choose_output_profile(in_path: str, out_path: str) -> OutputProfile:
    src = _probe_input_profile(in_path)
    container_format = _guess_container_format_from_path(out_path)
    if container_format is None:
        src_container = (src.get("container") or "").split(",")[0].strip() or None
        if src_container in {"matroska", "webm", "mp4", "mov", "avi"}:
            container_format = src_container
    video_codec, video_options = _select_video_encoder(src.get("video_codec"))
    video_pix_fmt = _select_video_pix_fmt(video_codec, src.get("video_codec"), src.get("video_pix_fmt"))
    audio_codec, audio_bitrate = _select_audio_encoder(src.get("audio_codec"))
    return OutputProfile(container_format, video_codec, video_pix_fmt, video_options, audio_codec, audio_bitrate, src.get("container"), src.get("video_codec"), src.get("video_pix_fmt"), src.get("audio_codec"))


def open_video_writer(out_path: str, width: int, height: int, fps: float, profile: OutputProfile):
    out = av.open(out_path, mode="w", format=profile.container_format) if profile.container_format else av.open(out_path, mode="w")
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


# Preview selection functions removed: output video/audio is always ChaCha-ciphered.

def _payload_common_kwargs(frame_idx: int, track_id: int, cls_id: int, policy_tag: str, bbox, mask_hash: str, roi_index: int):
    return {"frame_idx": int(frame_idx), "track_id": int(track_id), "cls_id": int(cls_id), "policy_tag": str(policy_tag), "bbox": None if bbox is None else tuple(int(v) for v in bbox), "mask_hash": str(mask_hash or ""), "roi_index": int(roi_index)}


def _payload_aead_key(seed: bytes, frame_class: str, domain: str) -> bytes:
    return hmac.new(seed, f"payload-aead-key|{frame_class}|{domain}".encode("utf-8"), hashlib.sha256).digest()[:32]


def _payload_aad(frame_class: str, domain: str, common: dict, plain_len: int) -> bytes:
    rec = {
        "scheme": "chacha20poly1305_v1",
        "domain": str(domain),
        "frame_class": str(frame_class),
        "frame_idx": int(common.get("frame_idx", -1)),
        "track_id": int(common.get("track_id", -1)),
        "cls_id": int(common.get("cls_id", -1)),
        "policy_tag": str(common.get("policy_tag", "")),
        "bbox": None if common.get("bbox") is None else [int(v) for v in common.get("bbox")],
        "mask_hash": str(common.get("mask_hash", "")),
        "roi_index": int(common.get("roi_index", -1)),
        "plain_len": int(plain_len),
    }
    return json.dumps(rec, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _payload_encrypt_bytes(plain: bytes, seed: bytes, frame_class: str, domain: str, common: dict):
    work_seed = derive_stream_seed(seed, f"payload|{domain}")
    meta = dict(prepare_aead_meta(work_seed, frame_class, **common))
    nonce = normalize_aead_nonce(work_seed, frame_class, _b64d(str(meta.get("stream_nonce_b64", ""))), label=f"payload-{domain}")
    meta["stream_nonce_b64"] = _b64e(nonce)
    meta["scheme"] = "chacha20poly1305_v1"
    aad = _payload_aad(frame_class, domain, common, len(plain))
    key = _payload_aead_key(work_seed, frame_class, domain)
    ct_with_tag = ChaCha20Poly1305(key).encrypt(nonce, plain, aad)
    meta["tags_b64"] = _b64e(ct_with_tag[-16:])
    return ct_with_tag[:-16], meta


def _payload_decrypt_bytes(cipher: bytes, seed: bytes, frame_class: str, meta: dict, domain: str, common: dict):
    work_seed = derive_stream_seed(seed, f"payload|{domain}")
    nonce = normalize_aead_nonce(work_seed, frame_class, _b64d(str(meta.get("stream_nonce_b64", ""))), label=f"payload-{domain}")
    tag = _b64d(str(meta.get("tags_b64", "")))
    if len(tag) != 16:
        raise ValueError(f"Invalid ChaCha20-Poly1305 tag for payload {domain}")
    aad = _payload_aad(frame_class, domain, common, int(common.get("plain_len")) if int(common.get("plain_len", -1)) >= 0 else len(cipher))
    key = _payload_aead_key(work_seed, frame_class, domain)
    return ChaCha20Poly1305(key).decrypt(nonce, bytes(cipher) + tag, aad)


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


def process_video(in_path: str, out_path: str, master_key: str, mode: str, roi_sidecar_path: Optional[str], payload_path: Optional[str], manifest_path: Optional[str], detect_width: int, keystream_dump_path: Optional[str], reuse_rois: bool = False):
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

    det = load_detectors() if (mode == "encrypt" and not reuse_rois_for_encrypt) else None
    if mode == "encrypt" and det is not None:
        print("[INFO] Warmup detectors...")
        try:
            warmup_detectors(det)
        except Exception:
            pass
    in_container = av.open(in_path)
    in_stream = in_container.streams.video[0]
    in_stream.thread_type = "AUTO"
    audio_container = av.open(in_path)
    audio_info = _decode_audio_pcm_s16(audio_container)
    audio_container.close()
    has_audio = audio_info is not None
    pcm_all, audio_sr, audio_layout = audio_info if has_audio else (None, None, None)
    fps = float(in_stream.average_rate) if in_stream.average_rate else 30.0
    width = in_stream.codec_context.width
    height = in_stream.codec_context.height
    input_identity = _video_identity(in_path)
    session_id = _session_id(master_key, input_identity["width"], input_identity["height"], input_identity["fps"], input_identity["frames"], has_audio)
    output_profile = choose_output_profile(in_path, out_path)
    print(f"[INFO] PUBLIC_OUTPUT_MATCH: container={output_profile.container_format or 'auto'} video={output_profile.video_codec}/{output_profile.video_pix_fmt} audio={output_profile.audio_codec}")
    out_container, out_stream = open_video_writer(out_path, width, height, fps, output_profile)
    aout = add_audio_stream(out_container, audio_sr, audio_layout, output_profile) if has_audio else None
    sidecar_f = None
    payload_f = None
    if mode == "encrypt":
        if roi_sidecar_path:
            sidecar_f = open(roi_sidecar_path, "w", encoding="utf-8", buffering=1024 * 1024)
        if payload_path:
            payload_f = open(payload_path, "w", encoding="utf-8", buffering=1024 * 1024)
            _store_payload_record(payload_f, {"type": "meta", "crypto_scheme": "chacha20poly1305_v1", "version": 1, "session_id": session_id, "width": int(input_identity["width"]), "height": int(input_identity["height"]), "fps": float(input_identity["fps"]), "frames": int(input_identity["frames"]), "video_output_mode": "chacha", "audio_output_mode": "chacha", "audio_sr": audio_sr, "audio_layout": audio_layout, "has_audio": bool(has_audio)})
    payload_header: dict = {}
    payload_records: Dict[int, dict] = {}
    payload_tail: Optional[dict] = None
    if mode == "decrypt":
        if not roi_sidecar_path:
            raise ValueError("Decrypt requires --roi_sidecar")
        with open(roi_sidecar_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    roi_records[int(rec["frame_idx"])] = rec
        payload_header, payload_records, payload_tail = _load_payload_records(payload_path)
        _verify_decrypt_inputs(manifest_path, in_path, roi_sidecar_path, payload_path, payload_header, session_id, input_identity)
        has_audio = bool(payload_header.get("has_audio", has_audio))
        audio_sr = int(payload_header.get("audio_sr", audio_sr or 48000)) if has_audio else None
        audio_layout = payload_header.get("audio_layout", audio_layout or "stereo") if has_audio else None
    total_frames = in_stream.frames if in_stream.frames else None
    pbar = tqdm(total=total_frames, desc=f"{mode} frames", unit="frame")
    ks_f = open(keystream_dump_path, "wb", buffering=1024 * 1024) if (mode == "encrypt" and keystream_dump_path) else None
    audio_pts = 0
    audio_prev_end = 0
    frame_idx = 0
    last_frame_class = "UNK"
    for av_frame in in_container.decode(in_stream):
        frame = av_frame.to_ndarray(format="bgr24")
        pict_raw = av_frame.pict_type
        frame_class = frame_class_for_pict_type(pict_raw)
        last_frame_class = frame_class
        if mode == "encrypt":
            if reuse_rois_for_encrypt:
                rec = roi_records.get(frame_idx, {"rois": [], "frame_class": frame_class})
                frame_class = str(rec.get("frame_class", frame_class))
                rois = rec.get("rois", [])
            else:
                rois = detect_rois_with_masks(det, frame, frame_idx=frame_idx, detect_width=detect_width, pack_for_sidecar=True)
            for roi_index, r in enumerate(rois):
                r["roi_index"] = int(r.get("roi_index", roi_index))
                cls_id = int(r.get("cls", -1))
                track_id = int(r.get("track_id", -1))
                policy_tag = str(r.get("policy_tag", policy_tag_for_class(cls_id)))
                r["policy_tag"] = policy_tag
                roi_mask0 = r.get("mask", None)
                if roi_mask0 is None:
                    roi_mask0 = unpack_mask(r["mask_pack"])
                r["mask_hash"] = str(r.get("mask_hash", _compute_mask_hash(roi_mask0)))
        else:
            rec = roi_records.get(frame_idx, {"rois": [], "frame_class": frame_class})
            frame_class = str(rec.get("frame_class", frame_class))
            rois = rec.get("rois", [])
        if mode == "encrypt" and sidecar_f is not None:
            rois_for_json = []
            for roi_index, r in enumerate(rois):
                mask_obj = r.get("mask_pack") if r.get("mask_pack") is not None else pack_mask(r["mask"])
                rois_for_json.append({"bbox": r["bbox"], "mask_pack": mask_obj, "cls": int(r.get("cls", -1)), "conf": float(r.get("conf", 0.0)), "track_id": int(r.get("track_id", -1)), "policy_tag": str(r.get("policy_tag", policy_tag_for_class(int(r.get("cls", -1))))), "roi_index": int(r.get("roi_index", roi_index)), "mask_hash": str(r.get("mask_hash", ""))})
            sidecar_f.write(json.dumps({"frame_idx": frame_idx, "frame_class": frame_class, "rois": rois_for_json}, separators=(",", ":")) + "\n")
        payload_frame = {"type": "frame", "frame_idx": frame_idx, "frame_class": frame_class, "rois": []}
        rois_to_apply = rois if mode == "encrypt" else list(reversed(rois))
        for r in rois_to_apply:
            x1, y1, x2, y2 = map(int, r["bbox"])
            x1 = clamp(x1, 0, width - 1)
            y1 = clamp(y1, 0, height - 1)
            x2 = clamp(x2, 1, width)
            y2 = clamp(y2, 1, height)
            if x2 <= x1 or y2 <= y1:
                continue
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            roi_mask = r.get("mask")
            if roi_mask is None:
                roi_mask = unpack_mask(r["mask_pack"])
            if roi_mask.shape[:2] != (roi.shape[0], roi.shape[1]):
                roi_mask = cv2.resize(roi_mask.astype(np.uint8), (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            if not np.any(roi_mask):
                continue
            track_id = int(r.get("track_id", -1))
            cls_id = int(r.get("cls", -1))
            policy_tag = str(r.get("policy_tag", policy_tag_for_class(cls_id)))
            roi_index = int(r.get("roi_index", 0))
            mask_hash = str(r.get("mask_hash", _compute_mask_hash(roi_mask)))
            bbox = (x1, y1, x2, y2)
            seed = derive_policy_seed(master_key=master_key, frame_idx=frame_idx, track_id=track_id, cls_id=cls_id, frame_class=frame_class, policy_tag=policy_tag, scope=POLICY_SCOPE, bbox=bbox, mask_hash=mask_hash, roi_index=roi_index)
            common = _payload_common_kwargs(frame_idx, track_id, cls_id, policy_tag, bbox, mask_hash, roi_index)
            if mode == "encrypt":
                orig_pixels = np.ascontiguousarray(roi[roi_mask], dtype=np.uint8).reshape(-1)
                comp = zlib.compress(orig_pixels.tobytes(), level=PAYLOAD_ZLIB_LEVEL)
                common["plain_len"] = int(len(comp))
                cipher, meta = _payload_encrypt_bytes(comp, seed, frame_class, "roi", common)
                rec_wo_tag = {"roi_index": roi_index, "bbox": [x1, y1, x2, y2], "cls": cls_id, "track_id": track_id, "policy_tag": policy_tag, "mask_hash": mask_hash, "plain_len": int(orig_pixels.size), "comp_len": int(len(comp)), "crypto_meta": meta}
                tag = _payload_tag(master_key, rec_wo_tag, cipher)
                payload_frame["rois"].append({**rec_wo_tag, "cipher_b64": _b64e(cipher), "tag": tag})
                if ks_f is not None and cipher:
                    ks_f.write(cipher)
                public_roi, _public_meta = encrypt_masked_bytes_aead(
                    roi.copy(), roi_mask, seed, frame_class,
                    frame_idx=frame_idx, track_id=track_id, cls_id=cls_id,
                    policy_tag=policy_tag, bbox=bbox, mask_hash=mask_hash, roi_index=roi_index,
                    crypto_meta=prepare_aead_meta(seed, frame_class),
                )
                frame[y1:y2, x1:x2] = public_roi
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
                rec_wo_tag = {"roi_index": int(match["roi_index"]), "bbox": match["bbox"], "cls": int(match["cls"]), "track_id": int(match["track_id"]), "policy_tag": str(match["policy_tag"]), "mask_hash": str(match["mask_hash"]), "plain_len": int(match["plain_len"]), "comp_len": int(match["comp_len"]), "crypto_meta": match["crypto_meta"]}
                expect = _payload_tag(master_key, rec_wo_tag, cipher)
                if not hmac.compare_digest(expect, str(match.get("tag", ""))):
                    raise ValueError(f"Payload authentication failed for frame {frame_idx} roi {roi_index}")
                common["plain_len"] = int(match.get("comp_len", -1)) if int(match.get("comp_len", -1)) >= 0 else int(match.get("plain_len", -1))
                plain_comp = _payload_decrypt_bytes(cipher, seed, frame_class, match["crypto_meta"], "roi", common)
                plain = zlib.decompress(plain_comp)
                pix = np.frombuffer(plain, dtype=np.uint8)
                if pix.size != int(match["plain_len"]):
                    raise ValueError(f"Payload ROI length mismatch for frame {frame_idx} roi {roi_index}")
                roi_out = roi.copy()
                roi_out[roi_mask] = pix.reshape(-1, 3)
                frame[y1:y2, x1:x2] = roi_out
        out_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        try:
            out_frame.pts = frame_idx
            out_frame.time_base = Fraction(1, int(round(fps)) if fps and fps > 0 else 30)
        except Exception:
            pass
        for packet in out_stream.encode(out_frame):
            out_container.mux(packet)
        if has_audio and aout is not None:
            start = int(round(frame_idx * audio_sr / fps))
            end = int(round((frame_idx + 1) * audio_sr / fps))
            if mode == "encrypt":
                start = max(start, audio_prev_end)
                end = max(end, start)
                end = min(end, int(pcm_all.shape[0]))
                if end > start:
                    seg = np.ascontiguousarray(pcm_all[start:end, :], dtype=np.int16)
                    aseed = derive_policy_seed(master_key=master_key, frame_idx=frame_idx, track_id=0, cls_id=-1, frame_class=frame_class, policy_tag="audio_default", scope="class")
                    audio_plain = zlib.compress(seg.tobytes(), level=PAYLOAD_ZLIB_LEVEL)
                    common_audio = _payload_common_kwargs(frame_idx, 0, -1, "audio_default", None, "", 0)
                    common_audio["plain_len"] = int(len(audio_plain))
                    acipher, ameta = _payload_encrypt_bytes(audio_plain, aseed, frame_class, "audio", common_audio)
                    arec_wo_tag = {"shape": [int(seg.shape[0]), int(seg.shape[1])], "dtype": "int16", "plain_len": int(seg.size * seg.dtype.itemsize), "comp_len": int(len(audio_plain)), "crypto_meta": ameta}
                    atag = _payload_tag(master_key, {"frame_idx": frame_idx, **arec_wo_tag}, acipher)
                    payload_frame["audio"] = {**arec_wo_tag, "cipher_b64": _b64e(acipher), "tag": atag}
                    if ks_f is not None and acipher:
                        ks_f.write(acipher)
                    seg_out = audio_preview_segment_chacha(seg.copy(), aseed, frame_class, decrypt=False)
                    audio_pts = _mux_pcm_segment(out_container, aout, seg_out, audio_sr, audio_layout, audio_pts)
                audio_prev_end = end
            else:
                frame_payload = payload_records.get(frame_idx, {})
                arec = frame_payload.get("audio")
                if arec is not None:
                    common_audio = _payload_common_kwargs(frame_idx, 0, -1, "audio_default", None, "", 0)
                    common_audio["plain_len"] = int(arec.get("comp_len", -1)) if int(arec.get("comp_len", -1)) >= 0 else int(arec.get("plain_len", -1))
                    aseed = derive_policy_seed(master_key=master_key, frame_idx=frame_idx, track_id=0, cls_id=-1, frame_class=frame_class, policy_tag="audio_default", scope="class")
                    acipher = _b64d(arec["cipher_b64"])
                    arec_wo_tag = {"frame_idx": frame_idx, "shape": arec["shape"], "dtype": arec["dtype"], "plain_len": int(arec["plain_len"]), "comp_len": int(arec["comp_len"]), "crypto_meta": arec["crypto_meta"]}
                    expect = _payload_tag(master_key, arec_wo_tag, acipher)
                    if not hmac.compare_digest(expect, str(arec.get("tag", ""))):
                        raise ValueError(f"Payload authentication failed for frame {frame_idx} audio")
                    plain_comp = _payload_decrypt_bytes(acipher, aseed, frame_class, arec["crypto_meta"], "audio", common_audio)
                    plain = zlib.decompress(plain_comp)
                    seg = np.frombuffer(plain, dtype=np.int16).reshape(int(arec["shape"][0]), int(arec["shape"][1]))
                    audio_pts = _mux_pcm_segment(out_container, aout, seg, audio_sr, audio_layout, audio_pts)
        if mode == "encrypt" and payload_f is not None:
            _store_payload_record(payload_f, payload_frame)
        if frame_idx % 30 == 0:
            print(f"frame={frame_idx} rois={len(rois)} class={frame_class}")
        frame_idx += 1
        pbar.update(1)
    if mode == "encrypt" and has_audio and payload_f is not None and aout is not None and audio_prev_end < int(pcm_all.shape[0]):
        seg = np.ascontiguousarray(pcm_all[audio_prev_end:, :], dtype=np.int16)
        last_fi = max(frame_idx - 1, 0)
        aseed = derive_policy_seed(master_key=master_key, frame_idx=last_fi, track_id=0, cls_id=-1, frame_class=last_frame_class, policy_tag="audio_default", scope="class")
        common_audio = _payload_common_kwargs(last_fi, 0, -1, "audio_default", None, "", 0)
        audio_plain = zlib.compress(seg.tobytes(), level=PAYLOAD_ZLIB_LEVEL)
        common_audio["plain_len"] = int(len(audio_plain))
        acipher, ameta = _payload_encrypt_bytes(audio_plain, aseed, last_frame_class, "audio", common_audio)
        arec_wo_tag = {"type": "tail_audio", "frame_idx": last_fi, "shape": [int(seg.shape[0]), int(seg.shape[1])], "dtype": "int16", "plain_len": int(seg.size * seg.dtype.itemsize), "comp_len": int(len(audio_plain)), "crypto_meta": ameta}
        atag = _payload_tag(master_key, arec_wo_tag, acipher)
        tail = {**arec_wo_tag, "cipher_b64": _b64e(acipher), "tag": atag}
        _store_payload_record(payload_f, tail)
        if ks_f is not None and acipher:
            ks_f.write(acipher)
        seg_out = audio_preview_segment_chacha(seg.copy(), aseed, last_frame_class, decrypt=False)
        audio_pts = _mux_pcm_segment(out_container, aout, seg_out, audio_sr, audio_layout, audio_pts)
    elif mode == "decrypt" and has_audio and payload_tail is not None and aout is not None:
        last_fi = int(payload_tail.get("frame_idx", max(frame_idx - 1, 0)))
        common_audio = _payload_common_kwargs(last_fi, 0, -1, "audio_default", None, "", 0)
        common_audio["plain_len"] = int(payload_tail.get("comp_len", -1)) if int(payload_tail.get("comp_len", -1)) >= 0 else int(payload_tail.get("plain_len", -1))
        aseed = derive_policy_seed(master_key=master_key, frame_idx=last_fi, track_id=0, cls_id=-1, frame_class=last_frame_class, policy_tag="audio_default", scope="class")
        acipher = _b64d(payload_tail["cipher_b64"])
        rec_wo_tag = {"type": "tail_audio", "frame_idx": last_fi, "shape": payload_tail["shape"], "dtype": payload_tail["dtype"], "plain_len": int(payload_tail["plain_len"]), "comp_len": int(payload_tail["comp_len"]), "crypto_meta": payload_tail["crypto_meta"]}
        expect = _payload_tag(master_key, rec_wo_tag, acipher)
        if not hmac.compare_digest(expect, str(payload_tail.get("tag", ""))):
            raise ValueError("Payload authentication failed for tail audio")
        plain_comp = _payload_decrypt_bytes(acipher, aseed, last_frame_class, payload_tail["crypto_meta"], "audio", common_audio)
        plain = zlib.decompress(plain_comp)
        seg = np.frombuffer(plain, dtype=np.int16).reshape(int(payload_tail["shape"][0]), int(payload_tail["shape"][1]))
        audio_pts = _mux_pcm_segment(out_container, aout, seg, audio_sr, audio_layout, audio_pts)
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

        # The encrypted container can report a wrong/unstable frame count after
        # re-encoding. The payload video frame records are the real recovery source
        # of truth, and tail_audio is a separate audio record, not a video frame.
        payload_frame_count = _count_payload_frame_records(payload_path)
        if payload_frame_count <= 0:
            payload_frame_count = int(frame_idx)

        # Keep the payload header and manifest consistent before hashing payload.
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Standalone ChaCha20-Poly1305 ROI encryption pipeline with encrypted payload recovery. Preview-mode selection removed."
    )
    ap.add_argument("--mode", choices=["encrypt", "decrypt"], required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--roi_sidecar", default="rois.jsonl")
    ap.add_argument("--payload", default="payload.jsonl")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--detect_width", type=int, default=_env_int("DETECT_WIDTH", 1280),
                    help="Detection resize width; 0 keeps full resolution. Default reads DETECT_WIDTH or 1280.")
    ap.add_argument("--reuse_rois", action="store_true",
                    help="Reuse existing ROI sidecar during encryption to skip YOLO detection.")
    ap.add_argument("--keystream_dump", default=None)

    # Compatibility-only: accepted so old commands do not crash, but ignored.
    ap.add_argument("--base_framework", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--video_preview_mode", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--audio_preview_mode", default=None, help=argparse.SUPPRESS)

    args = ap.parse_args()

    if args.base_framework:
        print("[WARN] --base_framework is ignored because chacha.py is standalone.")
    if args.video_preview_mode or args.audio_preview_mode:
        print("[WARN] Preview-mode arguments are ignored; output is always ChaCha-ciphered.")

    print("[INFO] Standalone file: chacha.py")
    print("[INFO] Output mode: chacha")
    print(f"[INFO] detect_width: {args.detect_width}")
    print(f"[INFO] YOLO model: {COCO_MODEL}")
    print(f"[INFO] TRACKER_CFG: {TRACKER_CFG}, RETINA_MASKS: {RETINA_MASKS}")

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
