
"""
robustness_noise_occlusion_framework_roi.py

Extends robustness_noise_occlusion_framework.py to support ROI-aware robustness for selective encryption.

Covers PDF V-G: "Resistance to noise and occlusion attacks"

Attacks (applied to cipher video frames):
- baseline (no corruption)
- gaussian noise
- salt & pepper noise
- rectangular occlusion ("sticker", ROA-inspired)

ROI support (requires --roi_sidecar and framework_faster.py providing unpack_mask()):
- attack_scope: where to apply corruption  [whole | roi | background]
- eval_scope: where to evaluate metrics    [whole | roi | background | all]

Important for selective/ROI encryption:
- Decrypt must use the SAME ROI sidecar to avoid re-detection differences after attack.
  So we call framework_faster.process_video(..., mode="decrypt", roi_sidecar_path=..., reuse_rois=True)

Outputs:
- out/report.json
- out/<scenario>/attacked_<scenario>.mkv
- out/<scenario>/decrypted_from_<scenario>.mkv

Example (one line):
python3 robustness_noise_occlusion_framework_roi.py --plain faces.mp4 --cipher encrypted.mkv --out results_rb \
  --framework_faster_path framework_faster.py --master_key secret --roi_sidecar rois.jsonl \
  --attack_scope roi --eval_scope all --max_frames 200 --verbose

Dependencies:
  pip install opencv-python numpy
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
try:
    from unified_utils import framework_cli_decrypt
except Exception:
    framework_cli_decrypt = None

DECRYPT_REQUIRES_ROI_SIDECAR = True  # avoids running detector during decrypt


def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    import hashlib
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
    import hashlib, hmac
    msg = f"session|w={int(width)}|h={int(height)}|fps={float(fps):.6f}|frames={int(frames)}|audio={1 if has_audio else 0}".encode("utf-8")
    return hmac.new(master_key.encode("utf-8"), msg, hashlib.sha256).hexdigest()[:32]


def _manifest_default_path(payload_path: str) -> str:
    return str(payload_path) + ".manifest.json"


def _load_payload_header(payload_path: str) -> dict:
    with open(payload_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "meta":
                return obj
    return {}


def _resolve_preview_modes(payload_path: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    # Only pass preview-mode CLI flags when a payload-aware decrypt path is actually in use.
    # Base frameworks generally do not expose these CLI arguments, so returning defaults like
    # "chaos"/"chaos" here can break Test 06 even when the decrypt logic itself is fine.
    if not payload_path or not os.path.exists(payload_path):
        return None, None

    video_mode: Optional[str] = os.environ.get("VIDEO_PREVIEW_MODE", "chaos")
    audio_mode: Optional[str] = os.environ.get("AUDIO_PREVIEW_MODE", "chaos")
    try:
        meta = _load_payload_header(payload_path)
        for k in ("video_preview_mode", "preview_mode", "video_mode"):
            if meta.get(k):
                video_mode = str(meta[k])
                break
        for k in ("audio_preview_mode", "audio_mode"):
            if meta.get(k):
                audio_mode = str(meta[k])
                break
    except Exception:
        pass
    return video_mode, audio_mode


def _clone_payload_for_video(src_payload_path: str, dst_payload_path: str, encrypted_video_path: str, master_key: str) -> str:
    header = _load_payload_header(src_payload_path)
    ident = _video_identity(encrypted_video_path)
    # attacked videos written by this test do not preserve the original audio stream
    has_audio = False
    session = _session_id(master_key, ident["width"], ident["height"], ident["fps"], ident["frames"], has_audio)
    with open(src_payload_path, "r", encoding="utf-8") as fin, open(dst_payload_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line_s = line.strip()
            if not line_s:
                continue
            obj = json.loads(line_s)
            t = str(obj.get("type", ""))
            if t == "meta":
                obj = dict(obj)
                obj["session_id"] = session
                obj["width"] = int(ident["width"])
                obj["height"] = int(ident["height"])
                obj["fps"] = float(ident["fps"])
                obj["frames"] = int(ident["frames"])
                obj["has_audio"] = False
                if not obj.get("video_preview_mode"):
                    obj["video_preview_mode"] = "chaos"
                if not obj.get("audio_preview_mode"):
                    obj["audio_preview_mode"] = "chaos"
                fout.write(json.dumps(obj) + "\n")
                continue
            # drop audio-tail / audio-specific records because attacked videos are rewritten without audio
            if t in {"tail_audio", "audio_meta", "audio_chunk", "audio"}:
                continue
            if obj.get("stream") == "audio" or obj.get("kind") == "audio":
                continue
            fout.write(json.dumps(obj) + "\n")
    return dst_payload_path


def _write_manifest(path: str, rec: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, sort_keys=True)


def _read_manifest(path: Optional[str]) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _candidate_roi_sidecars(search_dirs: List[Path], fallback_roi: Optional[str]) -> List[str]:
    seen = set()
    out: List[str] = []

    def add(p: Path):
        try:
            rp = str(p.resolve())
        except Exception:
            rp = str(p)
        if rp in seen or not p.is_file():
            return
        if p.suffix.lower() != ".jsonl":
            return
        if "roi" not in p.name.lower():
            return
        seen.add(rp)
        out.append(rp)

    if fallback_roi:
        add(Path(fallback_roi))
    for root in search_dirs:
        if not root or not root.exists() or not root.is_dir():
            continue
        for pat in ("roi*.jsonl", "rois*.jsonl", "*roi*.jsonl"):
            for p in root.glob(pat):
                add(p)
    return out


def _resolve_roi_sidecar_from_manifest(manifest_path: Optional[str], fallback_roi: Optional[str], search_dirs: List[Path]) -> Optional[str]:
    manifest = _read_manifest(manifest_path)
    if not manifest:
        return fallback_roi

    target_name = Path(str(manifest.get("roi_sidecar_path") or "")).name.lower()
    target_sha = str(manifest.get("roi_sidecar_sha256") or "").strip().lower()
    candidates = _candidate_roi_sidecars(search_dirs, fallback_roi)
    if not candidates:
        return fallback_roi

    def sha(path: str) -> str:
        try:
            return _sha256_file(path).lower()
        except Exception:
            return ""

    if target_name and target_sha:
        for c in candidates:
            if Path(c).name.lower() == target_name and sha(c) == target_sha:
                return c
    if target_sha:
        for c in candidates:
            if sha(c) == target_sha:
                return c
    if target_name:
        for c in candidates:
            if Path(c).name.lower() == target_name:
                return c
    return fallback_roi

def _build_decrypt_manifest(manifest_path: str, encrypted_video_path: str, roi_sidecar_path: Optional[str], payload_path: str, master_key: str) -> str:
    header = _load_payload_header(payload_path)
    ident = _video_identity(encrypted_video_path)
    has_audio = bool(header.get("has_audio", False))
    manifest = {
        "version": 1,
        "session_id": _session_id(master_key, ident["width"], ident["height"], ident["fps"], ident["frames"], has_audio),
        "width": int(ident["width"]),
        "height": int(ident["height"]),
        "fps": float(ident["fps"]),
        "frames": int(ident["frames"]),
        "has_audio": bool(has_audio),
        "encrypted_video_path": os.path.basename(encrypted_video_path),
        "roi_sidecar_path": os.path.basename(roi_sidecar_path) if roi_sidecar_path else None,
        "payload_path": os.path.basename(payload_path),
        "encrypted_video_sha256": _sha256_file(encrypted_video_path),
        "roi_sidecar_sha256": _sha256_file(roi_sidecar_path) if roi_sidecar_path else None,
        "payload_sha256": _sha256_file(payload_path),
    }
    _write_manifest(manifest_path, manifest)
    return manifest_path


def _framework_cli_decrypt_with_modes(framework_path: str, in_path: str, out_path: str, master_key: str,
                                      roi_sidecar: Optional[str] = None, payload: Optional[str] = None,
                                      manifest: Optional[str] = None, framework_base_path: Optional[str] = None,
                                      video_preview_mode: Optional[str] = None, audio_preview_mode: Optional[str] = None) -> None:
    cmd = [sys.executable, str(Path(framework_path).resolve()), "--mode", "decrypt", "--in", in_path, "--out", out_path, "--key", master_key]
    if roi_sidecar:
        cmd.extend(["--roi_sidecar", roi_sidecar])
    if payload:
        cmd.extend(["--payload", payload])
    if manifest:
        cmd.extend(["--manifest", manifest])
    if framework_base_path:
        cmd.extend(["--base_framework", framework_base_path])
    if video_preview_mode:
        cmd.extend(["--video_preview_mode", video_preview_mode])
    if audio_preview_mode:
        cmd.extend(["--audio_preview_mode", audio_preview_mode])

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        stdout_tail = (proc.stdout or "")[-4000:]
        stderr_tail = (proc.stderr or "")[-4000:]
        msg = (
            f"Decrypt subprocess failed with code {proc.returncode}.\n"
            f"Command: {' '.join(cmd)}\n"
            f"--- stdout tail ---\n{stdout_tail}\n"
            f"--- stderr tail ---\n{stderr_tail}"
        )
        raise RuntimeError(msg)


def resolve_logic_framework_path(framework_exec_path: str, framework_base_path: Optional[str] = None) -> str:
    if framework_base_path:
        return str(Path(framework_base_path).resolve())
    try:
        mod = _import_framework(framework_exec_path)
        if hasattr(mod, 'unpack_mask'):
            return str(Path(framework_exec_path).resolve())
    except Exception:
        pass
    fp = Path(framework_exec_path).resolve()
    for name in ('framework_succesful.py', 'framework_faster.py'):
        cand = fp.with_name(name)
        if cand.exists():
            return str(cand)
    return str(Path(framework_exec_path).resolve())


def call_framework_process_video(ff_mod, in_path: str, out_path: str, master_key: str, mode: str, roi_sidecar: Optional[str], payload: Optional[str] = None, manifest: Optional[str] = None, base_mod=None, framework_exec_path: Optional[str] = None, framework_base_path: Optional[str] = None, derive_payload_contract: bool = False):
    payload_use = payload
    manifest_path = manifest
    video_preview_mode, audio_preview_mode = _resolve_preview_modes(payload)

    if payload and mode == "decrypt":
        if derive_payload_contract:
            payload_use = str(Path(out_path).with_suffix(Path(out_path).suffix + ".verify.payload.jsonl"))
            _clone_payload_for_video(payload, payload_use, in_path, master_key)
            video_preview_mode, audio_preview_mode = _resolve_preview_modes(payload_use)
            manifest_path = str(Path(out_path).with_suffix(Path(out_path).suffix + ".verify.manifest.json"))
            _build_decrypt_manifest(manifest_path, in_path, roi_sidecar, payload_use, master_key)
        elif not manifest_path:
            manifest_path = _manifest_default_path(payload_use)

    if framework_exec_path and mode == "decrypt":
        _framework_cli_decrypt_with_modes(
            framework_path=str(Path(framework_exec_path).resolve()),
            in_path=in_path,
            out_path=out_path,
            master_key=master_key,
            roi_sidecar=roi_sidecar,
            payload=payload_use,
            manifest=manifest_path,
            framework_base_path=framework_base_path,
            video_preview_mode=video_preview_mode,
            audio_preview_mode=audio_preview_mode,
        )
        return

    sig = inspect.signature(ff_mod.process_video)
    kwargs = {
        "in_path": in_path,
        "out_path": out_path,
        "master_key": master_key,
        "mode": mode,
        "roi_sidecar_path": roi_sidecar,
        "payload_path": payload_use,
        "payload": payload_use,
        "manifest_path": manifest_path,
        "keystream_dump_path": None,
        "video_preview_mode": video_preview_mode,
        "audio_preview_mode": audio_preview_mode,
        "base": base_mod,
        "detect_every": 1,
        "detect_width": 0,
    }
    kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return ff_mod.process_video(**kwargs)

import cv2
import numpy as np


# -------------------------
# Logging
# -------------------------

def _now():
    return datetime.datetime.now().strftime("%H:%M:%S")

def vprint(verbose: bool, msg: str):
    if verbose:
        print(f"[{_now()}] {msg}", flush=True)

class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()
    def elapsed(self) -> float:
        return float(time.perf_counter() - self.t0)

def _eta(done: int, total: int, elapsed_s: float) -> str:
    if done <= 0 or elapsed_s <= 0:
        return "ETA: ?"
    rate = done / elapsed_s
    if rate <= 0:
        return "ETA: ?"
    rem = max(0, total - done) / rate
    return f"ETA {rem:,.1f}s @ {rate:,.2f} fps"


# -------------------------
# Video IO
# -------------------------

def video_info(path: str) -> Dict[str, object]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "fourcc": int(cap.get(cv2.CAP_PROP_FOURCC) or 0),
        "file_bytes": int(os.path.getsize(path)),
    }
    cap.release()
    return info

def iter_frames(path: str, max_frames: int, stride: int):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    idx = 0
    out_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if stride > 1 and (idx % stride) != 0:
            idx += 1
            continue
        yield out_idx, frame
        out_idx += 1
        idx += 1
        if max_frames and out_idx >= max_frames:
            break
    cap.release()

def write_video(frames: List[np.ndarray], out_path: str, fps: float, prefer_lossless: bool = True):
    if not frames:
        raise ValueError("No frames to write.")
    h, w = frames[0].shape[:2]
    out_path = str(out_path)

    if prefer_lossless:
        fourcc = cv2.VideoWriter_fourcc(*"FFV1")
        vw = cv2.VideoWriter(out_path, fourcc, fps if fps > 0 else 25.0, (w, h))
        if vw.isOpened():
            for fr in frames:
                if fr.shape[0] != h or fr.shape[1] != w:
                    fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
                vw.write(fr)
            vw.release()
            return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps if fps > 0 else 25.0, (w, h))
    if not vw.isOpened():
        raise RuntimeError("Could not open VideoWriter (FFV1 or mp4v).")
    for fr in frames:
        if fr.shape[0] != h or fr.shape[1] != w:
            fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
        vw.write(fr)
    vw.release()


# -------------------------
# ROI sidecar loader
# -------------------------

def _import_framework(framework_faster_path: str):
    resolved = Path(framework_faster_path).resolve()
    canonical_name = "framework_faster_robust"
    existing = sys.modules.get(canonical_name)
    if existing is not None and Path(getattr(existing, "__file__", "")).resolve() == resolved:
        return existing
    spec = importlib.util.spec_from_file_location(canonical_name, str(resolved))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import from {resolved}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[canonical_name] = mod
    spec.loader.exec_module(mod)
    return mod

def _build_orig_to_proc(max_frames: int, stride: int) -> Dict[int, int]:
    """
    Map original frame index -> processed frame index used by iter_frames() with stride.
    Only builds as far as needed for max_frames.
    """
    mapping: Dict[int, int] = {}
    proc = 0
    orig = 0
    while True:
        if stride <= 1 or (orig % stride) == 0:
            mapping[orig] = proc
            proc += 1
            if max_frames and proc >= max_frames:
                break
        orig += 1
        if max_frames and orig > max_frames * max(1, stride) + 10:
            break
    return mapping

def load_roi_masks(
    roi_sidecar_path: str,
    H: int,
    W: int,
    framework_faster_path: str,
    max_frames: int,
    stride: int,
) -> Dict[int, np.ndarray]:
    """
    Returns: processed_frame_idx -> combined ROI mask (bool HxW)
    """
    ff = _import_framework(framework_faster_path)
    if not hasattr(ff, "unpack_mask"):
        raise RuntimeError("framework_faster.py must provide unpack_mask(mask_pack, h, w)")

    orig_to_proc = _build_orig_to_proc(max_frames=max_frames, stride=stride)
    masks: Dict[int, np.ndarray] = {}

    with open(roi_sidecar_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            orig_idx = int(obj.get("frame_idx", -1))
            if orig_idx not in orig_to_proc:
                continue
            proc_idx = orig_to_proc[orig_idx]
            rois = obj.get("rois", []) or []
            m = np.zeros((H, W), dtype=bool)
            for r in rois:
                bbox = r.get("bbox", None)
                pack = r.get("mask_pack", None)
                if bbox is None or pack is None:
                    continue
                x1, y1, x2, y2 = [int(v) for v in bbox]
                x1 = max(0, min(W, x1)); x2 = max(0, min(W, x2))
                y1 = max(0, min(H, y1)); y2 = max(0, min(H, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                mh = y2 - y1
                mw = x2 - x1
                # framework_faster.py pack_mask() typically stores shape inside the dict.
                # Support both signatures:
                #   unpack_mask(obj: dict) -> mask
                #   unpack_mask(obj, h, w) -> mask
                import inspect
                if isinstance(pack, str):
                    try:
                        pack = json.loads(pack)
                    except Exception:
                        pass
                sig = inspect.signature(ff.unpack_mask)
                if len(sig.parameters) == 1:
                    m_local = ff.unpack_mask(pack)
                else:
                    m_local = ff.unpack_mask(pack, mh, mw)
                m[y1:y2, x1:x2] |= m_local
            masks[proc_idx] = m
    return masks


# -------------------------
# Metrics
# -------------------------

def to_gray_u8(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        g = img_bgr
    else:
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if g.dtype != np.uint8:
        g = np.clip(g, 0, 255).astype(np.uint8, copy=False)
    return g

def mse(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float64, copy=False) - b.astype(np.float64, copy=False)
    return float(np.mean(d * d))

def mae(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float64, copy=False) - b.astype(np.float64, copy=False)
    return float(np.mean(np.abs(d)))

def psnr(a: np.ndarray, b: np.ndarray, peak: float = 255.0) -> float:
    m = mse(a, b)
    if m <= 0:
        return float("inf")
    return float(10.0 * np.log10((peak * peak) / m))

def ssim_global(gray1: np.ndarray, gray2: np.ndarray, k1=0.01, k2=0.03, L=255.0) -> float:
    x = gray1.astype(np.float64, copy=False)
    y = gray2.astype(np.float64, copy=False)
    mu_x = float(np.mean(x)); mu_y = float(np.mean(y))
    sx = float(np.var(x)); sy = float(np.var(y))
    sxy = float(np.mean((x - mu_x) * (y - mu_y)))
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (sx + sy + c2)
    if den == 0:
        return 1.0 if num == 0 else 0.0
    return float(num / den)

def mean(vals: List[float]) -> float:
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def ssim_windowed(gray1: np.ndarray, gray2: np.ndarray, L: float = 255.0) -> float:
    I1 = gray1.astype(np.float32)
    I2 = gray2.astype(np.float32)
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    mu1 = cv2.GaussianBlur(I1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(I2, (11, 11), 1.5)
    mu1_2 = mu1 * mu1
    mu2_2 = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_2 = cv2.GaussianBlur(I1 * I1, (11, 11), 1.5) - mu1_2
    sigma2_2 = cv2.GaussianBlur(I2 * I2, (11, 11), 1.5) - mu2_2
    sigma12 = cv2.GaussianBlur(I1 * I2, (11, 11), 1.5) - mu1_mu2
    num = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)
    den = (mu1_2 + mu2_2 + C1) * (sigma1_2 + sigma2_2 + C2)
    ssim_map = num / np.maximum(den, 1e-12)
    return float(np.mean(ssim_map))


# -------------------------
# Attacks (support applying only on mask)
# -------------------------

def _apply_on_mask(base: np.ndarray, attacked: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    base: original frame
    attacked: fully attacked version
    mask: bool HxW ; True => take attacked pixels; False => keep base
    """
    if mask is None:
        return attacked
    out = base.copy()
    if out.ndim == 3:
        out[mask] = attacked[mask]
    else:
        out[mask] = attacked[mask]
    return out

def attack_gaussian(frame: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=frame.shape).astype(np.float32)
    return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def attack_salt_pepper(frame: np.ndarray, prob: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = frame.copy()
    h, w = out.shape[:2]
    m = rng.random((h, w))
    salt = m < (prob / 2.0)
    pepper = m > (1.0 - prob / 2.0)
    out[salt] = 255
    out[pepper] = 0
    return out


def attack_jpeg_recompress(frame: np.ndarray, crf_like: float) -> np.ndarray:
    """JPEG proxy for JPEG/CRF-style recompression robustness.

    OpenCV does not expose H.264 CRF per frame, so we map lower CRF to higher
    JPEG quality and higher CRF to lower quality. The scenario name preserves
    the CRF-like level for paper reporting.
    """
    crf = float(crf_like)
    quality = int(round(np.clip(100.0 - 2.0 * crf, 5.0, 95.0)))
    ok, enc = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return frame.copy()
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if dec is None or dec.shape != frame.shape:
        return frame.copy()
    return dec

def attack_rect_occlusion(
    frame: np.ndarray,
    frac: float,
    seed: int,
    fill: str,
    roi_mask: Optional[np.ndarray],
    scope: str,
) -> np.ndarray:
    """
    scope: whole|roi|background determines where rectangle is placed.
    If roi_mask is None, falls back to whole.
    Rectangle area is computed relative to target region area (ROI area if scope=roi, background area if background, else full).
    """
    rng = np.random.default_rng(seed)
    h, w = frame.shape[:2]
    out = frame.copy()

    if roi_mask is None:
        scope = "whole"

    if scope == "whole":
        target = np.ones((h, w), dtype=bool)
    elif scope == "roi":
        target = roi_mask
    elif scope == "background":
        target = ~roi_mask
    else:
        raise ValueError("scope")

    coords = np.argwhere(target)
    if coords.size == 0:
        return out

    target_area = coords.shape[0]
    area = max(1, int(target_area * frac))
    rh = max(1, int(np.sqrt(area)))
    rw = max(1, int(area / rh))
    rh = min(rh, h)
    rw = min(rw, w)

    # pick a random target pixel as anchor
    y0, x0 = coords[int(rng.integers(0, coords.shape[0]))]
    # choose top-left so that anchor lies inside the rectangle
    y = int(np.clip(y0 - rh // 2, 0, h - rh))
    x = int(np.clip(x0 - rw // 2, 0, w - rw))

    if fill == "black":
        out[y:y+rh, x:x+rw] = 0
    elif fill == "white":
        out[y:y+rh, x:x+rw] = 255
    elif fill == "mean":
        mu = int(np.mean(out))
        out[y:y+rh, x:x+rw] = mu
    elif fill == "random":
        patch = rng.integers(0, 256, size=(rh, rw, out.shape[2]), dtype=np.uint8)
        out[y:y+rh, x:x+rw] = patch
    else:
        raise ValueError("fill")
    return out


# -------------------------
# Orchestration
# -------------------------

@dataclass
class ScenarioScopeMetrics:
    scope: str
    frames_used: int
    mean_psnr: float
    mean_ssim_global: float
    mean_ssim_windowed: float
    mean_mse: float
    mean_mae: float

@dataclass
class ScenarioResult:
    scenario: str
    attack_scope: str
    attacked_cipher_path: str
    decrypted_path: str
    scopes: List[ScenarioScopeMetrics]
    decrypt_ok: bool = True
    decrypt_error: Optional[str] = None

def _eval_scopes(pg: np.ndarray, dg: np.ndarray, mask: Optional[np.ndarray], scopes: List[str]) -> Dict[str, Tuple[float,float,float,float,float]]:
    """
    Return dict scope -> (psnr, ssim_global, ssim_windowed, mse, mae) for one frame.
    """
    h, w = pg.shape
    if mask is None:
        mask = np.ones((h, w), dtype=bool)
    out = {}
    for sc in scopes:
        if sc == "whole":
            use = np.ones((h, w), dtype=bool)
        elif sc == "roi":
            use = mask
        elif sc == "background":
            use = ~mask
        else:
            continue
        if not np.any(use):
            out[sc] = (float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
            continue
        a = pg[use]
        b = dg[use]
        out[sc] = (psnr(a, b), ssim_global(a, b), ssim_windowed(a, b), mse(a, b), mae(a, b))
    return out

def run_scenario(
    name: str,
    cipher_frames: List[np.ndarray],
    roi_masks: Optional[Dict[int, np.ndarray]],
    attack_kind: str,
    attack_param: float,
    attack_scope: str,
    occl_fill: str,
    frame_loss_fill: str,
    fps: float,
    plain_path: str,
    out_dir: Path,
    ff_mod,

    master_key: str,
    roi_sidecar_path_for_decrypt: Optional[str],
    payload_path_for_decrypt: Optional[str],
    manifest_path_for_decrypt: Optional[str],
    base_mod,
    framework_exec_path: Optional[str],
    framework_base_path: Optional[str],
    max_frames: int,
    stride: int,
    eval_scopes: List[str],
    seed0: int,
    verbose: bool,
    attacked_path_override: Optional[str] = None,
) -> ScenarioResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    attacked_path = attacked_path_override or str(out_dir / f"attacked_{name}.mkv")
    dec_path = str(out_dir / f"decrypted_from_{name}.mkv")

    # Build attacked frames
    attacked_frames: List[np.ndarray] = []
    for i, fr in enumerate(cipher_frames):
        mask = roi_masks.get(i) if roi_masks is not None else None
        # choose region to modify based on attack_scope
        if attack_scope == "whole" or mask is None:
            region_mask = None
        elif attack_scope == "roi":
            region_mask = mask
        elif attack_scope == "background":
            region_mask = ~mask
        else:
            raise ValueError("attack_scope")

        if attack_kind == "baseline":
            out_fr = fr
        elif attack_kind == "gaussian":
            full_att = attack_gaussian(fr, sigma=attack_param, seed=seed0 + i)
            out_fr = _apply_on_mask(fr, full_att, region_mask)
        elif attack_kind == "saltpepper":
            full_att = attack_salt_pepper(fr, prob=attack_param, seed=seed0 + 1000 + i)
            out_fr = _apply_on_mask(fr, full_att, region_mask)
        elif attack_kind == "occlusion":
            # occlusion placement depends on scope (whole/roi/background)
            out_fr = attack_rect_occlusion(fr, frac=attack_param, seed=seed0 + 2000 + i, fill=occl_fill,
                                          roi_mask=mask, scope=attack_scope)
        elif attack_kind == "jpeg_crf":
            full_att = attack_jpeg_recompress(fr, crf_like=attack_param)
            out_fr = _apply_on_mask(fr, full_att, region_mask)
        elif attack_kind == "frame_loss":
            rng = np.random.default_rng(seed0 + 3000 + i)
            lost = bool(rng.random() < float(attack_param))
            if lost:
                if frame_loss_fill == "black" or i == 0:
                    repl = np.zeros_like(fr)
                elif frame_loss_fill in {"previous", "freeze"}:
                    repl = attacked_frames[-1].copy()
                else:
                    repl = np.zeros_like(fr)
                out_fr = _apply_on_mask(fr, repl, region_mask)
            else:
                out_fr = fr
        else:
            raise ValueError("attack_kind")
        attacked_frames.append(out_fr)

    if attacked_path_override is None:
        write_video(attacked_frames, attacked_path, fps=fps, prefer_lossless=True)
        vprint(verbose, f"Wrote attacked video: {attacked_path}")
    else:
        vprint(verbose, f"Using existing cipher as attacked input: {attacked_path}")

    # Decrypt attacked video through compatibility wrapper so proxy and base frameworks both work
    vprint(verbose, f"Decrypting attacked cipher -> {dec_path}")
    decrypt_ok = True
    decrypt_error = None
    try:
        call_framework_process_video(
            ff_mod,
            attacked_path,
            dec_path,
            master_key,
            mode="decrypt",
            roi_sidecar=roi_sidecar_path_for_decrypt,
            payload=payload_path_for_decrypt,
            manifest=manifest_path_for_decrypt,
            base_mod=base_mod,
            framework_exec_path=framework_exec_path,
            framework_base_path=framework_base_path,
            derive_payload_contract=(attacked_path_override is None),
        )
    except Exception as e:
        decrypt_ok = False
        decrypt_error = str(e)[-2000:]

    # Compare decrypted vs plain (scoped)
    if not decrypt_ok:
        scopes_out = [ScenarioScopeMetrics(scope=sc, frames_used=0, mean_psnr=float("nan"), mean_ssim_global=float("nan"), mean_ssim_windowed=float("nan"), mean_mse=float("nan"), mean_mae=float("nan")) for sc in eval_scopes]
        return ScenarioResult(scenario=name, attack_scope=attack_scope, attacked_cipher_path=attacked_path, decrypted_path=dec_path, scopes=scopes_out, decrypt_ok=False, decrypt_error=decrypt_error)

    per_scope = {sc: {"psnr": [], "ssim_global": [], "ssim_windowed": [], "mse": [], "mae": []} for sc in eval_scopes}

    t = Timer()
    total = max_frames if max_frames else 10**9
    total = min(total, int(video_info(plain_path)["frames"]))
    total = max(1, total)

    vprint(verbose, f"Comparing plain vs decrypted for scenario {name} (eval_scopes={eval_scopes}) ...")
    for (i, pfr), (_, dfr) in zip(iter_frames(plain_path, max_frames=max_frames, stride=stride),
                                 iter_frames(dec_path, max_frames=max_frames, stride=stride)):
        pg = to_gray_u8(pfr)
        dg = to_gray_u8(dfr)
        mask = roi_masks.get(i) if roi_masks is not None else None
        frame_vals = _eval_scopes(pg, dg, mask, eval_scopes)
        for sc, (pp, ssg, ssw, mm, aa) in frame_vals.items():
            per_scope[sc]["psnr"].append(pp)
            per_scope[sc]["ssim_global"].append(ssg)
            per_scope[sc]["ssim_windowed"].append(ssw)
            per_scope[sc]["mse"].append(mm)
            per_scope[sc]["mae"].append(aa)

        if verbose and (i == 0 or (i+1) % 50 == 0):
            vprint(True, f"  frame {i+1}/{total} {_eta(i+1, total, t.elapsed())}")

    scopes_out = []
    for sc in eval_scopes:
        scopes_out.append(ScenarioScopeMetrics(
            scope=sc,
            frames_used=len(per_scope[sc]["psnr"]),
            mean_psnr=mean(per_scope[sc]["psnr"]),
            mean_ssim_global=mean(per_scope[sc]["ssim_global"]),
            mean_ssim_windowed=mean(per_scope[sc]["ssim_windowed"]),
            mean_mse=mean(per_scope[sc]["mse"]),
            mean_mae=mean(per_scope[sc]["mae"]),
        ))

    return ScenarioResult(
        scenario=name,
        attack_scope=attack_scope,
        attacked_cipher_path=attacked_path,
        decrypted_path=dec_path,
        scopes=scopes_out,
        decrypt_ok=decrypt_ok,
        decrypt_error=decrypt_error,
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", required=True, help="original/plain video path")
    ap.add_argument("--cipher", required=True, help="encrypted/cipher video path")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--framework_faster_path", default="framework_proxy_payload_matched_removed_une.py", help="path to your encryption/decryption code")
    ap.add_argument("--master_key", required=True, help="master key for decryption")

    # ROI support
    ap.add_argument("--roi_sidecar", default=None, help="rois.jsonl from your encryption run. Strongly recommended: decryption uses reuse_rois=True so it does NOT need the detector model.")
    ap.add_argument("--payload", default=None, help="payload.jsonl for payload-aware decryption frameworks")
    ap.add_argument("--manifest", default=None, help="existing payload manifest for baseline decrypt and ROI contract resolution")
    ap.add_argument("--framework_base_path", default=None, help="base framework path for sidecar unpacking when the executable framework is payload-aware")
    ap.add_argument("--attack_scope", default="whole", choices=["whole","roi","background"], help="where corruption is applied on cipher frames")
    ap.add_argument("--eval_scope", default="whole", choices=["whole","roi","background","all"], help="where metrics are computed on plain vs decrypted")

    ap.add_argument("--max_frames", type=int, default=200)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--gaussian_sigmas", default="5,10,20,40", help="comma list of Gaussian sigmas")
    ap.add_argument("--sp_probs", default="0.001,0.005,0.01,0.05", help="comma list of salt&pepper probabilities")
    ap.add_argument("--jpeg_crfs", default="18,23,28,35", help="comma list of JPEG/CRF-like recompression levels")
    ap.add_argument("--frame_loss_probs", default="0.01,0.05,0.10", help="comma list of simulated packet/frame-loss probabilities")
    ap.add_argument("--frame_loss_fill", default="previous", choices=["previous","freeze","black"], help="replacement policy for simulated frame loss")
    ap.add_argument("--occl_fracs", default="0.10,0.25,0.50", help="comma list of occlusion area fractions (relative to target region)")
    ap.add_argument("--occl_fill", default="black", choices=["black","white","mean","random"])

    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    search_dirs: List[Path] = []
    for p in [args.cipher, args.roi_sidecar, args.manifest, args.payload]:
        if p:
            try:
                search_dirs.append(Path(p).resolve().parent)
            except Exception:
                pass
    args.roi_sidecar = _resolve_roi_sidecar_from_manifest(args.manifest, args.roi_sidecar, search_dirs)

    if DECRYPT_REQUIRES_ROI_SIDECAR and not args.roi_sidecar:
        raise SystemExit("ERROR: --roi_sidecar is required for robust decrypt (avoids detector). Provide the rois.jsonl produced during encryption.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    info_p = video_info(args.plain)
    info_c = video_info(args.cipher)
    fps = float(info_c["fps"]) if info_c["fps"] else float(info_p["fps"]) if info_p["fps"] else 25.0
    H = int(info_c["height"]); W = int(info_c["width"])

    vprint(args.verbose, f"Plain info: {info_p}")
    vprint(args.verbose, f"Cipher info: {info_c}")

    ff_mod = _import_framework(args.framework_faster_path)
    logic_framework_path = resolve_logic_framework_path(args.framework_faster_path, args.framework_base_path)
    base_mod = _import_framework(logic_framework_path) if logic_framework_path != str(Path(args.framework_faster_path).resolve()) else ff_mod
    if not hasattr(ff_mod, "process_video"):
        raise RuntimeError("framework_faster must expose process_video()")

    roi_masks = None
    if args.roi_sidecar:
        vprint(args.verbose, f"Loading ROI masks from sidecar: {args.roi_sidecar}")
        roi_masks = load_roi_masks(args.roi_sidecar, H=H, W=W, framework_faster_path=logic_framework_path,
                                   max_frames=args.max_frames, stride=args.stride)
        vprint(args.verbose, f"Loaded ROI masks for {len(roi_masks)} processed frames.")

    # Load cipher frames once
    vprint(args.verbose, "Loading cipher frames ...")
    cipher_frames = [fr for _, fr in iter_frames(args.cipher, max_frames=args.max_frames, stride=args.stride)]
    if not cipher_frames:
        raise RuntimeError("No frames read from cipher video.")

    eval_scopes = [args.eval_scope] if args.eval_scope != "all" else ["whole","roi","background"]
    gaussian_sigmas = [float(x) for x in args.gaussian_sigmas.split(",") if x.strip()]
    sp_probs = [float(x) for x in args.sp_probs.split(",") if x.strip()]
    occl_fracs = [float(x) for x in args.occl_fracs.split(",") if x.strip()]
    jpeg_crfs = [float(x) for x in args.jpeg_crfs.split(",") if x.strip()]
    frame_loss_probs = [float(x) for x in args.frame_loss_probs.split(",") if x.strip()]

    results: List[ScenarioResult] = []
    t_all = Timer()

    # Baseline
    vprint(args.verbose, "Scenario baseline ...")
    results.append(run_scenario(
        name="baseline",
        cipher_frames=cipher_frames,
        roi_masks=roi_masks,
        attack_kind="baseline",
        attack_param=0.0,
        attack_scope=args.attack_scope,
        occl_fill=args.occl_fill,
        frame_loss_fill=args.frame_loss_fill,
        fps=fps,
        plain_path=args.plain,
        out_dir=out_dir / "baseline",
        ff_mod=ff_mod,
        attacked_path_override=args.cipher,
        master_key=args.master_key,
        roi_sidecar_path_for_decrypt=args.roi_sidecar,
        payload_path_for_decrypt=args.payload,
        manifest_path_for_decrypt=args.manifest,
        base_mod=base_mod,
        framework_exec_path=args.framework_faster_path,
        framework_base_path=args.framework_base_path,
        max_frames=args.max_frames,
        stride=args.stride,
        eval_scopes=eval_scopes,
        seed0=args.seed,
        verbose=args.verbose,
    ))

    # Gaussian
    for sigma in gaussian_sigmas:
        name = f"gaussian_sigma{sigma:g}"
        vprint(args.verbose, f"Scenario {name} ...")
        results.append(run_scenario(
            name=name,
            cipher_frames=cipher_frames,
            roi_masks=roi_masks,
            attack_kind="gaussian",
            attack_param=sigma,
            attack_scope=args.attack_scope,
            occl_fill=args.occl_fill,
            frame_loss_fill=args.frame_loss_fill,
            fps=fps,
            plain_path=args.plain,
            out_dir=out_dir / name,
            ff_mod=ff_mod,
            master_key=args.master_key,
            roi_sidecar_path_for_decrypt=args.roi_sidecar,
            payload_path_for_decrypt=args.payload,
            manifest_path_for_decrypt=args.manifest,
            base_mod=base_mod,
            framework_exec_path=args.framework_faster_path,
            framework_base_path=args.framework_base_path,
            max_frames=args.max_frames,
            stride=args.stride,
            eval_scopes=eval_scopes,
            seed0=args.seed,
            verbose=args.verbose,
        ))

    # Salt & pepper
    for prob in sp_probs:
        name = f"saltpepper_p{prob:g}"
        vprint(args.verbose, f"Scenario {name} ...")
        results.append(run_scenario(
            name=name,
            cipher_frames=cipher_frames,
            roi_masks=roi_masks,
            attack_kind="saltpepper",
            attack_param=prob,
            attack_scope=args.attack_scope,
            occl_fill=args.occl_fill,
            frame_loss_fill=args.frame_loss_fill,
            fps=fps,
            plain_path=args.plain,
            out_dir=out_dir / name,
            ff_mod=ff_mod,
            master_key=args.master_key,
            roi_sidecar_path_for_decrypt=args.roi_sidecar,
            payload_path_for_decrypt=args.payload,
            manifest_path_for_decrypt=args.manifest,
            base_mod=base_mod,
            framework_exec_path=args.framework_faster_path,
            framework_base_path=args.framework_base_path,
            max_frames=args.max_frames,
            stride=args.stride,
            eval_scopes=eval_scopes,
            seed0=args.seed,
            verbose=args.verbose,
        ))

    # JPEG/CRF-like recompression
    for crf in jpeg_crfs:
        name = f"jpeg_crf{crf:g}"
        vprint(args.verbose, f"Scenario {name} ...")
        results.append(run_scenario(
            name=name,
            cipher_frames=cipher_frames,
            roi_masks=roi_masks,
            attack_kind="jpeg_crf",
            attack_param=crf,
            attack_scope=args.attack_scope,
            occl_fill=args.occl_fill,
            frame_loss_fill=args.frame_loss_fill,
            fps=fps,
            plain_path=args.plain,
            out_dir=out_dir / name,
            ff_mod=ff_mod,
            master_key=args.master_key,
            roi_sidecar_path_for_decrypt=args.roi_sidecar,
            payload_path_for_decrypt=args.payload,
            manifest_path_for_decrypt=args.manifest,
            base_mod=base_mod,
            framework_exec_path=args.framework_faster_path,
            framework_base_path=args.framework_base_path,
            max_frames=args.max_frames,
            stride=args.stride,
            eval_scopes=eval_scopes,
            seed0=args.seed,
            verbose=args.verbose,
        ))

    # Simulated packet/frame loss
    for prob in frame_loss_probs:
        name = f"frame_loss_p{prob:g}_{args.frame_loss_fill}"
        vprint(args.verbose, f"Scenario {name} ...")
        results.append(run_scenario(
            name=name,
            cipher_frames=cipher_frames,
            roi_masks=roi_masks,
            attack_kind="frame_loss",
            attack_param=prob,
            attack_scope=args.attack_scope,
            occl_fill=args.occl_fill,
            frame_loss_fill=args.frame_loss_fill,
            fps=fps,
            plain_path=args.plain,
            out_dir=out_dir / name,
            ff_mod=ff_mod,
            master_key=args.master_key,
            roi_sidecar_path_for_decrypt=args.roi_sidecar,
            payload_path_for_decrypt=args.payload,
            manifest_path_for_decrypt=args.manifest,
            base_mod=base_mod,
            framework_exec_path=args.framework_faster_path,
            framework_base_path=args.framework_base_path,
            max_frames=args.max_frames,
            stride=args.stride,
            eval_scopes=eval_scopes,
            seed0=args.seed,
            verbose=args.verbose,
        ))

    # Occlusion
    for frac in occl_fracs:
        name = f"occlusion_frac{int(frac*100)}_{args.occl_fill}"
        vprint(args.verbose, f"Scenario {name} ...")
        results.append(run_scenario(
            name=name,
            cipher_frames=cipher_frames,
            roi_masks=roi_masks,
            attack_kind="occlusion",
            attack_param=frac,
            attack_scope=args.attack_scope,
            occl_fill=args.occl_fill,
            frame_loss_fill=args.frame_loss_fill,
            fps=fps,
            plain_path=args.plain,
            out_dir=out_dir / name,
            ff_mod=ff_mod,
            master_key=args.master_key,
            roi_sidecar_path_for_decrypt=args.roi_sidecar,
            payload_path_for_decrypt=args.payload,
            manifest_path_for_decrypt=args.manifest,
            base_mod=base_mod,
            framework_exec_path=args.framework_faster_path,
            framework_base_path=args.framework_base_path,
            max_frames=args.max_frames,
            stride=args.stride,
            eval_scopes=eval_scopes,
            seed0=args.seed,
            verbose=args.verbose,
        ))

    report = {
        "focus": "Robustness to noise and occlusion (PDF V-G) with ROI-aware options",
        "tests_covered": {
            "baseline": "decrypt cipher without corruption and compare to plain",
            "gaussian_noise": {"sigmas": gaussian_sigmas, "description": "additive Gaussian noise"},
            "salt_pepper_noise": {"probs": sp_probs, "description": "impulse noise"},
            "jpeg_crf_recompression": {"crfs": jpeg_crfs, "description": "JPEG proxy for JPEG/CRF recompression"},
            "packet_frame_loss": {"probs": frame_loss_probs, "fill": args.frame_loss_fill, "description": "simulated frame loss / freeze or black replacement"},
            "rectangular_occlusion": {"fracs": occl_fracs, "fill": args.occl_fill, "description": "ROA-inspired rectangular sticker occlusion"},
            "attack_scope": args.attack_scope,
            "eval_scope": args.eval_scope,
            "metrics": ["PSNR", "SSIM_global", "SSIM_windowed", "MSE", "MAE", "decrypt_ok", "decrypt_error"],
        },
        "code_used": {
            "decrypt": f"{args.framework_faster_path} : process_video(mode='decrypt', roi_sidecar_path=..., reuse_rois=True when provided)",
            "roi_masks": f"{args.framework_faster_path} : unpack_mask(mask_pack,h,w) + sidecar JSONL parsing",
            "io": "OpenCV cv2.VideoCapture/VideoWriter",
            "attacks": "implemented here (Gaussian, salt&pepper, rectangular occlusion with optional ROI scope)",
        },
        "inputs": {"plain": args.plain, "cipher": args.cipher, "roi_sidecar": args.roi_sidecar, "manifest": args.manifest},
        "notes": [],
        "params": {"max_frames": args.max_frames, "stride": args.stride, "seed": args.seed},
        "results": [asdict(r) for r in results],
        "total_elapsed_s": t_all.elapsed(),
    }

    baseline = next((r for r in report["results"] if r.get("scenario") == "baseline"), None)
    if baseline:
        warn = []
        for sc in baseline.get("scopes", []):
            if (sc.get("mean_psnr") is not None and sc.get("mean_psnr", 0) < 30.0) or (sc.get("mean_ssim_windowed") is not None and sc.get("mean_ssim_windowed", 0) < 0.95):
                warn.append(f"Baseline reconstruction for scope={sc.get('scope')} is weak (PSNR={sc.get('mean_psnr'):.3f}, SSIM_windowed={sc.get('mean_ssim_windowed'):.6f}); attack robustness conclusions should be treated cautiously.")
        report["notes"].append("Decrypt uses reuse_rois=True with roi_sidecar to avoid running the detector model during robustness tests.")
        report["notes"].extend(warn)
    else:
        report["notes"].append("Decrypt uses reuse_rois=True with roi_sidecar to avoid running the detector model during robustness tests.")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    vprint(args.verbose, f"Wrote {out_dir/'report.json'} (elapsed {t_all.elapsed():.2f}s)")

if __name__ == "__main__":
    main()
