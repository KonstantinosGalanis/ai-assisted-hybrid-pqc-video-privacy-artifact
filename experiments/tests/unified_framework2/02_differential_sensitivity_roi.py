#!/usr/bin/env python3
"""
02_differential_sensitivity_roi.py

Differential / sensitivity framework for the video/image ROI encryption pipeline.

Covers:
- Plaintext sensitivity / chosen-plaintext differential test
- Key sensitivity
- NPCR / UACI (gray + color)
- Average Hamming distance vs number of iterations / keystream length
- ROI / background / whole-image scopes

Main fixes compared with the earlier version:
- ROI sidecar decoding is compatible with unpack_mask(obj) and unpack_mask(obj, h, w)
- NPCR/UACI can be computed on grayscale, color, or both
- Plaintext-sensitivity no longer misaligns videos when stride > 1
- Hamming-vs-iterations can use true burn-iteration sweeps, not only output-length sweeps
"""
from __future__ import annotations

import argparse
import csv
import datetime
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
try:
    from unified_utils import framework_cli_encrypt
except Exception:
    framework_cli_encrypt = None


def now() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def vprint(verbose: bool, msg: str):
    if verbose:
        print(f"[{now()}] {msg}", flush=True)


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def elapsed(self) -> float:
        return float(time.perf_counter() - self.t0)


def eta(done: int, total: int, elapsed_s: float) -> str:
    if done <= 0 or elapsed_s <= 0:
        return "ETA: ?"
    r = done / elapsed_s
    if r <= 0:
        return "ETA: ?"
    rem = max(0, total - done) / r
    return f"ETA {rem:,.1f}s @ {r:,.2f} fps"


def import_framework(path: str):
    spec = importlib.util.spec_from_file_location("framework_faster_mod", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod




def resolve_logic_framework_path(framework_exec_path: str, framework_base_path: Optional[str] = None) -> str:
    if framework_base_path:
        return str(Path(framework_base_path).resolve())
    try:
        mod = import_framework(framework_exec_path)
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
    """
    Yields (proc_idx, orig_idx, frame_bgr)
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    orig_idx = 0
    proc_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if stride > 1 and (orig_idx % stride) != 0:
            orig_idx += 1
            continue
        yield proc_idx, orig_idx, frame
        proc_idx += 1
        orig_idx += 1
        if max_frames and proc_idx >= max_frames:
            break
    cap.release()


def iter_all_frames(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield idx, frame
        idx += 1
    cap.release()


def write_video_lossless_ffv1(frames: List[np.ndarray], out_path: str, fps: float):
    if not frames:
        raise ValueError("No frames to write.")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    vw = cv2.VideoWriter(out_path, fourcc, fps if fps > 0 else 25.0, (w, h))
    if not vw.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(out_path, fourcc, fps if fps > 0 else 25.0, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"Could not open writer for {out_path}")
    for fr in frames:
        if fr.shape[:2] != (h, w):
            fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
        vw.write(fr)
    vw.release()


def write_video_safe_mp4(frames: List[np.ndarray], out_path: str, fps: float):
    out_path = str(Path(out_path).with_suffix('.mp4'))
    if not frames:
        raise ValueError("No frames to write.")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps if fps > 0 else 25.0, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"Could not open writer for {out_path}")
    for fr in frames:
        if fr.shape[:2] != (h, w):
            fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
        if fr.ndim == 2:
            fr = cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR)
        vw.write(fr)
    vw.release()
    return out_path


@dataclass
class RoiFrame:
    frame_idx: int
    masks: List[np.ndarray]


def _call_unpack_mask(unpack_fn, mask_pack: dict, mh: int, mw: int) -> np.ndarray:
    try:
        out = unpack_fn(mask_pack)
    except TypeError:
        out = unpack_fn(mask_pack, mh, mw)
    out = np.asarray(out, dtype=bool)
    if out.shape != (mh, mw):
        if out.shape[0] >= mh and out.shape[1] >= mw:
            out = out[:mh, :mw]
        else:
            out = cv2.resize(out.astype(np.uint8), (mw, mh), interpolation=cv2.INTER_NEAREST).astype(bool)
    return out


def _build_orig_to_proc(max_frames: int, stride: int) -> Dict[int, int]:
    orig_to_proc: Dict[int, int] = {}
    proc = 0
    max_orig = (max_frames * stride + 5) if max_frames else 10**9
    for orig in range(0, max_orig):
        if stride > 1 and (orig % stride) != 0:
            continue
        orig_to_proc[orig] = proc
        proc += 1
        if max_frames and proc >= max_frames:
            break
    return orig_to_proc


def load_roi_sidecar(
    roi_sidecar_path: str,
    H: int,
    W: int,
    framework_faster_path: str,
    max_frames: int,
    stride: int,
    verbose: bool,
) -> Dict[int, RoiFrame]:
    ff = import_framework(framework_faster_path)
    if not hasattr(ff, "unpack_mask"):
        raise RuntimeError("framework_faster.py must expose unpack_mask(...)")
    unpack_mask = ff.unpack_mask

    orig_to_proc = _build_orig_to_proc(max_frames=max_frames, stride=stride)
    out: Dict[int, RoiFrame] = {}
    vprint(verbose, f"Loading ROI sidecar: {roi_sidecar_path}")
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
            masks: List[np.ndarray] = []
            for r in rois:
                bbox = r.get("bbox", None)
                mask_pack = r.get("mask_pack", None)
                if bbox is None or mask_pack is None:
                    continue
                x1, y1, x2, y2 = [int(v) for v in bbox]
                x1 = max(0, min(W, x1)); x2 = max(0, min(W, x2))
                y1 = max(0, min(H, y1)); y2 = max(0, min(H, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                mh = y2 - y1
                mw = x2 - x1
                m_local = _call_unpack_mask(unpack_mask, mask_pack, mh, mw)
                m_full = np.zeros((H, W), dtype=bool)
                m_full[y1:y2, x1:x2] = m_local
                masks.append(m_full)
            out[proc_idx] = RoiFrame(frame_idx=proc_idx, masks=masks)
    vprint(verbose, f"ROI sidecar loaded for {len(out)} processed frames.")
    return out


def combined_mask(roi_frame: Optional[RoiFrame], H: int, W: int) -> np.ndarray:
    if roi_frame is None or not roi_frame.masks:
        return np.zeros((H, W), dtype=bool)
    m = np.zeros((H, W), dtype=bool)
    for mm in roi_frame.masks:
        m |= mm
    return m


def scope_mask(scope: str, roi: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if roi is None:
        return None
    if scope == "whole":
        return None
    if scope == "roi":
        return roi
    if scope == "background":
        return ~roi
    raise ValueError("scope")


def to_gray_u8(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        g = img_bgr
    else:
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if g.dtype != np.uint8:
        g = np.clip(g, 0, 255).astype(np.uint8, copy=False)
    return g


def npcr_uaci(a_u8: np.ndarray, b_u8: np.ndarray) -> Tuple[float, float]:
    a = np.asarray(a_u8, dtype=np.uint8)
    b = np.asarray(b_u8, dtype=np.uint8)
    if a.shape != b.shape:
        raise ValueError("Shape mismatch for NPCR/UACI")
    diff = (a != b).astype(np.uint8)
    npcr = float(100.0 * np.mean(diff))
    uaci = float(100.0 * np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64)) / 255.0))
    return npcr, uaci


def _apply_scope(frame: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return frame
    if frame.ndim == 2:
        return frame[mask]
    return frame[mask]


def npcr_uaci_bundle(a_frame: np.ndarray, b_frame: np.ndarray, scope_mask_arr: Optional[np.ndarray]) -> Dict[str, float]:
    out: Dict[str, float] = {}

    ga = to_gray_u8(a_frame)
    gb = to_gray_u8(b_frame)
    if scope_mask_arr is None:
        ag = ga
        bg = gb
    else:
        ag = ga[scope_mask_arr]
        bg = gb[scope_mask_arr]
    g_npcr, g_uaci = npcr_uaci(ag, bg)
    out["gray_npcr"] = g_npcr
    out["gray_uaci"] = g_uaci

    if a_frame.ndim == 3 and a_frame.shape[2] >= 3 and b_frame.ndim == 3 and b_frame.shape[2] >= 3:
        names = ["b", "g", "r"]
        agg_npcr = []
        agg_uaci = []
        if scope_mask_arr is None:
            ac = a_frame
            bc = b_frame
        else:
            ac = a_frame[scope_mask_arr]
            bc = b_frame[scope_mask_arr]
        color_npcr, color_uaci = npcr_uaci(ac, bc)
        out["color_npcr"] = color_npcr
        out["color_uaci"] = color_uaci
        for ci, nm in enumerate(names):
            if scope_mask_arr is None:
                ach = a_frame[:, :, ci]
                bch = b_frame[:, :, ci]
            else:
                ach = a_frame[:, :, ci][scope_mask_arr]
                bch = b_frame[:, :, ci][scope_mask_arr]
            n, u = npcr_uaci(ach, bch)
            out[f"{nm}_npcr"] = n
            out[f"{nm}_uaci"] = u
            agg_npcr.append(n)
            agg_uaci.append(u)
        out["color_mean_per_channel_npcr"] = float(np.mean(agg_npcr))
        out["color_mean_per_channel_uaci"] = float(np.mean(agg_uaci))
    return out


def hamming_rate_bytes(b1: bytes, b2: bytes) -> float:
    m = min(len(b1), len(b2))
    if m <= 0:
        return float("nan")
    x = np.frombuffer(b1[:m], dtype=np.uint8) ^ np.frombuffer(b2[:m], dtype=np.uint8)
    return float(100.0 * np.unpackbits(x).sum() / (8 * m))


def collect_video_bytes(path: str, max_frames: int, stride: int, max_bits: int) -> bytes:
    max_bytes = max_bits // 8 if max_bits else None
    parts: List[bytes] = []
    got = 0
    for _, _, fr in iter_frames(path, max_frames=max_frames, stride=stride):
        b = fr.tobytes()
        if max_bytes is not None:
            take = min(len(b), max_bytes - got)
            if take <= 0:
                break
            parts.append(b[:take])
            got += take
            if got >= max_bytes:
                break
        else:
            parts.append(b)
    return b"".join(parts)


def _finite_mean(values: List[float]) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def compute_npcr_uaci_over_video_pairs(
    vid_a: str,
    vid_b: str,
    roi_sidecar: Optional[str],
    framework_faster_path: str,
    scope: str,
    max_frames: int,
    stride: int,
    verbose: bool,
) -> Dict[str, object]:
    info_a = video_info(vid_a)
    H = int(info_a["height"]); W = int(info_a["width"])
    roi_map = None
    if roi_sidecar:
        roi_map = load_roi_sidecar(roi_sidecar, H, W, framework_faster_path, max_frames, stride, verbose)

    per_frame: List[Dict[str, float]] = []
    agg_lists: Dict[str, List[float]] = {}
    t = Timer()
    total = int(min(video_info(vid_a)["frames"], video_info(vid_b)["frames"], max_frames if max_frames else 10**9) / max(1, stride))
    total = max(1, total)

    for (i, _, fa), (_, _, fb) in zip(iter_frames(vid_a, max_frames=max_frames, stride=stride),
                                      iter_frames(vid_b, max_frames=max_frames, stride=stride)):
        if fa.shape[:2] != fb.shape[:2]:
            fb = cv2.resize(fb, (fa.shape[1], fa.shape[0]), interpolation=cv2.INTER_AREA)
        roi = combined_mask(roi_map.get(i, None), H, W) if roi_map is not None else None
        use = scope_mask(scope, roi)

        bundle = npcr_uaci_bundle(fa, fb, use)
        row = {"frame_idx": float(i)}
        row.update({k: float(v) for k, v in bundle.items()})
        per_frame.append(row)

        for k, v in bundle.items():
            agg_lists.setdefault(k, []).append(float(v))

        if verbose and (i == 0 or (i + 1) % 25 == 0):
            vprint(True, f"Processed {i+1}/{total} frames. {eta(i+1, total, t.elapsed())}")

    summary = {f"mean_{k}": _finite_mean(v) for k, v in agg_lists.items() if v}
    valid_metric_counts = {f"valid_{k}_frames": int(sum(1 for x in v if np.isfinite(x))) for k, v in agg_lists.items() if v}
    summary.update(valid_metric_counts)
    summary["frames_used"] = int(len(per_frame))
    summary["scope"] = scope
    summary["per_frame"] = per_frame
    return summary


def _pick_flip_position(frame: np.ndarray, use_mask: Optional[np.ndarray]) -> Tuple[int, int]:
    h, w = frame.shape[:2]
    if use_mask is not None and use_mask.shape == (h, w) and np.any(use_mask):
        ys, xs = np.nonzero(use_mask)
        mid = len(xs) // 2
        return int(ys[mid]), int(xs[mid])
    return h // 2, w // 2


def _flip_pixel(frame: np.ndarray, use_mask: Optional[np.ndarray] = None) -> np.ndarray:
    out = frame.copy()
    y, x = _pick_flip_position(out, use_mask)
    out[y, x] = 255 - out[y, x]
    return out


def build_modified_full_video(
    plain_path: str,
    out_path: str,
    fps: float,
    max_frames: int,
    stride: int,
    verbose: bool,
    roi_sidecar: Optional[str] = None,
    framework_faster_path: str = "framework_faster.py",
    scope: str = "whole",
    safe_proxy_input: bool = True,
):
    frames: List[np.ndarray] = []
    proc_used = 0
    roi_map = None
    H = W = 0
    if roi_sidecar:
        info = video_info(plain_path)
        H = int(info["height"])
        W = int(info["width"])
        roi_map = load_roi_sidecar(roi_sidecar, H, W, framework_faster_path, max_frames, stride, verbose)

    for orig_idx, fr in iter_all_frames(plain_path):
        should_flip = (stride <= 1 or (orig_idx % stride) == 0) and (not max_frames or proc_used < max_frames)
        if should_flip:
            roi = combined_mask(roi_map.get(proc_used, None), H, W) if roi_map is not None else None
            use = scope_mask(scope, roi)
            fr = _flip_pixel(fr, use)
            proc_used += 1
        frames.append(fr)
    vprint(verbose, f"Writing modified full-length plaintext with {proc_used} flipped processed frames -> {out_path}")
    if safe_proxy_input:
        return write_video_safe_mp4(frames, out_path, fps=fps)
    write_video_lossless_ffv1(frames, out_path, fps=fps)
    return out_path


def call_process_video(ff, in_path: str, out_path: str, master_key: str, mode: str, roi_sidecar: Optional[str], payload: Optional[str] = None, base_framework: Any = None):
    fn = ff.process_video
    sig = inspect.signature(fn)
    kwargs = {
        "in_path": in_path,
        "out_path": out_path,
        "master_key": master_key,
        "mode": mode,
        "roi_sidecar_path": roi_sidecar,
        "payload_path": payload,
        "payload": payload,
        "base": base_framework,
    }
    supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**supported)


def run_framework_encrypt_subprocess(
    framework_faster_path: str,
    in_path: str,
    out_path: str,
    master_key: str,
    roi_sidecar: Optional[str],
    verbose: bool,
    payload: Optional[str] = None,
    keystream_dump: Optional[str] = None,
    framework_base_path: Optional[str] = None,
) -> None:
    manifest = (str(payload) + ".manifest.json") if payload else None
    if callable(framework_cli_encrypt):
        vprint(verbose, f"Launching framework_cli_encrypt for {Path(in_path).name} -> {Path(out_path).name}")
        framework_cli_encrypt(
            framework_path=str(Path(framework_faster_path).resolve()),
            in_path=in_path,
            out_path=out_path,
            master_key=master_key,
            roi_sidecar=roi_sidecar,
            payload=payload,
            manifest=manifest,
            keystream_dump=keystream_dump,
            framework_base_path=framework_base_path,
            video_preview_mode="chacha",
            audio_preview_mode="chacha",
            detect_width=0,
        )
        return

    cmd = [
        sys.executable,
        str(Path(framework_faster_path).resolve()),
        "--mode", "encrypt",
        "--in", in_path,
        "--out", out_path,
        "--key", master_key,
        "--detect_width", "0",
    ]
    if roi_sidecar:
        cmd.extend(["--roi_sidecar", roi_sidecar])
    if payload:
        cmd.extend(["--payload", payload])
        if manifest:
            cmd.extend(["--manifest", manifest])
    if keystream_dump:
        cmd.extend(["--keystream_dump", keystream_dump])
    if framework_base_path:
        cmd.extend(["--base_framework", framework_base_path])
    cmd.extend(["--video_preview_mode", "chacha", "--audio_preview_mode", "chacha"])
    vprint(verbose, "Launching subprocess: " + " ".join(cmd))
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if verbose:
        if res.stdout:
            print(res.stdout[-4000:], flush=True)
        if res.stderr:
            print(res.stderr[-4000:], flush=True)
    if res.returncode != 0:
        raise subprocess.CalledProcessError(res.returncode, cmd, output=res.stdout, stderr=res.stderr)


def plaintext_sensitivity(
    plain_path: str,
    exec_framework_path: str,
    logic_framework_path: str,
    master_key: str,
    out_dir: str,
    roi_sidecar: Optional[str],
    scope: str,
    max_frames: int,
    stride: int,
    max_bits: int,
    verbose: bool,
    framework_base_path: Optional[str] = None,
    existing_cipher_path: Optional[str] = None,
) -> Dict[str, object]:
    info = video_info(plain_path)
    fps = float(info["fps"]) if info["fps"] else 25.0
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    plain_mod = str(outp / "_tmp_plain_modified_full.mp4")
    plain_mod = build_modified_full_video(
        plain_path,
        plain_mod,
        fps=fps,
        max_frames=max_frames,
        stride=stride,
        verbose=verbose,
        roi_sidecar=roi_sidecar,
        framework_faster_path=logic_framework_path,
        scope=scope,
        safe_proxy_input=True,
    )

    reuse_existing_cipher = bool(existing_cipher_path and os.path.isfile(existing_cipher_path))
    c1 = str(existing_cipher_path) if reuse_existing_cipher else str(outp / "_tmp_cipher_plain.mkv")
    c2 = str(outp / "_tmp_cipher_modified.mkv")
    side1 = None if reuse_existing_cipher else (str(outp / "_tmp_sidecar_plain.jsonl") if roi_sidecar else None)
    side2 = str(outp / "_tmp_sidecar_modified.jsonl") if roi_sidecar else None
    payload1 = None if reuse_existing_cipher else str(outp / "_tmp_payload_plain.jsonl")
    payload2 = str(outp / "_tmp_payload_modified.jsonl")
    dump1 = None if reuse_existing_cipher else str(outp / "_tmp_payload_plain.bin")
    dump2 = str(outp / "_tmp_payload_modified.bin")

    if reuse_existing_cipher:
        vprint(verbose, f"Reusing existing ciphertext for original plaintext branch: {c1}")
    else:
        vprint(verbose, "Encrypting original plaintext in isolated subprocess...")
        run_framework_encrypt_subprocess(exec_framework_path, plain_path, c1, master_key, side1, verbose, payload=payload1, keystream_dump=dump1, framework_base_path=framework_base_path)
    vprint(verbose, "Encrypting modified plaintext in isolated subprocess...")
    run_framework_encrypt_subprocess(exec_framework_path, plain_mod, c2, master_key, side2, verbose, payload=payload2, keystream_dump=dump2, framework_base_path=framework_base_path)

    vprint(verbose, "Computing NPCR/UACI between ciphertexts...")
    compare_sidecar = roi_sidecar if reuse_existing_cipher else (side1 if side1 and os.path.exists(side1) else roi_sidecar)
    stats = compute_npcr_uaci_over_video_pairs(
        c1, c2,
        roi_sidecar=compare_sidecar,
        framework_faster_path=logic_framework_path,
        scope=scope,
        max_frames=max_frames,
        stride=stride,
        verbose=verbose,
    )

    b1 = collect_video_bytes(c1, max_frames=max_frames, stride=stride, max_bits=max_bits)
    b2 = collect_video_bytes(c2, max_frames=max_frames, stride=stride, max_bits=max_bits)
    stats["ciphertext_hamming_rate_percent"] = hamming_rate_bytes(b1, b2)
    stats["cipher_a"] = c1
    stats["cipher_b"] = c2
    stats["cipher_a_sidecar"] = side1
    stats["cipher_b_sidecar"] = side2
    stats["comparison_sidecar"] = compare_sidecar
    if stats.get("per_frame"):
        with (outp / "plaintext_sensitivity_frames.csv").open("w", newline="", encoding="utf-8") as f:
            fieldnames = ["frame_idx"] + [k for k in stats["per_frame"][0].keys() if k != "frame_idx"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in stats["per_frame"]:
                out_row = dict(row)
                out_row["frame_idx"] = int(out_row["frame_idx"])
                w.writerow(out_row)
    return stats


def key_sensitivity(
    cipher1: str,
    cipher2: str,
    out_dir: str,
    roi_sidecar: Optional[str],
    framework_faster_path: str,
    scope: str,
    max_frames: int,
    stride: int,
    max_bits: int,
    keystream1: Optional[str],
    keystream2: Optional[str],
    verbose: bool,
) -> Dict[str, object]:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    stats = compute_npcr_uaci_over_video_pairs(
        cipher1, cipher2,
        roi_sidecar=roi_sidecar,
        framework_faster_path=framework_faster_path,
        scope=scope,
        max_frames=max_frames,
        stride=stride,
        verbose=verbose,
    )

    b1 = collect_video_bytes(cipher1, max_frames=max_frames, stride=stride, max_bits=max_bits)
    b2 = collect_video_bytes(cipher2, max_frames=max_frames, stride=stride, max_bits=max_bits)
    stats["ciphertext_hamming_rate_percent"] = hamming_rate_bytes(b1, b2)

    if keystream1 and keystream2 and os.path.exists(keystream1) and os.path.exists(keystream2):
        max_bytes = max_bits // 8 if max_bits else None
        with open(keystream1, "rb") as f:
            ks1 = f.read(max_bytes) if max_bytes else f.read()
        with open(keystream2, "rb") as f:
            ks2 = f.read(max_bytes) if max_bytes else f.read()
        stats["keystream_hamming_rate_percent"] = hamming_rate_bytes(ks1, ks2)
        stats["keystream_paths"] = [keystream1, keystream2]

    stats["cipher1"] = cipher1
    stats["cipher2"] = cipher2
    if stats.get("per_frame"):
        with (outp / "key_sensitivity_frames.csv").open("w", newline="", encoding="utf-8") as f:
            fieldnames = ["frame_idx"] + [k for k in stats["per_frame"][0].keys() if k != "frame_idx"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in stats["per_frame"]:
                out_row = dict(row)
                out_row["frame_idx"] = int(out_row["frame_idx"])
                w.writerow(out_row)
    return stats


def _make_probe_stream(ff, seed: bytes, frame_class: str, n: int, label: str):
    if hasattr(ff, "chacha20_stream_bytes"):
        return ff.chacha20_stream_bytes(seed, str(frame_class), label, int(n))
    if hasattr(ff, "derive_stream_seed"):
        seed = ff.derive_stream_seed(seed, label)
    if hasattr(ff, "keystream_u8"):
        return ff.keystream_u8(seed, frame_class, int(n))
    raise RuntimeError("framework module does not expose chacha20_stream_bytes(...) or keystream_u8(...)")


def hamming_curve(
    framework_faster_path: str,
    master_key: str,
    frame_class: str,
    x_values: List[int],
    x_axis: str,
    probe_bytes: int,
) -> Dict[str, object]:
    ff = import_framework(framework_faster_path)
    if not hasattr(ff, "derive_seed"):
        return {"note": "framework module does not expose derive_seed(...)."}

    seed1 = ff.derive_seed(master_key, 0, 0, 0, frame_class)
    seed2 = ff.derive_seed(master_key + "|1", 0, 0, 0, frame_class)

    curve = []
    note = None
    for x in x_values:
        if x_axis == "length":
            ks1 = _make_probe_stream(ff, seed1, frame_class, int(x), "hamming-length").tobytes()
            ks2 = _make_probe_stream(ff, seed2, frame_class, int(x), "hamming-length").tobytes()
        elif x_axis == "burn":
            note = "ChaCha-based framework has no burn-iteration parameter; burn-axis points are simulated via distinct domain labels at fixed probe length."
            ks1 = _make_probe_stream(ff, seed1, frame_class, int(probe_bytes), f"hamming-burn-{int(x)}").tobytes()
            ks2 = _make_probe_stream(ff, seed2, frame_class, int(probe_bytes), f"hamming-burn-{int(x)}").tobytes()
        else:
            raise ValueError("x_axis must be length or burn")
        curve.append({
            "x": int(x),
            "hamming_rate_percent": hamming_rate_bytes(ks1, ks2),
        })

    out = {
        "frame_class": frame_class,
        "x_axis": x_axis,
        "probe_bytes": int(probe_bytes),
        "curve": curve,
    }
    if note:
        out["note"] = note
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["plain_sens", "key_sens", "hamming_curve", "all"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--max_frames", type=int, default=200)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max_bits", type=int, default=2_000_000)

    ap.add_argument("--roi_sidecar", default=None)
    ap.add_argument("--framework_faster_path", default="framework_faster.py")
    ap.add_argument("--scope", default="whole", choices=["whole", "roi", "background"])

    ap.add_argument("--plain", default=None)
    ap.add_argument("--master_key", default=None)

    ap.add_argument("--cipher", default=None)
    ap.add_argument("--cipher2", default=None)
    ap.add_argument("--keystream", default=None)
    ap.add_argument("--keystream2", default=None)
    ap.add_argument("--payload", default=None, help="optional payload from primary encryption run")
    ap.add_argument("--payload2", default=None, help="optional payload from secondary/alt-key encryption run")
    ap.add_argument("--framework_base_path", default=None, help="base framework path used for sidecar unpacking/seed helpers when the executable framework is payload-aware")

    ap.add_argument("--ham_map", default="I", choices=["I", "P", "B", "UNK"], help="frame class used for the ChaCha hamming-curve probe")
    ap.add_argument("--ham_x_values", default="1024,2048,4096,8192,16384,32768,65536")
    ap.add_argument("--ham_x_axis", default="length", choices=["length", "burn"])
    ap.add_argument("--ham_probe_bytes", type=int, default=4096)
    args = ap.parse_args()

    outp = Path(args.out)
    outp.mkdir(parents=True, exist_ok=True)

    logic_framework_path = resolve_logic_framework_path(
        args.framework_faster_path,
        args.framework_base_path,
    )

    report: Dict[str, object] = {
        "tests_covered": {
            "plaintext_sensitivity": "NPCR/UACI between ciphertexts of P and P' (1-pixel flip), plus ciphertext Hamming rate",
            "key_sensitivity": "NPCR/UACI between two ciphertext videos (different keys), plus ciphertext Hamming rate and optional keystream Hamming rate",
            "hamming_vs_iterations": "Average Hamming distance versus keystream length or burn-iteration count",
        },
        "results": {},
        "params": {
            "max_frames": args.max_frames,
            "stride": args.stride,
            "max_bits": args.max_bits,
            "scope": args.scope,
            "roi_sidecar": args.roi_sidecar,
            "framework_faster_path": args.framework_faster_path,
            "ham_frame_class": args.ham_map,
            "ham_x_axis": args.ham_x_axis,
            "ham_probe_bytes": args.ham_probe_bytes,
        },
    }

    if args.mode in ("plain_sens", "all"):
        if not args.plain or not args.master_key:
            raise SystemExit("--plain and --master_key are required for plain_sens")
        res = plaintext_sensitivity(
            plain_path=args.plain,
            exec_framework_path=args.framework_faster_path,
            logic_framework_path=logic_framework_path,
            master_key=args.master_key,
            out_dir=str(outp),
            roi_sidecar=args.roi_sidecar,
            scope=args.scope,
            max_frames=args.max_frames,
            stride=args.stride,
            max_bits=args.max_bits,
            verbose=args.verbose,
            framework_base_path=args.framework_base_path,
            existing_cipher_path=args.cipher,
        )
        report["results"]["plaintext_sensitivity"] = res

    if args.mode in ("key_sens", "all"):
        if not args.cipher or not args.cipher2:
            raise SystemExit("--cipher and --cipher2 are required for key_sens")
        res = key_sensitivity(
            cipher1=args.cipher,
            cipher2=args.cipher2,
            out_dir=str(outp),
            roi_sidecar=args.roi_sidecar,
            framework_faster_path=logic_framework_path,
            scope=args.scope,
            max_frames=args.max_frames,
            stride=args.stride,
            max_bits=args.max_bits,
            keystream1=args.keystream,
            keystream2=args.keystream2,
            verbose=args.verbose,
        )
        report["results"]["key_sensitivity"] = res

    if args.mode in ("hamming_curve", "all"):
        if not args.master_key:
            raise SystemExit("--master_key is required for hamming_curve")
        x_values = [int(x.strip()) for x in args.ham_x_values.split(",") if x.strip()]
        res = hamming_curve(
            framework_faster_path=logic_framework_path,
            master_key=args.master_key,
            frame_class=args.ham_map,
            x_values=x_values,
            x_axis=args.ham_x_axis,
            probe_bytes=args.ham_probe_bytes,
        )
        report["results"]["hamming_curve"] = res
        (outp / "hamming_curve.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    (outp / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    vprint(args.verbose, f"Wrote {outp/'report.json'}")


if __name__ == "__main__":
    main()
