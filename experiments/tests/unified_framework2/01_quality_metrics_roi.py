
"""
video_quality_framework_roi.py

Extends video_quality_framework.py with ROI-only / background-only evaluation using your ROI sidecar (rois.jsonl).

PDF items covered (VI-A/B/C/R):
- SNR
- PSNR
- SSIM
- MSE
- MAE

New features:
- --roi_sidecar rois.jsonl  (produced by your framework_faster.py)
- --scope whole|roi|background|all
- ROI mask decoding is compatible with both unpack_mask(mask_pack) and unpack_mask(mask_pack, h, w)

How ROI scope works:
- For MSE/MAE/PSNR/SNR: computed only on pixels in the selected scope.
- For SSIM: computed on masked images where pixels outside the scope are set to the scope mean
  (this preserves a 2D structure needed for SSIM while focusing comparison on ROI/background).

Outputs:
- report.json
- frame_metrics_<scope>.csv (or frame_metrics.csv if a single scope)

Examples:
  python3 video_quality_framework_roi.py --plain faces.mp4 --test encrypted.mkv --out results_q --mode gray --prefer_skimage --verbose
  python3 video_quality_framework_roi.py --plain faces.mp4 --test encrypted.mkv --roi_sidecar rois.jsonl --scope roi --out results_q_roi --mode gray --prefer_skimage --verbose
  python3 video_quality_framework_roi.py --plain faces.mp4 --test encrypted.mkv --roi_sidecar rois.jsonl --scope all --out results_q_all --mode color --verbose
"""
from __future__ import annotations

import argparse
import csv
import datetime
import importlib.util
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

# Optional: scikit-image metrics
_SKIMAGE_OK = False
try:
    from skimage.metrics import mean_squared_error as sk_mse
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    from skimage.metrics import structural_similarity as sk_ssim
    _SKIMAGE_OK = True
except Exception:
    _SKIMAGE_OK = False


# -------------------------
# Logging
# -------------------------

def _now() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")

def vprint(verbose: bool, msg: str):
    if verbose:
        print(f"[{_now()}] {msg}", flush=True)

class StageTimer:
    def __init__(self):
        self.t0 = time.perf_counter()
    def elapsed(self) -> float:
        return float(time.perf_counter() - self.t0)

def eta(done: int, total: int, elapsed_s: float) -> str:
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


# -------------------------
# ROI sidecar
# -------------------------

@dataclass
class RoiFrame:
    frame_idx: int
    mask: np.ndarray  # bool mask in full-frame coords (H,W)

def _load_unpack_mask(framework_faster_path: str):
    spec = importlib.util.spec_from_file_location("framework_faster_sidecar", framework_faster_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import framework_faster from {framework_faster_path}")
    ff = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ff)
    if not hasattr(ff, "unpack_mask"):
        raise RuntimeError("framework_faster.py must provide unpack_mask(...)")
    return ff.unpack_mask

def _call_unpack_mask(unpack_fn, mask_pack: dict, mh: int, mw: int) -> np.ndarray:
    """
    Compatibility wrapper:
    - framework_faster.py in this project exposes unpack_mask(obj)
    - some older variants expose unpack_mask(obj, h, w)
    """
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

def load_roi_sidecar(
    roi_sidecar_path: str,
    H: int,
    W: int,
    framework_faster_path: str,
    max_frames: int,
    stride: int,
    verbose: bool,
) -> Dict[int, RoiFrame]:
    """
    Returns dict: processed_frame_idx -> RoiFrame aligned to iter_frames() output index.
    Assumes sidecar uses original frame_idx. We map to processed idx via stride.
    """
    vprint(verbose, f"Loading ROI sidecar: {roi_sidecar_path}")
    unpack_mask = _load_unpack_mask(framework_faster_path)

    # map original idx -> processed idx under stride/max_frames
    orig_to_proc: Dict[int, int] = {}
    proc = 0
    orig = 0
    while True:
        if stride <= 1 or (orig % stride) == 0:
            orig_to_proc[orig] = proc
            proc += 1
            if max_frames and proc >= max_frames:
                break
        orig += 1
        if max_frames and orig > (max_frames * stride + 10):
            break

    out: Dict[int, RoiFrame] = {}
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
            m_full = np.zeros((H, W), dtype=bool)
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
                m_local = _call_unpack_mask(unpack_mask, mask_pack, mh, mw)  # bool
                m_full[y1:y2, x1:x2] |= m_local
            out[proc_idx] = RoiFrame(frame_idx=proc_idx, mask=m_full)

    vprint(verbose, f"ROI sidecar loaded for {len(out)} processed frames.")
    return out


# -------------------------
# Metric primitives
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

def psnr_from_mse(m: float, peak: float = 255.0) -> float:
    if m <= 0:
        return float("inf")
    return float(10.0 * np.log10((peak * peak) / m))

def snr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    noise = a - b
    sig_pow = float(np.mean(a * a))
    noise_pow = float(np.mean(noise * noise))
    if noise_pow <= 0:
        return float("inf")
    if sig_pow <= 0:
        return float("-inf")
    return float(10.0 * np.log10(sig_pow / noise_pow))

def mssim_opencv_like(img1: np.ndarray, img2: np.ndarray, L: float = 255.0) -> float:
    I1 = img1.astype(np.float32)
    I2 = img2.astype(np.float32)
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    mu1 = cv2.GaussianBlur(I1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(I2, (11, 11), 1.5)
    mu1_2 = mu1 * mu1
    mu2_2 = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_2 = cv2.GaussianBlur(I1 * I1, (11, 11), 1.5) - mu1_2
    sigma2_2 = cv2.GaussianBlur(I2 * I2, (11, 11), 1.5) - mu2_2
    sigma12  = cv2.GaussianBlur(I1 * I2, (11, 11), 1.5) - mu1_mu2
    t1 = 2.0 * mu1_mu2 + C1
    t2 = 2.0 * sigma12 + C2
    t3 = t1 * t2
    t1 = mu1_2 + mu2_2 + C1
    t2 = sigma1_2 + sigma2_2 + C2
    t1 = t1 * t2
    ssim_map = t3 / t1
    return float(np.mean(ssim_map))

def _masked_img(gray_u8: np.ndarray, use_mask: np.ndarray) -> np.ndarray:
    out = gray_u8.copy()
    mu = int(np.mean(out[use_mask])) if np.any(use_mask) else int(np.mean(out))
    out[~use_mask] = mu
    return out

def _select_scope_mask(scope: str, roi_mask: Optional[np.ndarray], H: int, W: int) -> np.ndarray:
    if scope == "whole":
        return np.ones((H, W), dtype=bool)
    if roi_mask is None:
        return np.ones((H, W), dtype=bool) if scope == "roi" else np.zeros((H, W), dtype=bool)
    if scope == "roi":
        return roi_mask
    if scope == "background":
        return ~roi_mask
    raise ValueError("scope must be whole|roi|background")


def compute_metrics_pair(ref: np.ndarray, test: np.ndarray, mode: str, prefer_skimage: bool, scope_mask: np.ndarray) -> Dict[str, float]:
    if mode == "gray":
        a2d = to_gray_u8(ref)
        b2d = to_gray_u8(test)
        if b2d.shape != a2d.shape:
            b2d = cv2.resize(b2d, (a2d.shape[1], a2d.shape[0]), interpolation=cv2.INTER_AREA)

        use = scope_mask
        if not np.any(use):
            return {"mse": float("nan"), "mae": float("nan"), "psnr": float("nan"), "snr": float("nan"), "ssim": float("nan")}

        a = a2d[use]; b = b2d[use]
        if prefer_skimage and _SKIMAGE_OK:
            m = float(sk_mse(a.astype(np.float64), b.astype(np.float64)))
            p = psnr_from_mse(m)
        else:
            m = mse(a, b); p = psnr_from_mse(m)

        A = _masked_img(a2d, use)
        B = _masked_img(b2d, use)
        if prefer_skimage and _SKIMAGE_OK:
            s = float(sk_ssim(A, B, data_range=255))
        else:
            s = mssim_opencv_like(A, B)

        return {"mse": m, "mae": mae(a, b), "psnr": p, "snr": snr(a, b), "ssim": s}

    # color
    A3 = ref; B3 = test
    if A3.ndim != 3 or A3.shape[2] < 3 or B3.ndim != 3 or B3.shape[2] < 3:
        return compute_metrics_pair(ref, test, "gray", prefer_skimage, scope_mask)
    if B3.shape[0] != A3.shape[0] or B3.shape[1] != A3.shape[1]:
        B3 = cv2.resize(B3, (A3.shape[1], A3.shape[0]), interpolation=cv2.INTER_AREA)

    use = scope_mask
    if not np.any(use):
        return {"mse": float("nan"), "mae": float("nan"), "psnr": float("nan"), "snr": float("nan"), "ssim": float("nan")}

    out: Dict[str, float] = {}
    names = ["b", "g", "r"]
    mses = []; maes = []; psnrs = []; snrs = []; ssims = []
    for ci, nm in enumerate(names):
        a2d = A3[:, :, ci].astype(np.uint8, copy=False)
        b2d = B3[:, :, ci].astype(np.uint8, copy=False)
        a = a2d[use]; b = b2d[use]
        if prefer_skimage and _SKIMAGE_OK:
            m = float(sk_mse(a.astype(np.float64), b.astype(np.float64))); p = psnr_from_mse(m)
            s = float(sk_ssim(_masked_img(a2d, use), _masked_img(b2d, use), data_range=255))
        else:
            m = mse(a, b); p = psnr_from_mse(m)
            s = mssim_opencv_like(_masked_img(a2d, use), _masked_img(b2d, use))
        out[f"mse_{nm}"] = m
        out[f"mae_{nm}"] = mae(a, b)
        out[f"psnr_{nm}"] = p
        out[f"snr_{nm}"] = snr(a, b)
        out[f"ssim_{nm}"] = s
        mses.append(m); maes.append(out[f"mae_{nm}"]); psnrs.append(p); snrs.append(out[f"snr_{nm}"]); ssims.append(s)

    out["mse"] = float(np.mean(mses))
    out["mae"] = float(np.mean(maes))
    out["psnr"] = float(np.mean(psnrs))
    out["snr"] = float(np.mean(snrs))
    out["ssim"] = float(np.mean(ssims))
    return out


@dataclass
class FrameMetrics:
    frame_idx: int
    scope: str
    mse: float
    mae: float
    psnr: float
    snr: float
    ssim: float
    mse_b: float = float("nan")
    mse_g: float = float("nan")
    mse_r: float = float("nan")
    mae_b: float = float("nan")
    mae_g: float = float("nan")
    mae_r: float = float("nan")
    psnr_b: float = float("nan")
    psnr_g: float = float("nan")
    psnr_r: float = float("nan")
    snr_b: float = float("nan")
    snr_g: float = float("nan")
    snr_r: float = float("nan")
    ssim_b: float = float("nan")
    ssim_g: float = float("nan")
    ssim_r: float = float("nan")


def summarize(values: List[float]) -> Dict[str, float]:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "median": float("nan")}
    arr = np.array(vals, dtype=np.float64)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "min": float(np.min(arr)), "max": float(np.max(arr)), "median": float(np.median(arr))}


def run_one_scope(
    plain_path: str,
    test_path: str,
    scope: str,
    roi_map: Optional[Dict[int, RoiFrame]],
    out_dir: str,
    max_frames: int,
    stride: int,
    mode: str,
    resize_test_to_plain: bool,
    prefer_skimage: bool,
    verbose: bool,
    progress_every: int,
) -> Dict[str, object]:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    info_plain = video_info(plain_path)
    info_test = video_info(test_path)
    H = int(info_plain["height"]); W = int(info_plain["width"])

    rows: List[FrameMetrics] = []
    t = StageTimer()
    total = int(min(info_plain["frames"], info_test["frames"], max_frames if max_frames else 10**9) / max(1, stride))
    total = max(1, total)

    for (i, pfr), (_, tfr) in zip(iter_frames(plain_path, max_frames=max_frames, stride=stride),
                                 iter_frames(test_path, max_frames=max_frames, stride=stride)):
        if resize_test_to_plain and (pfr.shape[0] != tfr.shape[0] or pfr.shape[1] != tfr.shape[1]):
            tfr = cv2.resize(tfr, (pfr.shape[1], pfr.shape[0]), interpolation=cv2.INTER_AREA)

        roi_mask = roi_map.get(i).mask if (roi_map is not None and i in roi_map) else (np.zeros((H, W), dtype=bool) if roi_map is not None else None)
        use_mask = _select_scope_mask(scope, roi_mask, H, W)

        m = compute_metrics_pair(pfr, tfr, mode=mode, prefer_skimage=prefer_skimage, scope_mask=use_mask)

        rows.append(FrameMetrics(
            frame_idx=i, scope=scope,
            mse=m["mse"], mae=m["mae"], psnr=m["psnr"], snr=m["snr"], ssim=m["ssim"],
            mse_b=m.get("mse_b", float("nan")), mse_g=m.get("mse_g", float("nan")), mse_r=m.get("mse_r", float("nan")),
            mae_b=m.get("mae_b", float("nan")), mae_g=m.get("mae_g", float("nan")), mae_r=m.get("mae_r", float("nan")),
            psnr_b=m.get("psnr_b", float("nan")), psnr_g=m.get("psnr_g", float("nan")), psnr_r=m.get("psnr_r", float("nan")),
            snr_b=m.get("snr_b", float("nan")), snr_g=m.get("snr_g", float("nan")), snr_r=m.get("snr_r", float("nan")),
            ssim_b=m.get("ssim_b", float("nan")), ssim_g=m.get("ssim_g", float("nan")), ssim_r=m.get("ssim_r", float("nan")),
        ))

        if verbose and (i == 0 or (i + 1) % max(1, progress_every) == 0):
            vprint(True, f"[{scope}] Frame {i+1}/{total} {eta(i+1, total, t.elapsed())}")

    csv_path = outp / f"frame_metrics_{scope}.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
            w.writeheader()
            for r in rows:
                w.writerow(asdict(r))

    return {
        "scope": scope,
        "frames_used": len(rows),
        "outputs": {"frame_metrics_csv": str(csv_path)},
        "aggregates": {
            "mse": summarize([r.mse for r in rows]),
            "mae": summarize([r.mae for r in rows]),
            "psnr": summarize([r.psnr for r in rows]),
            "snr": summarize([r.snr for r in rows]),
            "ssim": summarize([r.ssim for r in rows]),
        },
    }


def run_framework(
    plain_path: str,
    test_path: str,
    out_dir: str,
    max_frames: int,
    stride: int,
    mode: str,
    resize_test_to_plain: bool,
    prefer_skimage: bool,
    verbose: bool,
    progress_every: int,
    roi_sidecar: Optional[str],
    scope: str,
    framework_faster_path: str,
) -> Dict[str, object]:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    vprint(verbose, f"Starting quality+ROI framework. mode={mode} scope={scope} max_frames={max_frames} stride={stride}")
    vprint(verbose, f"skimage_available={_SKIMAGE_OK} prefer_skimage={prefer_skimage}")

    info_plain = video_info(plain_path)
    info_test = video_info(test_path)

    roi_map: Optional[Dict[int, RoiFrame]] = None
    if roi_sidecar:
        roi_map = load_roi_sidecar(
            roi_sidecar_path=roi_sidecar,
            H=int(info_plain["height"]),
            W=int(info_plain["width"]),
            framework_faster_path=framework_faster_path,
            max_frames=max_frames,
            stride=stride,
            verbose=verbose,
        )

    scopes = [scope] if scope != "all" else ["whole", "roi", "background"]
    out_scopes = {}
    for sc in scopes:
        out_scopes[sc] = run_one_scope(
            plain_path=plain_path,
            test_path=test_path,
            scope=sc,
            roi_map=roi_map,
            out_dir=out_dir,
            max_frames=max_frames,
            stride=stride,
            mode=mode,
            resize_test_to_plain=resize_test_to_plain,
            prefer_skimage=prefer_skimage,
            verbose=verbose,
            progress_every=progress_every,
        )

    report = {
        "tests_covered": ["SNR", "PSNR", "SSIM", "MSE", "MAE", "Pixel resemblance/disparity (dedicated section via MSE/MAE/PSNR/SSIM)"],
        "backend": {"skimage_available": _SKIMAGE_OK, "prefer_skimage": bool(prefer_skimage)},
        "inputs": {
            "plain_path": plain_path,
            "test_path": test_path,
            "mode": mode,
            "scope": scope,
            "roi_sidecar": roi_sidecar,
            "max_frames": max_frames,
            "stride": stride,
            "framework_faster_path": framework_faster_path,
        },
        "video_info": {"plain": info_plain, "test": info_test},
        "outputs": out_scopes,
        "pixel_resemblance_disparity": {
            "definition": "Dedicated pixel-level resemblance/disparity view derived from MSE, MAE, PSNR and SSIM over the selected scope(s).",
            "scopes": {k: v["aggregates"] for k, v in out_scopes.items()},
        },
    }

    (outp / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    vprint(verbose, f"Wrote {outp/'report.json'}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", required=True, help="reference/plain video")
    ap.add_argument("--test", required=True, help="test video (cipher/decrypted/compressed etc.)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--max_frames", type=int, default=300, help="process up to N frames (0=all)")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--mode", choices=["gray", "color"], default="gray", help="metric mode")
    ap.add_argument("--resize_test_to_plain", action="store_true", help="resize test frames to match plain if needed")
    ap.add_argument("--prefer_skimage", action="store_true", help="use skimage.metrics if available")
    ap.add_argument("--roi_sidecar", default=None, help="rois.jsonl from framework_faster.py")
    ap.add_argument("--scope", choices=["whole", "roi", "background", "all"], default="whole")
    ap.add_argument("--framework_faster_path", default="framework_faster.py", help="path to framework_faster.py (for unpack_mask)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--progress_every", type=int, default=25)
    args = ap.parse_args()

    run_framework(
        plain_path=args.plain,
        test_path=args.test,
        out_dir=args.out,
        max_frames=args.max_frames,
        stride=args.stride,
        mode=args.mode,
        resize_test_to_plain=args.resize_test_to_plain,
        prefer_skimage=args.prefer_skimage,
        verbose=args.verbose,
        progress_every=args.progress_every,
        roi_sidecar=args.roi_sidecar,
        scope=args.scope,
        framework_faster_path=args.framework_faster_path,
    )

if __name__ == "__main__":
    main()
