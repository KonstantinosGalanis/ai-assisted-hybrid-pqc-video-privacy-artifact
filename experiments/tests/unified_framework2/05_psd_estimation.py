
"""
psd_testing_framework.py

Focus: PDF VI-I "PSD Estimation" only.

Implements two PSD approaches (inspired by open-source references):
1) 2D FFT power spectrum + radially averaged PSD ("RAPSD") for images/frames.
   - Inspired by keflavich/image_tools FFT PSD utilities.  (See repository reference in chat.)
2) Welch PSD (1D) for a flattened pixel stream (e.g., ROI pixels across frames).
   - Uses scipy.signal.welch when available.

Designed for video-encryption research:
- Run on cipher video (and optionally plain video for comparison).
- Supports ROI-only PSD using your ROI sidecar (rois.jsonl) + framework_faster.py unpack_mask.

Outputs:
- out/psd_report.json
- out/rapsd_<label>_whole_mean.csv + .png
- out/rapsd_<label>_whole_median.csv + .png
- out/rapsd_<label>_roi_mean.csv + .png (if ROI)
- out/rapsd_<label>_roi_median.csv + .png (if ROI)
- out/welch_cipher_roi.csv + .png (if ROI)

Example (whole-frame PSD):
  python3 psd_testing_framework.py --video encrypted.mkv --out psd_out --max_frames 200 --stride 2 --verbose

Example (compare plain vs cipher):
  python3 psd_testing_framework.py --plain faces.mp4 --video encrypted.mkv --out psd_out --max_frames 200 --stride 2 --verbose

Example (ROI-only PSD + Welch stream PSD):
  python3 psd_testing_framework.py --video encrypted.mkv --roi_sidecar rois.jsonl --framework_faster_path framework_faster.py --out psd_out --max_frames 200 --verbose

Notes:
- RAPSD uses simple radial binning of the 2D power spectrum |FFT|^2 (shifted).
- For ROI RAPSD we mask outside ROI to the ROI mean (reduces edge leakage vs zeroing).
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import matplotlib.pyplot as plt

try:
    from scipy.signal import welch as scipy_welch
except Exception:
    scipy_welch = None


# -------------------------
# Logging
# -------------------------

def _now() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")

def vprint(verbose: bool, msg: str):
    if verbose:
        print(f"[{_now()}] {msg}", flush=True)


# -------------------------
# Video IO
# -------------------------

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

def to_gray_u8(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        g = img_bgr
    else:
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if g.dtype != np.uint8:
        g = np.clip(g, 0, 255).astype(np.uint8, copy=False)
    return g


# -------------------------
# ROI sidecar loading
# -------------------------

@dataclass
class RoiFrame:
    frame_idx: int
    masks: List[np.ndarray]  # list of bool masks in full-frame coords

def _import_framework_faster(framework_faster_path: str):
    spec = importlib.util.spec_from_file_location("framework_faster_for_psd", framework_faster_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import framework_faster from {framework_faster_path}")
    ff = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ff)
    return ff

def load_roi_sidecar(
    roi_sidecar_path: str,
    H: int,
    W: int,
    framework_faster_path: str,
    max_frames: int,
    stride: int,
    verbose: bool,
) -> Dict[int, RoiFrame]:
    vprint(verbose, f"Loading ROI sidecar: {roi_sidecar_path}")
    ff = _import_framework_faster(framework_faster_path)
    if not hasattr(ff, "unpack_mask"):
        raise RuntimeError("framework_faster.py must provide unpack_mask(mask_pack, h, w)")
    unpack_mask = ff.unpack_mask

    # mapping original frame idx -> processed idx
    orig_to_proc = {}
    proc = 0
    limit_orig = (max_frames - 1) * max(1, stride) if max_frames else 10**9
    for orig in range(0, limit_orig + 1):
        if stride > 1 and (orig % stride) != 0:
            continue
        orig_to_proc[orig] = proc
        proc += 1
        if max_frames and proc >= max_frames:
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
                # framework_faster.py defines unpack_mask(obj: dict) -> bool[h,w]
                # Older code variants may use unpack_mask(obj, h, w). Handle both.
                try:
                    m_local = unpack_mask(mask_pack, mh, mw)
                except TypeError:
                    m_local = unpack_mask(mask_pack)
                # Sanity: ensure mask matches bbox size
                if m_local.shape != (mh, mw):
                    # If stored size differs, crop/pad to fit bbox
                    mm = np.zeros((mh, mw), dtype=bool)
                    hh = min(mh, m_local.shape[0]); ww = min(mw, m_local.shape[1])
                    mm[:hh, :ww] = m_local[:hh, :ww]
                    m_local = mm
                m_full = np.zeros((H, W), dtype=bool)
                m_full[y1:y2, x1:x2] = m_local
                masks.append(m_full)
            out[proc_idx] = RoiFrame(frame_idx=proc_idx, masks=masks)
    vprint(verbose, f"ROI sidecar loaded for {len(out)} processed frames.")
    return out

def combined_roi_mask(roi_frame: Optional[RoiFrame], H: int, W: int) -> np.ndarray:
    if roi_frame is None or not roi_frame.masks:
        return np.zeros((H, W), dtype=bool)
    m = np.zeros((H, W), dtype=bool)
    for mm in roi_frame.masks:
        m |= mm
    return m


# -------------------------
# PSD / RAPSD
# -------------------------

def power_spectrum_2d(gray: np.ndarray, detrend: bool = True) -> np.ndarray:
    x = gray.astype(np.float64, copy=False)
    if detrend:
        x = x - np.mean(x)
    F = np.fft.fft2(x)
    P = np.abs(F) ** 2
    return np.fft.fftshift(P)

def rapsd_from_power(P_shifted: np.ndarray, bin_width: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    H, W = P_shifted.shape
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    y, x = np.indices((H, W))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)

    rmax = r.max()
    nbins = int(np.floor(rmax / bin_width)) + 1
    bins = np.linspace(0, nbins * bin_width, nbins + 1)

    r_flat = r.reshape(-1)
    p_flat = P_shifted.reshape(-1)

    which = np.digitize(r_flat, bins) - 1
    which = np.clip(which, 0, nbins - 1)

    sums = np.bincount(which, weights=p_flat, minlength=nbins).astype(np.float64)
    counts = np.bincount(which, minlength=nbins).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        prof = np.where(counts > 0, sums / counts, 0.0)

    r_centers = (bins[:-1] + bins[1:]) / 2.0
    return r_centers, prof

def rapsd_from_image(gray: np.ndarray, bin_width: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    P = power_spectrum_2d(gray, detrend=True)
    return rapsd_from_power(P, bin_width=bin_width)

def masked_image_for_fft(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = gray.copy()
    if np.any(mask):
        mu = int(np.mean(out[mask]))
    else:
        mu = int(np.mean(out))
    out[~mask] = mu
    return out


# -------------------------
# Welch PSD (1D)
# -------------------------

def welch_psd_1d(x: np.ndarray, fs: float = 1.0, nperseg: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    if x.size < 8:
        return np.array([]), np.array([])
    if scipy_welch is not None:
        f, Pxx = scipy_welch(x, fs=fs, nperseg=min(nperseg, x.size))
        return f, Pxx

    # fallback
    nperseg = min(nperseg, x.size)
    step = nperseg // 2
    if step <= 0:
        step = nperseg
    segs = []
    for start in range(0, x.size - nperseg + 1, step):
        seg = x[start:start+nperseg]
        w = np.hanning(nperseg)
        seg = seg * w
        F = np.fft.rfft(seg)
        P = (np.abs(F) ** 2) / np.sum(w ** 2)
        segs.append(P)
    if not segs:
        return np.array([]), np.array([])
    Pxx = np.mean(np.stack(segs, axis=0), axis=0)
    f = np.fft.rfftfreq(nperseg, d=1.0/fs)
    return f, Pxx


# -------------------------
# Output helpers
# -------------------------

@dataclass
class PsdReport:
    inputs: Dict[str, object]
    rapsd_whole: Dict[str, object]
    rapsd_roi: Optional[Dict[str, object]]
    welch_roi: Optional[Dict[str, object]]
    notes: List[str]

def save_curve_csv(path: Path, x: np.ndarray, y: np.ndarray, x_name: str, y_name: str):
    import csv
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([x_name, y_name])
        for xi, yi in zip(x.tolist(), y.tolist()):
            w.writerow([xi, yi])

def plot_loglog(path: Path, x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str):
    plt.figure()
    m = (x > 0) & (y > 0)
    if np.any(m):
        plt.loglog(x[m], y[m])
    else:
        plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# -------------------------
# Main runner
# -------------------------

def run_psd(
    video_path: str,
    out_dir: str,
    plain_path: Optional[str],
    roi_sidecar: Optional[str],
    framework_faster_path: str,
    max_frames: int,
    stride: int,
    bin_width: float,
    welch_nperseg: int,
    verbose: bool,
) -> PsdReport:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    notes: List[str] = []

    first = next(iter_frames(video_path, max_frames=1, stride=1), None)
    if first is None:
        raise RuntimeError("No frames read from video.")
    H, W = first[1].shape[:2]

    roi_map = None
    if roi_sidecar:
        roi_map = load_roi_sidecar(roi_sidecar, H, W, framework_faster_path, max_frames=max_frames, stride=stride, verbose=verbose)

    def avg_rapsd(path: str, label: str, use_roi: bool) -> Dict[str, object]:
        vprint(verbose, f"Computing RAPSD ({'ROI' if use_roi else 'whole'}) for {label}: {path}")
        curves = []
        roi_stream = []
        frames_used = 0
        r_ref = None

        for i, fr in iter_frames(path, max_frames=max_frames, stride=stride):
            g = to_gray_u8(fr)
            if use_roi and roi_map is not None:
                m = combined_roi_mask(roi_map.get(i), H, W)
                if not np.any(m):
                    continue
                g2 = masked_image_for_fft(g, m)
                roi_stream.append(g[m].astype(np.float64))
            else:
                g2 = g

            r, pr = rapsd_from_image(g2, bin_width=bin_width)
            if r_ref is None:
                r_ref = r
            curves.append(pr)
            frames_used += 1
            if verbose and (frames_used == 1 or frames_used % 25 == 0):
                vprint(True, f"  {label}: {frames_used} frames processed for RAPSD")

        if not curves or r_ref is None:
            return {"frames_used": 0}

        stack = np.stack(curves, axis=0)
        Pmean = np.mean(stack, axis=0)
        Pmed = np.median(stack, axis=0)

        mean_csv = outp / f"rapsd_{label}_{'roi' if use_roi else 'whole'}_mean.csv"
        med_csv = outp / f"rapsd_{label}_{'roi' if use_roi else 'whole'}_median.csv"
        save_curve_csv(mean_csv, r_ref, Pmean, "radius_bin", "power_mean")
        save_curve_csv(med_csv, r_ref, Pmed, "radius_bin", "power_median")

        mean_png = outp / f"rapsd_{label}_{'roi' if use_roi else 'whole'}_mean.png"
        med_png = outp / f"rapsd_{label}_{'roi' if use_roi else 'whole'}_median.png"
        plot_loglog(mean_png, r_ref, Pmean, f"RAPSD mean ({label}, {'ROI' if use_roi else 'whole'})", "radius (pixels)", "power")
        plot_loglog(med_png, r_ref, Pmed, f"RAPSD median ({label}, {'ROI' if use_roi else 'whole'})", "radius (pixels)", "power")

        out = {
            "frames_used": int(frames_used),
            "bin_width": float(bin_width),
            "mean_csv": str(mean_csv),
            "median_csv": str(med_csv),
            "mean_png": str(mean_png),
            "median_png": str(med_png),
            "power_mean_summary": {
                "mean": float(np.mean(Pmean)),
                "median": float(np.median(Pmean)),
                "p95": float(np.percentile(Pmean, 95)),
            },
        }
        if use_roi and roi_stream:
            out["_roi_stream"] = np.concatenate(roi_stream, axis=0)
        return out

    # Cipher
    whole_cipher = avg_rapsd(video_path, "cipher", use_roi=False)

    roi_cipher = None
    welch_roi_obj = None
    if roi_map is not None:
        roi_cipher = avg_rapsd(video_path, "cipher", use_roi=True)
        stream = roi_cipher.pop("_roi_stream", None)
        if stream is not None and stream.size > 0:
            vprint(verbose, "Computing Welch PSD on ROI pixel stream (cipher)...")
            f, Pxx = welch_psd_1d(stream, fs=1.0, nperseg=welch_nperseg)
            if f.size:
                csv_path = outp / "welch_cipher_roi.csv"
                save_curve_csv(csv_path, f, Pxx, "freq", "Pxx")
                png_path = outp / "welch_cipher_roi.png"
                plot_loglog(png_path, f, Pxx, "Welch PSD (cipher ROI stream)", "frequency (arb)", "PSD")
                welch_roi_obj = {
                    "n_samples": int(stream.size),
                    "nperseg": int(min(welch_nperseg, stream.size)),
                    "backend": "scipy.signal.welch" if scipy_welch is not None else "fallback_welch",
                    "csv": str(csv_path),
                    "png": str(png_path),
                }
            else:
                notes.append("Welch PSD skipped: ROI stream too short.")

    # Optional plain comparison
    plain_whole = None
    plain_roi = None
    if plain_path:
        plain_whole = avg_rapsd(plain_path, "plain", use_roi=False)
        if roi_map is not None:
            plain_roi = avg_rapsd(plain_path, "plain", use_roi=True)
            plain_roi.pop("_roi_stream", None)

    report = PsdReport(
        inputs={
            "video": video_path,
            "plain": plain_path,
            "roi_sidecar": roi_sidecar,
            "framework_faster_path": framework_faster_path,
            "max_frames": int(max_frames),
            "stride": int(stride),
            "bin_width": float(bin_width),
            "welch_nperseg": int(welch_nperseg),
        },
        rapsd_whole={"cipher": whole_cipher, "plain": plain_whole},
        rapsd_roi={"cipher": roi_cipher, "plain": plain_roi} if roi_map is not None else None,
        welch_roi=welch_roi_obj,
        notes=notes,
    )

    (outp / "psd_report.json").write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    vprint(verbose, f"Wrote {outp/'psd_report.json'}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="cipher/encrypted video path")
    ap.add_argument("--plain", default=None, help="optional plain/original video path")
    ap.add_argument("--roi_sidecar", default=None, help="rois.jsonl (enables ROI PSD)")
    ap.add_argument("--framework_faster_path", default="framework_faster.py", help="used to decode ROI masks via unpack_mask")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--max_frames", type=int, default=200)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--bin_width", type=float, default=1.0)
    ap.add_argument("--welch_nperseg", type=int, default=4096)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    run_psd(
        video_path=args.video,
        out_dir=args.out,
        plain_path=args.plain,
        roi_sidecar=args.roi_sidecar,
        framework_faster_path=args.framework_faster_path,
        max_frames=args.max_frames,
        stride=args.stride,
        bin_width=args.bin_width,
        welch_nperseg=args.welch_nperseg,
        verbose=args.verbose,
    )

if __name__ == "__main__":
    main()
