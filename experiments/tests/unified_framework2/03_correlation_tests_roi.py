#!/usr/bin/env python3
"""
correlation_tests_framework_roi.py

Correlation tests (PDF VI-D / VI-E / VI-G) with OPTIONAL ROI-only and background-only modes.

Covers:
- Adjacent pixel Pearson correlation in Horizontal / Vertical / Diagonal directions (H/V/D)
- For: grayscale, binary (thresholded), and each color channel (B/G/R)
- Dispersion/scatter plots of adjacent pairs (H/V/D) for plain vs cipher
- NEW: Scope support (whole / roi / background / all) using your rois.jsonl sidecar masks:
    * whole: all adjacent pairs
    * roi: only pairs where BOTH pixels are inside the ROI mask
    * background: only pairs where BOTH pixels are outside the ROI mask

Why ROI scope matters:
- If you encrypt only ROIs (selective encryption), whole-frame correlation will remain high by design.
  ROI-only correlation is the meaningful security metric for the encrypted area.

Sources / inspiration:
- Teixeira007/image-encryption-chaotic-maps: adjacent pixel correlation + dispersion plots methodology
- gxli/Adjacent-Correlation-Analysis: optional ACA plots/maps if installed (not required)

Outputs:
- <out>/report.json
- <out>/plots/*.png

Examples (single line):
  # whole-frame correlation (default)
  python3 correlation_tests_framework_roi.py --plain faces.mp4 --cipher encrypted.mkv --out corr_out --max_frames 200 --verbose

  # ROI-only correlation (requires rois.jsonl produced by your encryption pipeline)
  python3 correlation_tests_framework_roi.py --plain faces.mp4 --cipher encrypted.mkv --roi_sidecar rois.jsonl --scope roi --out corr_roi --max_frames 200 --verbose

  # run all scopes in one go (whole + roi + background)
  python3 correlation_tests_framework_roi.py --plain faces.mp4 --cipher encrypted.mkv --roi_sidecar rois.jsonl --scope all --out corr_all --max_frames 200 --verbose
"""

from __future__ import annotations

import argparse
import datetime
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Logging
# -------------------------

def _now() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def vprint(verbose: bool, msg: str) -> None:
    if verbose:
        print(f"[{_now()}] {msg}", flush=True)


# -------------------------
# ROI sidecar helpers
# -------------------------

def unpack_mask(mask_pack: dict) -> np.ndarray:
    """
    Same format as your framework_faster.py:
      {"h": h, "w": w, "b64": base64(packbits(mask))}
    """
    import base64
    h, w = int(mask_pack["h"]), int(mask_pack["w"])
    raw = base64.b64decode(mask_pack["b64"].encode("ascii"))
    packed = np.frombuffer(raw, dtype=np.uint8)
    bits = np.unpackbits(packed)[: h * w]
    return bits.reshape(h, w).astype(bool)


def load_combined_roi_masks(roi_sidecar_path: str, H: int, W: int, verbose: bool) -> Dict[int, np.ndarray]:
    """
    Returns dict: original_frame_idx -> combined full-frame bool mask.
    """
    vprint(verbose, f"Loading ROI sidecar: {roi_sidecar_path}")
    out: Dict[int, np.ndarray] = {}
    with open(roi_sidecar_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            frame_idx = int(obj.get("frame_idx", -1))
            rois = obj.get("rois", []) or []
            full = np.zeros((H, W), dtype=bool)
            for r in rois:
                bbox = r.get("bbox", None)
                mp = r.get("mask_pack", None)
                if bbox is None or mp is None:
                    continue
                x1, y1, x2, y2 = [int(v) for v in bbox]
                x1 = max(0, min(W, x1)); x2 = max(0, min(W, x2))
                y1 = max(0, min(H, y1)); y2 = max(0, min(H, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                roi_h = y2 - y1
                roi_w = x2 - x1
                m_local = unpack_mask(mp)
                if m_local.shape != (roi_h, roi_w):
                    # Resize mask defensively if mismatch (shouldn't happen, but safe)
                    m_local_u8 = (m_local.astype(np.uint8) * 255)
                    m_local_u8 = cv2.resize(m_local_u8, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
                    m_local = (m_local_u8 > 127)
                full[y1:y2, x1:x2] |= m_local
            out[frame_idx] = full
    vprint(verbose, f"Loaded ROI masks for {len(out)} frames.")
    return out


# -------------------------
# Video I/O
# -------------------------

def _is_video(path: str) -> bool:
    ext = Path(path).suffix.lower()
    return ext in {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}


def _read_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def iter_video_frames(path: str, max_frames: int, stride: int):
    """
    Yields (proc_idx, orig_idx, frame_bgr)
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    orig = 0
    proc = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if stride > 1 and (orig % stride) != 0:
            orig += 1
            continue
        yield proc, orig, frame
        proc += 1
        orig += 1
        if max_frames and proc >= max_frames:
            break
    cap.release()


# -------------------------
# Correlation metrics
# -------------------------

def to_gray_u8(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        g = img_bgr
    else:
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if g.dtype != np.uint8:
        g = np.clip(g, 0, 255).astype(np.uint8, copy=False)
    return g


def to_binary_u8(gray_u8: np.ndarray, thr: int = 128) -> np.ndarray:
    g = to_gray_u8(gray_u8)
    return (g >= thr).astype(np.uint8) * 255


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    am = float(np.mean(a))
    bm = float(np.mean(b))
    cov = float(np.mean((a - am) * (b - bm)))
    sa = float(np.sqrt(np.mean((a - am) ** 2)))
    sb = float(np.sqrt(np.mean((b - bm) ** 2)))
    den = sa * sb
    return float(cov / den) if den else 0.0


def _adjacent_pairs_scoped(g: np.ndarray, mask: Optional[np.ndarray], direction: str, scope: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    scope: whole | roi | background
    For roi/background, we only keep pairs where BOTH pixels satisfy the scope.
    """
    H, W = g.shape
    if scope == "whole" or mask is None:
        if direction == "h":
            a = g[:, :-1].reshape(-1)
            b = g[:, 1:].reshape(-1)
        elif direction == "v":
            a = g[:-1, :].reshape(-1)
            b = g[1:, :].reshape(-1)
        elif direction == "d":
            a = g[:-1, :-1].reshape(-1)
            b = g[1:, 1:].reshape(-1)
        else:
            raise ValueError("direction")
        return a, b

    if scope == "roi":
        m = mask.astype(bool, copy=False)
    elif scope == "background":
        m = ~mask.astype(bool, copy=False)
    else:
        raise ValueError("scope")

    if direction == "h":
        m2 = m[:, :-1] & m[:, 1:]
        a = g[:, :-1][m2]
        b = g[:, 1:][m2]
    elif direction == "v":
        m2 = m[:-1, :] & m[1:, :]
        a = g[:-1, :][m2]
        b = g[1:, :][m2]
    elif direction == "d":
        m2 = m[:-1, :-1] & m[1:, 1:]
        a = g[:-1, :-1][m2]
        b = g[1:, 1:][m2]
    else:
        raise ValueError("direction")

    return a, b


def adjacent_pixel_correlation(g: np.ndarray, mask: Optional[np.ndarray], direction: str, scope: str, sample: int, seed: int) -> float:
    a, b = _adjacent_pairs_scoped(g, mask, direction, scope)
    n = a.size
    if n == 0:
        return float("nan")
    if sample and n > sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=sample, replace=False)
        a = a[idx]
        b = b[idx]
    return pearson_corr(a, b)


def correlation_suite(frame_bgr: np.ndarray, mask: Optional[np.ndarray], scope: str, sample: int, seed: int) -> Dict[str, float]:
    """
    Adjacent correlations for:
      gray_{h,v,d}, bin_{h,v,d}, b_{h,v,d}, g_{h,v,d}, r_{h,v,d}
    computed under the selected scope.
    """
    out: Dict[str, float] = {}
    g = to_gray_u8(frame_bgr)
    bimg = to_binary_u8(g)

    for d in ("h", "v", "d"):
        out[f"gray_{d}"] = adjacent_pixel_correlation(g, mask, d, scope, sample=sample, seed=seed + 1)
        out[f"bin_{d}"] = adjacent_pixel_correlation(bimg, mask, d, scope, sample=sample, seed=seed + 2)

    if frame_bgr.ndim == 3 and frame_bgr.shape[2] >= 3:
        names = ["b", "g", "r"]  # OpenCV order
        for ci, nm in enumerate(names):
            ch = frame_bgr[:, :, ci].astype(np.uint8, copy=False)
            for d in ("h", "v", "d"):
                out[f"{nm}_{d}"] = adjacent_pixel_correlation(ch, mask, d, scope, sample=sample, seed=seed + 10 + ci)

    # Compatibility aliases used by the sweep script and older reports.
    out["h_corr"] = out.get("gray_h", float("nan"))
    out["v_corr"] = out.get("gray_v", float("nan"))
    out["d_corr"] = out.get("gray_d", float("nan"))
    gray_vals = [out.get("gray_h"), out.get("gray_v"), out.get("gray_d")]
    gray_vals = [float(v) for v in gray_vals if v is not None and np.isfinite(v)]
    out["gray_adj_corr"] = float(np.mean(gray_vals)) if gray_vals else float("nan")
    for nm in ("r", "g", "b"):
        vals = [out.get(f"{nm}_h"), out.get(f"{nm}_v"), out.get(f"{nm}_d")]
        vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
        out[f"{nm}_adj_corr"] = float(np.mean(vals)) if vals else float("nan")

    return out


# -------------------------
# Plots (scatter / dispersion)
# -------------------------

def _scatter_adjacent(ax, g: np.ndarray, mask: Optional[np.ndarray], direction: str, scope: str, sample: int, seed: int, title: str) -> None:
    a, b = _adjacent_pairs_scoped(g, mask, direction, scope)
    n = a.size
    if n == 0:
        ax.set_title(title + " (no pairs)")
        ax.set_xlim(0, 255); ax.set_ylim(0, 255)
        return
    if sample and n > sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=sample, replace=False)
        a = a[idx]
        b = b[idx]
    ax.scatter(a, b, s=1, alpha=0.25)
    ax.set_xlim(0, 255); ax.set_ylim(0, 255)
    ax.set_xlabel("Pixel(i)")
    ax.set_ylabel("Pixel(i+1)")
    ax.set_title(title)


def save_dispersion_plots(out_dir: Path, frame_bgr: np.ndarray, mask: Optional[np.ndarray], scope: str, prefix: str, sample: int, seed: int) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    g = to_gray_u8(frame_bgr)
    files: List[str] = []
    for direction, name in [("h", "H"), ("v", "V"), ("d", "D")]:
        fig = plt.figure(figsize=(5, 5))
        ax = plt.gca()
        _scatter_adjacent(ax, g, mask, direction, scope, sample=sample, seed=seed, title=f"{prefix} gray {name} ({scope})")
        fig.tight_layout()
        fn = f"{prefix}_gray_{name}_{scope}.png"
        fig.savefig(out_dir / fn, dpi=150)
        plt.close(fig)
        files.append(str(Path("plots") / fn))
    return files


# -------------------------
# Report
# -------------------------

def _mean_dict(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    if not dicts:
        return {}
    keys = sorted({k for d in dicts for k in d.keys()})
    out: Dict[str, float] = {}
    for k in keys:
        vals = [d.get(k) for d in dicts if k in d]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


@dataclass
class CorrelationReport:
    tests_covered: List[str]
    sources_used: List[str]
    plain_path: str
    cipher_paths: List[str]
    roi_sidecar: Optional[str]
    scope: str
    max_frames: int
    stride: int
    sample_pairs: int
    plot_frames: int
    frames_processed: int
    results: Dict[str, Dict[str, Dict[str, float]]]  # cipher_label -> scope -> {plain,cipher}_means
    plots: Dict[str, Dict[str, List[str]]]           # cipher_label -> scope -> plot files
    notes: List[str]


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", required=True, help="plain image/video")
    ap.add_argument("--cipher", action="append", required=True, help="cipher image/video (repeatable)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--roi_sidecar", default=None, help="rois.jsonl from your encryption pipeline (enables roi/background scope)")
    ap.add_argument("--scope", default="whole", choices=["whole", "roi", "background", "all"])
    ap.add_argument("--max_frames", type=int, default=200)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--sample_pairs", type=int, default=200_000)
    ap.add_argument("--plot_frames", type=int, default=1)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    tests_covered = [
        "VI-D Correlation Test",
        "VI-E Adjacent pixel correlation coefficients (gray/color/binary)",
        "VI-G Horizontal/Vertical/Diagonal correlation (H/V/D) for plain and cipher",
        "ROI scope extension: whole vs ROI-only vs background-only (for selective encryption)",
    ]
    sources_used = [
        "Teixeira007/image-encryption-chaotic-maps (dispersion plot methodology)",
        "gxli/Adjacent-Correlation-Analysis (optional; not required here)",
    ]

    # Load ROI masks if provided (needs video dimensions)
    roi_masks: Optional[Dict[int, np.ndarray]] = None
    if args.roi_sidecar:
        # Get size from plain video/image
        if _is_video(args.plain):
            cap = cv2.VideoCapture(args.plain)
            if not cap.isOpened():
                raise FileNotFoundError(f"Could not open plain video: {args.plain}")
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cap.release()
        else:
            img = _read_image(args.plain)
            H, W = img.shape[:2]
        roi_masks = load_combined_roi_masks(args.roi_sidecar, H=H, W=W, verbose=args.verbose)

    scopes = [args.scope] if args.scope != "all" else ["whole", "roi", "background"]
    if args.scope in ("roi", "background", "all") and roi_masks is None:
        raise ValueError("--scope roi/background/all requires --roi_sidecar rois.jsonl")

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    plots: Dict[str, Dict[str, List[str]]] = {}
    notes: List[str] = []

    # Image mode: load once
    if not _is_video(args.plain):
        plain_img = _read_image(args.plain)
        for cipher_path in args.cipher:
            label = Path(cipher_path).stem
            cipher_img = _read_image(cipher_path)
            results[label] = {}
            plots[label] = {}
            image_mask = None
            if roi_masks is not None:
                image_mask = roi_masks.get(0, np.zeros(plain_img.shape[:2], dtype=bool))
            for sc in scopes:
                plain_corr = correlation_suite(plain_img, mask=image_mask, scope=sc, sample=args.sample_pairs, seed=1000)
                cipher_corr = correlation_suite(cipher_img, mask=image_mask, scope=sc, sample=args.sample_pairs, seed=2000)
                results[label][sc] = {"plain": plain_corr, "cipher": cipher_corr}
                plots[label][sc] = save_dispersion_plots(plots_dir, cipher_img, image_mask, sc, f"{label}_cipher_f0", sample=args.sample_pairs, seed=123)
        rep = CorrelationReport(
            tests_covered=tests_covered,
            sources_used=sources_used,
            plain_path=args.plain,
            cipher_paths=args.cipher,
            roi_sidecar=args.roi_sidecar,
            scope=args.scope,
            max_frames=args.max_frames,
            stride=args.stride,
            sample_pairs=args.sample_pairs,
            plot_frames=args.plot_frames,
            frames_processed=1,
            results=results,
            plots=plots,
            notes=notes,
        )
        (out_dir / "report.json").write_text(json.dumps(asdict(rep), indent=2), encoding="utf-8")
        vprint(args.verbose, f"Wrote {out_dir/'report.json'}")
        return

    # Video mode: stream frames for each cipher separately (keeps memory low)
    total_frames_done = 0
    for cipher_path in args.cipher:
        label = Path(cipher_path).stem
        results[label] = {}
        plots[label] = {sc: [] for sc in scopes}

        vprint(args.verbose, f"Processing cipher video: {cipher_path}")
        plain_corrs_by_scope = {sc: [] for sc in scopes}
        cipher_corrs_by_scope = {sc: [] for sc in scopes}

        # Iterate aligned frames (by proc order); we assume plain and cipher are same length/order
        plain_iter = iter_video_frames(args.plain, max_frames=args.max_frames, stride=args.stride)
        cipher_iter = iter_video_frames(cipher_path, max_frames=args.max_frames, stride=args.stride)

        frames_done = 0
        for (p_proc, p_orig, pfr), (c_proc, c_orig, cfr) in zip(plain_iter, cipher_iter):
            # Use ROI mask by ORIGINAL frame index (as saved by sidecar)
            m = roi_masks.get(p_orig) if roi_masks is not None else None

            for sc in scopes:
                plain_corrs_by_scope[sc].append(correlation_suite(pfr, mask=m, scope=sc, sample=args.sample_pairs, seed=1000 + p_proc))
                cipher_corrs_by_scope[sc].append(correlation_suite(cfr, mask=m, scope=sc, sample=args.sample_pairs, seed=2000 + p_proc))

            # plots for first few frames
            if p_proc < args.plot_frames:
                for sc in scopes:
                    plots[label][sc] += save_dispersion_plots(plots_dir, cfr, m, sc, f"{label}_cipher_f{p_proc}", sample=args.sample_pairs, seed=3000 + p_proc)

            frames_done += 1
            if args.verbose and (frames_done == 1 or frames_done % 25 == 0):
                vprint(True, f"{label}: processed {frames_done} frames")

        total_frames_done = max(total_frames_done, frames_done)
        for sc in scopes:
            results[label][sc] = {
                "plain": _mean_dict(plain_corrs_by_scope[sc]),
                "cipher": _mean_dict(cipher_corrs_by_scope[sc]),
            }

    rep = CorrelationReport(
        tests_covered=tests_covered,
        sources_used=sources_used,
        plain_path=args.plain,
        cipher_paths=args.cipher,
        roi_sidecar=args.roi_sidecar,
        scope=args.scope,
        max_frames=args.max_frames,
        stride=args.stride,
        sample_pairs=args.sample_pairs,
        plot_frames=args.plot_frames,
        frames_processed=total_frames_done,
        results=results,
        plots=plots,
        notes=notes,
    )
    (out_dir / "report.json").write_text(json.dumps(asdict(rep), indent=2), encoding="utf-8")
    vprint(args.verbose, f"Wrote {out_dir/'report.json'}")
    vprint(args.verbose, f"Plots in {plots_dir}")


if __name__ == "__main__":
    main()
