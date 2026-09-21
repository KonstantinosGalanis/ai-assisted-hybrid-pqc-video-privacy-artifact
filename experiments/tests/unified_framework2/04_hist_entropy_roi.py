#!/usr/bin/env python3
"""
04_hist_entropy_roi.py

Histogram / entropy framework with corrected 3D histogram handling.

Covers:
- Shannon entropy from 1D histograms
- 1D histograms (gray + B/G/R)
- 2D HSV(H,S) histograms
- 3D RGB histograms
- ROI / background / whole-image scope

Main fixes compared with the earlier version:
- 3D histograms are aggregated across all processed frames, not only the first frame
- 3D histograms honor ROI/background masks
- ROI-sidecar lookup uses original frame indices, so stride > 1 stays aligned
- For compatibility, the first-frame 3D histogram is also saved in addition to the corrected mean 3D histogram
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def vprint(verbose: bool, msg: str):
    if verbose:
        print(msg, flush=True)


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
        ok, fr = cap.read()
        if not ok:
            break
        if stride > 1 and (orig_idx % stride) != 0:
            orig_idx += 1
            continue
        yield proc_idx, orig_idx, fr
        proc_idx += 1
        orig_idx += 1
        if max_frames and proc_idx >= max_frames:
            break
    cap.release()


def _import_unpack_mask(framework_faster_path: str):
    spec = importlib.util.spec_from_file_location("framework_faster_sidecar", framework_faster_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import framework_faster from {framework_faster_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "unpack_mask"):
        raise RuntimeError("framework_faster.py must provide unpack_mask(...)")
    return mod.unpack_mask


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


def load_roi_masks(
    roi_sidecar_path: str,
    H: int,
    W: int,
    unpack_mask,
) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    with open(roi_sidecar_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            orig_idx = int(obj.get("frame_idx", -1))
            full = np.zeros((H, W), dtype=bool)
            for r in obj.get("rois", []) or []:
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
                local = _call_unpack_mask(unpack_mask, mask_pack, mh, mw)
                full[y1:y2, x1:x2] |= local
            out[orig_idx] = full
    return out


def shannon_entropy_from_hist(hist: np.ndarray) -> float:
    hist = hist.astype(np.float64, copy=False)
    s = float(hist.sum())
    if s <= 0:
        return 0.0
    p = hist / s
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p))) if p.size else 0.0


def hist1d_u8(img_1c_u8: np.ndarray, bins: int = 256, mask_u8: Optional[np.ndarray] = None) -> np.ndarray:
    h = cv2.calcHist([img_1c_u8], [0], mask_u8, [bins], [0, 256])
    return h.reshape(-1).astype(np.float64)


def hist2d_hs(bgr_u8: np.ndarray, bins_h: int, bins_s: int, mask_u8: Optional[np.ndarray] = None) -> np.ndarray:
    hsv = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], mask_u8, [bins_h, bins_s], [0, 180, 0, 256])
    return h.astype(np.float64)


def hist3d_rgb(bgr_u8: np.ndarray, bins: int, mask_u8: Optional[np.ndarray] = None) -> np.ndarray:
    rgb = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2RGB)
    if mask_u8 is None:
        pix = rgb.reshape(-1, 3)
    else:
        use = mask_u8.astype(bool)
        pix = rgb[use]
    H3 = np.zeros((bins, bins, bins), dtype=np.int64)
    if pix.size == 0:
        return H3.astype(np.float64)
    q = (pix.astype(np.uint16) * bins) // 256
    q = np.clip(q, 0, bins - 1).astype(np.int16)
    r = q[:, 0]
    g = q[:, 1]
    b = q[:, 2]
    np.add.at(H3, (r, g, b), 1)
    return H3.astype(np.float64)


def normalize_hist(h: np.ndarray) -> np.ndarray:
    s = float(h.sum())
    return (h.astype(np.float64, copy=False) / s) if s > 0 else h.astype(np.float64, copy=False)


def hist_intersection(h1: np.ndarray, h2: np.ndarray) -> float:
    a = normalize_hist(h1)
    b = normalize_hist(h2)
    return float(np.minimum(a, b).sum())


def l1_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    a = normalize_hist(h1)
    b = normalize_hist(h2)
    return float(np.abs(a - b).sum())


def analyze(
    cipher_path: str,
    out_dir: str,
    plain_path: Optional[str],
    roi_sidecar: Optional[str],
    framework_faster_path: str,
    scope: str,
    max_frames: int,
    stride: int,
    bins1d: int,
    bins2d_h: int,
    bins2d_s: int,
    bins3d: int,
    plots: bool,
    verbose: bool,
):
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    (outp / "plots").mkdir(parents=True, exist_ok=True)

    info_c = video_info(cipher_path)
    info_p = video_info(plain_path) if plain_path else None
    H = int(info_c["height"]); W = int(info_c["width"])

    roi_masks = None
    if roi_sidecar:
        unpack_mask = _import_unpack_mask(framework_faster_path)
        roi_masks = load_roi_masks(roi_sidecar, H=H, W=W, unpack_mask=unpack_mask)
        vprint(verbose, f"Loaded ROI masks for {len(roi_masks)} original frames.")

    h1_plain_sum = np.zeros((4, bins1d), dtype=np.float64) if plain_path else None
    h1_cipher_sum = np.zeros((4, bins1d), dtype=np.float64)
    h2_plain_sum = np.zeros((bins2d_h, bins2d_s), dtype=np.float64) if plain_path else None
    h2_cipher_sum = np.zeros((bins2d_h, bins2d_s), dtype=np.float64)

    h3_plain_sum = np.zeros((bins3d, bins3d, bins3d), dtype=np.float64) if plain_path else None
    h3_cipher_sum = np.zeros((bins3d, bins3d, bins3d), dtype=np.float64)
    h3_plain_first = None
    h3_cipher_first = None

    ent_plain_gray = []
    ent_cipher_gray = []
    ent_plain_bgr = {"b": [], "g": [], "r": []}
    ent_cipher_bgr = {"b": [], "g": [], "r": []}
    n_used = 0

    cipher_iter = iter_frames(cipher_path, max_frames=max_frames, stride=stride)
    plain_iter = iter_frames(plain_path, max_frames=max_frames, stride=stride) if plain_path else None
    iterable = ((ci, oi, cfr, None) for (ci, oi, cfr) in cipher_iter) if plain_iter is None else (
        (ci, oi, cfr, pfr) for (ci, oi, cfr), (_, _, pfr) in zip(cipher_iter, plain_iter)
    )

    for proc_idx, orig_idx, cimg, pimg in iterable:
        if cimg.shape[0] != H or cimg.shape[1] != W:
            cimg = cv2.resize(cimg, (W, H), interpolation=cv2.INTER_AREA)
        if pimg is not None and (pimg.shape[0] != H or pimg.shape[1] != W):
            pimg = cv2.resize(pimg, (W, H), interpolation=cv2.INTER_AREA)

        mask_u8 = None
        if roi_masks is not None:
            roi_mask = roi_masks.get(orig_idx, np.zeros((H, W), dtype=bool))
            if scope == "roi":
                use = roi_mask
            elif scope == "background":
                use = ~roi_mask
            else:
                use = np.ones((H, W), dtype=bool)
            mask_u8 = (use.astype(np.uint8) * 255)

        cgray = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
        h_gray_c = hist1d_u8(cgray, bins=bins1d, mask_u8=mask_u8)
        h_b_c = hist1d_u8(cimg[:, :, 0], bins=bins1d, mask_u8=mask_u8)
        h_g_c = hist1d_u8(cimg[:, :, 1], bins=bins1d, mask_u8=mask_u8)
        h_r_c = hist1d_u8(cimg[:, :, 2], bins=bins1d, mask_u8=mask_u8)

        h1_cipher_sum[0] += h_gray_c
        h1_cipher_sum[1] += h_b_c
        h1_cipher_sum[2] += h_g_c
        h1_cipher_sum[3] += h_r_c

        ent_cipher_gray.append(shannon_entropy_from_hist(h_gray_c))
        ent_cipher_bgr["b"].append(shannon_entropy_from_hist(h_b_c))
        ent_cipher_bgr["g"].append(shannon_entropy_from_hist(h_g_c))
        ent_cipher_bgr["r"].append(shannon_entropy_from_hist(h_r_c))

        h2_cipher_sum += hist2d_hs(cimg, bins_h=bins2d_h, bins_s=bins2d_s, mask_u8=mask_u8)
        h3_cur_c = hist3d_rgb(cimg, bins=bins3d, mask_u8=mask_u8)
        h3_cipher_sum += h3_cur_c
        if n_used == 0:
            h3_cipher_first = h3_cur_c.copy()

        if pimg is not None and plain_path:
            pgray = cv2.cvtColor(pimg, cv2.COLOR_BGR2GRAY)
            h_gray_p = hist1d_u8(pgray, bins=bins1d, mask_u8=mask_u8)
            h_b_p = hist1d_u8(pimg[:, :, 0], bins=bins1d, mask_u8=mask_u8)
            h_g_p = hist1d_u8(pimg[:, :, 1], bins=bins1d, mask_u8=mask_u8)
            h_r_p = hist1d_u8(pimg[:, :, 2], bins=bins1d, mask_u8=mask_u8)

            h1_plain_sum[0] += h_gray_p
            h1_plain_sum[1] += h_b_p
            h1_plain_sum[2] += h_g_p
            h1_plain_sum[3] += h_r_p

            ent_plain_gray.append(shannon_entropy_from_hist(h_gray_p))
            ent_plain_bgr["b"].append(shannon_entropy_from_hist(h_b_p))
            ent_plain_bgr["g"].append(shannon_entropy_from_hist(h_g_p))
            ent_plain_bgr["r"].append(shannon_entropy_from_hist(h_r_p))

            h2_plain_sum += hist2d_hs(pimg, bins_h=bins2d_h, bins_s=bins2d_s, mask_u8=mask_u8)
            h3_cur_p = hist3d_rgb(pimg, bins=bins3d, mask_u8=mask_u8)
            h3_plain_sum += h3_cur_p
            if n_used == 0:
                h3_plain_first = h3_cur_p.copy()

        n_used += 1
        if verbose and (n_used == 1 or (n_used % 25) == 0):
            vprint(True, f"Processed {n_used} frames...")

    h1_cipher_mean = h1_cipher_sum / max(1, n_used)
    h2_cipher_mean = h2_cipher_sum / max(1, n_used)
    h3_cipher_mean = h3_cipher_sum / max(1, n_used)

    np.save(outp / "hist1d_mean_cipher.npy", h1_cipher_mean)
    np.save(outp / "hist2d_hs_mean_cipher.npy", h2_cipher_mean)
    np.save(outp / "hist3d_rgb_mean_cipher.npy", h3_cipher_mean)
    if h3_cipher_first is not None:
        np.save(outp / "hist3d_rgb_first_cipher.npy", h3_cipher_first)

    sim = {}
    if plain_path and h1_plain_sum is not None and h2_plain_sum is not None and h3_plain_sum is not None:
        h1_plain_mean = h1_plain_sum / max(1, n_used)
        h2_plain_mean = h2_plain_sum / max(1, n_used)
        h3_plain_mean = h3_plain_sum / max(1, n_used)

        np.save(outp / "hist1d_mean_plain.npy", h1_plain_mean)
        np.save(outp / "hist2d_hs_mean_plain.npy", h2_plain_mean)
        np.save(outp / "hist3d_rgb_mean_plain.npy", h3_plain_mean)
        if h3_plain_first is not None:
            np.save(outp / "hist3d_rgb_first_plain.npy", h3_plain_first)

        sim["hist1d_gray_intersection"] = hist_intersection(h1_plain_mean[0], h1_cipher_mean[0])
        sim["hist1d_gray_l1"] = l1_distance(h1_plain_mean[0], h1_cipher_mean[0])
        sim["hist2d_hs_intersection"] = hist_intersection(h2_plain_mean, h2_cipher_mean)
        sim["hist2d_hs_l1"] = l1_distance(h2_plain_mean, h2_cipher_mean)
        sim["hist3d_rgb_intersection"] = hist_intersection(h3_plain_mean, h3_cipher_mean)
        sim["hist3d_rgb_l1"] = l1_distance(h3_plain_mean, h3_cipher_mean)
    else:
        h1_plain_mean = None
        h2_plain_mean = None
        h3_plain_mean = None

    report = {
        "inputs": {"plain": plain_path, "cipher": cipher_path},
        "scope": scope,
        "params": {
            "max_frames": max_frames,
            "stride": stride,
            "bins1d": bins1d,
            "bins2d_h": bins2d_h,
            "bins2d_s": bins2d_s,
            "bins3d": bins3d,
        },
        "video_info": {"plain": info_p, "cipher": info_c},
        "frames_used": n_used,
        "entropy": {
            "plain_gray_mean": float(np.mean(ent_plain_gray)) if ent_plain_gray else None,
            "cipher_gray_mean": float(np.mean(ent_cipher_gray)) if ent_cipher_gray else None,
            "plain_bgr_mean": {k: float(np.mean(v)) if v else None for k, v in ent_plain_bgr.items()},
            "cipher_bgr_mean": {k: float(np.mean(v)) if v else None for k, v in ent_cipher_bgr.items()},
        },
        "histogram_outputs": {
            "hist1d_mean_cipher": "hist1d_mean_cipher.npy",
            "hist2d_hs_mean_cipher": "hist2d_hs_mean_cipher.npy",
            "hist3d_rgb_mean_cipher": "hist3d_rgb_mean_cipher.npy",
            "hist3d_rgb_first_cipher": "hist3d_rgb_first_cipher.npy" if h3_cipher_first is not None else None,
            "hist1d_mean_plain": "hist1d_mean_plain.npy" if plain_path else None,
            "hist2d_hs_mean_plain": "hist2d_hs_mean_plain.npy" if plain_path else None,
            "hist3d_rgb_mean_plain": "hist3d_rgb_mean_plain.npy" if plain_path else None,
            "hist3d_rgb_first_plain": "hist3d_rgb_first_plain.npy" if h3_plain_first is not None else None,
        },
        "histogram_similarity": sim if sim else None,
        "tests_covered": [
            "Shannon entropy from 1D histograms (gray + per-channel B,G,R)",
            "1D histograms via cv2.calcHist (gray + B,G,R)",
            "2D HSV(H,S) histogram via cv2.calcHist",
            "3D RGB histogram cube (aggregated across all processed frames)",
            "ROI/background/whole scope via sidecar mask",
            "Optional plots (1D/2D + 3D projections)"
        ],
        "notes": [
            "3D histogram mean is the corrected VI-S implementation.",
            "First-frame 3D histograms are still saved for backward compatibility."
        ],
    }

    (outp / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if plots:
        try:
            import matplotlib.pyplot as plt
            plots_dir = outp / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            x = np.arange(bins1d)

            def plot_1d(idx: int, title: str, fname: str):
                plt.figure()
                plt.plot(x, normalize_hist(h1_cipher_mean[idx]), label="cipher")
                if h1_plain_mean is not None:
                    plt.plot(x, normalize_hist(h1_plain_mean[idx]), label="plain")
                plt.title(title)
                plt.xlabel("bin")
                plt.ylabel("probability")
                plt.legend()
                plt.tight_layout()
                plt.savefig(plots_dir / fname, dpi=150)
                plt.close()

            plot_1d(0, "1D Histogram (Grayscale)", "hist1d_gray.png")
            plot_1d(1, "1D Histogram (B channel)", "hist1d_b.png")
            plot_1d(2, "1D Histogram (G channel)", "hist1d_g.png")
            plot_1d(3, "1D Histogram (R channel)", "hist1d_r.png")

            def plot_2d(h2: np.ndarray, title: str, fname: str):
                plt.figure()
                plt.imshow(normalize_hist(h2).T, aspect="auto", origin="lower")
                plt.title(title)
                plt.xlabel("H bins")
                plt.ylabel("S bins")
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(plots_dir / fname, dpi=150)
                plt.close()

            plot_2d(h2_cipher_mean, "2D HS Histogram (Cipher)", "hist2d_hs_cipher.png")
            if h2_plain_mean is not None:
                plot_2d(h2_plain_mean, "2D HS Histogram (Plain)", "hist2d_hs_plain.png")

            def plot_3d_proj(h3: np.ndarray, title_prefix: str, tag: str):
                h3n = normalize_hist(h3)
                rg = h3n.sum(axis=2)
                rb = h3n.sum(axis=1)
                gb = h3n.sum(axis=0)
                for plane, nm in [(rg, "RG"), (rb, "RB"), (gb, "GB")]:
                    plt.figure()
                    plt.imshow(plane.T, aspect="auto", origin="lower")
                    plt.title(f"{title_prefix} 3D Hist Projection {nm}")
                    plt.xlabel(f"{nm[0]} bins")
                    plt.ylabel(f"{nm[1]} bins")
                    plt.colorbar()
                    plt.tight_layout()
                    plt.savefig(plots_dir / f"hist3d_{tag}_proj_{nm}.png", dpi=150)
                    plt.close()

            plot_3d_proj(h3_cipher_mean, "Cipher", "cipher_mean")
            if h3_plain_mean is not None:
                plot_3d_proj(h3_plain_mean, "Plain", "plain_mean")
        except Exception as e:
            vprint(verbose, f"Plotting skipped due to error: {e}")

    vprint(verbose, f"Done. Wrote {outp/'report.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", default=None, help="plain/original video (optional)")
    ap.add_argument("--cipher", required=True, help="cipher/encrypted video")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--max_frames", type=int, default=200)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--roi_sidecar", default=None)
    ap.add_argument("--framework_faster_path", default="framework_faster.py")
    ap.add_argument("--scope", default="whole", choices=["whole", "roi", "background"])
    ap.add_argument("--bins1d", type=int, default=256)
    ap.add_argument("--bins2d_h", type=int, default=30)
    ap.add_argument("--bins2d_s", type=int, default=32)
    ap.add_argument("--bins3d", type=int, default=32)
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    analyze(
        cipher_path=args.cipher,
        out_dir=args.out,
        plain_path=args.plain,
        roi_sidecar=args.roi_sidecar,
        framework_faster_path=args.framework_faster_path,
        scope=args.scope,
        max_frames=args.max_frames,
        stride=args.stride,
        bins1d=args.bins1d,
        bins2d_h=args.bins2d_h,
        bins2d_s=args.bins2d_s,
        bins3d=args.bins3d,
        plots=args.plots,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
