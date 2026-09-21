#!/usr/bin/env python3
"""
01_quality_metrics_full_frame.py

Full-frame version of the quality metrics test. No rois.jsonl argument is used.
Compares the original/plain video against a full-frame encrypted, decrypted, or attacked video.

Metrics: MSE, MAE, PSNR, SNR, SSIM.
Outputs: report.json and frame_metrics.csv.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import mean

import cv2
import numpy as np

from unified_utils import ensure_dir, iter_frames, mae, mse, psnr, select_mode, snr, video_info, write_csv, write_json

try:
    from skimage.metrics import structural_similarity as sk_ssim
except Exception:
    sk_ssim = None


def safe_mean(vals):
    vals = [float(v) for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(mean(vals)) if vals else float('nan')


def ssim_score(a: np.ndarray, b: np.ndarray, mode: str) -> float:
    a = select_mode(a, mode)
    b = select_mode(b, mode)
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a = a[:h, :w]
    b = b[:h, :w]
    if sk_ssim is None:
        # Lightweight fallback: not exact SSIM, but bounded similarity proxy.
        m = mse(a, b)
        return float(1.0 / (1.0 + m / (255.0 * 255.0)))
    kwargs = {'data_range': 255}
    if a.ndim == 3:
        kwargs['channel_axis'] = 2
    return float(sk_ssim(a, b, **kwargs))


def main():
    ap = argparse.ArgumentParser(description='Full-frame video quality metrics. No ROI sidecar is needed.')
    ap.add_argument('--plain', required=True, help='Original/plain video')
    ap.add_argument('--test', required=True, help='Video to compare against plain, e.g. encrypted.mkv or decrypted.mp4')
    ap.add_argument('--out', required=True, help='Output directory')
    ap.add_argument('--mode', choices=['gray', 'color'], default='gray')
    ap.add_argument('--max_frames', type=int, default=0)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    out = ensure_dir(args.out)
    plain_info = video_info(args.plain)
    test_info = video_info(args.test)

    rows = []
    for (proc_i, orig_i, fp), (_, _, ft) in zip(
        iter_frames(args.plain, args.max_frames, args.stride),
        iter_frames(args.test, args.max_frames, args.stride),
    ):
        apx = select_mode(fp, args.mode)
        bpx = select_mode(ft, args.mode)
        row = {
            'proc_frame_idx': proc_i,
            'orig_frame_idx': orig_i,
            'mse': mse(apx, bpx),
            'mae': mae(apx, bpx),
            'psnr_db': psnr(apx, bpx),
            'snr_db': snr(apx, bpx),
            'ssim': ssim_score(fp, ft, args.mode),
        }
        rows.append(row)
        if args.verbose and proc_i % 30 == 0:
            print(f'frame={proc_i} psnr={row["psnr_db"]:.3f} ssim={row["ssim"]:.4f}', flush=True)

    if not rows:
        raise RuntimeError('No paired frames were read. Check paths, max_frames, and stride.')

    report = {
        'test_type': 'full_frame_quality_metrics',
        'plain': args.plain,
        'test': args.test,
        'mode': args.mode,
        'max_frames': args.max_frames,
        'stride': args.stride,
        'plain_info': plain_info,
        'test_info': test_info,
        'frames_evaluated': len(rows),
        'summary': {k: safe_mean([r[k] for r in rows]) for k in ['mse', 'mae', 'psnr_db', 'snr_db', 'ssim']},
        'note': 'Full-frame evaluation: every pixel is part of the encrypted/tested region; no rois.jsonl is used.',
    }
    write_csv(out / 'frame_metrics.csv', rows)
    write_json(out / 'report.json', report)
    print(f'[OK] Wrote {out / "report.json"}')


if __name__ == '__main__':
    main()
