#!/usr/bin/env python3
"""
05_psd_estimation_full_frame.py

Full-frame PSD estimation. No rois.jsonl argument is used.
Computes radially averaged 2D FFT PSD and optional Welch PSD over flattened frames.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from unified_utils import ensure_dir, iter_frames, to_gray, write_csv, write_json

try:
    from scipy.signal import welch as scipy_welch
except Exception:
    scipy_welch = None


def radial_psd(gray: np.ndarray):
    g = gray.astype(np.float64)
    g = g - np.mean(g)
    f = np.fft.fftshift(np.fft.fft2(g))
    power = np.abs(f) ** 2
    h, w = power.shape
    y, x = np.indices((h, w))
    r = np.sqrt((x - w / 2.0) ** 2 + (y - h / 2.0) ** 2).astype(np.int32)
    max_r = int(r.max()) + 1
    sums = np.bincount(r.ravel(), weights=power.ravel(), minlength=max_r)
    counts = np.bincount(r.ravel(), minlength=max_r)
    psd = sums / np.maximum(counts, 1)
    return np.arange(max_r), psd


def process(path: str, max_frames: int, stride: int, keep_stream: bool):
    curves = []
    stream_chunks = []
    frames = 0
    for _, _, fr in iter_frames(path, max_frames, stride):
        g = to_gray(fr)
        r, p = radial_psd(g)
        curves.append(p)
        if keep_stream:
            stream_chunks.append(g.reshape(-1).astype(np.float64))
        frames += 1
    if not curves:
        raise RuntimeError(f'No frames read from {path}')
    mlen = min(len(c) for c in curves)
    arr = np.vstack([c[:mlen] for c in curves])
    result = {'r': np.arange(mlen), 'mean': arr.mean(axis=0), 'median': np.median(arr, axis=0), 'frames': frames}
    if keep_stream and scipy_welch is not None and stream_chunks:
        s = np.concatenate(stream_chunks).astype(np.float32)

        # Cap Welch input to avoid huge memory allocations.
        max_welch_samples = 500_000
        if s.size > max_welch_samples:
            step = max(1, s.size // max_welch_samples)
            s = s[::step][:max_welch_samples]

        if s.size > 0:
            nperseg = min(1024, s.size)
            f, Pxx = scipy_welch(s, nperseg=nperseg, noverlap=0)
            result['welch_f'] = f
            result['welch_psd'] = Pxx
    return result


def save_outputs(out: Path, label: str, res: dict):
    rows = [{'radius': int(r), 'mean_psd': float(m), 'median_psd': float(md)} for r, m, md in zip(res['r'], res['mean'], res['median'])]
    write_csv(out / f'rapsd_{label}_whole_frame.csv', rows)
    plt.figure(figsize=(7, 4))
    plt.plot(res['r'], np.log10(res['mean'] + 1e-12))
    plt.xlabel('radial frequency bin')
    plt.ylabel('log10 mean PSD')
    plt.title(f'{label} full-frame RAPSD')
    plt.tight_layout()
    plt.savefig(out / f'rapsd_{label}_whole_frame.png', dpi=160)
    plt.close()
    if 'welch_f' in res:
        write_csv(out / f'welch_{label}_full_frame.csv', [{'frequency': float(f), 'psd': float(p)} for f, p in zip(res['welch_f'], res['welch_psd'])])
        plt.figure(figsize=(7, 4))
        plt.semilogy(res['welch_f'], res['welch_psd'] + 1e-12)
        plt.title(f'{label} full-frame Welch PSD')
        plt.tight_layout()
        plt.savefig(out / f'welch_{label}_full_frame.png', dpi=160)
        plt.close()


def main():
    ap = argparse.ArgumentParser(description='Full-frame PSD estimation. No ROI sidecar is needed.')
    ap.add_argument('--video', '--cipher', dest='video', required=True)
    ap.add_argument('--plain', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max_frames', type=int, default=200)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    out = ensure_dir(args.out)
    report = {'test_type': 'full_frame_psd_estimation', 'inputs': {'video': args.video}, 'summary': {}, 'note': 'Full-frame PSD; no rois.jsonl is used.'}
    for label, path in ([('plain', args.plain)] if args.plain else []) + [('cipher', args.video)]:
        res = process(path, args.max_frames, args.stride, keep_stream=False)
        save_outputs(out, label, res)
        report['summary'][label] = {'frames': res['frames'], 'rapsd_bins': int(len(res['r'])), 'welch_available': 'welch_f' in res}
    write_json(out / 'psd_report.json', report)
    print(f'[OK] Wrote {out / "psd_report.json"}')


if __name__ == '__main__':
    main()
