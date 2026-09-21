#!/usr/bin/env python3
"""
03_correlation_tests_full_frame.py

Full-frame adjacent-pixel correlation tests. No rois.jsonl argument is used.
Computes H/V/D Pearson correlation for plain and cipher videos.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

import cv2
import matplotlib.pyplot as plt
import numpy as np

from unified_utils import adjacent_corr, ensure_dir, iter_frames, to_gray, write_csv, write_json


def sample_pairs(img: np.ndarray, direction: str, max_pairs: int = 5000):
    g = to_gray(img)
    if direction == 'h':
        x, y = g[:, :-1].reshape(-1), g[:, 1:].reshape(-1)
    elif direction == 'v':
        x, y = g[:-1, :].reshape(-1), g[1:, :].reshape(-1)
    else:
        x, y = g[:-1, :-1].reshape(-1), g[1:, 1:].reshape(-1)
    if x.size > max_pairs:
        idx = np.linspace(0, x.size - 1, max_pairs).astype(np.int64)
        x, y = x[idx], y[idx]
    return x, y


def main():
    ap = argparse.ArgumentParser(description='Full-frame adjacent pixel correlation. No ROI sidecar is needed.')
    ap.add_argument('--plain', required=True)
    ap.add_argument('--cipher', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max_frames', type=int, default=200)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--plots', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    out = ensure_dir(args.out)
    plot_dir = ensure_dir(out / 'plots')
    dirs = ['h', 'v', 'd']
    accum = {'plain': {d: [] for d in dirs}, 'cipher': {d: [] for d in dirs}}
    rows = []
    first_plain = first_cipher = None

    for (proc_i, orig_i, fp), (_, _, fc) in zip(
        iter_frames(args.plain, args.max_frames, args.stride),
        iter_frames(args.cipher, args.max_frames, args.stride),
    ):
        if first_plain is None:
            first_plain, first_cipher = fp.copy(), fc.copy()
        row = {'proc_frame_idx': proc_i, 'orig_frame_idx': orig_i}
        for d in dirs:
            cp = adjacent_corr(fp, d)
            cc = adjacent_corr(fc, d)
            accum['plain'][d].append(cp)
            accum['cipher'][d].append(cc)
            row[f'plain_corr_{d}'] = cp
            row[f'cipher_corr_{d}'] = cc
        rows.append(row)
        if args.verbose and proc_i % 30 == 0:
            print(f'frame={proc_i} cipher_h={row["cipher_corr_h"]:.5f}', flush=True)

    if not rows:
        raise RuntimeError('No paired frames read.')

    if args.plots and first_plain is not None:
        for label, img in [('plain', first_plain), ('cipher', first_cipher)]:
            for d in dirs:
                x, y = sample_pairs(img, d)
                plt.figure(figsize=(5, 5))
                plt.scatter(x, y, s=1, alpha=0.25)
                plt.xlabel('pixel i')
                plt.ylabel('neighbor pixel')
                plt.title(f'{label} {d.upper()} adjacent-pixel scatter')
                plt.tight_layout()
                plt.savefig(plot_dir / f'{label}_{d}_scatter.png', dpi=160)
                plt.close()

    summary = {}
    for label in ['plain', 'cipher']:
        summary[label] = {d: float(mean(accum[label][d])) for d in dirs}
    report = {
        'test_type': 'full_frame_adjacent_pixel_correlation',
        'plain': args.plain,
        'cipher': args.cipher,
        'frames_evaluated': len(rows),
        'summary': summary,
        'note': 'Full-frame evaluation: H/V/D adjacent-pixel correlations are computed over the entire frame; no rois.jsonl is used.',
    }
    write_csv(out / 'frame_correlations.csv', rows)
    write_json(out / 'report.json', report)
    print(f'[OK] Wrote {out / "report.json"}')


if __name__ == '__main__':
    main()
