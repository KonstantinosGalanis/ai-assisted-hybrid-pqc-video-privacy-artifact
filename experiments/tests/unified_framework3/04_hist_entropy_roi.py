#!/usr/bin/env python3
"""
04_hist_entropy_full_frame.py

Full-frame histogram and entropy tests. No rois.jsonl argument is used.
Computes Shannon entropy and histograms over the entire frame stream.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from unified_utils import ensure_dir, entropy_u8, iter_frames, to_gray, write_csv, write_json


def collect_histograms(path: str, max_frames: int, stride: int, verbose: bool):
    hist_gray = np.zeros(256, dtype=np.float64)
    hist_bgr = {c: np.zeros(256, dtype=np.float64) for c in ['b', 'g', 'r']}
    hsv_hs = np.zeros((180, 256), dtype=np.float64)
    rgb3 = np.zeros((16, 16, 16), dtype=np.float64)
    ent_rows = []
    count = 0
    for proc_i, orig_i, fr in iter_frames(path, max_frames, stride):
        g = to_gray(fr)
        hist_gray += np.bincount(g.reshape(-1), minlength=256)
        frame_row = {
            'proc_frame_idx': proc_i,
            'orig_frame_idx': orig_i,
            'gray_entropy': entropy_u8(g),
            'b_entropy': entropy_u8(fr[:, :, 0]),
            'g_entropy': entropy_u8(fr[:, :, 1]),
            'r_entropy': entropy_u8(fr[:, :, 2]),
        }
        ent_rows.append(frame_row)
        for idx, c in enumerate(['b', 'g', 'r']):
            hist_bgr[c] += np.bincount(fr[:, :, idx].reshape(-1), minlength=256)
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        h2, _, _ = np.histogram2d(hsv[:, :, 0].reshape(-1), hsv[:, :, 1].reshape(-1), bins=[180, 256], range=[[0, 180], [0, 256]])
        hsv_hs += h2
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        h3, _ = np.histogramdd(rgb.reshape(-1, 3), bins=(16, 16, 16), range=((0, 256), (0, 256), (0, 256)))
        rgb3 += h3
        count += 1
        if verbose and count % 30 == 0:
            print(f'processed {count} frames from {path}', flush=True)
    return {'gray': hist_gray, 'bgr': hist_bgr, 'hsv_hs': hsv_hs, 'rgb3': rgb3, 'frame_entropy': ent_rows, 'frames': count}


def save_hist_plots(out: Path, label: str, hists: dict):
    plot_dir = ensure_dir(out / 'plots')
    x = np.arange(256)
    plt.figure(figsize=(8, 4))
    plt.plot(x, hists['gray'])
    plt.title(f'{label} full-frame gray histogram')
    plt.tight_layout()
    plt.savefig(plot_dir / f'{label}_gray_hist.png', dpi=160)
    plt.close()
    for c in ['b', 'g', 'r']:
        plt.figure(figsize=(8, 4))
        plt.plot(x, hists['bgr'][c])
        plt.title(f'{label} full-frame {c.upper()} histogram')
        plt.tight_layout()
        plt.savefig(plot_dir / f'{label}_{c}_hist.png', dpi=160)
        plt.close()
    plt.figure(figsize=(7, 5))
    plt.imshow(np.log1p(hists['hsv_hs']).T, aspect='auto', origin='lower')
    plt.title(f'{label} HSV H-S histogram')
    plt.xlabel('H')
    plt.ylabel('S')
    plt.colorbar(label='log(1+count)')
    plt.tight_layout()
    plt.savefig(plot_dir / f'{label}_hsv_hs_hist.png', dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description='Full-frame histogram and entropy tests. No ROI sidecar is needed.')
    ap.add_argument('--plain', default=None)
    ap.add_argument('--cipher', '--video', dest='cipher', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max_frames', type=int, default=200)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--plots', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    out = ensure_dir(args.out)
    labels = [('cipher', args.cipher)]
    if args.plain:
        labels.insert(0, ('plain', args.plain))
    report = {'test_type': 'full_frame_histogram_entropy', 'inputs': {}, 'summary': {}, 'note': 'Whole-frame histograms/entropy; no rois.jsonl is used.'}
    for label, path in labels:
        h = collect_histograms(path, args.max_frames, args.stride, args.verbose)
        report['inputs'][label] = path
        report['summary'][label] = {
            'frames': h['frames'],
            'global_gray_entropy': entropy_u8(np.repeat(np.arange(256, dtype=np.uint8), np.maximum(h['gray'], 0).astype(np.int64))) if h['gray'].sum() < 5_000_000 else None,
            'mean_frame_gray_entropy': float(np.mean([r['gray_entropy'] for r in h['frame_entropy']])) if h['frame_entropy'] else float('nan'),
            'mean_frame_b_entropy': float(np.mean([r['b_entropy'] for r in h['frame_entropy']])) if h['frame_entropy'] else float('nan'),
            'mean_frame_g_entropy': float(np.mean([r['g_entropy'] for r in h['frame_entropy']])) if h['frame_entropy'] else float('nan'),
            'mean_frame_r_entropy': float(np.mean([r['r_entropy'] for r in h['frame_entropy']])) if h['frame_entropy'] else float('nan'),
        }
        write_csv(out / f'{label}_frame_entropy.csv', h['frame_entropy'])
        write_csv(out / f'{label}_gray_hist.csv', [{'value': i, 'count': int(h['gray'][i])} for i in range(256)])
        for c in ['b', 'g', 'r']:
            write_csv(out / f'{label}_{c}_hist.csv', [{'value': i, 'count': int(h['bgr'][c][i])} for i in range(256)])
        if args.plots:
            save_hist_plots(out, label, h)
    write_json(out / 'report.json', report)
    print(f'[OK] Wrote {out / "report.json"}')


if __name__ == '__main__':
    main()
