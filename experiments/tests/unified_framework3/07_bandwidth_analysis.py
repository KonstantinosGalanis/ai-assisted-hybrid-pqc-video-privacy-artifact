#!/usr/bin/env python3
"""
07_bandwidth_analysis_full_frame.py

Full-frame bandwidth and format analysis. No rois.jsonl argument is used.
Reports file size, average bitrate, codec/container metadata, and optional packet bitrate over time.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from unified_utils import ensure_dir, sha256_file, video_info, write_csv, write_json


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
    if p.returncode != 0:
        raise RuntimeError(f'Command failed: {cmd}\n{p.stderr[-2000:]}')
    return p.stdout


def ffprobe_json(path: str):
    ffprobe = shutil.which('ffprobe')
    if not ffprobe:
        return None
    out = run([ffprobe, '-v', 'error', '-print_format', 'json', '-show_format', '-show_streams', path])
    return json.loads(out)


def packet_bins(path: str, bin_s: float = 1.0):
    ffprobe = shutil.which('ffprobe')
    if not ffprobe:
        return []
    out = run([ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_packets', '-show_entries', 'packet=pts_time,size', '-of', 'csv=p=0', path])
    bins = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 2:
            continue
        try:
            t = float(parts[0])
            size = int(parts[1])
        except Exception:
            continue
        b = int(t // bin_s)
        bins[b] = bins.get(b, 0) + size
    return [{'second_bin': k, 'bytes': v, 'bitrate_bps': 8.0 * v / bin_s} for k, v in sorted(bins.items())]


def summarize(path: str):
    info = video_info(path)
    meta = ffprobe_json(path)
    duration = None
    bit_rate = None
    if meta:
        fmt = meta.get('format', {}) or {}
        try:
            duration = float(fmt.get('duration')) if fmt.get('duration') else None
        except Exception:
            duration = None
        try:
            bit_rate = float(fmt.get('bit_rate')) if fmt.get('bit_rate') else None
        except Exception:
            bit_rate = None
    if bit_rate is None and duration and duration > 0:
        bit_rate = 8.0 * info['file_bytes'] / duration
    return {'path': path, 'video_info': info, 'ffprobe': meta, 'duration_s': duration, 'average_bitrate_bps': bit_rate, 'sha256': sha256_file(path)}


def main():
    ap = argparse.ArgumentParser(description='Full-frame bandwidth/format analysis. No ROI sidecar is needed.')
    ap.add_argument('--plain', required=True)
    ap.add_argument('--cipher', required=True)
    ap.add_argument('--decrypted', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--packet_bins', action='store_true')
    args = ap.parse_args()

    out = ensure_dir(args.out)
    files = {'plain': args.plain, 'cipher': args.cipher}
    if args.decrypted:
        files['decrypted'] = args.decrypted
    report = {'test_type': 'full_frame_bandwidth_format', 'files': {}, 'ratios': {}, 'note': 'File/stream-level bandwidth for full-frame encryption; no rois.jsonl is used.'}
    for label, path in files.items():
        report['files'][label] = summarize(path)
        if args.packet_bins:
            write_csv(out / f'{label}_video_packet_bitrate_bins.csv', packet_bins(path))
    plain_bytes = report['files']['plain']['video_info']['file_bytes']
    cipher_bytes = report['files']['cipher']['video_info']['file_bytes']
    report['ratios']['cipher_over_plain_size'] = float(cipher_bytes / plain_bytes) if plain_bytes else None
    write_json(out / 'report.json', report)
    print(f'[OK] Wrote {out / "report.json"}')


if __name__ == '__main__':
    main()
