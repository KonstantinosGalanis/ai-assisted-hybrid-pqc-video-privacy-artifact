#!/usr/bin/env python3
"""
12_audio_eval_full_frame.py

Audio evaluation for full-frame encryption outputs. No rois.jsonl/payload/manifest arguments are used.
Compares audio extracted from plain, cipher, and optionally decrypted videos.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from unified_utils import ensure_dir, write_json


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
    if p.returncode != 0:
        raise RuntimeError(f'Command failed: {cmd}\n{p.stderr[-2000:]}')
    return p.stdout


def ffprobe_audio_info(path: str):
    if not shutil.which('ffprobe'):
        return {'has_audio': False, 'error': 'ffprobe not found'}
    out = run(['ffprobe', '-v', 'error', '-show_streams', '-select_streams', 'a:0', '-of', 'json', path])
    obj = json.loads(out)
    streams = obj.get('streams', [])
    if not streams:
        return {'has_audio': False}
    s = streams[0]
    return {'has_audio': True, 'codec_name': s.get('codec_name'), 'sample_rate': int(s.get('sample_rate', 0) or 0), 'channels': int(s.get('channels', 0) or 0), 'duration': float(s.get('duration', 0.0) or 0.0)}


def extract_pcm(path: str, wav_path: str, sr: int = 48000):
    if not shutil.which('ffmpeg'):
        raise RuntimeError('ffmpeg not found')
    run(['ffmpeg', '-y', '-i', path, '-vn', '-ac', '1', '-ar', str(sr), '-f', 's16le', '-acodec', 'pcm_s16le', wav_path])


def load_pcm_s16(raw_path: str):
    data = Path(raw_path).read_bytes()
    if not data:
        return np.asarray([], dtype=np.float64), b''
    y = np.frombuffer(data, dtype=np.int16).astype(np.float64) / 32768.0
    return y, data


def align(a, b):
    n = min(a.size, b.size)
    return a[:n], b[:n]


def entropy_bytes(data: bytes):
    if not data:
        return float('nan')
    arr = np.frombuffer(data, dtype=np.uint8)
    hist = np.bincount(arr, minlength=256).astype(np.float64)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def corr_adjacent(y):
    if y.size < 2:
        return float('nan')
    a, b = y[:-1], y[1:]
    if np.std(a) == 0 or np.std(b) == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def compare_audio(a, b):
    a, b = align(a, b)
    if a.size == 0:
        return {'samples': 0}
    d = a - b
    mse = float(np.mean(d * d))
    mae = float(np.mean(np.abs(d)))
    snr = float(10 * math.log10(np.mean(a * a) / mse)) if mse > 0 and np.mean(a * a) > 0 else float('inf')
    return {'samples': int(a.size), 'mse': mse, 'mae': mae, 'snr_db': snr, 'corr': float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float('nan')}


def main():
    ap = argparse.ArgumentParser(description='Audio evaluation for full-frame encrypted videos. No ROI sidecar is needed.')
    ap.add_argument('--plain', required=True)
    ap.add_argument('--cipher', required=True)
    ap.add_argument('--decrypted', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sample_rate', type=int, default=48000)
    args = ap.parse_args()

    out = ensure_dir(args.out)
    report = {'test_type': 'full_frame_audio_eval', 'inputs': {'plain': args.plain, 'cipher': args.cipher, 'decrypted': args.decrypted}, 'audio_info': {}, 'metrics': {}, 'note': 'Full-frame audio evaluation; no rois.jsonl/payload/manifest is used.'}
    with tempfile.TemporaryDirectory() as td:
        loaded = {}
        for label, path in [('plain', args.plain), ('cipher', args.cipher), ('decrypted', args.decrypted)]:
            if not path:
                continue
            report['audio_info'][label] = ffprobe_audio_info(path)
            if not report['audio_info'][label].get('has_audio'):
                continue
            raw = str(Path(td) / f'{label}.s16')
            extract_pcm(path, raw, sr=args.sample_rate)
            y, data = load_pcm_s16(raw)
            loaded[label] = y
            report['metrics'][f'{label}_standalone'] = {'samples': int(y.size), 'byte_entropy': entropy_bytes(data), 'adjacent_sample_corr': corr_adjacent(y)}
        if 'plain' in loaded and 'cipher' in loaded:
            report['metrics']['plain_vs_cipher'] = compare_audio(loaded['plain'], loaded['cipher'])
        if 'plain' in loaded and 'decrypted' in loaded:
            report['metrics']['plain_vs_decrypted'] = compare_audio(loaded['plain'], loaded['decrypted'])
    write_json(out / 'audio_report.json', report)
    print(f'[OK] Wrote {out / "audio_report.json"}')


if __name__ == '__main__':
    main()
