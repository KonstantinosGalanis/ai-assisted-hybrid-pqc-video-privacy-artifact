#!/usr/bin/env python3
"""
10_nist_runner_full_frame.py

Standalone NIST-style randomness runner for full-frame encryption.
No rois.jsonl argument is used. Prefer --bitstream from framework_full_encrypt.py --cipher_dump.
If --cipher_video is used, decoded encrypted frame bytes are tested instead of the container bytes.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from unified_utils import ensure_dir, raw_frame_bytes, write_bytes, write_json


def read_bytes_source(bitstream=None, cipher_video=None, max_bits=0, max_frames=0, stride=1):
    if bitstream:
        data = Path(bitstream).read_bytes()
    elif cipher_video:
        data = raw_frame_bytes(cipher_video, max_frames=max_frames, stride=stride)
    else:
        raise ValueError('Provide --bitstream or --cipher_video')
    if max_bits and max_bits > 0:
        data = data[: math.ceil(max_bits / 8)]
    return data


def bits_from_bytes(data: bytes):
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder='big').astype(np.uint8)


def monobit(bits):
    n = bits.size
    if n == 0:
        return {'p_value': float('nan'), 'passed': False}
    s = int(np.sum(2 * bits.astype(np.int16) - 1))
    p = math.erfc(abs(s) / math.sqrt(2.0 * n))
    return {'n_bits': int(n), 'statistic': int(s), 'p_value': float(p), 'passed': bool(p >= 0.01)}


def block_frequency(bits, block_size=128):
    n = bits.size
    M = int(block_size)
    N = n // M
    if N == 0:
        return {'p_value': float('nan'), 'passed': False, 'note': 'not enough bits'}
    x = bits[:N * M].reshape(N, M)
    pis = x.mean(axis=1)
    chi = 4.0 * M * float(np.sum((pis - 0.5) ** 2))
    # Wilson-Hilferty normal approximation for chi-square survival.
    z = ((chi / N) ** (1/3) - (1 - 2/(9*N))) / math.sqrt(2/(9*N)) if N > 0 else 0.0
    p = 0.5 * math.erfc(z / math.sqrt(2))
    return {'block_size': M, 'n_blocks': int(N), 'chi_square': float(chi), 'p_value_approx': float(p), 'passed': bool(p >= 0.01)}


def runs(bits):
    n = bits.size
    if n < 2:
        return {'p_value': float('nan'), 'passed': False}
    pi = float(bits.mean())
    tau = 2.0 / math.sqrt(n)
    if abs(pi - 0.5) >= tau:
        return {'pi': pi, 'p_value': 0.0, 'passed': False, 'note': 'frequency precondition failed'}
    V = 1 + int(np.sum(bits[1:] != bits[:-1]))
    p = math.erfc(abs(V - 2*n*pi*(1-pi)) / (2 * math.sqrt(2*n) * pi * (1-pi)))
    return {'pi': pi, 'runs': int(V), 'p_value': float(p), 'passed': bool(p >= 0.01)}


def byte_entropy(data):
    if not data:
        return {'entropy_bits_per_byte': float('nan')}
    arr = np.frombuffer(data, dtype=np.uint8)
    hist = np.bincount(arr, minlength=256).astype(np.float64)
    p = hist / hist.sum()
    p = p[p > 0]
    h = float(-(p * np.log2(p)).sum())
    return {'entropy_bits_per_byte': h, 'ideal': 8.0}


def serial_2bit(bits):
    n = bits.size
    if n < 4:
        return {'p_value_approx': float('nan'), 'passed': False}
    pairs = bits[:(n//2)*2].reshape(-1, 2)
    vals = pairs[:, 0] * 2 + pairs[:, 1]
    counts = np.bincount(vals, minlength=4).astype(np.float64)
    expected = counts.sum() / 4.0
    chi = float(np.sum((counts - expected) ** 2 / expected)) if expected else float('nan')
    # chi-square df=3 survival approximation via exp is rough but useful as a screening metric.
    p = math.exp(-0.5 * chi)
    return {'counts_00_01_10_11': counts.astype(int).tolist(), 'chi_square': chi, 'p_value_approx': float(p), 'passed': bool(p >= 0.01)}


def main():
    ap = argparse.ArgumentParser(description='NIST-style randomness tests for full-frame ciphertext bytes. No ROI sidecar is needed.')
    ap.add_argument('--bitstream', default=None, help='Raw ciphertext dump, preferably from framework_full_encrypt.py --cipher_dump')
    ap.add_argument('--cipher_video', default=None, help='Alternative: test decoded encrypted frame bytes')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max_bits', type=int, default=2_000_000)
    ap.add_argument('--max_frames', type=int, default=0)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--block_size', type=int, default=128)
    args = ap.parse_args()

    out = ensure_dir(args.out)
    data = read_bytes_source(args.bitstream, args.cipher_video, args.max_bits, args.max_frames, args.stride)
    bits = bits_from_bytes(data)
    if args.max_bits and args.max_bits > 0:
        bits = bits[:args.max_bits]
    write_bytes(out / 'tested_bytes.bin', data[: math.ceil(bits.size / 8)])
    report = {
        'test_type': 'full_frame_nist_style_runner',
        'source': args.bitstream or args.cipher_video,
        'n_bytes_loaded': len(data),
        'n_bits_tested': int(bits.size),
        'tests': {
            'monobit': monobit(bits),
            'block_frequency': block_frequency(bits, args.block_size),
            'runs': runs(bits),
            'serial_2bit': serial_2bit(bits),
            'byte_entropy': byte_entropy(data),
        },
        'note': 'No rois.jsonl is used. For the cleanest ciphertext bitstream, create --bitstream with framework_full_encrypt.py --cipher_dump.',
    }
    write_json(out / 'nist_report.json', report)
    print(f'[OK] Wrote {out / "nist_report.json"}')


if __name__ == '__main__':
    main()
