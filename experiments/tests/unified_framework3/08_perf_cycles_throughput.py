#!/usr/bin/env python3
"""
08_perf_cycles_throughput_full_frame.py

Full-frame AES performance test. No rois.jsonl, payload, or detector timing is used.
Measures end-to-end encryption/decryption time and throughput using framework_full_encrypt.py.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path
from statistics import mean

from unified_utils import ensure_dir, framework_cli_decrypt, framework_cli_encrypt, raw_frame_bytes, sha256_file, video_info, write_json


def main():
    ap = argparse.ArgumentParser(description='Full-frame AES performance and throughput. No ROI sidecar is needed.')
    ap.add_argument('--plain', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--framework_path', default='framework_full_encrypt.py')
    ap.add_argument('--master_key', required=True)
    ap.add_argument('--runs', type=int, default=1)
    ap.add_argument('--cipher', default=None, help='Optional existing ciphertext for decrypt timing')
    ap.add_argument('--make_cipher_dump', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    out = ensure_dir(args.out)
    info = video_info(args.plain)
    raw_bytes_est = int(info['width'] * info['height'] * 3 * info['frames']) if info['frames'] else 0
    enc_runs = []
    dec_runs = []
    last_cipher = args.cipher

    for i in range(max(1, args.runs)):
        cipher_path = out / f'perf_cipher_run{i}.mp4'
        dump_path = out / f'cipher_dump_run{i}.bin' if args.make_cipher_dump else None
        res = framework_cli_encrypt(args.framework_path, args.plain, str(cipher_path), args.master_key, cipher_dump=str(dump_path) if dump_path else None, verbose=args.verbose)
        cinfo = video_info(str(cipher_path))
        enc_runs.append({'run': i, 'elapsed_s': res['elapsed_s'], 'cipher_path': str(cipher_path), 'cipher_file_bytes': cinfo['file_bytes'], 'cipher_sha256': sha256_file(cipher_path), 'cipher_dump': str(dump_path) if dump_path else None})
        last_cipher = str(cipher_path)
        dec_path = out / f'perf_decrypted_run{i}.mp4'
        dres = framework_cli_decrypt(args.framework_path, str(cipher_path), str(dec_path), args.master_key, verbose=args.verbose)
        dinfo = video_info(str(dec_path))
        dec_runs.append({'run': i, 'elapsed_s': dres['elapsed_s'], 'decrypted_path': str(dec_path), 'decrypted_file_bytes': dinfo['file_bytes'], 'decrypted_sha256': sha256_file(dec_path)})

    report = {
        'test_type': 'full_frame_perf_cycles_throughput',
        'plain_info': info,
        'raw_video_bytes_estimate': raw_bytes_est,
        'encrypt_runs': enc_runs,
        'decrypt_runs': dec_runs,
        'summary': {
            'mean_encrypt_time_s': float(mean([r['elapsed_s'] for r in enc_runs])),
            'mean_decrypt_time_s': float(mean([r['elapsed_s'] for r in dec_runs])),
            'encrypt_file_throughput_bytes_per_s': float(info['file_bytes'] / mean([r['elapsed_s'] for r in enc_runs])) if enc_runs else None,
            'decrypt_file_throughput_bytes_per_s': float((video_info(last_cipher)['file_bytes'] if last_cipher else 0) / mean([r['elapsed_s'] for r in dec_runs])) if dec_runs and last_cipher else None,
            'encrypt_raw_pixel_throughput_bytes_per_s': float(raw_bytes_est / mean([r['elapsed_s'] for r in enc_runs])) if raw_bytes_est and enc_runs else None,
        },
        'note': 'Full-frame AES-CTR end-to-end timing; detector/ROI/payload timings are not applicable.',
    }
    write_json(out / 'report.json', report)
    print(f'[OK] Wrote {out / "report.json"}')


if __name__ == '__main__':
    main()
