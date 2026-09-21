#!/usr/bin/env python3
"""
02_differential_sensitivity_full_frame.py

Full-frame differential/sensitivity tests. No rois.jsonl argument is used.
Computes NPCR/UACI for plain-vs-cipher, optional plaintext sensitivity, and optional key sensitivity.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

from unified_utils import (
    derive_key_variant,
    ensure_dir,
    framework_cli_encrypt,
    iter_frames,
    make_one_pixel_variant,
    npcr_uaci,
    select_mode,
    video_info,
    write_csv,
    write_json,
)


def compare_videos(a_path: str, b_path: str, mode: str, max_frames: int, stride: int):
    rows = []
    for (proc_i, orig_i, fa), (_, _, fb) in zip(iter_frames(a_path, max_frames, stride), iter_frames(b_path, max_frames, stride)):
        a = select_mode(fa, mode)
        b = select_mode(fb, mode)
        npcr, uaci = npcr_uaci(a, b)
        rows.append({'proc_frame_idx': proc_i, 'orig_frame_idx': orig_i, 'npcr_percent': npcr, 'uaci_percent': uaci})
    return rows


def summarize(rows):
    if not rows:
        return {'frames': 0, 'mean_npcr_percent': float('nan'), 'mean_uaci_percent': float('nan')}
    return {
        'frames': len(rows),
        'mean_npcr_percent': float(mean([r['npcr_percent'] for r in rows])),
        'mean_uaci_percent': float(mean([r['uaci_percent'] for r in rows])),
    }


def main():
    ap = argparse.ArgumentParser(description='Full-frame NPCR/UACI and sensitivity tests. No ROI sidecar is needed.')
    ap.add_argument('--plain', required=True)
    ap.add_argument('--cipher', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--mode', choices=['gray', 'color'], default='color')
    ap.add_argument('--max_frames', type=int, default=0)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--framework_path', default='framework_full_encrypt.py')
    ap.add_argument('--master_key', default=None, help='Needed for plaintext/key sensitivity encryption runs')
    ap.add_argument('--plain2', default=None, help='Optional second plaintext video')
    ap.add_argument('--cipher2', default=None, help='Optional second ciphertext video')
    ap.add_argument('--run_plaintext_sensitivity', action='store_true')
    ap.add_argument('--run_key_sensitivity', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    out = ensure_dir(args.out)
    report = {
        'test_type': 'full_frame_differential_sensitivity',
        'plain_info': video_info(args.plain),
        'cipher_info': video_info(args.cipher),
        'mode': args.mode,
        'note': 'Full-frame evaluation: every pixel participates in NPCR/UACI; no rois.jsonl is used.',
        'summary': {},
    }

    rows = compare_videos(args.plain, args.cipher, args.mode, args.max_frames, args.stride)
    write_csv(out / 'npcr_uaci_plain_vs_cipher.csv', rows)
    report['summary']['plain_vs_cipher'] = summarize(rows)

    if args.plain2 and args.cipher2:
        rows2 = compare_videos(args.cipher, args.cipher2, args.mode, args.max_frames, args.stride)
        write_csv(out / 'plaintext_sensitivity_cipher_vs_cipher2.csv', rows2)
        report['summary']['plaintext_sensitivity_cipher_vs_cipher2'] = summarize(rows2)
    elif args.run_plaintext_sensitivity:
        if not args.master_key:
            report['summary']['plaintext_sensitivity'] = {'skipped': 'Provide --master_key to generate and encrypt a one-pixel plaintext variant.'}
        else:
            plain_variant = out / 'plain_one_pixel_variant.mkv'
            cipher_variant = out / 'cipher_one_pixel_variant.mp4'
            make_one_pixel_variant(args.plain, plain_variant, args.max_frames, args.stride)
            framework_cli_encrypt(args.framework_path, str(plain_variant), str(cipher_variant), args.master_key, verbose=args.verbose)
            rows3 = compare_videos(args.cipher, str(cipher_variant), args.mode, args.max_frames, 1)
            write_csv(out / 'plaintext_sensitivity_cipher_vs_one_pixel_variant.csv', rows3)
            report['summary']['plaintext_sensitivity_one_pixel'] = summarize(rows3)

    if args.run_key_sensitivity:
        if not args.master_key:
            report['summary']['key_sensitivity'] = {'skipped': 'Provide --master_key to generate a key-variant ciphertext.'}
        else:
            key2 = derive_key_variant(args.master_key, 'key_sensitivity')
            cipher_key_variant = out / 'cipher_key_variant.mp4'
            framework_cli_encrypt(args.framework_path, args.plain, str(cipher_key_variant), key2, verbose=args.verbose)
            rows4 = compare_videos(args.cipher, str(cipher_key_variant), args.mode, args.max_frames, args.stride)
            write_csv(out / 'key_sensitivity_cipher_vs_key_variant.csv', rows4)
            report['summary']['key_sensitivity_cipher_vs_key_variant'] = summarize(rows4)

    write_json(out / 'report.json', report)
    print(f'[OK] Wrote {out / "report.json"}')


if __name__ == '__main__':
    main()
