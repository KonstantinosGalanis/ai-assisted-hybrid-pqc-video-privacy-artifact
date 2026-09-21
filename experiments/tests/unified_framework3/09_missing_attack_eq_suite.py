#!/usr/bin/env python3
"""
09_attack_eq_suite_full_frame.py

Full-frame attack / encryption-quality suite. No rois.jsonl argument is used.
Covers COA, KPA, optional CPA, optional CCA, EQ, pixel resemblance/disparity.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

import cv2
import numpy as np

from unified_utils import (
    adjacent_corr,
    derive_key_variant,
    ensure_dir,
    entropy_u8,
    framework_cli_decrypt,
    framework_cli_encrypt,
    iter_frames,
    mae,
    make_inverted_variant,
    mse,
    npcr_uaci,
    psnr,
    select_mode,
    snr,
    video_info,
    write_csv,
    write_json,
)


def compare_rows(a_path, b_path, mode, max_frames, stride):
    rows = []
    for (proc_i, orig_i, a), (_, _, b) in zip(iter_frames(a_path, max_frames, stride), iter_frames(b_path, max_frames, stride)):
        ax = select_mode(a, mode)
        bx = select_mode(b, mode)
        npcr, uaci = npcr_uaci(ax, bx)
        rows.append({'proc_frame_idx': proc_i, 'orig_frame_idx': orig_i, 'mse': mse(ax, bx), 'mae': mae(ax, bx), 'psnr_db': psnr(ax, bx), 'snr_db': snr(ax, bx), 'npcr_percent': npcr, 'uaci_percent': uaci})
    return rows


def summarize(rows):
    if not rows:
        return {'frames': 0}
    keys = [k for k in rows[0].keys() if k not in {'proc_frame_idx', 'orig_frame_idx'}]
    return {'frames': len(rows), **{f'mean_{k}': float(mean([r[k] for r in rows])) for k in keys}}


def coa_stats(cipher_path, max_frames, stride):
    rows = []
    for proc_i, orig_i, fr in iter_frames(cipher_path, max_frames, stride):
        rows.append({'proc_frame_idx': proc_i, 'orig_frame_idx': orig_i, 'entropy_gray': entropy_u8(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)), 'corr_h': adjacent_corr(fr, 'h'), 'corr_v': adjacent_corr(fr, 'v'), 'corr_d': adjacent_corr(fr, 'd')})
    return rows


def main():
    ap = argparse.ArgumentParser(description='Full-frame attack/EQ suite. No ROI sidecar is needed.')
    ap.add_argument('--plain', required=True)
    ap.add_argument('--cipher', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--mode', choices=['gray', 'color'], default='color')
    ap.add_argument('--max_frames', type=int, default=500)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--framework_path', default='framework_full_encrypt.py')
    ap.add_argument('--master_key', default=None)
    ap.add_argument('--run_cpa', action='store_true')
    ap.add_argument('--run_cca', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    out = ensure_dir(args.out)
    report = {'test_type': 'full_frame_attack_eq_suite', 'plain_info': video_info(args.plain), 'cipher_info': video_info(args.cipher), 'summary': {}, 'note': 'Full-frame attack metrics; no rois.jsonl/payload/manifest is used.'}

    coa = coa_stats(args.cipher, args.max_frames, args.stride)
    write_csv(out / 'coa_cipher_stats.csv', coa)
    report['summary']['ciphertext_only'] = summarize(coa)

    kpa = compare_rows(args.plain, args.cipher, args.mode, args.max_frames, args.stride)
    write_csv(out / 'kpa_plain_vs_cipher_eq.csv', kpa)
    report['summary']['known_plaintext_eq'] = summarize(kpa)

    if args.run_cpa:
        if not args.master_key:
            report['summary']['chosen_plaintext'] = {'skipped': 'Provide --master_key to generate chosen-plaintext ciphertexts.'}
        else:
            chosen_plain = out / 'chosen_plain_inverted.mkv'
            chosen_cipher = out / 'chosen_cipher_inverted.mp4'
            make_inverted_variant(args.plain, chosen_plain, args.max_frames, args.stride)
            framework_cli_encrypt(args.framework_path, str(chosen_plain), str(chosen_cipher), args.master_key, verbose=args.verbose)
            cpa = compare_rows(args.cipher, str(chosen_cipher), args.mode, args.max_frames, args.stride)
            write_csv(out / 'cpa_cipher_vs_chosen_cipher.csv', cpa)
            report['summary']['chosen_plaintext'] = summarize(cpa)

    if args.run_cca:
        if not args.master_key:
            report['summary']['chosen_ciphertext'] = {'skipped': 'Provide --master_key to decrypt ciphertext.'}
        else:
            dec = out / 'cca_decrypted_cipher.mp4'
            framework_cli_decrypt(args.framework_path, args.cipher, str(dec), args.master_key, verbose=args.verbose)
            cca = compare_rows(args.plain, str(dec), args.mode, args.max_frames, args.stride)
            write_csv(out / 'cca_plain_vs_decrypted.csv', cca)
            report['summary']['chosen_ciphertext_decryption_quality'] = summarize(cca)

    write_json(out / 'report.json', report)
    print(f'[OK] Wrote {out / "report.json"}')


if __name__ == '__main__':
    main()
