#!/usr/bin/env python3
"""
11_param_sweep_full_frame.py

Full-frame AES key-variant sweep. No rois.jsonl argument is used.
AES-256-CTR has no chaotic map parameters, so this script sweeps key variants and reports full-frame metrics.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

from unified_utils import adjacent_corr, derive_key_variant, ensure_dir, entropy_u8, framework_cli_encrypt, iter_frames, npcr_uaci, select_mode, video_info, write_csv, write_json


def metrics_for_cipher(plain: str, cipher: str, mode: str, max_frames: int, stride: int):
    rows = []
    for (proc_i, orig_i, fp), (_, _, fc) in zip(iter_frames(plain, max_frames, stride), iter_frames(cipher, max_frames, stride)):
        p = select_mode(fp, mode)
        c = select_mode(fc, mode)
        npcr, uaci = npcr_uaci(p, c)
        rows.append({'proc_frame_idx': proc_i, 'orig_frame_idx': orig_i, 'npcr_percent': npcr, 'uaci_percent': uaci, 'cipher_entropy': entropy_u8(c), 'cipher_corr_h': adjacent_corr(c, 'h'), 'cipher_corr_v': adjacent_corr(c, 'v'), 'cipher_corr_d': adjacent_corr(c, 'd')})
    return rows


def avg(rows, key):
    return float(mean([r[key] for r in rows])) if rows else float('nan')


def main():
    ap = argparse.ArgumentParser(description='Full-frame AES key-variant sweep. No ROI sidecar is needed.')
    ap.add_argument('--plain', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--framework_path', default='framework_full_encrypt.py')
    ap.add_argument('--master_key', required=True)
    ap.add_argument('--variants', type=int, default=5)
    ap.add_argument('--mode', choices=['gray', 'color'], default='color')
    ap.add_argument('--max_frames', type=int, default=0)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    out = ensure_dir(args.out)
    sweep_rows = []
    details = {}
    for i in range(max(1, args.variants)):
        key = args.master_key if i == 0 else derive_key_variant(args.master_key, f'sweep_{i}')
        cipher = out / f'cipher_key_variant_{i}.mp4'
        res = framework_cli_encrypt(args.framework_path, args.plain, str(cipher), key, verbose=args.verbose)
        rows = metrics_for_cipher(args.plain, str(cipher), args.mode, args.max_frames, args.stride)
        detail_csv = out / f'variant_{i}_frame_metrics.csv'
        write_csv(detail_csv, rows)
        row = {
            'variant': i,
            'key_label': 'master_key' if i == 0 else f'sha256(master_key|variant=sweep_{i})',
            'cipher_path': str(cipher),
            'encrypt_time_s': res['elapsed_s'],
            'mean_npcr_percent': avg(rows, 'npcr_percent'),
            'mean_uaci_percent': avg(rows, 'uaci_percent'),
            'mean_cipher_entropy': avg(rows, 'cipher_entropy'),
            'mean_cipher_corr_h': avg(rows, 'cipher_corr_h'),
            'mean_cipher_corr_v': avg(rows, 'cipher_corr_v'),
            'mean_cipher_corr_d': avg(rows, 'cipher_corr_d'),
        }
        sweep_rows.append(row)
        details[str(i)] = {'frames': len(rows), 'frame_metrics_csv': str(detail_csv)}
    write_csv(out / 'sweep_summary.csv', sweep_rows)
    report = {'test_type': 'full_frame_aes_key_variant_sweep', 'plain_info': video_info(args.plain), 'summary_csv': str(out / 'sweep_summary.csv'), 'variants': sweep_rows, 'details': details, 'note': 'AES-256-CTR has no chaos parameters. This sweep varies the full-frame AES key instead; no rois.jsonl is used.'}
    write_json(out / 'report.json', report)
    print(f'[OK] Wrote {out / "report.json"}')


if __name__ == '__main__':
    main()
