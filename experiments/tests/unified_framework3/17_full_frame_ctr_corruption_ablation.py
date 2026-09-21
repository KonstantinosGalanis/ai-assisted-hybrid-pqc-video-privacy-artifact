#!/usr/bin/env python3
from __future__ import annotations

"""
17_full_frame_ctr_corruption_ablation.py

Full-frame AES-CTR corruption ablation.

This is the full-frame counterpart to ROI mask-expansion/robustness ablations.
Because full-frame encryption has no RoI masks, padding/dilation are not
applicable. Instead, this test sweeps corruption strengths and measures:
- ciphertext modification magnitude
- post-decryption plaintext damage
- runtime/decrypt success
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from unified_utils import ensure_dir, video_info, write_csv, write_json

# Reuse the implementation from Test 06 when available.
import importlib.util


def import_test06(this_dir: Path):
    p = this_dir / "06_robustness_noise_occlusion_roi.py"
    spec = importlib.util.spec_from_file_location("full_frame_test06_reuse", str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_float_list(text: str) -> List[float]:
    out = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Full-frame AES-CTR corruption/tamper ablation. No ROI masks are used.")
    ap.add_argument("--plain", required=True)
    ap.add_argument("--cipher", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--framework_path", default="framework_full_encrypt.py")
    ap.add_argument("--master_key", required=True)
    ap.add_argument("--attacks", default="gaussian,saltpepper,jpeg_crf,frame_loss")
    ap.add_argument("--gaussian_sigmas", default="5,10,20,40")
    ap.add_argument("--saltpepper_probs", default="0.001,0.005,0.01,0.05")
    ap.add_argument("--jpeg_crfs", default="18,23,28,35")
    ap.add_argument("--frame_loss_probs", default="0.01,0.05,0.10")
    ap.add_argument("--max_frames", type=int, default=50)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="Optional max number of configs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out = ensure_dir(args.out)
    t06 = import_test06(Path(__file__).resolve().parent)
    attack_map = {
        "gaussian": parse_float_list(args.gaussian_sigmas),
        "saltpepper": parse_float_list(args.saltpepper_probs),
        "jpeg_crf": parse_float_list(args.jpeg_crfs),
        "frame_loss": parse_float_list(args.frame_loss_probs),
    }
    requested = [a.strip() for a in args.attacks.split(",") if a.strip()]
    configs = []
    for attack in requested:
        for strength in attack_map.get(attack, []):
            configs.append((attack, strength))
    if args.limit and args.limit > 0:
        configs = configs[: args.limit]

    rows: List[Dict[str, Any]] = []
    for idx, (attack, strength) in enumerate(configs):
        label = f"{idx:03d}_{attack}_{str(strength).replace('.', 'p')}"
        cdir = ensure_dir(out / label)
        attacked = cdir / "attacked_cipher.mkv"
        decrypted = cdir / "decrypted_attacked.mkv"
        t0 = time.perf_counter()
        attack_meta = t06.write_attacked_video_lossless(args.cipher, str(attacked), attack, strength, max_frames=args.max_frames, seed=args.seed)
        attack_seconds = time.perf_counter() - t0
        dec_ok = True
        dec_err = ""
        t1 = time.perf_counter()
        try:
            from unified_utils import framework_cli_decrypt
            framework_cli_decrypt(args.framework_path, str(attacked), str(decrypted), args.master_key, verbose=args.verbose)
        except Exception as exc:
            dec_ok = False
            dec_err = repr(exc)
        decrypt_seconds = time.perf_counter() - t1
        cipher_delta = t06.compare_videos(args.cipher, str(attacked), args.max_frames, args.stride)
        plain_damage = t06.compare_videos(args.plain, str(decrypted), args.max_frames, args.stride) if dec_ok else {"frames": 0}
        row = {
            "label": label,
            "attack": attack,
            "strength": float(strength),
            "attack_seconds": attack_seconds,
            "decrypt_seconds": decrypt_seconds,
            "decrypt_ok": dec_ok,
            "decrypt_error": dec_err,
            "attacked_size_bytes": Path(attacked).stat().st_size if attacked.exists() else 0,
            "decrypted_size_bytes": Path(decrypted).stat().st_size if decrypted.exists() else 0,
            "frames_written": attack_meta.get("frames_written"),
            "frame_loss_replacements": attack_meta.get("frame_loss_replacements"),
            "cipher_delta_mean_mse": cipher_delta.get("mean_mse"),
            "cipher_delta_mean_psnr_db": cipher_delta.get("mean_psnr_db"),
            "plain_damage_mean_mse": plain_damage.get("mean_mse"),
            "plain_damage_mean_psnr_db": plain_damage.get("mean_psnr_db"),
            "plain_damage_mean_ssim": plain_damage.get("mean_ssim"),
        }
        rows.append(row)
        if args.verbose:
            print(label, row)

    write_csv(out / "ablation_rows.csv", rows)
    report = {
        "test_type": "full_frame_aes_ctr_corruption_ablation",
        "interpretation": "Full-frame AES-CTR has no mask expansion parameter. This ablation sweeps corruption strengths and measures decrypted damage.",
        "inputs": {"plain": args.plain, "cipher": args.cipher},
        "plain_info": video_info(args.plain),
        "cipher_info": video_info(args.cipher),
        "configs_requested": len(configs),
        "rows": rows,
        "summary": {"rows": len(rows), "decrypt_failures": sum(1 for r in rows if not r["decrypt_ok"]), "all_passed": all(r["decrypt_ok"] for r in rows)},
    }
    write_json(out / "report.json", report)
    print(json.dumps({"ok": True, "report": str(out / "report.json"), "rows": len(rows), "all_passed": report["summary"]["all_passed"]}))


if __name__ == "__main__":
    main()
