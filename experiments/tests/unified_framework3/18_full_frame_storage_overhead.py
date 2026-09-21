#!/usr/bin/env python3
from __future__ import annotations

"""
18_full_frame_storage_overhead.py

Full-frame AES storage/overhead analysis.
No RoI sidecar, payload, or manifest is used.

For full-frame encryption, encrypted fraction is 100%. This test reports:
- plain/cipher/decrypted sizes
- overhead ratio vs plain
- raw frame byte estimate
- optional cipher_dump / keystream size if present
- codec/container metadata through ffprobe when available
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from unified_utils import ensure_dir, sha256_file, video_info, write_csv, write_json

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}
BIT_EXTS = {".bin", ".bits", ".bit", ".dat", ".raw"}


def run(cmd: List[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{p.stderr[-2000:]}")
    return p.stdout


def ffprobe_json(path: str) -> Optional[Dict[str, Any]]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        return json.loads(run([ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path]))
    except Exception as exc:
        return {"error": repr(exc)}


def pick_file(d: Path, stems: List[str], exts: set[str]) -> Optional[Path]:
    files = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts]
    for stem in stems:
        for p in files:
            if p.stem.lower() == stem:
                return p
    for stem in stems:
        for p in files:
            if p.name.lower().startswith(stem):
                return p
    return None


def analyze_dataset(d: Path, plain_arg: Optional[str] = None, cipher_arg: Optional[str] = None, decrypted_arg: Optional[str] = None, bitstream_arg: Optional[str] = None) -> Dict[str, Any]:
    plain = Path(plain_arg) if plain_arg else pick_file(d, ["plain", "original", "source", "input"], VIDEO_EXTS)
    cipher = Path(cipher_arg) if cipher_arg else pick_file(d, ["cipher", "encrypted", "enc"], VIDEO_EXTS)
    decrypted = Path(decrypted_arg) if decrypted_arg else pick_file(d, ["decrypted", "decrypt", "recovered", "dec"], VIDEO_EXTS)
    bitstream = Path(bitstream_arg) if bitstream_arg else pick_file(d, ["cipher_dump", "bitstream", "keystream", "dump"], BIT_EXTS)

    row: Dict[str, Any] = {"dataset": str(d), "status": "ok"}
    if not plain or not plain.exists() or not cipher or not cipher.exists():
        row.update({"status": "missing_plain_or_cipher", "plain": str(plain) if plain else None, "cipher": str(cipher) if cipher else None})
        return row

    pinfo = video_info(plain)
    cinfo = video_info(cipher)
    plain_size = int(Path(plain).stat().st_size)
    cipher_size = int(Path(cipher).stat().st_size)
    raw_bytes = int(pinfo.get("width", 0) * pinfo.get("height", 0) * 3 * pinfo.get("frames", 0))
    row.update({
        "bucket": "100% full-frame",
        "encrypted_fraction": 1.0,
        "plain": str(plain),
        "cipher": str(cipher),
        "decrypted": str(decrypted) if decrypted and decrypted.exists() else None,
        "bitstream": str(bitstream) if bitstream and bitstream.exists() else None,
        "plain_video_size_bytes": plain_size,
        "cipher_video_size_bytes": cipher_size,
        "decrypted_video_size_bytes": int(decrypted.stat().st_size) if decrypted and decrypted.exists() else 0,
        "cipher_dump_or_keystream_size_bytes": int(bitstream.stat().st_size) if bitstream and bitstream.exists() else 0,
        "total_protected_package_size_bytes": cipher_size,
        "overhead_ratio_vs_plain": float(cipher_size / plain_size) if plain_size else None,
        "raw_frame_bytes_estimate_bgr": raw_bytes,
        "cipher_ratio_vs_raw_frame_bytes": float(cipher_size / raw_bytes) if raw_bytes else None,
        "width": pinfo.get("width"),
        "height": pinfo.get("height"),
        "fps": pinfo.get("fps"),
        "frames": pinfo.get("frames"),
        "plain_sha256": sha256_file(plain),
        "cipher_sha256": sha256_file(cipher),
        "plain_ffprobe": ffprobe_json(str(plain)),
        "cipher_ffprobe": ffprobe_json(str(cipher)),
    })
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Full-frame AES storage overhead. No ROI sidecar/payload/manifest needed.")
    ap.add_argument("--datasets", default=None, help="Comma-separated dataset folders. If omitted, uses --plain/--cipher parent or current folder.")
    ap.add_argument("--plain", default=None)
    ap.add_argument("--cipher", default=None)
    ap.add_argument("--decrypted", default=None)
    ap.add_argument("--bitstream", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = ensure_dir(args.out)
    if args.datasets:
        dirs = [Path(x.strip()).resolve() for x in args.datasets.replace(";", ",").split(",") if x.strip()]
    elif args.plain:
        dirs = [Path(args.plain).resolve().parent]
    else:
        dirs = [Path.cwd()]

    rows = [analyze_dataset(d, args.plain, args.cipher, args.decrypted, args.bitstream) for d in dirs]
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    groups: Dict[str, Any] = {}
    for r in ok_rows:
        bucket = r.get("bucket", "100% full-frame")
        g = groups.setdefault(bucket, {"count": 0, "datasets": [], "mean_overhead_ratio_vs_plain": 0.0, "mean_cipher_video_size_bytes": 0.0})
        g["count"] += 1
        g["datasets"].append(r["dataset"])
        g["mean_overhead_ratio_vs_plain"] += float(r.get("overhead_ratio_vs_plain") or 0.0)
        g["mean_cipher_video_size_bytes"] += float(r.get("cipher_video_size_bytes") or 0.0)
    for g in groups.values():
        if g["count"]:
            g["mean_overhead_ratio_vs_plain"] /= g["count"]
            g["mean_cipher_video_size_bytes"] /= g["count"]

    write_csv(out / "storage_rows.csv", rows)
    report = {"test_type": "full_frame_aes_storage_overhead", "interpretation": "Full-frame encryption protects 100% of decoded pixels; package is just ciphertext video plus optional dumps.", "rows": rows, "groups": groups, "summary": {"rows": len(rows), "ok_rows": len(ok_rows), "all_passed": len(ok_rows) == len(rows) and bool(rows)}}
    write_json(out / "report.json", report)
    print(json.dumps({"ok": True, "report": str(out / "report.json"), "rows": len(rows), "all_passed": report["summary"]["all_passed"]}))


if __name__ == "__main__":
    main()
