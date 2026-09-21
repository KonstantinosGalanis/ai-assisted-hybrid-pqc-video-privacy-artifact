#!/usr/bin/env python3
from __future__ import annotations

"""
16_full_frame_ctr_iv_context_binding.py

Checks AES-CTR IV/context uniqueness for the full-frame framework.
No rois.jsonl, payload, or manifest is used.

For CTR mode, the critical safety property is that the same AES key/IV pair is
never reused for two different plaintext streams. This script checks the frame
IV derivation exposed by framework_full_encrypt.py.
"""

import argparse
import base64
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List

from unified_utils import ensure_dir, video_info, write_json


def import_framework(path: str):
    spec = importlib.util.spec_from_file_location("framework_full_encrypt_under_test", str(Path(path).resolve()))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import framework from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def b64(x: bytes) -> str:
    return base64.b64encode(x).decode("ascii")


def main() -> None:
    ap = argparse.ArgumentParser(description="Full-frame AES-CTR IV/context uniqueness checks.")
    ap.add_argument("--plain", required=True, help="Plain/source video used to infer frame count, dimensions, fps")
    ap.add_argument("--out", required=True)
    ap.add_argument("--framework_path", default="framework_full_encrypt.py")
    ap.add_argument("--master_key", required=True)
    ap.add_argument("--master_key_alt", default=None)
    ap.add_argument("--max_frames", type=int, default=0)
    args = ap.parse_args()

    out = ensure_dir(args.out)
    fw = import_framework(args.framework_path)
    required = ["derive_master_key_from_password", "derive_subkey", "derive_ctr_iv"]
    missing = [name for name in required if not hasattr(fw, name)]
    if missing:
        raise RuntimeError(f"Framework missing required functions: {missing}")

    info = video_info(args.plain)
    frame_count = int(info.get("frames") or 0)
    if args.max_frames and args.max_frames > 0:
        frame_count = min(frame_count, int(args.max_frames)) if frame_count else int(args.max_frames)
    width = int(info.get("width") or 0)
    height = int(info.get("height") or 0)

    master = fw.derive_master_key_from_password(args.master_key)
    video_key = fw.derive_subkey(master, "video_full_aes256ctr")
    ivs: List[bytes] = []
    for fi in range(frame_count):
        ivs.append(fw.derive_ctr_iv(video_key, kind="video", frame_idx=fi, extra=f"w={width}|h={height}|fmt=bgr24"))

    iv_strings = [b64(x) for x in ivs]
    dup = sorted({x for x in iv_strings if iv_strings.count(x) > 1})
    malformed = [x for x in ivs if not isinstance(x, (bytes, bytearray)) or len(x) != 16]

    alt_section: Dict[str, Any] = {"skipped": "no --master_key_alt supplied"}
    if args.master_key_alt:
        alt_master = fw.derive_master_key_from_password(args.master_key_alt)
        alt_video_key = fw.derive_subkey(alt_master, "video_full_aes256ctr")
        alt_ivs = [fw.derive_ctr_iv(alt_video_key, kind="video", frame_idx=fi, extra=f"w={width}|h={height}|fmt=bgr24") for fi in range(frame_count)]
        same_positions = sum(1 for a, b in zip(ivs, alt_ivs) if a == b)
        alt_section = {
            "frames_compared": frame_count,
            "same_iv_same_position_under_alt_key": same_positions,
            "all_changed_under_alt_key": same_positions == 0,
        }

    # Audio-tail/frame segment examples are derived separately by kind, so check domain separation.
    audio_examples = []
    for fi in range(min(frame_count, 10)):
        vi = fw.derive_ctr_iv(video_key, kind="video", frame_idx=fi, extra=f"w={width}|h={height}|fmt=bgr24")
        ai = fw.derive_ctr_iv(video_key, kind="audio", frame_idx=fi, extra="samples=100|channels=2")
        audio_examples.append({"frame_idx": fi, "video_audio_iv_differ": vi != ai})

    checks = {
        "iv_length_16_bytes": len(malformed) == 0,
        "no_duplicate_video_ivs": len(dup) == 0,
        "has_at_least_one_frame": frame_count > 0,
        "video_audio_domain_separated": all(x["video_audio_iv_differ"] for x in audio_examples) if audio_examples else True,
    }
    if args.master_key_alt:
        checks["alt_key_changes_ivs"] = bool(alt_section.get("all_changed_under_alt_key"))

    report = {
        "test_type": "full_frame_aes_ctr_iv_context_binding",
        "interpretation": "AES-CTR requires no repeated key/IV pairs. This checks deterministic per-frame IV derivation and domain separation.",
        "inputs": {"plain": args.plain, "framework_path": args.framework_path},
        "video_info": info,
        "frame_count_checked": frame_count,
        "iv_summary": {
            "iv_count": len(ivs),
            "unique_iv_count": len(set(iv_strings)),
            "duplicate_ivs": dup[:20],
            "malformed_iv_count": len(malformed),
            "first_iv_b64": iv_strings[0] if iv_strings else None,
            "last_iv_b64": iv_strings[-1] if iv_strings else None,
        },
        "domain_separation_examples": audio_examples,
        "alt_key_check": alt_section,
        "checks": checks,
        "summary": {"all_passed": all(bool(v) for v in checks.values())},
    }
    write_json(out / "report.json", report)
    print(json.dumps({"ok": True, "report": str(out / "report.json"), "all_passed": report["summary"]["all_passed"]}))


if __name__ == "__main__":
    main()
