#!/usr/bin/env python3
from __future__ import annotations

"""
14_chaos_payload_tamper_response.py

Chaos payload/security test for the current payload-aware framework.
This is the chaos/HMAC-manifest counterpart of the later ChaCha AEAD test:
it mutates one field at a time and verifies that decrypt/reconstruction is
rejected through either manifest SHA-256 checks or payload HMAC tag checks.
"""

import argparse
import base64
import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: str) -> List[dict]:
    out: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def write_jsonl(path: str, records: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")


def load_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_manifest(path: str, manifest: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def flip_b64_byte(text: str) -> str:
    raw = bytearray(base64.b64decode(text.encode("ascii")))
    if not raw:
        raw = bytearray(b"\x00")
    raw[0] ^= 0x01
    return base64.b64encode(bytes(raw)).decode("ascii")


def flip_hex_char(text: str) -> str:
    if not text:
        return "00"
    c = text[0].lower()
    repl = "0" if c != "0" else "1"
    return repl + text[1:]


def find_first_roi(records: List[dict]) -> Tuple[int, int]:
    for i, rec in enumerate(records):
        if rec.get("type") in {"meta", "tail_audio"}:
            continue
        rois = rec.get("rois") or []
        if rois:
            return i, 0
    raise ValueError("No ROI payload record found in payload.jsonl")


def mutate_first_roi(records: List[dict], mut: Callable[[dict, dict], None]) -> List[dict]:
    out = copy.deepcopy(records)
    fi, ri = find_first_roi(out)
    mut(out[fi], out[fi]["rois"][ri])
    return out


def make_payload_case(
    base_records: List[dict],
    manifest: dict,
    out_dir: Path,
    name: str,
    mut: Callable[[List[dict]], List[dict]],
) -> Tuple[str, str]:
    payload_path = out_dir / f"{name}.payload.jsonl"
    manifest_path = out_dir / f"{name}.manifest.json"

    records = mut(base_records)
    write_jsonl(str(payload_path), records)

    m = copy.deepcopy(manifest)

    # Point to the tampered payload file, but keep the original payload_sha256.
    # This tests whether the protected package rejects modified payload data.
    m["payload_path"] = payload_path.name

    # IMPORTANT: do NOT recompute this for payload tamper cases.
    # m["payload_sha256"] = sha256_file(str(payload_path))

    write_manifest(str(manifest_path), m)
    return str(payload_path), str(manifest_path)


def make_manifest_case(payload_path: str, manifest: dict, out_dir: Path, name: str, mut: Callable[[dict], None]) -> Tuple[str, str]:
    manifest_path = out_dir / f"{name}.manifest.json"
    m = copy.deepcopy(manifest)
    mut(m)
    write_manifest(str(manifest_path), m)
    return payload_path, str(manifest_path)


def run_decrypt(args, payload_path: str, manifest_path: str, out_path: str) -> Tuple[bool, str, str, int]:
    cmd = [
        sys.executable, str(Path(args.framework_faster_path).resolve()),
        "--mode", "decrypt",
        "--in", args.cipher,
        "--out", out_path,
        "--key", args.master_key,
        "--roi_sidecar", args.roi_sidecar,
        "--payload", payload_path,
        "--manifest", manifest_path,
        "--video_preview_mode", "chaos",
        "--audio_preview_mode", "chaos",
    ]
    if args.framework_base_path:
        cmd.extend(["--base_framework", args.framework_base_path])
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return p.returncode == 0, (p.stdout or "")[-3000:], (p.stderr or "")[-3000:], p.returncode


def main() -> None:
    ap = argparse.ArgumentParser(description="Chaos payload tamper/manifest verification tests.")
    ap.add_argument("--cipher", required=True)
    ap.add_argument("--roi_sidecar", required=True)
    ap.add_argument("--payload", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--framework_faster_path", required=True)
    ap.add_argument("--framework_base_path", default=None)
    ap.add_argument("--master_key", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = out_dir / "mutated_inputs"
    cases_dir.mkdir(exist_ok=True)

    base_records = read_jsonl(args.payload)
    manifest = load_manifest(args.manifest)

    cases: List[Tuple[str, str, str, bool]] = []  # name, payload, manifest, expected_reject

    payload_mutators: Dict[str, Callable[[List[dict]], List[dict]]] = {
        "ciphertext_byte_flip": lambda recs: mutate_first_roi(recs, lambda frame, roi: roi.__setitem__("cipher_b64", flip_b64_byte(str(roi.get("cipher_b64", ""))))),
        "poly_tag_flip_hmac_tag": lambda recs: mutate_first_roi(recs, lambda frame, roi: roi.__setitem__("tag", flip_hex_char(str(roi.get("tag", ""))))),
        "stream_nonce_flip": lambda recs: mutate_first_roi(recs, lambda frame, roi: roi.setdefault("crypto_meta", {}).__setitem__("stream_nonce_b64", flip_b64_byte(str(roi.get("crypto_meta", {}).get("stream_nonce_b64", ""))))),
        "tags_nonce_flip": lambda recs: mutate_first_roi(recs, lambda frame, roi: roi.setdefault("crypto_meta", {}).__setitem__("tags_b64", flip_b64_byte(str(roi.get("crypto_meta", {}).get("tags_b64", ""))))),
        "track_id_flip": lambda recs: mutate_first_roi(recs, lambda frame, roi: roi.__setitem__("track_id", int(roi.get("track_id", -1)) + 1)),
        "bbox_flip": lambda recs: mutate_first_roi(recs, lambda frame, roi: roi.__setitem__("bbox", [int(roi.get("bbox", [0,0,1,1])[0]) + 1, *list(roi.get("bbox", [0,0,1,1]))[1:]])),
        "mask_hash_flip": lambda recs: mutate_first_roi(recs, lambda frame, roi: roi.__setitem__("mask_hash", flip_hex_char(str(roi.get("mask_hash", ""))))),
        "frame_index_replay_metadata": lambda recs: mutate_first_roi(recs, lambda frame, roi: frame.__setitem__("frame_idx", int(frame.get("frame_idx", 0)) + 1)),
    }

    for name, mut in payload_mutators.items():
        pth, mth = make_payload_case(base_records, manifest, cases_dir, name, mut)
        cases.append((name, pth, mth, True))

    manifest_mutators: Dict[str, Callable[[dict], None]] = {
        "manifest_payload_hash_flip": lambda m: m.__setitem__("payload_sha256", flip_hex_char(str(m.get("payload_sha256", "")))),
        "manifest_video_hash_flip": lambda m: m.__setitem__("encrypted_video_sha256", flip_hex_char(str(m.get("encrypted_video_sha256", "")))),
        "manifest_roi_hash_flip": lambda m: m.__setitem__("roi_sidecar_sha256", flip_hex_char(str(m.get("roi_sidecar_sha256", "")))),
    }
    for name, mut in manifest_mutators.items():
        pth, mth = make_manifest_case(args.payload, manifest, cases_dir, name, mut)
        cases.append((name, pth, mth, True))

    results = []
    for name, payload_path, manifest_path, expected_reject in cases:
        out_video = str(out_dir / f"decrypt_{name}.mkv")
        ok, stdout_tail, stderr_tail, returncode = run_decrypt(args, payload_path, manifest_path, out_video)
        rejected = not ok
        results.append({
            "case": name,
            "payload": payload_path,
            "manifest": manifest_path,
            "expected_reject": expected_reject,
            "rejected": rejected,
            "passed": rejected == expected_reject,
            "returncode": returncode,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "note": "If this passes unexpectedly, the current chaos payload may not bind that field strongly enough or may skip reconstruction silently.",
        })

    report = {
        "focus": "Chaos payload tamper response: manifest SHA-256 binding + HMAC-tagged payload metadata/ciphertext.",
        "inputs": {"cipher": args.cipher, "roi_sidecar": args.roi_sidecar, "payload": args.payload, "manifest": args.manifest},
        "summary": {
            "cases": len(results),
            "passed": sum(1 for r in results if r["passed"]),
            "unexpected_successes": [r["case"] for r in results if r["expected_reject"] and not r["rejected"]],
            "all_passed": all(r["passed"] for r in results),
        },
        "results": results,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "report": str(out_dir / "report.json"), "all_passed": report["summary"]["all_passed"]}))


if __name__ == "__main__":
    main()
