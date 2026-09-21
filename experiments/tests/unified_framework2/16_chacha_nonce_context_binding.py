#!/usr/bin/env python3
from __future__ import annotations

"""
16_chacha_nonce_context_binding.py

ChaCha20-Poly1305 nonce/context uniqueness and binding checks for the
payload-aware ROI framework.

Schema used by the current framework:
- rois.jsonl stores geometry/context only. crypto_meta is optional there.
- payload.jsonl stores encrypted recoverable ROI payloads and MUST contain crypto_meta.
- manifest.json optionally binds cipher + sidecar + payload by SHA-256.

This test therefore does NOT fail merely because rois.jsonl has no crypto_meta.
It requires crypto_meta only in payload records.
"""

import argparse
import base64
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def sha256_file(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Optional[str]) -> dict:
    if not path or not Path(path).is_file():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Optional[str]) -> List[dict]:
    if not path:
        return []
    p = Path(path)
    if not p.is_file():
        return []
    out: List[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def b64_len(text: str) -> int:
    try:
        return len(base64.b64decode(str(text).encode("ascii")))
    except Exception:
        return -1


def iter_sidecar_rois(records: List[dict]):
    for frame_rec in records:
        if frame_rec.get("type"):
            continue
        frame_idx = int(frame_rec.get("frame_idx", -1))
        frame_class = frame_rec.get("frame_class") or frame_rec.get("map_name") or "unknown"
        for idx, roi in enumerate(frame_rec.get("rois") or []):
            yield {
                "source": "sidecar",
                "frame_idx": frame_idx,
                "frame_class": frame_class,
                "roi_index": int(roi.get("roi_index", idx)),
                "cls": int(roi.get("cls", -1)),
                "track_id": int(roi.get("track_id", -1)),
                "bbox": roi.get("bbox"),
                "mask_hash": roi.get("mask_hash"),
                "policy_tag": roi.get("policy_tag"),
                "mask_pack_present": bool(roi.get("mask_pack")),
                "crypto_meta": roi.get("crypto_meta") or {},
            }


def iter_payload_rois(records: List[dict]):
    for frame_rec in records:
        if frame_rec.get("type") in {"meta", "tail_audio"}:
            continue
        frame_idx = int(frame_rec.get("frame_idx", -1))
        frame_class = str(frame_rec.get("frame_class", "unknown"))
        for idx, roi in enumerate(frame_rec.get("rois") or []):
            yield {
                "source": "payload",
                "frame_idx": frame_idx,
                "frame_class": frame_class,
                "roi_index": int(roi.get("roi_index", idx)),
                "cls": int(roi.get("cls", -1)),
                "track_id": int(roi.get("track_id", -1)),
                "bbox": roi.get("bbox"),
                "mask_hash": roi.get("mask_hash"),
                "policy_tag": roi.get("policy_tag"),
                "plain_len": roi.get("plain_len"),
                "comp_len": roi.get("comp_len"),
                "cipher_b64_present": bool(roi.get("cipher_b64")),
                "tag_present": bool(roi.get("tag")),
                "crypto_meta": roi.get("crypto_meta") or {},
            }


def context_key(item: dict) -> Tuple:
    return (
        int(item.get("frame_idx", -1)),
        int(item.get("roi_index", -1)),
        int(item.get("cls", -1)),
        int(item.get("track_id", -1)),
        tuple(item.get("bbox") or []),
        str(item.get("mask_hash") or ""),
        str(item.get("policy_tag") or ""),
    )


def analyze_sidecar(items: List[dict], require_crypto_meta: bool = False) -> dict:
    missing_context = []
    missing_mask_pack = []
    missing_meta = []
    malformed = []
    nonce_pairs = []

    for item in items:
        for field in ["frame_idx", "roi_index", "cls", "track_id", "bbox", "mask_hash", "policy_tag"]:
            if item.get(field) in (None, ""):
                missing_context.append({"field": field, "item": item})
        if not item.get("mask_pack_present"):
            missing_mask_pack.append(item)
        meta = item.get("crypto_meta") or {}
        a = str(meta.get("stream_nonce_b64", ""))
        b = str(meta.get("tags_b64", ""))
        if not a or not b:
            if require_crypto_meta:
                missing_meta.append(item)
            continue
        nonce_pairs.append((a, b))
        if b64_len(a) <= 0 or b64_len(b) <= 0:
            malformed.append(item)

    counts = Counter(nonce_pairs)
    duplicates = {"|".join(k): v for k, v in counts.items() if v > 1}
    return {
        "items": len(items),
        "crypto_meta_required": bool(require_crypto_meta),
        "crypto_meta_items": len(nonce_pairs),
        "missing_crypto_meta": len(missing_meta),
        "missing_context_fields": len(missing_context),
        "missing_mask_pack": len(missing_mask_pack),
        "malformed_nonce_fields": len(malformed),
        "unique_nonce_pairs": len(counts),
        "duplicate_nonce_pairs": duplicates,
        "checks": {
            "no_missing_crypto_meta": (len(missing_meta) == 0) if require_crypto_meta else True,
            "context_fields_present": len(missing_context) == 0,
            "mask_pack_present": len(missing_mask_pack) == 0,
            "all_nonce_fields_decode": len(malformed) == 0,
            "no_duplicate_nonce_pair": len(duplicates) == 0,
        },
    }


def analyze_payload(items: List[dict]) -> dict:
    missing_meta = []
    malformed_nonce = []
    wrong_nonce_len = []
    wrong_tag_len = []
    wrong_scheme = []
    missing_context = []
    missing_payload_fields = []
    nonce_pairs: List[Tuple[str, str]] = []
    nonce_to_contexts: Dict[Tuple[str, str], List[dict]] = defaultdict(list)

    for item in items:
        for field in ["frame_idx", "frame_class", "roi_index", "cls", "track_id", "bbox", "mask_hash", "policy_tag", "plain_len", "comp_len"]:
            if item.get(field) in (None, ""):
                missing_context.append({"field": field, "item": item})
        if not item.get("cipher_b64_present") or not item.get("tag_present"):
            missing_payload_fields.append(item)

        meta = item.get("crypto_meta") or {}
        a = str(meta.get("stream_nonce_b64", ""))
        b = str(meta.get("tags_b64", ""))
        if not a or not b:
            missing_meta.append(item)
            continue
        nonce_pairs.append((a, b))
        nonce_to_contexts[(a, b)].append({k: item.get(k) for k in ["frame_idx", "frame_class", "roi_index", "cls", "track_id", "bbox", "mask_hash", "policy_tag"]})

        la = b64_len(a)
        lb = b64_len(b)
        if la <= 0 or lb <= 0:
            malformed_nonce.append(item)
        if la != 12:
            wrong_nonce_len.append({"len": la, "item": item})
        if lb != 16:
            wrong_tag_len.append({"len": lb, "item": item})
        if str(meta.get("scheme", "")) != "chacha20poly1305_v1":
            wrong_scheme.append({"scheme": meta.get("scheme"), "item": item})

    counts = Counter(nonce_pairs)
    duplicates = {"|".join(k): v for k, v in counts.items() if v > 1}
    duplicate_contexts = {"|".join(k): v for k, v in nonce_to_contexts.items() if len(v) > 1}
    raw_nonce_counts = Counter([p[0] for p in nonce_pairs])
    raw_nonce_duplicates = {k: v for k, v in raw_nonce_counts.items() if v > 1}

    return {
        "items": len(items),
        "crypto_meta_required": True,
        "crypto_meta_items": len(nonce_pairs),
        "missing_crypto_meta": len(missing_meta),
        "malformed_nonce_fields": len(malformed_nonce),
        "wrong_nonce_length_not_12_bytes": len(wrong_nonce_len),
        "wrong_poly1305_tag_length_not_16_bytes": len(wrong_tag_len),
        "wrong_scheme": len(wrong_scheme),
        "missing_context_fields": len(missing_context),
        "missing_payload_fields": len(missing_payload_fields),
        "unique_nonce_pairs": len(counts),
        "duplicate_nonce_pairs": duplicates,
        "duplicate_nonce_contexts": duplicate_contexts,
        "unique_raw_nonces": len(raw_nonce_counts),
        "duplicate_raw_nonces": raw_nonce_duplicates,
        "checks": {
            "payload_present": len(items) > 0,
            "no_missing_crypto_meta": len(missing_meta) == 0,
            "all_nonce_fields_decode": len(malformed_nonce) == 0,
            "all_nonces_are_12_bytes": len(wrong_nonce_len) == 0,
            "all_poly1305_tags_are_16_bytes": len(wrong_tag_len) == 0,
            "scheme_is_chacha20poly1305_v1": len(wrong_scheme) == 0,
            "no_duplicate_nonce_pair": len(duplicates) == 0,
            "context_fields_present": len(missing_context) == 0,
            "payload_fields_present": len(missing_payload_fields) == 0,
        },
    }


def compare_sidecar_payload(sidecar_items: List[dict], payload_items: List[dict]) -> dict:
    side = Counter(context_key(i) for i in sidecar_items)
    pay = Counter(context_key(i) for i in payload_items)
    missing_in_payload = list((side - pay).elements())
    missing_in_sidecar = list((pay - side).elements())
    return {
        "sidecar_items": len(sidecar_items),
        "payload_items": len(payload_items),
        "matching_context_multiset": len(missing_in_payload) == 0 and len(missing_in_sidecar) == 0,
        "missing_in_payload_count": len(missing_in_payload),
        "missing_in_sidecar_count": len(missing_in_sidecar),
        "missing_in_payload_examples": [str(x) for x in missing_in_payload[:10]],
        "missing_in_sidecar_examples": [str(x) for x in missing_in_sidecar[:10]],
    }


def manifest_binding(manifest_path: Optional[str], cipher: Optional[str], roi_sidecar: str, payload: Optional[str]) -> dict:
    if not manifest_path:
        return {"skipped": "no manifest supplied", "checks": {"manifest_hashes_match": True}}
    manifest = read_json(manifest_path)
    checks = []
    details = {}
    mapping = [
        ("encrypted_video_sha256", cipher, "cipher"),
        ("roi_sidecar_sha256", roi_sidecar, "roi_sidecar"),
        ("payload_sha256", payload, "payload"),
    ]
    for key, path, label in mapping:
        expected = str(manifest.get(key, ""))
        actual = sha256_file(path)
        ok = bool(expected) and actual == expected
        details[label] = {"manifest_field": key, "expected": expected, "actual": actual, "matches": ok}
        checks.append(ok)
    return {"manifest": manifest_path, "details": details, "checks": {"manifest_hashes_match": all(checks)}}


def compare_repeat(a_items: List[dict], b_items: List[dict]) -> dict:
    a = {context_key(i): (i.get("crypto_meta") or {}) for i in a_items}
    b = {context_key(i): (i.get("crypto_meta") or {}) for i in b_items}
    common = set(a) & set(b)
    same = sum(1 for k in common if a[k] == b[k])
    diff = len(common) - same
    return {"common_contexts": len(common), "same_crypto_meta": same, "different_crypto_meta": diff, "deterministic_for_common_contexts": diff == 0 if common else None}


def main() -> None:
    ap = argparse.ArgumentParser(description="ChaCha20-Poly1305 nonce/context uniqueness checks for sidecar and payload records.")
    ap.add_argument("--roi_sidecar", required=True)
    ap.add_argument("--payload", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--cipher", default=None)
    ap.add_argument("--roi_sidecar_repeat", default=None)
    ap.add_argument("--payload_repeat", default=None)
    ap.add_argument("--strict_sidecar_crypto_meta", action="store_true", help="Require crypto_meta in rois.jsonl too. Default is false because current schema stores crypto_meta in payload.jsonl.")
    ap.add_argument("--skip_cross_file_context_check", action="store_true", help="Do not require sidecar/payload context multisets to match. Useful for checking only nonce format/uniqueness.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sidecar_items = list(iter_sidecar_rois(read_jsonl(args.roi_sidecar)))
    payload_items = list(iter_payload_rois(read_jsonl(args.payload)))

    sections = {
        "sidecar_geometry_context": analyze_sidecar(sidecar_items, require_crypto_meta=args.strict_sidecar_crypto_meta),
        "payload_aead_context": analyze_payload(payload_items),
    }
    if args.skip_cross_file_context_check:
        sections["sidecar_payload_cross_file_context"] = {"skipped": True, "checks": {"matching_context_multiset": True}}
    else:
        cross = compare_sidecar_payload(sidecar_items, payload_items)
        cross["checks"] = {"matching_context_multiset": bool(cross["matching_context_multiset"])}
        sections["sidecar_payload_cross_file_context"] = cross
    sections["manifest_hash_binding"] = manifest_binding(args.manifest, args.cipher, args.roi_sidecar, args.payload)

    repeat = None
    if args.payload_repeat or args.roi_sidecar_repeat:
        repeat_items = list(iter_payload_rois(read_jsonl(args.payload_repeat))) if args.payload_repeat else []
        repeat = compare_repeat(payload_items, repeat_items)

    all_checks = []
    for sec in sections.values():
        checks = sec.get("checks")
        if isinstance(checks, dict):
            all_checks.extend(bool(v) for v in checks.values())

    report = {
        "focus": "ChaCha20-Poly1305 metadata nonce/context uniqueness and binding checks.",
        "schema_note": "crypto_meta is required in payload.jsonl; rois.jsonl is geometry/context and does not need crypto_meta unless --strict_sidecar_crypto_meta is used.",
        "inputs": {"roi_sidecar": args.roi_sidecar, "payload": args.payload, "manifest": args.manifest, "cipher": args.cipher, "roi_sidecar_repeat": args.roi_sidecar_repeat, "payload_repeat": args.payload_repeat},
        "sections": sections,
        "repeat_reproducibility": repeat,
        "summary": {"all_passed": all(all_checks) if all_checks else False},
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "report": str(out_dir / "report.json"), "all_passed": report["summary"]["all_passed"]}))


if __name__ == "__main__":
    main()
