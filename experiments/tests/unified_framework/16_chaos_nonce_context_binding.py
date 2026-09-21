#!/usr/bin/env python3
from __future__ import annotations

"""
16_chaos_nonce_context_binding.py

Checks chaos crypto metadata uniqueness and context binding in ROI sidecars and
payload records. This is the chaos counterpart to a later AEAD nonce test.
"""

import argparse
import base64
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def read_jsonl(path: str) -> List[dict]:
    if not path:
        return []
    p = Path(path)
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


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
        for idx, roi in enumerate(frame_rec.get("rois") or []):
            yield {
                "source": "sidecar",
                "frame_idx": frame_idx,
                "roi_index": int(roi.get("roi_index", idx)),
                "cls": int(roi.get("cls", -1)),
                "track_id": int(roi.get("track_id", -1)),
                "bbox": roi.get("bbox"),
                "mask_hash": roi.get("mask_hash"),
                "policy_tag": roi.get("policy_tag"),
                "crypto_meta": roi.get("crypto_meta") or {},
            }


def iter_payload_rois(records: List[dict]):
    for frame_rec in records:
        if frame_rec.get("type") in {"meta", "tail_audio"}:
            continue
        frame_idx = int(frame_rec.get("frame_idx", -1))
        for idx, roi in enumerate(frame_rec.get("rois") or []):
            yield {
                "source": "payload",
                "frame_idx": frame_idx,
                "roi_index": int(roi.get("roi_index", idx)),
                "cls": int(roi.get("cls", -1)),
                "track_id": int(roi.get("track_id", -1)),
                "bbox": roi.get("bbox"),
                "mask_hash": roi.get("mask_hash"),
                "policy_tag": roi.get("policy_tag"),
                "crypto_meta": roi.get("crypto_meta") or {},
            }


def analyze_items(items: List[dict], require_crypto_meta: bool = True) -> dict:
    nonce_pairs: List[Tuple[str, str]] = []
    context_pairs: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    missing_meta = []
    malformed = []
    missing_context_fields = []

    context_fields = [
        "frame_idx",
        "roi_index",
        "cls",
        "track_id",
        "bbox",
        "mask_hash",
        "policy_tag",
    ]

    # First check context fields for BOTH sidecar and payload.
    # This must happen even when crypto_meta is absent.
    for item in items:
        for field in context_fields:
            if item.get(field) in (None, ""):
                missing_context_fields.append({"field": field, "item": item})

        meta = item.get("crypto_meta") or {}
        a = str(meta.get("stream_nonce_b64", ""))
        b = str(meta.get("tags_b64", ""))

        # For payload records, crypto_meta is required.
        # For sidecar records, crypto_meta is optional by design.
        if not a or not b:
            if require_crypto_meta:
                missing_meta.append(item)
            continue

        pair = (a, b)
        nonce_pairs.append(pair)
        context_pairs[pair].append({
            k: item.get(k)
            for k in ["source", "frame_idx", "roi_index", "cls", "track_id", "bbox", "mask_hash", "policy_tag"]
        })

        if b64_len(a) <= 0 or b64_len(b) <= 0:
            malformed.append(item)

    counts = Counter(nonce_pairs)
    duplicates = {"|".join(k): v for k, v in counts.items() if v > 1}
    duplicate_contexts = {"|".join(k): v for k, v in context_pairs.items() if len(v) > 1}

    return {
        "items": len(items),
        "crypto_meta_required": bool(require_crypto_meta),
        "crypto_meta_items": len(nonce_pairs),
        "missing_crypto_meta": len(missing_meta),
        "malformed_nonce_fields": len(malformed),
        "missing_context_fields": len(missing_context_fields),
        "unique_nonce_pairs": len(counts),
        "duplicate_nonce_pairs": duplicates,
        "duplicate_nonce_contexts": duplicate_contexts,
        "checks": {
            # If crypto_meta is not required, missing crypto_meta must not fail the test.
            "no_missing_crypto_meta": (len(missing_meta) == 0) if require_crypto_meta else True,
            "all_nonce_fields_decode": len(malformed) == 0,
            "no_duplicate_nonce_pair": len(duplicates) == 0,
            "context_fields_present": len(missing_context_fields) == 0,
        },
    }


def compare_repeat(a_items: List[dict], b_items: List[dict]) -> dict:
    def key(item: dict):
        return (item.get("source"), item.get("frame_idx"), item.get("roi_index"), item.get("cls"), item.get("track_id"), tuple(item.get("bbox") or []), item.get("mask_hash"), item.get("policy_tag"))
    a = {key(i): (i.get("crypto_meta") or {}) for i in a_items}
    b = {key(i): (i.get("crypto_meta") or {}) for i in b_items}
    common = set(a) & set(b)
    same = sum(1 for k in common if a[k] == b[k])
    diff = len(common) - same
    return {"common_contexts": len(common), "same_crypto_meta": same, "different_crypto_meta": diff, "deterministic_for_common_contexts": diff == 0 if common else None}


def main() -> None:
    ap = argparse.ArgumentParser(description="Chaos nonce/context uniqueness checks for sidecar and payload records.")
    ap.add_argument("--roi_sidecar", required=True)
    ap.add_argument("--payload", default=None)
    ap.add_argument("--roi_sidecar_repeat", default=None, help="optional second sidecar from repeat encryption for reproducibility check")
    ap.add_argument("--payload_repeat", default=None, help="optional second payload from repeat encryption for reproducibility check")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sidecar_items = list(iter_sidecar_rois(read_jsonl(args.roi_sidecar)))
    payload_items = list(iter_payload_rois(read_jsonl(args.payload))) if args.payload else []

    sections = {
        # Sidecar stores geometry/mask context. It does NOT need crypto_meta.
        "sidecar": analyze_items(sidecar_items, require_crypto_meta=False),

        # Payload stores encrypted ROI data. It MUST have crypto_meta.
        "payload": analyze_items(payload_items, require_crypto_meta=True)
        if payload_items else {"items": 0, "skipped": "no payload supplied", "checks": {"payload_present": False}},

        # Combined is diagnostic only. Do not require crypto_meta for sidecar records.
        "combined": analyze_items(sidecar_items + payload_items, require_crypto_meta=False),
    }

    repeat = None
    if args.roi_sidecar_repeat or args.payload_repeat:
        sidecar_repeat_items = list(iter_sidecar_rois(read_jsonl(args.roi_sidecar_repeat))) if args.roi_sidecar_repeat else []
        payload_repeat_items = list(iter_payload_rois(read_jsonl(args.payload_repeat))) if args.payload_repeat else []
        repeat = compare_repeat(sidecar_items + payload_items, sidecar_repeat_items + payload_repeat_items)

    sidecar_checks = sections.get("sidecar", {}).get("checks", {})
    payload_checks = sections.get("payload", {}).get("checks", {})
    combined_checks = sections.get("combined", {}).get("checks", {})

    all_checks = [
        # Sidecar must have valid geometry/context, but crypto_meta is optional.
        bool(sidecar_checks.get("context_fields_present", False)),

        # Payload must have crypto metadata and valid nonce/context fields.
        bool(payload_checks.get("no_missing_crypto_meta", False)),
        bool(payload_checks.get("all_nonce_fields_decode", False)),
        bool(payload_checks.get("no_duplicate_nonce_pair", False)),
        bool(payload_checks.get("context_fields_present", False)),

        # Combined duplicate nonce check is useful, but missing sidecar crypto_meta should not fail.
        bool(combined_checks.get("no_duplicate_nonce_pair", False)),
    ]

    report = {
        "focus": "Chaos metadata nonce/context uniqueness and binding checks.",
        "inputs": {"roi_sidecar": args.roi_sidecar, "payload": args.payload, "roi_sidecar_repeat": args.roi_sidecar_repeat, "payload_repeat": args.payload_repeat},
        "sections": sections,
        "repeat_reproducibility": repeat,
        "summary": {"all_passed": all(all_checks) if all_checks else False},
    }
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "report": str(out_dir / "report.json"), "all_passed": report["summary"]["all_passed"]}))


if __name__ == "__main__":
    main()
