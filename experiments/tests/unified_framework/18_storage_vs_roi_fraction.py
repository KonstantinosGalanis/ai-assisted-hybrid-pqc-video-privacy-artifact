#!/usr/bin/env python3
from __future__ import annotations

"""
18_storage_vs_roi_fraction.py

Storage overhead vs ROI density test. It groups one or more dataset folders by
mean ROI fraction and reports plain video, chaos ROI package, sidecar, payload,
manifest, and optional baseline sizes.
"""

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
import sys
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
from unified_utils import discover_assets


def import_framework(path: str):
    spec = importlib.util.spec_from_file_location("framework_storage_roi", str(Path(path).resolve()))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import framework from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def file_size(path: Optional[str]) -> int:
    return int(os.path.getsize(path)) if path and os.path.exists(path) else 0


def video_shape(path: str) -> Tuple[int, int, float, int]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0, 0, 0.0, 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return w, h, fps, frames


def call_unpack(unpack_fn, pack: dict, h: int, w: int) -> np.ndarray:
    try:
        out = unpack_fn(pack)
    except TypeError:
        out = unpack_fn(pack, h, w)
    out = np.asarray(out, dtype=bool)
    if out.shape != (h, w):
        out = cv2.resize(out.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return out


def roi_fraction(roi_sidecar: str, plain: str, framework_path: str, method: str) -> Dict[str, Any]:
    w, h, fps, frame_count = video_shape(plain)
    if w <= 0 or h <= 0:
        return {"error": "could not read video dimensions"}
    ff = import_framework(framework_path) if method == "mask" else None
    total_px = 0
    frames = 0
    frames_with_roi = 0
    objects = 0
    with open(roi_sidecar, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            roi_px = 0
            for r in rec.get("rois") or []:
                bbox = r.get("bbox")
                if not bbox:
                    continue
                x1, y1, x2, y2 = [int(v) for v in bbox]
                x1 = max(0, min(w, x1)); x2 = max(0, min(w, x2))
                y1 = max(0, min(h, y1)); y2 = max(0, min(h, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                if method == "mask" and r.get("mask_pack") is not None and hasattr(ff, "unpack_mask"):
                    m = call_unpack(ff.unpack_mask, r["mask_pack"], y2-y1, x2-x1)
                    roi_px += int(m.sum())
                else:
                    roi_px += int((x2-x1) * (y2-y1))
                objects += 1
            frames += 1
            if roi_px > 0:
                frames_with_roi += 1
            total_px += roi_px
    denom = float(max(1, frames * w * h))
    return {
        "width": w, "height": h, "fps": fps, "video_frame_count": frame_count,
        "sidecar_frames": frames, "frames_with_roi": frames_with_roi, "objects": objects,
        "roi_pixels": total_px, "mean_roi_fraction": float(total_px / denom),
        "estimated_raw_roi_payload_bytes_bgr": int(total_px * 3),
    }


def bucket(frac: float) -> str:
    pct = frac * 100.0
    if pct <= 5: return "0-5%"
    if pct <= 15: return "5-15%"
    if pct <= 30: return "15-30%"
    if pct <= 50: return "30-50%"
    return ">50%"


def find_optional_baselines(dataset: Path) -> Dict[str, Optional[str]]:
    media = [p for p in dataset.rglob("*") if p.is_file() and p.suffix.lower() in {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}]
    def pick(keys, excludes=()):
        hits = []
        for p in media:
            rel = str(p.relative_to(dataset)).lower()
            if any(k in rel for k in keys) and not any(e in rel for e in excludes):
                hits.append(p)
        return str(sorted(hits)[0]) if hits else None
    return {
        "frame_domain_full_pixel_encryption": pick(["full", "frame_domain", "fullframe"], ["plain", "original", "source"]),
        "compressed_bitstream_aead_baseline": pick(["bitstream", "container_aead", "compressed_aead", "aead_baseline"]),
        "chacha_roi": pick(["chacha", "poly1305", "aead_roi"]),
    }


def summarize_bucket(rows: List[dict]) -> dict:
    def avg(key):
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None
    return {
        "count": len(rows),
        "mean_roi_fraction": avg("mean_roi_fraction"),
        "mean_total_package_size_bytes": avg("total_protected_package_size_bytes"),
        "mean_overhead_ratio_vs_plain": avg("overhead_ratio_vs_plain"),
        "datasets": [r["dataset"] for r in rows],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Storage overhead vs ROI density grouping.")
    ap.add_argument("--datasets", nargs="+", required=True, help="one or more dataset folders")
    ap.add_argument("--framework_faster_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--roi_area_method", choices=["bbox", "mask"], default="mask")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    for ds in args.datasets:
        dataset = Path(ds).resolve()
        assets = discover_assets(str(dataset))
        optional = find_optional_baselines(dataset)
        if not (assets.get("plain") and assets.get("cipher") and assets.get("roi_sidecar")):
            rows.append({"dataset": str(dataset), "status": "skipped", "reason": "need plain, cipher, roi_sidecar", "assets": assets})
            continue
        rf = roi_fraction(assets["roi_sidecar"], assets["plain"], args.framework_faster_path, args.roi_area_method)
        frac = float(rf.get("mean_roi_fraction", 0.0))
        plain_size = file_size(assets.get("plain"))
        cipher_size = file_size(assets.get("cipher"))
        sidecar_size = file_size(assets.get("roi_sidecar"))
        payload_size = file_size(assets.get("payload"))
        manifest_size = file_size(assets.get("manifest"))
        total = cipher_size + sidecar_size + payload_size + manifest_size
        row = {
            "dataset": str(dataset), "status": "ok", "bucket": bucket(frac),
            "plain_video_size_bytes": plain_size,
            "chaos_roi_video_size_bytes": cipher_size,
            "sidecar_size_bytes": sidecar_size,
            "encrypted_payload_size_bytes": payload_size,
            "manifest_size_bytes": manifest_size,
            "total_protected_package_size_bytes": total,
            "overhead_ratio_vs_plain": (total / plain_size) if plain_size else None,
            **rf,
            "frame_domain_full_pixel_encryption_size_bytes": file_size(optional.get("frame_domain_full_pixel_encryption")),
            "compressed_bitstream_aead_baseline_size_bytes": file_size(optional.get("compressed_bitstream_aead_baseline")),
            "chacha_roi_size_bytes": file_size(optional.get("chacha_roi")),
            "assets": assets,
            "optional_baselines": optional,
        }
        rows.append(row)

    groups: Dict[str, List[dict]] = {}
    for r in rows:
        if r.get("status") == "ok":
            groups.setdefault(r["bucket"], []).append(r)
    group_summary = {k: summarize_bucket(v) for k, v in sorted(groups.items())}

    fieldnames = sorted({k for r in rows for k in r.keys() if k not in {"assets", "optional_baselines"}})
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})
    report = {"focus": "Storage overhead vs ROI fraction", "roi_area_method": args.roi_area_method, "groups": group_summary, "rows": rows}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "report": str(out_dir / "report.json"), "rows": len(rows)}))


if __name__ == "__main__":
    main()
