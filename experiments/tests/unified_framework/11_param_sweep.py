#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List, Any

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from unified_utils import (
    derive_key_variant,
    encrypt_frames_with_sidecar,
    load_sidecar_records,
    read_media,
    write_media,
    resolve_sidecar_module,
)


def stable_import_module(path: str, canonical_name: str):
    resolved = Path(path).resolve()
    existing = sys.modules.get(canonical_name)
    if existing is not None and Path(getattr(existing, "__file__", "")).resolve() == resolved:
        return existing

    spec = importlib.util.spec_from_file_location(canonical_name, str(resolved))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {resolved}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[canonical_name] = mod
    spec.loader.exec_module(mod)
    return mod


def call_unpack_mask(unpack_fn, mask_pack: dict, mh: int, mw: int) -> np.ndarray:
    try:
        out = unpack_fn(mask_pack)
    except TypeError:
        out = unpack_fn(mask_pack, mh, mw)
    out = np.asarray(out, dtype=bool)
    if out.shape != (mh, mw):
        if out.shape[0] >= mh and out.shape[1] >= mw:
            out = out[:mh, :mw]
        else:
            out = cv2.resize(out.astype(np.uint8), (mw, mh), interpolation=cv2.INTER_NEAREST).astype(bool)
    return out


def build_combined_masks(recs: List[dict], ff, frame_shapes: List[tuple]) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for idx, rec in enumerate(recs):
        fi = int(rec.get("frame_idx", idx))
        if not frame_shapes:
            continue
        h, w = frame_shapes[min(fi, len(frame_shapes) - 1)]
        mask = np.zeros((h, w), dtype=bool)
        for r in rec.get("rois", []) or []:
            x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
            x1 = max(0, min(w, x1)); x2 = max(0, min(w, x2))
            y1 = max(0, min(h, y1)); y2 = max(0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            mh, mw = y2 - y1, x2 - x1
            loc = call_unpack_mask(ff.unpack_mask, r["mask_pack"], mh, mw)
            mask[y1:y2, x1:x2] |= loc
        out[fi] = mask
    return out


def avg(items: List[Dict[str, Any]], key: str) -> float:
    vals: List[float] = []
    for x in items:
        if key not in x:
            continue
        v = x[key]
        if v is None:
            continue
        try:
            vf = float(v)
        except Exception:
            continue
        if np.isfinite(vf):
            vals.append(vf)
    return float(mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser(
        description="Dedicated sweep for correlation analysis over different key sizes and chaotic dimensions/maps."
    )
    ap.add_argument("--plain", required=True)
    ap.add_argument("--roi_sidecar", required=True)
    ap.add_argument("--framework_faster_path", default="framework_faster.py")
    ap.add_argument("--framework_base_path", default=None, help="base framework path for sidecar unpacking when using proxy/payload framework")
    ap.add_argument("--video_preview_mode", default="chaos", choices=["chaos","mosaic","blur","black"], help="preview mode used when the framework path is a proxy/payload framework")
    ap.add_argument("--correlation_script", default="03_correlation_tests_roi.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--master_key", required=True)
    ap.add_argument("--maps", default="chen,cubic,skew_tent")
    ap.add_argument("--key_sizes", default="16,24,32", help="sizes in bytes for derived key strings")
    ap.add_argument("--scope", default="roi", choices=["whole", "roi", "background"])
    ap.add_argument("--sample_pairs", type=int, default=50000)
    ap.add_argument("--max_frames", type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ff_path = str((Path(args.framework_faster_path) if Path(args.framework_faster_path).is_absolute() else (THIS_DIR / args.framework_faster_path)).resolve())
    corr_path = str((Path(args.correlation_script) if Path(args.correlation_script).is_absolute() else (THIS_DIR / args.correlation_script)).resolve())

    ff = stable_import_module(ff_path, "framework_sweep")
    sidecar_ff = resolve_sidecar_module(ff, args.framework_base_path)
    corr_mod = stable_import_module(corr_path, "corr_sweep")

    recs = load_sidecar_records(args.roi_sidecar)
    frames, fps = read_media(args.plain)
    if args.max_frames and args.max_frames > 0:
        frames = frames[:args.max_frames]
        recs = [r for r in recs if int(r.get("frame_idx", 0)) < len(frames)]

    frame_shapes = [fr.shape[:2] for fr in frames]
    masks = build_combined_masks(recs, sidecar_ff, frame_shapes)

    rows = []
    maps = [m.strip() for m in args.maps.split(",") if m.strip()]
    key_sizes = [int(x.strip()) for x in args.key_sizes.split(",") if x.strip()]
    map_dims = {"chen": 3, "cubic": 1, "skew_tent": 1}

    generated_dir = out_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    for map_name in maps:
        for key_bytes in key_sizes:
            derived_key = derive_key_variant(args.master_key, key_bytes * 2)
            cipher_frames = encrypt_frames_with_sidecar(frames, recs, ff, derived_key, force_map_name=map_name, preview_mode=args.video_preview_mode)
            out_cipher = write_media(cipher_frames, str(generated_dir / f"{map_name}_k{key_bytes}.mkv"), fps=fps)

            plain_corrs = []
            cipher_corrs = []
            frame_notes = []

            for idx, (pfr, cfr) in enumerate(zip(frames, cipher_frames)):
                mask = masks.get(idx, np.zeros(pfr.shape[:2], dtype=bool))
                try:
                    p = corr_mod.correlation_suite(
                        pfr, mask=mask, scope=args.scope, sample=args.sample_pairs, seed=1000 + idx
                    )
                    c = corr_mod.correlation_suite(
                        cfr, mask=mask, scope=args.scope, sample=args.sample_pairs, seed=2000 + idx
                    )
                    if isinstance(p, dict):
                        plain_corrs.append(p)
                    if isinstance(c, dict):
                        cipher_corrs.append(c)
                except Exception as e:
                    frame_notes.append(f"frame {idx}: {e}")

            row = {
                "map_name": map_name,
                "chaotic_dimension": map_dims.get(map_name, None),
                "key_bytes": key_bytes,
                "scope": args.scope,
                "generated_cipher": out_cipher,
                "frames_used": len(cipher_corrs),
                "plain_h": avg(plain_corrs, "h_corr"),
                "plain_v": avg(plain_corrs, "v_corr"),
                "plain_d": avg(plain_corrs, "d_corr"),
                "cipher_h": avg(cipher_corrs, "h_corr"),
                "cipher_v": avg(cipher_corrs, "v_corr"),
                "cipher_d": avg(cipher_corrs, "d_corr"),
                "plain_gray_adj": avg(plain_corrs, "gray_adj_corr"),
                "cipher_gray_adj": avg(cipher_corrs, "gray_adj_corr"),
                "plain_r_adj": avg(plain_corrs, "r_adj_corr"),
                "plain_g_adj": avg(plain_corrs, "g_adj_corr"),
                "plain_b_adj": avg(plain_corrs, "b_adj_corr"),
                "cipher_r_adj": avg(cipher_corrs, "r_adj_corr"),
                "cipher_g_adj": avg(cipher_corrs, "g_adj_corr"),
                "cipher_b_adj": avg(cipher_corrs, "b_adj_corr"),
                "notes": frame_notes,
            }
            rows.append(row)

    json_path = out_dir / "report.json"
    csv_path = out_dir / "summary.csv"

    json_path.write_text(
        json.dumps(
            {
                "plain": args.plain,
                "roi_sidecar": args.roi_sidecar,
                "scope": args.scope,
                "maps": maps,
                "key_sizes": key_sizes,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if rows:
        fieldnames = sorted({k for row in rows for k in row.keys() if k != "notes"})
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            import csv
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in rows:
                out_row = {k: v for k, v in row.items() if k != "notes"}
                w.writerow(out_row)

    print(json.dumps({"ok": True, "report": str(json_path), "rows": len(rows)}))


if __name__ == "__main__":
    main()
