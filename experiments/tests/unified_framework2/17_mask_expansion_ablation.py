#!/usr/bin/env python3
from __future__ import annotations

"""
17_chacha_mask_expansion_ablation.py

ChaCha-first privacy/utility ablation for mask expansion / silhouette leakage.
It rewrites an existing ROI sidecar with padding and mask dilation, reuses those
ROIs for the ChaCha/proxy payload framework, then runs detector leakage + quality
+ storage summary per configuration.
"""

import argparse
import base64
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent


def import_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, str(Path(path).resolve()))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def call_unpack(unpack_fn, pack: dict, h: int, w: int) -> np.ndarray:
    try:
        out = unpack_fn(pack)
    except TypeError:
        out = unpack_fn(pack, h, w)
    out = np.asarray(out, dtype=bool)
    if out.shape != (h, w):
        out = cv2.resize(out.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return out


def pack_mask(mask: np.ndarray) -> dict:
    mask = np.asarray(mask, dtype=bool)
    packed = np.packbits(mask.reshape(-1).astype(np.uint8))
    return {"h": int(mask.shape[0]), "w": int(mask.shape[1]), "b64": base64.b64encode(packed.tobytes()).decode("ascii")}


def video_shape(path: str) -> Tuple[int, int, float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    cap.release()
    return h, w, fps


def mask_sha(mask: np.ndarray) -> str:
    return hashlib.sha256(np.packbits(mask.astype(np.uint8).reshape(-1)).tobytes()).hexdigest()


def expand_sidecar(src: str, dst: str, framework_base_path: str, plain_video: str, padding: int, dilation: int) -> dict:
    ff = import_module(framework_base_path, "framework_mask_ablation_base")
    if not hasattr(ff, "unpack_mask"):
        raise RuntimeError("framework_base_path must expose unpack_mask")
    H, W, _ = video_shape(plain_video)
    kernel = np.ones((max(1, int(dilation)), max(1, int(dilation))), np.uint8) if int(dilation) > 0 else None
    frames = objects = 0
    roi_pixels = 0
    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            out_rois = []
            for roi in rec.get("rois") or []:
                bbox = roi.get("bbox")
                pack = roi.get("mask_pack")
                if not bbox or pack is None:
                    continue
                x1, y1, x2, y2 = [int(v) for v in bbox]
                x1 = max(0, min(W, x1)); x2 = max(0, min(W, x2))
                y1 = max(0, min(H, y1)); y2 = max(0, min(H, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                local = call_unpack(ff.unpack_mask, pack, y2 - y1, x2 - x1)
                full = np.zeros((H, W), dtype=bool)
                full[y1:y2, x1:x2] = local
                if kernel is not None:
                    full = cv2.dilate(full.astype(np.uint8), kernel, iterations=1).astype(bool)
                if padding > 0:
                    contours, _ = cv2.findContours(full.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    padded = np.zeros_like(full)
                    for c in contours:
                        bx, by, bw, bh = cv2.boundingRect(c)
                        px1 = max(0, bx - padding); py1 = max(0, by - padding)
                        px2 = min(W, bx + bw + padding); py2 = min(H, by + bh + padding)
                        padded[py1:py2, px1:px2] |= full[py1:py2, px1:px2]
                        # Also include a filled contour expansion rectangle to suppress boundary/context leakage.
                        padded[py1:py2, px1:px2] = True
                    full = padded
                ys, xs = np.where(full)
                if ys.size == 0:
                    continue
                nx1, nx2 = int(xs.min()), int(xs.max()) + 1
                ny1, ny2 = int(ys.min()), int(ys.max()) + 1
                new_local = full[ny1:ny2, nx1:nx2]
                new_roi = dict(roi)
                new_roi["bbox"] = [nx1, ny1, nx2, ny2]
                new_roi["mask_pack"] = pack_mask(new_local)
                new_roi["mask_hash"] = mask_sha(new_local)
                out_rois.append(new_roi)
                objects += 1
                roi_pixels += int(new_local.sum())
            rec["rois"] = out_rois
            fout.write(json.dumps(rec, separators=(",", ":")) + "\n")
            frames += 1
    return {"frames": frames, "objects": objects, "roi_pixels": roi_pixels, "mean_roi_fraction": roi_pixels / float(max(1, frames * H * W))}


def run_cmd(cmd: List[str], cwd: Path = THIS_DIR) -> dict:
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace")
    return {"returncode": p.returncode, "stdout_tail": (p.stdout or "")[-2000:], "stderr_tail": (p.stderr or "")[-2000:], "cmd": cmd}


def parse_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Mask expansion / silhouette leakage ablation for ChaCha ROI protection.")
    ap.add_argument("--plain", required=True)
    ap.add_argument("--roi_sidecar", required=True)
    ap.add_argument("--framework_faster_path", required=True)
    ap.add_argument("--framework_base_path", default=None)
    ap.add_argument("--master_key", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--paddings", default="0,4,8,16,32")
    ap.add_argument("--dilations", default="0,1,3,5,9")
    ap.add_argument("--preview_modes", default="chacha")
    ap.add_argument("--max_frames", type=int, default=50)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="optional limit on configurations for quick CI runs")
    ap.add_argument("--skip_detector", action="store_true")
    ap.add_argument("--skip_quality", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    base_path = args.framework_base_path or args.framework_faster_path
    paddings = [int(x) for x in args.paddings.split(',') if x.strip()]
    dilations = [int(x) for x in args.dilations.split(',') if x.strip()]
    preview_modes = [x.strip() for x in args.preview_modes.split(',') if x.strip()]

    rows = []
    cfgs = [(p, d, m) for p in paddings for d in dilations for m in preview_modes]
    if args.limit and args.limit > 0:
        cfgs = cfgs[:args.limit]

    for pad, dil, mode in cfgs:
        label = f"pad{pad}_dil{dil}_{mode}"
        cfg_dir = out_dir / label; cfg_dir.mkdir(exist_ok=True)
        sidecar_out = cfg_dir / "rois_expanded.jsonl"
        sidecar_stats = expand_sidecar(args.roi_sidecar, str(sidecar_out), base_path, args.plain, pad, dil)
        cipher_out = cfg_dir / "cipher.mkv"
        payload_out = cfg_dir / "payload.jsonl"
        manifest_out = cfg_dir / "payload.jsonl.manifest.json"
        cmd = [sys.executable, str(Path(args.framework_faster_path).resolve()), "--mode", "encrypt", "--in", args.plain, "--out", str(cipher_out), "--key", args.master_key, "--roi_sidecar", str(sidecar_out), "--reuse_rois"]
        if args.framework_base_path:
            # Payload/proxy framework: preview mode is configurable and original ROI pixels are stored in ChaCha payload.
            cmd.extend(["--base_framework", args.framework_base_path, "--payload", str(payload_out), "--manifest", str(manifest_out), "--video_preview_mode", mode, "--audio_preview_mode", "chacha"])
        else:
            # Base ChaCha framework has no public preview-mode CLI; it performs AEAD pixel substitution.
            cmd.extend(["--keystream_dump", str(cfg_dir / "keystream.bin")])
        t0 = time.perf_counter(); enc_res = run_cmd(cmd); enc_s = time.perf_counter() - t0
        row = {"label": label, "padding": pad, "dilation": dil, "preview_mode": mode, "encrypt_seconds": enc_s, "encrypt_returncode": enc_res["returncode"], **sidecar_stats}
        row["cipher_size_bytes"] = cipher_out.stat().st_size if cipher_out.exists() else None
        row["payload_size_bytes"] = payload_out.stat().st_size if payload_out.exists() else 0
        row["manifest_size_bytes"] = manifest_out.stat().st_size if manifest_out.exists() else 0
        row["total_package_size_bytes"] = sum(x for x in [row["cipher_size_bytes"] or 0, row["payload_size_bytes"] or 0, row["manifest_size_bytes"] or 0, sidecar_out.stat().st_size if sidecar_out.exists() else 0])
        row["encrypt_stdout_tail"] = enc_res["stdout_tail"]
        row["encrypt_stderr_tail"] = enc_res["stderr_tail"]

        if enc_res["returncode"] == 0 and cipher_out.exists() and not args.skip_detector:
            det_dir = cfg_dir / "detector"
            det_cmd = [sys.executable, "13_detector_privacy_leakage.py", "--plain", args.plain, "--cipher", str(cipher_out), "--roi_sidecar", args.roi_sidecar, "--framework_faster_path", base_path, "--out", str(det_dir), "--max_frames", str(args.max_frames), "--stride", str(args.stride)]
            det_res = run_cmd(det_cmd)
            det_rep = parse_json(det_dir / "report.json")
            summ = det_rep.get("privacy_leakage_summary", {})
            row.update({
                "detector_returncode": det_res["returncode"],
                "cipher_person_recall": summ.get("cipher_person_recall"),
                "cipher_vehicle_recall": summ.get("cipher_vehicle_recall"),
                "cipher_mean_matched_conf": summ.get("cipher_mean_matched_conf"),
                "cipher_recall_relative_to_plain": summ.get("cipher_recall_relative_to_plain"),
            })
        if enc_res["returncode"] == 0 and cipher_out.exists() and not args.skip_quality:
            q_dir = cfg_dir / "quality"
            q_cmd = [sys.executable, "01_quality_metrics_roi.py", "--plain", args.plain, "--test", str(cipher_out), "--roi_sidecar", args.roi_sidecar, "--framework_faster_path", base_path, "--out", str(q_dir), "--max_frames", str(args.max_frames), "--stride", str(args.stride), "--mode", "color", "--scope", "all"]
            q_res = run_cmd(q_cmd)
            q_rep = parse_json(q_dir / "report.json")
            scopes = q_rep.get("pixel_resemblance_disparity", {}).get("scopes", {})
            bg = scopes.get("background") or {}
            roi_scope = scopes.get("roi") or {}
            row.update({
                "quality_returncode": q_res["returncode"],
                "public_background_ssim": (bg.get("ssim") or {}).get("mean"),
                "public_background_psnr": (bg.get("psnr") or {}).get("mean"),
                "roi_psnr": (roi_scope.get("psnr") or {}).get("mean"),
            })
        rows.append(row)

    fieldnames = sorted({k for r in rows for k in r.keys() if not k.endswith("tail")})
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})
    report = {"focus": "ChaCha mask expansion / silhouette leakage ablation", "configs_requested": len(cfgs), "rows": rows}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "report": str(out_dir / "report.json"), "rows": len(rows)}))


if __name__ == "__main__":
    main()
