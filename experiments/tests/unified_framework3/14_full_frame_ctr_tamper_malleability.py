#!/usr/bin/env python3
from __future__ import annotations

"""
14_full_frame_ctr_tamper_malleability.py

AES-CTR full-frame tamper test.

This framework uses AES-256-CTR without an authentication tag. Therefore the
correct expectation is NOT rejection. The test demonstrates malleability:
modified ciphertext decrypts successfully, but the recovered plaintext is
corrupted.
"""

import argparse
import json
from fractions import Fraction
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from unified_utils import ensure_dir, framework_cli_decrypt, iter_frames, mae, mse, psnr, sha256_file, snr, video_info, write_csv, write_json

try:
    import av
except Exception:
    av = None

try:
    from skimage.metrics import structural_similarity as sk_ssim
except Exception:
    sk_ssim = None


def _safe_rate(fps: float) -> Fraction:
    return Fraction(float(fps) if fps and fps > 0 else 30.0).limit_denominator(100000)


def video_props(path: str) -> Tuple[int, int, float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    cap.release()
    return w, h, fps


def open_writer(path: str, w: int, h: int, fps: float):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if av is not None:
        c = av.open(path, mode="w", format="matroska")
        st = c.add_stream("libx264rgb", rate=_safe_rate(fps))
        st.width = w
        st.height = h
        st.pix_fmt = "bgr24"
        st.options = {"preset": "slower", "qp": "0", "x264-params": "keyint=10000:min-keyint=10000:scenecut=0:bframes=0"}
        return "pyav", c, st
    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    vw = cv2.VideoWriter(path, fourcc, fps if fps > 0 else 25.0, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"Could not open writer for {path}")
    return "opencv", vw, None


def write_frame(kind, writer, stream, frame):
    if kind == "pyav":
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="bgr24")
        for pkt in stream.encode(vf):
            writer.mux(pkt)
    else:
        writer.write(frame)


def close_writer(kind, writer, stream):
    if kind == "pyav":
        for pkt in stream.encode():
            writer.mux(pkt)
        writer.close()
    else:
        writer.release()


def ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
    a = a[:h, :w]; b = b[:h, :w]
    if sk_ssim is None:
        m = mse(a, b)
        return float(1.0 / (1.0 + m / (255.0 * 255.0)))
    return float(sk_ssim(a, b, data_range=255, channel_axis=2 if a.ndim == 3 else None))


def compare_videos(a: str, b: str, max_frames: int, stride: int) -> Dict[str, Any]:
    rows = []
    for (proc_i, orig_i, fa), (_, _, fb) in zip(iter_frames(a, max_frames, stride), iter_frames(b, max_frames, stride)):
        rows.append({"proc_frame_idx": proc_i, "orig_frame_idx": orig_i, "mse": mse(fa, fb), "mae": mae(fa, fb), "psnr_db": psnr(fa, fb), "snr_db": snr(fa, fb), "ssim": ssim_score(fa, fb)})
    if not rows:
        return {"frames": 0, "rows": []}
    keys = [k for k in rows[0] if k not in {"proc_frame_idx", "orig_frame_idx"}]
    return {"frames": len(rows), "rows": rows, **{f"mean_{k}": float(mean([r[k] for r in rows])) for k in keys}}


def make_tampered_video(cipher: str, out_path: str, case: str, max_frames: int, seed: int) -> Dict[str, Any]:
    w, h, fps = video_props(cipher)
    kind, writer, stream = open_writer(out_path, w, h, fps)
    rng = np.random.default_rng(seed)
    changed_frames = 0
    total_frames = 0
    try:
        for _, _, fr in iter_frames(cipher, max_frames, 1):
            out = fr.copy()
            if total_frames == 0 and case == "single_byte_flip":
                flat = out.reshape(-1)
                pos = min(flat.size - 1, 12345)
                flat[pos] = np.uint8(int(flat[pos]) ^ 1)
                changed_frames += 1
            elif total_frames == 0 and case == "block_zero":
                y1, y2 = h // 3, min(h, h // 3 + max(1, h // 8))
                x1, x2 = w // 3, min(w, w // 3 + max(1, w // 8))
                out[y1:y2, x1:x2] = 0
                changed_frames += 1
            elif case == "random_sparse_bitflips":
                if rng.random() < 0.25:
                    flat = out.reshape(-1)
                    n = max(1, flat.size // 10000)
                    idx = rng.choice(flat.size, size=n, replace=False)
                    flat[idx] ^= np.uint8(1)
                    changed_frames += 1
            elif total_frames == 0 and case == "full_frame_invert":
                out = cv2.bitwise_not(out)
                changed_frames += 1
            write_frame(kind, writer, stream, out)
            total_frames += 1
    finally:
        close_writer(kind, writer, stream)
    return {"case": case, "frames_written": total_frames, "changed_frames": changed_frames, "writer": kind}


def main() -> None:
    ap = argparse.ArgumentParser(description="AES-CTR tamper/malleability test. Decryption is expected to succeed but output should be corrupted.")
    ap.add_argument("--plain", required=True)
    ap.add_argument("--cipher", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--framework_path", default="framework_full_encrypt.py")
    ap.add_argument("--master_key", required=True)
    ap.add_argument("--max_frames", type=int, default=100)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out = ensure_dir(args.out)
    cases = ["single_byte_flip", "block_zero", "random_sparse_bitflips", "full_frame_invert"]
    baseline_dec = out / "baseline_decrypted.mkv"
    baseline_ok = True
    baseline_error = ""
    try:
        framework_cli_decrypt(args.framework_path, args.cipher, str(baseline_dec), args.master_key, verbose=args.verbose)
        baseline_metrics = compare_videos(args.plain, str(baseline_dec), args.max_frames, args.stride)
    except Exception as exc:
        baseline_ok = False
        baseline_error = repr(exc)
        baseline_metrics = None

    results = []
    for case in cases:
        cdir = ensure_dir(out / case)
        tampered = cdir / "tampered_cipher.mkv"
        dec = cdir / "decrypted_tampered.mkv"
        tamper_meta = make_tampered_video(args.cipher, str(tampered), case, args.max_frames, args.seed)
        decrypt_ok = True
        decrypt_error = ""
        try:
            framework_cli_decrypt(args.framework_path, str(tampered), str(dec), args.master_key, verbose=args.verbose)
        except Exception as exc:
            decrypt_ok = False
            decrypt_error = repr(exc)
        metrics = compare_videos(args.plain, str(dec), args.max_frames, args.stride) if decrypt_ok else None
        cipher_delta = compare_videos(args.cipher, str(tampered), args.max_frames, args.stride)
        if metrics and metrics.get("rows"):
            write_csv(cdir / "plain_vs_decrypted_tampered.csv", metrics["rows"])
        results.append({
            "case": case,
            "expected_reject": False,
            "decrypt_ok": decrypt_ok,
            "decrypt_error": decrypt_error,
            "tamper_meta": tamper_meta,
            "tampered_cipher_sha256": sha256_file(tampered),
            "cipher_vs_tampered": {k: v for k, v in cipher_delta.items() if k != "rows"},
            "plain_vs_decrypted_tampered": {k: v for k, v in (metrics or {}).items() if k != "rows"},
            "malleability_observed": bool(decrypt_ok and metrics and metrics.get("mean_mse", 0.0) > 0.0),
        })

    report = {
        "test_type": "full_frame_aes_ctr_tamper_malleability",
        "interpretation": "AES-CTR is unauthenticated. Tampered ciphertext is expected to decrypt without rejection, producing corrupted plaintext.",
        "inputs": {"plain": args.plain, "cipher": args.cipher, "framework_path": args.framework_path},
        "plain_info": video_info(args.plain),
        "cipher_info": video_info(args.cipher),
        "baseline": {"decrypt_ok": baseline_ok, "decrypt_error": baseline_error, "plain_vs_decrypted": {k: v for k, v in (baseline_metrics or {}).items() if k != "rows"}},
        "results": results,
        "summary": {"cases": len(results), "decrypt_rejections": sum(1 for r in results if not r["decrypt_ok"]), "malleability_cases": sum(1 for r in results if r["malleability_observed"]), "all_passed": baseline_ok and all(r["decrypt_ok"] and r["malleability_observed"] for r in results)},
    }
    write_json(out / "report.json", report)
    print(json.dumps({"ok": True, "report": str(out / "report.json"), "all_passed": report["summary"]["all_passed"]}))


if __name__ == "__main__":
    main()
