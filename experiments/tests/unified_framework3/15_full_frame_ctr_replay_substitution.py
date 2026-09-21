#!/usr/bin/env python3
from __future__ import annotations

"""
15_full_frame_ctr_replay_substitution.py

Full-frame AES-CTR replay/substitution test.

There is no payload/manifest/AEAD tag in this full-frame CTR framework, so the
correct behavior is not cryptographic rejection. The test demonstrates that
frame replay/substitution or wrong-key/session use decrypts successfully but
produces corrupted plaintext.
"""

import argparse
import json
from fractions import Fraction
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from unified_utils import derive_key_variant, ensure_dir, framework_cli_decrypt, iter_frames, mae, mse, psnr, snr, video_info, write_csv, write_json

try:
    import av
except Exception:
    av = None


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


def read_frames(path: str, max_frames: int) -> List[np.ndarray]:
    return [fr for _, _, fr in iter_frames(path, max_frames=max_frames, stride=1)]


def write_frames(frames: List[np.ndarray], out_path: str, fps: float) -> Dict[str, Any]:
    if not frames:
        raise ValueError("no frames")
    h, w = frames[0].shape[:2]
    kind, writer, stream = open_writer(out_path, w, h, fps)
    try:
        for fr in frames:
            write_frame(kind, writer, stream, fr)
    finally:
        close_writer(kind, writer, stream)
    return {"frames_written": len(frames), "writer": kind}


def compare_videos(a: str, b: str, max_frames: int, stride: int) -> Dict[str, Any]:
    rows = []
    for (proc_i, orig_i, fa), (_, _, fb) in zip(iter_frames(a, max_frames, stride), iter_frames(b, max_frames, stride)):
        rows.append({"proc_frame_idx": proc_i, "orig_frame_idx": orig_i, "mse": mse(fa, fb), "mae": mae(fa, fb), "psnr_db": psnr(fa, fb), "snr_db": snr(fa, fb)})
    if not rows:
        return {"frames": 0, "rows": []}
    keys = [k for k in rows[0] if k not in {"proc_frame_idx", "orig_frame_idx"}]
    return {"frames": len(rows), "rows": rows, **{f"mean_{k}": float(mean([r[k] for r in rows])) for k in keys}}


def build_replay_case(cipher: str, out_path: str, case: str, max_frames: int) -> Dict[str, Any]:
    w, h, fps = video_props(cipher)
    frames = read_frames(cipher, max_frames=max_frames)
    if len(frames) < 2:
        raise RuntimeError("need at least 2 frames")
    changed = 0
    if case == "frame0_to_frame1":
        frames[1] = frames[0].copy(); changed = 1
    elif case == "swap_frame0_frame1":
        frames[0], frames[1] = frames[1].copy(), frames[0].copy(); changed = 2
    elif case == "reverse_first_10":
        n = min(10, len(frames))
        frames[:n] = list(reversed([f.copy() for f in frames[:n]])); changed = n
    elif case == "duplicate_first_half_from_first_frame":
        n = max(1, len(frames) // 2)
        for i in range(n):
            frames[i] = frames[0].copy()
        changed = n
    else:
        raise ValueError(case)
    meta = write_frames(frames, out_path, fps)
    meta.update({"case": case, "changed_frames": changed})
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Full-frame AES-CTR replay/substitution test. No rejection expected; corruption expected.")
    ap.add_argument("--plain", required=True)
    ap.add_argument("--cipher", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--framework_path", default="framework_full_encrypt.py")
    ap.add_argument("--master_key", required=True)
    ap.add_argument("--max_frames", type=int, default=100)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out = ensure_dir(args.out)
    cases = ["frame0_to_frame1", "swap_frame0_frame1", "reverse_first_10", "duplicate_first_half_from_first_frame"]
    results = []
    for case in cases:
        cdir = ensure_dir(out / case)
        mutated = cdir / "mutated_cipher.mkv"
        dec = cdir / "decrypted_mutated.mkv"
        meta = build_replay_case(args.cipher, str(mutated), case, args.max_frames)
        decrypt_ok = True
        decrypt_error = ""
        try:
            framework_cli_decrypt(args.framework_path, str(mutated), str(dec), args.master_key, verbose=args.verbose)
        except Exception as exc:
            decrypt_ok = False
            decrypt_error = repr(exc)
        metrics = compare_videos(args.plain, str(dec), args.max_frames, args.stride) if decrypt_ok else None
        if metrics and metrics.get("rows"):
            write_csv(cdir / "plain_vs_decrypted_mutated.csv", metrics["rows"])
        results.append({
            "case": case,
            "expected_reject": False,
            "decrypt_ok": decrypt_ok,
            "decrypt_error": decrypt_error,
            "mutation_meta": meta,
            "plain_vs_decrypted_mutated": {k: v for k, v in (metrics or {}).items() if k != "rows"},
            "substitution_corruption_observed": bool(decrypt_ok and metrics and metrics.get("mean_mse", 0.0) > 0.0),
        })

    wrong_key = derive_key_variant(args.master_key, "wrong_session")
    wrong_dir = ensure_dir(out / "wrong_key_session")
    wrong_dec = wrong_dir / "decrypted_with_wrong_key.mkv"
    wrong_ok = True
    wrong_error = ""
    try:
        framework_cli_decrypt(args.framework_path, args.cipher, str(wrong_dec), wrong_key, verbose=args.verbose)
    except Exception as exc:
        wrong_ok = False
        wrong_error = repr(exc)
    wrong_metrics = compare_videos(args.plain, str(wrong_dec), args.max_frames, args.stride) if wrong_ok else None
    results.append({
        "case": "wrong_key_or_new_session",
        "expected_reject": False,
        "decrypt_ok": wrong_ok,
        "decrypt_error": wrong_error,
        "plain_vs_decrypted_mutated": {k: v for k, v in (wrong_metrics or {}).items() if k != "rows"},
        "substitution_corruption_observed": bool(wrong_ok and wrong_metrics and wrong_metrics.get("mean_mse", 0.0) > 0.0),
    })

    report = {
        "test_type": "full_frame_aes_ctr_replay_substitution",
        "interpretation": "AES-CTR has no built-in replay/substitution rejection. Mutations/wrong keys should not be accepted as authentic; this framework demonstrates corruption rather than rejection.",
        "inputs": {"plain": args.plain, "cipher": args.cipher},
        "plain_info": video_info(args.plain),
        "cipher_info": video_info(args.cipher),
        "results": results,
        "summary": {"cases": len(results), "decrypt_rejections": sum(1 for r in results if not r["decrypt_ok"]), "corruption_cases": sum(1 for r in results if r["substitution_corruption_observed"]), "all_passed": all(r["decrypt_ok"] and r["substitution_corruption_observed"] for r in results)},
    }
    write_json(out / "report.json", report)
    print(json.dumps({"ok": True, "report": str(out / "report.json"), "all_passed": report["summary"]["all_passed"]}))


if __name__ == "__main__":
    main()
