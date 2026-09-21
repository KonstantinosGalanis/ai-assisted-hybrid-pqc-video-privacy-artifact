#!/usr/bin/env python3
"""
06_robustness_noise_occlusion_full_frame.py

Full-frame AES-CTR robustness / tamper-propagation test.
No rois.jsonl, payload, or manifest is used.

Important interpretation:
- AES-CTR provides confidentiality only. It is malleable and unauthenticated.
- Corrupting ciphertext is not expected to be rejected by this framework.
- Instead, this test measures how corruption propagates into decrypted plaintext.

Scenarios:
- baseline decrypt of original ciphertext
- Gaussian noise: sigma list
- salt-and-pepper noise: probability list
- JPEG/CRF proxy recompression: CRF-like list mapped to JPEG quality
- packet/frame loss simulation: probability list, fill with previous/black
- rectangular occlusion: fraction list

Outputs:
- report.json
- scenario folders with attacked cipher and decrypted attacked videos
"""
from __future__ import annotations

import argparse
import math
from fractions import Fraction
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple
import json
import cv2
import numpy as np

from unified_utils import (
    ensure_dir,
    framework_cli_decrypt,
    iter_frames,
    mae,
    mse,
    psnr,
    snr,
    video_info,
    write_csv,
    write_json,
)

try:
    import av
except Exception:
    av = None

try:
    from skimage.metrics import structural_similarity as sk_ssim
except Exception:
    sk_ssim = None


def parse_float_list(text: str, default: Iterable[float]) -> List[float]:
    if text is None or str(text).strip() == "":
        return [float(x) for x in default]
    out: List[float] = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out


def _safe_rate(fps: float) -> Fraction:
    if fps is None or not np.isfinite(fps) or fps <= 0:
        return Fraction(30, 1)
    return Fraction(float(fps)).limit_denominator(100000)


def _read_video_props(path: str) -> Tuple[int, int, float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    cap.release()
    return w, h, fps


def _open_lossless_video_writer(out_path: str, w: int, h: int, fps: float):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if av is not None:
        container = av.open(out_path, mode="w", format="matroska")
        st = container.add_stream("libx264rgb", rate=_safe_rate(fps))
        st.width = int(w)
        st.height = int(h)
        st.pix_fmt = "bgr24"
        st.options = {
            "preset": "slower",
            "qp": "0",
            "x264-params": "keyint=10000:min-keyint=10000:scenecut=0:bframes=0",
        }
        return "pyav", container, st

    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    vw = cv2.VideoWriter(out_path, fourcc, fps if fps > 0 else 25.0, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"Could not open lossless writer for {out_path}. Install PyAV/FFmpeg with FFV1/libx264rgb.")
    return "opencv", vw, None


def _write_frame(writer_kind: str, writer, stream, frame_bgr: np.ndarray) -> None:
    if writer_kind == "pyav":
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame_bgr), format="bgr24")
        for packet in stream.encode(vf):
            writer.mux(packet)
    else:
        writer.write(frame_bgr)


def _close_writer(writer_kind: str, writer, stream) -> None:
    if writer_kind == "pyav":
        for packet in stream.encode():
            writer.mux(packet)
        writer.close()
    else:
        writer.release()


def jpeg_quality_from_crf(crf: float) -> int:
    # Proxy mapping: lower CRF = better quality, higher CRF = lower quality.
    # Clamp for OpenCV JPEG quality 1..100.
    return int(max(5, min(100, round(100 - (float(crf) - 18.0) * 3.0))))


def apply_attack(frame: np.ndarray, attack: str, strength: float, rng: np.random.Generator, previous: np.ndarray | None = None, frame_idx: int = 0, fill: str = "previous") -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    if attack == "gaussian":
        sigma = float(strength)
        noise = rng.normal(0.0, sigma, size=out.shape)
        return np.clip(out.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    if attack == "saltpepper":
        p = min(max(float(strength), 0.0), 1.0)
        mask = rng.random(out.shape[:2])
        out[mask < p / 2.0] = 0
        out[(mask >= p / 2.0) & (mask < p)] = 255
        return out
    if attack == "jpeg_crf":
        quality = jpeg_quality_from_crf(strength)
        ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return out
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return dec if dec is not None else out
    if attack == "frame_loss":
        p = min(max(float(strength), 0.0), 1.0)
        if rng.random() < p:
            if fill == "black" or previous is None:
                return np.zeros_like(out)
            return previous.copy()
        return out
    if attack == "occlusion":
        frac = min(max(float(strength), 0.0), 0.95)
        oh = max(1, int(h * frac))
        ow = max(1, int(w * frac))
        y1 = max(0, (h - oh) // 2)
        x1 = max(0, (w - ow) // 2)
        out[y1:y1 + oh, x1:x1 + ow] = 0
        return out
    raise ValueError(attack)


def write_attacked_video_lossless(in_path: str, out_path: str, attack: str, strength: float, max_frames: int = 0, seed: int = 0, frame_loss_fill: str = "previous") -> Dict[str, Any]:
    w, h, fps = _read_video_props(in_path)
    writer_kind, writer, stream = _open_lossless_video_writer(out_path, w, h, fps)
    rng = np.random.default_rng(int(seed))
    frames = 0
    replaced = 0
    previous = None
    try:
        for _, _, fr in iter_frames(in_path, max_frames=max_frames, stride=1):
            before = fr
            attacked = apply_attack(fr, attack, strength, rng, previous=previous, frame_idx=frames, fill=frame_loss_fill)
            if attack == "frame_loss" and not np.array_equal(before, attacked):
                replaced += 1
            _write_frame(writer_kind, writer, stream, attacked)
            previous = attacked.copy()
            frames += 1
    finally:
        _close_writer(writer_kind, writer, stream)
    return {"frames_written": frames, "writer": writer_kind, "frame_loss_replacements": replaced}


def ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[: min(a.shape[0], b.shape[0]), : min(a.shape[1], b.shape[1])], b[: min(a.shape[0], b.shape[0]), : min(a.shape[1], b.shape[1])]
    if sk_ssim is None:
        m = mse(a, b)
        return float(1.0 / (1.0 + m / (255.0 * 255.0)))
    return float(sk_ssim(a, b, data_range=255, channel_axis=2 if a.ndim == 3 else None))


def compare_videos(a_path: str, b_path: str, max_frames: int, stride: int) -> Dict[str, Any]:
    rows = []
    for (proc_i, orig_i, a), (_, _, b) in zip(iter_frames(a_path, max_frames, stride), iter_frames(b_path, max_frames, stride)):
        rows.append({
            "proc_frame_idx": proc_i,
            "orig_frame_idx": orig_i,
            "mse": mse(a, b),
            "mae": mae(a, b),
            "psnr_db": psnr(a, b),
            "snr_db": snr(a, b),
            "ssim": ssim_score(a, b),
        })
    if not rows:
        return {"frames": 0, "rows": rows}
    keys = [k for k in rows[0].keys() if k not in {"proc_frame_idx", "orig_frame_idx"}]
    return {"frames": len(rows), "rows": rows, **{f"mean_{k}": float(mean([r[k] for r in rows])) for k in keys}}


def scenario_label(attack: str, strength: float) -> str:
    s = str(strength).replace(".", "p").replace("-", "m")
    return f"{attack}_{s}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Full-frame AES-CTR robustness/tamper propagation. No ROI sidecar is needed.")
    ap.add_argument("--plain", required=True)
    ap.add_argument("--cipher", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--framework_path", default="framework_full_encrypt.py")
    ap.add_argument("--master_key", required=True)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--gaussian_sigmas", default="5,10,20,40")
    ap.add_argument("--saltpepper_probs", default="0.001,0.005,0.01,0.05")
    ap.add_argument("--jpeg_crfs", default="18,23,28,35")
    ap.add_argument("--frame_loss_probs", default="0.01,0.05,0.10")
    ap.add_argument("--occlusion_fracs", default="0.10,0.25,0.50")
    ap.add_argument("--frame_loss_fill", choices=["previous", "black"], default="previous")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out = ensure_dir(args.out)
    scenarios: List[Tuple[str, float]] = [("baseline", 0.0)]
    scenarios += [("gaussian", x) for x in parse_float_list(args.gaussian_sigmas, [5, 10, 20, 40])]
    scenarios += [("saltpepper", x) for x in parse_float_list(args.saltpepper_probs, [0.001, 0.005, 0.01, 0.05])]
    scenarios += [("jpeg_crf", x) for x in parse_float_list(args.jpeg_crfs, [18, 23, 28, 35])]
    scenarios += [("frame_loss", x) for x in parse_float_list(args.frame_loss_probs, [0.01, 0.05, 0.10])]
    scenarios += [("occlusion", x) for x in parse_float_list(args.occlusion_fracs, [0.10, 0.25, 0.50])]

    report: Dict[str, Any] = {
        "test_type": "full_frame_aes_ctr_robustness_tamper_propagation",
        "interpretation": "AES-CTR is unauthenticated/malleable: tampering is expected to decrypt without rejection but corrupt plaintext.",
        "plain_info": video_info(args.plain),
        "cipher_info": video_info(args.cipher),
        "scenarios_requested": [{"attack": a, "strength": s} for a, s in scenarios],
        "results": [],
    }

    for attack, strength in scenarios:
        label = "baseline" if attack == "baseline" else scenario_label(attack, strength)
        sdir = ensure_dir(out / label)
        if attack == "baseline":
            attacked_path = args.cipher
            attack_meta = {"frames_written": None, "writer": "original_cipher", "frame_loss_replacements": 0}
        else:
            attacked_path = str(sdir / "attacked_cipher.mkv")
            attack_meta = write_attacked_video_lossless(args.cipher, attacked_path, attack, strength, max_frames=args.max_frames, seed=args.seed, frame_loss_fill=args.frame_loss_fill)
        decrypted_path = str(sdir / "decrypted_from_attacked.mkv")
        dec_ok = True
        dec_error = ""
        try:
            dec_res = framework_cli_decrypt(args.framework_path, attacked_path, decrypted_path, args.master_key, verbose=args.verbose)
        except Exception as exc:
            dec_ok = False
            dec_error = repr(exc)
            dec_res = {"elapsed_s": None}

        cipher_vs_attacked = None if attack == "baseline" else compare_videos(args.cipher, attacked_path, args.max_frames, args.stride)
        plain_vs_decrypted = compare_videos(args.plain, decrypted_path, args.max_frames, args.stride) if dec_ok else None

        row = {
            "scenario": label,
            "attack": attack,
            "strength": float(strength),
            "attacked_cipher_path": attacked_path,
            "decrypted_path": decrypted_path if dec_ok else None,
            "decrypt_ok": dec_ok,
            "decrypt_error": dec_error,
            "decrypt_seconds": dec_res.get("elapsed_s"),
            "attack_meta": attack_meta,
            "cipher_vs_attacked_cipher": {k: v for k, v in (cipher_vs_attacked or {}).items() if k != "rows"},
            "plain_vs_decrypted_attacked": {k: v for k, v in (plain_vs_decrypted or {}).items() if k != "rows"},
        }
        report["results"].append(row)
        if plain_vs_decrypted and plain_vs_decrypted.get("rows"):
            write_csv(sdir / "plain_vs_decrypted_frame_metrics.csv", plain_vs_decrypted["rows"])
        if cipher_vs_attacked and cipher_vs_attacked.get("rows"):
            write_csv(sdir / "cipher_vs_attacked_frame_metrics.csv", cipher_vs_attacked["rows"])
        if args.verbose:
            print(f"{label}: decrypt_ok={dec_ok}", flush=True)

    write_json(out / "report.json", report)
    print(json.dumps({"ok": True, "report": str(out / "report.json"), "scenarios": len(report["results"])}))


if __name__ == "__main__":
    main()
