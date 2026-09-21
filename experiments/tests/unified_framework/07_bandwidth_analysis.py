
"""
bandwidth_format_framework_roi.py

Extends bandwidth_format_framework.py with ROI-aware analysis.

Important note:
- Container bitrate / bandwidth (VI-K) is inherently a *file/stream-level* property, not "ROI-only".
  You cannot directly measure "ROI-only bitrate" from an existing encoded file without re-encoding a
  ROI-masked version of the video.
- What we *can* do (and what this script adds) is ROI workload / coverage analysis using your rois.jsonl:
  * ROI coverage over time (fraction of frame pixels inside ROI masks)
  * Estimated encrypted payload (ROI_pixels * 3 bytes per frame for BGR XOR), and an estimated payload bitrate
  * Optional correlation export: bitrate_over_time vs roi_fraction per 1-second bin

This is useful for:
- explaining why bitrate/bandwidth changes with selective encryption,
- reporting "how much of the video was actually encrypted" (common in ROI encryption papers),
- giving a fair context to VI-K bandwidth consumption.

Covered tests:
1) Bandwidth consumption:
   - file size and cipher/plain ratio
   - average bitrate (ffprobe reported or computed)
2) Bitrate over time:
   - ffprobe packet timestamps + sizes -> bins -> stats + CSV
3) Format compliance:
   - container/codec/resolution/fps/duration/frame-count/stream-count comparisons
4) ROI coverage / encrypted payload estimate (NEW):
   - ROI fraction timeseries + mean/median/p95
   - estimated encrypted bytes + estimated encrypted payload bitrate (bps)

Dependencies:
- ffprobe in PATH (ffmpeg)
Optional:
- pymediainfo (pip install pymediainfo)
ROI sidecar:
- produced by your framework_faster.py as rois.jsonl
- requires framework_faster.py to provide unpack_mask(mask_pack, h, w) if you choose --roi_area_method mask.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def vprint(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg, flush=True)


def ensure_ffprobe() -> str:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe not found in PATH. Install ffmpeg (ffprobe) and try again.")
    return ffprobe


def run_cmd(cmd: List[str]) -> str:
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\nSTDERR:\n{p.stderr[:2000]}")
    return p.stdout


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "" or s.lower() == "n/a":
            return None
        return float(s)
    except Exception:
        return None


def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, int):
            return int(x)
        s = str(x).strip()
        if s == "" or s.lower() == "n/a":
            return None
        return int(float(s))
    except Exception:
        return None


# --------------------------
# ffprobe metadata
# --------------------------

def ffprobe_json(path: str, ffprobe: str) -> Dict[str, Any]:
    out = run_cmd([ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path])
    return json.loads(out)


def pick_stream(streams: List[Dict[str, Any]], codec_type: str) -> Optional[Dict[str, Any]]:
    for s in streams:
        if s.get("codec_type") == codec_type:
            return s
    return None


def parse_fps(avg_frame_rate: Optional[str]) -> Optional[float]:
    if not avg_frame_rate:
        return None
    s = str(avg_frame_rate)
    if "/" in s:
        num, den = s.split("/", 1)
        try:
            numf = float(num); denf = float(den)
            if denf == 0:
                return None
            return numf / denf
        except Exception:
            return None
    return safe_float(s)


def summarize_ffprobe(meta: Dict[str, Any]) -> Dict[str, Any]:
    fmt = meta.get("format", {}) or {}
    streams = meta.get("streams", []) or []
    v = pick_stream(streams, "video")
    a = pick_stream(streams, "audio")

    duration = safe_float(fmt.get("duration"))
    bit_rate = safe_float(fmt.get("bit_rate"))
    size = safe_int(fmt.get("size"))
    if size is None and fmt.get("filename"):
        try:
            size = os.path.getsize(fmt["filename"])
        except Exception:
            size = None

    computed_bit_rate = None
    if duration and duration > 0 and size is not None:
        computed_bit_rate = (size * 8.0) / duration

    out: Dict[str, Any] = {
        "container_format_name": fmt.get("format_name"),
        "container_format_long_name": fmt.get("format_long_name"),
        "duration_sec": duration,
        "bit_rate_bps": bit_rate,
        "bit_rate_bps_computed": computed_bit_rate,
        "file_size_bytes_ffprobe": size,
        "nb_streams": safe_int(fmt.get("nb_streams")),
    }

    if v:
        out.update({
            "video_codec": v.get("codec_name"),
            "video_codec_long": v.get("codec_long_name"),
            "video_profile": v.get("profile"),
            "video_pix_fmt": v.get("pix_fmt"),
            "width": safe_int(v.get("width")),
            "height": safe_int(v.get("height")),
            "avg_frame_rate": v.get("avg_frame_rate"),
            "fps": parse_fps(v.get("avg_frame_rate")),
            "nb_frames": safe_int(v.get("nb_frames")),
            "video_bit_rate_bps": safe_float(v.get("bit_rate")),
        })

    if a:
        out.update({
            "audio_codec": a.get("codec_name"),
            "audio_codec_long": a.get("codec_long_name"),
            "audio_sample_rate": safe_int(a.get("sample_rate")),
            "audio_channels": safe_int(a.get("channels")),
            "audio_bit_rate_bps": safe_float(a.get("bit_rate")),
        })

    return out


# --------------------------
# Optional pymediainfo
# --------------------------

def mediainfo_summary(path: str) -> Optional[Dict[str, Any]]:
    try:
        from pymediainfo import MediaInfo  # type: ignore
    except Exception:
        return None

    try:
        mi = MediaInfo.parse(path)
        general = None
        video = None
        audio = None
        for t in mi.tracks:
            if t.track_type == "General" and general is None:
                general = t
            elif t.track_type == "Video" and video is None:
                video = t
            elif t.track_type == "Audio" and audio is None:
                audio = t

        out: Dict[str, Any] = {}
        if general:
            out.update({
                "general_format": getattr(general, "format", None),
                "general_duration_ms": getattr(general, "duration", None),
                "general_overall_bit_rate": getattr(general, "overall_bit_rate", None),
                "general_file_size": getattr(general, "file_size", None),
            })
        if video:
            out.update({
                "mi_video_format": getattr(video, "format", None),
                "mi_video_codec_id": getattr(video, "codec_id", None),
                "mi_width": getattr(video, "width", None),
                "mi_height": getattr(video, "height", None),
                "mi_frame_rate": getattr(video, "frame_rate", None),
                "mi_bit_rate": getattr(video, "bit_rate", None),
                "mi_color_space": getattr(video, "color_space", None),
                "mi_chroma_subsampling": getattr(video, "chroma_subsampling", None),
                "mi_bit_depth": getattr(video, "bit_depth", None),
            })
        if audio:
            out.update({
                "mi_audio_format": getattr(audio, "format", None),
                "mi_audio_sampling_rate": getattr(audio, "sampling_rate", None),
                "mi_audio_channels": getattr(audio, "channel_s", None),
                "mi_audio_bit_rate": getattr(audio, "bit_rate", None),
            })
        return out
    except Exception:
        return None


# --------------------------
# Bitrate over time (plotbitrate-inspired)
# --------------------------

@dataclass
class BitrateStats:
    bin_size_sec: float
    bins: int
    mean_bps: float
    median_bps: float
    p95_bps: float
    p99_bps: float
    peak_bps: float


def bitrate_timeseries_ffprobe(
    path: str,
    ffprobe: str,
    bin_size_sec: float = 1.0,
    stream: str = "v:0",
    max_packets: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    cmd = [
        ffprobe, "-v", "error",
        "-select_streams", stream,
        "-show_packets",
        "-show_entries", "packet=pts_time,size",
        "-of", "csv=p=0",
        path,
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    pts: List[float] = []
    sizes: List[int] = []
    try:
        assert p.stdout is not None
        for i, line in enumerate(p.stdout):
            if max_packets is not None and i >= max_packets:
                break
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            t = safe_float(parts[0])
            s = safe_int(parts[1])
            if t is None or s is None:
                continue
            pts.append(t)
            sizes.append(s)
    finally:
        try:
            p.kill()
        except Exception:
            pass

    if not pts:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    pts_arr = np.array(pts, dtype=np.float64)
    size_arr = np.array(sizes, dtype=np.float64)

    t0 = float(np.min(pts_arr))
    t1 = float(np.max(pts_arr))
    duration = max(0.0, t1 - t0)
    nbins = int(np.floor(duration / bin_size_sec)) + 1
    bins = np.zeros((nbins,), dtype=np.float64)

    idx = np.floor((pts_arr - t0) / bin_size_sec).astype(np.int64)
    idx = np.clip(idx, 0, nbins - 1)
    np.add.at(bins, idx, size_arr)

    bps = (bins * 8.0) / bin_size_sec
    times = t0 + np.arange(nbins, dtype=np.float64) * bin_size_sec
    return times, bps


def bitrate_stats(bps: np.ndarray, bin_size_sec: float) -> Optional[BitrateStats]:
    if bps.size == 0:
        return None
    return BitrateStats(
        bin_size_sec=float(bin_size_sec),
        bins=int(bps.size),
        mean_bps=float(np.mean(bps)),
        median_bps=float(np.median(bps)),
        p95_bps=float(np.percentile(bps, 95)),
        p99_bps=float(np.percentile(bps, 99)),
        peak_bps=float(np.max(bps)),
    )


# --------------------------
# ROI coverage / payload estimate (NEW)
# --------------------------

def _import_framework_faster(framework_faster_path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("framework_faster_roi", framework_faster_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import framework_faster from {framework_faster_path}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def roi_coverage_timeseries(
    roi_sidecar_path: str,
    width: int,
    height: int,
    fps: float,
    out_csv: str,
    framework_faster_path: str,
    area_method: str = "bbox",   # bbox | mask
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Reads rois.jsonl and outputs a per-frame ROI fraction timeseries.
    area_method:
      - bbox: fast approximate area using bbox area sum (clipped)
      - mask: decode mask_pack with framework_faster.unpack_mask and sum pixels (more accurate)
    """
    H, W = int(height), int(width)
    frame_area = float(H * W)

    unpack_mask = None
    if area_method == "mask":
        ff = _import_framework_faster(framework_faster_path)
        if not hasattr(ff, "unpack_mask"):
            raise RuntimeError("framework_faster.py missing unpack_mask(mask_pack, h, w)")
        unpack_mask = ff.unpack_mask

    rows = []
    total_roi_pixels = 0.0
    frames = 0
    frames_with_roi = 0

    with open(roi_sidecar_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            frame_idx = int(obj.get("frame_idx", -1))
            rois = obj.get("rois", []) or []
            roi_pixels = 0.0

            if area_method == "bbox":
                # sum bbox area (fast, approximate; ignores holes in masks)
                for r in rois:
                    bbox = r.get("bbox")
                    if not bbox:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    x1 = max(0, min(W, x1)); x2 = max(0, min(W, x2))
                    y1 = max(0, min(H, y1)); y2 = max(0, min(H, y2))
                    if x2 > x1 and y2 > y1:
                        roi_pixels += float((x2 - x1) * (y2 - y1))
            else:
                # decode masks and sum pixels (accurate)
                for r in rois:
                    bbox = r.get("bbox")
                    mask_pack = r.get("mask_pack")
                    if not bbox or mask_pack is None:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    x1 = max(0, min(W, x1)); x2 = max(0, min(W, x2))
                    y1 = max(0, min(H, y1)); y2 = max(0, min(H, y2))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    mh = y2 - y1
                    mw = x2 - x1
                    m = unpack_mask(mask_pack)
                    m = np.array(m, dtype=bool)
                    if m.ndim == 1:
                        if m.size != mh * mw:
                            raise ValueError(f"mask size {m.size} != mh*mw {mh*mw} for bbox {bbox}")
                        m = m.reshape((mh, mw))
                    roi_pixels += float(np.sum(m))

            frac = float(roi_pixels / frame_area) if frame_area > 0 else 0.0
            t_sec = float(frame_idx / fps) if fps and fps > 0 else float(frame_idx)

            rows.append((t_sec, frame_idx, roi_pixels, frac))
            total_roi_pixels += roi_pixels
            frames += 1
            if roi_pixels > 0:
                frames_with_roi += 1

    rows.sort(key=lambda x: x[1])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        f.write("time_sec,frame_idx,roi_pixels,roi_fraction\n")
        for t_sec, frame_idx, roi_pixels, frac in rows:
            f.write(f"{t_sec:.6f},{frame_idx},{roi_pixels:.1f},{frac:.8f}\n")

    frac_arr = np.array([r[3] for r in rows], dtype=np.float64) if rows else np.array([], dtype=np.float64)

    # estimated encrypted payload: ROI_pixels * 3 bytes (BGR) per frame
    est_encrypted_bytes = float(total_roi_pixels * 3.0)
    duration_sec = float((frames / fps)) if fps and fps > 0 else None
    est_payload_bps = float((est_encrypted_bytes * 8.0) / duration_sec) if duration_sec and duration_sec > 0 else None

    summary = {
        "roi_sidecar_path": roi_sidecar_path,
        "area_method": area_method,
        "frames_seen_in_sidecar": frames,
        "frames_with_roi": frames_with_roi,
        "mean_roi_fraction": float(np.mean(frac_arr)) if frac_arr.size else 0.0,
        "median_roi_fraction": float(np.median(frac_arr)) if frac_arr.size else 0.0,
        "p95_roi_fraction": float(np.percentile(frac_arr, 95)) if frac_arr.size else 0.0,
        "total_roi_pixels": float(total_roi_pixels),
        "estimated_encrypted_bytes_bgr": est_encrypted_bytes,
        "estimated_payload_bitrate_bps": est_payload_bps,
        "roi_timeseries_csv": out_csv,
    }
    vprint(verbose, f"ROI coverage: mean={summary['mean_roi_fraction']:.4f}, p95={summary['p95_roi_fraction']:.4f}")
    return summary


# --------------------------
# Compliance checks
# --------------------------

def format_compliance_checks(plain: Dict[str, Any], cipher: Dict[str, Any]) -> Dict[str, Any]:
    def same(a, b) -> bool:
        return (a is not None) and (b is not None) and (a == b)

    checks: Dict[str, Any] = {}
    checks["same_container_format"] = same(plain.get("container_format_name"), cipher.get("container_format_name"))
    checks["same_video_codec"] = same(plain.get("video_codec"), cipher.get("video_codec"))
    checks["same_audio_codec"] = same(plain.get("audio_codec"), cipher.get("audio_codec"))
    checks["same_resolution"] = same(plain.get("width"), cipher.get("width")) and same(plain.get("height"), cipher.get("height"))

    fps_p = plain.get("fps"); fps_c = cipher.get("fps")
    checks["fps_delta"] = (fps_c - fps_p) if (isinstance(fps_p, (int,float)) and isinstance(fps_c, (int,float))) else None

    dur_p = plain.get("duration_sec"); dur_c = cipher.get("duration_sec")
    checks["duration_delta_sec"] = (dur_c - dur_p) if (isinstance(dur_p, (int,float)) and isinstance(dur_c, (int,float))) else None

    nbp = plain.get("nb_frames"); nbc = cipher.get("nb_frames")
    checks["nb_frames_plain"] = nbp
    checks["nb_frames_cipher"] = nbc
    checks["same_nb_frames"] = same(nbp, nbc)

    checks["nb_streams_plain"] = plain.get("nb_streams")
    checks["nb_streams_cipher"] = cipher.get("nb_streams")
    checks["same_nb_streams"] = same(plain.get("nb_streams"), cipher.get("nb_streams"))

    size_p = plain.get("file_size_bytes_ffprobe"); size_c = cipher.get("file_size_bytes_ffprobe")
    checks["file_size_ratio_cipher_over_plain"] = float(size_c / size_p) if (isinstance(size_p, int) and size_p > 0 and isinstance(size_c, int)) else None

    br_p = plain.get("bit_rate_bps") or plain.get("bit_rate_bps_computed")
    br_c = cipher.get("bit_rate_bps") or cipher.get("bit_rate_bps_computed")
    checks["avg_bitrate_ratio_cipher_over_plain"] = float(br_c / br_p) if (isinstance(br_p, (int,float)) and br_p > 0 and isinstance(br_c, (int,float))) else None
    checks["same_pixel_format"] = same(plain.get("video_pix_fmt"), cipher.get("video_pix_fmt"))
    checks["bitrate_ratio_interpretable"] = bool(checks["same_container_format"] and checks["same_video_codec"] and checks["same_resolution"] and checks["same_pixel_format"])

    return checks


# --------------------------
# Analysis routines
# --------------------------

def analyze_one(
    path: str,
    out_dir: str,
    ffprobe: str,
    bin_size_sec: float,
    max_packets: Optional[int],
    verbose: bool,
) -> Dict[str, Any]:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    vprint(verbose, f"Analyzing {path}")
    meta = ffprobe_json(path, ffprobe)
    summ = summarize_ffprobe(meta)
    mi = mediainfo_summary(path)

    t, bps = bitrate_timeseries_ffprobe(path, ffprobe, bin_size_sec=bin_size_sec, max_packets=max_packets)
    stats = bitrate_stats(bps, bin_size_sec)

    csv_path = outp / "bitrate_timeseries.csv"
    if bps.size:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            f.write("time_sec,bitrate_bps\n")
            for tt, bb in zip(t.tolist(), bps.tolist()):
                f.write(f"{tt:.6f},{bb:.3f}\n")

    return {
        "path": path,
        "ffprobe_summary": summ,
        "mediainfo_summary": mi,
        "bitrate_timeseries_csv": str(csv_path) if bps.size else None,
        "bitrate_stats": asdict(stats) if stats else None,
    }


def analyze_pair(
    plain_path: str,
    cipher_path: str,
    out_dir: str,
    ffprobe: str,
    bin_size_sec: float,
    max_packets: Optional[int],
    verbose: bool,
    roi_sidecar: Optional[str],
    framework_faster_path: str,
    roi_area_method: str,
) -> Dict[str, Any]:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    plain_res = analyze_one(plain_path, str(outp / "plain"), ffprobe, bin_size_sec, max_packets, verbose)
    cipher_res = analyze_one(cipher_path, str(outp / "cipher"), ffprobe, bin_size_sec, max_packets, verbose)

    checks = format_compliance_checks(plain_res["ffprobe_summary"], cipher_res["ffprobe_summary"])

    ratio_stats = None
    ps = plain_res.get("bitrate_stats")
    cs = cipher_res.get("bitrate_stats")
    if ps and cs:
        ratio_stats = {
            "bin_size_sec": ps["bin_size_sec"],
            "mean_bitrate_ratio": (cs["mean_bps"] / ps["mean_bps"]) if ps["mean_bps"] else None,
            "p95_bitrate_ratio": (cs["p95_bps"] / ps["p95_bps"]) if ps["p95_bps"] else None,
            "peak_bitrate_ratio": (cs["peak_bps"] / ps["peak_bps"]) if ps["peak_bps"] else None,
        }

    roi_summary = None
    if roi_sidecar:
        fps = cipher_res["ffprobe_summary"].get("fps") or plain_res["ffprobe_summary"].get("fps") or 0.0
        w = cipher_res["ffprobe_summary"].get("width") or plain_res["ffprobe_summary"].get("width") or 0
        h = cipher_res["ffprobe_summary"].get("height") or plain_res["ffprobe_summary"].get("height") or 0
        roi_csv = str(outp / "roi_timeseries.csv")
        if w and h and fps:
            roi_summary = roi_coverage_timeseries(
                roi_sidecar_path=roi_sidecar,
                width=int(w),
                height=int(h),
                fps=float(fps),
                out_csv=roi_csv,
                framework_faster_path=framework_faster_path,
                area_method=roi_area_method,
                verbose=verbose,
            )
        else:
            roi_summary = {"error": "Could not determine width/height/fps for ROI analysis."}

    fair_context = None
    if roi_summary and plain_res["ffprobe_summary"].get("fps") and plain_res["ffprobe_summary"].get("width") and plain_res["ffprobe_summary"].get("height"):
        fps = float(plain_res["ffprobe_summary"]["fps"])
        w = int(plain_res["ffprobe_summary"]["width"])
        h = int(plain_res["ffprobe_summary"]["height"])
        raw_full_frame_bitrate_bps = float(w * h * 3 * 8 * fps)
        fair_context = {
            "raw_full_frame_bgr_bitrate_bps": raw_full_frame_bitrate_bps,
            "estimated_roi_payload_ratio_of_raw_full_frame": (roi_summary.get("estimated_payload_bitrate_bps") / raw_full_frame_bitrate_bps) if raw_full_frame_bitrate_bps and roi_summary.get("estimated_payload_bitrate_bps") is not None else None,
            "bitrate_ratio_interpretable": checks.get("bitrate_ratio_interpretable"),
            "interpretation_note": "Cipher/plain bitrate ratios are only fair when codec, container, resolution, and pixel format match. Otherwise prefer the estimated ROI payload bitrate relative to raw full-frame bitrate."
        }
    return {
        "plain": plain_res,
        "cipher": cipher_res,
        "format_compliance": checks,
        "bitrate_ratio_stats": ratio_stats,
        "roi_summary": roi_summary,
        "fairness_context": fair_context,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--video", default=None, help="analyze a single video")
    ap.add_argument("--plain", default=None, help="plain/original video for comparison")
    ap.add_argument("--cipher", default=None, help="cipher/encrypted video for comparison")
    ap.add_argument("--bin_size_sec", type=float, default=1.0)
    ap.add_argument("--max_packets", type=int, default=0, help="limit packets parsed (0 = no limit)")
    ap.add_argument("--verbose", action="store_true")

    # ROI (NEW)
    ap.add_argument("--roi_sidecar", default=None, help="rois.jsonl to compute ROI coverage/payload estimate")
    ap.add_argument("--framework_faster_path", default="framework_faster.py", help="needed if roi_area_method=mask")
    ap.add_argument("--roi_area_method", default="bbox", choices=["bbox", "mask"], help="ROI area method: bbox fast or mask accurate")

    args = ap.parse_args()

    ffprobe = ensure_ffprobe()
    outp = Path(args.out)
    outp.mkdir(parents=True, exist_ok=True)

    max_packets = args.max_packets if args.max_packets and args.max_packets > 0 else None

    if args.video:
        report = analyze_one(args.video, str(outp), ffprobe, args.bin_size_sec, max_packets, args.verbose)
    else:
        if not (args.plain and args.cipher):
            raise SystemExit("Provide either --video OR both --plain and --cipher.")
        report = analyze_pair(
            args.plain, args.cipher, str(outp), ffprobe,
            args.bin_size_sec, max_packets, args.verbose,
            roi_sidecar=args.roi_sidecar,
            framework_faster_path=args.framework_faster_path,
            roi_area_method=args.roi_area_method,
        )

    (outp / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    vprint(args.verbose, f"Wrote {outp/'report.json'}")


if __name__ == "__main__":
    main()
