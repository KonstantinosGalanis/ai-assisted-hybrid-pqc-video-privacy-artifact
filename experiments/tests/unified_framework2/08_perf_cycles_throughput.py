#!/usr/bin/env python3
"""
08_perf_cycles_throughput.py

Performance framework for the ROI video-encryption pipeline.

Covers:
- Average encryption time and decryption time
- End-to-end throughput (file bytes/s, pixels/s)
- Component-time percentages (line_profiler, optional)
- Core-crypto throughput (in-memory replay using ROI sidecar; excludes detector + mux I/O)
- Cycles/byte and instructions/byte via Linux perf stat (optional)

Compared with the earlier version:
- End-to-end throughput is kept, but explicitly separated from core-crypto throughput
- cycles/byte is implemented with perf stat when available
"""
from __future__ import annotations

import argparse
import datetime
import inspect
import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np


def now() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def vprint(verbose: bool, msg: str) -> None:
    if verbose:
        print(f"[{now()}] {msg}", flush=True)


def safe_mkdir(p: str | Path) -> Path:
    pp = Path(p)
    pp.mkdir(parents=True, exist_ok=True)
    return pp




def _stable_import_from_path(path: str, canonical_name: str):
    """
    Import a module from a filesystem path using a stable module name so that
    Numba cache / pickle metadata do not break across dynamic imports.
    If the same canonical module is already loaded from the same file, reuse it.
    """
    import sys
    import importlib.util
    from pathlib import Path

    resolved = Path(path).resolve()
    if canonical_name in sys.modules:
        mod = sys.modules[canonical_name]
        if Path(getattr(mod, "__file__", "")).resolve() == resolved:
            return mod
        sys.modules.pop(canonical_name, None)

    spec = importlib.util.spec_from_file_location(canonical_name, str(resolved))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {resolved}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[canonical_name] = mod
    spec.loader.exec_module(mod)
    return mod

def load_module_from_path(path: str) -> Any:
    return _stable_import_from_path(path, "framework_faster")



def resolve_base_framework_path(module_path: str, framework_base_path: Optional[str] = None) -> Optional[str]:
    if framework_base_path:
        return str(Path(framework_base_path).resolve())
    try:
        mod = load_module_from_path(module_path)
        if hasattr(mod, 'unpack_mask') and (hasattr(mod, 'derive_seed') or hasattr(mod, 'derive_policy_seed')):
            return str(Path(module_path).resolve())
    except Exception:
        pass
    fp = Path(module_path).resolve()
    for name in ('framework_succesful.py', 'framework_faster.py'):
        cand = fp.with_name(name)
        if cand.exists():
            return str(cand)
    return None


def maybe_load_base_module(module_path: str, framework_base_path: Optional[str] = None):
    bp = resolve_base_framework_path(module_path, framework_base_path)
    if not bp or str(Path(bp).resolve()) == str(Path(module_path).resolve()):
        return None
    return _stable_import_from_path(bp, 'framework_base_perf')

def call_with_supported_kwargs(fn: Callable[..., Any], kwargs: Dict[str, Any]) -> Any:
    sig = inspect.signature(fn)
    supported = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            supported[k] = v
    return fn(**supported)



def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _session_id(master_key: str, width: int, height: int, fps: float, frames: int, has_audio: bool) -> str:
    import hashlib, hmac
    msg = f"session|w={int(width)}|h={int(height)}|fps={float(fps):.6f}|frames={int(frames)}|audio={1 if has_audio else 0}".encode("utf-8")
    return hmac.new(master_key.encode("utf-8"), msg, hashlib.sha256).hexdigest()[:32]


def _manifest_default_path(payload_path: str) -> str:
    return str(payload_path) + ".manifest.json"


def _load_payload_header(payload_path: str) -> dict:
    with open(payload_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "meta":
                return obj
    return {}


def _count_payload_frame_records(payload_path: str) -> int:
    """Return processed video frame count from the payload file itself.

    Payload frame records are authoritative for Test 8 because the freshly
    encoded MKV may report an unstable CAP_PROP_FRAME_COUNT such as 1798
    even when the payload contains records 0..1799.
    """
    max_frame_idx = -1
    count = 0
    with open(payload_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") in {"meta", "tail_audio"}:
                continue
            if "frame_idx" in obj:
                count += 1
                max_frame_idx = max(max_frame_idx, int(obj["frame_idx"]))
    if max_frame_idx >= 0:
        return max_frame_idx + 1
    return count


def _write_manifest(path: str, rec: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, sort_keys=True)


def _build_decrypt_manifest(manifest_path: str, encrypted_video_path: str, roi_sidecar_path: str, payload_path: str, master_key: str) -> str:
    header = _load_payload_header(payload_path)
    ident = video_info(encrypted_video_path)
    has_audio = bool(header.get("has_audio", False))

    payload_frames = _count_payload_frame_records(payload_path)
    frames = int(payload_frames or header.get("frames") or ident["frames"])
    width = int(header.get("width") or ident["width"])
    height = int(header.get("height") or ident["height"])
    fps = float(header.get("fps") or ident["fps"])
    session_id = str(header.get("session_id") or _session_id(master_key, width, height, fps, frames, has_audio))

    manifest = {
        "version": 1,
        "session_id": session_id,
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frames,
        "has_audio": bool(has_audio),
        "encrypted_video_path": os.path.basename(encrypted_video_path),
        "roi_sidecar_path": os.path.basename(roi_sidecar_path) if roi_sidecar_path else None,
        "payload_path": os.path.basename(payload_path),
        "encrypted_video_sha256": _sha256_file(encrypted_video_path),
        "roi_sidecar_sha256": _sha256_file(roi_sidecar_path) if roi_sidecar_path else None,
        "payload_sha256": _sha256_file(payload_path),
    }
    _write_manifest(manifest_path, manifest)
    return manifest_path


def video_info(path: str) -> Dict[str, Any]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "fourcc": int(cap.get(cv2.CAP_PROP_FOURCC) or 0),
        "file_bytes": int(os.path.getsize(path)),
        "suffix": Path(path).suffix.lower(),
    }
    cap.release()
    return info


def unique_outfile(out_dir: Path, stem: str, suffix: str) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    return str(out_dir / f"{stem}_{ts}_p{pid}{suffix}")


@dataclass
class TimingResult:
    mode: str
    seconds: float
    input_bytes: float
    output_bytes: float
    input_video: Dict[str, Any]
    output_path: str
    bytes_per_sec: float
    pixels_per_sec: float
    throughput_kind: str
    payload_path: Optional[str] = None
    # Benchmark-local sidecar used by end-to-end encrypt timing.
    # This prevents Test 08 from overwriting the dataset's original rois.jsonl.
    roi_sidecar_path: Optional[str] = None


@dataclass
class CoreTimingResult:
    mode: str
    seconds: float
    video_roi_bytes: int
    audio_bytes: int
    total_processed_bytes: int
    core_bytes_per_sec: float
    notes: List[str]


@dataclass
class PerfCounterResult:
    mode: str
    cycles: Optional[float]
    instructions: Optional[float]
    processed_bytes: int
    cycles_per_byte: Optional[float]
    instructions_per_byte: Optional[float]
    stderr_excerpt: Optional[str]
    notes: List[str]


@dataclass
class ComponentProfile:
    total_seconds: float
    by_function_seconds: Dict[str, float]
    by_component_seconds: Dict[str, float]
    by_component_percent: Dict[str, float]
    notes: List[str]


@dataclass
class PerfReport:
    module_path: str
    python: str
    platform: str
    args: Dict[str, Any]
    encrypt: Optional[TimingResult]
    decrypt: Optional[TimingResult]
    core_encrypt: Optional[CoreTimingResult]
    core_decrypt: Optional[CoreTimingResult]
    perf_encrypt: Optional[PerfCounterResult]
    perf_decrypt: Optional[PerfCounterResult]
    component_profile: Optional[ComponentProfile]
    notes: List[str]


def run_process_video(
    mod: Any,
    mode: str,
    in_path: str,
    out_path: str,
    key: str,
    roi_sidecar: Optional[str],
    detect_every: int,
    detect_width: int,
    verbose: bool,
    payload: Optional[str] = None,
    base_mod: Any = None,
) -> None:
    if not hasattr(mod, "process_video"):
        raise RuntimeError("Module does not have process_video()")
    fn = mod.process_video
    kwargs = dict(
        in_path=in_path,
        out_path=out_path,
        master_key=key,
        mode=mode,
        roi_sidecar_path=roi_sidecar,
        reuse_rois=(mode == "decrypt"),
        detect_every=detect_every,
        detect_width=detect_width,
        payload_path=payload,
        payload=payload,
        base=base_mod,
    )
    try:
        sig = inspect.signature(fn)
        if "manifest_path" in sig.parameters and payload:
            manifest_path = _manifest_default_path(payload)
            if mode == "decrypt" and roi_sidecar and os.path.exists(payload):
                manifest_path = _build_decrypt_manifest(manifest_path, in_path, roi_sidecar, payload, key)
            kwargs["manifest_path"] = manifest_path
        if "keystream_dump_path" in sig.parameters:
            kwargs["keystream_dump_path"] = None
        if "video_preview_mode" in sig.parameters:
            kwargs["video_preview_mode"] = "chacha"
        if "audio_preview_mode" in sig.parameters:
            kwargs["audio_preview_mode"] = "chacha"
    except Exception:
        pass
    vprint(verbose, f"Calling process_video(mode={mode}) -> {out_path}")
    call_with_supported_kwargs(fn, kwargs)


def bench_once(
    mod: Any,
    mode: str,
    in_path: str,
    out_dir: Path,
    key: str,
    roi_sidecar: Optional[str],
    detect_every: int,
    detect_width: int,
    verbose: bool,
    payload: Optional[str] = None,
    base_mod: Any = None,
) -> TimingResult:
    inp_info = video_info(in_path)
    out_path = unique_outfile(out_dir, f"{mode}", ".mkv")
    payload_out: Optional[str] = None
    payload_in = payload

    roi_sidecar_out: Optional[str] = None
    roi_sidecar_in = roi_sidecar

    if mode == "encrypt":
        payload_out = str(out_dir / f"{Path(out_path).stem}_payload.jsonl")
        payload_in = payload_out

        # IMPORTANT:
        # Proxy payload frameworks open roi_sidecar_path with "w" in encrypt mode.
        # Never pass the dataset's original rois.jsonl to a benchmark encryption,
        # otherwise Test 08 corrupts the package manifest used by later tests.
        if roi_sidecar:
            roi_sidecar_out = str(out_dir / f"{Path(out_path).stem}_rois.jsonl")
            roi_sidecar_in = roi_sidecar_out

    t0 = time.perf_counter()
    run_process_video(mod, mode, in_path, out_path, key, roi_sidecar_in, detect_every, detect_width, verbose, payload=payload_in, base_mod=base_mod)
    t1 = time.perf_counter()
    sec = float(t1 - t0)

    if payload_out and not os.path.exists(payload_out):
        payload_out = None
    if roi_sidecar_out and not os.path.exists(roi_sidecar_out):
        roi_sidecar_out = None

    out_bytes = float(os.path.getsize(out_path)) if os.path.exists(out_path) else 0.0
    in_bytes = float(inp_info["file_bytes"])
    pixels = float(inp_info["width"] * inp_info["height"] * max(1, inp_info["frames"]))
    bps = (in_bytes / sec) if sec > 0 else float("inf")
    pps = (pixels / sec) if sec > 0 else float("inf")
    return TimingResult(
        mode=mode,
        seconds=sec,
        input_bytes=in_bytes,
        output_bytes=out_bytes,
        input_video=inp_info,
        output_path=out_path,
        bytes_per_sec=bps,
        pixels_per_sec=pps,
        throughput_kind="end_to_end_file_wallclock",
        payload_path=payload_out if mode == "encrypt" else payload,
        roi_sidecar_path=roi_sidecar_out if mode == "encrypt" else roi_sidecar,
    )


def _call_unpack_mask(unpack_fn, mask_pack: dict, mh: int, mw: int) -> np.ndarray:
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


def load_sidecar_records(path: str) -> Dict[int, dict]:
    recs: Dict[int, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            recs[int(obj["frame_idx"])] = obj
    return recs


def iter_video_frames(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield idx, frame
        idx += 1
    cap.release()


def core_crypto_replay(
    mod: Any,
    base_mod: Any,
    mode: str,
    in_path: str,
    key: str,
    roi_sidecar: str,
    verbose: bool = False,
) -> CoreTimingResult:
    if not roi_sidecar:
        raise RuntimeError("core_crypto_replay requires --roi_sidecar")
    logic_mod = base_mod if base_mod is not None else mod
    required = ["unpack_mask", "derive_seed", "encrypt_masked_bytes_cpu", "decrypt_masked_bytes_cpu"]
    for name in required:
        if not hasattr(logic_mod, name):
            raise RuntimeError(f"Module does not expose required function: {name}")

    recs = load_sidecar_records(roi_sidecar)
    unpack_mask = logic_mod.unpack_mask
    video_roi_bytes = 0
    audio_bytes = 0
    notes: List[str] = []

    t0 = time.perf_counter()
    last_map = "cubic"
    frame_count = 0
    for orig_idx, frame in iter_video_frames(in_path):
        rec = recs.get(orig_idx, None)
        if rec is None:
            frame_count += 1
            continue
        map_name = str(rec.get("map_name", last_map))
        last_map = map_name
        rois = rec.get("rois", []) or []
        for r in rois:
            x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
            x1 = max(0, min(frame.shape[1], x1)); x2 = max(0, min(frame.shape[1], x2))
            y1 = max(0, min(frame.shape[0], y1)); y2 = max(0, min(frame.shape[0], y2))
            if x2 <= x1 or y2 <= y1:
                continue
            roi = frame[y1:y2, x1:x2]
            mh, mw = roi.shape[:2]
            mask = _call_unpack_mask(unpack_mask, r["mask_pack"], mh, mw)
            if not np.any(mask):
                continue
            packed_mask = np.packbits(mask.reshape(-1).astype(np.uint8))
            mask_hash = __import__("hashlib").blake2b(packed_mask.tobytes(), digest_size=8).hexdigest()
            seed = logic_mod.derive_seed(
                key,
                orig_idx,
                int(r.get("track_id", -1)),
                int(r.get("cls", -1)),
                map_name,
                bbox=(x1, y1, x2, y2),
                mask_hash=mask_hash,
            )
            if mode == "decrypt":
                logic_mod.decrypt_masked_bytes_cpu(roi, mask, seed, map_name)
            else:
                logic_mod.encrypt_masked_bytes_cpu(roi, mask, seed, map_name)
            video_roi_bytes += int(mask.sum()) * int(roi.shape[2] if roi.ndim == 3 else 1)
        frame_count += 1

    # Optional audio-core replay, if the module exposes the same helpers used by framework_faster.py
    if hasattr(logic_mod, "_decode_audio_pcm_s16") and (hasattr(logic_mod, "_xor_audio_segment") or hasattr(logic_mod, "_chaos_audio_segment")):
        try:
            if not hasattr(logic_mod, "av"):
                import av  # noqa
            audio_container = logic_mod.av.open(in_path) if hasattr(logic_mod, "av") else None
            if audio_container is None:
                import av
                audio_container = av.open(in_path)
            audio_info = logic_mod._decode_audio_pcm_s16(audio_container)
            audio_container.close()
            if audio_info is not None:
                pcm_all, audio_sr, audio_layout = audio_info
                info = video_info(in_path)
                fps = float(info["fps"]) if info["fps"] else 30.0
                audio_prev_end = 0
                total_frames = frame_count
                for frame_idx in range(total_frames):
                    rec = recs.get(frame_idx, None)
                    map_name = str(rec.get("map_name", last_map)) if rec is not None else last_map
                    start = int(round(frame_idx * audio_sr / fps))
                    end = int(round((frame_idx + 1) * audio_sr / fps))
                    start = max(start, audio_prev_end)
                    end = max(end, start)
                    end = min(end, int(pcm_all.shape[0]))
                    if end > start:
                        seg = pcm_all[start:end, :]
                        aseed = logic_mod.derive_seed(key, frame_idx, 0, 0, map_name)
                        if hasattr(logic_mod, "_xor_audio_segment"):
                            logic_mod._xor_audio_segment(seg, aseed, map_name)
                        else:
                            logic_mod._chaos_audio_segment(seg, aseed, map_name, decrypt=(mode == "decrypt"))
                        audio_bytes += int(seg.view(np.uint8).size)
                    audio_prev_end = end
                if audio_prev_end < int(pcm_all.shape[0]):
                    seg = pcm_all[audio_prev_end:, :]
                    aseed = logic_mod.derive_seed(key, max(frame_count - 1, 0), 0, 0, last_map)
                    if hasattr(logic_mod, "_xor_audio_segment"):
                        logic_mod._xor_audio_segment(seg, aseed, last_map)
                    else:
                        logic_mod._chaos_audio_segment(seg, aseed, last_map, decrypt=(mode == "decrypt"))
                    audio_bytes += int(seg.view(np.uint8).size)
        except Exception as e:
            notes.append(f"audio-core replay skipped: {e}")
    else:
        notes.append("audio-core replay skipped: module does not expose _decode_audio_pcm_s16 and _xor_audio_segment/_chaos_audio_segment")

    t1 = time.perf_counter()
    sec = float(t1 - t0)
    total_processed = int(video_roi_bytes + audio_bytes)
    core_bps = (total_processed / sec) if sec > 0 else float("inf")
    return CoreTimingResult(
        mode=mode,
        seconds=sec,
        video_roi_bytes=int(video_roi_bytes),
        audio_bytes=int(audio_bytes),
        total_processed_bytes=total_processed,
        core_bytes_per_sec=float(core_bps),
        notes=notes,
    )


def _parse_perf_stat_csv(text: str) -> Dict[str, Optional[float]]:
    out = {"cycles": None, "instructions": None}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        value_txt = parts[0].replace("<not supported>", "").replace("<not counted>", "").strip()
        event = parts[2].strip()
        value_txt = value_txt.replace(" ", "")
        if not value_txt or value_txt.startswith("<"):
            continue
        value_txt = value_txt.replace(".", "") if re.fullmatch(r"\d+\.\d+", value_txt) is None and "." in value_txt and value_txt.count(".") > 1 else value_txt
        value_txt = value_txt.replace(",", "")
        try:
            value = float(parts[0].replace(" ", ""))
        except Exception:
            try:
                value = float(value_txt)
            except Exception:
                continue
        if event == "cycles":
            out["cycles"] = value
        elif event == "instructions":
            out["instructions"] = value
    return out


def perf_stat_core(
    script_path: str,
    module_path: str,
    in_path: str,
    key: str,
    roi_sidecar: str,
    mode: str,
    framework_base_path: Optional[str] = None,
    verbose: bool = False,
) -> PerfCounterResult:
    notes: List[str] = []
    if platform.system() != "Linux":
        return PerfCounterResult(mode, None, None, 0, None, None, None, ["perf stat is only supported on Linux"])

    cmd = [
        "perf", "stat", "-x,", "-e", "cycles,instructions",
        sys.executable, script_path,
        "--_core_helper",
        "--module", module_path,
        "--in", in_path,
        "--key", key,
        "--roi_sidecar", roi_sidecar,
        "--core_mode", mode,
    ]
    if framework_base_path:
        cmd.extend(["--framework_base_path", framework_base_path])
    vprint(verbose, "Running perf stat on core helper...")
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError:
        return PerfCounterResult(mode, None, None, 0, None, None, None, ["perf is not installed or not in PATH"])

    helper_json = None
    for line in proc.stdout.splitlines()[::-1]:
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            helper_json = json.loads(line)
            break
    if helper_json is None:
        return PerfCounterResult(
            mode=mode,
            cycles=None,
            instructions=None,
            processed_bytes=0,
            cycles_per_byte=None,
            instructions_per_byte=None,
            stderr_excerpt=proc.stderr[-500:] if proc.stderr else None,
            notes=["Could not parse core helper JSON output"],
        )

    parsed = _parse_perf_stat_csv(proc.stderr)
    processed_bytes = int(helper_json.get("total_processed_bytes", 0))
    cycles = parsed.get("cycles")
    instructions = parsed.get("instructions")
    cpb = (cycles / processed_bytes) if (cycles is not None and processed_bytes > 0) else None
    ipb = (instructions / processed_bytes) if (instructions is not None and processed_bytes > 0) else None

    if proc.returncode != 0:
        notes.append(f"perf returned code {proc.returncode}")
    return PerfCounterResult(
        mode=mode,
        cycles=cycles,
        instructions=instructions,
        processed_bytes=processed_bytes,
        cycles_per_byte=cpb,
        instructions_per_byte=ipb,
        stderr_excerpt=proc.stderr[-1000:] if proc.stderr else None,
        notes=notes,
    )


def _try_import_line_profiler() -> Any:
    try:
        from line_profiler import LineProfiler  # type: ignore
        return LineProfiler
    except Exception:
        return None


def line_profile_components(
    mod: Any,
    base_mod: Any,
    mode: str,
    in_path: str,
    out_dir: Path,
    key: str,
    roi_sidecar: Optional[str],
    detect_every: int,
    detect_width: int,
    verbose: bool,
) -> ComponentProfile:
    LineProfiler = _try_import_line_profiler()
    notes: List[str] = []
    if LineProfiler is None:
        raise RuntimeError("line_profiler is not installed. Install with: pip install line_profiler")

    default_funcs = ["process_video", "detect_rois_with_masks", "encrypt_masked_bytes_cpu", "decrypt_masked_bytes_cpu", "_encrypt_masked_bytes_cpu_chaos", "_decrypt_masked_bytes_cpu_chaos", "_payload_encrypt_bytes", "_payload_decrypt_bytes", "open_video_writer", "choose_output_profile", "_add_pcm_audio_stream", "_mux_pcm_segment", "_xor_audio_segment", "_chaos_audio_segment"]
    prof = LineProfiler()
    added = 0
    owners = [mod] + ([base_mod] if base_mod is not None else [])
    seen = set()
    for owner in owners:
        for name in default_funcs:
            f = getattr(owner, name, None)
            if callable(f) and id(f) not in seen:
                try:
                    prof.add_function(f)
                    added += 1
                    seen.add(id(f))
                except Exception:
                    pass
    if added == 0:
        for name in default_funcs:
            notes.append(f"Function not found or not callable: {name}")
    if added == 0:
        raise RuntimeError("No functions were added to line_profiler.")

    out_path = unique_outfile(out_dir, f"{mode}_lineprof", ".mkv")
    lineprof_sidecar = str(out_dir / f"{Path(out_path).stem}_rois.jsonl") if (mode == "encrypt" and roi_sidecar) else roi_sidecar

    def driver():
        run_process_video(
            mod=mod,
            mode=mode,
            in_path=in_path,
            out_path=out_path,
            key=key,
            roi_sidecar=lineprof_sidecar,
            detect_every=detect_every,
            detect_width=detect_width,
            verbose=False,
            payload=str(out_dir / f"{mode}_lineprof_payload.jsonl"),
            base_mod=base_mod,
        )

    vprint(verbose, f"Running line_profiler on {added} functions...")
    prof_wrapper = prof(driver)
    t0 = time.perf_counter()
    prof_wrapper()
    t1 = time.perf_counter()

    stats = prof.get_stats()
    unit = float(stats.unit)
    by_function_seconds: Dict[str, float] = {}
    total = 0.0
    for (filename, lineno, func_name), timing_list in stats.timings.items():
        t_ticks = sum(t[2] for t in timing_list)
        sec = float(t_ticks * unit)
        keyname = f"{func_name} ({Path(filename).name}:{lineno})"
        by_function_seconds[keyname] = sec
        total += sec
    if total <= 0:
        total = float(t1 - t0)
        notes.append("line_profiler total time was 0; using wall-clock duration instead.")

    def comp_of(func_key: str) -> str:
        name = func_key.split(" ")[0]
        if name in ("detect_rois_with_masks", "load_detectors", "_maybe_resize_for_detect"):
            return "roi_detection"
        if name in ("encrypt_masked_bytes_cpu", "decrypt_masked_bytes_cpu", "_encrypt_masked_bytes_cpu_chaos", "_decrypt_masked_bytes_cpu_chaos", "_payload_encrypt_bytes", "_payload_decrypt_bytes", "_xor_masked_roi_chen_inplace", "_xor_masked_roi_cubic_inplace", "_xor_masked_roi_skew_tent_inplace", "keystream_u8"):
            return "roi_crypto"
        if name in ("xor_audio_bytes_inplace", "_xor_audio_segment", "_chaos_audio_segment", "_xor_u8_stream_inplace_chen", "_xor_u8_stream_inplace_cubic", "_xor_u8_stream_inplace_skew_tent"):
            return "audio_crypto"
        if name in ("open_lossless_writer", "open_compressed_writer", "open_video_writer", "choose_output_profile", "_mux_pcm_segment", "_add_pcm_audio_stream"):
            return "mux_io"
        if name in ("process_video",):
            return "pipeline_driver"
        return "other"

    by_component_seconds: Dict[str, float] = {}
    for fk, sec in by_function_seconds.items():
        comp = comp_of(fk)
        by_component_seconds[comp] = by_component_seconds.get(comp, 0.0) + sec
    by_component_percent = {k: (100.0 * v / total if total > 0 else float("nan")) for k, v in by_component_seconds.items()}
    return ComponentProfile(
        total_seconds=total,
        by_function_seconds=by_function_seconds,
        by_component_seconds=by_component_seconds,
        by_component_percent=by_component_percent,
        notes=notes,
    )


def main():
    if "--_core_helper" in sys.argv:
        ap = argparse.ArgumentParser()
        ap.add_argument("--_core_helper", action="store_true")
        ap.add_argument("--module", required=True)
        ap.add_argument("--in", dest="inp", required=True)
        ap.add_argument("--key", required=True)
        ap.add_argument("--roi_sidecar", required=True)
        ap.add_argument("--core_mode", default="encrypt", choices=["encrypt", "decrypt"])
        ap.add_argument("--framework_base_path", default=None)
        args = ap.parse_args()
        mod = load_module_from_path(args.module)
        base_mod = maybe_load_base_module(args.module, args.framework_base_path)
        res = core_crypto_replay(mod, base_mod, args.core_mode, args.inp, args.key, args.roi_sidecar, verbose=False)
        print(json.dumps(asdict(res)))
        return

    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True, help="path to encryption module (e.g., framework_faster.py)")
    ap.add_argument("--in", dest="inp", required=True, help="input video for encrypt benchmark")
    ap.add_argument("--out_dir", required=True, help="output directory for artifacts + report.json")
    ap.add_argument("--key", required=True, help="master key string")
    ap.add_argument("--roi_sidecar", default=None, help="optional rois.jsonl")
    ap.add_argument("--payload", default=None, help="payload.jsonl for payload-aware decrypt benchmarking")
    ap.add_argument("--framework_base_path", default=None, help="base framework path for payload-aware modules")
    ap.add_argument("--detect_every", type=int, default=1)
    ap.add_argument("--detect_width", type=int, default=0)
    ap.add_argument("--bench", action="store_true", help="one-shot end-to-end encrypt+decrypt timing")
    ap.add_argument("--core_bench", action="store_true", help="in-memory core-crypto throughput bench; requires --roi_sidecar")
    ap.add_argument("--perf_stat", action="store_true", help="Linux perf stat cycles/byte on the core helper; requires --roi_sidecar")
    ap.add_argument("--line_profile", action="store_true", help="line_profiler breakdown and component %% time")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out_dir = safe_mkdir(args.out_dir)
    notes: List[str] = []
    vprint(args.verbose, f"Loading module: {args.module}")
    mod = load_module_from_path(args.module)
    base_mod = maybe_load_base_module(args.module, args.framework_base_path)

    report = PerfReport(
        module_path=str(Path(args.module).resolve()),
        python=sys.version.replace("\n", " "),
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        args=vars(args),
        encrypt=None,
        decrypt=None,
        core_encrypt=None,
        core_decrypt=None,
        perf_encrypt=None,
        perf_decrypt=None,
        component_profile=None,
        notes=[],
    )

    if args.bench:
        vprint(args.verbose, "Running end-to-end timing for encrypt and decrypt...")
        enc = bench_once(mod, "encrypt", args.inp, out_dir, args.key, args.roi_sidecar, args.detect_every, args.detect_width, args.verbose, payload=None, base_mod=base_mod)
        dec = bench_once(mod, "decrypt", enc.output_path, out_dir, args.key, enc.roi_sidecar_path or args.roi_sidecar, args.detect_every, args.detect_width, args.verbose, payload=(enc.payload_path or args.payload), base_mod=base_mod)
        report.encrypt = enc
        report.decrypt = dec

    if args.core_bench:
        if not args.roi_sidecar:
            notes.append("core_bench skipped: --roi_sidecar is required")
        else:
            try:
                vprint(args.verbose, "Running in-memory core encrypt throughput bench...")
                report.core_encrypt = core_crypto_replay(mod, base_mod, "encrypt", args.inp, args.key, args.roi_sidecar, verbose=args.verbose)
                decrypt_in = report.encrypt.output_path if report.encrypt is not None else None
                if decrypt_in is None:
                    vprint(args.verbose, "Preparing cipher input for core decrypt bench...")
                    enc = bench_once(mod, "encrypt", args.inp, out_dir, args.key, args.roi_sidecar, args.detect_every, args.detect_width, args.verbose, payload=None, base_mod=base_mod)
                    report.encrypt = report.encrypt or enc
                    decrypt_in = enc.output_path
                vprint(args.verbose, "Running in-memory core decrypt throughput bench...")
                core_decrypt_sidecar = (report.encrypt.roi_sidecar_path if report.encrypt is not None else None) or args.roi_sidecar
                report.core_decrypt = core_crypto_replay(mod, base_mod, "decrypt", decrypt_in, args.key, core_decrypt_sidecar, verbose=args.verbose)
            except Exception as e:
                notes.append(f"core_bench failed: {e}")

    if args.perf_stat:
        if not args.roi_sidecar:
            notes.append("perf_stat skipped: --roi_sidecar is required")
        else:
            try:
                report.perf_encrypt = perf_stat_core(__file__, args.module, args.inp, args.key, args.roi_sidecar, "encrypt", framework_base_path=args.framework_base_path, verbose=args.verbose)
                decrypt_in = report.encrypt.output_path if report.encrypt is not None else None
                if decrypt_in is None:
                    enc = bench_once(mod, "encrypt", args.inp, out_dir, args.key, args.roi_sidecar, args.detect_every, args.detect_width, args.verbose, payload=None, base_mod=base_mod)
                    report.encrypt = report.encrypt or enc
                    decrypt_in = enc.output_path
                perf_decrypt_sidecar = (report.encrypt.roi_sidecar_path if report.encrypt is not None else None) or args.roi_sidecar
                report.perf_decrypt = perf_stat_core(__file__, args.module, decrypt_in, args.key, perf_decrypt_sidecar, "decrypt", framework_base_path=args.framework_base_path, verbose=args.verbose)
            except Exception as e:
                notes.append(f"perf_stat failed: {e}")

    if args.line_profile:
        try:
            report.component_profile = line_profile_components(
                mod=mod,
                base_mod=base_mod,
                mode="encrypt",
                in_path=args.inp,
                out_dir=out_dir,
                key=args.key,
                roi_sidecar=args.roi_sidecar,
                detect_every=args.detect_every,
                detect_width=args.detect_width,
                verbose=args.verbose,
            )
        except Exception as e:
            notes.append(f"line_profile failed: {e}")

    report.notes = notes
    (out_dir / "report.json").write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    vprint(args.verbose, f"Wrote {out_dir/'report.json'}")


if __name__ == "__main__":
    main()
