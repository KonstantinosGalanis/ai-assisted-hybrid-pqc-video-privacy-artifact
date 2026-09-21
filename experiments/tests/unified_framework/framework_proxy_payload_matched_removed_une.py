import argparse
import base64
import hashlib
import hmac
import importlib.util
import json
import os
import zlib
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Optional, Tuple, List

import av
import cv2
import numpy as np
import torch
from tqdm import tqdm


# -----------------------------
# Matched public-output settings
# -----------------------------
MATCHED_VIDEO_CRF = os.environ.get("MATCHED_VIDEO_CRF", "18")
MATCHED_VIDEO_PRESET = os.environ.get("MATCHED_VIDEO_PRESET", "medium")
MATCHED_AUDIO_BITRATE = int(os.environ.get("MATCHED_AUDIO_BITRATE", "192000"))

VIDEO_PREVIEW_MODE = os.environ.get("VIDEO_PREVIEW_MODE", "mosaic")  # mosaic|blur|black|chaos
AUDIO_PREVIEW_MODE = os.environ.get("AUDIO_PREVIEW_MODE", "chaos")  # chaos|mute|passthrough
MOSAIC_BLOCK = int(os.environ.get("VIDEO_MOSAIC_BLOCK", "20"))
BLUR_KSIZE = int(os.environ.get("VIDEO_BLUR_KSIZE", "31"))
PAYLOAD_ZLIB_LEVEL = int(os.environ.get("PAYLOAD_ZLIB_LEVEL", "6"))


# -----------------------------
# Helpers
# -----------------------------

def _load_base_framework(path: str):
    spec = importlib.util.spec_from_file_location("base_framework", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base framework: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _video_identity(path: str) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return {"width": width, "height": height, "fps": fps, "frames": frames}


def _session_id(master_key: str, width: int, height: int, fps: float, frames: int, has_audio: bool) -> str:
    msg = f"session|w={int(width)}|h={int(height)}|fps={float(fps):.6f}|frames={int(frames)}|audio={1 if has_audio else 0}".encode("utf-8")
    return hmac.new(master_key.encode("utf-8"), msg, hashlib.sha256).hexdigest()[:32]


def _manifest_default_path(payload_path: str) -> str:
    return str(payload_path) + ".manifest.json"


def _write_manifest(path: str, rec: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, sort_keys=True)


def _load_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _verify_decrypt_inputs(manifest_path: str, encrypted_video_path: str, roi_sidecar_path: str, payload_path: str, payload_header: dict, current_session_id: str, current_identity: dict) -> None:
    if not manifest_path or not os.path.isfile(manifest_path):
        raise ValueError(f"Decrypt verification failed: manifest not found: {manifest_path}")
    manifest = _load_manifest(manifest_path)

    checks = [
        ("encrypted video", _sha256_file(encrypted_video_path), str(manifest.get("encrypted_video_sha256", ""))),
        ("ROI sidecar", _sha256_file(roi_sidecar_path), str(manifest.get("roi_sidecar_sha256", ""))),
        ("payload", _sha256_file(payload_path), str(manifest.get("payload_sha256", ""))),
    ]
    for label, got, expect in checks:
        if got != expect:
            raise ValueError(f"Decrypt verification failed: {label} does not match manifest.")

    if str(manifest.get("session_id", "")) != current_session_id:
        raise ValueError("Decrypt verification failed: session_id mismatch for encrypted video.")

    for k in ("width", "height", "frames"):
        if int(manifest.get(k, -1)) != int(current_identity.get(k, -2)):
            raise ValueError(f"Decrypt verification failed: manifest {k} mismatch.")
    if abs(float(manifest.get("fps", 0.0)) - float(current_identity.get("fps", 0.0))) > 0.05:
        raise ValueError("Decrypt verification failed: manifest fps mismatch.")

    if payload_header:
        if str(payload_header.get("session_id", "")) != current_session_id:
            raise ValueError("Decrypt verification failed: payload header session_id mismatch.")
        for k in ("width", "height", "frames"):
            if int(payload_header.get(k, -1)) != int(current_identity.get(k, -2)):
                raise ValueError(f"Decrypt verification failed: payload header {k} mismatch.")
        if abs(float(payload_header.get("fps", 0.0)) - float(current_identity.get("fps", 0.0))) > 0.05:
            raise ValueError("Decrypt verification failed: payload header fps mismatch.")

def _b64e(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _b64d(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


def _safe_rate(fps: float) -> Fraction:
    if fps is None or not np.isfinite(fps) or fps <= 0:
        return Fraction(30, 1)
    return Fraction(fps).limit_denominator(100000)


def _guess_container_format_from_path(path: str) -> Optional[str]:
    ext = os.path.splitext(str(path))[1].lower()
    mapping = {
        ".mkv": "matroska",
        ".webm": "webm",
        ".mp4": "mp4",
        ".m4v": "ipod",
        ".mov": "mov",
        ".avi": "avi",
    }
    return mapping.get(ext)


def _extract_pix_fmt_name(vstream) -> Optional[str]:
    try:
        cc = getattr(vstream, "codec_context", None)
        fmt = getattr(cc, "pix_fmt", None)
        if isinstance(fmt, str) and fmt:
            return fmt
        fmt_obj = getattr(cc, "format", None)
        name = getattr(fmt_obj, "name", None)
        if isinstance(name, str) and name:
            return name
    except Exception:
        pass
    return None


def _probe_input_profile(in_path: str) -> Dict[str, Optional[str]]:
    info = {
        "container": None,
        "video_codec": None,
        "video_pix_fmt": None,
        "audio_codec": None,
    }
    try:
        probe = av.open(in_path)
        try:
            fmt = getattr(probe, "format", None)
            info["container"] = getattr(fmt, "name", None)
        except Exception:
            pass
        if probe.streams.video:
            vstream = probe.streams.video[0]
            cc = getattr(vstream, "codec_context", None)
            info["video_codec"] = getattr(cc, "name", None) or getattr(vstream.codec_context.codec, "name", None)
            info["video_pix_fmt"] = _extract_pix_fmt_name(vstream)
        if probe.streams.audio:
            astream = probe.streams.audio[0]
            acc = getattr(astream, "codec_context", None)
            info["audio_codec"] = getattr(acc, "name", None) or getattr(astream.codec_context.codec, "name", None)
        probe.close()
    except Exception:
        pass
    return info


def _matched_yuv_pix_fmt(source_pix_fmt: Optional[str]) -> str:
    spf = (source_pix_fmt or "").lower()
    if "444" in spf:
        return "yuv444p"
    if "422" in spf:
        return "yuv422p"
    if "420" in spf:
        return "yuv420p"
    return "yuv420p"


def _select_video_encoder(source_video_codec: Optional[str]):
    svc = (source_video_codec or "").lower().strip()
    if svc in {"h264", "avc1"}:
        return "libx264", {"crf": str(MATCHED_VIDEO_CRF), "preset": str(MATCHED_VIDEO_PRESET)}
    if svc in {"hevc", "h265", "hev1", "hvc1"}:
        return "libx265", {"crf": str(MATCHED_VIDEO_CRF), "preset": str(MATCHED_VIDEO_PRESET)}
    if svc == "vp9":
        return "libvpx-vp9", {"crf": "33", "b:v": "0"}
    if svc == "vp8":
        return "libvpx", {"crf": "10", "b:v": "0"}
    if svc in {"av1", "av01"}:
        return "libaom-av1", {"crf": "30", "b:v": "0"}
    return "libx264", {"crf": str(MATCHED_VIDEO_CRF), "preset": str(MATCHED_VIDEO_PRESET)}


def _select_video_pix_fmt(source_video_codec: Optional[str], source_pix_fmt: Optional[str]) -> str:
    svc = (source_video_codec or "").lower().strip()
    spf = (source_pix_fmt or "").lower().strip()

    # Normal delivery codecs should stay on broadly supported YUV formats.
    if svc in {"h264", "avc1", "hevc", "h265", "hev1", "hvc1", "vp9", "vp8", "av1", "av01"}:
        return _matched_yuv_pix_fmt(spf)

    # For generated/attacked/helper files, the probe can report formats like bgra/bgr0/rgba
    # or other formats that libx264/libx265 cannot accept directly. Do not preserve those
    # literally when we later fall back to a delivery codec.
    if spf in {
        "bgra", "bgr0", "rgba", "argb", "abgr", "rgb0",
        "bgr24", "rgb24",
        "gbrp", "gbrp10le", "gbrp12le",
        "pal8",
    }:
        return "yuv420p"

    # Gray inputs can stay gray only if you intentionally choose a compatible codec path.
    # For our matched public-output path, yuv420p is the safest default.
    if spf.startswith("gray"):
        return "yuv420p"

    # If the probed format is already a sane YUV/NV format, keep the closest supported family.
    if spf.startswith("yuv") or spf.startswith("nv"):
        return _matched_yuv_pix_fmt(spf)

    return "yuv420p"


def _select_audio_encoder(source_audio_codec: Optional[str]):
    sac = (source_audio_codec or "").lower().strip()
    if sac in {"aac", ""}:
        return "aac", MATCHED_AUDIO_BITRATE
    if sac == "flac":
        return "flac", None
    if sac in {"mp3", "mp2"}:
        return "libmp3lame", MATCHED_AUDIO_BITRATE
    if sac == "opus":
        return "libopus", MATCHED_AUDIO_BITRATE
    if sac == "vorbis":
        return "libvorbis", MATCHED_AUDIO_BITRATE
    if sac in {"alac", "pcm_s16le", "pcm_s24le", "pcm_s32le", "pcm_f32le", "pcm_f64le"}:
        return sac, None
    return "aac", MATCHED_AUDIO_BITRATE


@dataclass
class OutputProfile:
    container_format: Optional[str]
    video_codec: str
    video_pix_fmt: str
    video_options: Dict[str, str]
    audio_codec: str
    audio_bitrate: Optional[int]
    source_container: Optional[str] = None
    source_video_codec: Optional[str] = None
    source_video_pix_fmt: Optional[str] = None
    source_audio_codec: Optional[str] = None


def _sanitize_output_profile(profile: OutputProfile) -> OutputProfile:
    vcodec = (profile.video_codec or "").lower().strip()
    vpf = (profile.video_pix_fmt or "").lower().strip()

    # Delivery codecs in this framework should not be paired with raw RGB/BGRA style formats.
    if vcodec in {"libx264", "libx265", "libvpx", "libvpx-vp9", "libaom-av1"}:
        if (
            not vpf
            or vpf in {"bgra", "bgr0", "rgba", "argb", "abgr", "rgb0", "bgr24", "rgb24", "pal8"}
            or vpf.startswith("gray")
            or vpf.startswith("gbr")
            or (not (vpf.startswith("yuv") or vpf.startswith("nv")))
        ):
            profile.video_pix_fmt = "yuv420p"

    return profile


def choose_output_profile(in_path: str, out_path: str) -> OutputProfile:
    src = _probe_input_profile(in_path)
    container_format = _guess_container_format_from_path(out_path)
    if container_format is None:
        src_container = (src.get("container") or "").split(",")[0].strip() or None
        if src_container in {"matroska", "webm", "mp4", "mov", "avi"}:
            container_format = src_container
    video_codec, video_options = _select_video_encoder(src.get("video_codec"))
    video_pix_fmt = _select_video_pix_fmt(src.get("video_codec"), src.get("video_pix_fmt"))
    audio_codec, audio_bitrate = _select_audio_encoder(src.get("audio_codec"))
    profile = OutputProfile(
        container_format=container_format,
        video_codec=video_codec,
        video_pix_fmt=video_pix_fmt,
        video_options=video_options,
        audio_codec=audio_codec,
        audio_bitrate=audio_bitrate,
        source_container=src.get("container"),
        source_video_codec=src.get("video_codec"),
        source_video_pix_fmt=src.get("video_pix_fmt"),
        source_audio_codec=src.get("audio_codec"),
    )
    return _sanitize_output_profile(profile)


def open_video_writer(out_path: str, width: int, height: int, fps: float, profile: OutputProfile):
    profile = _sanitize_output_profile(profile)
    if profile.container_format:
        out = av.open(out_path, mode="w", format=profile.container_format)
    else:
        out = av.open(out_path, mode="w")
    rate = _safe_rate(fps)
    st = out.add_stream(profile.video_codec, rate=rate)
    st.width = width
    st.height = height
    st.pix_fmt = profile.video_pix_fmt
    st.options = dict(profile.video_options)
    try:
        st.codec_context.time_base = Fraction(1, int(rate))
    except Exception:
        pass
    return out, st


def add_audio_stream(out_container, sr: int, layout_name: str, profile: OutputProfile):
    aout = out_container.add_stream(profile.audio_codec, rate=sr)
    aout.layout = layout_name
    if profile.audio_bitrate is not None:
        try:
            aout.bit_rate = int(profile.audio_bitrate)
        except Exception:
            pass
    return aout


# -----------------------------
# Public preview transforms
# -----------------------------

def _mosaic_masked_roi(roi_bgr: np.ndarray, mask: np.ndarray, block: int = MOSAIC_BLOCK) -> np.ndarray:
    h, w = roi_bgr.shape[:2]
    bw = max(1, int(block))
    sw = max(1, w // bw)
    sh = max(1, h // bw)
    small = cv2.resize(roi_bgr, (sw, sh), interpolation=cv2.INTER_LINEAR)
    pix = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    out = roi_bgr.copy()
    out[mask] = pix[mask]
    return out


def _blur_masked_roi(roi_bgr: np.ndarray, mask: np.ndarray, ksize: int = BLUR_KSIZE) -> np.ndarray:
    kk = max(3, int(ksize) | 1)
    blurred = cv2.GaussianBlur(roi_bgr, (kk, kk), 0)
    out = roi_bgr.copy()
    out[mask] = blurred[mask]
    return out


def _black_masked_roi(roi_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = roi_bgr.copy()
    out[mask] = 0
    return out


def apply_public_video_preview(base, roi_bgr: np.ndarray, mask: np.ndarray, seed: bytes, map_name: str, mode: str) -> np.ndarray:
    mode = (mode or "mosaic").lower()
    if mode == "mosaic":
        return _mosaic_masked_roi(roi_bgr, mask)
    if mode == "blur":
        return _blur_masked_roi(roi_bgr, mask)
    if mode == "black":
        return _black_masked_roi(roi_bgr, mask)
    if mode == "chaos":
        preview, _ = base._encrypt_masked_bytes_cpu_chaos(
            roi_bgr.copy(),
            mask,
            seed,
            map_name,
            frame_idx=0,
            track_id=-1,
            cls_id=-1,
            policy_tag="preview",
            bbox=None,
            mask_hash="",
            roi_index=0,
            crypto_meta={
                "scheme": "chaos_psd_v2_local",
                "stream_nonce_b64": _b64e(bytes(16)),
                "tags_b64": _b64e(bytes(16)),
                "chunk_size": 1024,
                "block_size": 1024,
            },
        )
        return preview
    return _mosaic_masked_roi(roi_bgr, mask)


# -----------------------------
# Payload crypto
# -----------------------------

def _payload_common_kwargs(frame_idx: int, track_id: int, cls_id: int, policy_tag: str, bbox, mask_hash: str, roi_index: int):
    return dict(
        frame_idx=int(frame_idx),
        track_id=int(track_id),
        cls_id=int(cls_id),
        policy_tag=str(policy_tag),
        bbox=None if bbox is None else tuple(int(v) for v in bbox),
        mask_hash=str(mask_hash or ""),
        roi_index=int(roi_index),
    )


def _payload_encrypt_bytes(base, plain: bytes, seed: bytes, map_name: str, domain: str, common: dict):
    work_seed = base.derive_stream_seed(seed, f"payload|{domain}")
    meta = base._prepare_chaos_crypto_meta(work_seed, map_name, **common)
    nonce_a = base._b64d(str(meta.get("stream_nonce_b64", "")))
    nonce_b = base._b64d(str(meta.get("tags_b64", "")))
    core_seed = base._bind_roi_seed(work_seed, **common)
    stream_seed = hmac.new(core_seed, nonce_a + nonce_b, hashlib.sha256).digest()
    arr = np.frombuffer(plain, dtype=np.uint8).copy()
    ks = np.asarray(base.keystream_u8(base.derive_stream_seed(stream_seed, f"payload-xor|{domain}"), map_name, int(arr.size)), dtype=np.uint8)
    out = np.bitwise_xor(arr, ks).astype(np.uint8, copy=False)
    return out.tobytes(), meta


def _payload_decrypt_bytes(base, cipher: bytes, seed: bytes, map_name: str, meta: dict, domain: str, common: dict):
    work_seed = base.derive_stream_seed(seed, f"payload|{domain}")
    nonce_a = base._b64d(str(meta.get("stream_nonce_b64", "")))
    nonce_b = base._b64d(str(meta.get("tags_b64", "")))
    core_seed = base._bind_roi_seed(work_seed, **common)
    stream_seed = hmac.new(core_seed, nonce_a + nonce_b, hashlib.sha256).digest()
    arr = np.frombuffer(cipher, dtype=np.uint8).copy()
    ks = np.asarray(base.keystream_u8(base.derive_stream_seed(stream_seed, f"payload-xor|{domain}"), map_name, int(arr.size)), dtype=np.uint8)
    out = np.bitwise_xor(arr, ks).astype(np.uint8, copy=False)
    return out.tobytes()


def _payload_keystream_bytes(base, plain_len: int, seed: bytes, map_name: str, meta: dict, domain: str, common: dict) -> bytes:
    work_seed = base.derive_stream_seed(seed, f"payload|{domain}")
    nonce_a = base._b64d(str(meta.get("stream_nonce_b64", "")))
    nonce_b = base._b64d(str(meta.get("tags_b64", "")))
    core_seed = base._bind_roi_seed(work_seed, **common)
    stream_seed = hmac.new(core_seed, nonce_a + nonce_b, hashlib.sha256).digest()
    n = max(0, int(plain_len))
    if n == 0:
        return b""
    ks = np.asarray(
        base.keystream_u8(base.derive_stream_seed(stream_seed, f"payload-xor|{domain}"), map_name, n),
        dtype=np.uint8,
    )
    return ks.tobytes()


def _payload_tag(master_key: str, rec_wo_tag: dict, cipher: bytes) -> str:
    aad = json.dumps(rec_wo_tag, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hmac.new(master_key.encode("utf-8"), aad + cipher, hashlib.sha256).hexdigest()


def _store_payload_record(payload_f, rec: dict):
    payload_f.write(json.dumps(rec, separators=(",", ":")) + "\n")


def _load_payload_records(path: str) -> Tuple[dict, Dict[int, dict], Optional[dict]]:
    header = {}
    frames: Dict[int, dict] = {}
    tail = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            kind = rec.get("type", "frame")
            if kind == "meta":
                header = rec
            elif kind == "tail_audio":
                tail = rec
            else:
                frames[int(rec["frame_idx"])] = rec
    return header, frames, tail


# -----------------------------
# Main processing
# -----------------------------

def process_video(
    base,
    in_path: str,
    out_path: str,
    master_key: str,
    mode: str,
    roi_sidecar_path: Optional[str],
    payload_path: Optional[str],
    manifest_path: Optional[str],
    detect_width: int,
    keystream_dump_path: Optional[str],
    video_preview_mode: str,
    audio_preview_mode: str,
):
    assert mode in {"encrypt", "decrypt"}
    if mode == "decrypt" and not payload_path:
        raise ValueError("Decrypt requires --payload")
    if payload_path and not manifest_path:
        manifest_path = _manifest_default_path(payload_path)

    det = base.load_detectors() if mode == "encrypt" else None

    if mode == "encrypt" and det is not None:
        print("[INFO] Warmup YOLO + Numba...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        with torch.inference_mode():
            try:
                _ = det.coco.track(
                    dummy,
                    device=base.YOLO_DEVICE,
                    half=base.YOLO_HALF,
                    verbose=False,
                    persist=base.TRACK_PERSIST,
                    tracker=base.TRACKER_CFG,
                    classes=list(base.COCO_CLASSES_OF_INTEREST),
                )
            except Exception:
                _ = det.coco.predict(dummy, device=base.YOLO_DEVICE, half=base.YOLO_HALF, verbose=False)
        _ = base.cubic_keystream_u8(b"warmup", 64)
        _ = base.skew_tent_keystream_u8(b"warmup", 64)
        _ = base.chen_keystream_u8(b"warmup", 64)

    in_container = av.open(in_path)
    in_stream = in_container.streams.video[0]
    in_stream.thread_type = "AUTO"

    audio_container = av.open(in_path)
    audio_info = base._decode_audio_pcm_s16(audio_container)
    audio_container.close()
    has_audio = audio_info is not None
    if has_audio:
        pcm_all, audio_sr, audio_layout = audio_info
    else:
        pcm_all, audio_sr, audio_layout = None, None, None

    fps = float(in_stream.average_rate) if in_stream.average_rate else 30.0
    width = in_stream.codec_context.width
    height = in_stream.codec_context.height
    input_identity = _video_identity(in_path)
    session_id = _session_id(master_key, input_identity["width"], input_identity["height"], input_identity["fps"], input_identity["frames"], has_audio)

    output_profile = choose_output_profile(in_path, out_path)
    if mode == "decrypt" and has_audio:
        container_name = (output_profile.container_format or "").lower()
        if container_name in {"mp4", "ipod", "mov"}:
            output_profile.audio_codec = "alac"
        else:
            output_profile.audio_codec = "flac"
        output_profile.audio_bitrate = None
    print(
        f"[INFO] PUBLIC_OUTPUT_MATCH: container={output_profile.container_format or 'auto'} "
        f"video={output_profile.video_codec}/{output_profile.video_pix_fmt} "
        f"audio={output_profile.audio_codec}"
    )

    out_container, out_stream = open_video_writer(out_path, width, height, fps, output_profile)
    aout = add_audio_stream(out_container, audio_sr, audio_layout, output_profile) if has_audio else None

    sidecar_f = None
    payload_f = None
    if mode == "encrypt":
        if roi_sidecar_path:
            sidecar_f = open(roi_sidecar_path, "w", encoding="utf-8")
        if payload_path:
            payload_f = open(payload_path, "w", encoding="utf-8")
            _store_payload_record(payload_f, {
                "type": "meta",
                "version": 1,
                "session_id": session_id,
                "width": int(input_identity["width"]),
                "height": int(input_identity["height"]),
                "fps": float(input_identity["fps"]),
                "frames": int(input_identity["frames"]),
                "video_preview_mode": video_preview_mode,
                "audio_preview_mode": audio_preview_mode,
                "audio_sr": audio_sr,
                "audio_layout": audio_layout,
                "has_audio": bool(has_audio),
            })

    roi_records: Dict[int, dict] = {}
    payload_header: dict = {}
    payload_records: Dict[int, dict] = {}
    payload_tail: Optional[dict] = None
    if mode == "decrypt":
        if not roi_sidecar_path:
            raise ValueError("Decrypt requires --roi_sidecar")
        with open(roi_sidecar_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                roi_records[int(rec["frame_idx"])] = rec
        payload_header, payload_records, payload_tail = _load_payload_records(payload_path)
        _verify_decrypt_inputs(
            manifest_path=manifest_path,
            encrypted_video_path=in_path,
            roi_sidecar_path=roi_sidecar_path,
            payload_path=payload_path,
            payload_header=payload_header,
            current_session_id=session_id,
            current_identity=input_identity,
        )
        has_audio = bool(payload_header.get("has_audio", has_audio))
        audio_sr = int(payload_header.get("audio_sr", audio_sr or 48000)) if has_audio else None
        audio_layout = payload_header.get("audio_layout", audio_layout or "stereo") if has_audio else None

    total_frames = in_stream.frames if in_stream.frames else None
    pbar = tqdm(total=total_frames, desc=f"{mode} frames", unit="frame")

    ks_f = None
    if mode == "encrypt" and keystream_dump_path:
        ks_f = open(keystream_dump_path, "wb")
        print(f"[INFO] Payload-cipher dump enabled: {keystream_dump_path}")

    audio_pts = 0
    audio_prev_end = 0
    frame_idx = 0
    last_map_name = "cubic"

    for av_frame in in_container.decode(in_stream):
        frame = av_frame.to_ndarray(format="bgr24")
        pict_raw = av_frame.pict_type
        map_name = base.map_for_pict_type(pict_raw)
        last_map_name = map_name

        if mode == "encrypt":
            rois = base.detect_rois_with_masks(
                det,
                frame,
                frame_idx=frame_idx,
                detect_width=detect_width,
                pack_for_sidecar=True,
            )
            for roi_index, r in enumerate(rois):
                r["roi_index"] = int(r.get("roi_index", roi_index))
                cls_id = int(r.get("cls", -1))
                track_id = int(r.get("track_id", -1))
                policy_tag = str(r.get("policy_tag", base.policy_tag_for_class(cls_id)))
                r["policy_tag"] = policy_tag
                roi_mask0 = r.get("mask", None)
                if roi_mask0 is None:
                    roi_mask0 = base.unpack_mask(r["mask_pack"])
                r["mask_hash"] = str(r.get("mask_hash", base._compute_mask_hash(roi_mask0)))
        else:
            rec = roi_records.get(frame_idx, {"rois": [], "map_name": map_name})
            map_name = str(rec.get("map_name", map_name))
            rois = rec.get("rois", [])

        # sidecar write
        if mode == "encrypt" and sidecar_f is not None:
            rois_for_json = []
            for roi_index, r in enumerate(rois):
                mask_obj = r.get("mask_pack") if r.get("mask_pack") is not None else base.pack_mask(r["mask"])
                rois_for_json.append({
                    "bbox": r["bbox"],
                    "mask_pack": mask_obj,
                    "cls": int(r.get("cls", -1)),
                    "conf": float(r.get("conf", 0.0)),
                    "track_id": int(r.get("track_id", -1)),
                    "policy_tag": str(r.get("policy_tag", base.policy_tag_for_class(int(r.get("cls", -1))))),
                    "roi_index": int(r.get("roi_index", roi_index)),
                    "mask_hash": str(r.get("mask_hash", "")),
                })
            sidecar_f.write(json.dumps({
                "frame_idx": frame_idx,
                "map_name": map_name,
                "rois": rois_for_json,
            }) + "\n")

        payload_frame = {"type": "frame", "frame_idx": frame_idx, "map_name": map_name, "rois": []}

        # process ROI video
        rois_to_apply = rois if mode == "encrypt" else list(reversed(rois))
        for r in rois_to_apply:
            x1, y1, x2, y2 = map(int, r["bbox"])
            x1 = base.clamp(x1, 0, width - 1)
            y1 = base.clamp(y1, 0, height - 1)
            x2 = base.clamp(x2, 1, width)
            y2 = base.clamp(y2, 1, height)
            if x2 <= x1 or y2 <= y1:
                continue
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            roi_mask = r.get("mask")
            if roi_mask is None:
                roi_mask = base.unpack_mask(r["mask_pack"])
            if roi_mask.shape[:2] != (roi.shape[0], roi.shape[1]):
                roi_mask = cv2.resize(roi_mask.astype(np.uint8), (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            if roi_mask.sum() == 0:
                continue

            track_id = int(r.get("track_id", -1))
            cls_id = int(r.get("cls", -1))
            policy_tag = str(r.get("policy_tag", base.policy_tag_for_class(cls_id)))
            roi_index = int(r.get("roi_index", 0))
            mask_hash = str(r.get("mask_hash", base._compute_mask_hash(roi_mask)))
            bbox = (x1, y1, x2, y2)
            seed = base.derive_policy_seed(
                master_key=master_key,
                frame_idx=frame_idx,
                track_id=track_id,
                cls_id=cls_id,
                map_name=map_name,
                policy_tag=policy_tag,
                scope=base.POLICY_SCOPE,
                bbox=bbox,
                mask_hash=mask_hash,
                roi_index=roi_index,
            )
            common = _payload_common_kwargs(frame_idx, track_id, cls_id, policy_tag, bbox, mask_hash, roi_index)

            if mode == "encrypt":
                # store recoverable secret in payload
                orig_pixels = np.ascontiguousarray(roi[roi_mask], dtype=np.uint8).reshape(-1)
                comp = zlib.compress(orig_pixels.tobytes(), level=PAYLOAD_ZLIB_LEVEL)
                cipher, meta = _payload_encrypt_bytes(base, comp, seed, map_name, "roi", common)
                rec_wo_tag = {
                    "roi_index": roi_index,
                    "bbox": [x1, y1, x2, y2],
                    "cls": cls_id,
                    "track_id": track_id,
                    "policy_tag": policy_tag,
                    "mask_hash": mask_hash,
                    "plain_len": int(orig_pixels.size),
                    "comp_len": int(len(comp)),
                    "crypto_meta": meta,
                }
                tag = _payload_tag(master_key, rec_wo_tag, cipher)
                payload_frame["rois"].append({
                    **rec_wo_tag,
                    "cipher_b64": _b64e(cipher),
                    "tag": tag,
                })
                if ks_f is not None and cipher:
                    ks_f.write(cipher)

                # write codec-friendly public preview
                public_roi = apply_public_video_preview(base, roi, roi_mask, seed, map_name, video_preview_mode)
                frame[y1:y2, x1:x2] = public_roi
            else:
                frame_payload = payload_records.get(frame_idx, {"rois": []})
                match = None
                for pr in frame_payload.get("rois", []):
                    if int(pr.get("roi_index", -1)) == roi_index:
                        match = pr
                        break
                if match is None:
                    continue
                cipher = _b64d(match["cipher_b64"])
                rec_wo_tag = {
                    "roi_index": int(match["roi_index"]),
                    "bbox": match["bbox"],
                    "cls": int(match["cls"]),
                    "track_id": int(match["track_id"]),
                    "policy_tag": str(match["policy_tag"]),
                    "mask_hash": str(match["mask_hash"]),
                    "plain_len": int(match["plain_len"]),
                    "comp_len": int(match["comp_len"]),
                    "crypto_meta": match["crypto_meta"],
                }
                expect = _payload_tag(master_key, rec_wo_tag, cipher)
                if not hmac.compare_digest(expect, str(match.get("tag", ""))):
                    raise ValueError(f"Payload authentication failed for frame {frame_idx} roi {roi_index}")
                plain_comp = _payload_decrypt_bytes(base, cipher, seed, map_name, match["crypto_meta"], "roi", common)
                plain = zlib.decompress(plain_comp)
                pix = np.frombuffer(plain, dtype=np.uint8)
                if pix.size != int(match["plain_len"]):
                    raise ValueError(f"Payload ROI length mismatch for frame {frame_idx} roi {roi_index}")
                roi_out = roi.copy()
                roi_out[roi_mask] = pix.reshape(-1, 3)
                frame[y1:y2, x1:x2] = roi_out

        # mux video frame
        out_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        try:
            out_frame.pts = frame_idx
            out_frame.time_base = Fraction(1, int(round(fps)) if fps and fps > 0 else 30)
        except Exception:
            pass
        for packet in out_stream.encode(out_frame):
            out_container.mux(packet)

        # audio handling
        if has_audio and aout is not None:
            start = int(round(frame_idx * audio_sr / fps))
            end = int(round((frame_idx + 1) * audio_sr / fps))
            if mode == "encrypt":
                start = max(start, audio_prev_end)
                end = max(end, start)
                end = min(end, int(pcm_all.shape[0]))
                if end > start:
                    seg = np.ascontiguousarray(pcm_all[start:end, :], dtype=np.int16)
                    aseed = base.derive_policy_seed(
                        master_key=master_key,
                        frame_idx=frame_idx,
                        track_id=0,
                        cls_id=-1,
                        map_name=map_name,
                        policy_tag="audio_default",
                        scope="class",
                    )
                    # store recoverable audio in payload
                    audio_plain = zlib.compress(seg.tobytes(), level=PAYLOAD_ZLIB_LEVEL)
                    common_audio = _payload_common_kwargs(frame_idx, 0, -1, "audio_default", None, "", 0)
                    acipher, ameta = _payload_encrypt_bytes(base, audio_plain, aseed, map_name, "audio", common_audio)
                    arec_wo_tag = {
                        "shape": [int(seg.shape[0]), int(seg.shape[1])],
                        "dtype": "int16",
                        "plain_len": int(seg.size * seg.dtype.itemsize),
                        "comp_len": int(len(audio_plain)),
                        "crypto_meta": ameta,
                    }
                    atag = _payload_tag(master_key, {"frame_idx": frame_idx, **arec_wo_tag}, acipher)
                    payload_frame["audio"] = {
                        **arec_wo_tag,
                        "cipher_b64": _b64e(acipher),
                        "tag": atag,
                    }
                    # Do not mix audio payload bytes into the NIST keystream artifact.

                    # public playable encrypted audio preview
                    if audio_preview_mode == "mute":
                        seg_out = np.zeros_like(seg)
                    elif audio_preview_mode == "passthrough":
                        seg_out = seg
                    else:
                        seg_out = base._chaos_audio_segment(seg.copy(), aseed, map_name, decrypt=False)
                    audio_pts = base._mux_pcm_segment(out_container, aout, seg_out, audio_sr, audio_layout, audio_pts)
                audio_prev_end = end
            else:
                frame_payload = payload_records.get(frame_idx, {})
                arec = frame_payload.get("audio")
                if arec is not None:
                    common_audio = _payload_common_kwargs(frame_idx, 0, -1, "audio_default", None, "", 0)
                    aseed = base.derive_policy_seed(
                        master_key=master_key,
                        frame_idx=frame_idx,
                        track_id=0,
                        cls_id=-1,
                        map_name=map_name,
                        policy_tag="audio_default",
                        scope="class",
                    )
                    acipher = _b64d(arec["cipher_b64"])
                    arec_wo_tag = {
                        "frame_idx": frame_idx,
                        "shape": arec["shape"],
                        "dtype": arec["dtype"],
                        "plain_len": int(arec["plain_len"]),
                        "comp_len": int(arec["comp_len"]),
                        "crypto_meta": arec["crypto_meta"],
                    }
                    expect = _payload_tag(master_key, arec_wo_tag, acipher)
                    if not hmac.compare_digest(expect, str(arec.get("tag", ""))):
                        raise ValueError(f"Payload authentication failed for frame {frame_idx} audio")
                    plain_comp = _payload_decrypt_bytes(base, acipher, aseed, map_name, arec["crypto_meta"], "audio", common_audio)
                    plain = zlib.decompress(plain_comp)
                    seg = np.frombuffer(plain, dtype=np.int16).reshape(int(arec["shape"][0]), int(arec["shape"][1]))
                    audio_pts = base._mux_pcm_segment(out_container, aout, seg, audio_sr, audio_layout, audio_pts)

        if mode == "encrypt" and payload_f is not None:
            _store_payload_record(payload_f, payload_frame)

        if frame_idx % 30 == 0:
            print(f"frame={frame_idx} rois={len(rois)} map={map_name}")

        frame_idx += 1
        pbar.update(1)

    # tail audio after last frame
    if mode == "encrypt" and has_audio and payload_f is not None and aout is not None and audio_prev_end < int(pcm_all.shape[0]):
        seg = np.ascontiguousarray(pcm_all[audio_prev_end:, :], dtype=np.int16)
        last_fi = max(frame_idx - 1, 0)
        last_map = last_map_name
        aseed = base.derive_policy_seed(
            master_key=master_key,
            frame_idx=last_fi,
            track_id=0,
            cls_id=-1,
            map_name=last_map,
            policy_tag="audio_default",
            scope="class",
        )
        common_audio = _payload_common_kwargs(last_fi, 0, -1, "audio_default", None, "", 0)
        audio_plain = zlib.compress(seg.tobytes(), level=PAYLOAD_ZLIB_LEVEL)
        acipher, ameta = _payload_encrypt_bytes(base, audio_plain, aseed, last_map, "audio", common_audio)
        arec_wo_tag = {
            "type": "tail_audio",
            "frame_idx": last_fi,
            "shape": [int(seg.shape[0]), int(seg.shape[1])],
            "dtype": "int16",
            "plain_len": int(seg.size * seg.dtype.itemsize),
            "comp_len": int(len(audio_plain)),
            "crypto_meta": ameta,
        }
        atag = _payload_tag(master_key, arec_wo_tag, acipher)
        _store_payload_record(payload_f, {**arec_wo_tag, "cipher_b64": _b64e(acipher), "tag": atag})
        # Do not mix tail-audio payload bytes into the NIST keystream artifact.
        if audio_preview_mode == "mute":
            seg_out = np.zeros_like(seg)
        elif audio_preview_mode == "passthrough":
            seg_out = seg
        else:
            seg_out = base._chaos_audio_segment(seg.copy(), aseed, last_map, decrypt=False)
        audio_pts = base._mux_pcm_segment(out_container, aout, seg_out, audio_sr, audio_layout, audio_pts)

    if mode == "decrypt" and has_audio and aout is not None and payload_tail is not None:
        last_fi = int(payload_tail.get("frame_idx", max(frame_idx - 1, 0)))
        last_map = last_map_name
        aseed = base.derive_policy_seed(
            master_key=master_key,
            frame_idx=last_fi,
            track_id=0,
            cls_id=-1,
            map_name=last_map,
            policy_tag="audio_default",
            scope="class",
        )
        common_audio = _payload_common_kwargs(last_fi, 0, -1, "audio_default", None, "", 0)
        acipher = _b64d(payload_tail["cipher_b64"])
        arec_wo_tag = {
            "type": "tail_audio",
            "frame_idx": int(payload_tail["frame_idx"]),
            "shape": payload_tail["shape"],
            "dtype": payload_tail["dtype"],
            "plain_len": int(payload_tail["plain_len"]),
            "comp_len": int(payload_tail["comp_len"]),
            "crypto_meta": payload_tail["crypto_meta"],
        }
        expect = _payload_tag(master_key, arec_wo_tag, acipher)
        if not hmac.compare_digest(expect, str(payload_tail.get("tag", ""))):
            raise ValueError("Payload authentication failed for tail audio")
        plain_comp = _payload_decrypt_bytes(base, acipher, aseed, last_map, payload_tail["crypto_meta"], "audio", common_audio)
        plain = zlib.decompress(plain_comp)
        seg = np.frombuffer(plain, dtype=np.int16).reshape(int(payload_tail["shape"][0]), int(payload_tail["shape"][1]))
        audio_pts = base._mux_pcm_segment(out_container, aout, seg, audio_sr, audio_layout, audio_pts)

    if aout is not None:
        for pkt in aout.encode():
            out_container.mux(pkt)
    for packet in out_stream.encode():
        out_container.mux(packet)

    pbar.close()
    if sidecar_f is not None:
        sidecar_f.close()
    if payload_f is not None:
        payload_f.close()
    if ks_f is not None:
        ks_f.close()
    out_container.close()
    in_container.close()

    if mode == "encrypt" and payload_path and manifest_path:
        out_identity = _video_identity(out_path)
        manifest = {
            "version": 1,
            "session_id": session_id,
            "width": int(out_identity["width"]),
            "height": int(out_identity["height"]),
            "fps": float(out_identity["fps"]),
            "frames": int(out_identity["frames"]),
            "has_audio": bool(has_audio),
            "encrypted_video_path": os.path.basename(out_path),
            "roi_sidecar_path": os.path.basename(roi_sidecar_path) if roi_sidecar_path else None,
            "payload_path": os.path.basename(payload_path),
            "encrypted_video_sha256": _sha256_file(out_path),
            "roi_sidecar_sha256": _sha256_file(roi_sidecar_path) if roi_sidecar_path else None,
            "payload_sha256": _sha256_file(payload_path),
        }
        _write_manifest(manifest_path, manifest)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Matched public proxy + encrypted payload framework")
    ap.add_argument("--base_framework", default="framework_faster_removed_une.py.py")
    ap.add_argument("--mode", choices=["encrypt", "decrypt"], required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--roi_sidecar", default="rois.jsonl")
    ap.add_argument("--payload", default="payload.jsonl")
    ap.add_argument("--manifest", default=None, help="Decryption verification manifest; defaults to <payload>.manifest.json")
    ap.add_argument("--detect_width", type=int, default=0)
    ap.add_argument("--keystream_dump", default=None, help="Dump encrypted payload bytes for NIST/ciphertext testing")
    ap.add_argument("--video_preview_mode", choices=["mosaic", "blur", "black", "chaos"], default=VIDEO_PREVIEW_MODE)
    ap.add_argument("--audio_preview_mode", choices=["chaos", "mute", "passthrough"], default=AUDIO_PREVIEW_MODE)
    args = ap.parse_args()

    base = _load_base_framework(args.base_framework)
    print(f"[INFO] Loaded base framework: {args.base_framework}")
    print(f"[INFO] Video preview mode: {args.video_preview_mode}")
    print(f"[INFO] Audio preview mode: {args.audio_preview_mode}")

    process_video(
        base=base,
        in_path=args.inp,
        out_path=args.out,
        master_key=args.key,
        mode=args.mode,
        roi_sidecar_path=args.roi_sidecar,
        payload_path=args.payload,
        manifest_path=args.manifest,
        detect_width=args.detect_width,
        keystream_dump_path=args.keystream_dump,
        video_preview_mode=args.video_preview_mode,
        audio_preview_mode=args.audio_preview_mode,
    )
    print(f"[INFO] Done. Wrote: {args.out}")
