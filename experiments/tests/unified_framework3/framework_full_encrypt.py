import argparse
import base64
import hashlib
import hmac
import time
from fractions import Fraction
from typing import Optional, Tuple

import av
import cv2
import numpy as np
from tqdm import tqdm
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


# -----------------------------
# Utilities
# -----------------------------

def derive_master_key_from_password(master_key: str) -> bytes:
    """Derive a 32-byte AES-256 key from a password-like string for local testing."""
    return hashlib.sha256(master_key.encode("utf-8")).digest()


def derive_subkey(master_key_bytes: bytes, label: str) -> bytes:
    return hmac.new(master_key_bytes, f"subkey|{label}".encode("utf-8"), hashlib.sha256).digest()


def derive_ctr_iv(subkey: bytes, *, kind: str, frame_idx: int, extra: str = "") -> bytes:
    payload = f"iv|kind={kind}|frame={int(frame_idx)}|{extra}".encode("utf-8")
    return hmac.new(subkey, payload, hashlib.sha256).digest()[:16]


def _safe_rate(fps: float) -> Fraction:
    if fps is None or not np.isfinite(fps) or fps <= 0:
        return Fraction(30, 1)
    return Fraction(fps).limit_denominator(100000)


# -----------------------------
# AES-256-CTR helpers
# -----------------------------

def aes256_ctr_crypt_bytes(data: bytes, key32: bytes, iv16: bytes) -> bytes:
    cipher = Cipher(algorithms.AES(key32), modes.CTR(iv16))
    enc = cipher.encryptor()
    return enc.update(data) + enc.finalize()


def aes256_ctr_crypt_frame(frame_bgr: np.ndarray, key32: bytes, iv16: bytes) -> np.ndarray:
    if frame_bgr.dtype != np.uint8:
        frame_bgr = frame_bgr.astype(np.uint8, copy=False)

    raw = np.ascontiguousarray(frame_bgr).reshape(-1)
    out = aes256_ctr_crypt_bytes(raw.tobytes(), key32, iv16)
    out_arr = np.frombuffer(out, dtype=np.uint8).reshape(frame_bgr.shape)
    return out_arr.copy()


def aes256_ctr_crypt_audio_segment(seg_sc_i16: np.ndarray, key32: bytes, iv16: bytes) -> np.ndarray:
    if seg_sc_i16.size == 0:
        return seg_sc_i16

    seg = seg_sc_i16
    if seg.dtype != np.int16:
        seg = seg.astype(np.int16, copy=False)
    if not seg.flags["C_CONTIGUOUS"]:
        seg = np.ascontiguousarray(seg)

    raw = seg.view(np.uint8).reshape(-1)
    out = aes256_ctr_crypt_bytes(raw.tobytes(), key32, iv16)
    out_arr = np.frombuffer(out, dtype=np.uint8)
    raw[:] = out_arr
    return seg


# -----------------------------
# Writer (kept identical to your current full-AES baseline)
# -----------------------------

def open_lossless_writer(out_path: str, width: int, height: int, fps: float):
    out = av.open(out_path, mode="w", format="mp4")
    rate = _safe_rate(fps)

    st = out.add_stream("libx264rgb", rate=rate)
    st.width = width
    st.height = height
    st.pix_fmt = "bgr24"

    st.options = {
        "preset": "slower",
        "qp": "0",
        "x264-params": "keyint=10000:min-keyint=10000:scenecut=0:bframes=0",
    }

    return out, st


# -----------------------------
# Audio helpers
# -----------------------------

def _decode_audio_pcm_s16(in_container: av.container.InputContainer) -> Optional[Tuple[np.ndarray, int, str]]:
    """Decode first audio stream to packed s16 PCM."""
    if not in_container.streams.audio:
        return None

    astream = in_container.streams.audio[0]
    layout_name = astream.layout.name if astream.layout else "stereo"
    sr = int(astream.rate) if getattr(astream, "rate", None) else None

    def _channels_from_layout(layout) -> Optional[int]:
        if not layout:
            return None
        ch = getattr(layout, "channels", None)
        if ch is None:
            return None
        if isinstance(ch, int):
            return ch
        if isinstance(ch, (tuple, list)):
            return len(ch)
        return None

    stream_ch = getattr(astream, "channels", None)
    if isinstance(stream_ch, int) and stream_ch > 0:
        channels = stream_ch
    else:
        channels = _channels_from_layout(astream.layout) or 1

    chunks = []
    resampler = None

    for af in in_container.decode(astream):
        if sr is None and getattr(af, "sample_rate", None):
            sr = int(af.sample_rate)
            resampler = av.audio.resampler.AudioResampler(format="s16", layout=layout_name, rate=sr)

        if resampler is None and sr is not None:
            resampler = av.audio.resampler.AudioResampler(format="s16", layout=layout_name, rate=sr)

        if resampler is None:
            continue

        raf = resampler.resample(af)
        if raf is None:
            continue

        frames = raf if isinstance(raf, list) else [raf]
        for fr in frames:
            arr = fr.to_ndarray()

            fr_ch = _channels_from_layout(getattr(fr, "layout", None))
            ch = fr_ch or channels or 1

            if arr.ndim == 1:
                arr_sc = arr.reshape(-1, ch)
            elif arr.ndim == 2:
                if arr.shape[0] == 1:
                    arr_sc = arr.reshape(-1, ch)
                elif arr.shape[1] == ch:
                    arr_sc = arr
                elif arr.shape[0] == ch:
                    arr_sc = arr.T
                else:
                    arr_sc = arr.reshape(-1, ch)
            else:
                arr_sc = arr.reshape(-1, ch)

            chunks.append(np.ascontiguousarray(arr_sc, dtype=np.int16))

    if sr is None or not chunks:
        return None

    pcm = np.concatenate(chunks, axis=0)
    return pcm, sr, layout_name


def _add_pcm_audio_stream(out_container: av.container.OutputContainer, sr: int, layout_name: str):
    aout = out_container.add_stream("flac", rate=sr)
    aout.layout = layout_name
    return aout


def _mux_pcm_segment(
    out_container: av.container.OutputContainer,
    aout,
    seg_sc_i16: np.ndarray,
    sr: int,
    layout_name: str,
    pts_samples: int,
) -> int:
    if seg_sc_i16.size == 0:
        return pts_samples

    seg_sc_i16 = np.ascontiguousarray(seg_sc_i16, dtype=np.int16)
    packed = seg_sc_i16.reshape(1, -1)
    af = av.AudioFrame.from_ndarray(packed, format="s16", layout=layout_name)
    af.sample_rate = sr
    af.time_base = Fraction(1, sr)
    af.pts = pts_samples

    for pkt in aout.encode(af):
        out_container.mux(pkt)

    return pts_samples + int(seg_sc_i16.shape[0])


# -----------------------------
# Core full AES-256-CTR transform
# -----------------------------

def process_video_full_aes_ctr(
    in_path: str,
    out_path: str,
    master_key_bytes: bytes,
    mode: str,  # "encrypt" or "decrypt"
    cipher_dump_path: Optional[str] = None,
):
    assert mode in {"encrypt", "decrypt"}
    if len(master_key_bytes) != 32:
        raise ValueError("master_key_bytes must be exactly 32 bytes for AES-256")

    # CTR decrypt == encrypt with same key/IV
    video_key = derive_subkey(master_key_bytes, "video_full_aes256ctr")
    audio_key = derive_subkey(master_key_bytes, "audio_full_aes256ctr")

    in_container = av.open(in_path)
    try:
        in_stream = in_container.streams.video[0]
        in_stream.thread_type = "AUTO"

        audio_container = av.open(in_path)
        try:
            audio_info = _decode_audio_pcm_s16(audio_container)
        finally:
            audio_container.close()

        has_audio = audio_info is not None
        if has_audio:
            pcm_all, audio_sr, audio_layout = audio_info
        else:
            pcm_all, audio_sr, audio_layout = None, None, None

        fps = float(in_stream.average_rate) if in_stream.average_rate else 30.0
        width = in_stream.codec_context.width
        height = in_stream.codec_context.height

        out_container, out_stream = open_lossless_writer(out_path, width, height, fps)
        try:
            aout = _add_pcm_audio_stream(out_container, audio_sr, audio_layout) if has_audio else None
            audio_pts = 0
            audio_prev_end = 0

            total_frames = in_stream.frames if in_stream.frames else None
            pbar = tqdm(total=total_frames, desc=f"{mode} frames", unit="frame")

            dump_f = None
            if mode == "encrypt" and cipher_dump_path:
                dump_f = open(cipher_dump_path, "wb")
                print(f"[INFO] Cipher dump enabled: {cipher_dump_path}")

            frame_idx = 0

            try:
                for av_frame in in_container.decode(in_stream):
                    t0 = time.perf_counter()

                    frame = av_frame.to_ndarray(format="bgr24")
                    h, w = frame.shape[:2]

                    # Keep same timing columns as your previous experiments.
                    t1 = t0

                    t_enc0 = time.perf_counter()
                    video_iv = derive_ctr_iv(
                        video_key,
                        kind="video",
                        frame_idx=frame_idx,
                        extra=f"w={w}|h={h}|fmt=bgr24",
                    )
                    frame_out = aes256_ctr_crypt_frame(frame, video_key, video_iv)
                    if dump_f is not None:
                        dump_f.write(np.ascontiguousarray(frame_out).reshape(-1).tobytes())
                    t_enc1 = time.perf_counter()

                    t_write0 = time.perf_counter()
                    out_frame = av.VideoFrame.from_ndarray(frame_out, format="bgr24")
                    for packet in out_stream.encode(out_frame):
                        out_container.mux(packet)

                    if has_audio:
                        start = int(round(frame_idx * audio_sr / fps))
                        end = int(round((frame_idx + 1) * audio_sr / fps))

                        start = max(start, audio_prev_end)
                        end = max(end, start)
                        end = min(end, int(pcm_all.shape[0]))

                        if end > start:
                            seg = pcm_all[start:end, :]
                            audio_iv = derive_ctr_iv(
                                audio_key,
                                kind="audio",
                                frame_idx=frame_idx,
                                extra=f"samples={int(seg.shape[0])}|channels={int(seg.shape[1])}",
                            )
                            seg_out = aes256_ctr_crypt_audio_segment(seg.copy(), audio_key, audio_iv)
                            audio_pts = _mux_pcm_segment(
                                out_container,
                                aout,
                                seg_out,
                                audio_sr,
                                audio_layout,
                                audio_pts,
                            )
                        audio_prev_end = end

                    t_write1 = time.perf_counter()

                    if frame_idx % 30 == 0:
                        print(
                            f"frame={frame_idx} "
                            f"yolo={(t1 - t0):.3f}s "
                            f"crypt={(t_enc1 - t_enc0):.3f}s "
                            f"write={(t_write1 - t_write0):.3f}s "
                            f"rois=0 map=aes256ctr_full"
                        )

                    frame_idx += 1
                    pbar.update(1)

                if has_audio:
                    if audio_prev_end < int(pcm_all.shape[0]):
                        seg = pcm_all[audio_prev_end:, :]
                        last_fi = max(frame_idx - 1, 0)
                        audio_iv = derive_ctr_iv(
                            audio_key,
                            kind="audio_tail",
                            frame_idx=last_fi,
                            extra=f"samples={int(seg.shape[0])}|channels={int(seg.shape[1])}",
                        )
                        seg_out = aes256_ctr_crypt_audio_segment(seg.copy(), audio_key, audio_iv)
                        audio_pts = _mux_pcm_segment(out_container, aout, seg_out, audio_sr, audio_layout, audio_pts)

                    for pkt in aout.encode():
                        out_container.mux(pkt)

                for packet in out_stream.encode():
                    out_container.mux(packet)
            finally:
                pbar.close()
                if dump_f is not None:
                    dump_f.close()
        finally:
            out_container.close()
    finally:
        in_container.close()


# -----------------------------
# Convenience wrapper for local terminal tests
# -----------------------------

def process_video_full_aes_ctr_from_password(
    in_path: str,
    out_path: str,
    master_key: str,
    mode: str,
    cipher_dump_path: Optional[str] = None,
):
    master_key_bytes = derive_master_key_from_password(master_key)
    process_video_full_aes_ctr(
        in_path=in_path,
        out_path=out_path,
        master_key_bytes=master_key_bytes,
        mode=mode,
        cipher_dump_path=cipher_dump_path,
    )


# -----------------------------
# CLI
# -----------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["encrypt", "decrypt"], required=True)
    ap.add_argument("--in", dest="inp", required=True, help="input video path")
    ap.add_argument("--out", required=True, help="output video path")
    key_group = ap.add_mutually_exclusive_group(required=True)
    key_group.add_argument("--key", help="master key as string (SHA-256 derived locally)")
    key_group.add_argument("--key-b64", help="raw 32-byte master key as base64")
    ap.add_argument("--cipher_dump", default=None, help="(encrypt only) optional path to dump encrypted bytes")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.inp)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"Resolution: {w} x {h}")
    print(f"FPS: {fps}")

    print("[INFO] Full encryption baseline: AES-256-CTR")
    print("[INFO] Video: full-frame AES-256-CTR on bgr24 decoded frames")
    print("[INFO] Audio: full-segment AES-256-CTR on s16 PCM")
    print("[INFO] Writer: libx264rgb + qp=0, audio=FLAC")

    if args.key_b64:
        key_bytes = base64.b64decode(args.key_b64.encode("ascii"))
        if len(key_bytes) != 32:
            raise ValueError("--key-b64 must decode to exactly 32 bytes")
        process_video_full_aes_ctr(
            in_path=args.inp,
            out_path=args.out,
            master_key_bytes=key_bytes,
            mode=args.mode,
            cipher_dump_path=args.cipher_dump,
        )
    else:
        process_video_full_aes_ctr_from_password(
            in_path=args.inp,
            out_path=args.out,
            master_key=args.key,
            mode=args.mode,
            cipher_dump_path=args.cipher_dump,
        )

    print(f"[INFO] Done. Wrote: {args.out}")