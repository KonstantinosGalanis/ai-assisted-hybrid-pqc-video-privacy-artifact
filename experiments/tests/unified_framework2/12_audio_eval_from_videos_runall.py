#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
from scipy import signal
from skimage.metrics import structural_similarity as ssim


def ffprobe_audio_info(path: str) -> Dict[str, Any]:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_streams", "-select_streams", "a:0", "-of", "json", path,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, check=True)
    obj = json.loads(p.stdout)
    streams = obj.get("streams", [])
    if not streams:
        return {"has_audio": False}
    s = streams[0]
    return {
        "has_audio": True,
        "codec_name": s.get("codec_name"),
        "sample_rate": int(s.get("sample_rate", 0) or 0),
        "channels": int(s.get("channels", 0) or 0),
        "duration": float(s.get("duration", 0.0) or 0.0),
    }


def extract_audio_to_wav(video_path: str, wav_path: str, sample_rate: int = 48000, channels: int = 1) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", str(channels), "-ar", str(sample_rate),
        "-c:a", "pcm_s16le", wav_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def load_audio_mono(path: str, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    y, s = librosa.load(path, sr=sr, mono=True)
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        raise ValueError(f"No audio samples loaded from {path}")
    return y, int(s)


def align_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = min(a.size, b.size)
    if n <= 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    return a[:n], b[:n]


def to_int16_pcm(y: np.ndarray) -> np.ndarray:
    y = np.clip(y, -1.0, 1.0)
    return np.round(y * 32767.0).astype(np.int16)


def shannon_entropy_bytes(data: bytes) -> float:
    if not data:
        return float("nan")
    arr = np.frombuffer(data, dtype=np.uint8)
    hist = np.bincount(arr, minlength=256).astype(np.float64)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def adjacent_sample_corr(y: np.ndarray) -> float:
    if y.size < 3:
        return float("nan")
    x1 = y[:-1]
    x2 = y[1:]
    sx1 = float(np.std(x1))
    sx2 = float(np.std(x2))
    if sx1 == 0.0 or sx2 == 0.0:
        return float("nan")
    return float(np.corrcoef(x1, x2)[0, 1])


def npcr_uaci_bytes(a_bytes: bytes, b_bytes: bytes) -> Tuple[float, float]:
    a = np.frombuffer(a_bytes, dtype=np.uint8)
    b = np.frombuffer(b_bytes, dtype=np.uint8)
    n = min(a.size, b.size)
    if n <= 0:
        return float("nan"), float("nan")
    a = a[:n]
    b = b[:n]
    npcr = float(100.0 * np.mean(a != b))
    uaci = float(100.0 * np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64)) / 255.0))
    return npcr, uaci


def signal_to_noise_ratio(ref: np.ndarray, est: np.ndarray) -> float:
    ref, est = align_pair(ref, est)
    if ref.size == 0:
        return float("nan")
    noise = ref - est
    denom = np.sum(noise ** 2)
    if denom <= 0:
        return float("inf")
    num = np.sum(ref ** 2)
    return float(10.0 * np.log10((num + 1e-12) / (denom + 1e-12)))


def si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    ref, est = align_pair(ref, est)
    if ref.size == 0:
        return float("nan")
    ref_zm = ref - np.mean(ref)
    est_zm = est - np.mean(est)
    denom = np.dot(ref_zm, ref_zm)
    if denom <= 0:
        return float("nan")
    alpha = np.dot(est_zm, ref_zm) / denom
    target = alpha * ref_zm
    noise = est_zm - target
    num = np.sum(target ** 2)
    den = np.sum(noise ** 2)
    return float(10.0 * np.log10((num + 1e-12) / (den + 1e-12)))


def sdr(ref: np.ndarray, est: np.ndarray) -> float:
    ref, est = align_pair(ref, est)
    if ref.size == 0:
        return float("nan")
    noise = ref - est
    num = np.sum(ref ** 2)
    den = np.sum(noise ** 2)
    return float(10.0 * np.log10((num + 1e-12) / (den + 1e-12)))


def log_spectral_distance(ref: np.ndarray, est: np.ndarray, sr: int, n_fft: int = 1024, hop_length: int = 256) -> float:
    ref, est = align_pair(ref, est)
    if ref.size == 0:
        return float("nan")
    S1 = np.abs(librosa.stft(ref, n_fft=n_fft, hop_length=hop_length)) + 1e-10
    S2 = np.abs(librosa.stft(est, n_fft=n_fft, hop_length=hop_length)) + 1e-10
    n = min(S1.shape[1], S2.shape[1])
    S1 = S1[:, :n]
    S2 = S2[:, :n]
    l1 = 20.0 * np.log10(S1)
    l2 = 20.0 * np.log10(S2)
    d = np.sqrt(np.mean((l1 - l2) ** 2, axis=0))
    return float(np.mean(d))


def spectrogram_features(y: np.ndarray, sr: int, n_fft: int = 1024, hop_length: int = 256) -> np.ndarray:
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    L = librosa.amplitude_to_db(S + 1e-10, ref=np.max)
    return np.asarray(L, dtype=np.float64)


def compare_spectrograms(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    if h <= 0 or w <= 0:
        return {"spectrogram_corr": float("nan"), "spectrogram_ssim": float("nan"), "spectrogram_npcr": float("nan"), "spectrogram_uaci": float("nan")}
    a = a[:h, :w]
    b = b[:h, :w]
    af = a.reshape(-1)
    bf = b.reshape(-1)
    if np.std(af) == 0 or np.std(bf) == 0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(af, bf)[0, 1])
    amin = min(float(np.min(a)), float(np.min(b)))
    amax = max(float(np.max(a)), float(np.max(b)))
    dr = max(1e-9, amax - amin)
    ssim_val = float(ssim(a, b, data_range=dr))
    # normalize to uint8 for NPCR/UACI-style image comparison
    an = np.clip(np.round((a - amin) * 255.0 / dr), 0, 255).astype(np.uint8)
    bn = np.clip(np.round((b - amin) * 255.0 / dr), 0, 255).astype(np.uint8)
    diff = an != bn
    npcr = float(100.0 * np.mean(diff))
    uaci = float(100.0 * np.mean(np.abs(an.astype(np.float64) - bn.astype(np.float64)) / 255.0))
    return {
        "spectrogram_corr": corr,
        "spectrogram_ssim": ssim_val,
        "spectrogram_npcr": npcr,
        "spectrogram_uaci": uaci,
    }


def try_metric_stoi(ref: np.ndarray, est: np.ndarray, sr: int) -> Tuple[Optional[float], str]:
    try:
        pystoi = importlib.import_module("pystoi")
        ref16 = librosa.resample(ref, orig_sr=sr, target_sr=10000)
        est16 = librosa.resample(est, orig_sr=sr, target_sr=10000)
        ref16, est16 = align_pair(ref16, est16)
        return float(pystoi.stoi(ref16, est16, 10000, extended=False)), "ok"
    except Exception as e:
        return None, f"unavailable: {type(e).__name__}: {e}"


def try_metric_pesq(ref: np.ndarray, est: np.ndarray, sr: int) -> Tuple[Optional[float], str]:
    try:
        pesq_mod = importlib.import_module("pesq")
        target_sr = 16000 if sr >= 16000 else 8000
        ref_r = librosa.resample(ref, orig_sr=sr, target_sr=target_sr)
        est_r = librosa.resample(est, orig_sr=sr, target_sr=target_sr)
        ref_r, est_r = align_pair(ref_r, est_r)
        mode = "wb" if target_sr == 16000 else "nb"
        score = pesq_mod.pesq(target_sr, ref_r, est_r, mode)
        return float(score), "ok"
    except Exception as e:
        return None, f"unavailable: {type(e).__name__}: {e}"


def try_asr_whisper(audio_path: str) -> Tuple[Optional[str], str]:
    # module backends first
    try:
        fw = importlib.import_module("faster_whisper")
        model = fw.WhisperModel("small", device="cpu", compute_type="int8")
        segments, info = model.transcribe(audio_path, vad_filter=True)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text, f"ok:faster_whisper:{getattr(info, 'language', 'auto')}"
    except Exception:
        pass
    try:
        whisper = importlib.import_module("whisper")
        model = whisper.load_model("small")
        result = model.transcribe(audio_path)
        return str(result.get("text", "")).strip(), f"ok:openai_whisper:{result.get('language', 'auto')}"
    except Exception:
        pass
    return None, "unavailable: no whisper backend installed"


def word_error_rate(ref_text: str, hyp_text: str) -> float:
    ref = [w for w in ref_text.strip().lower().split() if w]
    hyp = [w for w in hyp_text.strip().lower().split() if w]
    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0
    dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(hyp) + 1):
        dp[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return float(dp[-1][-1] / max(1, len(ref)))


def summarize_wave(y: np.ndarray) -> Dict[str, float]:
    pcm = to_int16_pcm(y)
    b = pcm.tobytes()
    return {
        "duration_s": float(y.size),
        "byte_entropy": shannon_entropy_bytes(b),
        "adjacent_sample_corr": adjacent_sample_corr(y),
        "rms": float(np.sqrt(np.mean(y ** 2))),
        "mean": float(np.mean(y)),
        "std": float(np.std(y)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audio evaluation from plain/encrypted/decrypted videos")
    ap.add_argument("--plain_video", required=True)
    ap.add_argument("--encrypted_video", required=False, default=None)
    ap.add_argument("--cipher_video", required=False, default=None, help="Alias for --encrypted_video when called from unified runners")
    ap.add_argument("--decrypted_video", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--extract_sr", type=int, default=48000)
    ap.add_argument("--extract_channels", type=int, default=1)
    args = ap.parse_args()

    if args.encrypted_video is None:
        args.encrypted_video = args.cipher_video
    if args.encrypted_video is None:
        raise SystemExit("Need --encrypted_video or --cipher_video")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "inputs": {
            "plain_video": args.plain_video,
            "encrypted_video": args.encrypted_video,
            "decrypted_video": args.decrypted_video,
        },
        "stream_info": {},
        "audio_extraction": {},
        "availability": {},
        "metrics": {},
    }

    with tempfile.TemporaryDirectory(prefix="audio_eval_") as td:
        td_path = Path(td)
        wav_plain = str(td_path / "plain.wav")
        wav_enc = str(td_path / "encrypted.wav")
        wav_dec = str(td_path / "decrypted.wav")

        for key, path in [("plain", args.plain_video), ("encrypted", args.encrypted_video), ("decrypted", args.decrypted_video)]:
            report["stream_info"][key] = ffprobe_audio_info(path)
            if not report["stream_info"][key].get("has_audio"):
                raise RuntimeError(f"No audio stream found in {path}")

        extract_audio_to_wav(args.plain_video, wav_plain, sample_rate=args.extract_sr, channels=args.extract_channels)
        extract_audio_to_wav(args.encrypted_video, wav_enc, sample_rate=args.extract_sr, channels=args.extract_channels)
        extract_audio_to_wav(args.decrypted_video, wav_dec, sample_rate=args.extract_sr, channels=args.extract_channels)
        report["audio_extraction"] = {
            "sample_rate": args.extract_sr,
            "channels": args.extract_channels,
        }

        yp, sr = load_audio_mono(wav_plain, sr=None)
        ye, se = load_audio_mono(wav_enc, sr=None)
        yd, sd = load_audio_mono(wav_dec, sr=None)
        if not (sr == se == sd):
            raise RuntimeError(f"Resampled SR mismatch: {sr}, {se}, {sd}")

        report["metrics"]["plain_wave"] = summarize_wave(yp)
        report["metrics"]["encrypted_wave"] = summarize_wave(ye)
        report["metrics"]["decrypted_wave"] = summarize_wave(yd)
        for k in ["plain_wave", "encrypted_wave", "decrypted_wave"]:
            report["metrics"][k]["duration_s"] /= sr

        # secrecy/leakage metrics
        pcm_p = to_int16_pcm(yp)
        pcm_e = to_int16_pcm(ye)
        pcm_d = to_int16_pcm(yd)
        b_p = pcm_p.tobytes()
        b_e = pcm_e.tobytes()
        b_d = pcm_d.tobytes()

        npcr_pe, uaci_pe = npcr_uaci_bytes(b_p, b_e)
        npcr_pd, uaci_pd = npcr_uaci_bytes(b_p, b_d)
        report["metrics"]["sample_domain"] = {
            "plain_vs_encrypted_npcr": npcr_pe,
            "plain_vs_encrypted_uaci": uaci_pe,
            "plain_vs_decrypted_npcr": npcr_pd,
            "plain_vs_decrypted_uaci": uaci_pd,
        }

        spec_p = spectrogram_features(yp, sr)
        spec_e = spectrogram_features(ye, sr)
        spec_d = spectrogram_features(yd, sr)
        report["metrics"]["spectrogram_plain_vs_encrypted"] = compare_spectrograms(spec_p, spec_e)
        report["metrics"]["spectrogram_plain_vs_decrypted"] = compare_spectrograms(spec_p, spec_d)

        report["metrics"]["recovery_quality"] = {
            "snr_db": signal_to_noise_ratio(yp, yd),
            "sdr_db": sdr(yp, yd),
            "si_sdr_db": si_sdr(yp, yd),
            "lsd_db": log_spectral_distance(yp, yd, sr),
        }

        stoi_val, stoi_status = try_metric_stoi(yp, yd, sr)
        pesq_val, pesq_status = try_metric_pesq(yp, yd, sr)
        report["availability"]["stoi"] = stoi_status
        report["availability"]["pesq"] = pesq_status
        if stoi_val is not None:
            report["metrics"]["recovery_quality"]["stoi"] = stoi_val
        if pesq_val is not None:
            report["metrics"]["recovery_quality"]["pesq"] = pesq_val

        # ASR leakage (optional)
        text_p, status_p = try_asr_whisper(wav_plain)
        text_e, status_e = try_asr_whisper(wav_enc)
        text_d, status_d = try_asr_whisper(wav_dec)
        report["availability"]["asr_plain"] = status_p
        report["availability"]["asr_encrypted"] = status_e
        report["availability"]["asr_decrypted"] = status_d
        asr_block: Dict[str, Any] = {}
        if text_p is not None:
            asr_block["plain_text"] = text_p
        if text_e is not None:
            asr_block["encrypted_text"] = text_e
        if text_d is not None:
            asr_block["decrypted_text"] = text_d
        if text_p is not None and text_e is not None:
            asr_block["wer_plain_vs_encrypted"] = word_error_rate(text_p, text_e)
        if text_p is not None and text_d is not None:
            asr_block["wer_plain_vs_decrypted"] = word_error_rate(text_p, text_d)
        report["metrics"]["asr_leakage"] = asr_block

        # write extracted wavs for inspection
        shutil.copy2(wav_plain, out_dir / "plain_audio.wav")
        shutil.copy2(wav_enc, out_dir / "encrypted_audio.wav")
        shutil.copy2(wav_dec, out_dir / "decrypted_audio.wav")

    (out_dir / "audio_eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "report": str(out_dir / 'audio_eval_report.json')}))


if __name__ == "__main__":
    main()
