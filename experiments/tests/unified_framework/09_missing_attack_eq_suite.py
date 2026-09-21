#!/usr/bin/env python3
from __future__ import annotations

"""
Chaos-sidecar-aware attack suite for test 9.

Keeps thesis headings:
- Ciphertext-only (COA)
- Known-plaintext (KPA)
- Chosen-plaintext (CPA)
- Chosen-ciphertext (CCA)
- Encryption Quality (EQ)
- Pixel resemblance / disparity

Runner-compatible: keeps the thesis attack headings and accepts only the attack-suite CLI.
"""

import argparse
import importlib.util
import inspect
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}


def _stable_import_from_path(path: str, canonical_name: str = "framework_faster"):
    import sys
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


def import_framework(path: str):
    return _stable_import_from_path(path, "framework_faster")


def is_video(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


def read_frames(path: str, max_frames: int = 25) -> Tuple[List[np.ndarray], float]:
    if is_video(path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        frames: List[np.ndarray] = []

        while True:
            ok, fr = cap.read()
            if not ok:
                break

            frames.append(fr)

            if max_frames and len(frames) >= max_frames:
                break

        cap.release()
        return frames, fps

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return [img], 1.0


def save_absdiff_examples(out_dir: Path, ref_frames: List[np.ndarray], test_frames: List[np.ndarray], prefix: str, max_examples: int = 3):
    ex_dir = out_dir / "examples"
    ex_dir.mkdir(parents=True, exist_ok=True)
    for i, (a, b) in enumerate(zip(ref_frames, test_frames)):
        if i >= max_examples:
            break
        diff = cv2.absdiff(a, b)
        cv2.imwrite(str(ex_dir / f"{prefix}_absdiff_{i:04d}.png"), diff)


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


def combined_full_mask(record: dict, H: int, W: int, unpack_fn) -> np.ndarray:
    full = np.zeros((H, W), dtype=bool)
    for r in record.get("rois", []) or []:
        x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
        x1 = max(0, min(W, x1)); x2 = max(0, min(W, x2))
        y1 = max(0, min(H, y1)); y2 = max(0, min(H, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        mh = y2 - y1
        mw = x2 - x1
        full[y1:y2, x1:x2] |= _call_unpack_mask(unpack_fn, r["mask_pack"], mh, mw)
    return full


def _compute_mask_hash(mask_bool: np.ndarray) -> str:
    packed = np.packbits(mask_bool.reshape(-1).astype(np.uint8))
    return hashlib.blake2b(packed.tobytes(), digest_size=8).hexdigest()


def mse(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float64, copy=False) - b.astype(np.float64, copy=False)
    return float(np.mean(d * d))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float64, copy=False) - b.astype(np.float64, copy=False)
    return float(np.mean(np.abs(d)))


def psnr_from_mse(m: float, peak: float = 255.0) -> float:
    if m <= 0:
        return float("inf")
    return float(10.0 * np.log10((peak * peak) / m))


def to_gray_u8(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        return img_bgr.astype(np.uint8, copy=False)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def ssim_windowed(gray1: np.ndarray, gray2: np.ndarray, L: float = 255.0) -> float:
    I1 = gray1.astype(np.float32)
    I2 = gray2.astype(np.float32)
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    mu1 = cv2.GaussianBlur(I1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(I2, (11, 11), 1.5)
    mu1_2 = mu1 * mu1
    mu2_2 = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_2 = cv2.GaussianBlur(I1 * I1, (11, 11), 1.5) - mu1_2
    sigma2_2 = cv2.GaussianBlur(I2 * I2, (11, 11), 1.5) - mu2_2
    sigma12 = cv2.GaussianBlur(I1 * I2, (11, 11), 1.5) - mu1_mu2
    num = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)
    den = (mu1_2 + mu2_2 + C1) * (sigma1_2 + sigma2_2 + C2)
    ssim_map = num / np.maximum(den, 1e-12)
    return float(np.mean(ssim_map))


def evaluate_pair(ref: np.ndarray, test: np.ndarray, mask: Optional[np.ndarray]) -> Dict[str, float]:
    if ref.shape != test.shape:
        test = cv2.resize(test, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
    if mask is None:
        a = ref
        b = test
    else:
        if not np.any(mask):
            return {}
        a = ref[mask]
        b = test[mask]
    m = mse(a, b)
    return {
        "mse": m,
        "mae": mae(a, b),
        "psnr": psnr_from_mse(m),
        "ssim_gray_windowed": ssim_windowed(to_gray_u8(ref), to_gray_u8(test)),
        "exact_match_rate_percent": float(100.0 * np.mean((a.astype(np.uint8) == b.astype(np.uint8)).astype(np.float64))),
    }


def summarize_metric_rows(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({k for r in rows for k in r.keys()})
    out = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r and np.isfinite(r[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def encryption_quality_single(plain: np.ndarray, cipher: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    pg = to_gray_u8(plain)
    cg = to_gray_u8(cipher)
    hp = cv2.calcHist([pg], [0], None, [256], [0, 256]).reshape(-1)
    hc = cv2.calcHist([cg], [0], None, [256], [0, 256]).reshape(-1)
    out["eq_gray"] = float(np.mean(np.abs(hc - hp)))
    if plain.ndim == 3 and cipher.ndim == 3 and plain.shape[2] >= 3 and cipher.shape[2] >= 3:
        vals = []
        for ci, nm in enumerate(("b", "g", "r")):
            hp = cv2.calcHist([plain[:, :, ci]], [0], None, [256], [0, 256]).reshape(-1)
            hc = cv2.calcHist([cipher[:, :, ci]], [0], None, [256], [0, 256]).reshape(-1)
            v = float(np.mean(np.abs(hc - hp)))
            out[f"eq_{nm}"] = v
            vals.append(v)
        out["eq_color_mean"] = float(np.mean(vals))
    return out


def _derive_seed_for_roi(ff, master_key: str, frame_idx: int, track_id: int, cls_id: int,
                         map_name: str, policy_tag: str, bbox: Tuple[int, int, int, int], mask_hash: str) -> bytes:
    if hasattr(ff, "derive_policy_seed"):
        sig = inspect.signature(ff.derive_policy_seed)
        kwargs = {
            "master_key": master_key,
            "frame_idx": frame_idx,
            "track_id": track_id,
            "cls_id": cls_id,
            "map_name": map_name,
            "policy_tag": policy_tag,
        }
        if "scope" in sig.parameters and hasattr(ff, "POLICY_SCOPE"):
            kwargs["scope"] = ff.POLICY_SCOPE
        if "bbox" in sig.parameters:
            kwargs["bbox"] = bbox
        if "mask_hash" in sig.parameters:
            kwargs["mask_hash"] = mask_hash
        return ff.derive_policy_seed(**kwargs)
    sig = inspect.signature(ff.derive_seed)
    kwargs = {
        "master_key": master_key,
        "frame_idx": frame_idx,
        "track_id": track_id,
        "cls_id": cls_id,
        "map_name": map_name,
    }
    if "bbox" in sig.parameters:
        kwargs["bbox"] = bbox
    if "mask_hash" in sig.parameters:
        kwargs["mask_hash"] = mask_hash
    return ff.derive_seed(**kwargs)


def _call_with_supported_kwargs(func, *args, **kwargs):
    sig = inspect.signature(func)
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return func(*args, **accepted)


def oracle_apply_frames(
    frames: List[np.ndarray],
    recs: Dict[int, dict],
    ff,
    master_key: str,
    mode: str,
    reuse_crypto_meta_for_encrypt: bool = False,
) -> List[np.ndarray]:
    out_frames = [fr.copy() for fr in frames]
    enc_helper = getattr(ff, "_encrypt_masked_bytes_cpu_chaos", None)
    if enc_helper is None:
        enc_helper = getattr(ff, "_encrypt_masked_bytes_cpu_aead", None)
    dec_helper = getattr(ff, "_decrypt_masked_bytes_cpu_chaos", None)
    if dec_helper is None:
        dec_helper = getattr(ff, "_decrypt_masked_bytes_cpu_aead", None)
    for frame_idx, frame in enumerate(out_frames):
        rec = recs.get(frame_idx, None)
        if rec is None:
            continue
        map_name = str(rec.get("map_name", "cubic"))
        for roi_index, r in enumerate(rec.get("rois", []) or []):
            x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
            x1 = max(0, min(frame.shape[1], x1)); x2 = max(0, min(frame.shape[1], x2))
            y1 = max(0, min(frame.shape[0], y1)); y2 = max(0, min(frame.shape[0], y2))
            if x2 <= x1 or y2 <= y1:
                continue
            roi = frame[y1:y2, x1:x2]
            mh, mw = roi.shape[:2]
            mask = _call_unpack_mask(ff.unpack_mask, r["mask_pack"], mh, mw)
            if not np.any(mask):
                continue
            track_id = int(r.get("track_id", -1))
            cls_id = int(r.get("cls", -1))
            policy_tag = str(r.get("policy_tag", "default_sensitive"))
            mask_hash = str(r.get("mask_hash", _compute_mask_hash(mask)))
            seed = _derive_seed_for_roi(
                ff, master_key, frame_idx, track_id, cls_id,
                map_name, policy_tag, (x1, y1, x2, y2), mask_hash,
            )
            common_kwargs = dict(
                frame_idx=frame_idx,
                track_id=track_id,
                cls_id=cls_id,
                policy_tag=policy_tag,
                bbox=(x1, y1, x2, y2),
                mask_hash=mask_hash,
                roi_index=int(r.get("roi_index", roi_index)),
            )
            if mode == "encrypt":
                if enc_helper is not None:
                    try:
                        roi_out, _ = _call_with_supported_kwargs(
                            enc_helper,
                            roi.copy(), mask, seed, map_name,
                            crypto_meta=(r.get("crypto_meta") if reuse_crypto_meta_for_encrypt else None),
                            **common_kwargs,
                        )
                        frame[y1:y2, x1:x2] = roi_out
                    except Exception as e:
                        raise RuntimeError(
                            f"Chaos encrypt oracle failed at frame={frame_idx} roi={roi_index}; "
                            f"refusing silent fallback to legacy deterministic path"
                        ) from e
                else:
                    ff.encrypt_masked_bytes_cpu(roi, mask, seed, map_name)
            elif mode == "decrypt":
                if dec_helper is not None and r.get("crypto_meta") is not None:
                    try:
                        roi_out = _call_with_supported_kwargs(
                            dec_helper,
                            roi.copy(), mask, seed, map_name,
                            crypto_meta=r.get("crypto_meta"),
                            **common_kwargs,
                        )
                        frame[y1:y2, x1:x2] = roi_out
                    except Exception as e:
                        raise RuntimeError(
                            f"Chaos decrypt oracle failed at frame={frame_idx} roi={roi_index}; "
                            f"refusing silent fallback to legacy deterministic path"
                        ) from e
                else:
                    ff.decrypt_masked_bytes_cpu(roi, mask, seed, map_name)
            else:
                raise ValueError(mode)
    return out_frames


def ciphertext_only_analysis(plain_frames, cipher_frames, recs, ff, out_dir: Path) -> Dict[str, object]:
    rows = []
    for i, (p, c) in enumerate(zip(plain_frames, cipher_frames)):
        rec = recs.get(i, {"rois": []})
        mask = combined_full_mask(rec, p.shape[0], p.shape[1], ff.unpack_mask)
        whole = evaluate_pair(p, c, None)
        roi = evaluate_pair(p, c, mask)
        row = {f"whole_{k}": v for k, v in whole.items()}
        row.update({f"roi_{k}": v for k, v in roi.items()})
        rows.append(row)
    save_absdiff_examples(out_dir, plain_frames, cipher_frames, "coa")
    return {"metrics_mean": summarize_metric_rows(rows), "notes": ["Ciphertext-only concealment/leakage proxy."]}


def known_plaintext_attack(plain_frames, cipher_frames, recs, ff, out_dir: Path) -> Dict[str, object]:
    if len(plain_frames) < 2:
        return {"notes": ["Need at least two frames."]}
    ref_plain = plain_frames[0]
    ref_cipher = cipher_frames[0]
    rows = []
    recovers = []
    for i in range(1, len(cipher_frames)):
        rec = recs.get(i, {"rois": []})
        mask = combined_full_mask(rec, cipher_frames[i].shape[0], cipher_frames[i].shape[1], ff.unpack_mask)
        cand = cipher_frames[i].copy()
        if np.any(mask):
            cand[mask] = np.bitwise_xor(cipher_frames[i][mask], np.bitwise_xor(ref_plain[mask], ref_cipher[mask]))
        recovers.append(cand)
        whole = evaluate_pair(plain_frames[i], cand, None)
        roi = evaluate_pair(plain_frames[i], cand, mask)
        row = {f"whole_{k}": v for k, v in whole.items()}
        row.update({f"roi_{k}": v for k, v in roi.items()})
        rows.append(row)
    save_absdiff_examples(out_dir, plain_frames[1:], recovers, "kpa")
    return {"metrics_mean": summarize_metric_rows(rows), "notes": ["Known-plaintext generic transfer baseline; poor ROI recovery is desirable."]}


def chosen_plaintext_attack(plain_frames, cipher_frames, recs, ff, master_key, out_dir: Path) -> Dict[str, object]:
    zero_frames = [np.zeros_like(fr) for fr in plain_frames]
    zero_cipher_1 = oracle_apply_frames(
        zero_frames,
        recs,
        ff,
        master_key,
        mode="encrypt",
        reuse_crypto_meta_for_encrypt=False,
    )
    zero_cipher_2 = oracle_apply_frames(
        zero_frames,
        recs,
        ff,
        master_key,
        mode="encrypt",
        reuse_crypto_meta_for_encrypt=False,
    )
    det_rows, transfer_rows = [], []
    transfer_frames = []
    for i, c in enumerate(cipher_frames):
        rec = recs.get(i, {"rois": []})
        mask = combined_full_mask(rec, c.shape[0], c.shape[1], ff.unpack_mask)
        if np.any(mask):
            det_rows.append({
                "roi_repeat_exact_match_rate_percent": float(100.0 * np.mean((zero_cipher_1[i][mask] == zero_cipher_2[i][mask]).astype(np.float64))),
                "roi_repeat_npcr_percent": float(100.0 * np.mean((zero_cipher_1[i][mask] != zero_cipher_2[i][mask]).astype(np.float64)))
            })
        cand = c.copy()
        if np.any(mask):
            cand[mask] = np.bitwise_xor(c[mask], zero_cipher_1[i][mask])
        transfer_frames.append(cand)
        whole = evaluate_pair(plain_frames[i], cand, None)
        roi = evaluate_pair(plain_frames[i], cand, mask)
        row = {f"whole_{k}": v for k, v in whole.items()}
        row.update({f"roi_{k}": v for k, v in roi.items()})
        transfer_rows.append(row)
    save_absdiff_examples(out_dir, plain_frames, transfer_frames, "cpa")
    return {
        "determinism_mean": summarize_metric_rows(det_rows),
        "chosen_pair_transfer_mean": summarize_metric_rows(transfer_rows),
        "notes": [
            "Repeated chosen-plaintext queries use fresh per-encryption crypto metadata instead of reusing stored sidecar values.",
            "Chosen-pair transfer exact-match is also ROI-scoped to avoid background inflation."
        ]
    }


def _tamper_cipher_bitflip(cipher_frames, recs, ff):
    out = [fr.copy() for fr in cipher_frames]
    stats = {"tampered_frames": 0, "tampered_pixels": 0}
    for i, fr in enumerate(out):
        rec = recs.get(i)
        if rec is None:
            continue
        mask = combined_full_mask(rec, fr.shape[0], fr.shape[1], ff.unpack_mask)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        n = max(1, int(0.01 * len(xs)))
        idx = np.linspace(0, len(xs) - 1, num=n, dtype=int)
        stats["tampered_frames"] += 1
        stats["tampered_pixels"] += int(n)
        for j in idx:
            y, x = int(ys[j]), int(xs[j])
            fr[y, x, 0] ^= np.uint8(0x01)
    return out, stats


def _tamper_crypto_meta(recs: Dict[int, dict]) -> Tuple[Dict[int, dict], Dict[str, int]]:
    out = json.loads(json.dumps(recs))
    out = {int(k): v for k, v in out.items()}
    stats = {"rois_tampered": 0, "tags_tampered": 0, "nonce_tampered": 0, "chunk_size_tampered": 0,
             "mask_hash_tampered": 0, "bbox_tampered": 0, "track_id_tampered": 0}
    for fi, rec in out.items():
        for r in rec.get("rois", []) or []:
            stats["rois_tampered"] += 1
            cm = r.get("crypto_meta")
            if isinstance(cm, dict):
                if isinstance(cm.get("tags_b64"), str) and cm["tags_b64"]:
                    cm["tags_b64"] = "A" + cm["tags_b64"][1:]
                    stats["tags_tampered"] += 1
                if isinstance(cm.get("stream_nonce_b64"), str) and cm["stream_nonce_b64"]:
                    cm["stream_nonce_b64"] = "A" + cm["stream_nonce_b64"][1:]
                    stats["nonce_tampered"] += 1
                if "chunk_size" in cm:
                    cm["chunk_size"] = int(cm.get("chunk_size", 0)) + 1
                    stats["chunk_size_tampered"] += 1
            if isinstance(r.get("mask_hash"), str) and r["mask_hash"]:
                r["mask_hash"] = "f" + r["mask_hash"][1:]
                stats["mask_hash_tampered"] += 1
            if "bbox" in r and len(r["bbox"]) == 4:
                b = list(r["bbox"])
                b[0] = int(b[0]) + 1
                r["bbox"] = b
                stats["bbox_tampered"] += 1
            r["track_id"] = int(r.get("track_id", -1)) + 99991
            stats["track_id_tampered"] += 1
    return out, stats


def _swap_sidecar_roi_records(recs: Dict[int, dict]) -> Tuple[Dict[int, dict], Dict[str, object]]:
    out = json.loads(json.dumps(recs))
    out = {int(k): v for k, v in out.items()}
    stats = {"swap_performed": False, "frame_a": None, "frame_b": None, "rois_a": 0, "rois_b": 0}
    for a, b in zip(sorted(out.keys())[:-1], sorted(out.keys())[1:]):
        ra = out.get(a, {})
        rb = out.get(b, {})
        la = list(ra.get("rois", []) or [])
        lb = list(rb.get("rois", []) or [])
        if not la or not lb:
            continue
        ra["rois"], rb["rois"] = lb, la
        out[a], out[b] = ra, rb
        stats.update({"swap_performed": True, "frame_a": a, "frame_b": b, "rois_a": len(la), "rois_b": len(lb)})
        break
    return out, stats


def _tamper_disruption_rows(attacked_frames, decrypted_frames, recs, ff):
    rows = []
    for i, (att, dec) in enumerate(zip(attacked_frames, decrypted_frames)):
        rec = recs.get(i, {"rois": []})
        mask = combined_full_mask(rec, att.shape[0], att.shape[1], ff.unpack_mask)
        roi = evaluate_pair(att, dec, mask)
        rows.append({f"roi_{k}": v for k, v in roi.items()})
    return summarize_metric_rows(rows)


def chosen_ciphertext_attack(plain_frames, cipher_frames, recs, ff, master_key, out_dir: Path) -> Dict[str, object]:
    bitflip_cipher, bitflip_stats = _tamper_cipher_bitflip(cipher_frames, recs, ff)
    tampered_recs, meta_stats = _tamper_crypto_meta(recs)
    swapped_recs, swap_stats = _swap_sidecar_roi_records(recs)

    dec_bitflip = oracle_apply_frames(bitflip_cipher, recs, ff, master_key, mode="decrypt")
    dec_meta = oracle_apply_frames(cipher_frames, tampered_recs, ff, master_key, mode="decrypt")
    dec_swap = oracle_apply_frames(cipher_frames, swapped_recs, ff, master_key, mode="decrypt")

    rows_bitflip, rows_meta, rows_swap = [], [], []
    for i, p in enumerate(plain_frames):
        mask_norm = combined_full_mask(recs.get(i, {"rois": []}), p.shape[0], p.shape[1], ff.unpack_mask)
        mask_swap = combined_full_mask(swapped_recs.get(i, {"rois": []}), p.shape[0], p.shape[1], ff.unpack_mask)
        wb = evaluate_pair(p, dec_bitflip[i], None); rb = evaluate_pair(p, dec_bitflip[i], mask_norm)
        wm = evaluate_pair(p, dec_meta[i], None); rm = evaluate_pair(p, dec_meta[i], mask_norm)
        ws = evaluate_pair(p, dec_swap[i], None); rs = evaluate_pair(p, dec_swap[i], mask_swap)
        rows_bitflip.append({**{f"whole_{k}": v for k, v in wb.items()}, **{f"roi_{k}": v for k, v in rb.items()}})
        rows_meta.append({**{f"whole_{k}": v for k, v in wm.items()}, **{f"roi_{k}": v for k, v in rm.items()}})
        rows_swap.append({**{f"whole_{k}": v for k, v in ws.items()}, **{f"roi_{k}": v for k, v in rs.items()}})

    save_absdiff_examples(out_dir, plain_frames, dec_bitflip, "cca_bitflip")
    save_absdiff_examples(out_dir, plain_frames, dec_meta, "cca_meta")
    save_absdiff_examples(out_dir, plain_frames, dec_swap, "cca_swap")

    return {
        "bitflip_mean": summarize_metric_rows(rows_bitflip),
        "bitflip_tamper_disruption_mean": _tamper_disruption_rows(bitflip_cipher, dec_bitflip, recs, ff),
        "crypto_meta_tamper_mean": summarize_metric_rows(rows_meta),
        "crypto_meta_tamper_disruption_mean": _tamper_disruption_rows(cipher_frames, dec_meta, recs, ff),
        "roi_record_swap_mean": summarize_metric_rows(rows_swap),
        "roi_record_swap_tamper_disruption_mean": _tamper_disruption_rows(cipher_frames, dec_swap, swapped_recs, ff),
        "tamper_stats": {
            "bitflip": bitflip_stats,
            "crypto_meta": meta_stats,
            "roi_swap": swap_stats,
        },
        "notes": [
            "CCA is evaluated as tamper disruption and recovery corruption under chaos-sidecar dependence.",
            "Decrypt-vs-cipher proximity inside ROI is treated as a tamper-disruption proxy, not as accept/reject authentication semantics."
        ]
    }


def eq_and_pixel_disparity(plain_frames, cipher_frames, recs, ff, out_dir: Path) -> Dict[str, object]:
    eq_rows, disparity_rows = [], []
    for i, (p, c) in enumerate(zip(plain_frames, cipher_frames)):
        eq_rows.append(encryption_quality_single(p, c))
        rec = recs.get(i, {"rois": []})
        mask = combined_full_mask(rec, p.shape[0], p.shape[1], ff.unpack_mask)
        whole = evaluate_pair(p, c, None)
        roi = evaluate_pair(p, c, mask)
        row = {f"whole_{k}": v for k, v in whole.items()}
        row.update({f"roi_{k}": v for k, v in roi.items()})
        disparity_rows.append(row)
    save_absdiff_examples(out_dir, plain_frames, cipher_frames, "eqdisp")
    return {
        "encryption_quality_mean": summarize_metric_rows(eq_rows),
        "pixel_resemblance_disparity_mean": summarize_metric_rows(disparity_rows),
        "notes": ["EQ and disparity are statistical/concealment measures, separate from oracle resistance."]
    }


def main():
    ap = argparse.ArgumentParser(description="Chaos-sidecar-aware attack suite for test 9.")
    ap.add_argument("--framework_faster_path", required=True)
    ap.add_argument("--master_key", required=True)
    ap.add_argument("--plain", required=True)
    ap.add_argument("--cipher", required=True)
    ap.add_argument("--roi_sidecar", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", default="all")
    ap.add_argument("--cipher2", default=None)
    ap.add_argument("--plain2", default=None)
    ap.add_argument("--plain_ref", default=None)
    ap.add_argument("--cipher_ref", default=None)
    ap.add_argument("--max_frames", type=int, default=1000)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ff = import_framework(args.framework_faster_path)
    plain_frames, _ = read_frames(args.plain, args.max_frames)
    cipher_frames, _ = read_frames(args.cipher, args.max_frames)
    recs = load_sidecar_records(args.roi_sidecar)
    n = min(len(plain_frames), len(cipher_frames))
    plain_frames = plain_frames[:n]
    cipher_frames = cipher_frames[:n]

    report = {
        "suite": "Chaos-sidecar-aware attack suite",
        "framework": str(Path(args.framework_faster_path).resolve()),
        "frames": n,
        "ciphertext_only": ciphertext_only_analysis(plain_frames, cipher_frames, recs, ff, out_dir / "coa"),
        "known_plaintext": known_plaintext_attack(plain_frames, cipher_frames, recs, ff, out_dir / "kpa"),
        "chosen_plaintext": chosen_plaintext_attack(plain_frames, cipher_frames, recs, ff, args.master_key, out_dir / "cpa"),
        "chosen_ciphertext": chosen_ciphertext_attack(plain_frames, cipher_frames, recs, ff, args.master_key, out_dir / "cca"),
        "eq_and_pixel_disparity": eq_and_pixel_disparity(plain_frames, cipher_frames, recs, ff, out_dir / "eq_pixel"),
        "notes": [
            "Preserves thesis attack headings without relying on XOR-keystream recovery equations.",
            "Chosen-pair transfer and ROI-swap metrics are ROI-scoped to avoid background inflation and scope mismatches.",
            "Metadata tamper and ROI swap are paired with a decrypt-vs-cipher proxy to interpret disruption under chaos-sidecar dependence rather than AEAD-style rejection.",
            "Oracle path is strict: if chaos helper exists but raises, the suite fails instead of silently falling back to legacy deterministic functions."
        ]
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "out": str(out_dir / "report.json"), "frames": n}))


if __name__ == "__main__":
    main()
