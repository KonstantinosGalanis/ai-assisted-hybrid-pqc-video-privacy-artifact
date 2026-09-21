from __future__ import annotations

import hashlib
import inspect
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
BIT_EXTS = {".bin", ".bits", ".bit", ".dat", ".raw"}

def is_video(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS

def is_image(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS

def is_bitstream(path: str) -> bool:
    return Path(path).suffix.lower() in BIT_EXTS

def import_module_from_path(path: str, module_name: str = "loaded_module"):
    resolved = str(Path(path).resolve())
    existing = sys.modules.get(module_name)
    if existing is not None and str(Path(getattr(existing, "__file__", "")).resolve()) == resolved:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {resolved}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

def read_media(path: str) -> Tuple[List[np.ndarray], float]:
    path = str(path)
    if is_image(path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return [img], 1.0
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open media: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frames: List[np.ndarray] = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {path}")
    return frames, fps

def write_media(frames: List[np.ndarray], out_path: str, fps: float = 25.0) -> str:
    out_path = str(out_path)
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("No frames to write")
    if is_image(out_path):
        cv2.imwrite(out_path, frames[0])
        return out_path
    h, w = frames[0].shape[:2]
    if outp.suffix.lower() != '.mkv':
        outp = outp.with_suffix('.mkv')
        out_path = str(outp)
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    vw = cv2.VideoWriter(out_path, fourcc, fps if fps > 0 else 25.0, (w, h))
    if not vw.isOpened():
        out_path = str(outp.with_suffix('.mp4'))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(out_path, fourcc, fps if fps > 0 else 25.0, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"Could not open writer for {out_path}")
    for fr in frames:
        if fr.shape[:2] != (h, w):
            fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
        vw.write(fr)
    vw.release()
    return out_path

def load_sidecar_records(path: str) -> List[dict]:
    recs: List[dict] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs

def _mask_for_roi(ff, roi: dict, roi_shape: Tuple[int, int]) -> np.ndarray:
    mh, mw = roi_shape
    try:
        mask = ff.unpack_mask(roi['mask_pack'])
    except TypeError:
        mask = ff.unpack_mask(roi['mask_pack'], mh, mw)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (mh, mw):
        mask = cv2.resize(mask.astype(np.uint8), (mw, mh), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask

def _roi_mask_hash(mask: np.ndarray) -> str:
    packed = np.packbits(mask.reshape(-1).astype(np.uint8))
    return hashlib.blake2b(packed.tobytes(), digest_size=8).hexdigest()

def _derive_seed_for_roi(ff, master_key: str, frame_idx: int, track_id: int, cls_id: int, map_name: str,
                         bbox: Optional[Tuple[int, int, int, int]] = None,
                         mask: Optional[np.ndarray] = None) -> bytes:
    bbox_tuple = tuple(int(v) for v in bbox) if bbox is not None else None
    mask_hash = _roi_mask_hash(mask) if mask is not None and mask.size else None

    if hasattr(ff, 'derive_policy_seed') and hasattr(ff, 'policy_tag_for_class'):
        try:
            sig = inspect.signature(ff.derive_policy_seed)
            kwargs = {
                'master_key': master_key,
                'frame_idx': int(frame_idx),
                'track_id': int(track_id),
                'cls_id': int(cls_id),
                'map_name': map_name,
                'policy_tag': ff.policy_tag_for_class(int(cls_id)),
            }
            if hasattr(ff, 'POLICY_SCOPE'):
                kwargs['scope'] = getattr(ff, 'POLICY_SCOPE')
            if 'bbox' in sig.parameters and bbox_tuple is not None:
                kwargs['bbox'] = bbox_tuple
            if 'mask_hash' in sig.parameters and mask_hash is not None:
                kwargs['mask_hash'] = mask_hash
            return ff.derive_policy_seed(**kwargs)
        except Exception:
            pass

    if hasattr(ff, 'derive_seed'):
        try:
            sig = inspect.signature(ff.derive_seed)
            params = sig.parameters
            if 'bbox' in params or 'mask_hash' in params:
                return ff.derive_seed(
                    master_key, int(frame_idx), int(track_id), int(cls_id), map_name,
                    bbox=bbox_tuple, mask_hash=mask_hash
                )
        except Exception:
            pass
        return ff.derive_seed(master_key, int(frame_idx), int(track_id), int(cls_id), map_name)

    raise RuntimeError('framework module does not expose derive_seed/derive_policy_seed')

def _keystream_bytes_for_roi(ff, seed: bytes, map_name: str, nbytes: int) -> bytes:
    stream_seed = seed
    if hasattr(ff, 'derive_stream_seed'):
        try:
            stream_seed = ff.derive_stream_seed(seed, 'diff2')
        except Exception:
            try:
                stream_seed = ff.derive_stream_seed(seed, 'diff')
            except Exception:
                stream_seed = seed
    ks = ff.keystream_u8(stream_seed, map_name, nbytes)
    return np.asarray(ks, dtype=np.uint8).tobytes()

def encrypt_frames_with_sidecar(frames: List[np.ndarray], recs: List[dict], ff, master_key: str, force_map_name: Optional[str] = None, preview_mode: str = "chaos") -> List[np.ndarray]:
    out: List[np.ndarray] = []
    rec_by_idx = {int(r.get('frame_idx', i)): r for i, r in enumerate(recs)}
    base_ff = resolve_sidecar_module(ff)
    for idx, fr in enumerate(frames):
        rec = rec_by_idx.get(idx, {'frame_idx': idx, 'rois': [], 'map_name': force_map_name or 'cubic'})
        map_name = force_map_name or str(rec.get('map_name', 'cubic'))
        cur = fr.copy()
        h, w = cur.shape[:2]
        for r in rec.get('rois', []) or []:
            x1, y1, x2, y2 = [int(v) for v in r['bbox']]
            x1 = max(0, min(w, x1)); x2 = max(0, min(w, x2))
            y1 = max(0, min(h, y1)); y2 = max(0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            roi = cur[y1:y2, x1:x2]
            mask = _mask_for_roi(base_ff, r, roi.shape[:2])
            track_id = int(r.get('track_id', -1))
            cls_id = int(r.get('cls', -1))
            seed = _derive_seed_for_roi(base_ff, master_key, idx, track_id, cls_id, map_name, bbox=(x1, y1, x2, y2), mask=mask)
            if hasattr(ff, 'apply_public_video_preview'):
                roi_out = ff.apply_public_video_preview(base_ff, roi, mask, seed, map_name, preview_mode)
            else:
                roi_out = base_ff.encrypt_masked_bytes_cpu(roi, mask, seed, map_name)
            if roi_out is not roi:
                cur[y1:y2, x1:x2] = roi_out
        out.append(cur)
    return out

def derive_used_keystream_bytes(recs: List[dict], ff, master_key: str, force_map_name: Optional[str] = None) -> bytes:
    if module_supports_payload(ff):
        raise RuntimeError('Payload-aware framework: used ciphertext bytes are not derivable from roi_sidecar alone; use --keystream_dump output from encryption.')
    chunks: List[bytes] = []
    core_ff = resolve_sidecar_module(ff)
    for rec in recs:
        frame_idx = int(rec.get('frame_idx', 0))
        map_name = force_map_name or str(rec.get('map_name', 'cubic'))
        for r in rec.get('rois', []) or []:
            bbox = r.get('bbox')
            if bbox is None or 'mask_pack' not in r:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            mh = max(0, y2 - y1)
            mw = max(0, x2 - x1)
            if mh <= 0 or mw <= 0:
                continue
            mask = _mask_for_roi(core_ff, r, (mh, mw))
            nbytes = int(mask.sum()) * 3
            if nbytes <= 0:
                continue
            track_id = int(r.get('track_id', -1))
            cls_id = int(r.get('cls', -1))
            seed = _derive_seed_for_roi(core_ff, master_key, frame_idx, track_id, cls_id, map_name, bbox=(x1, y1, x2, y2), mask=mask)
            chunks.append(_keystream_bytes_for_roi(core_ff, seed, map_name, nbytes))
    return b''.join(chunks)

def write_bytes(path: str, data: bytes) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return str(p)

def derive_key_variant(master_key: str, size_chars: int) -> str:
    digest = hashlib.sha256(master_key.encode('utf-8')).hexdigest()
    return (digest * ((size_chars // len(digest)) + 2))[:size_chars]

def make_checkerboard_variant(frames: List[np.ndarray]) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for idx, fr in enumerate(frames):
        h, w = fr.shape[:2]
        yy, xx = np.indices((h, w))
        pat = (((xx // 16) + (yy // 16) + idx) % 2 * 255).astype(np.uint8)
        img = np.stack([pat, np.roll(pat, 4, axis=1), np.roll(pat, 8, axis=0)], axis=2)
        out.append(img)
    return out

def make_inverted_variant(frames: List[np.ndarray]) -> List[np.ndarray]:
    return [255 - fr for fr in frames]

def make_pixel_flip_variant(frames: List[np.ndarray]) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for idx, fr in enumerate(frames):
        cur = fr.copy()
        h, w = cur.shape[:2]
        y = (idx * 7 + h // 2) % h
        x = (idx * 13 + w // 2) % w
        cur[y, x, :] ^= np.uint8(0xFF)
        out.append(cur)
    return out


def module_supports_payload(ff) -> bool:
    fn = getattr(ff, "process_video", None)
    if not callable(fn):
        return False
    try:
        sig = inspect.signature(fn)
    except Exception:
        return False
    return "payload_path" in sig.parameters or "payload" in sig.parameters


def resolve_sidecar_module(ff_or_path, framework_base_path: Optional[str] = None):
    if isinstance(ff_or_path, str):
        ff = import_module_from_path(ff_or_path, "resolved_framework")
        ff_path = Path(ff_or_path).resolve()
    else:
        ff = ff_or_path
        ff_path = Path(getattr(ff, "__file__", "")).resolve() if getattr(ff, "__file__", None) else None
    if framework_base_path:
        return import_module_from_path(framework_base_path, "resolved_framework_base")
    if hasattr(ff, "unpack_mask") and (hasattr(ff, "derive_policy_seed") or hasattr(ff, "derive_seed")):
        return ff
    if ff_path is not None:
        for name in ("framework_succesful.py", "framework_faster.py"):
            cand = ff_path.with_name(name)
            if cand.exists():
                try:
                    return import_module_from_path(str(cand), f"resolved_base_{cand.stem}")
                except Exception:
                    pass
    return ff


def framework_cli_encrypt(framework_path: str, in_path: str, out_path: str, master_key: str,
                          roi_sidecar: Optional[str] = None, payload: Optional[str] = None,
                          manifest: Optional[str] = None,
                          keystream_dump: Optional[str] = None, framework_base_path: Optional[str] = None,
                          video_preview_mode: Optional[str] = None, audio_preview_mode: Optional[str] = None,
                          detect_width: int = 0) -> None:
    cmd = [sys.executable, str(Path(framework_path).resolve()), "--mode", "encrypt", "--in", in_path, "--out", out_path, "--key", master_key, "--detect_width", str(detect_width)]
    if roi_sidecar:
        cmd.extend(["--roi_sidecar", roi_sidecar])
    if payload:
        cmd.extend(["--payload", payload])
    if manifest:
        cmd.extend(["--manifest", manifest])
    if keystream_dump:
        cmd.extend(["--keystream_dump", keystream_dump])
    if framework_base_path:
        cmd.extend(["--base_framework", framework_base_path])
    if video_preview_mode:
        cmd.extend(["--video_preview_mode", video_preview_mode])
    if audio_preview_mode:
        cmd.extend(["--audio_preview_mode", audio_preview_mode])
    subprocess.run(cmd, check=True)


def framework_cli_decrypt(framework_path: str, in_path: str, out_path: str, master_key: str,
                          roi_sidecar: Optional[str] = None, payload: Optional[str] = None,
                          manifest: Optional[str] = None,
                          framework_base_path: Optional[str] = None) -> None:
    cmd = [sys.executable, str(Path(framework_path).resolve()), "--mode", "decrypt", "--in", in_path, "--out", out_path, "--key", master_key]
    if roi_sidecar:
        cmd.extend(["--roi_sidecar", roi_sidecar])
    if payload:
        cmd.extend(["--payload", payload])
    if manifest:
        cmd.extend(["--manifest", manifest])
    if framework_base_path:
        cmd.extend(["--base_framework", framework_base_path])
    subprocess.run(cmd, check=True)


def discover_assets(input_path: str) -> Dict[str, Optional[str]]:
    p = Path(input_path)
    assets = {
        'plain': None,
        'cipher': None,
        'decrypted': None,
        'roi_sidecar': None,
        'payload': None,
        'manifest': None,
        'bitstream': None,
        'keystream': None,
        'keystream_altkey': None,
        'plain2': None,
        'cipher2': None,
        'plain_ref': None,
        'cipher_ref': None,
        'cipher_altkey': None,
        'payload_altkey': None,
        'manifest_altkey': None,
    }

    def _rel_lower(f: Path) -> str:
        try:
            return str(f.relative_to(p)).lower()
        except Exception:
            return str(f).lower()

    def _sort_key(f: Path):
        rel = _rel_lower(f)
        parts = [x.lower() for x in Path(rel).parts[:-1]]
        penalty = 0
        if 'results_unified' in parts:
            penalty += 50
        if 'generated' in parts:
            penalty += 20
        if 'results' in parts:
            penalty += 10
        return (penalty, len(parts), rel)

    def _choose(files):
        files = list(files)
        return str(sorted(files, key=_sort_key)[0]) if files else None

    def _name(f: Path) -> str:
        return f.name.lower()

    def _stem(f: Path) -> str:
        return f.stem.lower()

    def _has_any(text: str, pats) -> bool:
        return any(p in text for p in pats)

    def _media_exact(stems):
        return _choose(f for f in media_files if _stem(f) in stems)

    def _media_match(include, exclude=()):
        return _choose(
            f for f in media_files
            if (_has_any(_name(f), include) or _has_any(_rel_lower(f), include) or _has_any(_stem(f), include))
            and not (_has_any(_name(f), exclude) or _has_any(_rel_lower(f), exclude) or _has_any(_stem(f), exclude))
        )

    def _jsonl_match(include, exclude=()):
        return _choose(
            f for f in jsonl_files
            if (_has_any(_name(f), include) or _has_any(_rel_lower(f), include) or _has_any(_stem(f), include))
            and not (_has_any(_name(f), exclude) or _has_any(_rel_lower(f), exclude) or _has_any(_stem(f), exclude))
        )

    def _bin_match(include, exclude=()):
        return _choose(
            f for f in bin_files
            if (_has_any(_name(f), include) or _has_any(_rel_lower(f), include) or _has_any(_stem(f), include))
            and not (_has_any(_name(f), exclude) or _has_any(_rel_lower(f), exclude) or _has_any(_stem(f), exclude))
        )

    def _manifest_match(include, exclude=()):
        return _choose(
            f for f in manifest_files
            if (_has_any(_name(f), include) or _has_any(_rel_lower(f), include) or _has_any(_stem(f), include))
            and not (_has_any(_name(f), exclude) or _has_any(_rel_lower(f), exclude) or _has_any(_stem(f), exclude))
        )

    if p.is_file():
        ext = p.suffix.lower()
        name = p.name.lower()
        if is_bitstream(str(p)):
            if _has_any(name, ['alt', 'altkey', 'keystream2', 'keystream_2']):
                assets['keystream_altkey'] = str(p)
            elif 'bitstream' in name:
                assets['bitstream'] = str(p)
            else:
                assets['keystream'] = str(p)
        elif is_image(str(p)) or is_video(str(p)):
            if _has_any(name, ['cipher_altkey', 'altkey_cipher', 'cipher_alt', 'encrypted_altkey', 'altkey_encrypted', 'encrypted_alt', 'enc_altkey', 'enc_alt']):
                assets['cipher_altkey'] = str(p)
            elif _has_any(name, ['cipher_ref', 'known_cipher', 'ref_cipher']):
                assets['cipher_ref'] = str(p)
            elif _has_any(name, ['cipher2', 'cipher_2', 'alt_cipher', 'encrypted2', 'encrypted_2', 'enc2', 'enc_2']):
                assets['cipher2'] = str(p)
            elif _has_any(name, ['plain_ref', 'known_plain', 'ref_plain']):
                assets['plain_ref'] = str(p)
            elif _has_any(name, ['plain2', 'plain_2', 'alt_plain']):
                assets['plain2'] = str(p)
            elif _has_any(name, ['decrypt', 'decrypted', 'recovered', 'dec']):
                assets['decrypted'] = str(p)
            elif _stem(p) in {'plain', 'original', 'input', 'source'}:
                assets['plain'] = str(p)
            elif _stem(p) in {'cipher', 'encrypted', 'enc'}:
                assets['cipher'] = str(p)
            else:
                assets['plain'] = str(p)
        elif ext == '.jsonl':
            if 'payload' in name:
                if _has_any(name, ['alt', 'altkey', 'payload2', 'payload_2']):
                    assets['payload_altkey'] = str(p)
                else:
                    assets['payload'] = str(p)
            else:
                assets['roi_sidecar'] = str(p)
        elif name.endswith('.manifest.json') or 'manifest' in name:
            if _has_any(name, ['alt', 'altkey', 'payload2', 'payload_2']):
                assets['manifest_altkey'] = str(p)
            else:
                assets['manifest'] = str(p)
        if assets['bitstream'] is None and assets['keystream'] is not None:
            assets['bitstream'] = assets['keystream']
        return assets

    files = [x for x in p.rglob('*') if x.is_file()]
    media_files = [x for x in files if is_video(str(x)) or is_image(str(x))]
    bin_files = [x for x in files if is_bitstream(str(x))]
    jsonl_files = [x for x in files if x.suffix.lower() == '.jsonl']
    manifest_files = [x for x in files if x.name.lower().endswith('.manifest.json') or 'manifest' in x.name.lower()]

    assets['plain'] = _media_exact({'plain', 'original', 'input', 'source'}) or _media_match(
        ['plain', 'original', 'input', 'source'],
        ['plain2', 'plain_2', 'alt_plain', 'plain_ref', 'known_plain', 'ref_plain', 'decrypt', 'decrypted', 'recovered', 'cipher', 'encrypted', 'enc']
    )
    assets['cipher'] = _media_exact({'cipher', 'encrypted', 'enc'}) or _media_match(
        ['cipher', 'encrypted', 'enc'],
        ['cipher_altkey', 'altkey_cipher', 'cipher_alt', 'encrypted_altkey', 'altkey_encrypted', 'encrypted_alt', 'enc_altkey', 'enc_alt',
         'cipher2', 'cipher_2', 'alt_cipher', 'encrypted2', 'encrypted_2', 'enc2', 'enc_2', 'cipher_ref', 'known_cipher', 'ref_cipher',
         'decrypt', 'decrypted', 'recovered']
    )
    assets['decrypted'] = _media_exact({'decrypted', 'decrypt', 'recovered', 'dec'}) or _media_match(['decrypt', 'decrypted', 'recovered'], [])
    assets['plain2'] = _media_match(['plain2', 'plain_2', 'alt_plain'])
    assets['cipher2'] = _media_match(['cipher2', 'cipher_2', 'alt_cipher', 'encrypted2', 'encrypted_2', 'enc2', 'enc_2'])
    assets['cipher_altkey'] = _media_match(['cipher_altkey', 'altkey_cipher', 'cipher_alt', 'encrypted_altkey', 'altkey_encrypted', 'encrypted_alt', 'enc_altkey', 'enc_alt'])
    assets['plain_ref'] = _media_match(['plain_ref', 'known_plain', 'ref_plain'])
    assets['cipher_ref'] = _media_match(['cipher_ref', 'known_cipher', 'ref_cipher'])

    if assets['plain'] is None and media_files:
        assets['plain'] = _choose(media_files)
    if assets['cipher'] is None and len(media_files) > 1:
        assets['cipher'] = _choose(
            f for f in media_files
            if str(f) not in {assets['plain'], assets['cipher2'], assets['cipher_altkey'], assets['cipher_ref'], assets['decrypted'], assets['plain2'], assets['plain_ref']}
        )

    assets['payload'] = _jsonl_match(['payload'], ['alt', 'altkey', 'payload2', 'payload_2', 'plain2', 'plainref', 'ref'])
    assets['payload_altkey'] = _jsonl_match(['payload_altkey', 'altkey', 'payload2', 'payload_2'], [])
    assets['roi_sidecar'] = _jsonl_match(['rois.jsonl', 'rois', 'roi.jsonl', 'roi'], ['payload'])
    if assets['roi_sidecar'] is None and jsonl_files:
        roi_like = [x for x in jsonl_files if 'payload' not in x.name.lower()]
        assets['roi_sidecar'] = _choose(roi_like)

    assets['bitstream'] = _bin_match(['bitstream'], ['alt', 'altkey', 'keystream2', 'keystream_2'])
    assets['keystream'] = _bin_match(['keystream', 'cipher_payload'], ['alt', 'altkey', 'keystream2', 'keystream_2'])
    assets['keystream_altkey'] = _bin_match(['keystream_altkey', 'altkey', 'keystream2', 'keystream_2'], [])

    assets['manifest'] = _manifest_match(['payload.jsonl.manifest', 'payload.manifest', 'manifest'], ['alt', 'altkey', 'payload2', 'payload_2', 'plain2', 'plainref'])
    assets['manifest_altkey'] = _manifest_match(['payload_altkey', 'altkey', 'payload2', 'payload_2'], [])

    if assets['bitstream'] is None and assets['keystream'] is not None:
        assets['bitstream'] = assets['keystream']
    elif assets['bitstream'] is None and bin_files:
        assets['bitstream'] = _choose(bin_files)

    return assets


    files = [x for x in p.rglob('*') if x.is_file()]
    media_files = [x for x in files if is_video(str(x)) or is_image(str(x))]
    bin_files = [x for x in files if is_bitstream(str(x))]
    jsonl_files = [x for x in files if x.suffix.lower() == '.jsonl']
    manifest_files = [x for x in files if x.name.lower().endswith('.manifest.json') or 'manifest' in x.name.lower()]

    def pick_media(keywords: List[str]) -> Optional[str]:
        hits = []
        for f in media_files:
            stem = f.stem.lower()
            name = f.name.lower()
            rel = str(f.relative_to(p)).lower()
            if any(k in stem or k in name or k in rel for k in keywords):
                hits.append(f)
        return str(sorted(hits)[0]) if hits else None

    assets['plain'] = pick_media(['plain', 'original', 'input', 'source'])
    assets['cipher'] = pick_media(['cipher', 'encrypted', 'enc'])
    assets['decrypted'] = pick_media(['decrypt', 'decrypted', 'recovered', 'dec'])
    assets['plain2'] = pick_media(['plain2', 'plain_2', 'alt_plain'])
    assets['cipher2'] = pick_media(['cipher2', 'cipher_2', 'alt_cipher', 'encrypted2', 'encrypted_2', 'enc2', 'enc_2'])
    assets['cipher_altkey'] = pick_media(['cipher_altkey', 'altkey_cipher', 'cipher_alt', 'encrypted_altkey', 'altkey_encrypted', 'encrypted_alt', 'enc_altkey', 'enc_alt'])
    assets['plain_ref'] = pick_media(['plain_ref', 'known_plain', 'ref_plain'])
    assets['cipher_ref'] = pick_media(['cipher_ref', 'known_cipher', 'ref_cipher'])

    if assets['plain'] is None and media_files:
        assets['plain'] = str(sorted(media_files)[0])
    if assets['cipher'] is None and len(media_files) > 1:
        for f in sorted(media_files):
            s = str(f)
            if s not in {assets['plain'], assets['cipher2'], assets['cipher_altkey'], assets['decrypted']}:
                assets['cipher'] = s
                break

    for f in sorted(jsonl_files):
        name = f.name.lower()
        rel = str(f.relative_to(p)).lower()
        if ('payload' in name or 'payload' in rel) and ('manifest' not in name):
            if 'alt' in name or 'alt' in rel or name.startswith('payload2') or 'payload_2' in name:
                assets['payload_altkey'] = str(f)
            elif assets['payload'] is None:
                assets['payload'] = str(f)
            continue
        if 'roi' in name or 'rois' in name or 'roi' in rel:
            assets['roi_sidecar'] = str(f)
            break
    if assets['roi_sidecar'] is None and jsonl_files:
        roi_like = [x for x in sorted(jsonl_files) if 'payload' not in x.name.lower()]
        if roi_like:
            assets['roi_sidecar'] = str(roi_like[0])

    for f in sorted(bin_files):
        name = f.name.lower()
        rel = str(f.relative_to(p)).lower()
        if 'bitstream' in name or 'bitstream' in rel:
            assets['bitstream'] = str(f)
        elif 'keystream' in name or 'keystream' in rel or 'cipher_payload' in name or 'cipher_payload' in rel:
            if 'alt' in name or 'alt' in rel or name.startswith('keystream2') or 'keystream_2' in name:
                assets['keystream_altkey'] = str(f)
            else:
                assets['keystream'] = str(f)

    for f in sorted(manifest_files):
        name = f.name.lower()
        rel = str(f.relative_to(p)).lower()
        if 'alt' in name or 'alt' in rel or name.startswith('payload2') or 'payload_2' in name:
            assets['manifest_altkey'] = str(f)
        elif assets['manifest'] is None:
            assets['manifest'] = str(f)

    if assets['bitstream'] is None and bin_files:
        assets['bitstream'] = str(sorted(bin_files)[0])

    return assets
