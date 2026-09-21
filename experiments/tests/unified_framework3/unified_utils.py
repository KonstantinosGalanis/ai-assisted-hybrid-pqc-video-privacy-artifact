from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

VIDEO_EXTS = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.m4v'}
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def is_video(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def video_info(path: str | Path) -> Dict[str, Any]:
    path = str(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f'Could not open video: {path}')
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        'fps': float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC) or 0),
        'file_bytes': int(os.path.getsize(path)),
    }
    cap.release()
    return info


def iter_frames(path: str | Path, max_frames: int = 0, stride: int = 1):
    path = str(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f'Could not open media: {path}')
    orig_idx = 0
    proc_idx = 0
    stride = max(1, int(stride or 1))
    max_frames = int(max_frames or 0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if orig_idx % stride == 0:
            yield proc_idx, orig_idx, frame
            proc_idx += 1
            if max_frames and proc_idx >= max_frames:
                break
        orig_idx += 1
    cap.release()


def read_media(path: str | Path, max_frames: int = 0, stride: int = 1) -> Tuple[List[np.ndarray], float]:
    path = str(path)
    if is_image(path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f'Could not read image: {path}')
        return [img], 1.0
    info = video_info(path)
    return [fr for _, _, fr in iter_frames(path, max_frames=max_frames, stride=stride)], float(info.get('fps') or 25.0)


def write_media(frames: List[np.ndarray], out_path: str | Path, fps: float = 25.0) -> str:
    if not frames:
        raise ValueError('No frames to write.')
    out_path = str(out_path)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if is_image(out_path):
        cv2.imwrite(out_path, frames[0])
        return out_path
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    if p.suffix.lower() not in VIDEO_EXTS:
        p = p.with_suffix('.mkv')
    vw = cv2.VideoWriter(str(p), fourcc, fps if fps > 0 else 25.0, (w, h))
    if not vw.isOpened():
        p = p.with_suffix('.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(str(p), fourcc, fps if fps > 0 else 25.0, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f'Could not open writer for {p}')
    for fr in frames:
        if fr.shape[:2] != (h, w):
            fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
        vw.write(fr)
    vw.release()
    return str(p)


def to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame.astype(np.uint8, copy=False)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def select_mode(frame: np.ndarray, mode: str) -> np.ndarray:
    mode = (mode or 'color').lower()
    if mode == 'gray':
        return to_gray(frame)
    return frame.astype(np.uint8, copy=False)


def align_frames(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a2 = a[:h, :w]
    b2 = b[:h, :w]
    if a2.ndim != b2.ndim:
        if a2.ndim == 3:
            a2 = to_gray(a2)
        if b2.ndim == 3:
            b2 = to_gray(b2)
    if a2.ndim == 3 and b2.ndim == 3:
        c = min(a2.shape[2], b2.shape[2])
        a2 = a2[:, :, :c]
        b2 = b2[:, :, :c]
    return a2, b2


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a, b = align_frames(a, b)
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(d * d))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    a, b = align_frames(a, b)
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    m = mse(a, b)
    if m <= 0:
        return float('inf')
    return float(20.0 * math.log10(255.0 / math.sqrt(m)))


def snr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = align_frames(a, b)
    aa = a.astype(np.float64)
    noise = aa - b.astype(np.float64)
    p_signal = float(np.mean(aa * aa))
    p_noise = float(np.mean(noise * noise))
    if p_noise <= 0:
        return float('inf')
    if p_signal <= 0:
        return float('-inf')
    return float(10.0 * math.log10(p_signal / p_noise))


def npcr_uaci(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    a, b = align_frames(a, b)
    aa = a.astype(np.uint8, copy=False)
    bb = b.astype(np.uint8, copy=False)
    n = aa.size
    if n == 0:
        return float('nan'), float('nan')
    npcr = 100.0 * float(np.mean(aa != bb))
    uaci = 100.0 * float(np.mean(np.abs(aa.astype(np.float64) - bb.astype(np.float64)) / 255.0))
    return npcr, uaci


def entropy_u8(data: np.ndarray) -> float:
    arr = np.asarray(data, dtype=np.uint8).reshape(-1)
    if arr.size == 0:
        return float('nan')
    hist = np.bincount(arr, minlength=256).astype(np.float64)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def adjacent_corr(img: np.ndarray, direction: str = 'h', max_pairs: int = 200000) -> float:
    if img.ndim == 3:
        img = to_gray(img)
    if direction == 'h':
        x = img[:, :-1].reshape(-1)
        y = img[:, 1:].reshape(-1)
    elif direction == 'v':
        x = img[:-1, :].reshape(-1)
        y = img[1:, :].reshape(-1)
    else:
        x = img[:-1, :-1].reshape(-1)
        y = img[1:, 1:].reshape(-1)
    if x.size == 0:
        return float('nan')
    if x.size > max_pairs:
        idx = np.linspace(0, x.size - 1, max_pairs).astype(np.int64)
        x, y = x[idx], y[idx]
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx == 0.0 or sy == 0.0:
        return float('nan')
    return float(np.corrcoef(x.astype(np.float64), y.astype(np.float64))[0, 1])


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_json_default)


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_csv(path: str | Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for k in row.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def run_cmd(cmd: List[str], cwd: Optional[str | Path] = None, verbose: bool = False) -> Dict[str, Any]:
    if verbose:
        print('[CMD]', ' '.join(map(str, cmd)), flush=True)
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
    elapsed = time.perf_counter() - t0
    return {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr, 'elapsed_s': elapsed, 'cmd': cmd}


def framework_cli_encrypt(framework_path: str, in_path: str, out_path: str, master_key: str, cipher_dump: Optional[str] = None, verbose: bool = False) -> Dict[str, Any]:
    cmd = [sys.executable, str(framework_path), '--mode', 'encrypt', '--in', str(in_path), '--out', str(out_path), '--key', str(master_key)]
    if cipher_dump:
        cmd += ['--cipher_dump', str(cipher_dump)]
    res = run_cmd(cmd, cwd=Path(framework_path).resolve().parent, verbose=verbose)
    if res['returncode'] != 0:
        raise RuntimeError(f'Encryption failed:\nSTDOUT:\n{res["stdout"][-2000:]}\nSTDERR:\n{res["stderr"][-4000:]}')
    return res


def framework_cli_decrypt(framework_path: str, in_path: str, out_path: str, master_key: str, verbose: bool = False) -> Dict[str, Any]:
    cmd = [sys.executable, str(framework_path), '--mode', 'decrypt', '--in', str(in_path), '--out', str(out_path), '--key', str(master_key)]
    res = run_cmd(cmd, cwd=Path(framework_path).resolve().parent, verbose=verbose)
    if res['returncode'] != 0:
        raise RuntimeError(f'Decryption failed:\nSTDOUT:\n{res["stdout"][-2000:]}\nSTDERR:\n{res["stderr"][-4000:]}')
    return res


def derive_key_variant(master_key: str, variant: str) -> str:
    return hashlib.sha256(f'{master_key}|variant={variant}'.encode('utf-8')).hexdigest()


def make_inverted_variant(in_path: str, out_path: str, max_frames: int = 0, stride: int = 1) -> str:
    frames, fps = read_media(in_path, max_frames=max_frames, stride=stride)
    out_frames = [cv2.bitwise_not(fr) for fr in frames]
    return write_media(out_frames, out_path, fps)


def make_one_pixel_variant(in_path: str, out_path: str, max_frames: int = 0, stride: int = 1) -> str:
    frames, fps = read_media(in_path, max_frames=max_frames, stride=stride)
    if frames:
        fr = frames[0].copy()
        fr[0, 0, 0] = np.uint8(int(fr[0, 0, 0]) ^ 1)
        frames[0] = fr
    return write_media(frames, out_path, fps)


def raw_frame_bytes(path: str | Path, max_frames: int = 0, stride: int = 1) -> bytes:
    chunks: List[bytes] = []
    for _, _, fr in iter_frames(path, max_frames=max_frames, stride=stride):
        chunks.append(np.ascontiguousarray(fr).reshape(-1).tobytes())
    return b''.join(chunks)


def write_bytes(path: str | Path, data: bytes) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(data)


def discover_assets(dataset_dir: str | Path) -> Dict[str, Optional[str]]:
    d = Path(dataset_dir)
    files = [p for p in d.iterdir() if p.is_file()]
    def pick(prefixes: Iterable[str], exts=VIDEO_EXTS | IMAGE_EXTS):
        for p in files:
            name = p.name.lower()
            if p.suffix.lower() in exts and any(name.startswith(x) for x in prefixes):
                return str(p)
        return None
    assets = {
        'plain': pick(['plain', 'original', 'source']),
        'cipher': pick(['cipher', 'encrypted', 'enc']),
        'decrypted': pick(['decrypted', 'recovered', 'dec']),
        'plain2': pick(['plain2', 'alt_plain', 'known_plain']),
        'cipher2': pick(['cipher2', 'alt_cipher', 'encrypted2', 'known_cipher']),
        'bitstream': None,
        'master_key_file': None,
    }
    for p in files:
        name = p.name.lower()
        if p.suffix.lower() in {'.bin', '.bits', '.bit', '.dat', '.raw'} and any(x in name for x in ['bitstream', 'keystream', 'cipher_dump', 'dump']):
            assets['bitstream'] = str(p)
        if name in {'master_key.txt', 'key.txt'}:
            assets['master_key_file'] = str(p)
    return assets


def load_master_key(dataset_dir: str | Path, cli_key: Optional[str]) -> Optional[str]:
    if cli_key:
        return cli_key
    d = Path(dataset_dir)
    for name in ['master_key.txt', 'key.txt']:
        p = d / name
        if p.is_file():
            return p.read_text(encoding='utf-8').strip()
    for name in ['config.json', 'dataset.json']:
        p = d / name
        if p.is_file():
            try:
                obj = json.loads(p.read_text(encoding='utf-8'))
                for k in ['master_key', 'key', 'secret_key']:
                    if isinstance(obj.get(k), str) and obj[k].strip():
                        return obj[k].strip()
            except Exception:
                pass
    return None


def module_supports_payload(*args, **kwargs) -> bool:
    return False

# Backward-compatible stubs: full-frame AES tests do not use ROI sidecars.
def load_sidecar_records(path: str):
    raise RuntimeError('Full-frame test suite does not use rois.jsonl / ROI sidecars.')

def derive_used_keystream_bytes(*args, **kwargs):
    raise RuntimeError('Full-frame AES-CTR suite uses cipher_dump/raw frame bytes for randomness tests, not ROI sidecar-derived keystream.')


def resolve_sidecar_module(*args, **kwargs):
    raise RuntimeError('Full-frame test suite does not use ROI sidecars.')
