#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from unified_utils import (
    derive_key_variant,
    derive_used_keystream_bytes,
    discover_assets,
    encrypt_frames_with_sidecar,
    framework_cli_decrypt,
    framework_cli_encrypt,
    import_module_from_path,
    load_sidecar_records,
    make_checkerboard_variant,
    make_inverted_variant,
    module_supports_payload,
    read_media,
    write_bytes,
    write_media,
)

THIS_DIR = Path(__file__).resolve().parent


def _abs_cli_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return str(Path(path).resolve())


def run_cmd(label: str, cmd: list, report: dict):
    try:
        res = subprocess.run(
            cmd,
            cwd=str(THIS_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        report[label] = {
            "status": "ok",
            "cmd": cmd,
            "stdout_tail": (res.stdout or "")[-2000:],
            "stderr_tail": (res.stderr or "")[-2000:],
        }
    except subprocess.CalledProcessError as e:
        report[label] = {
            "status": "failed",
            "cmd": cmd,
            "stdout_tail": (e.stdout or "")[-2000:],
            "stderr_tail": (e.stderr or "")[-2000:],
            "returncode": e.returncode,
        }


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def resolve_master_key(dataset_dir: Path, cli_master_key: Optional[str], notes: List[str]) -> Optional[str]:
    if cli_master_key:
        notes.append("Using master key from --master_key.")
        return cli_master_key

    txt_candidates = [
        dataset_dir / "master_key.txt",
        dataset_dir / "key.txt",
        dataset_dir / "config" / "master_key.txt",
        dataset_dir / "config" / "key.txt",
    ]
    for p in txt_candidates:
        if p.is_file():
            notes.append(f"Using master key from {p.name}.")
            return read_text_file(p)

    json_candidates = [
        dataset_dir / "config.json",
        dataset_dir / "dataset.json",
        dataset_dir / "config" / "config.json",
    ]
    for p in json_candidates:
        if not p.is_file():
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            for key in ["master_key", "key", "secret_key"]:
                if isinstance(obj.get(key), str) and obj[key].strip():
                    notes.append(f"Using master key from {p.name}:{key}.")
                    return obj[key].strip()
        except Exception as e:
            notes.append(f"Ignored unreadable JSON config {p.name}: {e}")

    notes.append("No master key file found in dataset; key-dependent tests may be skipped.")
    return None


def expected_dataset_layout() -> dict:
    return {
        "required_for_full_run": {
            "plain media": "plain.* or original.* or source.*",
            "cipher media": "cipher.* or encrypted.* or enc.*",
            "ROI sidecar": "rois.jsonl or roi*.jsonl",
            "payload": "payload.jsonl for payload-aware frameworks",
            "manifest": "payload.jsonl.manifest.json for verified payload-aware frameworks (recommended)",
            "bitstream/keystream": "bitstream.bin or keystream.bin (or derivable from sidecar + master key)",
            "master key": "master_key.txt or key.txt or config.json with master_key",
        },
        "recommended": {
            "decrypted media": "decrypted.* or recovered.* for Tests 12 and optional 13"
        },
        "optional": {
            "plain2": "plain2.* or alt_plain.*",
            "cipher2": "cipher2.* or alt_cipher.* or encrypted2.*",
            "plain_ref": "plain_ref.* or known_plain.*",
            "cipher_ref": "cipher_ref.* or known_cipher.*",
        },
    }


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


def choose_roi_sidecar_for_manifest(dataset_dir: Path, manifest_path: Optional[str], fallback_roi: Optional[str], notes: Optional[List[str]] = None) -> Optional[str]:
    if not manifest_path:
        return fallback_roi
    mp = Path(manifest_path)
    if not mp.is_file():
        return fallback_roi
    try:
        manifest = json.loads(mp.read_text(encoding="utf-8"))
    except Exception as e:
        if notes is not None:
            notes.append(f"Could not read manifest for ROI matching: {e}")
        return fallback_roi

    target_name = Path(str(manifest.get("roi_sidecar_path") or "")).name.lower()
    target_sha = str(manifest.get("roi_sidecar_sha256") or "").strip().lower()

    candidates: List[Path] = []
    seen = set()

    def add(p: Path):
        try:
            rp = str(p.resolve())
        except Exception:
            rp = str(p)
        if rp in seen or not p.is_file() or p.suffix.lower() != ".jsonl":
            return
        if "roi" not in p.name.lower():
            return
        seen.add(rp)
        candidates.append(p)

    if fallback_roi:
        add(Path(fallback_roi))
    for pat in ("roi*.jsonl", "rois*.jsonl", "*roi*.jsonl"):
        for p in dataset_dir.glob(pat):
            add(p)

    def file_sha(p: Path) -> str:
        try:
            return _sha256_file(str(p)).lower()
        except Exception:
            return ""

    if target_name and target_sha:
        for p in candidates:
            if p.name.lower() == target_name and file_sha(p) == target_sha:
                if notes is not None and fallback_roi and str(p.resolve()) != str(Path(fallback_roi).resolve()):
                    notes.append(f"Using ROI sidecar matched from manifest: {p.name}")
                return str(p.resolve())
    if target_sha:
        for p in candidates:
            if file_sha(p) == target_sha:
                if notes is not None and fallback_roi and str(p.resolve()) != str(Path(fallback_roi).resolve()):
                    notes.append(f"Using ROI sidecar matched from manifest by hash: {p.name}")
                return str(p.resolve())
    if target_name:
        for p in candidates:
            if p.name.lower() == target_name:
                if notes is not None and fallback_roi and str(p.resolve()) != str(Path(fallback_roi).resolve()):
                    notes.append(f"Using ROI sidecar matched from manifest by name: {p.name}")
                return str(p.resolve())

    return fallback_roi

def validate_dataset_assets(assets: Dict[str, Optional[str]], master_key: Optional[str]) -> dict:
    full_run_ready = True
    missing: List[str] = []
    if not assets.get("plain"):
        full_run_ready = False
        missing.append("plain media")
    if not assets.get("cipher"):
        full_run_ready = False
        missing.append("cipher media")
    if not assets.get("roi_sidecar"):
        full_run_ready = False
        missing.append("ROI sidecar")
    if not (assets.get("bitstream") or assets.get("keystream") or (assets.get("roi_sidecar") and master_key)):
        full_run_ready = False
        missing.append("bitstream/keystream (or derivation inputs)")
    if not master_key:
        full_run_ready = False
        missing.append("master key")
    return {"full_run_ready": full_run_ready, "missing_for_full_run": missing}


def ensure_generated_core_assets(
    assets: Dict[str, Optional[str]],
    dataset_dir: Path,
    out_dir: Path,
    master_key: Optional[str],
    framework_path: str,
    max_frames: int,
    notes: List[str],
    framework_base_path: Optional[str] = None,
    selected_tests: Optional[set[int]] = None,
) -> None:
    out_gen = out_dir / "generated"
    out_gen.mkdir(parents=True, exist_ok=True)

    def needs(*tests: int) -> bool:
        return selected_tests is None or any(t in selected_tests for t in tests)

    needs_keystream = needs(2, 10)
    needs_decrypted = needs(12)
    needs_second_cipher = needs(2)
    needs_payload_tests = needs(6, 12, 14, 15)

    # Avoid importing the executable framework unless the selected tests actually
    # need asset generation or keystream derivation. This prevents spurious heavy
    # imports (for example torch via the proxy/payload framework) during tests
    # like Test 09 that only use the logic/base framework path.
    maybe_need_primary_generation = bool(
        assets.get("plain") and master_key and (
            not assets.get("cipher")
            or not assets.get("roi_sidecar")
            or (needs_payload_tests and not assets.get("payload"))
        )
    )
    maybe_need_decrypted_generation = bool(
        needs_decrypted and assets.get("cipher") and assets.get("roi_sidecar") and master_key and not assets.get("decrypted")
    )
    maybe_need_second_cipher_generation = bool(
        needs_second_cipher and assets.get("plain") and assets.get("roi_sidecar") and master_key and not (assets.get("cipher_altkey") or assets.get("cipher2"))
    )
    maybe_need_keystream_derivation = bool(
        needs_keystream and assets.get("plain") and assets.get("roi_sidecar") and not assets.get("bitstream") and not assets.get("keystream") and master_key
    )

    if not (
        maybe_need_primary_generation
        or maybe_need_decrypted_generation
        or maybe_need_second_cipher_generation
        or maybe_need_keystream_derivation
    ):
        notes.append("Skipped executable framework import in ensure_generated_core_assets; selected tests do not require generation.")
        return

    ff = import_module_from_path(framework_path, "framework_runall")
    payload_mode = module_supports_payload(ff)
    needs_payload_for_existing_cipher = payload_mode and needs_payload_tests

    if needs_keystream and assets.get("plain") and assets.get("roi_sidecar") and not assets.get("bitstream") and not assets.get("keystream") and master_key and not payload_mode:
        try:
            recs = load_sidecar_records(assets["roi_sidecar"])
            ks = derive_used_keystream_bytes(recs, ff, master_key)
            derived_path = write_bytes(str(out_gen / "keystream.bin"), ks)
            assets["bitstream"] = assets.get("bitstream") or derived_path
            assets["keystream"] = assets.get("keystream") or derived_path
            notes.append("Derived keystream.bin from roi_sidecar + master_key for NIST/differential tests.")
        except Exception as e:
            notes.append(f"Could not derive keystream from roi_sidecar: {e}")

    need_encrypt = bool(
        assets.get("plain") and master_key and (
            not assets.get("cipher")
            or not assets.get("roi_sidecar")
            or (needs_payload_for_existing_cipher and not assets.get("payload"))
        )
    )
    if need_encrypt:
        try:
            plain_path = assets["plain"]
            cipher_out = str(out_gen / "cipher.mkv")
            roi_out = str(out_gen / "rois.jsonl")
            ks_out = str(out_gen / "keystream.bin")
            payload_out = str(out_gen / "payload.jsonl") if payload_mode else None
            manifest_out = str(out_gen / "payload.jsonl.manifest.json") if payload_mode else None
            if payload_mode:
                framework_cli_encrypt(
                    framework_path,
                    plain_path,
                    cipher_out,
                    master_key,
                    roi_sidecar=roi_out,
                    payload=payload_out,
                    manifest=manifest_out,
                    keystream_dump=ks_out,
                    framework_base_path=framework_base_path,
                    video_preview_mode='chaos',
                    audio_preview_mode='chaos',
                )
                assets["payload"] = assets.get("payload") or payload_out
                assets["manifest"] = assets.get("manifest") or manifest_out
            else:
                ff.process_video(in_path=plain_path, out_path=cipher_out, master_key=master_key, mode='encrypt', roi_sidecar_path=roi_out, reuse_rois=False, detect_every=1, detect_width=0, keystream_dump_path=ks_out)
            assets["cipher"] = assets.get("cipher") or cipher_out
            assets["roi_sidecar"] = assets.get("roi_sidecar") or roi_out
            assets["bitstream"] = assets.get("bitstream") or ks_out
            assets["keystream"] = assets.get("keystream") or ks_out
            notes.append("Generated only the missing primary cipher/roi_sidecar/keystream assets.")
        except Exception as e:
            notes.append(f"Auto-generation of primary assets failed: {e}")

    if needs_decrypted and assets.get("cipher") and assets.get("roi_sidecar") and master_key and not assets.get("decrypted") and (not payload_mode or assets.get("payload")):
        try:
            dec_out = str(out_gen / "decrypted.mkv")
            if payload_mode:
                framework_cli_decrypt(framework_path, assets["cipher"], dec_out, master_key, roi_sidecar=assets.get("roi_sidecar"), payload=assets.get("payload"), manifest=assets.get("manifest"), framework_base_path=framework_base_path)
            else:
                ff.process_video(in_path=assets["cipher"], out_path=dec_out, master_key=master_key, mode='decrypt', roi_sidecar_path=assets["roi_sidecar"], reuse_rois=True, detect_every=1, detect_width=0, keystream_dump_path=None)
            assets["decrypted"] = dec_out
            notes.append("Generated decrypted media because the selected tests require it.")
        except Exception as e:
            notes.append(f"Auto-generation of decrypted media failed: {e}")

    if needs_second_cipher and assets.get("plain") and assets.get("roi_sidecar") and master_key and not (assets.get("cipher_altkey") or assets.get("cipher2")):
        try:
            if payload_mode:
                alt_key = derive_key_variant(master_key + "_alt", 64)
                alt_cipher = str(out_gen / "cipher_altkey.mkv")
                alt_payload = str(out_gen / "payload_altkey.jsonl")
                alt_manifest = str(out_gen / "payload_altkey.jsonl.manifest.json")
                alt_ks = str(out_gen / "keystream_altkey.bin")
                framework_cli_encrypt(
                    framework_path,
                    assets["plain"],
                    alt_cipher,
                    alt_key,
                    roi_sidecar=str(out_gen / "rois_altkey.jsonl"),
                    payload=alt_payload,
                    manifest=alt_manifest,
                    keystream_dump=alt_ks,
                    framework_base_path=framework_base_path,
                    video_preview_mode='chaos',
                    audio_preview_mode='chaos',
                )
                assets["cipher_altkey"] = alt_cipher
                assets["payload_altkey"] = alt_payload
                assets["manifest_altkey"] = alt_manifest
                assets["keystream_altkey"] = alt_ks
                notes.append("Generated only the alternate-key ciphertext bundle needed for Test 02.")
            else:
                frames, fps = read_media(assets["plain"])
                if max_frames and max_frames > 0:
                    frames = frames[:max_frames]
                recs = load_sidecar_records(assets["roi_sidecar"])
                recs = [r for r in recs if int(r.get("frame_idx", 0)) < len(frames)]
                alt_key = derive_key_variant(master_key + "_alt", 64)
                alt_cipher = write_media(encrypt_frames_with_sidecar(frames, recs, ff, alt_key), str(out_gen / "cipher_altkey.mkv"), fps=fps)
                alt_ks = write_bytes(str(out_gen / "keystream_altkey.bin"), derive_used_keystream_bytes(recs, ff, alt_key))
                assets["cipher_altkey"] = alt_cipher
                assets["keystream_altkey"] = alt_ks
                notes.append("Generated only the alternate-key ciphertext needed for Test 02.")
        except Exception as e:
            notes.append(f"Selective auxiliary asset generation failed: {e}")


def add_skip(report: dict, label: str, reason: str) -> None:
    report[label] = {"status": "skipped", "reason": reason}


def main() -> None:
    ap = argparse.ArgumentParser(description="Dataset-folder unified runner for the complete encryption evaluation framework.")
    ap.add_argument("dataset", help="Folder containing plain/cipher media, ROI sidecar, bitstream/keystream, and optional master_key.txt")
    ap.add_argument("--out", default=None, help="Defaults to <dataset>/results_unified")
    ap.add_argument("--master_key", default=None, help="Optional override. Otherwise the runner reads master_key.txt / key.txt / config.json from the dataset folder.")
    ap.add_argument("--framework_faster_path", default=str(THIS_DIR / "framework_faster_removed_une.py"))
    ap.add_argument("--framework_base_path", default=None, help="base framework path when using payload-aware executable framework")
    ap.add_argument("--max_frames", type=int, default=500)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--scope", default="roi", choices=["whole", "roi", "background"])
    ap.add_argument("--skip_perf_stat", action="store_true")
    ap.add_argument("--skip_sweep", action="store_true")
    ap.add_argument("--mask_ablation_paddings", default="0,8,16", help="Test 17 padding values")
    ap.add_argument("--mask_ablation_dilations", default="0,3,5", help="Test 17 dilation values")
    ap.add_argument("--mask_ablation_preview_modes", default="chaos", help="Test 17 preview modes for payload/proxy framework")
    ap.add_argument("--mask_ablation_limit", type=int, default=0, help="Optional Test 17 config limit for quick runs")
    ap.add_argument(
        "--tests",
        nargs="+",
        type=int,
        choices=range(1, 19),
        help="Run only the selected test numbers, e.g. --tests 2 6 10 14",
    )
    args = ap.parse_args()

    selected_tests = set(args.tests) if args.tests else None

    def wants(test_num: int) -> bool:
        return selected_tests is None or test_num in selected_tests

    framework_exec_path = _abs_cli_path(args.framework_faster_path)
    framework_base_path = _abs_cli_path(args.framework_base_path)
    framework_logic_path = framework_base_path or framework_exec_path

    dataset_dir = Path(args.dataset).resolve()
    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset folder not found: {dataset_dir}")

    out_dir = Path(args.out).resolve() if args.out else (dataset_dir / "results_unified")
    out_dir.mkdir(parents=True, exist_ok=True)

    assets = discover_assets(str(dataset_dir))
    notes: List[str] = []
    master_key = resolve_master_key(dataset_dir, args.master_key, notes)
    dataset_check = validate_dataset_assets(assets, master_key)

    master_report = {
        "dataset": str(dataset_dir),
        "out": str(out_dir),
        "expected_layout": expected_dataset_layout(),
        "assets": assets.copy(),
        "master_key_present": bool(master_key),
        "dataset_validation": dataset_check,
        "selected_tests": sorted(selected_tests) if selected_tests else "all",
        "framework_exec_path": framework_exec_path,
        "framework_base_path": framework_base_path,
        "framework_logic_path": framework_logic_path,
        "tests": {},
        "notes": notes,
    }

    ensure_generated_core_assets(assets, dataset_dir, out_dir, master_key, framework_exec_path, args.max_frames, notes, framework_base_path=framework_base_path, selected_tests=selected_tests)
    master_report["assets"] = assets.copy()
    master_report["dataset_validation"] = validate_dataset_assets(assets, master_key)

    if wants(1):
        if assets.get("plain") and assets.get("cipher"):
            cmd = [
                sys.executable, "01_quality_metrics_roi.py",
                "--plain", assets["plain"],
                "--test", assets["cipher"],
                "--out", str(out_dir / "01_quality"),
                "--max_frames", str(args.max_frames),
                "--stride", str(args.stride),
                "--mode", "color",
                "--framework_faster_path", framework_logic_path,
            ]
            if assets.get("roi_sidecar"):
                cmd.extend(["--roi_sidecar", assets["roi_sidecar"], "--scope", args.scope])
            else:
                cmd.extend(["--scope", "whole"])
            run_cmd("01_quality", cmd, master_report["tests"])
        else:
            add_skip(master_report["tests"], "01_quality", "Need plain and cipher media.")

    if wants(2):
        second_cipher = assets.get("cipher_altkey") or assets.get("cipher2")
        second_keystream = assets.get("keystream_altkey") or ""
        second_payload = assets.get("payload_altkey")
        if assets.get("plain") and master_key and second_cipher:
            original_keystream = assets.get("bitstream") or assets.get("keystream") or ""
            cmd = [
                sys.executable, "02_differential_sensitivity_roi.py",
                "--mode", "all",
                "--out", str(out_dir / "02_diff"),
                "--plain", assets["plain"],
                "--master_key", master_key,
                "--cipher", assets["cipher"],
                "--cipher2", second_cipher,
                "--keystream", original_keystream,
                "--keystream2", second_keystream,
                "--max_frames", str(args.max_frames),
                "--stride", str(args.stride),
                "--scope", args.scope,
                "--framework_faster_path", framework_exec_path,
            ]
            if assets.get("roi_sidecar"):
                cmd.extend(["--roi_sidecar", assets["roi_sidecar"]])
            if assets.get("payload"):
                cmd.extend(["--payload", assets["payload"]])
            if second_payload:
                cmd.extend(["--payload2", second_payload])
            if framework_base_path:
                cmd.extend(["--framework_base_path", framework_base_path])
            run_cmd("02_differential", [x for x in cmd if x != ""], master_report["tests"])
        else:
            add_skip(master_report["tests"], "02_differential", "Need plain media, master key, and a second ciphertext (cipher_altkey or cipher2 / encrypted2).")

    if wants(3):
        if assets.get("plain") and assets.get("cipher"):
            cmd = [
                sys.executable, "03_correlation_tests_roi.py",
                "--plain", assets["plain"],
                "--cipher", assets["cipher"],
                "--out", str(out_dir / "03_corr"),
                "--max_frames", str(args.max_frames),
                "--stride", str(args.stride),
                "--scope", args.scope,
            ]
            if assets.get("roi_sidecar"):
                cmd.extend(["--roi_sidecar", assets["roi_sidecar"]])
            run_cmd("03_correlation", cmd, master_report["tests"])
        else:
            add_skip(master_report["tests"], "03_correlation", "Need plain and cipher media.")

    if wants(4):
        if assets.get("cipher"):
            cmd = [
                sys.executable, "04_hist_entropy_roi.py",
                "--cipher", assets["cipher"],
                "--out", str(out_dir / "04_hist"),
                "--max_frames", str(args.max_frames),
                "--stride", str(args.stride),
                "--scope", args.scope,
                "--plots",
            ]
            if assets.get("plain"):
                cmd.extend(["--plain", assets["plain"]])
            if assets.get("roi_sidecar"):
                cmd.extend(["--roi_sidecar", assets["roi_sidecar"], "--framework_faster_path", framework_logic_path])
            run_cmd("04_hist_entropy", cmd, master_report["tests"])
        else:
            add_skip(master_report["tests"], "04_hist_entropy", "Need cipher media.")

    if wants(5):
        if assets.get("cipher"):
            cmd = [
                sys.executable, "05_psd_estimation.py",
                "--video", assets["cipher"],
                "--out", str(out_dir / "05_psd"),
                "--max_frames", str(args.max_frames),
                "--stride", str(args.stride),
            ]
            if assets.get("plain"):
                cmd.extend(["--plain", assets["plain"]])
            if assets.get("roi_sidecar"):
                cmd.extend(["--roi_sidecar", assets["roi_sidecar"], "--framework_faster_path", framework_logic_path])
            run_cmd("05_psd", cmd, master_report["tests"])
        else:
            add_skip(master_report["tests"], "05_psd", "Need cipher media.")

    if wants(6):
        roi_for_test6 = choose_roi_sidecar_for_manifest(dataset_dir, assets.get("manifest"), assets.get("roi_sidecar"), notes)
        if assets.get("plain") and assets.get("cipher") and roi_for_test6 and master_key:
            cmd = [
                sys.executable, "06_robustness_noise_occlusion_roi.py",
                "--plain", assets["plain"],
                "--cipher", assets["cipher"],
                "--out", str(out_dir / "06_robust"),
                "--framework_faster_path", framework_exec_path,
                "--master_key", master_key,
                "--roi_sidecar", roi_for_test6,
                "--payload", (assets.get("payload") or ""),
                "--manifest", (assets.get("manifest") or ""),
                "--attack_scope", args.scope,
                "--eval_scope", args.scope,
                "--max_frames", str(args.max_frames),
                "--stride", str(args.stride),
            ]
            if framework_base_path:
                cmd.extend(["--framework_base_path", framework_base_path])
            run_cmd("06_robustness", [x for x in cmd if x != ""], master_report["tests"])
        else:
            add_skip(master_report["tests"], "06_robustness", "Need plain media, cipher media, ROI sidecar, and master key.")

    if wants(7):
        if assets.get("plain") and assets.get("cipher"):
            cmd = [
                sys.executable, "07_bandwidth_analysis.py",
                "--out", str(out_dir / "07_bandwidth"),
                "--plain", assets["plain"],
                "--cipher", assets["cipher"],
            ]
            if assets.get("roi_sidecar"):
                cmd.extend(["--roi_sidecar", assets["roi_sidecar"], "--framework_faster_path", framework_logic_path, "--roi_area_method", "mask"])
            run_cmd("07_bandwidth", cmd, master_report["tests"])
        else:
            add_skip(master_report["tests"], "07_bandwidth", "Need plain and cipher media.")

    if wants(8):
        if assets.get("plain") and master_key:
            cmd = [
                sys.executable, "08_perf_cycles_throughput.py",
                "--module", framework_exec_path,
                "--in", assets["plain"],
                "--out_dir", str(out_dir / "08_perf"),
                "--key", master_key,
                "--bench",
                "--core_bench",
                "--line_profile",
            ]
            if framework_base_path:
                cmd.extend(["--framework_base_path", framework_base_path])
            if not args.skip_perf_stat:
                cmd.append("--perf_stat")
            if assets.get("roi_sidecar"):
                cmd.extend(["--roi_sidecar", assets["roi_sidecar"]])
            if assets.get("payload"):
                cmd.extend(["--payload", assets["payload"]])
            run_cmd("08_performance", cmd, master_report["tests"])
        else:
            add_skip(master_report["tests"], "08_performance", "Need plain media and master key.")

    if wants(9):
        if assets.get("plain") and assets.get("cipher") and assets.get("roi_sidecar") and master_key:
            cmd = [
                sys.executable, "09_missing_attack_eq_suite.py",
                "--framework_faster_path", framework_logic_path,
                "--master_key", master_key,
                "--plain", assets["plain"],
                "--cipher", assets["cipher"],
                "--roi_sidecar", assets["roi_sidecar"],
                "--max_frames", str(args.max_frames),
                "--out", str(out_dir / "09_attacks"),
            ]
            run_cmd("09_attacks_eq", cmd, master_report["tests"])
        else:
            add_skip(master_report["tests"], "09_attacks_eq", "Need plain media, cipher media, ROI sidecar, and master key.")

    if wants(10):
        nist_cmd = [sys.executable, "10_nist_runner.py", "--out", str(out_dir / "10_nist")]
        if assets.get("bitstream"):
            nist_cmd.extend(["--bitstream", assets["bitstream"]])
        elif assets.get("keystream"):
            nist_cmd.extend(["--keystream", assets["keystream"]])
        elif assets.get("roi_sidecar") and master_key:
            nist_cmd.extend(["--roi_sidecar", assets["roi_sidecar"], "--master_key", master_key, "--framework_faster_path", framework_logic_path])
        elif assets.get("cipher"):
            nist_cmd.extend(["--cipher", assets["cipher"]])
        else:
            nist_cmd = []
        if nist_cmd:
            run_cmd("10_nist", nist_cmd, master_report["tests"])
        else:
            add_skip(master_report["tests"], "10_nist", "Need bitstream/keystream, or roi_sidecar + master_key, or cipher media.")

    if wants(11):
        if not args.skip_sweep and assets.get("plain") and assets.get("roi_sidecar") and master_key:
            cmd = [
                sys.executable, "11_param_sweep.py",
                "--plain", assets["plain"],
                "--roi_sidecar", assets["roi_sidecar"],
                "--framework_faster_path", framework_exec_path,
                "--out", str(out_dir / "11_param_sweep"),
                "--master_key", master_key,
                "--scope", args.scope,
                "--max_frames", str(args.max_frames),
            ]
            if framework_base_path:
                cmd.extend(["--framework_base_path", framework_base_path])
            run_cmd("11_param_sweep", cmd, master_report["tests"])
        else:
            add_skip(master_report["tests"], "11_param_sweep", "Need plain media, ROI sidecar, and master key.")


    if wants(12):
        if assets.get("plain") and assets.get("cipher") and assets.get("decrypted"):
            cmd = [
                sys.executable, "12_audio_eval_from_videos_runall.py",
                "--plain_video", assets["plain"],
                "--cipher_video", assets["cipher"],
                "--decrypted_video", assets["decrypted"],
                "--out_dir", str(out_dir / "12_audio_eval"),
            ]
            run_cmd("12_audio_eval", cmd, master_report["tests"])
        else:
            add_skip(master_report["tests"], "12_audio_eval", "Need plain media, cipher media, and decrypted media.")

    if wants(13):
        if assets.get("plain") and assets.get("cipher") and assets.get("roi_sidecar"):
            cmd = [
                sys.executable, "13_detector_privacy_leakage.py",
                "--plain", assets["plain"],
                "--cipher", assets["cipher"],
                "--roi_sidecar", assets["roi_sidecar"],
                "--framework_faster_path", framework_logic_path,
                "--out", str(out_dir / "13_detector_privacy"),
                "--max_frames", str(args.max_frames),
                "--stride", str(args.stride),
            ]
            if assets.get("decrypted"):
                cmd.extend(["--decrypted", assets["decrypted"]])
            run_cmd("13_detector_privacy", cmd, master_report["tests"])
        else:
            add_skip(master_report["tests"], "13_detector_privacy", "Need plain media, cipher media, and ROI sidecar.")


    if wants(14):
        roi_for_test14 = choose_roi_sidecar_for_manifest(dataset_dir, assets.get("manifest"), assets.get("roi_sidecar"), notes)
        if assets.get("cipher") and roi_for_test14 and assets.get("payload") and assets.get("manifest") and master_key:
            cmd = [
                sys.executable, "14_chaos_payload_tamper_response.py",
                "--cipher", assets["cipher"],
                "--roi_sidecar", roi_for_test14,
                "--payload", assets["payload"],
                "--manifest", assets["manifest"],
                "--framework_faster_path", framework_exec_path,
                "--master_key", master_key,
                "--out", str(out_dir / "14_chaos_tamper"),
            ]
            if framework_base_path:
                cmd.extend(["--framework_base_path", framework_base_path])
            run_cmd("14_chaos_tamper", cmd, master_report["tests"])
        else:
            add_skip(master_report["tests"], "14_chaos_tamper", "Need cipher, ROI sidecar, payload, manifest, and master key. Use the payload/proxy chaos framework first.")

    if wants(15):
        roi_for_test15 = choose_roi_sidecar_for_manifest(dataset_dir, assets.get("manifest"), assets.get("roi_sidecar"), notes)
        if assets.get("cipher") and roi_for_test15 and assets.get("payload") and assets.get("manifest") and master_key:
            cmd = [
                sys.executable, "15_chaos_roi_replay_substitution.py",
                "--cipher", assets["cipher"],
                "--roi_sidecar", roi_for_test15,
                "--payload", assets["payload"],
                "--manifest", assets["manifest"],
                "--framework_faster_path", framework_exec_path,
                "--master_key", master_key,
                "--out", str(out_dir / "15_chaos_replay_substitution"),
            ]
            if framework_base_path:
                cmd.extend(["--framework_base_path", framework_base_path])
            run_cmd("15_chaos_replay_substitution", cmd, master_report["tests"])
        else:
            add_skip(master_report["tests"], "15_chaos_replay_substitution", "Need cipher, ROI sidecar, payload, manifest, and master key. Use the payload/proxy chaos framework first.")

    if wants(16):
        if assets.get("roi_sidecar"):
            cmd = [
                sys.executable, "16_chaos_nonce_context_binding.py",
                "--roi_sidecar", assets["roi_sidecar"],
                "--out", str(out_dir / "16_chaos_nonce_context"),
            ]
            if assets.get("payload"):
                cmd.extend(["--payload", assets["payload"]])
            run_cmd("16_chaos_nonce_context", cmd, master_report["tests"])
        else:
            add_skip(master_report["tests"], "16_chaos_nonce_context", "Need ROI sidecar; payload is optional but recommended.")

    if wants(17):
        if assets.get("plain") and assets.get("roi_sidecar") and master_key:
            cmd = [
                sys.executable, "17_mask_expansion_ablation.py",
                "--plain", assets["plain"],
                "--roi_sidecar", assets["roi_sidecar"],
                "--framework_faster_path", framework_exec_path,
                "--master_key", master_key,
                "--out", str(out_dir / "17_mask_expansion_ablation"),
                "--paddings", args.mask_ablation_paddings,
                "--dilations", args.mask_ablation_dilations,
                "--preview_modes", args.mask_ablation_preview_modes,
                "--max_frames", str(args.max_frames),
                "--stride", str(args.stride),
            ]
            if args.mask_ablation_limit and args.mask_ablation_limit > 0:
                cmd.extend(["--limit", str(args.mask_ablation_limit)])
            if framework_base_path:
                cmd.extend(["--framework_base_path", framework_base_path])
            run_cmd("17_mask_expansion_ablation", cmd, master_report["tests"])
        else:
            add_skip(master_report["tests"], "17_mask_expansion_ablation", "Need plain media, ROI sidecar, and master key.")

    if wants(18):
        if assets.get("plain") and assets.get("cipher") and assets.get("roi_sidecar"):
            cmd = [
                sys.executable, "18_storage_vs_roi_fraction.py",
                "--datasets", str(dataset_dir),
                "--framework_faster_path", framework_logic_path,
                "--out", str(out_dir / "18_storage_vs_roi_fraction"),
                "--roi_area_method", "mask",
            ]
            run_cmd("18_storage_vs_roi_fraction", cmd, master_report["tests"])
        else:
            add_skip(master_report["tests"], "18_storage_vs_roi_fraction", "Need plain media, cipher media, and ROI sidecar.")

    (out_dir / "dataset_manifest.json").write_text(json.dumps({
        "dataset": str(dataset_dir),
        "assets": assets,
        "master_key_present": bool(master_key),
        "dataset_validation": validate_dataset_assets(assets, master_key),
        "selected_tests": sorted(selected_tests) if selected_tests else "all",
    }, indent=2), encoding="utf-8")

    (out_dir / "master_report.json").write_text(json.dumps(master_report, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "report": str(out_dir / "master_report.json")}))


if __name__ == "__main__":
    main()