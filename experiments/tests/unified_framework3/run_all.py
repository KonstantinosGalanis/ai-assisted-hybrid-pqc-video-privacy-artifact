#!/usr/bin/env python3
r"""
run_all.py - full-frame AES evaluation runner

Default behavior:
  - no --dataset_dir is required
  - it looks for a folder named content3 automatically
  - it writes all results to content3/unified_results
  - every test writes stdout/stderr logs and a detailed JSON report

Typical command:
  python .\unified_framework3\run_all.py

Selective examples:
  python .\unified_framework3\run_all.py --tests 01,02,10
  python .\unified_framework3\run_all.py --tests quality,nist,detector
  python .\unified_framework3\run_all.py --skip_tests audio,detector
  python .\unified_framework3\run_all.py --list_tests
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CONTENT_DIR_NAME = "content3"
DEFAULT_RESULTS_DIR_NAME = "unified_results"
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
BITSTREAM_EXTS = {".bin", ".bits", ".bit", ".dat", ".raw"}

ALL_TEST_LABELS = [
    "01_quality_plain_vs_cipher",
    "01_quality_plain_vs_decrypted",
    "02_differential",
    "03_correlation",
    "04_hist_entropy",
    "05_psd",
    "06_robustness",
    "07_bandwidth",
    "08_perf",
    "09_attack_eq",
    "10_nist",
    "10_nist_decoded_cipher_frames",
    "11_param_sweep",
    "12_audio",
    "13_detector_privacy_leakage",
    "14_ctr_tamper_malleability",
    "15_ctr_replay_substitution",
    "16_ctr_iv_context_binding",
    "17_ctr_corruption_ablation",
    "18_storage_overhead",
]

TEST_ALIASES = {
    "all": ALL_TEST_LABELS,
    "01": ["01_quality_plain_vs_cipher", "01_quality_plain_vs_decrypted"],
    "1": ["01_quality_plain_vs_cipher", "01_quality_plain_vs_decrypted"],
    "quality": ["01_quality_plain_vs_cipher", "01_quality_plain_vs_decrypted"],
    "02": ["02_differential"],
    "2": ["02_differential"],
    "differential": ["02_differential"],
    "npcr": ["02_differential"],
    "uaci": ["02_differential"],
    "03": ["03_correlation"],
    "3": ["03_correlation"],
    "correlation": ["03_correlation"],
    "04": ["04_hist_entropy"],
    "4": ["04_hist_entropy"],
    "hist": ["04_hist_entropy"],
    "histogram": ["04_hist_entropy"],
    "entropy": ["04_hist_entropy"],
    "05": ["05_psd"],
    "5": ["05_psd"],
    "psd": ["05_psd"],
    "06": ["06_robustness"],
    "6": ["06_robustness"],
    "robustness": ["06_robustness"],
    "07": ["07_bandwidth"],
    "7": ["07_bandwidth"],
    "bandwidth": ["07_bandwidth"],
    "08": ["08_perf"],
    "8": ["08_perf"],
    "perf": ["08_perf"],
    "performance": ["08_perf"],
    "09": ["09_attack_eq"],
    "9": ["09_attack_eq"],
    "attack": ["09_attack_eq"],
    "attacks": ["09_attack_eq"],
    "10": ["10_nist", "10_nist_decoded_cipher_frames"],
    "nist": ["10_nist", "10_nist_decoded_cipher_frames"],
    "randomness": ["10_nist", "10_nist_decoded_cipher_frames"],
    "11": ["11_param_sweep"],
    "param": ["11_param_sweep"],
    "params": ["11_param_sweep"],
    "parameter": ["11_param_sweep"],
    "sweep": ["11_param_sweep"],
    "12": ["12_audio"],
    "audio": ["12_audio"],
    "13": ["13_detector_privacy_leakage"],
    "detector": ["13_detector_privacy_leakage"],
    "privacy": ["13_detector_privacy_leakage"],
    "detector_privacy": ["13_detector_privacy_leakage"],
    "14": ["14_ctr_tamper_malleability"],
    "tamper": ["14_ctr_tamper_malleability"],
    "malleability": ["14_ctr_tamper_malleability"],
    "15": ["15_ctr_replay_substitution"],
    "replay": ["15_ctr_replay_substitution"],
    "substitution": ["15_ctr_replay_substitution"],
    "16": ["16_ctr_iv_context_binding"],
    "iv": ["16_ctr_iv_context_binding"],
    "nonce": ["16_ctr_iv_context_binding"],
    "context": ["16_ctr_iv_context_binding"],
    "17": ["17_ctr_corruption_ablation"],
    "ablation": ["17_ctr_corruption_ablation"],
    "corruption": ["17_ctr_corruption_ablation"],
    "18": ["18_storage_overhead"],
    "storage": ["18_storage_overhead"],
    "overhead": ["18_storage_overhead"],
}

SELECTED_TESTS: Optional[set[str]] = None
SKIPPED_TESTS: set[str] = set()


def split_test_tokens(value: Optional[str]) -> List[str]:
    if not value:
        return []
    tokens: List[str] = []
    for part in value.replace(";", ",").split(","):
        for token in part.split():
            token = token.strip().lower().replace("-", "_")
            if token:
                tokens.append(token)
    return tokens


def expand_test_selection(value: Optional[str]) -> Optional[set[str]]:
    tokens = split_test_tokens(value)
    if not tokens:
        return None
    expanded: set[str] = set()
    unknown: List[str] = []
    for token in tokens:
        if token in TEST_ALIASES:
            expanded.update(TEST_ALIASES[token])
        elif token in ALL_TEST_LABELS:
            expanded.add(token)
        else:
            unknown.append(token)
    if unknown:
        valid = sorted(set(ALL_TEST_LABELS) | set(TEST_ALIASES.keys()))
        raise SystemExit(
            "Unknown test name(s): "
            + ", ".join(unknown)
            + "\n\nValid values include:\n  "
            + "\n  ".join(valid)
        )
    return expanded


def test_is_enabled(label: str) -> bool:
    if SELECTED_TESTS is not None and label not in SELECTED_TESTS:
        return False
    if label in SKIPPED_TESTS:
        return False
    return True


def selection_reason(label: str) -> Optional[str]:
    if SELECTED_TESTS is not None and label not in SELECTED_TESTS:
        return "Not selected by --tests."
    if label in SKIPPED_TESTS:
        return "Skipped by --skip_tests."
    return None


def _json_default(obj: Any) -> Any:
    try:
        import numpy as np  # type: ignore
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    tmp.replace(path)


def now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def is_media_file(path: Path) -> bool:
    return path.suffix.lower() in (VIDEO_EXTS | IMAGE_EXTS)


def is_bitstream_file(path: Path) -> bool:
    return path.suffix.lower() in BITSTREAM_EXTS


def resolve_content_dir(cli_value: Optional[str]) -> Path:
    candidates: List[Path] = []
    if cli_value:
        candidates.append(Path(cli_value))
    candidates.extend([
        Path.cwd() / DEFAULT_CONTENT_DIR_NAME,
        THIS_DIR.parent / DEFAULT_CONTENT_DIR_NAME,
        THIS_DIR / DEFAULT_CONTENT_DIR_NAME,
    ])
    for cand in candidates:
        if cand.is_dir():
            return cand.resolve()
    return (Path.cwd() / DEFAULT_CONTENT_DIR_NAME).resolve()


def pick_file(
    files: Iterable[Path],
    exact_stems: Iterable[str] = (),
    prefixes: Iterable[str] = (),
    excludes: Iterable[str] = (),
) -> Optional[str]:
    exact = {x.lower() for x in exact_stems}
    pref = tuple(x.lower() for x in prefixes)
    excl = tuple(x.lower() for x in excludes)
    pool = [p for p in files if p.is_file()]

    def ok_excl(p: Path) -> bool:
        name = p.name.lower()
        stem = p.stem.lower()
        return not any(x in name or x in stem for x in excl)

    for p in sorted(pool, key=lambda x: x.name.lower()):
        if p.stem.lower() in exact and ok_excl(p):
            return str(p.resolve())
    for p in sorted(pool, key=lambda x: x.name.lower()):
        name = p.name.lower()
        stem = p.stem.lower()
        if ok_excl(p) and any(name.startswith(x) or stem.startswith(x) for x in pref):
            return str(p.resolve())
    return None


def discover_assets(content_dir: Path) -> Dict[str, Optional[str]]:
    if not content_dir.is_dir():
        return {
            "plain": None,
            "cipher": None,
            "decrypted": None,
            "plain2": None,
            "cipher2": None,
            "bitstream": None,
            "master_key_file": None,
        }

    files = [p for p in content_dir.iterdir() if p.is_file()]
    media = [p for p in files if is_media_file(p)]
    bits = [p for p in files if is_bitstream_file(p)]

    assets: Dict[str, Optional[str]] = {
        "plain": pick_file(
            media,
            exact_stems=["plain", "original", "source", "input"],
            prefixes=["plain", "original", "source", "input"],
            excludes=["plain2", "plain_2", "alt_plain", "decrypted", "decrypt", "recovered", "cipher", "encrypted", "enc"],
        ),
        "cipher": pick_file(
            media,
            exact_stems=["cipher", "encrypted", "enc"],
            prefixes=["cipher", "encrypted", "enc"],
            excludes=["cipher2", "cipher_2", "encrypted2", "encrypted_2", "enc2", "enc_2", "alt", "decrypted", "decrypt", "recovered"],
        ),
        "decrypted": pick_file(
            media,
            exact_stems=["decrypted", "decrypt", "recovered", "dec"],
            prefixes=["decrypted", "decrypt", "recovered", "dec"],
            excludes=["decrypted2", "decrypt2", "recovered2", "dec2"],
        ),
        "plain2": pick_file(
            media,
            exact_stems=["plain2", "plain_2", "alt_plain", "known_plain"],
            prefixes=["plain2", "plain_2", "alt_plain", "known_plain"],
        ),
        "cipher2": pick_file(
            media,
            exact_stems=["cipher2", "cipher_2", "encrypted2", "encrypted_2", "enc2", "enc_2", "alt_cipher", "known_cipher"],
            prefixes=["cipher2", "cipher_2", "encrypted2", "encrypted_2", "enc2", "enc_2", "alt_cipher", "known_cipher"],
        ),
        "bitstream": pick_file(
            bits,
            exact_stems=["bitstream", "keystream", "cipher_dump", "dump"],
            prefixes=["bitstream", "keystream", "cipher_dump", "dump"],
            excludes=["keystream2", "keystream_2", "bitstream2", "bitstream_2", "alt"],
        ),
        "master_key_file": None,
    }

    for name in ("master_key.txt", "key.txt"):
        p = content_dir / name
        if p.is_file():
            assets["master_key_file"] = str(p.resolve())
            break
    return assets


def load_master_key(content_dir: Path, cli_key: Optional[str], notes: List[str]) -> Optional[str]:
    if cli_key:
        notes.append("Using master key from --master_key.")
        return cli_key
    for name in ("master_key.txt", "key.txt"):
        p = content_dir / name
        if p.is_file():
            notes.append(f"Using master key from {p.name}.")
            return p.read_text(encoding="utf-8").strip()
    for name in ("config.json", "dataset.json"):
        p = content_dir / name
        if p.is_file():
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                for key in ("master_key", "key", "secret_key"):
                    val = obj.get(key)
                    if isinstance(val, str) and val.strip():
                        notes.append(f"Using master key from {p.name}:{key}.")
                        return val.strip()
            except Exception as exc:
                notes.append(f"Could not parse {p.name} for a key: {exc}")
    notes.append("No master key found. Tests that require decryption/encryption will be skipped.")
    return None


def framework_cmd(
    framework_path: str,
    mode: str,
    in_path: str,
    out_path: str,
    key: str,
    cipher_dump: Optional[str] = None,
) -> List[str]:
    cmd = [
        sys.executable,
        str(Path(framework_path).resolve()),
        "--mode", mode,
        "--in", str(in_path),
        "--out", str(out_path),
        "--key", str(key),
    ]
    if cipher_dump and mode == "encrypt":
        cmd += ["--cipher_dump", str(cipher_dump)]
    return cmd


def run_command(
    label: str,
    cmd: List[str],
    out_dir: Path,
    report: Dict[str, Any],
    timeout_s: Optional[int] = None,
) -> None:
    if not test_is_enabled(label):
        add_skip(label, selection_reason(label) or "Not selected by --tests/--skip_tests.", report, out_dir)
        return

    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / f"{label}.stdout.txt"
    stderr_path = logs_dir / f"{label}.stderr.txt"

    rec: Dict[str, Any] = {
        "label": label,
        "cmd": cmd,
        "cwd": str(THIS_DIR),
        "started_at": now_iso(),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(THIS_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
        elapsed = time.perf_counter() - t0
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        stdout_path.write_text(stdout, encoding="utf-8", errors="replace")
        stderr_path.write_text(stderr, encoding="utf-8", errors="replace")
        rec.update({
            "returncode": proc.returncode,
            "elapsed_s": elapsed,
            "status": "ok" if proc.returncode == 0 else "failed",
            "stdout_tail": stdout[-5000:],
            "stderr_tail": stderr[-5000:],
        })
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - t0
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", "replace")
        stdout_path.write_text(stdout or "", encoding="utf-8", errors="replace")
        stderr_path.write_text((stderr or "") + f"\nTIMEOUT after {timeout_s} seconds\n", encoding="utf-8", errors="replace")
        rec.update({
            "returncode": None,
            "elapsed_s": elapsed,
            "status": "timeout",
            "timeout_s": timeout_s,
            "stdout_tail": (stdout or "")[-5000:],
            "stderr_tail": ((stderr or "") + f"\nTIMEOUT after {timeout_s} seconds\n")[-5000:],
        })
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        tb = traceback.format_exc()
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(tb, encoding="utf-8", errors="replace")
        rec.update({
            "returncode": None,
            "elapsed_s": elapsed,
            "status": "error",
            "exception": repr(exc),
            "traceback": tb,
            "stderr_tail": tb[-5000:],
        })
    finally:
        rec["finished_at"] = now_iso()
        report["tests"][label] = rec
        write_json(out_dir / "run_all_report.json", report)


def add_skip(label: str, reason: str, report: Dict[str, Any], out_dir: Path) -> None:
    override = selection_reason(label)
    if override:
        reason = override
    report["tests"][label] = {
        "label": label,
        "status": "skipped",
        "reason": reason,
        "finished_at": now_iso(),
    }
    write_json(out_dir / "run_all_report.json", report)


def script_path(script_name: str) -> Path:
    return THIS_DIR / script_name


def run_test(
    label: str,
    script_name: str,
    args: List[str],
    report: Dict[str, Any],
    out_dir: Path,
    notes: List[str],
    timeout_s: Optional[int] = None,
) -> None:
    if not test_is_enabled(label):
        add_skip(label, selection_reason(label) or "Not selected by --tests/--skip_tests.", report, out_dir)
        return
    sp = script_path(script_name)
    if not sp.is_file():
        add_skip(label, f"Missing script: {sp}", report, out_dir)
        return
    run_command(label, [sys.executable, str(sp)] + args, out_dir, report, timeout_s=timeout_s)


def finalize_summary(report: Dict[str, Any], out_dir: Path) -> None:
    counts: Dict[str, int] = {}
    for rec in report.get("tests", {}).values():
        status = str(rec.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    report["summary"] = counts
    if counts.get("failed", 0) or counts.get("error", 0) or counts.get("timeout", 0):
        report["overall_status"] = "completed_with_failures"
    elif counts.get("ok", 0):
        report["overall_status"] = "ok"
    else:
        report["overall_status"] = "no_tests_ran"
    report["finished_at"] = now_iso()
    write_json(out_dir / "run_all_report.json", report)


def main() -> int:
    global SELECTED_TESTS, SKIPPED_TESTS

    ap = argparse.ArgumentParser(description="Run full-frame tests. Defaults to content3 -> content3/unified_results.")
    ap.add_argument("content_dir_pos", nargs="?", default=None, help="Optional compatibility positional path. Not required; defaults to content3.")
    ap.add_argument("--content_dir", default=None, help="Optional override for the content folder. Not required.")
    ap.add_argument("--dataset_dir", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out", default=None, help="Optional output directory. Default: <content3>/unified_results")
    ap.add_argument("--results_name", default=DEFAULT_RESULTS_DIR_NAME, help="Folder name inside content3 when --out is not used.")
    ap.add_argument("--framework_path", default=str(THIS_DIR / "framework_full_encrypt.py"))
    ap.add_argument("--master_key", default=None)
    ap.add_argument("--max_frames", type=int, default=500)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--skip_detector", action="store_true")
    ap.add_argument("--skip_audio", action="store_true")
    ap.add_argument("--skip_slow", action="store_true")
    ap.add_argument("--generate_cipher_if_missing", action="store_true")
    ap.add_argument("--generate_decrypted_if_missing", action="store_true")
    ap.add_argument("--timeout_per_test", type=int, default=0, help="Optional timeout in seconds for each test; 0 = no timeout.")
    ap.add_argument("--tests", default=None, help="Run only selected tests. Examples: --tests 01,02,10 or --tests quality,nist,detector")
    ap.add_argument("--skip_tests", default=None, help="Skip selected tests. Examples: --skip_tests audio,detector or --skip_tests 12,13")
    ap.add_argument("--list_tests", action="store_true", help="List available test names and aliases, then exit.")
    args = ap.parse_args()

    if args.list_tests:
        print("Available concrete test labels:")
        for label in ALL_TEST_LABELS:
            print(f"  {label}")
        print("\nUseful aliases:")
        for alias in sorted(TEST_ALIASES):
            print(f"  {alias} -> {', '.join(TEST_ALIASES[alias])}")
        return 0

    SELECTED_TESTS = expand_test_selection(args.tests)
    SKIPPED_TESTS = expand_test_selection(args.skip_tests) or set()

    # Final full-frame suite: disable ROI-style/privacy-ablation tests by default.
    # Test 13 is only a detector sanity baseline for full-frame encryption,
    # and Test 17 is a corruption ablation, not the ROI mask-expansion test.
    SKIPPED_TESTS.update({
        "13_detector_privacy_leakage",
        "17_ctr_corruption_ablation",
    })

    cli_content = args.content_dir or args.dataset_dir or args.content_dir_pos
    content_dir = resolve_content_dir(cli_content)
    out_dir = Path(args.out).resolve() if args.out else (content_dir / args.results_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    notes: List[str] = [
        "Full-frame suite: no rois.jsonl, payload, or manifest is used.",
        f"Default content folder resolution chose: {content_dir}",
        f"Results folder: {out_dir}",
    ]
    if SELECTED_TESTS is not None:
        notes.append("Selected tests: " + ", ".join(sorted(SELECTED_TESTS)))
    if SKIPPED_TESTS:
        notes.append("Explicitly skipped tests: " + ", ".join(sorted(SKIPPED_TESTS)))

    report: Dict[str, Any] = {
        "created_at": now_iso(),
        "runner": str(Path(__file__).resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "content_dir": str(content_dir),
        "out_dir": str(out_dir),
        "framework_path": str(Path(args.framework_path).resolve()),
        "args": vars(args),
        "selected_tests": sorted(SELECTED_TESTS) if SELECTED_TESTS is not None else None,
        "skipped_tests": sorted(SKIPPED_TESTS),
        "assets": {},
        "notes": notes,
        "tests": {},
        "summary": {},
        "overall_status": "running",
    }
    write_json(out_dir / "run_all_report.json", report)

    try:
        if not content_dir.is_dir():
            notes.append(f"Content folder does not exist: {content_dir}")
            report["setup_error"] = f"Content folder does not exist: {content_dir}"
            finalize_summary(report, out_dir)
            print(f"[ERROR] Content folder not found. Report: {out_dir / 'run_all_report.json'}")
            return 2

        assets = discover_assets(content_dir)
        key = load_master_key(content_dir, args.master_key, notes)
        report["assets"] = assets.copy()
        report["master_key_present"] = bool(key)
        write_json(out_dir / "run_all_report.json", report)

        framework_path = str(Path(args.framework_path).resolve())
        timeout_s = int(args.timeout_per_test or 0) or None

        if not assets.get("plain"):
            notes.append("Missing plain/original/source video. Most tests will be skipped.")

        if not assets.get("cipher") and args.generate_cipher_if_missing and assets.get("plain") and key:
            gen_cipher = out_dir / "generated_cipher.mp4"
            gen_dump = out_dir / "generated_cipher_dump.bin"
            run_command(
                "00_generate_cipher",
                framework_cmd(framework_path, "encrypt", assets["plain"], str(gen_cipher), key, cipher_dump=str(gen_dump)),
                out_dir,
                report,
                timeout_s=timeout_s,
            )
            if gen_cipher.is_file():
                assets["cipher"] = str(gen_cipher.resolve())
            if gen_dump.is_file():
                assets["bitstream"] = str(gen_dump.resolve())
            report["assets"] = assets.copy()
            write_json(out_dir / "run_all_report.json", report)

        if not assets.get("cipher"):
            notes.append("Missing cipher/encrypted/enc video. Cipher-dependent tests will be skipped.")

        if not assets.get("decrypted") and args.generate_decrypted_if_missing and assets.get("cipher") and key:
            gen_dec = out_dir / "generated_decrypted.mp4"
            run_command(
                "00_generate_decrypted",
                framework_cmd(framework_path, "decrypt", assets["cipher"], str(gen_dec), key),
                out_dir,
                report,
                timeout_s=timeout_s,
            )
            if gen_dec.is_file():
                assets["decrypted"] = str(gen_dec.resolve())
            report["assets"] = assets.copy()
            write_json(out_dir / "run_all_report.json", report)

        plain = assets.get("plain")
        cipher = assets.get("cipher")
        decrypted = assets.get("decrypted")
        common = ["--max_frames", str(args.max_frames), "--stride", str(args.stride)]

        if plain and cipher:
            run_test("01_quality_plain_vs_cipher", "01_quality_metrics_roi.py", ["--plain", plain, "--test", cipher, "--out", str(out_dir / "01_quality_plain_vs_cipher"), "--mode", "gray"] + common, report, out_dir, notes, timeout_s)
        else:
            add_skip("01_quality_plain_vs_cipher", "Need plain and cipher videos.", report, out_dir)

        if plain and decrypted:
            run_test("01_quality_plain_vs_decrypted", "01_quality_metrics_roi.py", ["--plain", plain, "--test", decrypted, "--out", str(out_dir / "01_quality_plain_vs_decrypted"), "--mode", "gray"] + common, report, out_dir, notes, timeout_s)
        else:
            add_skip("01_quality_plain_vs_decrypted", "Need plain and decrypted videos.", report, out_dir)

        if plain and cipher:
            cmd02 = ["--plain", plain, "--cipher", cipher, "--out", str(out_dir / "02_differential"), "--mode", "color", "--framework_path", framework_path] + common
            if assets.get("plain2"):
                cmd02 += ["--plain2", assets["plain2"]]
            if assets.get("cipher2"):
                cmd02 += ["--cipher2", assets["cipher2"]]
            if key:
                cmd02 += ["--master_key", key, "--run_plaintext_sensitivity", "--run_key_sensitivity"]
            run_test("02_differential", "02_differential_sensitivity_roi.py", cmd02, report, out_dir, notes, timeout_s)
        else:
            add_skip("02_differential", "Need plain and cipher videos.", report, out_dir)

        if plain and cipher:
            run_test("03_correlation", "03_correlation_tests_roi.py", ["--plain", plain, "--cipher", cipher, "--out", str(out_dir / "03_correlation")] + common, report, out_dir, notes, timeout_s)
            run_test("04_hist_entropy", "04_hist_entropy_roi.py", ["--plain", plain, "--cipher", cipher, "--out", str(out_dir / "04_hist_entropy")] + common, report, out_dir, notes, timeout_s)
            run_test("05_psd", "05_psd_estimation.py", ["--plain", plain, "--video", cipher, "--out", str(out_dir / "05_psd")] + common, report, out_dir, notes, timeout_s)
            cmd07 = ["--plain", plain, "--cipher", cipher, "--out", str(out_dir / "07_bandwidth")]
            if decrypted:
                cmd07 += ["--decrypted", decrypted]
            run_test("07_bandwidth", "07_bandwidth_analysis.py", cmd07, report, out_dir, notes, timeout_s)
            cmd09 = ["--plain", plain, "--cipher", cipher, "--out", str(out_dir / "09_attack_eq"), "--mode", "color", "--framework_path", framework_path] + common
            if key:
                cmd09 += ["--master_key", key, "--run_cpa", "--run_cca"]
            run_test("09_attack_eq", "09_missing_attack_eq_suite.py", cmd09, report, out_dir, notes, timeout_s)
        else:
            for label in ("03_correlation", "04_hist_entropy", "05_psd", "07_bandwidth", "09_attack_eq"):
                add_skip(label, "Need plain and cipher videos.", report, out_dir)

        # NIST handling. Default preserves old behavior: bitstream if present, else decoded cipher frames.
        # With --tests, you may explicitly choose either label or use alias 10/nist for both.
        if SELECTED_TESTS is None:
            if assets.get("bitstream"):
                run_test("10_nist", "10_nist_runner.py", ["--bitstream", assets["bitstream"], "--out", str(out_dir / "10_nist")], report, out_dir, notes, timeout_s)
                add_skip("10_nist_decoded_cipher_frames", "Default NIST path used bitstream/keystream; decoded cipher-frame NIST was not run.", report, out_dir)
            elif cipher:
                add_skip("10_nist", "No bitstream/keystream file found; using decoded cipher frames instead.", report, out_dir)
                run_test("10_nist_decoded_cipher_frames", "10_nist_runner.py", ["--cipher_video", cipher, "--out", str(out_dir / "10_nist")] + common, report, out_dir, notes, timeout_s)
            else:
                add_skip("10_nist", "Need a bitstream/keystream file.", report, out_dir)
                add_skip("10_nist_decoded_cipher_frames", "Need a cipher video.", report, out_dir)
        else:
            if "10_nist" in SELECTED_TESTS:
                if assets.get("bitstream"):
                    run_test("10_nist", "10_nist_runner.py", ["--bitstream", assets["bitstream"], "--out", str(out_dir / "10_nist")], report, out_dir, notes, timeout_s)
                else:
                    add_skip("10_nist", "Need a bitstream/keystream file.", report, out_dir)
            else:
                add_skip("10_nist", "Not selected by --tests.", report, out_dir)

            if "10_nist_decoded_cipher_frames" in SELECTED_TESTS:
                if cipher:
                    run_test("10_nist_decoded_cipher_frames", "10_nist_runner.py", ["--cipher_video", cipher, "--out", str(out_dir / "10_nist_decoded_cipher_frames")] + common, report, out_dir, notes, timeout_s)
                else:
                    add_skip("10_nist_decoded_cipher_frames", "Need a cipher video.", report, out_dir)
            else:
                add_skip("10_nist_decoded_cipher_frames", "Not selected by --tests.", report, out_dir)

        if key and not args.skip_slow and plain and cipher:
            run_test("06_robustness", "06_robustness_noise_occlusion_roi.py", ["--plain", plain, "--cipher", cipher, "--out", str(out_dir / "06_robustness"), "--framework_path", framework_path, "--master_key", key] + common, report, out_dir, notes, timeout_s)
            cmd08 = ["--plain", plain, "--out", str(out_dir / "08_perf"), "--framework_path", framework_path, "--master_key", key, "--runs", "1"]
            if cipher:
                cmd08 += ["--cipher", cipher]
            run_test("08_perf", "08_perf_cycles_throughput.py", cmd08, report, out_dir, notes, timeout_s)
            run_test("11_param_sweep", "11_param_sweep.py", ["--plain", plain, "--out", str(out_dir / "11_param_sweep"), "--framework_path", framework_path, "--master_key", key, "--variants", "3"] + common, report, out_dir, notes, timeout_s)
        else:
            reason = "Need master key, plain, cipher, and --skip_slow must be false."
            if args.skip_slow:
                reason = "--skip_slow was set."
            for label in ("06_robustness", "08_perf", "11_param_sweep"):
                add_skip(label, reason, report, out_dir)

        if not args.skip_audio and plain and cipher:
            cmd12 = ["--plain", plain, "--cipher", cipher, "--out", str(out_dir / "12_audio")]
            if decrypted:
                cmd12 += ["--decrypted", decrypted]
            run_test("12_audio", "12_audio_eval_from_videos_runall.py", cmd12, report, out_dir, notes, timeout_s)
        else:
            add_skip("12_audio", "--skip_audio was set or plain/cipher is missing.", report, out_dir)

        if not args.skip_detector and plain and cipher:
            cmd13 = ["--plain", plain, "--cipher", cipher, "--out", str(out_dir / "13_detector")] + common
            if decrypted:
                cmd13 += ["--decrypted", decrypted]
            run_test("13_detector_privacy_leakage", "13_detector_privacy_leakage.py", cmd13, report, out_dir, notes, timeout_s)
        else:
            add_skip("13_detector_privacy_leakage", "--skip_detector was set or plain/cipher is missing.", report, out_dir)

        # Additional full-frame AES-CTR tests. These are deliberately CTR-specific:
        # they measure malleability/corruption and IV uniqueness, not AEAD rejection.
        if key and plain and cipher:
            run_test("14_ctr_tamper_malleability", "14_full_frame_ctr_tamper_malleability.py", ["--plain", plain, "--cipher", cipher, "--out", str(out_dir / "14_ctr_tamper"), "--framework_path", framework_path, "--master_key", key] + common, report, out_dir, notes, timeout_s)
            run_test("15_ctr_replay_substitution", "15_full_frame_ctr_replay_substitution.py", ["--plain", plain, "--cipher", cipher, "--out", str(out_dir / "15_ctr_replay"), "--framework_path", framework_path, "--master_key", key] + common, report, out_dir, notes, timeout_s)
        else:
            reason = "Need master key, plain, and cipher videos."
            for label in ("14_ctr_tamper_malleability", "15_ctr_replay_substitution", "17_ctr_corruption_ablation"):
                add_skip(label, reason, report, out_dir)

        if key and plain:
            run_test("16_ctr_iv_context_binding", "16_full_frame_ctr_iv_context_binding.py", ["--plain", plain, "--out", str(out_dir / "16_ctr_iv_context"), "--framework_path", framework_path, "--master_key", key, "--master_key_alt", key + "|alt_iv_test", "--max_frames", str(args.max_frames)], report, out_dir, notes, timeout_s)
        else:
            add_skip("16_ctr_iv_context_binding", "Need master key and plain video.", report, out_dir)

        add_skip("17_ctr_corruption_ablation", "Disabled for the final full-frame suite.", report, out_dir)

        if plain and cipher:
            cmd18 = ["--plain", plain, "--cipher", cipher, "--out", str(out_dir / "18_storage_overhead")]
            if decrypted:
                cmd18 += ["--decrypted", decrypted]
            if assets.get("bitstream"):
                cmd18 += ["--bitstream", assets["bitstream"]]
            run_test("18_storage_overhead", "18_full_frame_storage_overhead.py", cmd18, report, out_dir, notes, timeout_s)
        else:
            add_skip("18_storage_overhead", "Need plain and cipher videos.", report, out_dir)

        finalize_summary(report, out_dir)
        print(f"[DONE] Report written to: {out_dir / 'run_all_report.json'}")
        print(f"[DONE] Logs written to:   {out_dir / 'logs'}")
        print(f"[DONE] Summary: {report.get('summary')}")
        return 0 if report.get("overall_status") == "ok" else 1

    except Exception as exc:
        tb = traceback.format_exc()
        report["fatal_error"] = repr(exc)
        report["fatal_traceback"] = tb
        notes.append(f"Fatal runner error: {exc}")
        finalize_summary(report, out_dir)
        (out_dir / "logs").mkdir(parents=True, exist_ok=True)
        (out_dir / "logs" / "fatal_error.stderr.txt").write_text(tb, encoding="utf-8", errors="replace")
        print(f"[FATAL] {exc}")
        print(f"[FATAL] Report written to: {out_dir / 'run_all_report.json'}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
