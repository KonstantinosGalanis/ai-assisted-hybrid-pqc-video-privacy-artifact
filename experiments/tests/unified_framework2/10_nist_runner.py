#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List

import math
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR / 'vendor_nist'))

import sp800_22_approximate_entropy_test as t_approx
import sp800_22_binary_matrix_rank_test as t_rank
import sp800_22_cumulative_sums_test as t_cumsum
import sp800_22_dft_test as t_dft
import sp800_22_frequency_within_block_test as t_block
import sp800_22_linear_complexity_test as t_lin
import sp800_22_longest_run_ones_in_a_block_test as t_longest
import sp800_22_maurers_universal_test as t_universal
import sp800_22_monobit_test as t_monobit
import sp800_22_non_overlapping_template_matching_test as t_nonover
import sp800_22_overlapping_template_matching_test as t_over
import sp800_22_random_excursion_test as t_rexc
import sp800_22_random_excursion_variant_test as t_rexcv
import sp800_22_runs_test as t_runs
import sp800_22_serial_test as t_serial

from unified_utils import write_bytes, load_sidecar_records, derive_used_keystream_bytes


def import_framework_module(path: str):
    import importlib.util

    resolved = Path(path).resolve()
    name = "framework_faster"
    if name in sys.modules:
        mod = sys.modules[name]
        if Path(getattr(mod, "__file__", "")).resolve() == resolved:
            return mod
        sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, str(resolved))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {resolved}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_bits_limited(path: str, max_bits: int, big_endian: bool = False):
    """Read only as many bytes as needed for max_bits, then unpack safely.
    Returns a Python list because the bundled NIST functions expect list-like integer bits.
    """
    p = Path(path)
    if max_bits and max_bits > 0:
        bytes_needed = int(math.ceil(max_bits / 8.0))
        data = p.read_bytes()[:bytes_needed]
    else:
        data = p.read_bytes()
    arr = np.frombuffer(data, dtype=np.uint8)
    bitorder = "big" if big_endian else "little"
    bits = np.unpackbits(arr, bitorder=bitorder)
    if max_bits and max_bits > 0:
        bits = bits[:max_bits]
    return bits.astype(np.uint8, copy=False).tolist()


def _extract_cipher_payload_from_jsonl_dump(dump_path: str, out_dir: Path) -> Dict[str, str]:
    """Extract raw encrypted payload bytes from a JSONL dump.

    Supports both older top-level payload fields and the current ChaCha payload
    JSONL layout where ciphertext is stored inside frame records under:
      - rois[*].cipher_b64
      - audio.cipher_b64
      - tail_audio.cipher_b64

    Also accepts older top-level base64 field names:
      - cipher_payload_b64
      - ciphertext_b64
      - encrypted_payload_b64
      - payload_b64
      - cipher_b64
    """
    supported_fields = (
        "cipher_payload_b64",
        "ciphertext_b64",
        "encrypted_payload_b64",
        "payload_b64",
        "cipher_b64",
    )

    def _append_b64(parts: List[bytes], b64: str, where: str, lineno: int) -> None:
        try:
            parts.append(base64.b64decode(b64, validate=True))
        except Exception as e:
            raise SystemExit(f"Bad base64 in {where} at line {lineno}: {e}") from e

    payload_parts: List[bytes] = []
    seen_any_payload = False
    saw_old_debug_records = False
    saw_current_payload_header = False

    with open(dump_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except json.JSONDecodeError as e:
                raise SystemExit(f"Invalid JSONL dump at line {lineno}: {e}") from e

            if rec.get("type") == "chaos_psd_transform_debug_v1":
                saw_old_debug_records = True
            if rec.get("type") == "meta" and rec.get("crypto_scheme"):
                saw_current_payload_header = True

            found = False
            for field in supported_fields:
                b64 = rec.get(field)
                if isinstance(b64, str) and b64:
                    _append_b64(payload_parts, b64, f"field '{field}'", lineno)
                    found = True
                    seen_any_payload = True
                    break
            if found:
                continue
            if any(k in rec for k in supported_fields):
                raise SystemExit(f"Record at line {lineno} has a payload field but it is empty or invalid.")

            rois = rec.get("rois")
            if isinstance(rois, list):
                for idx, roi in enumerate(rois):
                    if not isinstance(roi, dict):
                        continue
                    b64 = roi.get("cipher_b64")
                    if isinstance(b64, str) and b64:
                        _append_b64(payload_parts, b64, f"rois[{idx}].cipher_b64", lineno)
                        seen_any_payload = True

            audio = rec.get("audio")
            if isinstance(audio, dict):
                b64 = audio.get("cipher_b64")
                if isinstance(b64, str) and b64:
                    _append_b64(payload_parts, b64, "audio.cipher_b64", lineno)
                    seen_any_payload = True

    if not seen_any_payload:
        if saw_current_payload_header:
            raise SystemExit(
                "The provided payload JSONL was recognized, but no encrypted ROI/audio payload bytes were found. "
                "Expected fields such as rois[*].cipher_b64, audio.cipher_b64, or tail_audio.cipher_b64."
            )
        if saw_old_debug_records:
            raise SystemExit(
                "The provided dump looks like an old debug dump and does not contain raw encrypted payload bytes. "
                "Use --cipher for a ciphertext file, --cipher_payload for a raw payload dump, or pass the current payload.jsonl file."
            )
        raise SystemExit(
            "No encrypted payload field found in the JSONL dump. Supported locations include: "
            "rois[*].cipher_b64, audio.cipher_b64, tail_audio.cipher_b64, "
            + ", ".join(supported_fields)
        )

    out_path = write_bytes(str(out_dir / "derived_cipher_payload.bin"), b"".join(payload_parts))
    return {
        "source": "cipher_payload_dump_bytes",
        "path": out_path,
        "derived_path": out_path,
        "notes": "Extracted concatenated encrypted payload bytes from JSONL dump.",
    }


def resolve_bytes(args) -> Dict[str, str]:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cipher_payload:
        return {
            "source": "cipher_payload_bytes",
            "path": args.cipher_payload,
            "derived_path": args.cipher_payload,
            "notes": "Preferred mode: tests raw encrypted payload bytes instead of whole-container video bytes.",
        }
    if args.cipher_payload_dump:
        return _extract_cipher_payload_from_jsonl_dump(args.cipher_payload_dump, out_dir)
    if args.payload:
        return _extract_cipher_payload_from_jsonl_dump(args.payload, out_dir)
    if args.cipher:
        return {
            "source": "cipher_file_bytes",
            "path": args.cipher,
            "derived_path": args.cipher,
            "notes": "Whole-file ciphertext mode. This may include container / codec headers and unchanged bytes.",
        }
    if args.bitstream:
        return {
            "source": "bitstream",
            "path": args.bitstream,
            "derived_path": args.bitstream,
            "notes": "Generic raw bitstream input.",
        }
    if args.keystream:
        return {
            "source": "compatibility_keystream_bytes",
            "path": args.keystream,
            "derived_path": args.keystream,
            "notes": "Compatibility mode for raw keystream bytes only; encrypted payload bytes are preferred.",
        }
    if args.roi_sidecar and args.master_key and args.framework_faster_path:
        ff = import_framework_module(args.framework_faster_path)
        recs = load_sidecar_records(args.roi_sidecar)
        data = derive_used_keystream_bytes(recs, ff, args.master_key)
        out_path = write_bytes(str(out_dir / 'derived_keystream.bin'), data)
        return {
            'source': 'compatibility_derived_keystream_from_sidecar',
            'path': out_path,
            'derived_path': out_path,
            'notes': 'Compatibility-only path. Prefer --payload, --cipher_payload_dump, --cipher_payload, or --cipher.',
        }
    raise SystemExit(
        'Need one of: --payload, --cipher_payload, --cipher_payload_dump, --cipher, --bitstream, '
        '--keystream (compatibility), or (--roi_sidecar + --master_key + --framework_faster_path) (compatibility).'
    )


def safe_call(func: Callable[[List[int]], tuple], bits: List[int]):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        success, p, plist = func(bits)
    return success, p, plist, buf.getvalue()


def main():
    ap = argparse.ArgumentParser(
        description='NIST SP800-22 runner. Preferred input is encrypted payload bytes from the current ChaCha payload framework.'
    )
    ap.add_argument('--out', required=True)
    ap.add_argument('--cipher_payload', default=None, help='raw encrypted payload bytes (preferred)')
    ap.add_argument('--cipher_payload_dump', default=None, help='JSONL dump that contains encrypted payload base64 fields')
    ap.add_argument('--cipher', default=None, help='whole ciphertext file (for example encrypted video file bytes)')
    ap.add_argument('--bitstream', default=None, help='generic raw bitstream file')
    ap.add_argument('--keystream', default=None, help='compatibility raw keystream bytes')
    ap.add_argument('--payload', default=None, help='current payload.jsonl file produced by the ChaCha payload framework')
    ap.add_argument('--roi_sidecar', default=None, help='compatibility path: derive raw keystream from ROI sidecar')
    ap.add_argument('--master_key', default=None)
    ap.add_argument('--framework_faster_path', default=None)
    ap.add_argument('--big_endian', action='store_true')
    ap.add_argument('--max_bits', type=int, default=2_000_000, help='0 = all bits')
    args = ap.parse_args()

    resolved = resolve_bytes(args)
    bits = read_bits_limited(resolved['path'], max_bits=args.max_bits, big_endian=args.big_endian)

    tests = [
        ('monobit', t_monobit.monobit_test),
        ('block_frequency', t_block.frequency_within_block_test),
        ('runs', t_runs.runs_test),
        ('longest_runs', t_longest.longest_run_ones_in_a_block_test),
        ('rank', t_rank.binary_matrix_rank_test),
        ('dft', t_dft.dft_test),
        ('non_overlapping_template_matching', t_nonover.non_overlapping_template_matching_test),
        ('overlapping_template_matching', t_over.overlapping_template_matching_test),
        ('universal', t_universal.maurers_universal_test),
        ('linear_complexity', t_lin.linear_complexity_test),
        ('serial', t_serial.serial_test),
        ('approximate_entropy', t_approx.approximate_entropy_test),
        ('cumulative_sums', t_cumsum.cumulative_sums_test),
        ('random_excursions', t_rexc.random_excursion_test),
        ('random_excursions_variant', t_rexcv.random_excursion_variant_test),
    ]

    report = {
        'suite': 'NIST SP800-22 Rev1a wrapper',
        'bit_source': resolved['source'],
        'input_path': resolved['path'],
        'bits_tested': len(bits),
        'notes': resolved.get('notes', ''),
        'tests': {},
        'nist_pdf_mapping': {
            '1': 'Monobit', '2': 'Block frequency', '3': 'Runs', '4': 'Longest Runs',
            '5': 'Rank', '6': 'DFT', '7': 'Non-overlapping and overlapping template matching',
            '8': 'Universal', '9': 'Linear Complexity', '10': 'Serial 1/2',
            '11': 'Approximate Entropy', '12': 'Cumulative Sums Forward',
            '13': 'Cumulative Sums Backward', '14': 'Random Excursions', '15': 'Random Excursions Variant'
        },
    }

    for name, func in tests:
        success, p, plist, captured = safe_call(func, bits)
        entry = {'success': bool(success), 'stdout': captured.strip()}
        if p is not None:
            entry['p_value'] = float(p)
        if plist is not None:
            entry['p_values'] = [float(x) for x in plist]
        report['tests'][name] = entry

    report['pdf_table'] = {
        'monobit': report['tests']['monobit'],
        'block_frequency': report['tests']['block_frequency'],
        'runs': report['tests']['runs'],
        'longest_runs': report['tests']['longest_runs'],
        'rank': report['tests']['rank'],
        'dft': report['tests']['dft'],
        'non_overlapping_template_matching': report['tests']['non_overlapping_template_matching'],
        'overlapping_template_matching': report['tests']['overlapping_template_matching'],
        'universal': report['tests']['universal'],
        'linear_complexity': report['tests']['linear_complexity'],
        'serial_1_2': report['tests']['serial'],
        'approximate_entropy': report['tests']['approximate_entropy'],
        'cumulative_sums_forward_backward': report['tests']['cumulative_sums'],
        'random_excursions': report['tests']['random_excursions'],
        'random_excursions_variant': report['tests']['random_excursions_variant'],
    }

    out_dir = Path(args.out)
    (out_dir / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    import csv
    with open(out_dir / 'summary.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['test', 'success', 'p_value', 'p_values'])
        for name, entry in report['tests'].items():
            w.writerow([name, entry.get('success'), entry.get('p_value'), '|'.join(str(x) for x in entry.get('p_values', []))])

    print(json.dumps({'ok': True, 'out': str(out_dir / 'report.json'), 'bits_tested': len(bits), 'source': resolved['source']}))


if __name__ == '__main__':
    main()
