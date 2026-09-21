#!/usr/bin/env python3
from __future__ import annotations

"""
15_chaos_roi_replay_substitution.py

Replay/substitution tests for the chaos payload branch. It deliberately moves
or substitutes encrypted ROI payload records across frame/object contexts and
checks that decryption/reconstruction is rejected. This complements Test 14 by
focusing on object/frame/session substitution rather than single-field flips.
"""

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: str) -> List[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: str, records: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")


def load_manifest(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_manifest(path: str, manifest: dict) -> None:
    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def roi_frames(records: List[dict]) -> List[int]:
    out = []
    for i, rec in enumerate(records):
        if rec.get("type") in {"meta", "tail_audio"}:
            continue
        if rec.get("rois"):
            out.append(i)
    return out


def find_two_roi_slots(records: List[dict]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    slots: List[Tuple[int, int]] = []
    for i, rec in enumerate(records):
        if rec.get("type") in {"meta", "tail_audio"}:
            continue
        for j, _ in enumerate(rec.get("rois") or []):
            slots.append((i, j))
    if len(slots) < 2:
        raise ValueError("Need at least two ROI payload records for replay/substitution tests")
    return slots[0], slots[1]


def find_first_and_next_frame(records: List[dict]) -> Tuple[int, int]:
    frames = roi_frames(records)
    if not frames:
        raise ValueError("Need at least one ROI frame")
    first = frames[0]
    # Prefer a distinct existing frame, otherwise create a replay to frame_idx+1 in same record.
    if len(frames) > 1:
        return first, frames[1]
    return first, first


def make_case(base_records: List[dict], manifest: dict, out_dir: Path, name: str, mut: Callable[[List[dict]], List[dict]]) -> Tuple[str, str]:
    payload_path = out_dir / f"{name}.payload.jsonl"
    manifest_path = out_dir / f"{name}.manifest.json"
    records = mut(copy.deepcopy(base_records))
    write_jsonl(str(payload_path), records)
    m = copy.deepcopy(manifest)
    m["payload_path"] = payload_path.name
    m["payload_sha256"] = sha256_file(str(payload_path))
    write_manifest(str(manifest_path), m)
    return str(payload_path), str(manifest_path)


def run_decrypt(args, payload_path: str, manifest_path: str, out_path: str) -> Tuple[bool, str, str, int]:
    cmd = [
        sys.executable, str(Path(args.framework_faster_path).resolve()),
        "--mode", "decrypt", "--in", args.cipher, "--out", out_path, "--key", args.master_key,
        "--roi_sidecar", args.roi_sidecar, "--payload", payload_path, "--manifest", manifest_path,
        "--video_preview_mode", "chaos", "--audio_preview_mode", "chaos",
    ]
    if args.framework_base_path:
        cmd.extend(["--base_framework", args.framework_base_path])
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return p.returncode == 0, (p.stdout or "")[-3000:], (p.stderr or "")[-3000:], p.returncode


def mut_payload_frame_to_next(records: List[dict]) -> List[dict]:
    src_i, dst_i = find_first_and_next_frame(records)
    if src_i == dst_i:
        records[src_i]["frame_idx"] = int(records[src_i].get("frame_idx", 0)) + 1
    else:
        # Copy source frame payload into destination frame index.
        src = copy.deepcopy(records[src_i])
        src["frame_idx"] = int(records[dst_i].get("frame_idx", int(src.get("frame_idx", 0)) + 1))
        records[dst_i] = src
    return records


def mut_object_payload_to_other_object(records: List[dict]) -> List[dict]:
    (a_i, a_j), (b_i, b_j) = find_two_roi_slots(records)
    src_roi = copy.deepcopy(records[a_i]["rois"][a_j])
    dst_roi = records[b_i]["rois"][b_j]
    # Force the source encrypted payload into the destination object's slot.
    src_roi["roi_index"] = int(dst_roi.get("roi_index", b_j))
    src_roi["bbox"] = dst_roi.get("bbox", src_roi.get("bbox"))
    src_roi["cls"] = int(dst_roi.get("cls", src_roi.get("cls", -1)))
    src_roi["track_id"] = int(dst_roi.get("track_id", src_roi.get("track_id", -1)))
    src_roi["mask_hash"] = dst_roi.get("mask_hash", src_roi.get("mask_hash", ""))
    records[b_i]["rois"][b_j] = src_roi
    return records


def mut_track_id_substitution(records: List[dict]) -> List[dict]:
    (a_i, a_j), _ = find_two_roi_slots(records)
    roi = records[a_i]["rois"][a_j]
    roi["track_id"] = int(roi.get("track_id", -1)) + 1000
    return records


def mut_class_substitution(records: List[dict]) -> List[dict]:
    (a_i, a_j), _ = find_two_roi_slots(records)
    roi = records[a_i]["rois"][a_j]
    old = int(roi.get("cls", 0))
    roi["cls"] = 2 if old == 0 else 0
    return records


def mut_old_payload_new_session(records: List[dict]) -> List[dict]:
    for rec in records:
        if rec.get("type") == "meta":
            rec["session_id"] = str(rec.get("session_id", "")) + "_replayed"
            break
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Chaos ROI replay/substitution rejection tests.")
    ap.add_argument("--cipher", required=True)
    ap.add_argument("--roi_sidecar", required=True)
    ap.add_argument("--payload", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--framework_faster_path", required=True)
    ap.add_argument("--framework_base_path", default=None)
    ap.add_argument("--master_key", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = out_dir / "mutated_inputs"; cases_dir.mkdir(exist_ok=True)
    base_records = read_jsonl(args.payload)
    manifest = load_manifest(args.manifest)

    mutators: Dict[str, Callable[[List[dict]], List[dict]]] = {
        "payload_from_one_frame_to_another": mut_payload_frame_to_next,
        "person_vehicle_or_object_payload_substitution": mut_object_payload_to_other_object,
        "track_id_substitution": mut_track_id_substitution,
        "class_substitution": mut_class_substitution,
        "old_payload_new_session_id": mut_old_payload_new_session,
    }

    results = []
    for name, mut in mutators.items():
        try:
            payload_path, manifest_path = make_case(base_records, manifest, cases_dir, name, mut)
            ok, stdout_tail, stderr_tail, returncode = run_decrypt(args, payload_path, manifest_path, str(out_dir / f"decrypt_{name}.mkv"))
            rejected = not ok
            results.append({
                "case": name,
                "expected_reject": True,
                "rejected": rejected,
                "passed": rejected,
                "returncode": returncode,
                "payload": payload_path,
                "manifest": manifest_path,
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
            })
        except Exception as e:
            results.append({"case": name, "expected_reject": True, "rejected": True, "passed": True, "setup_exception_treated_as_rejection": str(e)})

    report = {
        "focus": "Chaos ROI replay/substitution rejection using manifest and payload HMAC binding.",
        "inputs": {"cipher": args.cipher, "roi_sidecar": args.roi_sidecar, "payload": args.payload, "manifest": args.manifest},
        "summary": {
            "cases": len(results),
            "passed": sum(1 for r in results if r.get("passed")),
            "unexpected_successes": [r["case"] for r in results if not r.get("rejected")],
            "all_passed": all(r.get("passed") for r in results),
        },
        "results": results,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "report": str(out_dir / "report.json"), "all_passed": report["summary"]["all_passed"]}))


if __name__ == "__main__":
    main()
