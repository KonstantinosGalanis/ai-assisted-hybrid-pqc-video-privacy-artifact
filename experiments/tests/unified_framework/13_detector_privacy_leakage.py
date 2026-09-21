#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

CLASS_NAMES = {
    0: "person",
    2: "car",
    3: "motorbike",
    5: "bus",
    7: "truck",
}


def stable_import_module(path: str, canonical_name: str):
    resolved = Path(path).resolve()
    existing = sys.modules.get(canonical_name)
    if existing is not None and Path(getattr(existing, "__file__", "")).resolve() == resolved:
        return existing
    spec = importlib.util.spec_from_file_location(canonical_name, str(resolved))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {resolved}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[canonical_name] = mod
    spec.loader.exec_module(mod)
    return mod


def vprint(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg, flush=True)


@dataclass
class RoiObj:
    frame_idx: int
    bbox: Tuple[int, int, int, int]
    mask: np.ndarray
    cls_id: int
    track_id: int
    conf: float


class VideoReader:
    def __init__(self, path: str, max_frames: int, stride: int):
        self.path = path
        self.max_frames = max_frames
        self.stride = stride

    def iter_frames(self):
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {self.path}")
        orig_idx = 0
        proc_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if self.stride > 1 and (orig_idx % self.stride) != 0:
                orig_idx += 1
                continue
            yield proc_idx, orig_idx, frame
            proc_idx += 1
            orig_idx += 1
            if self.max_frames and proc_idx >= self.max_frames:
                break
        cap.release()



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


def _build_orig_to_proc(max_frames: int, stride: int) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    proc = 0
    orig = 0
    while True:
        if stride <= 1 or (orig % stride) == 0:
            mapping[orig] = proc
            proc += 1
            if max_frames and proc >= max_frames:
                break
        orig += 1
        if max_frames and orig > max_frames * max(1, stride) + 10:
            break
    return mapping


def load_sidecar_as_gt(
    roi_sidecar_path: str,
    framework_faster_path: str,
    H: int,
    W: int,
    max_frames: int,
    stride: int,
) -> Dict[int, List[RoiObj]]:
    ff = stable_import_module(framework_faster_path, "framework_privacy_sidecar")
    if not hasattr(ff, "unpack_mask"):
        raise RuntimeError("framework_faster.py must expose unpack_mask(...)")
    unpack_fn = ff.unpack_mask
    orig_to_proc = _build_orig_to_proc(max_frames=max_frames, stride=stride)
    out: Dict[int, List[RoiObj]] = {}
    with open(roi_sidecar_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            orig_idx = int(obj.get("frame_idx", -1))
            if orig_idx not in orig_to_proc:
                continue
            proc_idx = orig_to_proc[orig_idx]
            recs: List[RoiObj] = []
            for r in obj.get("rois", []) or []:
                bbox = r.get("bbox")
                mask_pack = r.get("mask_pack")
                if bbox is None or mask_pack is None:
                    continue
                x1, y1, x2, y2 = [int(v) for v in bbox]
                x1 = max(0, min(W, x1)); x2 = max(0, min(W, x2))
                y1 = max(0, min(H, y1)); y2 = max(0, min(H, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                mh, mw = y2 - y1, x2 - x1
                local = _call_unpack_mask(unpack_fn, mask_pack, mh, mw)
                full = np.zeros((H, W), dtype=bool)
                full[y1:y2, x1:x2] = local
                recs.append(RoiObj(
                    frame_idx=proc_idx,
                    bbox=(x1, y1, x2, y2),
                    mask=full,
                    cls_id=int(r.get("cls", -1)),
                    track_id=int(r.get("track_id", -1)),
                    conf=float(r.get("conf", 0.0)),
                ))
            out[proc_idx] = recs
    return out


def bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = float(iw * ih)
    area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
    area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def detect_frame_rois(ff, det_bundle, frame_bgr: np.ndarray, frame_idx: int, detect_width: int) -> List[RoiObj]:
    recs = ff.detect_rois_with_masks(
        det_bundle,
        frame_bgr,
        frame_idx=frame_idx,
        detect_width=detect_width,
        pack_for_sidecar=False,
    )
    H, W = frame_bgr.shape[:2]
    out: List[RoiObj] = []
    for r in recs:
        x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
        x1 = max(0, min(W, x1)); x2 = max(0, min(W, x2))
        y1 = max(0, min(H, y1)); y2 = max(0, min(H, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        mask_local = np.asarray(r.get("mask"), dtype=bool)
        if mask_local.shape != (y2 - y1, x2 - x1):
            mask_local = cv2.resize(mask_local.astype(np.uint8), (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST).astype(bool)
        full = np.zeros((H, W), dtype=bool)
        full[y1:y2, x1:x2] = mask_local
        out.append(RoiObj(
            frame_idx=frame_idx,
            bbox=(x1, y1, x2, y2),
            mask=full,
            cls_id=int(r.get("cls", -1)),
            track_id=int(r.get("track_id", -1)),
            conf=float(r.get("conf", 0.0)),
        ))
    return out


def greedy_match(gt: List[RoiObj], pred: List[RoiObj], iou_thr: float) -> List[Tuple[int, int, float]]:
    pairs: List[Tuple[float, int, int]] = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            if g.cls_id != p.cls_id:
                continue
            mi = mask_iou(g.mask, p.mask)
            bi = bbox_iou(g.bbox, p.bbox)
            score = max(mi, bi)
            if score >= iou_thr:
                pairs.append((score, gi, pi))
    pairs.sort(reverse=True, key=lambda x: x[0])
    used_g = set()
    used_p = set()
    out: List[Tuple[int, int, float]] = []
    for score, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        out.append((gi, pi, float(score)))
    return out


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def summarize_rows(rows: List[Dict[str, float]], keys: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r and np.isfinite(float(r[k]))]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def evaluate_video(
    label: str,
    video_path: str,
    gt_map: Dict[int, List[RoiObj]],
    framework_faster_path: str,
    max_frames: int,
    stride: int,
    detect_width: int,
    iou_thr: float,
    min_track_len: int,
    verbose: bool,
) -> Dict[str, object]:
    ff = stable_import_module(framework_faster_path, f"framework_privacy_eval_{label}")
    if not hasattr(ff, "load_detectors") or not hasattr(ff, "detect_rois_with_masks"):
        raise RuntimeError("framework_faster.py must expose load_detectors() and detect_rois_with_masks()")
    det_bundle = ff.load_detectors()

    reader = VideoReader(video_path, max_frames=max_frames, stride=stride)
    per_frame: List[Dict[str, float]] = []
    class_totals: Dict[int, Dict[str, float]] = {}
    gt_track_frames: Dict[int, int] = {}
    gt_track_matched: Dict[int, int] = {}
    total_gt = 0
    total_pred = 0
    total_matched = 0
    matched_ious: List[float] = []
    matched_confs: List[float] = []

    for proc_idx, orig_idx, frame in reader.iter_frames():
        gt = gt_map.get(proc_idx, [])
        pred = detect_frame_rois(ff, det_bundle, frame, frame_idx=orig_idx, detect_width=detect_width)
        matches = greedy_match(gt, pred, iou_thr=iou_thr)

        for g in gt:
            if g.track_id >= 0:
                gt_track_frames[g.track_id] = gt_track_frames.get(g.track_id, 0) + 1
        for gi, pi, score in matches:
            g = gt[gi]
            p = pred[pi]
            matched_ious.append(score)
            matched_confs.append(p.conf)
            if g.track_id >= 0:
                gt_track_matched[g.track_id] = gt_track_matched.get(g.track_id, 0) + 1

        total_gt += len(gt)
        total_pred += len(pred)
        total_matched += len(matches)

        classes_here = sorted(set([g.cls_id for g in gt] + [p.cls_id for p in pred]))
        for cls_id in classes_here:
            ctot = class_totals.setdefault(cls_id, {"gt": 0.0, "pred": 0.0, "matched": 0.0, "matched_iou_sum": 0.0, "matched_conf_sum": 0.0})
            gt_cls_idx = [i for i, g in enumerate(gt) if g.cls_id == cls_id]
            pred_cls_idx = [i for i, p in enumerate(pred) if p.cls_id == cls_id]
            matched_cls = [(gi, pi, sc) for gi, pi, sc in matches if gi in gt_cls_idx and pi in pred_cls_idx]
            ctot["gt"] += len(gt_cls_idx)
            ctot["pred"] += len(pred_cls_idx)
            ctot["matched"] += len(matched_cls)
            ctot["matched_iou_sum"] += sum(sc for _, _, sc in matched_cls)
            ctot["matched_conf_sum"] += sum(pred[pi].conf for _, pi, _ in matched_cls)

        row = {
            "frame_idx": int(proc_idx),
            "orig_frame_idx": int(orig_idx),
            "gt_count": float(len(gt)),
            "pred_count": float(len(pred)),
            "matched_count": float(len(matches)),
            "false_remaining_detections": float(max(0, len(pred) - len(matches))),
            "missed_gt_count": float(max(0, len(gt) - len(matches))),
            "frame_recall": safe_div(len(matches), len(gt)),
            "frame_precision": safe_div(len(matches), len(pred)),
            "mean_matched_iou": float(np.mean([m[2] for m in matches])) if matches else float("nan"),
            "mean_matched_conf": float(np.mean([pred[pi].conf for _, pi, _ in matches])) if matches else float("nan"),
        }
        per_frame.append(row)
        if verbose and (proc_idx % 10 == 0):
            print(f"[{label}] frame={proc_idx} gt={len(gt)} pred={len(pred)} matched={len(matches)}", flush=True)

    class_report: Dict[str, Dict[str, float]] = {}
    for cls_id, d in sorted(class_totals.items()):
        class_report[CLASS_NAMES.get(cls_id, str(cls_id))] = {
            "class_id": int(cls_id),
            "gt_count": float(d["gt"]),
            "pred_count": float(d["pred"]),
            "matched_count": float(d["matched"]),
            "false_remaining_detections": float(max(0.0, d["pred"] - d["matched"])),
            "missed_gt_count": float(max(0.0, d["gt"] - d["matched"])),
            "recall": safe_div(d["matched"], d["gt"]),
            "precision": safe_div(d["matched"], d["pred"]),
            "mean_matched_iou": safe_div(d["matched_iou_sum"], d["matched"]),
            "mean_matched_conf": safe_div(d["matched_conf_sum"], d["matched"]),
        }

    def _group_report(names: List[str]) -> Dict[str, float]:
        gt = pred = matched = conf_sum = iou_sum = 0.0
        for name in names:
            d = class_report.get(name)
            if not d:
                continue
            gt += float(d.get("gt_count", 0.0))
            pred += float(d.get("pred_count", 0.0))
            matched += float(d.get("matched_count", 0.0))
            mc = float(d.get("matched_count", 0.0))
            if mc > 0:
                ci = float(d.get("mean_matched_conf", float("nan")))
                ii = float(d.get("mean_matched_iou", float("nan")))
                if np.isfinite(ci):
                    conf_sum += ci * mc
                if np.isfinite(ii):
                    iou_sum += ii * mc
        return {
            "gt_count": gt,
            "pred_count": pred,
            "matched_count": matched,
            "false_remaining_detections": max(0.0, pred - matched),
            "missed_gt_count": max(0.0, gt - matched),
            "recall": safe_div(matched, gt),
            "precision": safe_div(matched, pred),
            "mean_matched_conf": safe_div(conf_sum, matched),
            "mean_matched_iou": safe_div(iou_sum, matched),
        }

    semantic_groups = {
        "person": _group_report(["person"]),
        "vehicle": _group_report(["car", "motorbike", "bus", "truck"]),
    }

    track_ratios = []
    eligible_tracks = 0
    survived_25 = 0
    survived_50 = 0
    survived_75 = 0
    for tid, n_frames in gt_track_frames.items():
        if n_frames < min_track_len:
            continue
        eligible_tracks += 1
        ratio = gt_track_matched.get(tid, 0) / float(n_frames)
        track_ratios.append(ratio)
        if ratio >= 0.25:
            survived_25 += 1
        if ratio >= 0.50:
            survived_50 += 1
        if ratio >= 0.75:
            survived_75 += 1

    frame_aggs = summarize_rows(per_frame, ["frame_recall", "frame_precision", "mean_matched_iou", "mean_matched_conf", "false_remaining_detections", "missed_gt_count"])
    return {
        "video_path": video_path,
        "totals": {
            "gt_count": int(total_gt),
            "pred_count": int(total_pred),
            "matched_count": int(total_matched),
            "false_remaining_detections": int(max(0, total_pred - total_matched)),
            "missed_gt_count": int(max(0, total_gt - total_matched)),
            "recall": safe_div(total_matched, total_gt),
            "precision": safe_div(total_matched, total_pred),
            "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else float("nan"),
            "mean_matched_conf": float(np.mean(matched_confs)) if matched_confs else float("nan"),
        },
        "frame_averages": frame_aggs,
        "per_class": class_report,
        "semantic_groups": semantic_groups,
        "track_survival": {
            "min_track_len": int(min_track_len),
            "eligible_tracks": int(eligible_tracks),
            "mean_track_frame_recall": float(np.mean(track_ratios)) if track_ratios else float("nan"),
            "survival_ratio_at_25pct": safe_div(survived_25, eligible_tracks),
            "survival_ratio_at_50pct": safe_div(survived_50, eligible_tracks),
            "survival_ratio_at_75pct": safe_div(survived_75, eligible_tracks),
        },
        "per_frame_rows": per_frame,
    }


def write_frame_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)



def main() -> None:
    ap = argparse.ArgumentParser(description="Detector/tracker privacy-leakage evaluation on encrypted video.")
    ap.add_argument("--plain", required=True, help="reference plain video used to generate the ROI sidecar")
    ap.add_argument("--cipher", required=True, help="encrypted/cipher video")
    ap.add_argument("--decrypted", default=None, help="optional decrypted/recovered video")
    ap.add_argument("--roi_sidecar", required=True, help="rois.jsonl produced on the plain video")
    ap.add_argument("--framework_faster_path", default="framework_faster.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_frames", type=int, default=100)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--detect_width", type=int, default=0)
    ap.add_argument("--iou_thr", type=float, default=0.30)
    ap.add_argument("--min_track_len", type=int, default=3)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.plain)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open plain video: {args.plain}")
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    cap.release()
    if H <= 0 or W <= 0:
        raise RuntimeError("Could not determine plain video dimensions.")

    gt_map = load_sidecar_as_gt(
        roi_sidecar_path=args.roi_sidecar,
        framework_faster_path=args.framework_faster_path,
        H=H,
        W=W,
        max_frames=args.max_frames,
        stride=args.stride,
    )

    report = {
        "definition": {
            "reference": "ROI sidecar detections from the plain/encryption pass are treated as pseudo-ground-truth.",
            "matching": "Greedy one-to-one matching, same class, using max(mask IoU, bbox IoU).",
            "privacy_goal": "Encrypted video should show strong drops in recall/precision/track survival relative to plain, while decrypted video should recover.",
        },
        "params": {
            "plain": args.plain,
            "cipher": args.cipher,
            "decrypted": args.decrypted,
            "roi_sidecar": args.roi_sidecar,
            "framework_faster_path": args.framework_faster_path,
            "max_frames": int(args.max_frames),
            "stride": int(args.stride),
            "detect_width": int(args.detect_width),
            "iou_thr": float(args.iou_thr),
            "min_track_len": int(args.min_track_len),
        },
        "reference_sidecar": {
            "frames_with_gt": int(len(gt_map)),
            "total_gt_objects": int(sum(len(v) for v in gt_map.values())),
        },
        "videos": {},
        "privacy_leakage_summary": {},
    }

    plain_eval = evaluate_video(
        label="plain",
        video_path=args.plain,
        gt_map=gt_map,
        framework_faster_path=args.framework_faster_path,
        max_frames=args.max_frames,
        stride=args.stride,
        detect_width=args.detect_width,
        iou_thr=args.iou_thr,
        min_track_len=args.min_track_len,
        verbose=args.verbose,
    )
    report["videos"]["plain"] = {k: v for k, v in plain_eval.items() if k != "per_frame_rows"}
    write_frame_csv(out_dir / "plain_per_frame.csv", plain_eval["per_frame_rows"])

    cipher_eval = evaluate_video(
        label="cipher",
        video_path=args.cipher,
        gt_map=gt_map,
        framework_faster_path=args.framework_faster_path,
        max_frames=args.max_frames,
        stride=args.stride,
        detect_width=args.detect_width,
        iou_thr=args.iou_thr,
        min_track_len=args.min_track_len,
        verbose=args.verbose,
    )
    report["videos"]["cipher"] = {k: v for k, v in cipher_eval.items() if k != "per_frame_rows"}
    write_frame_csv(out_dir / "cipher_per_frame.csv", cipher_eval["per_frame_rows"])

    dec_eval = None
    if args.decrypted:
        dec_eval = evaluate_video(
            label="decrypted",
            video_path=args.decrypted,
            gt_map=gt_map,
            framework_faster_path=args.framework_faster_path,
            max_frames=args.max_frames,
            stride=args.stride,
            detect_width=args.detect_width,
            iou_thr=args.iou_thr,
            min_track_len=args.min_track_len,
            verbose=args.verbose,
        )
        report["videos"]["decrypted"] = {k: v for k, v in dec_eval.items() if k != "per_frame_rows"}
        write_frame_csv(out_dir / "decrypted_per_frame.csv", dec_eval["per_frame_rows"])

    plain_recall = float(plain_eval["totals"]["recall"])
    cipher_recall = float(cipher_eval["totals"]["recall"])
    plain_track = float(plain_eval["track_survival"]["survival_ratio_at_50pct"])
    cipher_track = float(cipher_eval["track_survival"]["survival_ratio_at_50pct"])
    def _g(ev: dict, group: str, key: str) -> float:
        try:
            return float(ev.get("semantic_groups", {}).get(group, {}).get(key, float("nan")))
        except Exception:
            return float("nan")

    summary = {
        "cipher_recall_relative_to_plain": safe_div(cipher_recall, plain_recall) if np.isfinite(plain_recall) else float("nan"),
        "cipher_track_survival_relative_to_plain": safe_div(cipher_track, plain_track) if np.isfinite(plain_track) else float("nan"),
        "cipher_person_recall": _g(cipher_eval, "person", "recall"),
        "cipher_vehicle_recall": _g(cipher_eval, "vehicle", "recall"),
        "cipher_person_false_remaining_detections": _g(cipher_eval, "person", "false_remaining_detections"),
        "cipher_vehicle_false_remaining_detections": _g(cipher_eval, "vehicle", "false_remaining_detections"),
        "cipher_mean_matched_conf": float(cipher_eval["totals"].get("mean_matched_conf", float("nan"))),
    }
    if dec_eval is not None:
        dec_recall = float(dec_eval["totals"]["recall"])
        dec_track = float(dec_eval["track_survival"]["survival_ratio_at_50pct"])
        summary["decrypted_recall_relative_to_plain"] = safe_div(dec_recall, plain_recall) if np.isfinite(plain_recall) else float("nan")
        summary["decrypted_track_survival_relative_to_plain"] = safe_div(dec_track, plain_track) if np.isfinite(plain_track) else float("nan")
    report["privacy_leakage_summary"] = summary

    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "report": str(out_dir / 'report.json')}))


if __name__ == "__main__":
    main()
