#!/usr/bin/env python3
"""
13_detector_privacy_leakage_full_frame.py

Detector privacy leakage test for full-frame encryption.
No rois.jsonl argument is used.

Plain detections are treated as pseudo-ground-truth. Cipher detections are matched
against them by class and IoU. For a full-frame encrypted video, detector recall
on ciphertext should be near zero; false remaining detections should also be low.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from unified_utils import ensure_dir, iter_frames, video_info, write_csv, write_json

CLASS_NAMES = {0: "person", 2: "car", 3: "motorbike", 5: "bus", 7: "truck"}
VEHICLE_CLASSES = {2, 3, 5, 7}


@dataclass
class DetObj:
    frame_idx: int
    cls_id: int
    conf: float
    bbox: Tuple[float, float, float, float]


def load_yolo(model_name: str):
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("ultralytics is required for detector privacy leakage. Install with: pip install ultralytics") from e
    model = YOLO(model_name)
    try:
        model.fuse()
    except Exception:
        pass
    return model


def bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def detect_video(model, path: str, max_frames: int, stride: int, conf: float, classes: List[int]) -> Dict[int, List[DetObj]]:
    out: Dict[int, List[DetObj]] = {}
    for _, orig_i, fr in iter_frames(path, max_frames, stride):
        res = model.predict(fr, conf=conf, classes=list(classes), verbose=False)
        detections: List[DetObj] = []
        if res and getattr(res[0], "boxes", None) is not None:
            boxes = res[0].boxes
            for j in range(len(boxes)):
                try:
                    cls_id = int(boxes.cls[j].item())
                    cf = float(boxes.conf[j].item())
                    xyxy = tuple(float(x) for x in boxes.xyxy[j].tolist())
                except Exception:
                    continue
                detections.append(DetObj(frame_idx=int(orig_i), cls_id=cls_id, conf=cf, bbox=xyxy))
        out[int(orig_i)] = detections
    return out


def flatten(label: str, dets: Dict[int, List[DetObj]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fi in sorted(dets):
        if not dets[fi]:
            rows.append({"video": label, "frame_idx": fi, "cls_id": "", "class_name": "", "conf": "", "bbox": ""})
        for d in dets[fi]:
            rows.append({"video": label, "frame_idx": fi, "cls_id": d.cls_id, "class_name": CLASS_NAMES.get(d.cls_id, str(d.cls_id)), "conf": d.conf, "bbox": list(d.bbox)})
    return rows


def evaluate_detection_leakage(gt: Dict[int, List[DetObj]], pred: Dict[int, List[DetObj]], iou_thr: float) -> Dict[str, Any]:
    per_frame_rows: List[Dict[str, Any]] = []
    per_class: Dict[int, Dict[str, float]] = {}
    total_gt = total_pred = total_matched = total_false = total_missed = 0
    matched_ious: List[float] = []
    matched_confs: List[float] = []
    all_frames = sorted(set(gt) | set(pred))

    for fi in all_frames:
        gts = list(gt.get(fi, []))
        prs = list(pred.get(fi, []))
        used = set()
        matches: List[Tuple[int, int, float]] = []
        for gi, g in enumerate(gts):
            best_j, best_iou = None, 0.0
            for pj, p in enumerate(prs):
                if pj in used or p.cls_id != g.cls_id:
                    continue
                iou = bbox_iou(g.bbox, p.bbox)
                if iou > best_iou:
                    best_j, best_iou = pj, iou
            if best_j is not None and best_iou >= iou_thr:
                used.add(best_j)
                matches.append((gi, best_j, best_iou))

        frame_gt = len(gts)
        frame_pred = len(prs)
        frame_matched = len(matches)
        frame_false = frame_pred - frame_matched
        frame_missed = frame_gt - frame_matched
        total_gt += frame_gt
        total_pred += frame_pred
        total_matched += frame_matched
        total_false += frame_false
        total_missed += frame_missed
        matched_ious.extend([m[2] for m in matches])
        matched_confs.extend([prs[m[1]].conf for m in matches])

        for d in gts:
            pc = per_class.setdefault(d.cls_id, {"gt_count": 0.0, "pred_count": 0.0, "matched_count": 0.0, "false_remaining_detections": 0.0, "missed_gt_count": 0.0, "conf_sum": 0.0})
            pc["gt_count"] += 1
        for d in prs:
            pc = per_class.setdefault(d.cls_id, {"gt_count": 0.0, "pred_count": 0.0, "matched_count": 0.0, "false_remaining_detections": 0.0, "missed_gt_count": 0.0, "conf_sum": 0.0})
            pc["pred_count"] += 1
            pc["conf_sum"] += d.conf
        matched_gt_idxs = {m[0] for m in matches}
        matched_pred_idxs = {m[1] for m in matches}
        for gi, pj, _ in matches:
            per_class[gts[gi].cls_id]["matched_count"] += 1
        for gi, g in enumerate(gts):
            if gi not in matched_gt_idxs:
                per_class[g.cls_id]["missed_gt_count"] += 1
        for pj, p in enumerate(prs):
            if pj not in matched_pred_idxs:
                per_class[p.cls_id]["false_remaining_detections"] += 1

        per_frame_rows.append({
            "frame_idx": fi,
            "gt_count": frame_gt,
            "pred_count": frame_pred,
            "matched_count": frame_matched,
            "false_remaining_detections": frame_false,
            "missed_gt_count": frame_missed,
            "recall": frame_matched / frame_gt if frame_gt else None,
            "precision": frame_matched / frame_pred if frame_pred else None,
        })

    per_class_out: Dict[str, Dict[str, Any]] = {}
    for cls_id, d in sorted(per_class.items()):
        gt_c = d["gt_count"]
        pred_c = d["pred_count"]
        matched_c = d["matched_count"]
        per_class_out[CLASS_NAMES.get(cls_id, str(cls_id))] = {
            "class_id": int(cls_id),
            "gt_count": gt_c,
            "pred_count": pred_c,
            "matched_count": matched_c,
            "false_remaining_detections": d["false_remaining_detections"],
            "missed_gt_count": d["missed_gt_count"],
            "recall": matched_c / gt_c if gt_c else None,
            "precision": matched_c / pred_c if pred_c else None,
            "mean_conf": d["conf_sum"] / pred_c if pred_c else None,
        }

    vehicle_gt = sum(per_class.get(c, {}).get("gt_count", 0.0) for c in VEHICLE_CLASSES)
    vehicle_matched = sum(per_class.get(c, {}).get("matched_count", 0.0) for c in VEHICLE_CLASSES)
    person_gt = per_class.get(0, {}).get("gt_count", 0.0)
    person_matched = per_class.get(0, {}).get("matched_count", 0.0)

    return {
        "totals": {
            "gt_count": total_gt,
            "pred_count": total_pred,
            "matched_count": total_matched,
            "false_remaining_detections": total_false,
            "missed_gt_count": total_missed,
            "recall": total_matched / total_gt if total_gt else None,
            "precision": total_matched / total_pred if total_pred else None,
            "mean_matched_iou": float(mean(matched_ious)) if matched_ious else None,
            "mean_matched_conf": float(mean(matched_confs)) if matched_confs else None,
            "person_recall": person_matched / person_gt if person_gt else None,
            "vehicle_recall": vehicle_matched / vehicle_gt if vehicle_gt else None,
        },
        "per_class": per_class_out,
        "per_frame": per_frame_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Full-frame detector privacy leakage. No ROI sidecar is needed.")
    ap.add_argument("--plain", required=True)
    ap.add_argument("--cipher", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--decrypted", default=None)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou_thr", type=float, default=0.3)
    ap.add_argument("--classes", default="0,2,3,5,7", help="Comma-separated COCO class ids")
    ap.add_argument("--max_frames", type=int, default=100)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    out = ensure_dir(args.out)
    classes = [int(x) for x in args.classes.split(",") if x.strip()]
    model = load_yolo(args.model)

    plain_dets = detect_video(model, args.plain, args.max_frames, args.stride, args.conf, classes)
    cipher_dets = detect_video(model, args.cipher, args.max_frames, args.stride, args.conf, classes)
    cipher_eval = evaluate_detection_leakage(plain_dets, cipher_dets, args.iou_thr)

    write_csv(out / "detections_flat.csv", flatten("plain", plain_dets) + flatten("cipher", cipher_dets))
    write_csv(out / "cipher_leakage_per_frame.csv", cipher_eval["per_frame"])

    report: Dict[str, Any] = {
        "test_type": "full_frame_detector_privacy_leakage",
        "definition": {
            "reference": "Plain-video detections are treated as pseudo-ground-truth.",
            "matching": "Greedy one-to-one matching, same class, bbox IoU threshold.",
            "privacy_goal": "Encrypted full-frame video should have near-zero detector recall and few false remaining detections.",
        },
        "params": vars(args),
        "video_info": {"plain": video_info(args.plain), "cipher": video_info(args.cipher)},
        "plain_detection_count": sum(len(v) for v in plain_dets.values()),
        "cipher_detection_count": sum(len(v) for v in cipher_dets.values()),
        "cipher_leakage": cipher_eval,
    }

    if args.decrypted:
        dec_dets = detect_video(model, args.decrypted, args.max_frames, args.stride, args.conf, classes)
        dec_eval = evaluate_detection_leakage(plain_dets, dec_dets, args.iou_thr)
        report["video_info"]["decrypted"] = video_info(args.decrypted)
        report["decrypted_recovery"] = dec_eval
        write_csv(out / "decrypted_recovery_per_frame.csv", dec_eval["per_frame"])

    write_json(out / "report.json", report)
    print(json.dumps({"ok": True, "report": str(out / "report.json"), "cipher_recall": cipher_eval["totals"].get("recall")}))


if __name__ == "__main__":
    main()
