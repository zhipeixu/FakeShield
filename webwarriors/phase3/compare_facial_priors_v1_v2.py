#!/usr/bin/env python3
"""Compare v1 and v2 facial prior maps against FaceAPP ground-truth masks."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def read_index(path):
    with Path(path).open(encoding="utf-8") as f:
        return {Path(r["image"]).name: r for line in f if line.strip()
                for r in [json.loads(line)]}


def load_gt(path):
    # Matches FakeShield calculate_metrics.py: grayscale GT > 25.5.
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 25.5


def load_prior(index_dir, record):
    path = Path(index_dir) / record["prior_map"]
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def scores(gt, prior, threshold=0.5):
    pred = prior > threshold
    tp = int(np.logical_and(gt, pred).sum())
    fp = int(np.logical_and(~gt, pred).sum())
    fn = int(np.logical_and(gt, ~pred).sum())
    union = tp + fp + fn
    denom = 2 * tp + fp + fn
    return {
        "gt_foreground_pct": float(gt.mean() * 100),
        "prior_foreground_pct": float(pred.mean() * 100),
        "iou": tp / union if union else float("nan"),
        "pixel_f1": (2 * tp) / denom if denom else float("nan"),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v1-index", required=True)
    p.add_argument("--v1-dir", required=True)
    p.add_argument("--v2-index", required=True)
    p.add_argument("--v2-dir", required=True)
    p.add_argument("--mask-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--prior-threshold", type=float, default=0.5)
    args = p.parse_args()

    v1 = read_index(args.v1_index)
    v2 = read_index(args.v2_index)
    mask_dir = Path(args.mask_dir)
    names = sorted(set(v1) & set(v2))
    rows = []

    for name in names:
        mask_candidates = list(mask_dir.glob(Path(name).stem + ".*"))
        if not mask_candidates:
            print(f"SKIP {name}: no ground-truth mask")
            continue
        gt = load_gt(mask_candidates[0])
        s1 = scores(gt, load_prior(args.v1_dir, v1[name]), args.prior_threshold)
        s2 = scores(gt, load_prior(args.v2_dir, v2[name]), args.prior_threshold)
        rows.append({
            "image": name,
            "gt_foreground_pct": round(s1["gt_foreground_pct"], 4),
            "v1_prior_foreground_pct": round(s1["prior_foreground_pct"], 4),
            "v1_iou": s1["iou"],
            "v1_pixel_f1": s1["pixel_f1"],
            "v2_prior_foreground_pct": round(s2["prior_foreground_pct"], 4),
            "v2_iou": s2["iou"],
            "v2_pixel_f1": s2["pixel_f1"],
            "delta_iou_v2_minus_v1": s2["iou"] - s1["iou"],
            "delta_f1_v2_minus_v1": s2["pixel_f1"] - s1["pixel_f1"],
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    finite = lambda key: [r[key] for r in rows if np.isfinite(r[key])]
    print(f"Images compared: {len(rows)}")
    print(f"V1 mean diagnostic IoU: {np.mean(finite('v1_iou')):.4f}")
    print(f"V2 mean diagnostic IoU: {np.mean(finite('v2_iou')):.4f}")
    print(f"V1 mean diagnostic pixel F1: {np.mean(finite('v1_pixel_f1')):.4f}")
    print(f"V2 mean diagnostic pixel F1: {np.mean(finite('v2_pixel_f1')):.4f}")
    print(f"Saved per-image comparison: {output}")
    print("These are prior-only thresholded diagnostics, not MFLM model metrics.")


if __name__ == "__main__":
    main()
