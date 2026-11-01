import csv
import os
from pathlib import Path
from PIL import Image

ROOT = Path(".")
GT_DIR = ROOT / "dataset/deepfake/FaceAPP_Train/mask"
BASELINE_DIR = ROOT / "phase3/results/baseline_masks"
REFINED_DIR = ROOT / "phase3/results/refined_masks"
OUT_DIR = ROOT / "phase3/results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def evaluate(gt_path, pred_path):
    gt = Image.open(gt_path).convert("L")
    pred = Image.open(pred_path).convert("L")

    if gt.size != pred.size:
        raise ValueError(f"Size mismatch: {gt_path.name}: GT={gt.size}, pred={pred.size}")

    gt_values = list(gt.getdata())
    pred_values = list(pred.getdata())

    tp = fp = fn = 0
    gt_foreground = pred_foreground = 0

    for g, p in zip(gt_values, pred_values):
        g = g > 25.5
        p = p > 127

        gt_foreground += int(g)
        pred_foreground += int(p)

        if g and p:
            tp += 1
        elif p and not g:
            fp += 1
        elif g and not p:
            fn += 1

    union = tp + fp + fn
    denom = 2 * tp + fp + fn
    iou = tp / union if union else 1.0
    f1 = 2 * tp / denom if denom else 1.0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "gt_foreground": gt_foreground,
        "pred_foreground": pred_foreground,
        "iou": iou, "pixel_f1": f1,
    }

def main():
    dirs = {
        "baseline": BASELINE_DIR,
        "refined": REFINED_DIR,
    }

    print("Phase 3 paired localization evaluation")
    print(f"Ground truth: {GT_DIR}")
    print(f"Baseline:     {BASELINE_DIR}")
    print(f"Refined:      {REFINED_DIR}")

    rows = []
    missing = []
    failed = []

    # Evaluate only stems that have both baseline and refined predictions.
    common = sorted(
        p.stem for p in BASELINE_DIR.glob("*.png")
        if (REFINED_DIR / p.name).is_file()
    )
    print(f"\nPaired prediction masks: {len(common)}")

    for stem in common:
        gt_path = GT_DIR / f"{stem}.png"
        if not gt_path.is_file():
            # Allow matching GT images with common image extensions.
            matches = [
                GT_DIR / f"{stem}{ext}"
                for ext in (".jpg", ".jpeg", ".bmp")
                if (GT_DIR / f"{stem}{ext}").is_file()
            ]
            if matches:
                gt_path = matches[0]
            else:
                missing.append(stem)
                continue

        try:
            b = evaluate(gt_path, BASELINE_DIR / f"{stem}.png")
            r = evaluate(gt_path, REFINED_DIR / f"{stem}.png")
            rows.append({
                "image": stem,
                **{f"baseline_{k}": v for k, v in b.items()},
                **{f"refined_{k}": v for k, v in r.items()},
                "iou_delta": r["iou"] - b["iou"],
                "pixel_f1_delta": r["pixel_f1"] - b["pixel_f1"],
            })
        except Exception as e:
            failed.append((stem, str(e)))

    if not rows:
        raise SystemExit(
            "No images evaluated. Check the GT mask path and filename matching."
        )

    csv_path = OUT_DIR / "phase3_pilot_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    baseline_iou = sum(x["baseline_iou"] for x in rows) / n
    refined_iou = sum(x["refined_iou"] for x in rows) / n
    baseline_f1 = sum(x["baseline_pixel_f1"] for x in rows) / n
    refined_f1 = sum(x["refined_pixel_f1"] for x in rows) / n

    print(f"\nSuccessfully evaluated: {n}")
    print(f"Missing GT masks:       {len(missing)}")
    print(f"Failed images:          {len(failed)}")
    print("\nMacro-average results")
    print(f"{'Metric':<16}{'Baseline':>12}{'Refined':>12}{'Delta':>12}")
    print("-" * 52)
    print(f"{'IoU':<16}{baseline_iou:>12.4f}{refined_iou:>12.4f}{refined_iou-baseline_iou:>+12.4f}")
    print(f"{'Pixel F1':<16}{baseline_f1:>12.4f}{refined_f1:>12.4f}{refined_f1-baseline_f1:>+12.4f}")
    print(f"\nPer-image CSV: {csv_path}")

    if missing:
        print("\nMissing GT:", ", ".join(missing))
    if failed:
        print("\nFailed images:")
        for stem, error in failed:
            print(f"  {stem}: {error}")

if __name__ == "__main__":
    main()
