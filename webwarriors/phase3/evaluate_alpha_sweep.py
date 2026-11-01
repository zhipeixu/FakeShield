import csv
from pathlib import Path
from PIL import Image

GT_DIR = Path("dataset/deepfake/FaceAPP_Train/mask")
BASELINE_DIR = Path("phase3/results/baseline_masks")
RESULTS_DIR = Path("phase3/results")
ALPHAS = ["0.00", "0.25", "0.50", "0.75"]

def metrics(gt_path, pred_path):
    gt = Image.open(gt_path).convert("L")
    pred = Image.open(pred_path).convert("L")

    if gt.size != pred.size:
        raise ValueError(f"Image size mismatch: {pred_path}")

    g = [v > 25.5 for v in gt.getdata()]
    p = [v > 127 for v in pred.getdata()]

    tp = sum(a and b for a, b in zip(g, p))
    fp = sum((not a) and b for a, b in zip(g, p))
    fn = sum(a and (not b) for a, b in zip(g, p))

    union = tp + fp + fn
    denom = 2 * tp + fp + fn

    return (
        tp / union if union else 1.0,
        2 * tp / denom if denom else 1.0
    )

def find_gt(stem):
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        p = GT_DIR / f"{stem}{ext}"
        if p.is_file():
            return p
    return None

def main():
    baseline_files = sorted(BASELINE_DIR.glob("*.png"))
    rows = []
    summaries = []

    for alpha in ALPHAS:
        pred_dir = (
            BASELINE_DIR if alpha == "0.00"
            else RESULTS_DIR / f"alpha_{alpha}" / "masks"
        )

        if not pred_dir.is_dir():
            print(f"WARNING: Missing directory for alpha={alpha}: {pred_dir}")
            continue

        ious, f1s = [], []
        for base in baseline_files:
            stem = base.stem
            gt_path = find_gt(stem)
            pred_path = pred_dir / base.name

            if gt_path is None or not pred_path.is_file():
                continue

            iou, f1 = metrics(gt_path, pred_path)
            ious.append(iou)
            f1s.append(f1)
            rows.append({
                "alpha": alpha,
                "image": stem,
                "iou": iou,
                "pixel_f1": f1
            })

        if not ious:
            print(f"WARNING: No paired masks for alpha={alpha}")
            continue

        summaries.append({
            "alpha": alpha,
            "n": len(ious),
            "mean_iou": sum(ious) / len(ious),
            "mean_pixel_f1": sum(f1s) / len(f1s)
        })

    out_rows = RESULTS_DIR / "phase3_alpha_sweep_per_image.csv"
    with out_rows.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["alpha", "image", "iou", "pixel_f1"])
        w.writeheader()
        w.writerows(rows)

    out_summary = RESULTS_DIR / "phase3_alpha_sweep_summary.csv"
    with out_summary.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["alpha", "n", "mean_iou", "mean_pixel_f1"]
        )
        w.writeheader()
        w.writerows(summaries)

    print("\nPhase 3 alpha sensitivity (macro averages)")
    print(f"{'Alpha':<10}{'N':>5}{'Mean IoU':>14}{'Mean pixel F1':>17}")
    print("-" * 46)
    for s in summaries:
        print(
            f"{s['alpha']:<10}{s['n']:>5}"
            f"{s['mean_iou']:>14.4f}{s['mean_pixel_f1']:>17.4f}"
        )

    print(f"\nPer-image CSV: {out_rows}")
    print(f"Summary CSV:   {out_summary}")

if __name__ == "__main__":
    main()
