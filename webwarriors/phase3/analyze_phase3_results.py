#!/usr/bin/env python3
"""Analyze per-image Phase 3 baseline vs refined metrics.

Reads phase3/results/phase3_full_metrics.csv by default.
Writes a text summary and top-improvement/top-decline CSVs beside the input.
This script does not modify masks, ground truth, or evaluation metrics.
"""
import argparse
import csv
import math
from pathlib import Path


def number(row, key):
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="phase3/results/phase3_full_metrics.csv")
    args = parser.parse_args()
    source = Path(args.csv)
    if not source.is_file():
        raise SystemExit(f"CSV not found: {source}")

    required = ["image", "baseline_iou", "refined_iou",
                "baseline_pixel_f1", "refined_pixel_f1"]
    rows = []
    with source.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit("Missing CSV columns: " + ", ".join(missing))
        for row in reader:
            vals = [number(row, c) for c in required[1:]]
            if any(v is None for v in vals):
                continue
            bi, ri, bf, rf = vals
            row["_di"] = ri - bi
            row["_df"] = rf - bf
            row["_bi"], row["_ri"] = bi, ri
            row["_bf"], row["_rf"] = bf, rf
            rows.append(row)

    if not rows:
        raise SystemExit("No valid paired metric rows found.")

    def avg(key):
        return sum(r[key] for r in rows) / len(rows)

    def counts(key):
        up = sum(r[key] > 1e-12 for r in rows)
        down = sum(r[key] < -1e-12 for r in rows)
        return up, down, len(rows) - up - down

    iu, idn, iz = counts("_di")
    fu, fdn, fz = counts("_df")
    bi, ri, bf, rf = avg("_bi"), avg("_ri"), avg("_bf"), avg("_rf")

    report = [
        "PHASE 3 PER-IMAGE ANALYSIS",
        "=" * 52,
        f"Input CSV: {source}",
        f"Valid paired images: {len(rows)}",
        "",
        "MACRO AVERAGES",
        f"Baseline IoU: {bi:.6f}",
        f"Refined IoU:  {ri:.6f}",
        f"IoU delta:    {ri-bi:+.6f}",
        f"Baseline F1:  {bf:.6f}",
        f"Refined F1:   {rf:.6f}",
        f"F1 delta:     {rf-bf:+.6f}",
        "",
        "PER-IMAGE CHANGES",
        f"IoU improved / worsened / unchanged: {iu} / {idn} / {iz}",
        f"F1 improved / worsened / unchanged:  {fu} / {fdn} / {fz}",
        "",
        "Note: Per-image improvement counts do not replace the aggregate",
        "macro-average comparison. Report all results transparently."
    ]

    outdir = source.parent
    summary_path = outdir / "phase3_full_analysis_summary.txt"
    better_path = outdir / "phase3_full_top_improvements.csv"
    worse_path = outdir / "phase3_full_top_declines.csv"
    summary_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    fields = ["image", "baseline_iou", "refined_iou", "iou_delta",
              "baseline_pixel_f1", "refined_pixel_f1", "pixel_f1_delta"]
    for destination, ranked in [
        (better_path, sorted(rows, key=lambda r: r["_di"], reverse=True)[:20]),
        (worse_path, sorted(rows, key=lambda r: r["_di"])[:20])
    ]:
        with destination.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in ranked:
                writer.writerow({
                    "image": r["image"],
                    "baseline_iou": r["_bi"],
                    "refined_iou": r["_ri"],
                    "iou_delta": r["_di"],
                    "baseline_pixel_f1": r["_bf"],
                    "refined_pixel_f1": r["_rf"],
                    "pixel_f1_delta": r["_df"]
                })

    print("\n".join(report))
    print("\nCreated:")
    print(summary_path)
    print(better_path)
    print(worse_path)


if __name__ == "__main__":
    main()
