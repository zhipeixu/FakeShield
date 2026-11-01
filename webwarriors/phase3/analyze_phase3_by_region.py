#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path
from collections import defaultdict

def load_metrics(path):
    out = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        needed = {"image","baseline_iou","refined_iou","baseline_pixel_f1","refined_pixel_f1"}
        missing = needed - set(reader.fieldnames or [])
        if missing:
            raise ValueError("Missing metric columns: " + ", ".join(sorted(missing)))
        for r in reader:
            try:
                bi, ri = float(r["baseline_iou"]), float(r["refined_iou"])
                bf, rf = float(r["baseline_pixel_f1"]), float(r["refined_pixel_f1"])
            except (TypeError, ValueError):
                continue
            stem = Path(r["image"]).stem
            out[stem] = {"baseline_iou":bi,"refined_iou":ri,"iou_delta":ri-bi,
                         "baseline_pixel_f1":bf,"refined_pixel_f1":rf,"pixel_f1_delta":rf-bf}
    return out

def load_priors(path):
    out = {}
    with path.open(encoding="utf-8") as f:
        for n,line in enumerate(f,1):
            if not line.strip(): continue
            try: r=json.loads(line)
            except json.JSONDecodeError as e: raise ValueError(f"Bad JSONL line {n}: {e}")
            stem=Path(str(r.get("image",""))).stem
            if stem: out[stem]=r
    return out

def summarize(group_type, name, rows):
    n=len(rows)
    result={"group_type":group_type,"group":name,"images":n}
    for k in ("baseline_iou","refined_iou","iou_delta","baseline_pixel_f1",
              "refined_pixel_f1","pixel_f1_delta"):
        result[k]=sum(x[k] for x in rows)/n if n else 0
    result["iou_improved"]=sum(x["iou_delta"]>1e-12 for x in rows)
    result["iou_worsened"]=sum(x["iou_delta"] < -1e-12 for x in rows)
    result["iou_unchanged"]=n-result["iou_improved"]-result["iou_worsened"]
    return result

def save_csv(path, rows):
    if rows:
        with path.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description="Compare baseline/refined metrics by prior status and region.")
    ap.add_argument("--metrics",default="phase3/results/phase3_full_metrics.csv")
    ap.add_argument("--priors",default="phase3/results/facial_priors_v2_full/facial_prior_metadata.jsonl")
    a=ap.parse_args()
    mp,pp=Path(a.metrics),Path(a.priors)
    if not mp.is_file(): raise SystemExit(f"Metrics CSV not found: {mp}")
    if not pp.is_file(): raise SystemExit(f"Prior metadata not found: {pp}")
    metrics,priors=load_metrics(mp),load_priors(pp)
    matched=sorted(set(metrics)&set(priors))
    print(f"Metric records: {len(metrics)} | Prior records: {len(priors)} | Matched: {len(matched)}")
    statuses=defaultdict(list); regions=defaultdict(list)
    for stem in matched:
        m=metrics[stem]; prior=priors[stem]
        statuses[prior.get("status","unknown")].append(m)
        for region in set(prior.get("regions_detected") or []):
            regions[region].append(m)
    status_rows=[summarize("status",k,v) for k,v in sorted(statuses.items())]
    region_rows=[summarize("region",k,v) for k,v in sorted(regions.items())]
    out=mp.parent
    save_csv(out/"phase3_status_analysis.csv",status_rows)
    save_csv(out/"phase3_region_analysis.csv",region_rows)
    lines=["PHASE 3 FACIAL-PRIOR SUBGROUP ANALYSIS","="*54,
           f"Evaluated records: {len(metrics)}",f"Matched with prior metadata: {len(matched)}",
           f"Missing prior metadata: {len(set(metrics)-set(priors))}","","STATUS GROUPS (exclusive)"]
    for r in status_rows:
        lines += [f"{r['group']} (n={r['images']}): IoU {r['baseline_iou']:.6f} -> {r['refined_iou']:.6f} (delta {r['iou_delta']:+.6f}); F1 {r['baseline_pixel_f1']:.6f} -> {r['refined_pixel_f1']:.6f} (delta {r['pixel_f1_delta']:+.6f}); IoU up/down/same {r['iou_improved']}/{r['iou_worsened']}/{r['iou_unchanged']}"]
    lines += ["","REGION GROUPS (overlapping; an image can occur in multiple groups)"]
    for r in region_rows:
        lines += [f"{r['group']} (n={r['images']}): IoU {r['baseline_iou']:.6f} -> {r['refined_iou']:.6f} (delta {r['iou_delta']:+.6f}); F1 {r['baseline_pixel_f1']:.6f} -> {r['refined_pixel_f1']:.6f} (delta {r['pixel_f1_delta']:+.6f}); IoU up/down/same {r['iou_improved']}/{r['iou_worsened']}/{r['iou_unchanged']}"]
    lines += ["","Note: subgroup results are descriptive. Region groups overlap; use validation data for method selection and held-out data for final claims."]
    report="\n".join(lines)+"\n"
    (out/"phase3_region_analysis_summary.txt").write_text(report,encoding="utf-8")
    print("\n"+report)
    print("Saved:",out/"phase3_region_analysis_summary.txt")
    print("Saved:",out/"phase3_region_analysis.csv")
    print("Saved:",out/"phase3_status_analysis.csv")

if __name__=="__main__": main()
