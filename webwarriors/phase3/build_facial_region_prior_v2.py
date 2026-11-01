#!/usr/bin/env python3
"""Build v2 soft facial-region prior maps from DFFD metadata and DTE-FDM outputs.

Prototype for Phase 3. This creates spatial prior maps only; it does not modify MFLM/SAM.
Expected metadata schema matches extract_faceapp_train_metadata.py output.
"""
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

REGION_PATTERNS = {
    "eyes": [r"\beye\b", r"\beyes\b", r"eyelid", r"eyebrow", r"eyebrows", r"under[- ]eye"],
    "nose": [r"\bnose\b", r"nasal"],
    "mouth": [r"\bmouth\b", r"\blips?\b", r"\bsmile\b", r"\bteeth\b", r"\blipstick\b"],
    "cheeks": [r"\bcheek(s)?\b", r"\bcheekbone(s)?\b"],
    "jaw": [r"\b(jaw|jawline)\b", r"\bchin\b", r"\blower face\b"],
    "forehead": [r"\bforehead\b", r"\bhairline\b", r"\btemple(s)?\b"],
    "whole_face": [r"\bwhole face\b", r"\bfull face\b", r"\bentire face\b", r"\bface[- ]wide\b", r"\baging\b", r"\bage(d|ing)?\b", r"\bskin texture\b", r"\bskin\b"],
}


def load_records(path):
    """Load JSONL, one JSON object, or a directory of per-image JSON files."""
    path = Path(path)
    if path.is_dir():
        rows = []
        for file_path in sorted(path.glob("*.json")):
            try:
                obj = json.loads(file_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {file_path}: {exc}") from exc
            if isinstance(obj, dict):
                rows.append(obj)
            elif isinstance(obj, list):
                rows.extend(x for x in obj if isinstance(x, dict))
        return rows
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
    except json.JSONDecodeError:
        pass
    rows = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on {path}:{line_no}: {exc}") from exc
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def normalize_text(row):
    # DTE-FDM output schema used by FakeShield stores generated explanation in `outputs`.
    fields = [row.get("outputs"), row.get("explanation"), row.get("text"), row.get("response")]
    return "\n".join(str(v) for v in fields if v is not None)


def extract_regions(text):
    lower = text.lower()
    found = []
    for region, patterns in REGION_PATTERNS.items():
        if any(re.search(p, lower) for p in patterns):
            found.append(region)
    # Whole-face/skin mentions should remain a broad prior, not a hard crop.
    if "whole_face" in found:
        found = [r for r in found if r != "whole_face"] + ["whole_face"]
    return list(dict.fromkeys(found))


def gaussian_map(height, width, cx, cy, sx, sy):
    yy, xx = np.mgrid[0:height, 0:width]
    sx = max(float(sx), 1.0)
    sy = max(float(sy), 1.0)
    return np.exp(-0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2)).astype(np.float32)


def make_prior(row, regions, base_weight=0.08, region_weight=0.92):
    w, h = int(row["width"]), int(row["height"])
    box = row.get("selected_box_xyxy_score")
    points = row.get("selected_landmarks_5x2")
    if not box or not points or len(points) != 5:
        return np.zeros((h, w), dtype=np.float32), "missing_selected_face_geometry"

    x1, y1, x2, y2 = map(float, box[:4])
    fw, fh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    # RetinaFace 5-point order: left eye, right eye, nose, left mouth corner, right mouth corner.
    le, reye, nose, lm, rm = [np.asarray(p, dtype=np.float32) for p in points]
    face = gaussian_map(h, w, (x1+x2)/2, (y1+y2)/2, fw*0.38, fh*0.42)
    prior = base_weight * face

    # Region priors are deliberately soft Gaussians. They guide, rather than constrain, prediction.
    region_specs = {
        "eyes": [((le + reye)/2, fw*0.18, fh*0.075), (le, fw*0.075, fh*0.050), (reye, fw*0.075, fh*0.050)],
        "nose": [(nose, fw*0.095, fh*0.095)],
        "mouth": [((lm + rm)/2, fw*0.13, fh*0.070), (lm, fw*0.060, fh*0.050), (rm, fw*0.060, fh*0.050)],
        "cheeks": [((le + nose + lm)/3, fw*0.12, fh*0.105), ((reye + nose + rm)/3, fw*0.12, fh*0.105)],
        "jaw": [((lm + rm)/2 + np.array([0, fh*0.13], dtype=np.float32), fw*0.21, fh*0.095)],
        "forehead": [((le + reye)/2 + np.array([0, -fh*0.18], dtype=np.float32), fw*0.20, fh*0.095)],
        "whole_face": [(((x1+x2)/2, (y1+y2)/2), fw*0.38, fh*0.42)],
    }
    heat = np.zeros((h, w), dtype=np.float32)
    used = []
    for region in regions:
        for center, sx, sy in region_specs.get(region, []):
            heat = np.maximum(heat, gaussian_map(h, w, center[0], center[1], sx, sy))
            used.append(region)
    if used:
        prior = np.maximum(prior, region_weight * heat)
        status = "ok"
    else:
        # Explicit fallback: a moderate broad face prior when explanation keywords
        # fail. This is guidance, not a hard crop; the full image remains available.
        fallback = 0.55 * face
        prior = np.maximum(prior, fallback)
        status = "fallback_face_prior_no_region_keyword_match"
    # Clip the prior map to the image. Source landmarks/boxes remain unchanged.
    return np.clip(prior, 0.0, 1.0), status


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata", required=True, help="Face metadata JSONL from RetinaFace extraction")
    ap.add_argument("--dte-output", required=True, help="DTE-FDM JSONL file, JSON file, or directory containing per-image JSON outputs")
    ap.add_argument("--output-dir", required=True, help="Directory for prior PNGs and index files")
    ap.add_argument("--base-weight", type=float, default=0.08)
    ap.add_argument("--region-weight", type=float, default=0.92)
    args = ap.parse_args()

    metadata_rows = load_records(args.metadata)
    dte_rows = load_records(args.dte_output)
    metadata_by_name = {Path(str(r.get("image", ""))).name: r for r in metadata_rows}
    out_dir = Path(args.output_dir)
    map_dir = out_dir / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = out_dir / "facial_prior_metadata.jsonl"
    counts = Counter()
    unmatched_dte = []
    seen = set()
    with out_jsonl.open("w", encoding="utf-8") as out:
        for dte in dte_rows:
            image_ref = dte.get("image", dte.get("image_path", ""))
            image_name = Path(str(image_ref)).name
            if not image_name or image_name not in metadata_by_name:
                unmatched_dte.append({"image": image_ref, "reason": "no_metadata_basename_match"})
                counts["unmatched_metadata"] += 1
                continue
            meta = metadata_by_name[image_name]
            text = normalize_text(dte)
            regions = extract_regions(text)
            prior, status = make_prior(meta, regions, args.base_weight, args.region_weight)
            map_name = Path(image_name).stem + "_prior.png"
            Image.fromarray(np.round(prior * 255).astype(np.uint8), mode="L").save(map_dir / map_name)
            rec = {
                "image": image_name,
                "source_image_reference": image_ref,
                "prior_map": str(Path("maps") / map_name),
                "width": int(meta["width"]),
                "height": int(meta["height"]),
                "regions_detected": regions,
                "dte_text_used": text,
                "status": status,
                "prior_type": "soft_gaussian_landmark_prior_v2",
                "base_weight": args.base_weight,
                "region_weight": args.region_weight,
                "selected_face_index": meta.get("selected_face_index"),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts[status] += 1
            seen.add(image_name)

    summary = {
        "metadata_records": len(metadata_rows),
        "dte_records": len(dte_rows),
        "matched_unique_images": len(seen),
        "unmatched_dte_records": len(unmatched_dte),
        "status_counts": dict(counts),
        "output_index": str(out_jsonl),
        "prior_map_directory": str(map_dir),
        "note": "V2 prototype only; narrower local Gaussians and explicit face-prior fallback. Not integrated into MFLM/SAM or trained.",
    }
    (out_dir / "facial_prior_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "unmatched_dte_records.json").write_text(json.dumps(unmatched_dte, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
