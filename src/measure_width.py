"""Fully automated stage 6 (SAM2 variant, fallback/comparison - SAM3 via
measure_width_sam3.py is the default): for every crossing that needs a
structure, fetch an imagery crop (Esri World Imagery - see imagery.py for
why not Google), segment the road surface with SAM2 prompted from the
known OSM road centreline, measure width on 5 perpendicular transects,
and write a width-survey CSV. Merge it into the register with:
    python src/measure_width_sam3.py --measured-csv output/gujarat_width_survey.csv

No human annotation, no per-image manual work - SAM2 is a zero-shot
promptable segmentation model, prompted automatically from geometry this
pipeline already knows (the crossing point + road tangent direction).
A JPG per crossing is saved to output/width_crops/ so the automated
measurement can be spot-checked visually.

Usage:
    python src/measure_width.py [--limit N] [--half-extent-m 100]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

from imagery import fetch_crop
from irc_classify import plausible_range
from width_core import measure_transects, road_tangent_bearing, save_qa_image, write_csv

CROSSINGS = "output/gujarat_crossings.geojson"
ROADS = "shared/roads_gujarat.geojson"
OUT_CSV = "output/gujarat_width_survey.csv"
CROP_DIR = Path("output/width_crops")
EPSG = 32643

MODEL_ID = "facebook/sam2.1-hiera-small"


def pick_mask(masks: np.ndarray, scores: np.ndarray, center_xy: tuple[int, int], crop_area: int) -> np.ndarray | None:
    cx, cy = center_xy
    best, best_score = None, -1.0
    for mask, score in zip(masks, scores):
        if not mask[cy, cx]:
            continue
        area_frac = mask.sum() / crop_area
        if area_frac > 0.6:  # almost certainly bled into surrounding fields
            continue
        if score > best_score:
            best, best_score = mask, score
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--half-extent-m", type=float, default=100.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {MODEL_ID} on {device} ...")
    predictor = SAM2ImagePredictor.from_pretrained(MODEL_ID, device=device)

    cross = gpd.read_file(CROSSINGS)
    cross = cross[cross["decision"] != "no_crossing"].reset_index(drop=True)
    if args.limit:
        cross = cross.head(args.limit)
    roads = gpd.read_file(ROADS).to_crs(EPSG)
    roads_sindex = roads.sindex
    cross_utm = cross.to_crs(EPSG)

    print(f"measuring {len(cross)} crossing(s) ...")
    rows = []
    for i, row in cross.iterrows():
        cid = row["crossing_id"]
        lon, lat = row.geometry.x, row.geometry.y
        pt_utm = cross_utm.geometry.iloc[i]

        bearing = road_tangent_bearing(roads_sindex, roads, pt_utm)
        crop = fetch_crop(lon, lat, half_extent_m=args.half_extent_m)
        if bearing is None or crop is None:
            print(f"  {cid}: skip (no road geometry or no imagery available)")
            rows.append({"crossing_id": cid, "measured_width_m": None, "measured_by": None,
                          "notes": "no imagery or road geometry available"})
            continue

        rgb, mpp, zoom = crop
        h, w = rgb.shape[:2]
        center_px = (w / 2.0, h / 2.0)

        predictor.set_image(rgb)
        brg = math.radians(bearing)
        along = np.array([math.sin(brg), -math.cos(brg)])
        perp = np.array([math.cos(brg), math.sin(brg)])
        fg_pts = [center_px + k * 6 / mpp * along for k in (-1, 0, 1)]
        bg_pts = [center_px + k * 15 / mpp * perp for k in (-1, 1)]
        point_coords = np.array(fg_pts + bg_pts)
        point_labels = np.array([1, 1, 1, 0, 0])

        with torch.inference_mode(), torch.autocast(device_type=device if device == "cuda" else "cpu", dtype=torch.bfloat16, enabled=device == "cuda"):
            masks, scores, _ = predictor.predict(
                point_coords=point_coords, point_labels=point_labels, multimask_output=True
            )
        mask = pick_mask(masks.astype(bool), scores, (int(center_px[0]), int(center_px[1])), h * w)

        widths = measure_transects(mask, mpp, bearing, center_px) if mask is not None else []
        save_qa_image(CROP_DIR / f"{cid}.jpg", rgb, mask, bearing, mpp, center_px, widths)

        if widths:
            med = float(np.median(widths))
            iqr = float(np.percentile(widths, 75) - np.percentile(widths, 25)) if len(widths) > 1 else 0.0
            conf = "high" if len(widths) >= 4 and iqr < 1.0 else "medium" if len(widths) >= 3 else "low"
            lo, hi = plausible_range(row["irc_class"])
            suspect = not (lo <= med <= hi) or iqr > 3.0
            if suspect:
                conf = "low"
            flag = " SUSPECT" if suspect else ""
            print(f"  {cid}: {med:.1f} m (n={len(widths)}, iqr={iqr:.2f}, conf={conf}, zoom={zoom}){flag}")
            rows.append({"crossing_id": cid, "measured_width_m": round(med, 2),
                         "measured_by": f"SAM2+EsriWorldImagery-auto(z{zoom})",
                         "notes": f"n={len(widths)} transects, IQR={iqr:.2f} m, confidence={conf}"
                                  + (" - SUSPECT: mask likely over-extended (tree cover/shadow), check QA crop" if suspect else "")})
        else:
            print(f"  {cid}: no usable mask/transects")
            rows.append({"crossing_id": cid, "measured_width_m": None, "measured_by": None,
                         "notes": "SAM2 mask/transects failed - check QA crop"})

    out = pd.DataFrame(rows)
    Path(OUT_CSV).parent.mkdir(exist_ok=True)
    write_csv(out, Path(OUT_CSV))
    n_ok = out["measured_width_m"].notna().sum()
    print(f"\nwrote {OUT_CSV} ({n_ok}/{len(out)} measured)")
    print(f"QA crops in {CROP_DIR}/")
    print(f"Next: python src/measure_width_sam3.py --measured-csv {OUT_CSV}")


if __name__ == "__main__":
    main()
