"""One command: measure road width at every crossing using SAM3 segmentation
on Esri World Imagery, and merge the results straight into the crossing
register - populating design_width_m/span_m/width_source, saving a QA
image per crossing, and writing the raw measurement data to a CSV.

SAM3 (facebook/sam3, "Promptable Concept Segmentation") is prompted with
a text concept (default "road") and returns every matching instance in
the crop; pick_instance() picks whichever one sits at the known crossing
point. The OSM road tangent (from the roads file crossing_identify.py
already wrote) is only used to orient the 5 perpendicular transects, not
for prompting. No per-image human interaction.

A measurement flagged SUSPECT (implausible for its road class, or the
transects disagree too much - usually tree canopy or a divided
carriageway) is never allowed to override the IRC design-width floor -
see resolve_width() - so one bad mask can't silently inflate a span.

Needs local weights in ./model/sam3 - see src/download_sam3.py.
facebook/sam3 is gated with manual approval from Meta; weights are kept
in this project folder only, not the shared Hugging Face cache.

Usage:
    python src/measure_width_sam3.py
    python src/measure_width_sam3.py --crossings my_crossings.geojson --roads my_roads.geojson -o my_crossings_final.geojson
    python src/measure_width_sam3.py --measured-csv output/gujarat_width_recheck.csv   # re-merge a manual recheck, skip SAM3
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from PIL import Image

from geo_utils import get_utm_epsg, write_csv, write_geojson
from imagery import fetch_crop
from irc_classify import plausible_range
from width_core import measure_transects, road_tangent_bearing, save_qa_image

DEFAULT_CROSSINGS = "output/gujarat_crossings.geojson"
DEFAULT_ROADS = "shared/roads_gujarat.geojson"
DEFAULT_OUTPUT = "output/gujarat_crossings_final.geojson"
DEFAULT_CSV_OUTPUT = "output/gujarat_crossings_final.csv"
DEFAULT_SURVEY_CSV = "output/gujarat_width_survey.csv"
DEFAULT_IMAGES_DIR = "output/width_crops_sam3"
DEFAULT_MODEL_DIR = "model/sam3"

DETECT_THRESHOLD = 0.4  # SAM3's own example uses 0.5; slightly looser since a small crop has few distractors


def load_model(model_dir: Path, device: str):
    if not model_dir.exists() or not any(model_dir.glob("*.safetensors")):
        raise SystemExit(
            f"{model_dir} has no weights yet. Run:\n"
            f"  python src/download_sam3.py --token hf_xxxxxxxxxxxx\n"
            f"(needs Meta's manual approval for facebook/sam3 first - see that script's docstring)"
        )
    from transformers import Sam3Model, Sam3Processor
    model = Sam3Model.from_pretrained(str(model_dir)).to(device).eval()
    processor = Sam3Processor.from_pretrained(str(model_dir))
    return model, processor


def pick_instance(results: dict, center_xy: tuple[float, float], crop_area: int):
    """Among SAM3's detected instances, pick the one at the known crossing
    point - by mask containment first, else by nearest box centre."""
    masks = results.get("masks")
    boxes = results.get("boxes")
    scores = results.get("scores")
    if masks is None or len(masks) == 0:
        return None
    cx, cy = center_xy

    def as_bool(m):
        m = m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)
        return m.astype(bool)

    containing = []
    for i in range(len(masks)):
        m = as_bool(masks[i])
        if m.shape[0] > cy >= 0 and m.shape[1] > cx >= 0 and m[int(cy), int(cx)]:
            if m.sum() / crop_area <= 0.6:
                containing.append((i, float(scores[i]) if scores is not None else 0.0))
    if containing:
        best_i = max(containing, key=lambda t: t[1])[0]
        return as_bool(masks[best_i])

    if boxes is None:
        return None
    best_i, best_d = None, float("inf")
    for i in range(len(boxes)):
        x0, y0, x1, y1 = [float(v) for v in boxes[i]]
        bcx, bcy = (x0 + x1) / 2, (y0 + y1) / 2
        d = (bcx - cx) ** 2 + (bcy - cy) ** 2
        if d < best_d:
            best_d, best_i = d, i
    return as_bool(masks[best_i]) if best_i is not None else None


def measure_all(cross: gpd.GeoDataFrame, roads: gpd.GeoDataFrame, epsg: int, args) -> pd.DataFrame:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading SAM3 from {args.model_dir} on {device} ...")
    model, processor = load_model(Path(args.model_dir), device)

    roads = roads.to_crs(epsg)
    roads_sindex = roads.sindex
    to_measure = cross[cross["decision"] != "no_crossing"].reset_index(drop=True)
    if args.limit:
        to_measure = to_measure.head(args.limit)
    cross_utm = to_measure.to_crs(epsg)

    print(f"measuring {len(to_measure)} crossing(s) with text prompt {args.text!r} ...")
    rows = []
    for i, row in to_measure.iterrows():
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

        inputs = processor(images=Image.fromarray(rgb), text=args.text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_instance_segmentation(
            outputs, threshold=args.detect_threshold, mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        mask = pick_instance(results, center_px, h * w)
        widths = measure_transects(mask, mpp, bearing, center_px) if mask is not None else []
        save_qa_image(Path(args.images_dir) / f"{cid}.jpg", rgb, mask, bearing, mpp, center_px, widths)

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
                         "measured_by": f"SAM3({args.text})+EsriWorldImagery-auto(z{zoom})",
                         "notes": f"n={len(widths)} transects, IQR={iqr:.2f} m, confidence={conf}"
                                  + (" - SUSPECT: mask likely over-extended (tree cover/shadow), check QA crop" if suspect else "")})
        elif mask is None:
            print(f"  {cid}: no '{args.text}' instance detected at the crossing point")
            rows.append({"crossing_id": cid, "measured_width_m": None, "measured_by": None,
                         "notes": f"SAM3 found no '{args.text}' instance at the crossing point - check QA crop"})
        else:
            print(f"  {cid}: instance found but all transects failed (likely a divided/complex carriageway shape)")
            rows.append({"crossing_id": cid, "measured_width_m": None, "measured_by": None,
                         "notes": "SAM3 detected a road instance but the transect walk found no clean crossing - "
                                  "check QA crop (often a divided highway, median gap in the mask)"})
    return pd.DataFrame(rows)


def resolve_width(row):
    """design_width_m = max(measured, IRC minimum), unless the measurement is
    SUSPECT, in which case it falls back to the IRC floor (or stays
    'needs survey') rather than being trusted."""
    if pd.isna(row["measured_width_m"]):
        return row["design_width_m"], row["width_source"], row["span_m"]
    irc_width = row["design_width_m"] if str(row["width_source"]).startswith("IRC") else 0.0
    if "SUSPECT" in str(row["notes"]):
        if irc_width > 0:
            return irc_width, "IRC (measurement flagged SUSPECT, verify manually)", row["span_m"]
        return row["design_width_m"], "needs field/imagery survey (auto measurement flagged SUSPECT)", row["span_m"]
    w = max(row["measured_width_m"], irc_width)
    source = "measured" if row["measured_width_m"] >= irc_width else "IRC (exceeds measured)"
    span = round(w / math.sin(math.radians(max(row["angle_deg"], 1e-3))), 2)
    return w, source, span


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crossings", default=DEFAULT_CROSSINGS, help="Input crossings GeoJSON (from crossing_identify.py)")
    ap.add_argument("--roads", default=DEFAULT_ROADS, help="Roads GeoJSON (from crossing_identify.py) - used for the transect bearing")
    ap.add_argument("--epsg", type=int, default=None, help="Metric CRS for measurement. Default: auto-detect UTM zone")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Local SAM3 weights directory")
    ap.add_argument("--text", default="road", help="Concept prompt passed to SAM3")
    ap.add_argument("--half-extent-m", type=float, default=100.0, help="Half-width of the imagery crop fetched per crossing")
    ap.add_argument("--detect-threshold", type=float, default=DETECT_THRESHOLD, help="SAM3 instance-detection confidence threshold")
    ap.add_argument("--limit", type=int, default=None, help="Only measure the first N crossings (for testing)")
    ap.add_argument("--measured-csv", default=None,
                     help="Skip SAM3 entirely and merge this CSV instead (crossing_id, measured_width_m[, measured_by, notes]) "
                          "- e.g. a manual recheck from make_survey_worksheet.py")
    ap.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="Output final crossings GeoJSON")
    ap.add_argument("--csv-output", default=DEFAULT_CSV_OUTPUT, help="Output final crossings CSV")
    ap.add_argument("--survey-csv", default=DEFAULT_SURVEY_CSV, help="Where to save the raw SAM3 measurement data")
    ap.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR, help="Directory for the per-crossing QA images")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if not Path(args.crossings).exists():
        raise SystemExit(f"{args.crossings} not found - run src/crossing_identify.py first")
    cross = gpd.read_file(args.crossings)
    centroid = cross.geometry.union_all().centroid
    epsg = args.epsg or get_utm_epsg(centroid.x, centroid.y)

    if args.measured_csv:
        print(f"using {args.measured_csv} directly - skipping SAM3")
        survey = pd.read_csv(args.measured_csv)
    else:
        if not Path(args.roads).exists():
            raise SystemExit(f"{args.roads} not found - run src/crossing_identify.py first")
        roads = gpd.read_file(args.roads)
        survey = measure_all(cross, roads, epsg, args)
        write_csv(survey, Path(args.survey_csv))
        n_ok = survey["measured_width_m"].notna().sum()
        print(f"\nwrote {args.survey_csv} ({n_ok}/{len(survey)} measured)")
        print(f"QA crops in {args.images_dir}/")

    survey = survey[survey["measured_width_m"].notna()]
    for col in ("measured_by", "notes"):
        if col not in survey.columns:
            survey[col] = None
    survey = survey[["crossing_id", "measured_width_m", "measured_by", "notes"]]
    print(f"\n{len(survey)} measured row(s) to merge")

    cross = cross.drop(columns=["measured_width_m", "measured_by", "notes"], errors="ignore")
    cross = cross.merge(survey, on="crossing_id", how="left")
    cross[["design_width_m", "width_source", "span_m"]] = cross.apply(resolve_width, axis=1, result_type="expand")

    write_geojson(cross, Path(args.output))
    write_csv(cross.drop(columns="geometry"), Path(args.csv_output))
    print(f"\nwrote {args.output}")
    print(f"wrote {args.csv_output}")
    print(cross["width_source"].value_counts().to_string())
    still_missing = cross[cross["width_source"].str.startswith("needs field/imagery survey")]
    print(f"\n{len(still_missing)} crossing(s) still have no usable width at all - keep surveying.")


if __name__ == "__main__":
    main()
