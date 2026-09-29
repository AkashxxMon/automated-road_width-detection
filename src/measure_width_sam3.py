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

Design width and span come entirely from this measurement - there is no
IRC minimum floor. The IRC class (from crossing_identify.py) is used only
to flag a measurement as SUSPECT when it's implausible for that class of
road (or the 5 transects disagree too much) - see resolve_width() - so a
reviewer knows to check it, without the pipeline silently substituting or
capping the number.

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
from PIL import Image, ImageFilter
from shapely.geometry import LineString, Point
from shapely.ops import substring

from geo_utils import get_utm_epsg, write_csv, write_geojson
from imagery import fetch_crop
from irc_classify import plausible_range
from width_core import measure_transects_detailed, road_tangent_bearing, save_qa_image

DEFAULT_CROSSINGS = "output/gujarat_crossings.geojson"
DEFAULT_ROADS = "shared/roads_gujarat.geojson"
DEFAULT_ALIGNMENT = "shared/alignment_gujarat.geojson"
DEFAULT_OUTPUT = "output/gujarat_crossings_final_gl.geojson"
DEFAULT_CSV_OUTPUT = "output/gujarat_crossings_final_gl.csv"
DEFAULT_SURVEY_CSV = "output/gujarat_width_survey_gl.csv"
DEFAULT_IMAGES_DIR = "output/width_crops_sam3_gl_1"
DEFAULT_WIDTH_GEOMETRY_OUTPUT = "output/gujarat_width_geometry_gl.geojson"
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


MERGE_NEARBY_M = 18.0  # bridge a gap up to this wide between separate SAM3 instances - covers a
                       # typical divided-highway median so both carriageways measure as one road


def _dilate(mask: np.ndarray, kernel_px: int) -> np.ndarray:
    """Morphological dilation via PIL's MaxFilter (no scipy dependency)."""
    if kernel_px < 3:
        return mask
    if kernel_px % 2 == 0:
        kernel_px += 1
    img = Image.fromarray((mask.astype(np.uint8) * 255))
    dilated = img.filter(ImageFilter.MaxFilter(kernel_px))
    return np.asarray(dilated) > 0


def pick_instance(results: dict, center_xy: tuple[float, float], crop_area: int, mpp: float):
    """Among SAM3's detected instances, pick the one at the known crossing
    point - by mask containment first, else by nearest box centre. Then
    union in any OTHER instance whose mask is within MERGE_NEARBY_M of the
    picked one - a divided highway is often detected as 2+ separate
    instances (one per carriageway, split by the median), and using only
    one badly under-measures the true design width."""
    masks = results.get("masks")
    boxes = results.get("boxes")
    scores = results.get("scores")
    if masks is None or len(masks) == 0:
        return None
    cx, cy = center_xy

    def as_bool(m):
        m = m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)
        return m.astype(bool)

    all_masks = [as_bool(masks[i]) for i in range(len(masks))]

    containing = []
    for i, m in enumerate(all_masks):
        if m.shape[0] > cy >= 0 and m.shape[1] > cx >= 0 and m[int(cy), int(cx)]:
            if m.sum() / crop_area <= 0.6:
                containing.append((i, float(scores[i]) if scores is not None else 0.0))

    if containing:
        best_i = max(containing, key=lambda t: t[1])[0]
    elif boxes is not None:
        best_i, best_d = None, float("inf")
        for i in range(len(boxes)):
            x0, y0, x1, y1 = [float(v) for v in boxes[i]]
            bcx, bcy = (x0 + x1) / 2, (y0 + y1) / 2
            d = (bcx - cx) ** 2 + (bcy - cy) ** 2
            if d < best_d:
                best_d, best_i = d, i
    else:
        return None

    combined = all_masks[best_i].copy()
    kernel_px = int(round(MERGE_NEARBY_M / mpp))
    dilated = _dilate(combined, kernel_px)
    for i, m in enumerate(all_masks):
        if i != best_i and (m & dilated).any() and m.sum() / crop_area <= 0.6:
            combined |= m
    return combined


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

        mask = pick_instance(results, center_px, h * w, mpp)
        detailed = measure_transects_detailed(mask, mpp, bearing, center_px) if mask is not None else []
        widths = [d["width_m"] for d in detailed]
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

            # the transect whose width is closest to the reported median is the one whose
            # edge pixels become the width-line geometry - not necessarily the geographically
            # centre (offset=0) transect, whose own width can disagree with the reported median
            center_transect = min(detailed, key=lambda d: abs(d["width_m"] - med))
            p1x, p1y = center_transect["p1_px"]
            p2x, p2y = center_transect["p2_px"]
            width_p1_x = pt_utm.x + (p1x - center_px[0]) * mpp
            width_p1_y = pt_utm.y - (p1y - center_px[1]) * mpp
            width_p2_x = pt_utm.x + (p2x - center_px[0]) * mpp
            width_p2_y = pt_utm.y - (p2y - center_px[1]) * mpp

            rows.append({"crossing_id": cid, "measured_width_m": round(med, 2),
                         "measured_by": f"SAM3({args.text})+EsriWorldImagery-auto(z{zoom})",
                         "notes": f"n={len(widths)} transects, IQR={iqr:.2f} m, confidence={conf}"
                                  + (" - SUSPECT: mask likely over-extended (tree cover/shadow), check QA crop" if suspect else ""),
                         "width_p1_x": width_p1_x, "width_p1_y": width_p1_y,
                         "width_p2_x": width_p2_x, "width_p2_y": width_p2_y})
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
    """design_width_m/span_m come from the SAM3 measurement alone - no IRC
    floor, no substitution. A SUSPECT measurement is still used, just
    labelled for review rather than trusted silently; there's no other
    number to fall back to."""
    if pd.isna(row["measured_width_m"]):
        return None, "needs field/imagery survey", None
    span = round(row["measured_width_m"] / math.sin(math.radians(max(row["angle_deg"], 1e-3))), 2)
    source = "measured (SUSPECT - verify manually)" if "SUSPECT" in str(row["notes"]) else "measured"
    return row["measured_width_m"], source, span


def write_width_geometry(cross: gpd.GeoDataFrame, epsg: int, args) -> None:
    """One combined layer, geometry drawn to scale, two kinds of line per
    crossing:
      - "width": the actual SAM3 measurement, the two road-edge pixels the
        transect walk found - only present where a live SAM3 measurement ran
        (not for a --measured-csv merge, which has no pixel geometry).
      - "span": the length of structure needed *along the railway track*
        (design_width_m / sin(angle) - see resolve_width()), drawn as the
        real alignment geometry between chainage-span/2 and chainage+span/2,
        not just a straight tangent segment, so it stays accurate on curves.
    """
    props = ["crossing_id", "align_name", "chainage_m", "irc_class", "angle_deg",
              "measured_width_m", "design_width_m", "span_m", "width_source", "decision"]
    props = [c for c in props if c in cross.columns]

    width_rows = []
    has_width_geom = cross[["width_p1_x", "width_p1_y", "width_p2_x", "width_p2_y"]].notna().all(axis=1) \
        if "width_p1_x" in cross.columns else pd.Series(False, index=cross.index)
    for _, r in cross[has_width_geom].iterrows():
        geom = LineString([(r["width_p1_x"], r["width_p1_y"]), (r["width_p2_x"], r["width_p2_y"])])
        width_rows.append({**{c: r[c] for c in props}, "kind": "width", "geometry": geom})

    span_rows = []
    if Path(args.alignment).exists() and "align_id" in cross.columns:
        align = gpd.read_file(args.alignment).to_crs(epsg)
        align_by_id = {row["align_id"]: row.geometry for _, row in align.iterrows()}
        has_span = cross["span_m"].notna() & cross["chainage_m"].notna() & cross["align_id"].notna()
        for _, r in cross[has_span].iterrows():
            line = align_by_id.get(r["align_id"])
            if line is None:
                continue
            half = r["span_m"] / 2.0
            start = max(r["chainage_m"] - half, 0.0)
            end = min(r["chainage_m"] + half, line.length)
            if end <= start:
                continue
            geom = substring(line, start, end)
            span_rows.append({**{c: r[c] for c in props}, "kind": "span", "geometry": geom})
    else:
        print(f"\n  [warn] {args.alignment} not found - skipping span-line geometry (width lines still written if available)")

    if not width_rows and not span_rows:
        print("\nno width/span geometry available (likely a --measured-csv run with no matching alignment) - "
              "skipping width-geometry output")
        return

    combined = gpd.GeoDataFrame(width_rows + span_rows, geometry="geometry", crs=epsg).to_crs(4326)
    write_geojson(combined, Path(args.width_geometry_output))
    print(f"wrote {args.width_geometry_output} ({len(width_rows)} width line(s), {len(span_rows)} span line(s))")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crossings", default=DEFAULT_CROSSINGS, help="Input crossings GeoJSON (from crossing_identify.py)")
    ap.add_argument("--roads", default=DEFAULT_ROADS, help="Roads GeoJSON (from crossing_identify.py) - used for the transect bearing")
    ap.add_argument("--alignment", default=DEFAULT_ALIGNMENT,
                     help="Clipped alignment GeoJSON (from crossing_identify.py) - used to draw the span-line geometry")
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
    ap.add_argument("--width-geometry-output", default=DEFAULT_WIDTH_GEOMETRY_OUTPUT,
                     help="Output GeoJSON: one layer with both the measured width lines and the along-track span lines, drawn to scale")
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
    for col in ("measured_by", "notes", "width_p1_x", "width_p1_y", "width_p2_x", "width_p2_y"):
        if col not in survey.columns:
            survey[col] = None
    edge_cols = ["width_p1_x", "width_p1_y", "width_p2_x", "width_p2_y"]
    survey = survey[["crossing_id", "measured_width_m", "measured_by", "notes"] + edge_cols]
    print(f"\n{len(survey)} measured row(s) to merge")

    cross = cross.drop(columns=["measured_width_m", "measured_by", "notes"] + edge_cols, errors="ignore")
    cross = cross.merge(survey, on="crossing_id", how="left")
    cross[["design_width_m", "width_source", "span_m"]] = cross.apply(resolve_width, axis=1, result_type="expand")

    write_geojson(cross.drop(columns=edge_cols), Path(args.output))
    write_csv(cross.drop(columns="geometry"), Path(args.csv_output))
    print(f"\nwrote {args.output}")
    print(f"wrote {args.csv_output}")
    print(cross["width_source"].value_counts().to_string())
    still_missing = cross[cross["width_source"].str.startswith("needs field/imagery survey")]
    print(f"\n{len(still_missing)} crossing(s) still have no usable width at all - keep surveying.")

    write_width_geometry(cross, epsg, args)


if __name__ == "__main__":
    main()
