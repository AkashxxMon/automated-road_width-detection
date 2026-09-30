"""One command: measure road width at every crossing using SAM3 segmentation
on Esri World Imagery, and merge the results straight into the crossing
register - populating design_width_m/width_source (plus the combined
layout of every road within 20 m of it - the other carriageway of a
divided road, a service road - where the imagery shows one), saving a QA
image per crossing, and writing the raw measurement data to a CSV.
Box/bridge call, box size and span along track are the next step,
structure_decide.py.

SAM3 (facebook/sam3, "Promptable Concept Segmentation") is prompted with
a text concept (default "road") and returns every matching instance in
the crop; pick_instance() picks whichever one sits at the known crossing
point. The OSM road tangent (from the roads file crossing_identify.py
already wrote) is only used to orient the 5 perpendicular transects, not
for prompting. No per-image human interaction.

Design width comes entirely from this measurement - there is no
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
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter

from geo_utils import fill_region_paths, get_utm_epsg, latest, write_csv, write_geojson
from imagery import fetch_crop
from irc_classify import plausible_range
from width_core import (MAX_ROAD_GAP_M, combine_roads, measure_transects_detailed, road_layout_text,
                        road_tangent_bearing, save_qa_image, write_width_geometry)

# every input/output path defaults from the region (geo_utils.region_paths): --region, else the one
# crossing_identify.py last ran for - any path passed explicitly wins
REGION_PATH_ARGS = {"crossings": "crossings", "roads": "roads", "alignment": "alignment", "output": "final",
                    "csv_output": "final_csv", "survey_csv": "survey_csv", "images_dir": "images_dir",
                    "width_geometry_output": "width_geometry"}
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


MERGE_NEARBY_M = MAX_ROAD_GAP_M  # union in any other SAM3 instance within this (20 m) - the other
                                 # carriageway across a median, a service road - so the transect walk
                                 # sees the whole road corridor (width_core.combine_roads)


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
        # other roads within MAX_ROAD_GAP_M (the other carriageway, a service road) - kept alongside
        # the single-road width, not folded into it: structure_decide.py spans the whole layout
        combo = combine_roads(detailed) if widths else None
        combo_label = (f"combined: {road_layout_text(combo['widths_m'], combo['gaps_m'])} = {combo['total_m']:.1f} m"
                       if combo else None)
        save_qa_image(Path(args.images_dir) / f"{cid}.jpg", rgb, mask, bearing, mpp, center_px, widths,
                      extra_label=combo_label)

        if widths:
            med = float(np.median(widths))
            iqr = float(np.percentile(widths, 75) - np.percentile(widths, 25)) if len(widths) > 1 else 0.0
            conf = "high" if len(widths) >= 4 and iqr < 1.0 else "medium" if len(widths) >= 3 else "low"
            lo, hi = plausible_range(row["irc_class"])
            suspect = not (lo <= med <= hi) or iqr > 3.0
            if suspect:
                conf = "low"
            flag = " SUSPECT" if suspect else ""
            flag += f" ({combo_label})" if combo else ""
            print(f"  {cid}: {med:.1f} m (n={len(widths)}, iqr={iqr:.2f}, conf={conf}, zoom={zoom}){flag}")

            def to_utm(px):
                return pt_utm.x + (px[0] - center_px[0]) * mpp, pt_utm.y - (px[1] - center_px[1]) * mpp

            combo_cols = {"n_roads": len(combo["widths_m"]) if combo else 1}
            if combo:
                (ext_p1_x, ext_p1_y), (ext_p2_x, ext_p2_y) = to_utm(combo["e1_px"]), to_utm(combo["e2_px"])
                combo_cols.update({
                    "road_layout_m": road_layout_text(combo["widths_m"], combo["gaps_m"]),
                    "total_width_m": round(combo["total_m"], 2),
                    "ext_p1_x": ext_p1_x, "ext_p1_y": ext_p1_y, "ext_p2_x": ext_p2_x, "ext_p2_y": ext_p2_y,
                })
                if len(combo["widths_m"]) == 2:
                    combo_cols.update({"cw1_width_m": round(combo["widths_m"][0], 2),
                                       "median_width_m": round(combo["gaps_m"][0], 2),
                                       "cw2_width_m": round(combo["widths_m"][1], 2)})

            # the transect whose width is closest to the reported median is the one whose
            # edge pixels become the width-line geometry - not necessarily the geographically
            # centre (offset=0) transect, whose own width can disagree with the reported median
            center_transect = min(detailed, key=lambda d: abs(d["width_m"] - med))
            p1x, p1y = center_transect["p1_px"]
            p2x, p2y = center_transect["p2_px"]
            width_p1_x, width_p1_y = to_utm((p1x, p1y))
            width_p2_x, width_p2_y = to_utm((p2x, p2y))

            rows.append({"crossing_id": cid, "measured_width_m": round(med, 2),
                         "measured_by": f"SAM3({args.text})+EsriWorldImagery-auto(z{zoom})",
                         "notes": f"n={len(widths)} transects, IQR={iqr:.2f} m, confidence={conf}"
                                  + (" - SUSPECT: mask likely over-extended (tree cover/shadow), check QA crop" if suspect else ""),
                         "width_iqr_m": round(iqr, 2), "n_transects": len(widths), **combo_cols,
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
    """design_width_m comes from the SAM3 measurement alone - no IRC floor,
    no substitution. A SUSPECT measurement is still used, just labelled for
    review rather than trusted silently; there's no other number to fall
    back to. span_m is left for structure_decide.py, since whether a span
    applies at all depends on the box/bridge call (a box is always square)
    and the skew gate."""
    if pd.isna(row["measured_width_m"]):
        return None, "needs field/imagery survey"
    source = "measured (SUSPECT - verify manually)" if "SUSPECT" in str(row["notes"]) else "measured"
    return row["measured_width_m"], source


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", default=None,
                     help="Region whose files to use (e.g. maharashtra, as crossing_identify.py printed). Default: the "
                          "region crossing_identify.py last ran for. Any path below passed explicitly overrides it")
    ap.add_argument("--crossings", default=None, help="Input crossings GeoJSON. Default: output/<region>_crossings.geojson")
    ap.add_argument("--roads", default=None,
                     help="Roads GeoJSON - used for the transect bearing. Default: shared/roads_<region>.geojson")
    ap.add_argument("--alignment", default=None,
                     help="Clipped alignment GeoJSON - used to draw span lines. Default: shared/alignment_<region>.geojson")
    ap.add_argument("--epsg", type=int, default=None, help="Metric CRS for measurement. Default: auto-detect UTM zone")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Local SAM3 weights directory")
    ap.add_argument("--text", default="road", help="Concept prompt passed to SAM3")
    ap.add_argument("--half-extent-m", type=float, default=100.0, help="Half-width of the imagery crop fetched per crossing")
    ap.add_argument("--detect-threshold", type=float, default=DETECT_THRESHOLD, help="SAM3 instance-detection confidence threshold")
    ap.add_argument("--limit", type=int, default=None, help="Only measure the first N crossings (for testing)")
    ap.add_argument("--measured-csv", default=None,
                     help="Skip SAM3 entirely and merge this CSV instead (crossing_id, measured_width_m[, measured_by, notes, "
                          "cw1_width_m, median_width_m, cw2_width_m, total_width_m]) "
                          "- e.g. a manual recheck from make_survey_worksheet.py")
    ap.add_argument("-o", "--output", default=None,
                     help="Output final crossings GeoJSON. Default: output/<region>_crossings_final_gl.geojson")
    ap.add_argument("--csv-output", default=None, help="Output final crossings CSV")
    ap.add_argument("--survey-csv", default=None, help="Where to save the raw SAM3 measurement data")
    ap.add_argument("--images-dir", default=None, help="Directory for the per-crossing QA images")
    ap.add_argument("--width-geometry-output", default=None,
                     help="Output GeoJSON: the measured width lines, drawn to scale (span lines are added by structure_decide.py)")
    args = ap.parse_args()
    fill_region_paths(args, REGION_PATH_ARGS)
    return args


def main() -> None:
    args = parse_args()

    crossings_path = latest(args.crossings)
    if not crossings_path.exists():
        raise SystemExit(f"{args.crossings} not found - run src/crossing_identify.py first "
                         f"(or pass --region / --crossings for the region you ran it for)")
    print(f"crossings: {crossings_path}")
    cross = gpd.read_file(crossings_path)
    centroid = cross.geometry.union_all().centroid
    epsg = args.epsg or get_utm_epsg(centroid.x, centroid.y)

    if args.measured_csv:
        print(f"using {args.measured_csv} directly - skipping SAM3")
        survey = pd.read_csv(args.measured_csv)
    else:
        roads_path = latest(args.roads)
        if not roads_path.exists():
            raise SystemExit(f"{args.roads} not found - run src/crossing_identify.py first")
        print(f"roads: {roads_path}")
        roads = gpd.read_file(roads_path)
        survey = measure_all(cross, roads, epsg, args)
        write_csv(survey, Path(args.survey_csv))
        n_ok = survey["measured_width_m"].notna().sum()
        print(f"\nwrote {args.survey_csv} ({n_ok}/{len(survey)} measured)")
        print(f"QA crops in {args.images_dir}/")

    survey = survey[survey["measured_width_m"].notna()].copy()
    edge_cols = ["width_p1_x", "width_p1_y", "width_p2_x", "width_p2_y",
                 "ext_p1_x", "ext_p1_y", "ext_p2_x", "ext_p2_y"]
    # n_roads/road_layout_m/total/cw*/median: the combined-road reading (width_core.combine_roads) -
    # cw1/median/cw2 are also fillable by hand in a --measured-csv recheck, for a divided road the
    # imagery pass couldn't split
    width_cols = ["measured_width_m", "measured_by", "notes", "width_iqr_m", "n_transects",
                  "n_roads", "road_layout_m", "total_width_m",
                  "cw1_width_m", "median_width_m", "cw2_width_m"]
    for col in width_cols + edge_cols:
        if col not in survey.columns:
            survey[col] = None
    survey = survey[["crossing_id"] + width_cols + edge_cols]
    print(f"\n{len(survey)} measured row(s) to merge")

    cross = cross.drop(columns=width_cols + edge_cols, errors="ignore")
    cross = cross.merge(survey, on="crossing_id", how="left")
    cross[["design_width_m", "width_source"]] = cross.apply(resolve_width, axis=1, result_type="expand")
    cross["span_m"] = None  # a span from an earlier run no longer matches this width - structure_decide.py sets it

    # edge coordinates stay in the GeoJSON too - structure_decide.py redraws the width lines from them
    write_geojson(cross, Path(args.output))
    write_csv(cross.drop(columns="geometry"), Path(args.csv_output))
    print(f"\nwrote {args.output}")
    print(f"wrote {args.csv_output}")
    print(cross["width_source"].value_counts().to_string())
    still_missing = cross[cross["width_source"].str.startswith("needs field/imagery survey")]
    print(f"\n{len(still_missing)} crossing(s) still have no usable width at all - keep surveying.")

    write_width_geometry(cross, epsg, str(latest(args.alignment)), args.width_geometry_output)
    print(f"\nBox/bridge call, box size and span along track - run next:\n"
          f"  python src/structure_decide.py --region {args.region}")


if __name__ == "__main__":
    main()
