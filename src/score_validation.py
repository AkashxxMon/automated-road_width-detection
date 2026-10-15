"""Score the crossings against the designers' bridge lines: TP, FP, FN,
precision, recall and F1 per state.

The reference is the designers' CAD "bridge leader/mark" lines: each starts on
the alignment at the bridge and runs ~98 m out to its label, so only the end
on the alignment is the bridge. That end is snapped onto the alignment and
matched one-to-one (closest pairs first, within --match-m) to our crossing
points (where the road-data line meets the alignment - closer to the
designers' bridges than where the measured road meets it: median 2.7 m vs
15 m in Jharkhand). A bridge whose end is more than --max-ref-offset-m off our
alignment is on another line (43 in Maharashtra, around Dahanu, are on a
corridor that isn't in the DFCC master file) - counted as off_corridor, not
as missed.

The prediction is what the structure layer draws (width_core.in_layer): every
crossing with a SAM3 width and a structure decision of its own - not MERGED,
not NONE. A predicted crossing matched to a designers' bridge is a TP, an
unmatched one an FP, an unmatched bridge an FN.

Within the predictions, the keep/re-route decision is broken down too:
  KEPT_BUILT        we give a structure, so did the designers
  KEPT_NOT_BUILT    we give a structure, they don't
  REROUTE_BUILT     we re-route, they built one
  REROUTE_NOT_BUILT we re-route, they don't

Usage:
    python src/score_validation.py                                  # every region with a reference file
    python src/score_validation.py --region maharashtra jharkhand --match-m 50
    python src/score_validation.py --region jharkhand --structures some/other_structures.geojson
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from geo_utils import latest, region_paths, region_slug, write_csv
from width_core import in_layer

REFERENCE = "validation/Bridges_road_crossings_lines_{region}.geojson"
EPSG = 7755  # India LCC, metres - one CRS for every state
KEPT = {"BOX", "BRIDGE", "DUAL_DESIGNER_DECISION", "PENDING_SURVEY"}


def reference_points(path: str, align: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """The bridge end of each leader line (the end nearer the alignment), snapped onto it,
    with how far off the alignment that end was."""
    lines = gpd.read_file(path).to_crs(EPSG)
    track = align.geometry.union_all()
    pts, off = [], []
    for g in lines.geometry:
        a, b = Point(g.coords[0]), Point(g.coords[-1])
        end = a if track.distance(a) <= track.distance(b) else b
        pts.append(track.interpolate(track.project(end)))
        off.append(round(track.distance(end), 1))
    return gpd.GeoDataFrame({"line_fid": range(len(lines)), "offset_m": off}, geometry=pts, crs=EPSG)


def match(ours: gpd.GeoSeries, ref: gpd.GeoDataFrame, max_m: float) -> dict:
    """{our index: (reference line_fid, distance)}, one-to-one, closest pairs first."""
    if ours.empty or ref.empty:
        return {}
    a = np.c_[ours.x, ours.y]
    b = np.c_[ref.geometry.x, ref.geometry.y]
    d = np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1))
    pairs = sorted(zip(*np.nonzero(d <= max_m)), key=lambda ij: d[ij])
    used_a, used_b, out = set(), set(), {}
    for i, j in pairs:
        if i not in used_a and j not in used_b:
            used_a.add(i), used_b.add(j)
            out[ours.index[i]] = (int(ref["line_fid"].iloc[j]), round(float(d[i, j]), 1))
    return out


def score_region(region: str, structures: str | None, match_m: float, max_ref_offset_m: float):
    paths = region_paths(region)
    cross = gpd.read_file(structures or latest(paths["structures"])).to_crs(EPSG)
    align = gpd.read_file(latest(paths["alignment"])).to_crs(EPSG)
    ref = reference_points(REFERENCE.format(region=region), align)
    n_off = int((ref["offset_m"] > max_ref_offset_m).sum())
    ref = ref[ref["offset_m"] <= max_ref_offset_m]

    pred = cross[in_layer(cross)].copy()
    m = match(pred.geometry, ref, match_m)
    pred["designers_line_fid"] = [m.get(i, (None, None))[0] for i in pred.index]
    pred["designers_dist_m"] = [m.get(i, (None, None))[1] for i in pred.index]
    built = pred["designers_line_fid"].notna()
    kept = pred["structure_type"].isin(KEPT)
    pred["outcome"] = np.select([kept & built, kept & ~built, ~kept & built],
                                ["KEPT_BUILT", "KEPT_NOT_BUILT", "REROUTE_BUILT"], "REROUTE_NOT_BUILT")
    pred["state"] = region
    found = set(pred["designers_line_fid"].dropna().astype(int))
    missing = ref[~ref["line_fid"].isin(found)].to_crs(4326)
    missing = pd.DataFrame({"state": region, "line_fid": missing["line_fid"], "outcome": "NOT_FOUND",
                            "lon": missing.geometry.x.round(6), "lat": missing.geometry.y.round(6)})
    return pred, missing, len(ref), n_off


def summary(pred: pd.DataFrame, n_ref: int, n_off: int) -> dict:
    o = pred["outcome"].value_counts()
    kb, knb, rb, rnb = (int(o.get(k, 0)) for k in ("KEPT_BUILT", "KEPT_NOT_BUILT", "REROUTE_BUILT", "REROUTE_NOT_BUILT"))
    tp, fp = kb + rb, knb + rnb
    fn = n_ref - tp

    def ratio(a, b):
        return round(a / b, 3) if b else None

    p, r = ratio(tp, tp + fp), ratio(tp, n_ref)
    return {"designers_bridges": n_ref, "off_corridor": n_off, "predicted": tp + fp, "TP": tp, "FP": fp, "FN": fn,
            "precision": p, "recall": r, "F1": round(2 * p * r / (p + r), 3) if p and r else None,
            "kept_built": kb, "kept_not_built": knb, "reroute_built": rb, "reroute_not_built": rnb,
            "structure_precision": ratio(kb, kb + knb), "decision_accuracy": ratio(kb + rnb, tp + fp)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", nargs="+", default=None,
                     help="Regions to score. Default: every region with a " + REFERENCE.format(region="<region>"))
    ap.add_argument("--structures", default=None,
                     help="Structure register to score (one region only). Default: output/<region>_structures.geojson")
    ap.add_argument("--match-m", type=float, default=30.0, help="A designers' bridge within this of a crossing matches it")
    ap.add_argument("--max-ref-offset-m", type=float, default=200.0,
                     help="A designers' bridge further than this off our alignment is on another line - not scored")
    ap.add_argument("-o", "--output-dir", default="output", help="Where to write <region>_validation_*.csv")
    args = ap.parse_args()
    regions = [region_slug(r) for r in args.region] if args.region else sorted(
        p.stem.replace("Bridges_road_crossings_lines_", "") for p in Path("validation").glob("Bridges_road_crossings_lines_*.geojson")
        if Path(region_paths(p.stem.replace("Bridges_road_crossings_lines_", ""))["structures"]).exists())
    if args.structures and len(regions) != 1:
        raise SystemExit("--structures needs exactly one --region")

    table, all_rows, n_all, off_all = {}, [], 0, 0
    out = Path(args.output_dir)
    for region in regions:
        pred, missing, n_ref, n_off = score_region(region, args.structures, args.match_m, args.max_ref_offset_m)
        cols = ["state", "crossing_id", "road_source", "irc_class", "structure_type", "road_category", "measured_width_m",
                "nearest_major_crossing_m", "road_surface", "decision", "outcome", "designers_line_fid", "designers_dist_m"]
        write_csv(pred[[c for c in cols if c in pred.columns]], out / f"{region}_validation_crossings.csv")
        write_csv(missing, out / f"{region}_validation_not_found.csv")
        table[region] = summary(pred, n_ref, n_off)
        all_rows.append(pred)
        n_all += n_ref
        off_all += n_off
    if len(regions) > 1:
        table["all"] = summary(pd.concat(all_rows), n_all, off_all)
    result = pd.DataFrame(table).T
    write_csv(result.reset_index(names="state"), out / "validation_summary.csv")
    print(f"matched within {args.match_m:g} m of the bridge end of each designers' leader line; "
          f"bridges more than {args.max_ref_offset_m:g} m off our alignment not scored (off_corridor)\n")
    print(result.to_string())
    print(f"\nwrote {out}/<region>_validation_crossings.csv, _not_found.csv, validation_summary.csv")


if __name__ == "__main__":
    main()
