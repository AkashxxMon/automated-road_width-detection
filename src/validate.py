"""Validate the automated Gujarat crossing register against a manually
marked ground-truth set (bridge leader/mark lines digitised from the
design drawings, source_zone 2382_Z43 etc. - a CAD COGO-label export).

Each ground-truth feature is a short leader line pointing at a bridge
location, not the bridge geometry itself, so it's reduced to a single
point (its endpoint nearest the alignment) before matching.

Usage:
    python src/validate.py [--tolerance-m 60]
"""

from __future__ import annotations

import argparse

import geopandas as gpd
import networkx as nx
from shapely.geometry import Point
from shapely.strtree import STRtree

EPSG = 32643
VALIDATION = "validation/Bridges_corrected_lines.geojson"
ALIGNMENT = "shared/alignment_gujarat.geojson"
ROADS = "shared/roads_gujarat.geojson"
CROSSINGS = "output/gujarat_crossings.geojson"
CORRIDOR = "shared/corridor_2km.geojson"
OUT_PATH = "output/gujarat_validation.geojson"

DEDUP_TOL_M = 15.0  # this CAD export puts ~3 separate leader lines (label/elevation/coords) per bridge
NEAR_ROAD_M = 20.0  # a missed bridge with no OSM road within this distance is almost certainly a
                    # non-road structure (drainage/waterway/canal/rail), not an OSM coverage gap -
                    # this validation file covers every structure type, not just road crossings


def leader_anchor(line, align_barrier) -> Point:
    """The endpoint of a bridge leader/mark line that sits on the alignment."""
    p0, p1 = Point(line.coords[0]), Point(line.coords[-1])
    return p0 if align_barrier.distance(p0) <= align_barrier.distance(p1) else p1


def dedup_by_distance(gdf: gpd.GeoDataFrame, tol: float) -> gpd.GeoDataFrame:
    """Collapse points within tol of each other into one (their centroid) -
    the source file has multiple CAD annotation lines per physical bridge."""
    sindex = gdf.sindex
    G = nx.Graph()
    G.add_nodes_from(range(len(gdf)))
    for i, geom in enumerate(gdf.geometry):
        for j in sindex.query(geom.buffer(tol)):
            if j != i:
                G.add_edge(i, j)
    rows = []
    for cluster in nx.connected_components(G):
        idx = list(cluster)
        pts = gdf.geometry.iloc[idx]
        centroid = Point(pts.x.mean(), pts.y.mean())
        row = gdf.iloc[idx[0]].copy()
        row["geometry"] = centroid
        row["n_source_lines"] = len(idx)
        rows.append(row)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=gdf.crs).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tolerance-m", type=float, default=60.0)
    ap.add_argument("--belongs-to-alignment-m", type=float, default=200.0,
                     help="Drop validation features whose anchor is farther than this from the "
                          "Gujarat alignment - the raw file covers every DFC section nationwide, "
                          "and a 2 km corridor bbox can still pick up a neighbouring section's "
                          "bridges near the state border.")
    args = ap.parse_args()

    align = gpd.read_file(ALIGNMENT).to_crs(EPSG)
    corridor = gpd.read_file(CORRIDOR).to_crs(EPSG).geometry.iloc[0]
    align_barrier = align.geometry.union_all()

    val = gpd.read_file(VALIDATION).to_crs(EPSG)
    print(f"validation set: {len(val)} bridge leader/mark line(s), all states")

    val = val[val.geometry.intersects(corridor)].reset_index(drop=True)
    val["geometry"] = val.geometry.apply(lambda g: leader_anchor(g, align_barrier))
    val["dist_to_alignment_m"] = val.geometry.apply(align_barrier.distance)

    before = len(val)
    val = val[val["dist_to_alignment_m"] <= args.belongs_to_alignment_m].reset_index(drop=True)
    print(f"  {before} inside the Gujarat 2 km corridor bbox, {len(val)} of those within "
          f"{args.belongs_to_alignment_m:.0f} m of the Gujarat alignment itself (the rest belong "
          f"to a neighbouring section, e.g. PB-Sec-4/5 just across the Maharashtra border)")
    if val.empty:
        raise SystemExit("No validation features fall on the Gujarat alignment - nothing to check.")

    val = dedup_by_distance(val, DEDUP_TOL_M)
    print(f"  {len(val)} unique bridge(s) after collapsing CAD annotation lines within {DEDUP_TOL_M:.0f} m "
          f"of each other (this export puts ~3 lines - label, elevation, coordinates - per bridge)")

    cross = gpd.read_file(CROSSINGS).to_crs(EPSG)
    print(f"predicted crossings: {len(cross)}")
    roads = gpd.read_file(ROADS).to_crs(EPSG)
    roads_sindex = roads.sindex

    def nearest_road_dist(pt) -> float:
        near = list(roads_sindex.query(pt.buffer(200)))
        if not near:
            return float("inf")
        return min(roads.geometry.iloc[i].distance(pt) for i in near)

    pred_tree = STRtree(cross.geometry.values)
    val_tree = STRtree(val.geometry.values)

    # ground truth -> nearest prediction (recall)
    matches = []
    for i, row in val.iterrows():
        idx = pred_tree.nearest(row.geometry)
        pred = cross.iloc[idx]
        d = row.geometry.distance(pred.geometry)
        matched = d <= args.tolerance_m
        matches.append({
            "val_idx": i,
            "matched": matched,
            "dist_m": round(d, 1),
            "pred_crossing_id": pred["crossing_id"] if matched else None,
            "pred_decision": pred["decision"] if matched else None,
        })
    val_match = gpd.GeoDataFrame(matches).set_index("val_idx")
    val = val.join(val_match)

    found = val[val["matched"]]
    missed = val[~val["matched"]].copy()
    wrong_decision = found[found["pred_decision"] != "crossing"]

    if len(missed):
        missed["nearest_road_m"] = missed.geometry.apply(nearest_road_dist)
    likely_non_road = missed[missed.get("nearest_road_m", 0) > NEAR_ROAD_M] if len(missed) else missed
    likely_osm_gap = missed[missed.get("nearest_road_m", 0) <= NEAR_ROAD_M] if len(missed) else missed

    road_relevant = len(val) - len(likely_non_road)  # ground truth that's actually a road-crossing candidate

    print(f"\n--- Recall (does the pipeline find each known bridge?) tolerance={args.tolerance_m} m ---")
    print(f"  {len(val)} ground-truth features total. This validation file covers every structure type on "
          f"the drawings (road bridges, but also drainage/waterway/canal/rail crossings etc.), not just road "
          f"crossings, and has no field distinguishing them - {len(likely_non_road)} have no OSM road within "
          f"{NEAR_ROAD_M:.0f} m at all and are almost certainly one of those non-road structures, out of "
          f"scope for this pipeline. Treating those as N/A rather than misses:")
    print(f"  {len(found)}/{road_relevant} road-relevant ground-truth features matched to a predicted crossing "
          f"({100 * len(found) / road_relevant:.1f}%)" if road_relevant else "  no road-relevant features to check")
    print(f"  of those, {len(wrong_decision)} matched a candidate NOT decided 'crossing' "
          f"(dropped or sent to review by the R1-R4 rule - a false negative in the decision logic):")
    if len(wrong_decision):
        print(wrong_decision[["pred_crossing_id", "pred_decision", "dist_m"]].to_string())
    print(f"  {len(likely_osm_gap)} have a nearby OSM road (<= {NEAR_ROAD_M:.0f} m) but still no predicted "
          f"candidate within {args.tolerance_m} m - a genuine OSM/connectivity gap, worth checking:")
    if len(likely_osm_gap):
        print(likely_osm_gap[["dist_to_alignment_m", "nearest_road_m", "source_zone"]]
              .sort_values("dist_to_alignment_m").to_string())
    print(f"  {len(likely_non_road)} have no OSM road nearby at all - likely a non-road structure, not counted above")

    # prediction -> nearest ground truth (precision, crossing-decided only)
    crossing_only = cross[cross["decision"] == "crossing"].reset_index(drop=True)
    extra = []
    for i, row in crossing_only.iterrows():
        idx = val_tree.nearest(row.geometry)
        d = row.geometry.distance(val.iloc[idx].geometry)
        extra.append(d > args.tolerance_m)
    crossing_only["no_ground_truth_nearby"] = extra
    n_extra = sum(extra)
    print(f"\n--- Precision (are predicted crossings real?) ---")
    print(f"  {len(crossing_only) - n_extra}/{len(crossing_only)} predicted 'crossing' decisions have a "
          f"ground-truth bridge within {args.tolerance_m} m")
    print(f"  {n_extra} have none nearby - either a road the design drawing didn't need a structure for, "
          f"a genuinely new/unsurveyed crossing, or an OSM misclassification; spot-check before trusting them")

    val_wgs = val.to_crs(4326)
    val_wgs.to_file(OUT_PATH, driver="GeoJSON")
    print(f"\nwrote {OUT_PATH} (ground truth points with match status, for the review map)")


if __name__ == "__main__":
    main()
