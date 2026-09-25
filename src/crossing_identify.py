"""One command: clip a rail/road alignment to a boundary (optional), fetch
the OpenStreetMap road network for a corridor around it, and find every
point where a road crosses the alignment - chainage, crossing angle, IRC
road class, the connectivity/dead-end rule (R1-R4), and design width/span
from the IRC minimum (stage 6 - automated width measurement from imagery -
is a separate step, see measure_width_sam3.py).

Nothing about the project area is hardcoded: the alignment file, the
clip boundary, the corridor width, the output CRS and every rule
threshold are all arguments. Run with no arguments on this project and
you reproduce the existing Gujarat outputs; point it at a different
alignment/boundary pair to run it anywhere else.

Usage:
    python src/crossing_identify.py
    python src/crossing_identify.py --alignment my_line.geojson --boundary my_state.geojson --corridor-km 1.5
    python src/crossing_identify.py --alignment my_line.geojson --boundary ""   # unclipped: use the whole input
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from shapely.geometry import Point
from shapely.ops import split, unary_union

from geo_utils import flatten_geom, get_utm_epsg, write_csv, write_geojson
from irc_classify import classify, roadway_width
from shapely.strtree import STRtree

DEFAULT_ALIGNMENT = "input/DFCC_master_all_sections_Lines.geojson"
DEFAULT_BOUNDARY = "input/boundaries/gujarat.geojson"
DEFAULT_OUTPUT = "output/gujarat_crossings.geojson"
DEFAULT_SUMMARY_CSV = "output/gujarat_crossings_summary.csv"
DEFAULT_ALIGNMENT_OUTPUT = "shared/alignment_gujarat.geojson"
DEFAULT_ROADS_OUTPUT = "shared/roads_gujarat.geojson"
DEFAULT_CORRIDOR_OUTPUT = "shared/corridor_2km.geojson"

OSM_HIGHWAY_VALUES = [
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "track", "path", "footway",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
]


# ---------------------------------------------------------------- stage 1: alignment

def prepare_alignment(args) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(args.alignment)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf = gdf.to_crs(4326)

    if args.feature_type_field and args.feature_type_field in gdf.columns:
        gdf = gdf[gdf[args.feature_type_field] == args.feature_type_value].reset_index(drop=True)
    gdf = gdf[gdf.geometry.notna()].reset_index(drop=True)
    if gdf.empty:
        raise SystemExit(
            f"No features left after filtering {args.alignment} on "
            f"{args.feature_type_field}={args.feature_type_value!r} - check --feature-type-field/--feature-type-value"
        )

    id_field = args.id_field if args.id_field in gdf.columns else None

    if args.boundary:
        boundary = gpd.read_file(args.boundary).to_crs(4326)
        boundary_geom = boundary.geometry.union_all()
        print(f"clipping {len(gdf)} alignment feature(s) to {args.boundary}")
    else:
        boundary_geom = None
        print(f"no --boundary given - using all {len(gdf)} alignment feature(s) unclipped")

    rows = []
    for i, row in gdf.iterrows():
        base_id = str(row[id_field]) if id_field else f"seg{i + 1}"
        geom = row.geometry.intersection(boundary_geom) if boundary_geom is not None else row.geometry
        for part in flatten_geom(geom):
            rows.append({"base_id": base_id, "geometry": part})

    if not rows:
        raise SystemExit("No alignment geometry falls inside the given boundary.")

    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)

    centroid = out.geometry.union_all().centroid
    epsg = args.epsg or get_utm_epsg(centroid.x, centroid.y)
    print(f"working CRS: EPSG:{epsg}" + (" (auto-detected UTM zone)" if not args.epsg else ""))

    out = out.to_crs(epsg)
    out["length_m"] = out.geometry.length
    before = len(out)
    out = out[out["length_m"] >= args.min_segment_m].reset_index(drop=True)
    dropped = before - len(out)
    if dropped:
        print(f"dropped {dropped} sliver part(s) under {args.min_segment_m:.0f} m (boundary-clip artefacts)")
    if out.empty:
        raise SystemExit("Nothing left after dropping slivers - check --min-segment-m or the input geometry.")

    out = out.sort_values("base_id").reset_index(drop=True)
    out["part_index"] = out.groupby("base_id").cumcount() + 1
    parts_per_id = out.groupby("base_id")["part_index"].transform("max")
    out["align_name"] = out["base_id"].where(parts_per_id == 1, out["base_id"] + "-part" + out["part_index"].astype(str))
    out["section_id"] = out["base_id"]
    out.insert(0, "align_id", range(1, len(out) + 1))
    out.attrs["epsg"] = epsg

    total_km = out["length_m"].sum() / 1000
    print(f"alignment: {len(out)} segment(s), {total_km:.2f} km total")
    return out


# ---------------------------------------------------------------- stage 2: roads

def fetch_road_network(corridor_wgs84, args) -> gpd.GeoDataFrame:
    print("Querying Overpass for the OSM road network in the corridor ...")
    ox.settings.requests_timeout = 300
    custom_filter = '["highway"~"^({})$"]'.format("|".join(OSM_HIGHWAY_VALUES))
    G = ox.graph_from_polygon(
        corridor_wgs84, custom_filter=custom_filter, retain_all=True, simplify=False, truncate_by_edge=True,
    )
    print(f"  graph: {G.number_of_nodes()} node(s), {G.number_of_edges()} edge(s) (directed)")

    # graph_from_polygon returns a MultiDiGraph: every two-way street is two
    # directed edges (u->v and v->u) with the same geometry. Collapse those
    # back to one edge each, or every crossing gets counted twice.
    G = ox.convert.to_undirected(G)
    print(f"  graph: {G.number_of_nodes()} node(s), {G.number_of_edges()} edge(s) (undirected)")

    edges = ox.graph_to_gdfs(G, nodes=False, edges=True).reset_index()
    keep_cols = [c for c in ["u", "v", "key", "osmid", "highway", "ref", "name",
                              "lanes", "width", "oneway", "geometry"] if c in edges.columns]
    edges = edges[keep_cols]
    for col in ["highway", "ref", "name", "lanes", "width"]:
        if col in edges.columns:
            edges[col] = edges[col].apply(lambda v: v[0] if isinstance(v, list) else v)
    print(f"roads: {len(edges)} edge(s)")
    return edges


# ---------------------------------------------------------------- stage 4: crossings

def bearing(p0: tuple[float, float], p1: tuple[float, float]) -> float:
    return math.degrees(math.atan2(p1[0] - p0[0], p1[1] - p0[1]))


def tangent_bearing(line, at_point: Point, win: float = 10.0) -> float:
    d = line.project(at_point)
    p = line.interpolate(max(d - win, 0.0))
    q = line.interpolate(min(d + win, line.length))
    if p.equals(q):
        return 0.0
    return bearing((p.x, p.y), (q.x, q.y))


def acute_angle(a_deg: float, r_deg: float) -> float:
    diff = abs(a_deg - r_deg) % 180
    return diff if diff <= 90 else 180 - diff


def iter_points(geom):
    if geom.is_empty:
        return
    if geom.geom_type == "Point":
        yield geom
    elif geom.geom_type in ("MultiPoint", "GeometryCollection"):
        for part in geom.geoms:
            yield from iter_points(part)
    # LineString/overlap results (alignment running along a road) are skipped.


def find_candidates(align: gpd.GeoDataFrame, roads: gpd.GeoDataFrame, align_barrier) -> gpd.GeoDataFrame:
    align_sindex = align.sindex
    rows = []
    for _, road in roads.iterrows():
        rgeom = road.geometry
        if rgeom is None or rgeom.is_empty or not rgeom.intersects(align_barrier):
            continue
        inter = rgeom.intersection(align_barrier)
        for pt in iter_points(inter):
            near_ids = list(align_sindex.query(pt.buffer(1.0)))
            if not near_ids:
                continue
            align_row = min((align.iloc[i] for i in near_ids), key=lambda r: r.geometry.distance(pt))
            chainage_m = align_row.geometry.project(pt)
            a_brg = tangent_bearing(align_row.geometry, pt)
            r_brg = tangent_bearing(rgeom, pt)
            theta = acute_angle(a_brg, r_brg)
            rows.append({
                "align_id": align_row["align_id"],
                "align_name": align_row["align_name"],
                "section_id": align_row["section_id"],
                "chainage_m": round(chainage_m, 1),
                "angle_deg": round(theta, 1),
                "highway": road.get("highway"),
                "ref": road.get("ref"),
                "name": road.get("name"),
                "lanes": road.get("lanes"),
                "irc_class": classify(road.get("highway"), road.get("ref")),
                "geometry": pt,
            })
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=align.crs)


def merge_dual_carriageways(cand: gpd.GeoDataFrame, tol_m: float) -> gpd.GeoDataFrame:
    """R5: two OSM ways within tol_m on the same align segment that share a
    ref are one physical crossing (divided highway) - keep one row."""
    cand = cand.sort_values(["align_id", "chainage_m"]).reset_index(drop=True)
    cand["dual_carriageway"] = False
    used = set()
    out_rows = []
    for i, row in cand.iterrows():
        if i in used:
            continue
        if row["ref"]:
            group = cand[
                (cand["align_id"] == row["align_id"])
                & (cand["ref"] == row["ref"])
                & (~cand.index.isin(used))
            ]
            group = group[group.geometry.distance(row.geometry) <= tol_m]
        else:
            group = cand.loc[[i]]
        used.update(group.index)
        rep = row.copy()
        if len(group) > 1:
            rep["dual_carriageway"] = True
        out_rows.append(rep)
    return gpd.GeoDataFrame(out_rows, geometry="geometry", crs=cand.crs).reset_index(drop=True)


def build_barrier_graph(roads: gpd.GeoDataFrame, align: gpd.GeoDataFrame, align_barrier) -> nx.Graph:
    """Split every road that crosses the alignment at the crossing point, so
    the two halves never share a graph node - the alignment acts as a wall."""

    def side_of(pt: Point, align_row) -> int:
        d = align_row.geometry.project(pt)
        a = align_row.geometry.interpolate(max(d - 1.0, 0.0))
        b = align_row.geometry.interpolate(min(d + 1.0, align_row.geometry.length))
        cross = (b.x - a.x) * (pt.y - a.y) - (b.y - a.y) * (pt.x - a.x)
        return 1 if cross > 0 else -1

    def node_key(xy, side):
        k = (round(xy[0], 1), round(xy[1], 1))
        on_barrier = align_barrier.distance(Point(xy)) < 0.5
        return ("X", k, side) if on_barrier else k

    align_rows = list(align.itertuples())
    G = nx.Graph()
    for _, road in roads.iterrows():
        rgeom = road.geometry
        if rgeom is None or rgeom.is_empty:
            continue
        if rgeom.crosses(align_barrier):
            try:
                parts = list(split(rgeom, align_barrier).geoms)
            except Exception:
                parts = [rgeom]
        else:
            parts = [rgeom]
        for part in parts:
            if part.length == 0:
                continue
            mid = part.interpolate(0.5, normalized=True)
            nearest_align = min(align_rows, key=lambda r: r.geometry.distance(mid))
            s = side_of(mid, nearest_align)
            u = node_key(part.coords[0], s)
            v = node_key(part.coords[-1], s)
            if u == v:
                continue
            G.add_edge(u, v, length=part.length)
    return G


def apply_connectivity_rule(cand: gpd.GeoDataFrame, G: nx.Graph, corridor,
                             min_component_len_m: float, junction_flag_m: float, skew_flag_deg: float) -> gpd.GeoDataFrame:
    inner = corridor.buffer(-5.0)

    def connected(n) -> bool:
        comp = nx.node_connected_component(G, n)
        leaves_window = any(
            not inner.contains(Point(k)) for k in comp if not (isinstance(k, tuple) and k and k[0] == "X")
        )
        subg = G.subgraph(comp)
        has_junction = any(deg >= 3 for _, deg in subg.degree())
        comp_len = sum(d.get("length", 0.0) for _, _, d in subg.edges(data=True))
        return leaves_window or (has_junction and comp_len >= min_component_len_m)

    def decide(pt: Point):
        xy = (round(pt.x, 1), round(pt.y, 1))
        ends = [n for n in ((("X", xy, 1)), (("X", xy, -1))) if n in G]
        if len(ends) < 2:
            return "no_crossing", "R2", "road ends at the alignment - only one side present in the network"
        ok = [connected(n) for n in ends]
        if all(ok):
            return "crossing", "R1", "both halves reach another road or leave the corridor window"
        if not any(ok):
            return "no_crossing", "R3", "neither half connects to the wider network - check the map"
        return "review", "R4", "one half dead-ends close to the alignment - check imagery for buildings/habitation"

    decisions = cand.geometry.apply(decide)
    cand["decision"] = decisions.apply(lambda t: t[0])
    cand["rule"] = decisions.apply(lambda t: t[1])
    cand["reason"] = decisions.apply(lambda t: t[2])

    junction_pts = [Point(k) for k, d in G.degree() if isinstance(k, tuple) and k[0] != "X" and d >= 3]
    jtree = STRtree(junction_pts) if junction_pts else None
    cand["near_junction"] = cand.geometry.apply(
        lambda pt: bool(jtree is not None and len(jtree.query(pt.buffer(junction_flag_m))) > 0)
    )
    cand["skewed"] = cand["angle_deg"] < skew_flag_deg
    return cand


def flag_nearby_crossings(cand: gpd.GeoDataFrame, flag_m: float) -> gpd.GeoDataFrame:
    """R8: two crossings this close together are often the same physical road
    mapped as two separate/parallel OSM ways (a duplicate trace, or a paved
    road with a parallel unpaved service track) rather than genuinely two
    structures - not auto-merged (unlike R5's same-ref case), since it can
    also be a real, separate road; flagged for the designer to decide.
    Requires cand["crossing_id"] to already be assigned."""
    cand = cand.reset_index(drop=True)
    nearby_id = [None] * len(cand)
    nearby_dist = [None] * len(cand)
    for _, idx in cand.groupby("align_id").groups.items():
        idx = list(idx)
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                ia, ib = idx[a], idx[b]
                d = cand.geometry.iloc[ia].distance(cand.geometry.iloc[ib])
                if d <= flag_m:
                    if nearby_dist[ia] is None or d < nearby_dist[ia]:
                        nearby_id[ia], nearby_dist[ia] = cand["crossing_id"].iloc[ib], round(d, 1)
                    if nearby_dist[ib] is None or d < nearby_dist[ib]:
                        nearby_id[ib], nearby_dist[ib] = cand["crossing_id"].iloc[ia], round(d, 1)
    cand["nearby_crossing_id"] = nearby_id
    cand["nearby_dist_m"] = nearby_dist
    return cand


def compute_width_span(cand: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    def width_span(row):
        w = roadway_width(row["irc_class"])
        if w is None:
            return pd.Series({"design_width_m": None, "span_m": None, "width_source": "needs field/imagery survey"})
        span = w / math.sin(math.radians(max(row["angle_deg"], 1e-3)))
        return pd.Series({"design_width_m": w, "span_m": round(span, 2), "width_source": f"IRC {row['irc_class']}"})

    ws = cand.apply(width_span, axis=1)
    return pd.concat([cand, ws], axis=1)


# ---------------------------------------------------------------- CLI

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alignment", default=DEFAULT_ALIGNMENT, help="Alignment centreline file (any format geopandas reads)")
    ap.add_argument("--feature-type-field", default="feature_type",
                     help="Property to filter alignment features by, e.g. a mixed export with ticks/ROW/centreline. Pass '' to disable.")
    ap.add_argument("--feature-type-value", default="centerline", help="Value to keep when --feature-type-field is set")
    ap.add_argument("--id-field", default="section_id", help="Property used to name/group alignment segments, if present")
    ap.add_argument("--boundary", default=DEFAULT_BOUNDARY,
                     help="State/country/AOI boundary polygon to clip the alignment to. Pass '' to use the whole input, unclipped.")
    ap.add_argument("--corridor-km", type=float, default=2.0, help="Corridor half-width (km) for the OSM fetch and the connectivity-rule window")
    ap.add_argument("--min-segment-m", type=float, default=50.0, help="Drop alignment slivers shorter than this after clipping")
    ap.add_argument("--epsg", type=int, default=None, help="Metric CRS to work in. Default: auto-detect UTM zone from the data's centroid")
    ap.add_argument("--id-prefix", default="GJ", help="Prefix for generated crossing_id values, e.g. GJ-001")
    ap.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="Output crossings GeoJSON")
    ap.add_argument("--summary-csv", default=DEFAULT_SUMMARY_CSV, help="Output crossings summary CSV")
    ap.add_argument("--alignment-output", default=DEFAULT_ALIGNMENT_OUTPUT, help="Where to save the clipped alignment")
    ap.add_argument("--roads-output", default=DEFAULT_ROADS_OUTPUT, help="Where to save the fetched OSM road network")
    ap.add_argument("--corridor-output", default=DEFAULT_CORRIDOR_OUTPUT, help="Where to save the corridor polygon")
    ap.add_argument("--min-component-len-m", type=float, default=300.0, help="Min network length for a component without leaving the corridor to count as connected")
    ap.add_argument("--dual-merge-tol-m", type=float, default=30.0, help="R5: merge distance for same-ref dual-carriageway crossings")
    ap.add_argument("--junction-flag-m", type=float, default=50.0, help="R6: flag a crossing within this of an existing junction")
    ap.add_argument("--skew-flag-deg", type=float, default=45.0, help="R7: flag a crossing angle below this")
    ap.add_argument("--nearby-crossing-flag-m", type=float, default=40.0, help="R8: flag another crossing on the same segment within this distance")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    align = prepare_alignment(args)
    epsg = align.attrs["epsg"]
    write_geojson(align.to_crs(4326), Path(args.alignment_output))
    print(f"wrote {args.alignment_output}")

    corridor_m = align.geometry.buffer(args.corridor_km * 1000).union_all()
    corridor_wgs = gpd.GeoSeries([corridor_m], crs=epsg).to_crs(4326)
    write_geojson(gpd.GeoDataFrame(geometry=corridor_wgs, crs=4326), Path(args.corridor_output))
    print(f"wrote {args.corridor_output}")

    roads = fetch_road_network(corridor_wgs.iloc[0], args).to_crs(epsg)
    write_geojson(roads.to_crs(4326), Path(args.roads_output))
    print(f"wrote {args.roads_output}")

    corridor = corridor_m
    align_barrier = unary_union(align.geometry.values)

    cand = find_candidates(align, roads, align_barrier)
    if cand.empty:
        print("No crossings found.")
        return
    print(f"\nstage 4: {len(cand)} crossing candidate(s) before R5 merge")

    cand = merge_dual_carriageways(cand, args.dual_merge_tol_m)
    print(f"stage 4: {len(cand)} crossing(s) after R5 dual-carriageway merge")

    G = build_barrier_graph(roads, align, align_barrier)
    cand = apply_connectivity_rule(cand, G, corridor, args.min_component_len_m, args.junction_flag_m, args.skew_flag_deg)
    cand = compute_width_span(cand)

    cand = cand.sort_values(["align_id", "chainage_m"]).reset_index(drop=True)
    cand.insert(0, "crossing_id", [f"{args.id_prefix}-{i + 1:03d}" for i in range(len(cand))])
    cand = flag_nearby_crossings(cand, args.nearby_crossing_flag_m)  # needs metres, before reprojecting to WGS84
    cand_wgs = cand.to_crs(4326)

    write_geojson(cand_wgs, Path(args.output))
    write_csv(cand_wgs.drop(columns="geometry"), Path(args.summary_csv))

    print(f"\nwrote {args.output}")
    print(f"wrote {args.summary_csv}")
    print("\nDecision counts:")
    print(cand_wgs["decision"].value_counts().to_string())
    print("\nBy IRC class:")
    print(cand_wgs.groupby(["irc_class", "decision"]).size().to_string())
    needs_survey = cand_wgs[(cand_wgs["decision"] != "no_crossing") & (cand_wgs["design_width_m"].isna())]
    print(f"\n{len(needs_survey)} crossing(s) need a field/imagery width survey (track/footpath class, no IRC minimum).")
    nearby = cand_wgs[cand_wgs["nearby_crossing_id"].notna()]
    if len(nearby):
        print(f"\n{len(nearby) // 2} pair(s) flagged R8 (another crossing within {args.nearby_crossing_flag_m:.0f} m - "
              f"possibly the same road mapped twice, or a parallel service track; verify before designing separate structures):")
        seen = set()
        for _, r in nearby.iterrows():
            pair = tuple(sorted([r["crossing_id"], r["nearby_crossing_id"]]))
            if pair in seen:
                continue
            seen.add(pair)
            print(f"  {pair[0]} <-> {pair[1]}  ({r['nearby_dist_m']} m)")


if __name__ == "__main__":
    main()
