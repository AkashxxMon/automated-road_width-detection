"""One command: clip a rail/road alignment to a boundary (optional), load
the road network for a corridor around it from local NIC road data, plus
OpenStreetMap roads where NIC has none (Geofabrik extracts, downloaded once -
see osm_roads.py), and find every point where a road crosses the alignment -
chainage, crossing angle, IRC road class, the connectivity/dead-end rule (R1,
R2) and the identification filters (R10, R11). The IRC class assigned here is used
later only to flag an implausible measurement for review - design
width/span come entirely from the SAM3 measurement (stage 6, a separate
step - see measure_width_sam3.py), not from an IRC minimum.

Nothing about the project area is hardcoded: the alignment file, the
clip boundary, the corridor width, the output CRS and every rule
threshold are all arguments. Run with no arguments on this project and
you reproduce the existing Gujarat outputs; point it at a different
alignment/boundary pair to run it anywhere else.

Usage:
    python src/crossing_identify.py
    python src/crossing_identify.py --alignment my_line.geojson --state Maharashtra --corridor-km 1.5
    python src/crossing_identify.py --alignment my_line.geojson --boundary my_state.geojson
    python src/crossing_identify.py --alignment my_line.geojson --boundary ""   # unclipped: use the whole input
"""

from __future__ import annotations

import argparse
import glob
import math
import re
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import Point
from shapely.ops import split, unary_union

from geo_utils import (flatten_geom, get_utm_epsg, region_paths, region_prefix, region_slug, remember_region,
                       write_csv, write_geojson)
from irc_classify import classify
from osm_roads import load_osm_roads
from shapely.strtree import STRtree

DEFAULT_ALIGNMENT = "input/DFCC_master_all_sections_Lines.geojson"
DEFAULT_BOUNDARY = "input/boundaries/gujarat.geojson"
DEFAULT_STATES_FILE = "input/boundaries/india_states.geojson"
DEFAULT_STATE_NAME_FIELD = "NAME_1"
DEFAULT_ROADS_GLOB = "input/Roads/NIC_Roads.*.parquet"
# output paths and the crossing_id prefix default from the region (--region, else the --state
# name, else the boundary file's name) - see geo_utils.region_paths()

# accepted here, and by irc_classify._HIGHWAY_TO_IRC - a road classified into
# one of these OSM-vocabulary buckets (either passed through directly from a
# NIC column that already used it, or inferred from NIC's free-text class
# names below) is what classify() keys its IRC-class guess off of
_OSM_HIGHWAY_VALUES = {
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "track", "path", "footway",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
}


# ---------------------------------------------------------------- stage 1: alignment

def resolve_boundary(args):
    """The clip boundary, from one of three sources (at most one may be
    given): --state (looked up by name in --states-file), --boundary (a
    polygon file directly), or neither passed at all (falls back to this
    project's own default boundary file). Pass --boundary "" explicitly to
    get no clip at all (use the whole alignment input) - or --boundary none, which
    PowerShell passes through where it drops an empty ""."""
    if args.boundary is not None and args.boundary.strip().lower() in ("none", "all"):
        args.boundary = ""  # a spelling of "" that survives PowerShell, which drops empty arguments
    if args.state is not None and args.boundary is not None:
        raise SystemExit("Pass --state or --boundary, not both.")

    if args.state is not None:
        states = gpd.read_file(args.states_file)
        if args.state_name_field not in states.columns:
            raise SystemExit(
                f"{args.states_file} has no '{args.state_name_field}' property - "
                f"available properties: {list(states.columns)}"
            )
        match = states[states[args.state_name_field].str.strip().str.casefold() == args.state.strip().casefold()]
        if match.empty:
            available = sorted(states[args.state_name_field].dropna().unique())
            raise SystemExit(f"No state named {args.state!r} found in {args.states_file}. Available: {available}")
        print(f"using state boundary for {args.state!r} from {args.states_file} "
              f"({len(match)} feature(s) matched)")
        return match.to_crs(4326).geometry.union_all()

    if args.boundary is not None:
        if args.boundary == "":
            return None  # explicitly requested: no clip
        return gpd.read_file(args.boundary).to_crs(4326).geometry.union_all()

    # neither --state nor --boundary given at all - this project's own default
    return gpd.read_file(DEFAULT_BOUNDARY).to_crs(4326).geometry.union_all()


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

    boundary_geom = resolve_boundary(args)
    if boundary_geom is not None:
        print(f"clipping {len(gdf)} alignment feature(s) to the given boundary")
    else:
        print(f"no --state/--boundary given - using all {len(gdf)} alignment feature(s) unclipped")

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
#
# NIC's road files are a nationwide merge of many different states' own GIS
# exports, so the columns that carry road class/route-number/name are not
# consistent from row to row - one state's data might put "National Highway"
# in `Class`, another puts the literal short code "NH" in `road_type_1`, a
# third already used plain OSM highway-tag vocabulary ("primary", "trunk", ...)
# in `type_2`. The three _nic_*() helpers below coalesce across every variant
# column actually observed in this dataset (see the NIC_REF/CLASS/NAME column
# lists), in priority order, so downstream code (classify(), R5's ref-match,
# etc.) can keep treating `highway`/`ref`/`name` the same way it did for OSM.

_NIC_REF_COLS = ["NH_no", "road_code_3", "Road_code_1", "Road_Code", "road_type_1", "CLASS_2"]
_NIC_CLASS_TEXT_COLS = ["Class", "class_1", "CLASS_2", "road_type_1"]
_NIC_NAME_COLS = ["Name", "name_1", "road_name", "NAME_2"]
# surface: most state layers fill one of these with "Metalled (Road)" / "Unmetalled (Road)";
# Gujarat, Odisha, Rajasthan, Haryana, Punjab, Assam, Delhi don't record it at all
_NIC_SURFACE_COLS = ["Type", "type_2", "TYPE_1", "Road_type", "ROADTYPE"]
_NIC_RAW_HIGHWAY_COLS = ["type_2"]  # already OSM-vocabulary in some source states' rows
_NIC_REF_PATTERN = re.compile(r"\b(NH|SH|MDR|ODR)\b", re.IGNORECASE)


def _nic_ref(row: pd.Series) -> str | None:
    """Several of the _NIC_REF_COLS candidates are actually generic
    classification codes reused across thousands of unrelated roads (a plain
    "42" or "village_road", not a route number) - only accept a value that
    genuinely looks like a highway reference (contains NH/SH/MDR/ODR), or R5's
    same-ref dual-carriageway merge could wrongly fuse two unrelated nearby
    minor roads that happen to share the same generic code."""
    for col in _NIC_REF_COLS:
        v = row.get(col)
        if isinstance(v, str) and v.strip() and _NIC_REF_PATTERN.search(v):
            return v.strip()
    return None


def _nic_highway(row: pd.Series) -> str | None:
    for col in _NIC_RAW_HIGHWAY_COLS:
        v = row.get(col)
        if isinstance(v, str):
            v = v.strip().lower()
            if v in _OSM_HIGHWAY_VALUES:
                return v
    text = None
    for col in _NIC_CLASS_TEXT_COLS:
        v = row.get(col)
        if isinstance(v, str) and v.strip():
            text = v.strip().lower()
            break
    if text is None:
        return None
    if "express" in text or "asian highway" in text:
        return "motorway"
    if "national" in text or text == "nh":
        return "trunk"
    if "state" in text or text == "sh":
        return "primary"
    if "district" in text or text in ("mdr", "odr"):
        return "tertiary"
    if "village" in text:
        return "residential"
    if "street" in text or "town road" in text:
        return "residential"
    if "other" in text:
        return "track"
    return None


def _nic_surface(row: pd.Series) -> str | None:
    """'paved' (metalled), 'unpaved' (unmetalled) or None (not recorded)."""
    for col in _NIC_SURFACE_COLS:
        v = row.get(col)
        if isinstance(v, str) and v.strip():
            v = v.strip().lower()
            if v.startswith(("unmetal", "kutcha", "kachcha", "earthen", "unpaved")):
                return "unpaved"
            if v.startswith(("metal", "pucca", "paved", "bt", "black top", "cc", "concrete")):
                return "paved"
    return None


def _nic_name(row: pd.Series) -> str | None:
    for col in _NIC_NAME_COLS:
        v = row.get(col)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def load_road_network(corridor_wgs84, args) -> gpd.GeoDataFrame:
    """Road network for the corridor, read straight from local NIC road
    parquet file(s) - no live API, no network dependency at all. Only the
    row groups whose bbox overlaps the corridor are actually read (GeoParquet
    bbox pushdown), then the result is clipped precisely to the corridor
    polygon before the per-row class/ref/name mapping runs."""
    paths = sorted(glob.glob(args.roads_glob))
    if not paths:
        raise SystemExit(f"No files matched --roads-glob {args.roads_glob!r}")

    minx, miny, maxx, maxy = corridor_wgs84.bounds
    print(f"Loading NIC road data for the corridor from {len(paths)} file(s): "
          f"{', '.join(Path(p).name for p in paths)}")

    parts = []
    for p in paths:
        part = gpd.read_parquet(p, bbox=(minx, miny, maxx, maxy))
        print(f"  {Path(p).name}: {len(part)} row(s) in bbox")
        if len(part):
            parts.append(part)

    if not parts:
        raise SystemExit("No NIC road data intersects this corridor's bounding box.")

    roads = pd.concat(parts, ignore_index=True)
    roads = gpd.GeoDataFrame(roads, geometry="geometry", crs=parts[0].crs).to_crs(4326)
    roads = roads[roads.geometry.notna() & roads.intersects(corridor_wgs84)].reset_index(drop=True)
    # NIC data mixes LineString and MultiLineString rows - split every
    # MultiLineString into its own simple-LineString rows (duplicating the
    # other columns), since build_barrier_graph/find_candidates below work
    # on individual LineString coordinate sequences only.
    roads = roads.explode(index_parts=False).reset_index(drop=True)
    print(f"  {len(roads)} road(s) after precise corridor clip")

    roads["highway"] = roads.apply(_nic_highway, axis=1)
    roads["ref"] = roads.apply(_nic_ref, axis=1)
    roads["name"] = roads.apply(_nic_name, axis=1)
    roads["surface"] = roads.apply(_nic_surface, axis=1)
    roads = roads[["highway", "ref", "name", "surface", "geometry"]]
    print(f"roads: {len(roads)} edge(s)")
    return roads


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
                "surface": road.get("surface"),
                "irc_class": classify(road.get("highway"), road.get("ref")),
                "road_source": road.get("road_source", "nic"),
                "geometry": pt,
            })
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=align.crs)


def _is_route_number(ref) -> bool:
    """A ref that names one route (NH48, SH-17, NH-6/AH-46), not just a road
    class - most NIC refs are the bare class code ("SH", "ODR", "NH"), which
    every road of that class shares."""
    return isinstance(ref, str) and any(ch.isdigit() for ch in ref)


def merge_dual_carriageways(cand: gpd.GeoDataFrame, tol_m: float) -> gpd.GeoDataFrame:
    """R5: two road segments within tol_m on the same align segment that share
    a route number are one physical crossing (divided highway) - keep one row.
    A bare class code never merges: two different district roads 30 m apart
    would otherwise lose one of them here, before it's ever measured. Divided
    roads without a route number are merged later, from the measured widths
    (structure_decide.py R9)."""
    cand = cand.sort_values(["align_id", "chainage_m"]).reset_index(drop=True)
    cand["dual_carriageway"] = False
    used = set()
    out_rows = []
    for i, row in cand.iterrows():
        if i in used:
            continue
        if _is_route_number(row["ref"]):
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
    """Road network as a graph: every road vertex is a node and every segment
    between two vertices an edge, so two roads that share a vertex anywhere
    along their length are connected - NIC road lines run straight through
    junctions, with side roads joining at an intermediate vertex, so joining
    lines by their end points only would miss most junctions. Every road
    that crosses the alignment is split at the crossing point, so the two
    halves never share a graph node - the alignment acts as a wall."""

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
            coords = [c[:2] for c in part.coords]
            # only a part's two ends can sit on the alignment (it was split there); every
            # vertex in between is a plain node, shared with any other road that passes through it
            keys = [node_key(coords[0], s)] + [(round(x, 1), round(y, 1)) for x, y in coords[1:-1]] \
                + [node_key(coords[-1], s)]
            for (a, b), u, v in zip(zip(coords, coords[1:]), keys, keys[1:]):
                if u != v:
                    G.add_edge(u, v, length=math.dist(a, b))
    return G


def apply_connectivity_rule(cand: gpd.GeoDataFrame, G: nx.Graph, junction_flag_m: float,
                            skew_flag_deg: float) -> gpd.GeoDataFrame:
    # the two halves' nodes at each crossing point: the nearest on each side within 0.5 m, rather
    # than an exact 0.1 m-rounded key - a rounding miss would otherwise drop a real crossing as
    # R2. Nearest, because another road can cross the alignment a few decimetres away
    x_nodes = [n for n in G.nodes if isinstance(n, tuple) and n and n[0] == "X"]

    def decide(pt: Point):
        nearest = {}
        for n in x_nodes:
            d = math.dist(n[1], (pt.x, pt.y))
            if d < 0.5 and d < nearest.get(n[2], (math.inf,))[0]:
                nearest[n[2]] = (d, n)
        ends = [n for _, n in nearest.values()]
        if len(ends) < 2:
            return "no_crossing", "R2", "road ends at the alignment - only one side present in the network"
        # no R3/R4 (a road whose halves don't join the wider network): a road the alignment physically
        # cuts through is never dropped or queried because the road data doesn't join it up - NIC is a
        # merge of state exports with gaps between them, so an unconnected line is a data gap
        return "crossing", "R1", "the road is present on both sides of the alignment"

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


def drop_known_crossings(extra: gpd.GeoDataFrame, known: gpd.GeoDataFrame, buffer_m: float,
                         angle_deg: float) -> gpd.GeoDataFrame:
    """OSM crossings that are a NIC road already found: within buffer_m of a
    NIC crossing on the same alignment piece, at a similar crossing angle. NIC
    lines can sit tens of metres off the real road, so a plain distance alone
    would either keep duplicates or drop separate roads close together."""
    keep = []
    for _, r in extra.iterrows():
        k = known[known["align_id"] == r["align_id"]]
        near = k[(k.geometry.distance(r.geometry) <= buffer_m) & ((k["angle_deg"] - r["angle_deg"]).abs() <= angle_deg)]
        keep.append(near.empty)
    return extra[keep].reset_index(drop=True)


def apply_identification_filters(cand: gpd.GeoDataFrame, args) -> gpd.GeoDataFrame:
    """R10 and R11: crossings that, scored against the designers' bridges, are
    almost never one (angle under 15 deg 4%, tracks 19%) become no_crossing - kept in the register with the
    reason, but not measured or given a structure."""
    drop_classes = {c.strip().upper() for c in args.drop_classes.split(",") if c.strip()}
    rules = [
        (cand["angle_deg"] < args.min_angle_deg, "R10",
         f"road runs along the alignment (crossing angle below {args.min_angle_deg:g} deg), not across it"),
        (cand["irc_class"].isin(drop_classes), "R11", "road class excluded (--drop-classes)"),
    ]
    for mask, rule, reason in rules:
        mask = mask & (cand["decision"] != "no_crossing")
        cand.loc[mask, ["decision", "rule", "reason"]] = ["no_crossing", rule, reason]
        if mask.any():
            print(f"{rule}: {int(mask.sum())} crossing(s) -> no_crossing ({reason})")
    return cand


def flag_nearby_crossings(cand: gpd.GeoDataFrame, flag_m: float) -> gpd.GeoDataFrame:
    """R8: two crossings this close together are often the same physical road
    mapped as two separate/parallel segments (a duplicate trace, or a paved
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


def init_width_columns(cand: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """design_width_m/span_m come only from the SAM3 measurement in
    measure_width_sam3.py - this just marks every row as pending that
    measurement. irc_class still feeds the SUSPECT/review check there,
    but no longer sets or floors the design value itself."""
    cand["design_width_m"] = None
    cand["span_m"] = None
    cand["width_source"] = "needs field/imagery survey"
    return cand


# ---------------------------------------------------------------- CLI

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alignment", default=DEFAULT_ALIGNMENT, help="Alignment centreline file (any format geopandas reads)")
    ap.add_argument("--feature-type-field", default="feature_type",
                     help="Property to filter alignment features by, e.g. a mixed export with ticks/ROW/centreline. Pass '' to disable.")
    ap.add_argument("--feature-type-value", default="centerline", help="Value to keep when --feature-type-field is set")
    ap.add_argument("--id-field", default="section_id", help="Property used to name/group alignment segments, if present")
    ap.add_argument("--boundary", default=None,
                     help=f"State/country/AOI boundary polygon to clip the alignment to. Pass '' to use the whole "
                          f"input, unclipped. Default if neither this nor --state is given: {DEFAULT_BOUNDARY}")
    ap.add_argument("--state", default=None,
                     help="Indian state name (e.g. 'Gujarat') to clip to, looked up in --states-file instead of "
                          "needing your own boundary file. Cannot be combined with --boundary.")
    ap.add_argument("--states-file", default=DEFAULT_STATES_FILE, help="All-India state boundaries file used by --state")
    ap.add_argument("--state-name-field", default=DEFAULT_STATE_NAME_FIELD, help="Property in --states-file holding the state name")
    ap.add_argument("--corridor-km", type=float, default=2.0, help="Corridor half-width (km) for the road-data crop")
    ap.add_argument("--roads-glob", default=DEFAULT_ROADS_GLOB,
                     help="Glob for local NIC road parquet file(s) to load and crop to the corridor")
    ap.add_argument("--min-segment-m", type=float, default=50.0, help="Drop alignment slivers shorter than this after clipping")
    ap.add_argument("--epsg", type=int, default=None, help="Metric CRS to work in. Default: auto-detect UTM zone from the data's centroid")
    ap.add_argument("--region", default=None,
                     help="Name for this run's files (output/<region>_crossings.geojson, shared/roads_<region>.geojson, "
                          "...) - pass the same --region to the later steps. Default: the --state name, else the "
                          "--boundary file's name (gujarat for the default boundary), or 'all' when unclipped")
    ap.add_argument("--id-prefix", default=None,
                     help="Prefix for generated crossing_id values, e.g. MH-001. Default: the region's state code")
    ap.add_argument("-o", "--output", default=None, help="Output crossings GeoJSON. Default: output/<region>_crossings.geojson")
    ap.add_argument("--summary-csv", default=None, help="Output crossings summary CSV")
    ap.add_argument("--alignment-output", default=None, help="Where to save the clipped alignment")
    ap.add_argument("--roads-output", default=None, help="Where to save the loaded road network")
    ap.add_argument("--corridor-output", default=None, help="Where to save the corridor polygon")
    ap.add_argument("--dual-merge-tol-m", type=float, default=30.0, help="R5: merge distance for same-ref dual-carriageway crossings")
    ap.add_argument("--junction-flag-m", type=float, default=50.0, help="R6: flag a crossing within this of an existing junction")
    ap.add_argument("--skew-flag-deg", type=float, default=45.0, help="R7: flag a crossing angle below this")
    ap.add_argument("--nearby-crossing-flag-m", type=float, default=40.0, help="R8: flag another crossing on the same segment within this distance")
    ap.add_argument("--osm", action=argparse.BooleanOptionalAction, default=True,
                     help="Add OpenStreetMap roads (Geofabrik extracts, see osm_roads.py) where the NIC data has none")
    ap.add_argument("--osm-buffer-m", type=float, default=60.0,
                     help="An OSM crossing within this of a NIC crossing on the same alignment piece, at a similar "
                          "angle (--osm-angle-deg), is the same road - the NIC one is kept")
    ap.add_argument("--osm-angle-deg", type=float, default=20.0, help="See --osm-buffer-m")
    ap.add_argument("--osm-max-age-days", type=float, default=30.0, help="Re-download an OSM extract older than this")
    ap.add_argument("--min-angle-deg", type=float, default=15.0,
                     help="R10: a road crossing at less than this runs along the alignment, not across it -> no_crossing")
    ap.add_argument("--drop-classes", default="TRK",
                     help="R11: IRC classes that never get a crossing (comma-separated; '' for none) -> no_crossing")
    args = ap.parse_args()

    if args.boundary is not None and args.boundary.strip().lower() in ("none", "all"):
        args.boundary = ""
    if args.region is None:
        if args.state is not None:
            args.region = args.state
        elif args.boundary == "":
            args.region = "all"
        else:
            args.region = Path(args.boundary or DEFAULT_BOUNDARY).stem
    args.region = region_slug(args.region)
    paths = region_paths(args.region)
    for arg, key in (("output", "crossings"), ("summary_csv", "crossings_summary"), ("alignment_output", "alignment"),
                     ("roads_output", "roads"), ("corridor_output", "corridor")):
        if getattr(args, arg) is None:
            setattr(args, arg, paths[key])
    if args.id_prefix is None:
        args.id_prefix = region_prefix(args.region)
    return args


def main() -> None:
    args = parse_args()
    print(f"region: {args.region} - crossing ids {args.id_prefix}-001..., outputs {args.output}, {args.roads_output}")

    align = prepare_alignment(args)
    epsg = align.attrs["epsg"]
    write_geojson(align.to_crs(4326), Path(args.alignment_output))
    print(f"wrote {args.alignment_output}")

    corridor_m = align.geometry.buffer(args.corridor_km * 1000).union_all()
    corridor_wgs = gpd.GeoSeries([corridor_m], crs=epsg).to_crs(4326)
    write_geojson(gpd.GeoDataFrame(geometry=corridor_wgs, crs=4326), Path(args.corridor_output))
    print(f"wrote {args.corridor_output}")

    nic = load_road_network(corridor_wgs.iloc[0], args).to_crs(epsg).assign(road_source="nic")
    osm = (load_osm_roads(corridor_wgs.iloc[0], region_paths(args.region)["osm_roads"], args.osm_max_age_days)
           .to_crs(epsg).assign(road_source="osm") if args.osm else nic.iloc[:0])
    roads = gpd.GeoDataFrame(pd.concat([nic, osm.drop(columns="osm_id", errors="ignore")], ignore_index=True),
                             geometry="geometry", crs=epsg)
    write_geojson(roads.to_crs(4326), Path(args.roads_output))
    print(f"wrote {args.roads_output}")

    align_barrier = unary_union(align.geometry.values)

    cand = find_candidates(align, nic, align_barrier)
    if cand.empty:
        print("No crossings found.")
        return
    print(f"\nstage 4: {len(cand)} crossing candidate(s) before R5 merge")
    cand = merge_dual_carriageways(cand, args.dual_merge_tol_m)
    print(f"stage 4: {len(cand)} crossing(s) after R5 dual-carriageway merge")
    if len(osm):
        extra = find_candidates(align, osm, align_barrier)
        extra = merge_dual_carriageways(extra, args.dual_merge_tol_m) if len(extra) else extra
        extra = drop_known_crossings(extra, cand, args.osm_buffer_m, args.osm_angle_deg)
        print(f"stage 4: + {len(extra)} OSM crossing(s) with no NIC crossing within {args.osm_buffer_m:g} m")
        cand = gpd.GeoDataFrame(pd.concat([cand, extra], ignore_index=True), geometry="geometry", crs=cand.crs)

    G = build_barrier_graph(roads, align, align_barrier)
    cand = apply_connectivity_rule(cand, G, args.junction_flag_m, args.skew_flag_deg)
    cand = apply_identification_filters(cand, args)
    cand = init_width_columns(cand)

    # NIC crossings first, numbered as they always were; OSM crossings after them - so adding OSM never
    # renumbers a NIC crossing, and measure_width_sam3.py --resume can reuse every width already measured
    cand["_osm"] = cand["road_source"] == "osm"
    cand = cand.sort_values(["_osm", "align_id", "chainage_m"]).drop(columns="_osm").reset_index(drop=True)
    cand.insert(0, "crossing_id", [f"{args.id_prefix}-{i + 1:03d}" for i in range(len(cand))])
    cand = flag_nearby_crossings(cand, args.nearby_crossing_flag_m)  # needs metres, before reprojecting to WGS84
    cand_wgs = cand.to_crs(4326)

    write_geojson(cand_wgs, Path(args.output))
    write_csv(cand_wgs.drop(columns="geometry"), Path(args.summary_csv))
    remember_region(args.region)  # the later steps default to this region

    print(f"\nwrote {args.output}")
    print(f"wrote {args.summary_csv}")
    print("\nDecision counts:")
    print(cand_wgs["decision"].value_counts().to_string())
    print("\nBy IRC class:")
    print(cand_wgs.groupby(["irc_class", "decision"]).size().to_string())
    print(f"\nAll design widths/spans are pending SAM3 measurement - run next:\n"
          f"  python src/measure_width_sam3.py --region {args.region}")
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
