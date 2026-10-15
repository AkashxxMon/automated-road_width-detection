"""Stage 7: decide the structure at every crossing - box culvert, bridge,
re-route or nothing - from the measured road width, the crossing angle and
the road class, then size it: a box snaps to a standard RDSO size, a bridge
gets its span along the track. Pure rules, no imagery or model, so it can
be rerun as often as the thresholds or the box table change without
re-measuring anything.

First, R9 (corridor merge): crossings on the same alignment segment whose
roads sit within --corridor-merge-m (20 m) of each other, edge to edge,
are one road corridor - the two carriageways of a divided highway, a
service road beside it, the same road mapped as two lines - so they get
one structure over their combined width. The other crossings in the group
become MERGED, pointing at it (merged_into). The combined layout is the
union of where each one's road actually sits in the imagery (overlapping
measurements count once), or, without imagery edges, its measured width
centred on its crossing point.

After S0-S6 below, R12 (thinning): a minor road within --thin-m (100 m) of a
higher-ranked crossing (higher class, then measured, then wider) is diverted
to it - MERGED, merged_into pointing at it.

Then, per structure (the first match sets structure_type; flags add up and
never change it):
  S0  decision = no_crossing (R2)                         -> NONE
  S1  minor road (irc_class not in --major-classes),      -> NIL_REROUTE, no structure
      --minor-rule spacing (default): narrower than
      --minor-keep-width-m (4 m, combined width) or within
      --minor-divert-m (500 m) of a major-road crossing
      (nearest_major_crossing_m) - it's diverted there -
      or with no measured width. Otherwise kept
      (road_category MINOR_KEPT) and sized by S3-S6 like
      a major road, with --minor-clearance-m.
      --minor-rule surface (the earlier rule): re-routed
      if unpaved; paved if EITHER SAM3 (surface_sam3) or
      the road data (surface) says so -> kept
      (MINOR_PAVED); unknown -> kept, flag SURFACE_UNKNOWN
  S2  kept road, no measured width                        -> PENDING_SURVEY
  S3  crossing angle below --skew-flag-deg (same as R7)   -> flag HIGH_SKEW, span_m left empty
                                                             whatever the structure turns out to be
  S4  more than one road (divided road per R5, or two+    -> DUAL_DESIGNER_DECISION: one box over the
      roads in the combined layout)                          whole width, a split (twin-cell) box and
                                                             a bridge are all worked out;
                                                             structure_suggested = first that fits
  S5  one road, opening fits a standard box               -> BOX, square to the road (no span)
  S6  one road, opening wider than any box                -> BRIDGE, span_m = opening / sin(angle)

opening_width_m is the combined width plus --edge-buffer-m on each side.
Every box is the smallest standard size (--box-sizes) whose span covers the
opening and whose height clears --clearance-m.

Usage:
    python src/structure_decide.py
    python src/structure_decide.py --register my_crossings_final.geojson -o my_structures.geojson
    python src/structure_decide.py --edge-buffer-m 1.5 --box-sizes config/my_box_sizes.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from geo_utils import fill_region_paths, get_utm_epsg, latest, write_csv, write_geojson
from irc_classify import MAJOR_CLASSES
from width_core import (MAX_ROAD_GAP_M, MIN_ROAD_GAP_M, load_tracks, road_layout_text, road_meets_track, track_for,
                        write_width_geometry)

# input/output paths default from the region (geo_utils.region_paths): --region, else the one
# crossing_identify.py last ran for - any path passed explicitly wins
REGION_PATH_ARGS = {"register": "final", "alignment": "alignment", "output": "structures",
                    "csv_output": "structures_csv", "geometry_output": "structure_geometry"}
DEFAULT_BOX_SIZES = "config/box_sizes.csv"

DEFAULT_WALL_M = 0.5  # twin-box intermediate wall, if the box table doesn't give one (RDSO DFC twin: 0.45-0.5 m)
CLASS_RANK = {c: i for i, c in enumerate(("EXP", "NH", "SH", "MDR", "ODR", "VR", "TRK", "PTH"))}

STRUCTURE_COLUMNS = [
    "road_category", "road_surface", "surface_source", "nearest_major_crossing_m", "skew_angle_deg", "merged_into", "merged_ids", "n_roads", "combined_layout_m",
    "combined_width_m", "opening_width_m", "structure_type", "structure_suggested", "structure_rule",
    "box_size", "box_split_size", "span_m", "span_each_m", "span_centre_chainage_m", "clearance_m",
    "structure_flags", "structure_reason",
]


def load_box_sizes(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        raise SystemExit(f"{path} not found - the standard box size table (cells, span_m, height_m[, intermediate_wall_m])")
    boxes = pd.read_csv(path)
    missing = {"cells", "span_m", "height_m"} - set(boxes.columns)
    if missing:
        raise SystemExit(f"{path} is missing column(s) {sorted(missing)}")
    if "intermediate_wall_m" not in boxes.columns:
        boxes["intermediate_wall_m"] = None
    return boxes.sort_values(["cells", "span_m", "height_m"]).reset_index(drop=True)


def box_label(box) -> str:
    return f"{int(box['cells'])} x {box['span_m']:g} x {box['height_m']:g}"


def pick_box(boxes: pd.DataFrame, opening_m: float, clearance_m: float):
    """Smallest standard single-cell box whose span covers opening_m and whose
    height clears clearance_m, or None. Sorted by span first, so a wider
    opening never trades span for extra height."""
    fit = boxes[(boxes["cells"] == 1) & (boxes["span_m"] >= opening_m - 1e-9)
                & (boxes["height_m"] >= clearance_m - 1e-9)]
    return None if fit.empty else fit.iloc[0]


def pick_split_box(boxes: pd.DataFrame, cw1_m: float, median_m: float, cw2_m: float,
                   buffer_m: float, clearance_m: float):
    """Smallest standard twin-cell box that takes one carriageway per cell with
    the intermediate wall centred in the median. Each cell has to hold the
    edge buffer, its carriageway and its half of the median beside the wall
    - at least one edge buffer there too, so a median narrower than
    (2 x buffer + wall) is widened locally through the box rather than
    squeezing the buffer. Returns (box, cell_span_needed_m); box is None if
    no standard twin fits."""
    need = None
    for _, box in boxes[(boxes["cells"] == 2) & (boxes["height_m"] >= clearance_m - 1e-9)].iterrows():
        wall = box["intermediate_wall_m"] if pd.notna(box["intermediate_wall_m"]) else DEFAULT_WALL_M
        need = max(cw1_m, cw2_m) + buffer_m + max(buffer_m, (median_m - wall) / 2)
        if box["span_m"] >= need - 1e-9:
            return box, need
    if need is None:  # no twin in the table clears the height at all
        need = max(cw1_m, cw2_m) + buffer_m + max(buffer_m, (median_m - DEFAULT_WALL_M) / 2)
    return None, need


def along_track(opening_m: float, angle_deg: float) -> float:
    return round(opening_m / math.sin(math.radians(max(angle_deg, 1e-3))), 2)


def _is_true(v) -> bool:
    return bool(v) if v is not None and pd.notna(v) else False


def _num(v) -> float | None:
    return float(v) if v is not None and pd.notna(v) else None


def _xy(row, prefix: str) -> tuple[np.ndarray, np.ndarray] | None:
    cols = [f"{prefix}_p1_x", f"{prefix}_p1_y", f"{prefix}_p2_x", f"{prefix}_p2_y"]
    vals = [_num(row.get(c)) for c in cols]
    if None in vals:
        return None
    return np.array(vals[:2]), np.array(vals[2:])


# ---------------------------------------------------------------- R9: corridor merge

def own_layout(row) -> tuple[list[float], list[float]] | None:
    """(road widths, gaps between them) at this crossing, one side to the
    other: the imagery's combined-road layout if it found more than one road,
    else hand-filled carriageways from a recheck, else the single measured
    width. None if there's no width at all."""
    if _num(row.get("measured_width_m")) is None:
        return None
    text = row.get("road_layout_m")
    if isinstance(text, str) and text.strip():
        vals = [float(v) for v in text.split("|")]
        return vals[0::2], vals[1::2]
    cw1, med, cw2 = (_num(row.get(c)) for c in ("cw1_width_m", "median_width_m", "cw2_width_m"))
    if None not in (cw1, med, cw2):
        return [cw1, cw2], [med]
    return [float(row["measured_width_m"])], []


def footprint(row, base_m: float, track_dir: np.ndarray | None) -> tuple[list[tuple[float, float]], bool]:
    """Where this crossing's road(s) sit across the road direction, as
    [(lo, hi), ...] in metres: base_m is the crossing point's own position on
    that axis, increasing with chainage along track_dir. Placed from the
    imagery's road edges when the measurement has them, else centred on the
    crossing point. Returns (intervals, placed_from_imagery)."""
    layout = own_layout(row)
    if layout is None:
        return [(base_m, base_m)], False
    widths, gaps = layout
    p = np.array([row.geometry.x, row.geometry.y])
    edges = _xy(row, "ext") or (_xy(row, "width") if len(widths) == 1 else None)
    if edges is not None and track_dir is not None:
        e1, e2 = edges
        n = (e2 - e1) / (np.linalg.norm(e2 - e1) or 1.0)  # transect direction = road normal
        if n @ track_dir < 0:
            n = -n
        a, b = float((e1 - p) @ n), float((e2 - p) @ n)
        if a > b:  # the layout reads e1 -> e2; flip it to run with chainage
            widths, gaps, a = widths[::-1], gaps[::-1], b
        cursor = base_m + a
        placed = True
    else:
        cursor = base_m - (sum(widths) + sum(gaps)) / 2
        placed = False
    out = []
    for i, w in enumerate(widths):
        out.append((cursor, cursor + w))
        cursor += w + (gaps[i] if i < len(gaps) else 0.0)
    return out, placed


def _track_dir(rows) -> np.ndarray | None:
    a, b = rows[0].geometry, rows[-1].geometry
    d = np.array([b.x - a.x, b.y - a.y])
    norm = np.linalg.norm(d)
    return d / norm if norm > 1e-6 else None


def union_layout(intervals: list[tuple[float, float]]) -> tuple[list[float], list[float], float, float]:
    """Merge overlapping road intervals (and any split by less than
    MIN_ROAD_GAP_M) into one side-to-side layout: (widths, gaps, lo, hi)."""
    merged = []
    for lo, hi in sorted(intervals):
        if merged and lo - merged[-1][1] < MIN_ROAD_GAP_M:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    widths = [hi - lo for lo, hi in merged]
    gaps = [merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1)]
    return widths, gaps, merged[0][0], merged[-1][1]


def corridor_groups(cross: gpd.GeoDataFrame, args) -> list[list]:
    """R9: runs of consecutive crossings (by chainage, same alignment segment,
    roughly parallel roads) whose road footprints are within
    --corridor-merge-m of each other, edge to edge. Every eligible crossing
    lands in exactly one group, most in a group of one."""
    eligible = cross[cross["decision"] != "no_crossing"]
    groups = []
    for _, seg in eligible.groupby("align_id"):
        rows = [r for _, r in seg.sort_values("chainage_m").iterrows()]
        current = [rows[0]]
        for a, b in zip(rows, rows[1:]):
            linked = False
            if abs(a["angle_deg"] - b["angle_deg"]) <= args.corridor_angle_deg:
                sin_t = math.sin(math.radians((a["angle_deg"] + b["angle_deg"]) / 2))
                t = _track_dir([a, b])
                fa, _ = footprint(a, 0.0, t)
                fb, _ = footprint(b, (b["chainage_m"] - a["chainage_m"]) * sin_t, t)
                lo_a, hi_a = min(v[0] for v in fa), max(v[1] for v in fa)
                lo_b, hi_b = min(v[0] for v in fb), max(v[1] for v in fb)
                linked = max(0.0, lo_b - hi_a, lo_a - hi_b) <= args.corridor_merge_m
            if linked:
                current.append(b)
            else:
                groups.append(current)
                current = [b]
        groups.append(current)
    return groups


# ---------------------------------------------------------------- S0-S6

def road_surface(row) -> tuple[str, str]:
    """(paved / unpaved / unknown, where from). A paved road can't be
    re-routed, so evidence of paved from EITHER source wins: SAM3's imagery
    verdict or the road data's metalled flag (when both say paved the source
    is 'imagery+road data'). Otherwise unpaved if either says so, else unknown."""
    s3, nic = row.get("surface_sam3"), row.get("surface")
    img = s3 if s3 in ("paved", "unpaved") else None
    dat = nic if nic in ("paved", "unpaved") else None
    if "paved" in (img, dat):
        return "paved", " + ".join(n for n, v in (("imagery", img), ("road data", dat)) if v == "paved")
    if img or dat:
        return "unpaved", " + ".join(n for n, v in (("imagery", img), ("road data", dat)) if v == "unpaved")
    return "unknown", "none"


def decide(rep, layout, members: list, boxes: pd.DataFrame, args) -> dict:
    """Structure for one road corridor: rep is its representative crossing
    (highest road class), layout its (widths, gaps) or None, members every
    crossing in it (rep included)."""
    out = dict.fromkeys(STRUCTURE_COLUMNS)
    angle = float(rep["angle_deg"])
    out["skew_angle_deg"] = round(90.0 - angle, 1)  # org convention: |90 - skew number|, 0 = square
    major = rep["irc_class"] in args.major_classes
    # a paved minor road is kept like a major one - it can't be assumed reroutable; so is one whose
    # surface nobody knows, unless told otherwise. Any paved road in a merged corridor keeps it all
    surfaces = [road_surface(m) for m in members]
    surface, source = next((x for x in surfaces if x[0] == "paved"), None) or \
        next((x for x in surfaces if x[0] == "unpaved"), ("unknown", "none"))
    out["road_surface"], out["surface_source"] = surface, source
    widths, gaps = layout if layout is not None else ([], [])
    whole = sum(widths) + sum(gaps) if widths else None
    d_major = _num(rep.get("nearest_major_crossing_m"))
    out["nearest_major_crossing_m"] = d_major
    if args.minor_rule == "spacing":
        # the designers keep a village road that is wide enough and has no major-road structure close by to
        # divert it to (validation: 24% built within 500 m of a major-road crossing, 66% beyond 1 km)
        near_major = d_major is not None and d_major <= args.minor_divert_m
        narrow = whole is None or whole < args.minor_keep_width_m  # unmeasured: 17% built, so re-routed
        spaced = not major and not near_major and not narrow
        paved = unknown_kept = False
    else:
        spaced = near_major = narrow = False
        paved = args.keep_paved_minor and surface == "paved"
        unknown_kept = args.keep_paved_minor and surface == "unknown" and args.surface_known and \
            args.unknown_surface == "keep"
    out["road_category"] = "MAJOR" if major else "MINOR_KEPT" if spaced else "MINOR_PAVED" if paved else \
        "MINOR_SURFACE_UNKNOWN" if unknown_kept else "MINOR"
    kept = major or spaced or paved or unknown_kept
    ids = {m["crossing_id"] for m in members}
    if len(members) > 1:
        out["merged_ids"] = ";".join(sorted(ids - {rep["crossing_id"]}))
    flags = []

    def finish(structure_type: str, rule: str, reason: str) -> dict:
        out.update(structure_type=structure_type, structure_rule=rule, structure_reason=reason,
                   structure_flags=";".join(flags) or None)
        return out

    if rep["decision"] == "no_crossing":
        return finish("NONE", "S0", f"no structure needed ({rep['rule']}: {rep['reason']})")

    if widths:
        out.update(n_roads=len(widths), combined_layout_m=road_layout_text(widths, gaps), combined_width_m=round(whole, 2))
    merged_note = (f"{len(members)} crossings ({', '.join(sorted(ids))}) within {args.corridor_merge_m:g} m of each "
                   f"other, one structure (R9); " if len(members) > 1 else "")

    if not kept:
        if (whole is not None and whole > args.class_check_width_m) or len(widths) > 1:
            flags.append("CLASS_CHECK")
        if args.minor_rule == "spacing":
            why = " and ".join(x for x in (
                ("no measured width" if whole is None else f"only {whole:.1f} m wide (under {args.minor_keep_width_m:g} m)")
                if narrow else "",
                f"a major-road crossing is {d_major:.0f} m away" if near_major else "") if x)
            return finish("NIL_REROUTE", "S1", f"{merged_note}{rep['irc_class']} is a minor road, {why} - "
                                               f"re-route it to the nearest structure")
        known = f", {surface} ({source})" if source != "none" else ""
        return finish("NIL_REROUTE", "S1", f"{merged_note}{rep['irc_class']} is a minor road{known} - "
                                           f"re-route it to the nearest structure")

    if len(members) > 1:
        flags.append("CORRIDOR_MERGED")
    if any("SUSPECT" in str(m.get("width_source")) for m in members):
        flags.append("SUSPECT_WIDTH")
    if any(m["decision"] == "review" for m in members):
        flags.append("CONNECTIVITY_REVIEW")
    if any(_is_true(m.get("near_junction")) for m in members):
        flags.append("NEAR_JUNCTION")
    if any(pd.notna(m.get("nearby_crossing_id")) and m["nearby_crossing_id"] not in ids for m in members):
        flags.append("NEARBY_CROSSING")  # R8 pair not already folded in by R9
    high_skew = angle < args.skew_flag_deg
    if high_skew:
        flags.append("HIGH_SKEW")
    dual_r5 = any(_is_true(m.get("dual_carriageway")) for m in members)
    live = any(pd.notna(m.get("n_transects")) for m in members)  # an imagery pass ran and could have seen it
    if widths and live and dual_r5 != (len(widths) > 1):
        flags.append("DUAL_MISMATCH")
    skew_note = (f"; span left empty - crossing angle {angle:g} deg is below {args.skew_flag_deg:g} deg "
                 f"(realign the road, or the designer sets the span)") if high_skew else ""

    clearance = args.clearance_m if major else args.minor_clearance_m
    out["clearance_m"] = clearance
    if spaced:
        far = f"{d_major:.0f} m" if d_major is not None else "no major-road crossing in this region"
        merged_note += f"{rep['irc_class']} is {whole:.1f} m wide, nearest major-road crossing {far} - kept (S1 spacing); "
    elif paved and not major:
        merged_note += f"{rep['irc_class']} is paved ({source}) - kept, not re-routed; "
    elif unknown_kept and not major:
        flags.append("SURFACE_UNKNOWN")
        merged_note += f"{rep['irc_class']} surface unknown (no imagery verdict, not in the road data) - kept to be safe; "
    if not widths:
        return finish("PENDING_SURVEY", "S2", f"{merged_note}road kept, but no usable width yet - survey it")

    b = args.edge_buffer_m
    opening = whole + 2 * b
    out["opening_width_m"] = round(opening, 2)

    if len(widths) > 1 or dual_r5:
        source = ("R5 same-ref merge" if dual_r5 else "") + (" + " if dual_r5 and len(widths) > 1 else "") + \
                 (f"{len(widths)} roads in the combined layout" if len(widths) > 1 else "")
        parts = [f"{merged_note}divided/multiple road ({source}), whole width {whole:.2f} m -> opening {opening:.2f} m"]

        single = pick_box(boxes, opening, clearance)
        if single is not None:
            out["box_size"] = box_label(single)
        parts.append(f"one box over the whole width: {out['box_size'] or 'wider than any standard box'}")

        twin = None
        if len(widths) == 2:
            twin, need = pick_split_box(boxes, widths[0], gaps[0], widths[1], b, clearance)
            if twin is not None:
                out["box_split_size"] = box_label(twin)
            parts.append(f"split box (one cell per carriageway, {road_layout_text(widths, gaps)} m, "
                         f"cell needs {need:.2f} m): {out['box_split_size'] or 'no standard twin box that wide'}")
        elif len(widths) > 2:
            parts.append(f"split box not evaluated - {len(widths)} roads, the standard twin box has 2 cells")
        else:
            parts.append("split box not evaluated - carriageways not separated in the imagery "
                         "(fill cw1/median/cw2 in a --measured-csv recheck)")

        if not high_skew:
            out["span_m"] = along_track(opening, angle)
            parts.append(f"bridge span along track {out['span_m']} m")
            if len(widths) > 1:
                each = [along_track(w + 2 * b, angle) for w in widths]
                out["span_each_m"] = " | ".join(f"{s:g}" for s in each)
                parts[-1] += f" (or {out['span_each_m']} m as a separate structure per road)"
        out["structure_suggested"] = "BOX" if single is not None else "BOX_SPLIT" if twin is not None else "BRIDGE"
        return finish("DUAL_DESIGNER_DECISION", "S4", "; ".join(parts) + skew_note)

    measured = widths[0]
    opening_note = f"{merged_note}opening {opening:.2f} m (measured {measured:.2f} + 2 x {b:g} m buffer)"
    fits_height = boxes[(boxes["cells"] == 1) & (boxes["height_m"] >= clearance - 1e-9)]
    max_span = fits_height["span_m"].max() if not fits_height.empty else None
    iqr = max((_num(m.get("width_iqr_m")) or 0.0) for m in members)
    if max_span is not None and abs(opening - max_span) <= max(args.borderline_m, iqr):
        flags.append("BORDERLINE")

    box = pick_box(boxes, opening, clearance)
    if box is not None:
        out["box_size"] = box_label(box)
        squared = "; crossing is skewed - the road is squared up to the box" if high_skew else ""
        return finish("BOX", "S5", f"{opening_note} fits standard box {out['box_size']} "
                                   f"(clear height >= {clearance:g} m); a box is square to the road, so no span{squared}")

    if not high_skew:
        out["span_m"] = along_track(opening, angle)
    too_wide = (f"wider than the largest standard box span ({max_span:g} m)" if max_span is not None
                else f"no standard box clears {clearance:g} m")
    span_note = f"; span along track {out['span_m']} m at {angle:g} deg" if out["span_m"] is not None else skew_note
    return finish("BRIDGE", "S6", f"{opening_note} is {too_wide}{span_note}")


def road_centre_chainage(row, tracks: dict) -> float | None:
    """Chainage where the measured road (or, for several roads, the middle of
    their combined extent) actually meets the track - the crossing point
    itself is only where the road data's line meets it."""
    track = track_for(tracks, row)
    edges = _xy(row, "ext") or _xy(row, "width")
    if track is None or edges is None:
        return None
    hit = road_meets_track(tuple(edges[0]), tuple(edges[1]), track)
    return hit[2] if hit else None


def decide_all(cross: gpd.GeoDataFrame, boxes: pd.DataFrame, tracks: dict, args) -> pd.DataFrame:
    """Structure columns for every row of cross (in the metric CRS)."""
    major_pts = cross.geometry[cross["irc_class"].isin(args.major_classes) & (cross["decision"] != "no_crossing")]
    cross["nearest_major_crossing_m"] = (cross.geometry.apply(lambda p: round(float(major_pts.distance(p).min()), 1))
                                         if len(major_pts) else np.nan)
    decided = {}
    for _, r in cross[cross["decision"] == "no_crossing"].iterrows():
        decided[r.name] = decide(r, None, [r], boxes, args)

    for members in corridor_groups(cross, args):
        rep = min(members, key=lambda m: (CLASS_RANK.get(m["irc_class"], 99),
                                          abs(m["chainage_m"] - np.median([x["chainage_m"] for x in members]))))
        if len(members) == 1:
            out = decide(rep, own_layout(rep), members, boxes, args)
            centre = road_centre_chainage(rep, tracks)
            if centre is not None:
                out["span_centre_chainage_m"] = round(centre, 1)
                if out["road_category"] != "MINOR" and abs(centre - rep["chainage_m"]) > args.road_offset_flag_m:
                    out["structure_flags"] = ";".join(
                        [f for f in (out["structure_flags"] or "").split(";") if f] + ["ROAD_OFFSET"])
                    out["structure_reason"] += (f"; the measured road meets the track {centre - rep['chainage_m']:+.1f} m "
                                                f"from the road-data crossing point - check the QA crop")
            decided[rep.name] = out
            continue
        sin_t = math.sin(math.radians(rep["angle_deg"]))
        t = _track_dir(members)
        ch0 = members[0]["chainage_m"]
        intervals, all_placed = [], True
        for m in members:
            if own_layout(m) is None:
                continue  # an unmeasured crossing adds no width - it's still part of the group
            f, placed = footprint(m, (m["chainage_m"] - ch0) * sin_t, t)
            intervals += f
            all_placed &= placed
        layout = None
        centre = rep["chainage_m"]
        if intervals:
            widths, gaps, lo, hi = union_layout(intervals)
            layout = (widths, gaps)
            centre = ch0 + (lo + hi) / 2 / sin_t
        out = decide(rep, layout, members, boxes, args)
        if out["road_category"] != "MINOR" and intervals and not all_placed:
            flags = [f for f in (out["structure_flags"] or "").split(";") if f] + ["LAYOUT_APPROX"]
            out["structure_flags"] = ";".join(flags)
            out["structure_reason"] += ("; layout approximate - some crossings have no imagery road edges, "
                                        "so their width is centred on the crossing point")
        out["span_centre_chainage_m"] = round(centre, 1)
        decided[rep.name] = out
        for m in members:
            if m.name == rep.name:
                continue
            decided[m.name] = {**dict.fromkeys(STRUCTURE_COLUMNS),
                               "road_category": out["road_category"], "skew_angle_deg": round(90.0 - m["angle_deg"], 1),
                               "nearest_major_crossing_m": m["nearest_major_crossing_m"],
                               "merged_into": rep["crossing_id"], "structure_type": "MERGED", "structure_rule": "R9",
                               "structure_reason": f"same road corridor as {rep['crossing_id']} (roads within "
                                                   f"{args.corridor_merge_m:g} m, edge to edge) - one structure, "
                                                   f"see {rep['crossing_id']}"}
    thin_nearby(cross, decided, args)
    return pd.DataFrame.from_dict(decided, orient="index").reindex(cross.index)[STRUCTURE_COLUMNS]


def thin_nearby(cross: gpd.GeoDataFrame, decided: dict, args) -> None:
    """R12: a minor road within --thin-m of a higher-ranked crossing (higher
    road class, then measured, then wider) is diverted to it - MERGED into it.
    The designers built at only 16-23% of crossings within 100 m of another."""
    if args.thin_m <= 0:
        return
    live = [i for i, o in decided.items() if o["structure_type"] not in ("MERGED", "NONE")]
    width = {i: _num(decided[i]["combined_width_m"]) for i in live}
    live.sort(key=lambda i: (CLASS_RANK.get(cross.at[i, "irc_class"], 99), width[i] is None, -(width[i] or 0)))
    kept = []
    for i in live:
        pt = cross.geometry.at[i]
        near = [(pt.distance(cross.geometry.at[k]), k) for k in kept]
        near = [x for x in near if x[0] <= args.thin_m]
        if cross.at[i, "irc_class"] in args.major_classes or not near:
            kept.append(i)
            continue
        d, k = min(near)
        rep = cross.at[k, "crossing_id"]
        decided[i] = {**dict.fromkeys(STRUCTURE_COLUMNS), "road_category": decided[i]["road_category"],
                      "road_surface": decided[i]["road_surface"], "surface_source": decided[i]["surface_source"],
                      "skew_angle_deg": decided[i]["skew_angle_deg"],
                      "nearest_major_crossing_m": decided[i]["nearest_major_crossing_m"],
                      "merged_into": rep, "structure_type": "MERGED", "structure_rule": "R12",
                      "structure_reason": f"minor road {d:.0f} m from {rep} ({cross.at[k, 'irc_class']}) - "
                                          f"diverted to its crossing (--thin-m {args.thin_m:g})"}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", default=None,
                     help="Region whose files to use (e.g. maharashtra). Default: the region crossing_identify.py "
                          "last ran for. Any path below passed explicitly overrides it")
    ap.add_argument("--register", default=None,
                     help="Measured crossing register. Default: output/<region>_crossings_final_gl.geojson")
    ap.add_argument("--alignment", default=None,
                     help="Clipped alignment GeoJSON - used to draw the span lines. Default: shared/alignment_<region>.geojson")
    ap.add_argument("--box-sizes", default=DEFAULT_BOX_SIZES,
                     help="Standard box size table: cells, span_m, height_m[, intermediate_wall_m] (see config/box_sizes.csv)")
    ap.add_argument("--major-classes", default=",".join(MAJOR_CLASSES),
                     help="IRC classes that get a structure; every other class is re-routed")
    ap.add_argument("--corridor-merge-m", type=float, default=MAX_ROAD_GAP_M,
                     help="R9: crossings whose roads are within this of each other (edge to edge) share one structure")
    ap.add_argument("--corridor-angle-deg", type=float, default=15.0,
                     help="R9: only merge roads whose crossing angles differ by at most this (i.e. roughly parallel)")
    ap.add_argument("--thin-m", type=float, default=100.0,
                     help="R12: a minor road this close to a higher-ranked crossing is diverted to it (MERGED). 0 to disable")
    ap.add_argument("--edge-buffer-m", type=float, default=1.0, help="Added to the combined width on each side")
    ap.add_argument("--skew-flag-deg", type=float, default=45.0,
                     help="S3: below this crossing angle, flag HIGH_SKEW and leave span_m empty (same threshold as R7)")
    ap.add_argument("--clearance-m", type=float, default=5.5,
                     help="Vertical clearance a box must give a major road (org bridge-list template: major road 5.5 m)")
    ap.add_argument("--minor-clearance-m", type=float, default=4.0,
                     help="Vertical clearance for a kept (paved) minor road (org bridge-list template: minor road 4.0 m)")
    ap.add_argument("--minor-rule", choices=("spacing", "surface"), default="spacing",
                     help="S1: how a minor road is kept or re-routed. spacing (default): kept if at least "
                          "--minor-keep-width-m wide and more than --minor-divert-m from a major-road crossing. "
                          "surface: kept if paved (the earlier rule; see --keep-paved-minor, --unknown-surface)")
    ap.add_argument("--minor-keep-width-m", type=float, default=4.0,
                     help="S1 spacing: a minor road narrower than this (combined width) is re-routed")
    ap.add_argument("--minor-divert-m", type=float, default=500.0,
                     help="S1 spacing: a minor road this close to a major-road crossing is diverted to it")
    ap.add_argument("--keep-paved-minor", action=argparse.BooleanOptionalAction, default=True,
                     help="S1 surface rule only: keep a paved minor road (imagery verdict, else road data)")
    ap.add_argument("--unknown-surface", choices=("keep", "reroute"), default="keep",
                     help="S1 surface rule only: a minor road whose surface neither the imagery nor the road data settles")
    ap.add_argument("--borderline-m", type=float, default=0.5,
                     help="Flag BORDERLINE when the opening is this close to the largest box span (or the transect IQR, if wider)")
    ap.add_argument("--road-offset-flag-m", type=float, default=10.0,
                     help="Flag ROAD_OFFSET when the measured road meets the track this far from the crossing point")
    ap.add_argument("--class-check-width-m", type=float, default=7.0,
                     help="Flag CLASS_CHECK on a minor road measured wider than this (maybe mis-classed in the road data)")
    ap.add_argument("--epsg", type=int, default=None, help="Metric CRS of the width-edge coordinates. Default: auto-detect UTM zone")
    ap.add_argument("-o", "--output", default=None, help="Output structure register GeoJSON. Default: output/<region>_structures.geojson")
    ap.add_argument("--csv-output", default=None, help="Output structure register CSV")
    ap.add_argument("--geometry-output", default=None,
                     help="Output GeoJSON: measured road width lines (and combined extent where several roads), drawn to scale")
    args = ap.parse_args()
    args.major_classes = {c.strip().upper() for c in args.major_classes.split(",") if c.strip()}
    fill_region_paths(args, REGION_PATH_ARGS)
    args.alignment = str(latest(args.alignment))
    return args


def main() -> None:
    args = parse_args()
    register = latest(args.register)
    if not register.exists():
        raise SystemExit(f"{args.register} not found - run src/measure_width_sam3.py first "
                         f"(or pass --region / --register for the region you measured)")
    print(f"register: {register}")
    cross = gpd.read_file(register)
    boxes = load_box_sizes(args.box_sizes)
    # a register with no surface information at all (made before surface was recorded) keeps the old
    # behaviour - every minor road re-routed - rather than keeping all of them as "unknown"
    args.surface_known = bool({"surface", "surface_sam3"} & set(cross.columns))
    if not args.surface_known:
        print("  [note] no road surface in this register - minor roads re-routed as before; re-run "
              "crossing_identify.py / measure_width_sam3.py to get it")
    centroid = cross.geometry.union_all().centroid
    epsg = args.epsg or get_utm_epsg(centroid.x, centroid.y)

    cross_utm = cross.to_crs(epsg)
    tracks = load_tracks(args.alignment, epsg)
    decided = decide_all(cross_utm, boxes, tracks, args)
    cross = cross.drop(columns=STRUCTURE_COLUMNS, errors="ignore")
    cross = gpd.GeoDataFrame(pd.concat([cross, decided], axis=1), geometry="geometry", crs=cross.crs)
    # the same values under "Assumed_" names (imagery and rules, not a survey) - the map layer copies these
    for col in ("irc_class", "structure_type", "box_size", "opening_width_m", "span_m"):
        cross["Assumed_" + col] = cross[col]

    write_geojson(cross, Path(args.output))
    write_csv(cross.drop(columns="geometry"), Path(args.csv_output))
    print(f"wrote {args.output}")
    print(f"wrote {args.csv_output}")
    write_width_geometry(cross, epsg, args.alignment, args.geometry_output)

    print("\nStructure type:")
    print(cross["structure_type"].value_counts().to_string())
    print("\nBy IRC class:")
    print(cross.groupby(["irc_class", "structure_type"]).size().to_string())
    merged = cross[cross["merged_ids"].notna()]
    if len(merged):
        print(f"\nR9 corridor merges (roads within {args.corridor_merge_m:g} m of each other - one structure each):")
        for _, r in merged.iterrows():
            print(f"  {r['crossing_id']} + {r['merged_ids']}: {r['combined_layout_m']} m -> "
                  f"{r['structure_type']}" + (f" (suggested {r['structure_suggested']})" if pd.notna(r['structure_suggested']) else ""))
    boxes_used = cross.loc[cross["structure_type"] == "BOX", "box_size"].value_counts()
    if len(boxes_used):
        print("\nBox sizes:")
        print(boxes_used.to_string())
    flags = cross["structure_flags"].dropna().str.split(";").explode().value_counts()
    if len(flags):
        print("\nFlags (major roads, plus CLASS_CHECK on minor ones):")
        print(flags.to_string())



if __name__ == "__main__":
    main()
