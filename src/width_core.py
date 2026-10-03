"""Shared logic between the SAM2 (src/measure_width.py) and SAM3
(src/measure_width_sam3.py) width-measurement pipelines: transect
measurement, the QA overlay image, and the file-lock-safe CSV writer.
Model loading and prompting differ per model, so those stay in each
model-specific script.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from shapely.geometry import LineString
from shapely.ops import substring

from geo_utils import write_geojson

TRANSECT_OFFSETS_M = (-20, -10, 0, 10, 20)
MAX_PLAUSIBLE_WIDTH_M = 60.0  # wider than that is almost certainly the mask leaking into a field


def bearing_deg(p0, p1) -> float:
    return math.degrees(math.atan2(p1[0] - p0[0], p1[1] - p0[1]))


def road_tangent_bearing(roads_sindex, roads: gpd.GeoDataFrame, pt_utm) -> float | None:
    near = list(roads_sindex.query(pt_utm.buffer(30)))
    if not near:
        return None
    geom = min((roads.iloc[i].geometry for i in near), key=lambda g: g.distance(pt_utm))
    d = geom.project(pt_utm)
    p = geom.interpolate(max(d - 10, 0))
    q = geom.interpolate(min(d + 10, geom.length))
    if p.equals(q):
        return None
    return bearing_deg((p.x, p.y), (q.x, q.y))


MAX_SNAP_M = 25.0  # how far off-road the OSM-derived crossing point is allowed to be before we give up
                   # finding the real road in the mask - rural OSM traces are commonly this far off
                   # from the true (imagery-visible) road position, this is not a bug, just data noise

# combined-road reading: every other road run on the same transect within MAX_ROAD_GAP_M of the
# measured road (edge to edge) is part of one road corridor - the other carriageway of a divided road,
# a service road - and the structure spans all of it. Matches measure_width_sam3.MERGE_NEARBY_M (the
# widest gap whose far road gets unioned into the mask at all) and structure_decide.py's R9
MAX_ROAD_GAP_M = 20.0
MIN_ROAD_GAP_M = 1.0   # a narrower gap is lane-marking/shadow speckle inside one road, not a median
MIN_ROAD_RUN_M = 2.5   # a narrower run is a kerb/footpath/verge sliver - counted as gap, not as a road


def _nearest_mask_pixel(mask: np.ndarray, cx: float, cy: float, max_radius_px: int) -> tuple[int, int] | None:
    """Nearest True pixel to (cx, cy) within max_radius_px, searched in every
    direction (not just vertically) - handles the road sitting anywhere near
    the expected point, not just above/below it."""
    h, w = mask.shape
    cxi, cyi = int(round(cx)), int(round(cy))
    if 0 <= cyi < h and 0 <= cxi < w and mask[cyi, cxi]:
        return cxi, cyi
    y0, y1 = max(cyi - max_radius_px, 0), min(cyi + max_radius_px + 1, h)
    x0, x1 = max(cxi - max_radius_px, 0), min(cxi + max_radius_px + 1, w)
    if y1 <= y0 or x1 <= x0:
        return None
    sub = mask[y0:y1, x0:x1]
    ys, xs = np.nonzero(sub)
    if len(xs) == 0:
        return None
    ys, xs = ys + y0, xs + x0
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    idx = np.argmin(d2)
    return int(xs[idx]), int(ys[idx])


def _outward_runs(mask: np.ndarray, cx: float, cy: float, dx: float, dy: float, edge: int,
                  max_gap_px: int, min_run_px: int) -> list[tuple[int, int]]:
    """Past the run edge at `edge` steps from (cx, cy) along (dx, -dy): every
    further road run reached across a gap of at most max_gap_px, outward,
    as (gap_px, run_px). Runs under min_run_px count as gap. Stops at the
    first gap too wide, or at the crop edge (a run cut by it is dropped -
    its width is unknown)."""
    h, w = mask.shape

    def on(step):
        px = int(round(cx + step * dx))
        py = int(round(cy - step * dy))
        if not (0 <= px < w and 0 <= py < h):
            return None
        return bool(mask[py, px])

    runs = []
    step, gap = edge + 1, 0
    while True:
        v = on(step)
        if v is None:
            return runs
        if not v:
            gap += 1
            if gap > max_gap_px:
                return runs
            step += 1
            continue
        start = step
        while True:
            nxt = on(step + 1)
            if nxt is None:
                return runs
            if not nxt:
                break
            step += 1
        run = step - start + 1
        if run < min_run_px:
            gap += run
        else:
            runs.append((gap, run))
            gap = 0
        step += 1


def measure_transects_detailed(mask: np.ndarray, mpp: float, bearing: float, center_px: tuple[float, float],
                                max_snap_m: float = MAX_SNAP_M) -> list[dict]:
    """One entry per successful transect (offsets that fell off the mask/crop
    are omitted): {"offset_m", "width_m", "p1_px", "p2_px", "far"} - p1_px/p2_px
    are the two road-edge pixels the width was measured between, so a caller
    can turn the measurement into real geometry (a line, or the two edge
    points) instead of just a number. "far" holds the further road runs
    past each edge within MAX_ROAD_GAP_M, outward ({"p1": [...], "p2": [...]},
    each run {"gap_m", "width_m", "edge_px"} with edge_px its outer edge) -
    the other carriageway of a divided road, a service road; see
    combine_roads()."""
    h, w = mask.shape
    brg = math.radians(bearing)
    perp = brg + math.pi / 2
    dx_perp, dy_perp = math.sin(perp), math.cos(perp)  # bearing: 0=north(+y up in world, -y in image)
    dx_along, dy_along = math.sin(brg), math.cos(brg)
    max_radius_px = max(1, int(round(max_snap_m / mpp)))
    max_gap_px = max(1, int(round(MAX_ROAD_GAP_M / mpp)))
    min_run_px = max(1, int(round(MIN_ROAD_RUN_M / mpp)))

    results = []
    for off in TRANSECT_OFFSETS_M:
        cx = center_px[0] + (off * dx_along) / mpp
        cy = center_px[1] - (off * dy_along) / mpp
        snapped = _nearest_mask_pixel(mask, cx, cy, max_radius_px)
        if snapped is None:
            continue
        cx, cy = snapped

        pos = neg = 0
        step = 0
        while True:
            step += 1
            px = int(round(cx + step * dx_perp))
            py = int(round(cy - step * dy_perp))
            if not (0 <= px < w and 0 <= py < h) or not mask[py, px]:
                break
            pos = step
        step = 0
        while True:
            step += 1
            px = int(round(cx - step * dx_perp))
            py = int(round(cy + step * dy_perp))
            if not (0 <= px < w and 0 <= py < h) or not mask[py, px]:
                break
            neg = step
        width_m = (pos + neg + 1) * mpp
        if width_m <= MAX_PLAUSIBLE_WIDTH_M:
            # edge points sit at the outer half-pixel boundary of the run, not
            # the centre of the outermost True pixel - pos+neg would only span
            # pos+neg pixels (undercounting by 1 vs width_m, which correctly
            # counts pos+neg+1 pixels inclusive of the centre pixel itself)
            p1 = (cx - (neg + 0.5) * dx_perp, cy + (neg + 0.5) * dy_perp)
            p2 = (cx + (pos + 0.5) * dx_perp, cy - (pos + 0.5) * dy_perp)
            far = {}
            for side, sgn, edge in (("p2", 1, pos), ("p1", -1, neg)):
                far[side] = []
                outer = edge
                for gap_px, run_px in _outward_runs(mask, cx, cy, sgn * dx_perp, sgn * dy_perp, edge,
                                                    max_gap_px, min_run_px):
                    outer += gap_px + run_px
                    far[side].append({"gap_m": gap_px * mpp, "width_m": run_px * mpp,
                                      "edge_px": (cx + sgn * (outer + 0.5) * dx_perp,
                                                  cy - sgn * (outer + 0.5) * dy_perp)})
            results.append({"offset_m": off, "width_m": width_m, "p1_px": p1, "p2_px": p2, "far": far})
    return results


def combine_roads(detailed: list[dict]) -> dict | None:
    """The whole road corridor around the measured road: on each transect,
    every road run from the outermost one reached on the p1 side to the
    outermost on the p2 side (gaps up to MAX_ROAD_GAP_M). Uses the transects
    with the most common road count and, among those, the one with the
    median overall width. Returns {"widths_m", "gaps_m", "total_m", "e1_px",
    "e2_px"} - widths/gaps ordered p1 side to p2 side, e1/e2 the two outer
    edges - or None if no other road is within reach."""
    layouts = []
    for d in detailed:
        left, right = d["far"]["p1"], d["far"]["p2"]
        widths = [r["width_m"] for r in reversed(left)] + [d["width_m"]] + [r["width_m"] for r in right]
        gaps = [r["gap_m"] for r in reversed(left)] + [r["gap_m"] for r in right]
        # fold speckle-sized gaps back into one road
        w_out, g_out = [widths[0]], []
        for g, wd in zip(gaps, widths[1:]):
            if g < MIN_ROAD_GAP_M:
                w_out[-1] += g + wd
            else:
                g_out.append(g)
                w_out.append(wd)
        e1 = left[-1]["edge_px"] if left else d["p1_px"]
        e2 = right[-1]["edge_px"] if right else d["p2_px"]
        layouts.append({"widths_m": w_out, "gaps_m": g_out, "total_m": sum(w_out) + sum(g_out),
                        "e1_px": e1, "e2_px": e2})
    if not layouts:
        return None
    counts = [len(lay["widths_m"]) for lay in layouts]
    n = max(set(counts), key=lambda c: (counts.count(c), c))  # most common road count; ties -> more roads
    if n < 2:
        return None
    same = [lay for lay in layouts if len(lay["widths_m"]) == n]
    med = float(np.median([lay["total_m"] for lay in same]))
    return min(same, key=lambda lay: abs(lay["total_m"] - med))


def road_layout_text(widths_m: list[float], gaps_m: list[float]) -> str:
    """"7.50 | 4.50 | 7.30" - road, gap, road, ... from one side to the other."""
    parts = [f"{widths_m[0]:.2f}"]
    for g, w in zip(gaps_m, widths_m[1:]):
        parts += [f"{g:.2f}", f"{w:.2f}"]
    return " | ".join(parts)


def measure_transects(mask: np.ndarray, mpp: float, bearing: float, center_px: tuple[float, float],
                       max_snap_m: float = MAX_SNAP_M) -> list[float]:
    """Back-compat wrapper - just the widths, for callers that don't need
    the edge geometry (measure_width.py)."""
    return [r["width_m"] for r in measure_transects_detailed(mask, mpp, bearing, center_px, max_snap_m)]


def save_qa_image(path: Path, rgb: np.ndarray, mask: np.ndarray | None, bearing: float,
                   mpp: float, center_px: tuple[float, float], widths: list[float],
                   max_snap_m: float = MAX_SNAP_M, extra_label: str | None = None) -> None:
    img = Image.fromarray(rgb).convert("RGBA")
    if mask is not None:
        overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
        overlay[mask] = (255, 30, 30, 90)
        img = Image.alpha_composite(img, Image.fromarray(overlay))
    draw = ImageDraw.Draw(img)
    brg = math.radians(bearing)
    perp = brg + math.pi / 2
    dx_perp, dy_perp = math.sin(perp), math.cos(perp)
    dx_along, dy_along = math.sin(brg), math.cos(brg)
    max_radius_px = max(1, int(round(max_snap_m / mpp)))
    for off in TRANSECT_OFFSETS_M:
        cx = center_px[0] + (off * dx_along) / mpp
        cy = center_px[1] - (off * dy_along) / mpp
        x0, y0 = cx - 25 * dx_perp, cy + 25 * dy_perp
        x1, y1 = cx + 25 * dx_perp, cy - 25 * dy_perp
        draw.line([(x0, y0), (x1, y1)], fill=(255, 255, 0, 255), width=1)
        # small green dot where this transect actually snapped to on the mask, if
        # it differs from the naive OSM-derived point - makes an OSM/imagery
        # position offset (the road sitting away from the expected point) visible
        if mask is not None:
            snapped = _nearest_mask_pixel(mask, cx, cy, max_radius_px)
            if snapped is not None and (abs(snapped[0] - cx) > 2 or abs(snapped[1] - cy) > 2):
                sx, sy = snapped
                draw.ellipse([sx - 3, sy - 3, sx + 3, sy + 3], fill=(0, 255, 0, 255))
    r = 4
    draw.ellipse([center_px[0] - r, center_px[1] - r, center_px[0] + r, center_px[1] + r], outline=(0, 255, 255, 255), width=2)
    label = f"{np.median(widths):.1f} m (n={len(widths)})" if widths else "no measurement"
    draw.text((4, 4), label, fill=(255, 255, 0, 255))
    if extra_label:
        draw.text((4, 16), extra_label, fill=(0, 255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path, quality=85)


def load_tracks(alignment_path: str, epsg: int) -> dict:
    """{align_id: row} of the clipped alignment in the metric CRS, or {} if
    the file isn't there. Look rows up with track_for(), not directly."""
    if not Path(alignment_path).exists():
        return {}
    align = gpd.read_file(alignment_path).to_crs(epsg)
    return {row["align_id"]: row for _, row in align.iterrows()}


def track_for(tracks: dict, r):
    """The alignment segment a crossing row sits on, or None. align_id is just
    a running number per crossing_identify.py run - an alignment file from a
    different run (other clip/boundary) reuses the same ids for other
    segments - so the align_name has to match too, where both have one."""
    seg = tracks.get(r.get("align_id"))
    if seg is None:
        return None
    if "align_name" in seg.index and pd.notna(r.get("align_name")) and seg["align_name"] != r["align_name"]:
        return None
    return seg.geometry


def road_meets_track(p1: tuple[float, float], p2: tuple[float, float], track, reach_m: float = 100.0):
    """Where a measured road actually meets the track: the road centreline
    through the midpoint of its width line (p1-p2, which runs across the
    road), followed along the road until it hits the alignment. Returns
    (x, y, chainage_m) of the hit nearest the width line, or None if the
    road doesn't reach the track within reach_m. Differs from the crossing
    point when the transect used sits along the road from it, or when the
    road in the imagery sits off the road line in the road data."""
    (x1, y1), (x2, y2) = p1, p2
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return None
    ux, uy = -(y2 - y1) / length, (x2 - x1) / length  # along the road
    probe = LineString([(mx - reach_m * ux, my - reach_m * uy), (mx + reach_m * ux, my + reach_m * uy)])
    hit = probe.intersection(track)
    points = [g for g in getattr(hit, "geoms", [hit]) if g.geom_type == "Point" and not g.is_empty]
    if not points:
        return None
    best = min(points, key=lambda q: math.hypot(q.x - mx, q.y - my))
    return best.x, best.y, track.project(best)


def write_width_geometry(cross: gpd.GeoDataFrame, epsg: int, alignment_path: str, out_path: str) -> None:
    """One combined layer, geometry drawn to scale, up to three kinds of line
    per crossing, all drawn where the measured road meets the track (see
    road_meets_track) so they cross each other at the structure:
      - "width": the SAM3 measurement across the road, from the two road-edge
        pixels the transect walk found - moved along the road from the
        transect it was measured on - only present where a live SAM3
        measurement ran (not for a --measured-csv merge, which has no pixel
        geometry).
      - "extent": outer edge to outer edge of every road within 20 m of it
        (a divided road's other carriageway, a service road), where the
        imagery found any.
      - "span": the length of structure needed *along the railway track*
        (span_m, set by structure_decide.py - absent for boxes and high-skew
        crossings), drawn as the real alignment geometry either side of
        span_centre_chainage_m, not just a straight tangent segment, so it
        stays accurate on curves.
    A line whose road doesn't meet the track stays where it was measured.
    Every line carries its own length_m (the width, extent or span it shows)
    and a few of its crossing's structure attributes, for styling/labelling.
    """
    # only what a map reader needs - the full record is in the register the layer was drawn from
    props = ["crossing_id", "project_ch", "irc_class", "angle_deg", "structure_type", "box_size",
             "opening_width_m", "span_m", "road_lowering_m", "structure_flags"]
    props = [c for c in props if c in cross.columns]
    tracks = load_tracks(alignment_path, epsg) if "align_id" in cross.columns else {}
    if not tracks:
        print(f"\n  [warn] {alignment_path} not found - width lines drawn where measured, no span lines")

    width_rows = []
    for kind, prefix in (("width", "width"), ("extent", "ext")):
        cols = [f"{prefix}_p1_x", f"{prefix}_p1_y", f"{prefix}_p2_x", f"{prefix}_p2_y"]
        if not set(cols) <= set(cross.columns):
            continue
        for _, r in cross[cross[cols].notna().all(axis=1)].iterrows():
            p1, p2 = (r[cols[0]], r[cols[1]]), (r[cols[2]], r[cols[3]])
            track = track_for(tracks, r)
            hit = road_meets_track(p1, p2, track) if track is not None else None
            dx, dy = (hit[0] - (p1[0] + p2[0]) / 2, hit[1] - (p1[1] + p2[1]) / 2) if hit else (0.0, 0.0)
            geom = LineString([(p1[0] + dx, p1[1] + dy), (p2[0] + dx, p2[1] + dy)])
            width_rows.append({"crossing_id": r["crossing_id"], "kind": kind, "length_m": round(geom.length, 2),
                               **{c: r[c] for c in props if c != "crossing_id"}, "geometry": geom})

    span_rows = []
    mismatched = 0
    if tracks and "span_m" in cross.columns:
        for _, r in cross[cross["span_m"].notna() & cross["chainage_m"].notna()].iterrows():
            line = track_for(tracks, r)
            if line is None:
                mismatched += 1
                continue
            half = r["span_m"] / 2.0
            centre = r["span_centre_chainage_m"] if pd.notna(r.get("span_centre_chainage_m")) else r["chainage_m"]
            start = max(centre - half, 0.0)
            end = min(centre + half, line.length)
            if end <= start:
                continue
            geom = substring(line, start, end)
            span_rows.append({"crossing_id": r["crossing_id"], "kind": "span", "length_m": round(geom.length, 2),
                              **{c: r[c] for c in props if c != "crossing_id"}, "geometry": geom})
    if mismatched:
        print(f"\n  [warn] {mismatched} span line(s) skipped - their align_id/align_name isn't in {alignment_path} "
              f"(alignment file from a different crossing_identify.py run? pass the matching --alignment)")

    if not width_rows and not span_rows:
        print("\nno width/span geometry available (likely a --measured-csv run with no matching alignment) - "
              "skipping width-geometry output")
        return

    combined = gpd.GeoDataFrame(width_rows + span_rows, geometry="geometry", crs=epsg).to_crs(4326)
    write_geojson(combined, Path(out_path))
    print(f"wrote {out_path} ({len(width_rows)} width/extent line(s), {len(span_rows)} span line(s))")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    try:
        os.replace(tmp, path)
    except PermissionError:
        print(f"  [warn] {path} is locked (open in an editor?) - wrote {tmp} instead")
