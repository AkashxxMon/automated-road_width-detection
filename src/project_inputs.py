"""Optional project data attached to every crossing before the structure call
(structure_decide.py): rail and ground levels (the over/under call, and how
far a road must be lowered for an RUB), project chainage, and the designers'
own ROB/RUB list to check the result against. Each input is optional - a
missing file just leaves its columns empty. File formats are described in
input/ewdfc/SOURCES.md (the EWDFC versions); any project can supply its own
in the same columns.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from width_core import track_for

LEVEL_COLUMNS = ["rail_level_m", "ground_level_m", "headroom_m", "level_tag"]
CHAINAGE_COLUMNS = ["project_ch_m", "project_ch"]
REFERENCE_COLUMNS = ["ref_bridge_no", "ref_kind", "ref_config", "ref_dist_m"]


def _empty(cross: gpd.GeoDataFrame, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(index=cross.index, columns=columns, dtype=object)


def _points_near(path: str, cross: gpd.GeoDataFrame, margin_deg: float = 0.05, **read_kw) -> gpd.GeoDataFrame | None:
    """Rows of a lon/lat point table (parquet or csv) inside the crossings'
    bounding box plus margin, as points in the crossings' CRS - or None if
    the file isn't there."""
    if not path or not Path(path).exists():
        return None
    df = pd.read_parquet(path, **read_kw) if str(path).endswith(".parquet") else pd.read_csv(path, **read_kw)
    lon0, lat0, lon1, lat1 = cross.to_crs(4326).total_bounds
    df = df[df["lon"].between(lon0 - margin_deg, lon1 + margin_deg) & df["lat"].between(lat0 - margin_deg, lat1 + margin_deg)]
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=4326).to_crs(cross.crs)


def attach_levels(cross: gpd.GeoDataFrame, path: str, max_dist_m: float) -> pd.DataFrame:
    """Rail design level and existing ground level at each crossing, from the
    nearest level point within max_dist_m (the 5 m interval sheet). headroom_m
    = rail - ground: how much height an RUB has before the road is lowered.
    level_tag is Viaduct/Tunnel where the railway is on one there."""
    out = _empty(cross, LEVEL_COLUMNS)
    pts = _points_near(path, cross)
    if pts is None:
        print(f"  [note] no levels file at {path} - over/under (ROB vs RUB) check skipped")
        return out
    if pts.empty:
        print(f"  [note] {path} has no level points in this region - over/under check skipped")
        return out
    j = gpd.sjoin_nearest(cross[["geometry"]], pts[["rail_level_m", "ground_level_m", "tag", "geometry"]],
                          max_distance=max_dist_m, distance_col="_d")
    j = j[~j.index.duplicated()]
    out.loc[j.index, "rail_level_m"] = j["rail_level_m"].round(3)
    out.loc[j.index, "ground_level_m"] = j["ground_level_m"].round(3)
    out.loc[j.index, "headroom_m"] = (j["rail_level_m"] - j["ground_level_m"]).round(2)
    out.loc[j.index, "level_tag"] = j["tag"].where(j["tag"].notna(), None)
    print(f"  levels: {out['headroom_m'].notna().sum()}/{len(cross)} crossing(s) have rail and ground levels "
          f"(nearest point within {max_dist_m:g} m)")
    return out


def format_chainage(ch_m: float) -> str:
    """1234567.8 -> '1234+568', the project's km+m notation."""
    sign = "-" if ch_m < 0 else ""
    m = int(round(abs(ch_m)))
    return f"{sign}{m // 1000}+{m % 1000:03d}"


def attach_project_chainage(cross: gpd.GeoDataFrame, tracks: dict, path: str,
                            search_m: float = 300.0, on_line_m: float = 60.0) -> pd.DataFrame:
    """Project chainage at each crossing, interpolated along its alignment
    piece between the two nearest chainage labels on that piece (labels every
    100 m in the EWDFC alignment). The labels set the direction too, so a
    piece digitised against the chainage still comes out right. Labels are
    text positions, offset from the centreline (~35 m for km/100 m labels,
    ~15 m for curve points in the EWDFC KMZs), so each is projected onto
    the line; on_line_m keeps labels of a neighbouring line out."""
    out = _empty(cross, CHAINAGE_COLUMNS)
    if not path or not Path(path).exists():
        print(f"  [note] no chainage labels at {path} - project chainage left empty")
        return out
    labels = gpd.read_file(path).to_crs(cross.crs)
    sidx = labels.sindex
    for idx, row in cross.iterrows():
        line = track_for(tracks, row)
        if line is None:
            continue
        near = labels.iloc[list(sidx.query(row.geometry.buffer(search_m)))]
        near = near[near.geometry.distance(line) <= on_line_m]
        if len(near) < 2:
            continue
        pos = np.array([line.project(g) for g in near.geometry])
        ch = near["ch_m"].to_numpy(dtype=float)
        here = line.project(row.geometry)
        order = np.argsort(np.abs(pos - here))
        i, k = order[0], next((o for o in order[1:] if abs(pos[o] - pos[order[0]]) > 1.0), None)
        if k is None:
            continue
        slope = (ch[k] - ch[i]) / (pos[k] - pos[i])  # ~ +1 or -1 m of chainage per m of line
        value = ch[i] + slope * (here - pos[i])
        out.loc[idx, "project_ch_m"] = round(value, 1)
        out.loc[idx, "project_ch"] = format_chainage(value)
    print(f"  chainage: {out['project_ch_m'].notna().sum()}/{len(cross)} crossing(s) placed on project chainage")
    return out


def attach_reference(cross: gpd.GeoDataFrame, path: str, max_dist_m: float) -> pd.DataFrame:
    """The designers' own structure (bridge no, RUB/ROB, configuration) at
    each crossing, if their list has one within max_dist_m - to check our
    call against theirs."""
    out = _empty(cross, REFERENCE_COLUMNS)
    pts = _points_near(path, cross)
    if pts is None:
        print(f"  [note] no reference structure list at {path} - comparison skipped")
        return out
    if pts.empty:
        return out
    j = gpd.sjoin_nearest(cross[["geometry"]], pts[["ref_bridge_no", "ref_kind", "ref_config", "geometry"]],
                          max_distance=max_dist_m, distance_col="ref_dist_m")
    j = j[~j.index.duplicated()]
    for c in ("ref_bridge_no", "ref_kind", "ref_config"):
        out.loc[j.index, c] = j[c]
    out.loc[j.index, "ref_dist_m"] = j["ref_dist_m"].round(1)
    print(f"  reference: {out['ref_kind'].notna().sum()}/{len(cross)} crossing(s) have a designers' structure "
          f"within {max_dist_m:g} m")
    return out


def reference_coverage(cross: gpd.GeoDataFrame, tracks: dict, path: str, max_dist_m: float,
                       on_line_m: float = 25.0) -> tuple[int, int] | None:
    """(designers' structures on this region's alignment, how many of them
    have one of our crossings within max_dist_m) - roads we missed, or
    didn't treat as needing a structure, show up as the difference."""
    pts = _points_near(path, cross)
    if pts is None or pts.empty or not tracks:
        return None
    lines = gpd.GeoSeries([r.geometry for r in tracks.values()], crs=cross.crs).union_all()
    pts = pts[pts.geometry.distance(lines) <= on_line_m]
    if pts.empty:
        return 0, 0
    d = gpd.sjoin_nearest(pts[["geometry"]], cross[["geometry"]], max_distance=max_dist_m, distance_col="_d")
    return len(pts), int(d.index.unique().size)
