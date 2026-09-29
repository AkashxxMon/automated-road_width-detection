"""Small helpers shared across scripts: safe file writes and UTM zone
auto-detection, so no script has to hardcode a CRS or an output path."""

from __future__ import annotations

import math
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd


def write_geojson(gdf: gpd.GeoDataFrame, path: Path) -> None:
    """Write via a temp file + replace - an editor/extension holding the
    output open for preview (common for .geojson) can block a direct
    write-in-place on Windows, but not a rename over it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.geojson")
    gdf.to_file(tmp, driver="GeoJSON")
    try:
        os.replace(tmp, path)
    except PermissionError:
        print(f"  [warn] {path} is locked (open in an editor?) - wrote {tmp} instead")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    try:
        os.replace(tmp, path)
    except PermissionError:
        print(f"  [warn] {path} is locked (open in an editor?) - wrote {tmp} instead")


def get_utm_epsg(lon: float, lat: float) -> int:
    """UTM zone EPSG code for a lon/lat, north or south hemisphere."""
    zone = int(math.floor((lon + 180) / 6) % 60) + 1
    return (32600 if lat >= 0 else 32700) + zone


def offset_lonlat(lon: float, lat: float, east_m: float, north_m: float) -> tuple[float, float]:
    """(lon, lat) shifted by (east_m, north_m) metres, via a local flat-earth
    approximation - fine at the ~100-200m scale these imagery crops cover,
    consistent with the local mpp assumption width_core.py's transects
    already make."""
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
    return lon + east_m / m_per_deg_lon, lat + north_m / m_per_deg_lat


def flatten_geom(geom):
    """Yield LineString parts out of a LineString/MultiLineString/GeometryCollection."""
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == "LineString":
        yield geom
    elif geom.geom_type == "MultiLineString":
        yield from geom.geoms
    elif geom.geom_type == "GeometryCollection":
        for part in geom.geoms:
            yield from flatten_geom(part)
