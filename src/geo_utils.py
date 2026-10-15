"""Small helpers shared across scripts: safe file writes, per-region default
paths and UTM zone auto-detection, so no script has to hardcode a CRS or an
output path."""

from __future__ import annotations

import math
import os
import re
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


def latest(path) -> Path:
    """path, or the .tmp copy beside it if that's newer. write_geojson/write_csv
    leave one when the real file was locked (open in QGIS/an editor) - the next
    stage has to read what the previous one actually wrote, not the stale
    locked file."""
    path = Path(path)
    tmp = path.with_suffix(".tmp" + path.suffix)
    if tmp.exists() and (not path.exists() or tmp.stat().st_mtime > path.stat().st_mtime):
        print(f"  [note] reading {tmp} - {path.name} was locked when the last step wrote it (close it in QGIS "
              f"and rerun that step to replace it)")
        return tmp
    return path


# ISO 3166-2:IN codes, keyed by region_slug() of the state/UT name - the default crossing_id prefix
STATE_CODES = {
    "andaman_and_nicobar": "AN", "andhra_pradesh": "AP", "arunachal_pradesh": "AR", "assam": "AS", "bihar": "BR",
    "chandigarh": "CH", "chhattisgarh": "CT", "dadra_and_nagar_haveli": "DN", "daman_and_diu": "DD", "delhi": "DL",
    "goa": "GA", "gujarat": "GJ", "haryana": "HR", "himachal_pradesh": "HP", "jammu_and_kashmir": "JK",
    "jharkhand": "JH", "karnataka": "KA", "kerala": "KL", "ladakh": "LA", "lakshadweep": "LD", "madhya_pradesh": "MP",
    "maharashtra": "MH", "manipur": "MN", "meghalaya": "ML", "mizoram": "MZ", "nagaland": "NL", "odisha": "OR",
    "orissa": "OR", "puducherry": "PY", "punjab": "PB", "rajasthan": "RJ", "sikkim": "SK", "tamil_nadu": "TN",
    "telangana": "TG", "tripura": "TR", "uttar_pradesh": "UP", "uttarakhand": "UT", "uttaranchal": "UT",
    "west_bengal": "WB",
}


def region_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_") or "region"


def region_prefix(region: str) -> str:
    """Default crossing_id prefix for a region: its state code, else its first two letters."""
    slug = region_slug(region)
    return STATE_CODES.get(slug, slug[:2].upper())


def region_paths(region: str) -> dict[str, str]:
    """Every stage's default input/output path for one region (a state name,
    or a boundary file's name) - so a run for one state never overwrites, or
    gets read in place of, another's."""
    r = region_slug(region)
    return {
        "alignment": f"shared/alignment_{r}.geojson",
        "corridor": f"shared/corridor_{r}.geojson",
        "roads": f"shared/roads_{r}.geojson",
        "osm_roads": f"shared/osm_roads_{r}.parquet",
        "crossings": f"output/{r}_crossings.geojson",
        "crossings_summary": f"output/{r}_crossings_summary.csv",
        "final": f"output/{r}_crossings_final_gl.geojson",
        "final_csv": f"output/{r}_crossings_final_gl.csv",
        "survey_csv": f"output/{r}_width_survey_gl.csv",
        "images_dir": f"output/{r}_width_crops_sam3_gl",
        "width_geometry": f"output/{r}_width_geometry_gl.geojson",
        "structures": f"output/{r}_structures.geojson",
        "structures_csv": f"output/{r}_structures.csv",
        "structure_geometry": f"output/{r}_structure_geometry.geojson",
        "review_map": f"output/{r}_review_map.html",
        "recheck_csv": f"output/{r}_width_recheck.csv",
    }


LAST_REGION_FILE = Path("output/last_region.txt")  # written by crossing_identify.py, read by every later step


def remember_region(region: str) -> None:
    LAST_REGION_FILE.parent.mkdir(parents=True, exist_ok=True)
    LAST_REGION_FILE.write_text(region_slug(region) + "\n", encoding="utf-8")


def resolve_region(region: str | None) -> str:
    """The region a later step works on: --region if given, else the one
    crossing_identify.py last ran for, else gujarat (this checkout's example)."""
    if region:
        return region_slug(region)
    try:
        last = LAST_REGION_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        last = ""
    resolved = region_slug(last) if last else "gujarat"
    source = f"last crossing_identify.py run, {LAST_REGION_FILE}" if last else "default"
    print(f"region: {resolved} ({source} - pass --region to pick another)")
    return resolved


def fill_region_paths(args, defaults: dict[str, str]) -> None:
    """Resolve args.region, then set every path argument still None to that
    region's default - defaults maps argument name -> region_paths() key."""
    args.region = resolve_region(args.region)
    paths = region_paths(args.region)
    for arg, key in defaults.items():
        if getattr(args, arg) is None:
            setattr(args, arg, paths[key])


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
