"""OpenStreetMap roads for a corridor, from Geofabrik's India zone extracts.

Not the Overpass API: it allows each IP only a couple of query slots and
starts refusing (HTTP 429/504) after a handful of large queries - the jam seen
after ~8 queries. Geofabrik instead publishes each India zone as one .osm.pbf
file, refreshed daily: one plain download per zone, cached in input/osm/ and
reused for every state in it, no query limits at all. The roads are then read
locally (GDAL's OSM driver), cropped to the corridor, and cached per region in
shared/osm_roads_<region>.parquet, so later runs don't touch the network or
re-read the extract.

Which zones a corridor needs comes from Geofabrik's zone outlines (.poly).
"""

from __future__ import annotations

import re
import shutil
import time
import urllib.request
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio
from shapely.geometry import Polygon
from shapely.ops import unary_union

GEOFABRIK = "https://download.geofabrik.de/asia/india"
ZONES = ("central-zone", "eastern-zone", "north-eastern-zone", "northern-zone", "southern-zone", "western-zone")
CACHE_DIR = Path("input/osm")
# drivable roads only - no service/driveway, footway, path, cycleway, steps, construction or proposed
ROAD_TYPES = ("motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary",
              "secondary_link", "tertiary", "tertiary_link", "unclassified", "residential", "living_street", "road", "track")
PAVED = {"paved", "asphalt", "concrete", "concrete:plates", "paving_stones", "sett", "chipseal", "metal"}
UNPAVED = {"unpaved", "dirt", "earth", "ground", "gravel", "fine_gravel", "compacted", "sand", "mud", "grass", "pebblestone"}


def _download(url: str, dest: Path, retries: int = 4) -> Path:
    """One GET to a temp file, moved into place when complete; retried with backoff."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            print(f"  downloading {url} ...", flush=True)
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            tmp.replace(dest)
            return dest
        except OSError as e:
            wait = 30 * 2 ** attempt
            print(f"  [warn] {e} - retrying in {wait} s")
            time.sleep(wait)
    raise SystemExit(f"could not download {url}")


def _poly(zone: str) -> Polygon:
    """Geofabrik's zone outline (.poly: rings of 'lon lat' lines, END-terminated; '!' marks a hole)."""
    path = CACHE_DIR / f"{zone}.poly"
    if not path.exists():
        _download(f"{GEOFABRIK}/{zone}.poly", path)
    rings, holes, cur, hole = [], [], [], False
    for line in path.read_text().splitlines()[1:]:
        s = line.strip()
        if s == "END":
            if cur:
                (holes if hole else rings).append(Polygon(cur))
                cur = []
            continue
        if not s:
            continue
        nums = s.split()
        if len(nums) >= 2 and re.match(r"^[-+]?[\d.]", nums[0]):
            cur.append((float(nums[0]), float(nums[1])))
        else:  # a ring's name line, e.g. "1" or "!2" (a hole)
            hole = s.startswith("!")
    return unary_union(rings).difference(unary_union(holes)) if holes else unary_union(rings)


def zones_for(area_wgs84) -> list[str]:
    return [z for z in ZONES if _poly(z).intersects(area_wgs84)]


def _extract(zone: str, max_age_days: float) -> Path:
    path = CACHE_DIR / f"{zone}-latest.osm.pbf"
    if not path.exists() or (time.time() - path.stat().st_mtime) / 86400 > max_age_days:
        _download(f"{GEOFABRIK}/{zone}-latest.osm.pbf", path)
    return path


def _tag(other_tags, key: str):
    if not isinstance(other_tags, str):
        return None
    m = re.search(rf'"{re.escape(key)}"=>"([^"]*)"', other_tags)
    return m.group(1) if m else None


def _surface(v) -> str | None:
    v = (v or "").lower()
    return "paved" if v in PAVED else "unpaved" if v in UNPAVED else None


def load_osm_roads(corridor_wgs84, cache_path: str, max_age_days: float = 30.0) -> gpd.GeoDataFrame:
    """OSM roads inside the corridor (EPSG:4326), as highway/ref/name/surface/osm_id/geometry -
    the same columns as the NIC roads. Cached at cache_path; delete it to re-read."""
    cache = Path(cache_path)
    if cache.exists():
        roads = gpd.read_parquet(cache)
        print(f"OSM roads: {len(roads)} from cache {cache} (delete it to refresh)")
        return roads
    zones = zones_for(corridor_wgs84)
    print(f"OSM roads: Geofabrik zone(s) {', '.join(zones)}")
    where = "highway IN (" + ", ".join(f"'{t}'" for t in ROAD_TYPES) + ")"
    parts = []
    for zone in zones:
        pbf = _extract(zone, max_age_days)
        t = time.time()
        part = pyogrio.read_dataframe(pbf, layer="lines", bbox=corridor_wgs84.bounds, where=where,
                                      columns=["osm_id", "name", "highway", "other_tags"])
        print(f"  {zone}: {len(part)} road line(s) in the corridor's bbox ({time.time() - t:.0f} s)")
        parts.append(part)
    roads = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=4326)
    roads = roads[roads.geometry.notna() & roads.intersects(corridor_wgs84)]
    roads = roads.drop_duplicates("osm_id").explode(index_parts=False).reset_index(drop=True)
    roads["ref"] = roads["other_tags"].apply(lambda t: _tag(t, "ref"))
    roads["surface"] = roads["other_tags"].apply(lambda t: _surface(_tag(t, "surface")))
    roads = roads[["highway", "ref", "name", "surface", "osm_id", "geometry"]]
    cache.parent.mkdir(parents=True, exist_ok=True)
    roads.to_parquet(cache)
    print(f"  {len(roads)} OSM road(s) in the corridor - cached to {cache}")
    return roads
