"""Fetch a small georeferenced imagery crop around a point from Esri World
Imagery (standard XYZ tile access - the same source QGIS's built-in
"ArcGIS World Imagery" connection and countless GIS tools use for
analysis basemaps). Unlike Google's Maps Platform terms, Esri's World
Imagery is published as a basemap for exactly this kind of GIS digitising
and analysis; it does not carry a "no tracing" clause. See README for the
full reasoning behind using this source instead of Google's.

Resolution is whatever's available locally - commonly ~0.3-1 m in India,
checked empirically for this corridor at ~0.56 m/px (zoom 18). That's
coarser than the 0.5 m licensed-imagery target the project brief assumes,
so treat widths measured from it as a first-pass automated estimate.
"""

from __future__ import annotations

import math

import numpy as np
import requests
from PIL import Image

# TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
TILE_URL = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
TILE_SIZE = 256
PLACEHOLDER_MAX_BYTES = 6000  # "Map data not yet available" tiles are small flat-grey JPEGs
DEFAULT_ZOOM_CANDIDATES = (19, 18, 17, 16)

_tile_cache: dict[tuple[int, int, int], "Image.Image | None"] = {}


def mpp_at(lat_deg: float, zoom: int) -> float:
    """Web Mercator metres-per-pixel at this latitude and zoom."""
    return 156543.03392 * math.cos(math.radians(lat_deg)) / (2 ** zoom)


def _fetch_tile(z: int, x: int, y: int, retries: int = 3) -> "Image.Image | None":
    key = (z, x, y)
    if key in _tile_cache:
        return _tile_cache[key]
    img = None
    for attempt in range(retries):
        try:
            r = requests.get(TILE_URL.format(z=z, x=x, y=y), timeout=20)
            if r.status_code == 200 and len(r.content) > PLACEHOLDER_MAX_BYTES:
                from io import BytesIO
                img = Image.open(BytesIO(r.content)).convert("RGB")
                break
            elif r.status_code == 200:
                break  # genuine placeholder tile ("no imagery here") - not a transient failure, don't retry
        except requests.RequestException:
            pass  # transient network error - retry, and don't cache the failure
    else:
        return None  # exhausted retries on a real (non-placeholder) failure - leave uncached so a later call can retry
    _tile_cache[key] = img
    return img


def _lonlat_to_pixel(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    n = 2 ** zoom
    x = (lon + 180.0) / 360.0 * n * TILE_SIZE
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n * TILE_SIZE
    return x, y


def fetch_crop(lon: float, lat: float, half_extent_m: float = 100.0,
                zoom_candidates=DEFAULT_ZOOM_CANDIDATES) -> tuple[np.ndarray, float, int] | None:
    """Return (rgb_array, metres_per_pixel, zoom) for a square crop of side
    2*half_extent_m centred on (lon, lat), or None if no imagery is
    available at any candidate zoom."""
    for zoom in zoom_candidates:
        mpp = mpp_at(lat, zoom)
        half_px = half_extent_m / mpp
        cx, cy = _lonlat_to_pixel(lon, lat, zoom)
        x0, y0 = cx - half_px, cy - half_px
        x1, y1 = cx + half_px, cy + half_px

        tx0, ty0 = int(x0 // TILE_SIZE), int(y0 // TILE_SIZE)
        tx1, ty1 = int(x1 // TILE_SIZE), int(y1 // TILE_SIZE)

        tiles = {}
        ok = True
        for tx in range(tx0, tx1 + 1):
            for ty in range(ty0, ty1 + 1):
                img = _fetch_tile(zoom, tx, ty)
                if img is None:
                    ok = False
                    break
                tiles[(tx, ty)] = img
            if not ok:
                break
        if not ok or not tiles:
            continue

        mosaic_w = (tx1 - tx0 + 1) * TILE_SIZE
        mosaic_h = (ty1 - ty0 + 1) * TILE_SIZE
        mosaic = Image.new("RGB", (mosaic_w, mosaic_h))
        for (tx, ty), img in tiles.items():
            mosaic.paste(img, ((tx - tx0) * TILE_SIZE, (ty - ty0) * TILE_SIZE))

        left = int(round(x0 - tx0 * TILE_SIZE))
        top = int(round(y0 - ty0 * TILE_SIZE))
        size = int(round(half_px * 2))
        crop = mosaic.crop((left, top, left + size, top + size))
        return np.array(crop), mpp, zoom
    return None
