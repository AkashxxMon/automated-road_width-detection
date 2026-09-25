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

TRANSECT_OFFSETS_M = (-20, -10, 0, 10, 20)
MAX_PLAUSIBLE_WIDTH_M = 40.0  # wider than that is almost certainly the mask leaking into a field


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


def measure_transects(mask: np.ndarray, mpp: float, bearing: float, center_px: tuple[float, float]) -> list[float]:
    h, w = mask.shape
    brg = math.radians(bearing)
    perp = brg + math.pi / 2
    dx_perp, dy_perp = math.sin(perp), math.cos(perp)  # bearing: 0=north(+y up in world, -y in image)
    dx_along, dy_along = math.sin(brg), math.cos(brg)

    widths = []
    for off in TRANSECT_OFFSETS_M:
        cx = center_px[0] + (off * dx_along) / mpp
        cy = center_px[1] - (off * dy_along) / mpp
        cxi, cyi = int(round(cx)), int(round(cy))
        if not (0 <= cxi < w and 0 <= cyi < h):
            continue
        if not mask[cyi, cxi]:
            found = False
            for r in range(1, 6):
                for s in (-1, 1):
                    yy = cyi + s * r
                    if 0 <= yy < h and mask[yy, cxi]:
                        cyi = yy
                        found = True
                        break
                if found:
                    break
            if not found:
                continue

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
            widths.append(width_m)
    return widths


def save_qa_image(path: Path, rgb: np.ndarray, mask: np.ndarray | None, bearing: float,
                   mpp: float, center_px: tuple[float, float], widths: list[float]) -> None:
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
    for off in TRANSECT_OFFSETS_M:
        cx = center_px[0] + (off * dx_along) / mpp
        cy = center_px[1] - (off * dy_along) / mpp
        x0, y0 = cx - 25 * dx_perp, cy + 25 * dy_perp
        x1, y1 = cx + 25 * dx_perp, cy - 25 * dy_perp
        draw.line([(x0, y0), (x1, y1)], fill=(255, 255, 0, 255), width=1)
    r = 4
    draw.ellipse([center_px[0] - r, center_px[1] - r, center_px[0] + r, center_px[1] + r], outline=(0, 255, 255, 255), width=2)
    label = f"{np.median(widths):.1f} m (n={len(widths)})" if widths else "no measurement"
    draw.text((4, 4), label, fill=(255, 255, 0, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path, quality=85)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    try:
        os.replace(tmp, path)
    except PermissionError:
        print(f"  [warn] {path} is locked (open in an editor?) - wrote {tmp} instead")
