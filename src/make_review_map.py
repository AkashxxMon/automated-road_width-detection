"""Stage 8: build an interactive HTML map of the Gujarat crossing register
for manual review.

Includes a Google Satellite Hybrid tile layer as an optional basemap the
reviewer can switch on, purely so they can eyeball road width and nearby
habitation for each candidate before accepting or overriding a decision.
The tiles are fetched live by the browser when a reviewer opens the map
and looks at a given crossing - this pipeline does not download, store,
trace over, or train anything on them. Replace this step with licensed
imagery (drone ortho, Cartosat-3, Pleiades Neo, Maxar) before using the
register for anything beyond prototyping - see the imagery-licensing
note in the project brief / README.

Usage:
    python src/make_review_map.py
"""

from __future__ import annotations

from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
from folium.plugins import MeasureControl

CROSSINGS_BASE = "output/gujarat_crossings.geojson"
CROSSINGS_FINAL = "output/gujarat_crossings_final.geojson"  # written by src/measure_width_sam3.py, if it's been run
ALIGNMENT = "shared/alignment_gujarat.geojson"
VALIDATION = "output/gujarat_validation.geojson"  # optional - written by src/validate.py
OUT_HTML = "output/gujarat_review_map.html"

DECISION_COLOR = {"crossing": "#EA4335", "review": "#F9AB00", "no_crossing": "#80868B"}
DECISION_LABEL = {"crossing": "Crossing required", "review": "Review", "no_crossing": "No crossing"}

GOOGLE_SAT_URL = "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}"
GOOGLE_SAT_ATTR = (
    "Imagery (c) Google - live tiles for on-screen research/review only; "
    "not downloaded, traced, or used for training by this pipeline"
)


def popup_html(row) -> str:
    def fmt(v, suffix=""):
        return "-" if v is None or (isinstance(v, float) and v != v) else f"{v}{suffix}"

    measured_bits = ""
    if "measured_width_m" in row.index and pd.notna(row.get("measured_width_m")):
        suspect = "notes" in row.index and isinstance(row.get("notes"), str) and "SUSPECT" in row["notes"]
        color = "#B31412" if suspect else "#188038"
        measured_bits = (
            f'<b>Auto-measured:</b> <span style="color:{color}">{row["measured_width_m"]} m</span> '
            f'({fmt(row.get("measured_by"))})<br>'
            f'<span style="color:#5f6368;font-size:12px">{fmt(row.get("notes"))}</span><br>'
            f'<a href="width_crops/{row["crossing_id"]}.jpg" target="_blank">QA crop &rarr;</a><br>'
        )

    return f"""
    <div style="font-family:sans-serif;font-size:13px;min-width:240px">
      <b>{row['crossing_id']}</b> &middot; {DECISION_LABEL[row['decision']]}<br>
      <span style="color:#5f6368">{row['align_name']} &middot; chainage {row['chainage_m']:.0f} m</span>
      <hr style="margin:6px 0">
      <b>Road:</b> {row['name'] or row['highway'] or '-'} ({row['irc_class']})<br>
      <b>Rule:</b> {row['rule']} - {row['reason']}<br>
      <b>Crossing angle:</b> {fmt(row['angle_deg'], '&deg;')}<br>
      {measured_bits}
      <b>Design width:</b> {fmt(row['design_width_m'], ' m')} ({row['width_source']})<br>
      <b>Span along track:</b> {fmt(row['span_m'], ' m')}<br>
      <b>Near junction:</b> {row['near_junction']} &middot; <b>Skewed (&lt;45&deg;):</b> {row['skewed']}
    </div>
    """


def main() -> None:
    if Path(CROSSINGS_FINAL).exists():
        crossings_path = CROSSINGS_FINAL
    elif Path(CROSSINGS_BASE).exists():
        crossings_path = CROSSINGS_BASE
    else:
        raise SystemExit(f"Neither {CROSSINGS_FINAL} nor {CROSSINGS_BASE} exists - run src/crossing_identify.py first")
    print(f"using {crossings_path}")

    crossings = gpd.read_file(crossings_path)
    align = gpd.read_file(ALIGNMENT)

    center = [crossings.geometry.y.mean(), crossings.geometry.x.mean()]
    m = folium.Map(location=center, zoom_start=10, tiles="OpenStreetMap", control_scale=True)

    folium.TileLayer(
        tiles=GOOGLE_SAT_URL,
        attr=GOOGLE_SAT_ATTR,
        name="Google Satellite Hybrid (research reference only)",
        overlay=False,
        show=False,
        max_zoom=21,
    ).add_to(m)

    align_layer = folium.FeatureGroup(name="Proposed alignment (Gujarat)", show=True)
    for _, row in align.iterrows():
        folium.PolyLine(
            [(lat, lon) for lon, lat, *_ in row.geometry.coords],
            color="#1A73E8", weight=4, opacity=0.85,
            tooltip=row["align_name"],
        ).add_to(align_layer)
    align_layer.add_to(m)

    for decision, label in DECISION_LABEL.items():
        layer = folium.FeatureGroup(name=f"{label} ({(crossings['decision'] == decision).sum()})", show=True)
        subset = crossings[crossings["decision"] == decision]
        for _, row in subset.iterrows():
            folium.CircleMarker(
                location=(row.geometry.y, row.geometry.x),
                radius=6,
                color="#202124",
                weight=1,
                fill=True,
                fill_color=DECISION_COLOR[decision],
                fill_opacity=0.9,
                popup=folium.Popup(popup_html(row), max_width=320),
                tooltip=f"{row['crossing_id']} - {row['name'] or row['highway']}",
            ).add_to(layer)
        layer.add_to(m)

    if Path(VALIDATION).exists():
        val = gpd.read_file(VALIDATION)
        for matched, label, color in [(True, "matched", "#188038"), (False, "missed", "#8E24AA")]:
            subset = val[val["matched"] == matched]
            layer = folium.FeatureGroup(name=f"Ground-truth bridges - {label} ({len(subset)})", show=False)
            for _, row in subset.iterrows():
                folium.CircleMarker(
                    location=(row.geometry.y, row.geometry.x),
                    radius=4,
                    color=color,
                    weight=2,
                    fill=True,
                    fill_opacity=0.3,
                    tooltip=f"ground truth - {'matched ' + str(row['pred_crossing_id']) if matched else 'no prediction nearby'}",
                ).add_to(layer)
            layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Ruler for manually measuring road width against the Google Satellite
    # layer - a human draws the line and reads the length, same as using
    # the ruler in Google Earth Pro. See src/make_survey_worksheet.py.
    MeasureControl(primary_length_unit="meters", secondary_length_unit=None, active_color="#EA4335").add_to(m)

    Path(OUT_HTML).parent.mkdir(exist_ok=True)
    m.save(OUT_HTML)
    print(f"wrote {OUT_HTML}")
    print("Open it in a browser. Turn on 'Google Satellite Hybrid' in the layer")
    print("control (top right) to visually check width/habitation for a given")
    print("crossing - tiles load live from Google and are not saved anywhere.")
    print("Use the ruler icon (top-left) to click across a road and read its")
    print("width in metres - see src/make_survey_worksheet.py to log it.")


if __name__ == "__main__":
    main()
