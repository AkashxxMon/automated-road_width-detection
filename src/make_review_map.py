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
    python src/make_review_map.py                       # the region crossing_identify.py last ran for
    python src/make_review_map.py --region maharashtra
"""

from __future__ import annotations

import argparse
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
from folium.plugins import MeasureControl

from geo_utils import latest, region_paths, resolve_region

VALIDATION = "output/gujarat_validation.geojson"  # optional - written by src/validate.py (Gujarat ground truth)

DECISION_COLOR = {"crossing": "#EA4335", "review": "#F9AB00", "no_crossing": "#80868B"}
DECISION_LABEL = {"crossing": "Crossing required", "review": "Review", "no_crossing": "No crossing"}
STRUCTURE_COLOR = {"BRIDGE": "#EA4335", "BOX": "#1E8E3E", "DUAL_DESIGNER_DECISION": "#8E24AA",
                   "PENDING_SURVEY": "#F9AB00",
                   "NIL_REROUTE": "#9AA0A6", "MERGED": "#BDC1C6", "NONE": "#5F6368"}
STRUCTURE_LABEL = {"BRIDGE": "Bridge (RUB)", "BOX": "Box culvert (RUB)",
                   "DUAL_DESIGNER_DECISION": "Divided road - designer decides",
                   "PENDING_SURVEY": "Major road - width pending",
                   "NIL_REROUTE": "Minor road - re-route",
                   "MERGED": "Merged into another crossing's structure", "NONE": "No structure"}

GOOGLE_SAT_URL = "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}"
GOOGLE_SAT_ATTR = (
    "Imagery (c) Google - live tiles for on-screen research/review only; "
    "not downloaded, traced, or used for training by this pipeline"
)


def popup_html(row, crops_dir: str) -> str:
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
            f'<a href="{crops_dir}/{row["crossing_id"]}.jpg" target="_blank">QA crop &rarr;</a><br>'
        )

    structure_bits = ""
    if "structure_type" in row.index:
        suggested = f" (suggested: {row['structure_suggested']})" if pd.notna(row.get("structure_suggested")) else ""
        structure_bits = (
            f"<b>Structure:</b> {STRUCTURE_LABEL.get(row['structure_type'], row['structure_type'])}{suggested} "
            f"&middot; {row['structure_rule']}<br>"
            f'<span style="color:#5f6368;font-size:12px">{fmt(row.get("structure_reason"))}</span><br>'
            f"<b>Opening:</b> {fmt(row.get('opening_width_m'), ' m')} &middot; <b>Box:</b> {fmt(row.get('box_size'))}"
            f" &middot; <b>Split box:</b> {fmt(row.get('box_split_size'))}<br>"
            f"<b>Flags:</b> {fmt(row.get('structure_flags'))}<br>"
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
      {structure_bits}
      <b>Span along track:</b> {fmt(row['span_m'], ' m')}<br>
      <b>Near junction:</b> {row['near_junction']} &middot; <b>Chainage:</b> {fmt(row.get('chainage_m'), ' m')}
    </div>
    """


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", default=None, help="Region to map. Default: the region crossing_identify.py last ran for")
    args = ap.parse_args()
    region = resolve_region(args.region)
    paths = region_paths(region)
    out_html = paths["review_map"]
    # the furthest-along register there is: structures, else measured widths, else just the crossings
    for key in ("structures", "final", "crossings"):
        crossings_path = latest(paths[key])
        if crossings_path.exists():
            break
    else:
        raise SystemExit(f"No {paths['crossings']} - run src/crossing_identify.py for {region} first")
    crops_dir = Path(paths["images_dir"]).relative_to(Path(out_html).parent).as_posix()
    print(f"using {crossings_path}")

    crossings = gpd.read_file(crossings_path)
    align = gpd.read_file(latest(paths["alignment"]))

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

    align_layer = folium.FeatureGroup(name=f"Proposed alignment ({region})", show=True)
    for _, row in align.iterrows():
        folium.PolyLine(
            [(lat, lon) for lon, lat, *_ in row.geometry.coords],
            color="#1A73E8", weight=4, opacity=0.85,
            tooltip=row["align_name"],
        ).add_to(align_layer)
    align_layer.add_to(m)

    # one layer per structure type once structure_decide.py has run, else per connectivity decision
    if "structure_type" in crossings.columns:
        by, labels, colors = "structure_type", STRUCTURE_LABEL, STRUCTURE_COLOR
    else:
        by, labels, colors = "decision", DECISION_LABEL, DECISION_COLOR
    for key, label in labels.items():
        layer = folium.FeatureGroup(name=f"{label} ({(crossings[by] == key).sum()})", show=key not in ("NIL_REROUTE", "NONE", "MERGED"))
        subset = crossings[crossings[by] == key]
        for _, row in subset.iterrows():
            folium.CircleMarker(
                location=(row.geometry.y, row.geometry.x),
                radius=6,
                color="#202124",
                weight=1,
                fill=True,
                fill_color=colors[key],
                fill_opacity=0.9,
                popup=folium.Popup(popup_html(row, crops_dir), max_width=320),
                tooltip=f"{row['crossing_id']} - {row['name'] or row['highway']}",
            ).add_to(layer)
        layer.add_to(m)

    if region == "gujarat" and Path(VALIDATION).exists():
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

    Path(out_html).parent.mkdir(exist_ok=True)
    m.save(out_html)
    print(f"wrote {out_html}")
    print("Open it in a browser. Turn on 'Google Satellite Hybrid' in the layer")
    print("control (top right) to visually check width/habitation for a given")
    print("crossing - tiles load live from Google and are not saved anywhere.")
    print("Use the ruler icon (top-left) to click across a road and read its")
    print("width in metres - see src/make_survey_worksheet.py to log it.")


if __name__ == "__main__":
    main()
