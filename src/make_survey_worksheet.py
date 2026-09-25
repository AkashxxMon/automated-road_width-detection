"""Build a short recheck worksheet for the crossings the automated width
measurement (src/measure_width_sam3.py) couldn't resolve confidently:
flagged SUSPECT, or no imagery/road geometry to measure at all. Everything
else already has an automated width and doesn't need this.

Optional - only useful if someone wants to manually verify/override a
specific low-confidence result (e.g. with the ruler tool in the review
map, QGIS, or a field visit). Writes to a separate file from
measure_width_sam3.py's output so it never overwrites the automated results.

Usage:
    python src/make_survey_worksheet.py
    # fill in measured_width_m for rows you've rechecked, then:
    python src/measure_width_sam3.py --measured-csv output/gujarat_width_recheck.csv
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

CROSSINGS_FINAL = "output/gujarat_crossings_final.geojson"
CROSSINGS_BASE = "output/gujarat_crossings.geojson"
OUT_CSV = "output/gujarat_width_recheck.csv"


def main() -> None:
    if Path(CROSSINGS_FINAL).exists():
        cross = gpd.read_file(CROSSINGS_FINAL)
    elif Path(CROSSINGS_BASE).exists():
        cross = gpd.read_file(CROSSINGS_BASE)
    else:
        raise SystemExit("No crossings file found - run src/crossing_identify.py first")

    cross = cross[cross["decision"] != "no_crossing"].copy()
    needs_recheck = cross["width_source"].str.contains("SUSPECT|needs field", regex=True, na=False)
    cross = cross[needs_recheck].sort_values(["align_name", "chainage_m"])

    if cross.empty:
        print("Nothing needs a recheck - every crossing already has a usable width.")
        return

    sheet = pd.DataFrame({
        "crossing_id": cross["crossing_id"],
        "decision": cross["decision"],
        "align_name": cross["align_name"],
        "chainage_m": cross["chainage_m"],
        "road_name": cross["name"],
        "irc_class": cross["irc_class"],
        "current_width_source": cross["width_source"],
        "current_design_width_m": cross["design_width_m"],
        "angle_deg": cross["angle_deg"],
        "lat": cross.geometry.y,
        "lon": cross.geometry.x,
        "measured_width_m": None,   # <- fill this in after rechecking
        "measured_by": None,        # <- name/initials
        "notes": None,
    })

    Path(OUT_CSV).parent.mkdir(exist_ok=True)
    sheet.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} ({len(sheet)} row(s) that need a second opinion)")
    print("\nTo recheck one:")
    print("  1. python src/make_review_map.py, open output/gujarat_review_map.html")
    print("  2. find the crossing_id, open its QA crop link in the popup (or check output/width_crops/<id>.jpg directly)")
    print("  3. turn on 'Google Satellite Hybrid' and the ruler tool for a second visual opinion")
    print("  4. put the corrected number in measured_width_m, save the CSV")
    print("  5. python src/measure_width_sam3.py --measured-csv output/gujarat_width_recheck.csv")


if __name__ == "__main__":
    main()
