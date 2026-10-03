"""Build a short recheck worksheet for the crossings the automated width
measurement (src/measure_width_sam3.py) couldn't resolve confidently:
flagged SUSPECT, or no imagery/road geometry to measure at all. Everything
else already has an automated width and doesn't need this. Minor roads are
left out - they're re-routed (structure_decide.py rule S1), so their width
never reaches a design.

Optional - only useful if someone wants to manually verify/override a
specific low-confidence result (e.g. with the ruler tool in the review
map, QGIS, or a field visit). Writes to a separate file from
measure_width_sam3.py's output so it never overwrites the automated results.

Usage:
    python src/make_survey_worksheet.py                    # the region crossing_identify.py last ran for
    python src/make_survey_worksheet.py --region maharashtra
    # fill in measured_width_m for rows you've rechecked, then:
    python src/measure_width_sam3.py --region maharashtra --measured-csv output/maharashtra_width_recheck.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

from geo_utils import latest, region_paths, resolve_region
from irc_classify import MAJOR_CLASSES


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", default=None, help="Region to check. Default: the region crossing_identify.py last ran for")
    region = resolve_region(ap.parse_args().region)
    paths = region_paths(region)
    out_csv = paths["recheck_csv"]
    final, base = latest(paths["final"]), latest(paths["crossings"])
    if final.exists():
        cross = gpd.read_file(final)
    elif base.exists():
        cross = gpd.read_file(base)
    else:
        raise SystemExit("No crossings file found - run src/crossing_identify.py first")

    kept = cross["irc_class"].isin(MAJOR_CLASSES)
    for col in ("surface", "surface_sam3"):  # a paved minor road is kept too (structure_decide.py S1)
        if col in cross.columns:
            kept |= cross[col].eq("paved")
    cross = cross[(cross["decision"] != "no_crossing") & kept].copy()
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
        "dual_carriageway": cross["dual_carriageway"],
        "lat": cross.geometry.y,
        "lon": cross.geometry.x,
        "measured_width_m": None,   # <- fill this in after rechecking
        "cw1_width_m": None,        # <- divided road only: each carriageway and the median between them
        "median_width_m": None,
        "cw2_width_m": None,
        "measured_by": None,        # <- name/initials
        "notes": None,
    })

    Path(out_csv).parent.mkdir(exist_ok=True)
    sheet.to_csv(out_csv, index=False)
    print(f"wrote {out_csv} ({len(sheet)} row(s) that need a second opinion)")
    print("\nTo recheck one:")
    print(f"  1. python src/make_review_map.py --region {region}, open {paths['review_map']}")
    print(f"  2. find the crossing_id, open its QA crop link in the popup (or check {paths['images_dir']}/<id>.jpg directly)")
    print("  3. turn on 'Google Satellite Hybrid' and the ruler tool for a second visual opinion")
    print("  4. put the corrected number in measured_width_m, save the CSV")
    print(f"  5. python src/measure_width_sam3.py --region {region} --measured-csv {out_csv}")


if __name__ == "__main__":
    main()
