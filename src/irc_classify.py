"""Road hierarchy: OSM highway/ref tags -> IRC class.

The IRC class no longer sets the design width (that comes from the SAM3
measurement only, in measure_width_sam3.py) - it's kept purely to flag a
measurement as SUSPECT when it's implausible for that class of road (e.g.
a "village road" mask that's actually 20m wide almost certainly bled into
a field), so a reviewer knows to check it rather than trust it blindly.
"""

from __future__ import annotations

_HIGHWAY_TO_IRC = {
    "motorway": "EXP", "motorway_link": "EXP",
    "trunk": "NH", "trunk_link": "NH",
    "primary": "SH", "primary_link": "SH",
    "secondary": "MDR", "secondary_link": "MDR",
    "tertiary": "ODR", "tertiary_link": "ODR",
    "unclassified": "VR", "residential": "VR",
    "track": "TRK",
    "path": "PTH", "footway": "PTH",
}


def classify(highway: str | None, ref: str | None) -> str:
    """OSM ref tags (NH-/SH- numbers) override the highway-tag guess."""
    if isinstance(ref, str):
        r = ref.strip().upper()
        if r.startswith("NH"):
            return "NH"
        if r.startswith("SH"):
            return "SH"
        if r.startswith("MDR"):
            return "MDR"
        if r.startswith("ODR"):
            return "ODR"
    return _HIGHWAY_TO_IRC.get(highway, "TRK")


# rough sanity envelope for an automated width measurement - used only to
# flag an implausible SAM3 mask (e.g. bled into a field, or collapsed to a
# sliver) for review, never to set or override the design value itself
PLAUSIBLE_RANGE_M = {
    "EXP": (7.0, 40.0), "NH": (5.0, 25.0), "SH": (5.0, 25.0),
    "MDR": (3.5, 20.0), "ODR": (2.5, 15.0), "VR": (2.0, 12.0),
    "TRK": (1.5, 8.0), "PTH": (0.8, 4.0),
}


def plausible_range(irc_class: str) -> tuple[float, float]:
    return PLAUSIBLE_RANGE_M.get(irc_class, (2.0, 12.0))
