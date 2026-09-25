"""Road hierarchy: OSM highway/ref tags -> IRC class -> design roadway width.

IRC:73 roadway widths (plain/rolling terrain, open areas). Cross-check
against the current IRC:73 revision and the project's own spec before
release - see the Sources section of the project brief.
"""

from __future__ import annotations

# name, IRC roadway width in m (None = connectivity rule / measurement decides)
IRC = {
    "EXP": ("Expressway", None),
    "NH": ("National Highway", 12.0),
    "SH": ("State Highway", 12.0),
    "MDR": ("Major District Road", 9.0),
    "ODR": ("Other District Road", 7.5),
    "VR": ("Village Road", 7.5),
    "TRK": ("Cart track / field access", None),
    "PTH": ("Footpath", None),
}

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


def roadway_width(irc_class: str) -> float | None:
    return IRC.get(irc_class, (None, None))[1]


# rough sanity envelope for an automated width measurement, independent of
# the IRC design minimum - used only to flag an implausible SAM2 mask
# (e.g. bled into a field, or collapsed to a sliver), not as a design value
PLAUSIBLE_RANGE_M = {
    "EXP": (7.0, 40.0), "NH": (5.0, 25.0), "SH": (5.0, 25.0),
    "MDR": (3.5, 20.0), "ODR": (2.5, 15.0), "VR": (2.0, 12.0),
    "TRK": (1.5, 8.0), "PTH": (0.8, 4.0),
}


def plausible_range(irc_class: str) -> tuple[float, float]:
    return PLAUSIBLE_RANGE_M.get(irc_class, (2.0, 12.0))
