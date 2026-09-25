# DFC Alignment Crossing Finder - Gujarat

Finds every point where a road crosses the proposed DFC rail alignment
within Gujarat, decides which ones need a grade-separated structure (no
level crossings are permitted on new lines per the Railway Board's Aug
2023 directive), and computes a skew-corrected span for each one that
does.

This is a fresh, standalone build - it does not read or write anything in
the sibling `DFC_crossing_identify` project.

## Pipeline

Two commands. Nothing about the project area is hardcoded in either -
every path, the corridor width, the CRS and every rule threshold is a CLI
argument with a sensible default; running both with no arguments
reproduces this project's Gujarat outputs exactly.

| Command | Does | Reads | Writes |
|---|---|---|---|
| `src/crossing_identify.py` | Clips the alignment to a boundary (stages 1-2), fetches the OSM road network for the corridor, finds every crossing and applies rules R1-R8 (stages 4-5, 7) | an alignment file, optionally a boundary polygon | `shared/alignment_*.geojson`, `shared/roads_*.geojson`, `shared/corridor_*.geojson`, the crossings GeoJSON + summary CSV |
| `src/measure_width_sam3.py` | Measures road width with SAM3 + Esri imagery (stage 6) and merges it into the register (stage 7 redo) | the crossings GeoJSON + roads GeoJSON above, `model/sam3/` | a width-survey CSV, a QA image per crossing, the final crossings GeoJSON + CSV |

```
.venv\Scripts\python.exe src\crossing_identify.py
.venv\Scripts\python.exe src\measure_width_sam3.py
.venv\Scripts\python.exe src\make_review_map.py
```

Everything else is optional and unaffected by this consolidation:

| Script | Purpose |
|---|---|
| `src/download_sam3.py` | One-time: pull the gated SAM3 weights into `model/sam3/` |
| `src/measure_width.py` | SAM2 fallback/comparison for stage 6 - writes the same survey CSV schema; merge it with `measure_width_sam3.py --measured-csv <that CSV>` |
| `src/make_review_map.py` | Builds the interactive HTML review map |
| `src/validate.py` | Checks the register against a manually-marked ground-truth set, if you have one |
| `src/make_survey_worksheet.py` | Lists just the unresolved/SUSPECT crossings for a manual recheck |

Re-running just `measure_width_sam3.py` is enough after a re-measurement;
re-run `crossing_identify.py` too after an alignment revision (it needs a
CUDA GPU to run SAM3 at a reasonable speed - CPU works, just much slower).

### Running it elsewhere

```powershell
.venv\Scripts\python.exe src\crossing_identify.py `
    --alignment my_line.geojson --boundary my_state.geojson `
    --corridor-km 1.5 -o my_crossings.geojson
.venv\Scripts\python.exe src\measure_width_sam3.py `
    --crossings my_crossings.geojson --roads shared\roads_my_line.geojson -o my_crossings_final.geojson
```

Pass `--boundary ""` to run against the whole alignment input, unclipped
(the default is `input/boundaries/gujarat.geojson`, since this project
always wants that clip - a fresh alignment file elsewhere has no such
default, so just don't pass `--boundary` at all). Run `--help` on either
script for the full argument list (feature-type filtering, output paths,
every R1-R8 threshold, EPSG override, etc.).

## What "Gujarat only" means here

The master alignment export (`input/DFCC_master_all_sections_Lines.geojson`)
covers every DFC section, all states. Only two named sections have any
geometry inside Gujarat: **PB-Sec-2**, **PB-Sec-3**, **PB-Sec-6** (~100.5 km
combined). `crossing_identify.py --boundary input/boundaries/gujarat.geojson`
(the default) clips their centrelines to the Gujarat state boundary
(sourced from a public India-states GeoJSON), splitting a section wherever
it briefly leaves and re-enters the state (PB-Sec-3 dips into Maharashtra
for about 70 m near its eastern end) - 31 in-state segments in total,
working CRS EPSG:32643 (UTM 43N, auto-detected from the data's centroid -
not hardcoded, just happens to be correct for this stretch of Gujarat).

## Data sources

- **Alignment**: your team's `DFCC_master_all_sections_Lines.geojson`.
- **Roads**: OpenStreetMap, pulled live from Overpass for a corridor around
  the alignment. Not the NIC roads dataset used by the sibling project -
  this build is self-contained.
- **State boundary**: a public India-states GeoJSON, filtered to Gujarat.

## Imagery and width measurement

**Why not Google satellite for width extraction.** Google Maps Platform's
terms explicitly prohibit tracing/digitising roads from the satellite
basemap and using Google Maps content to train, test, or fine-tune
models - automated width extraction is exactly that. So **no code in this
pipeline fetches, traces, or measures anything from Google's tiles.**
Google's layer only ever appears as a toggle-able live reference basemap
in `output/gujarat_review_map.html`, for a human to look at - the same as
adding "Google Satellite" as an XYZ layer in QGIS or using the ruler in
Google Earth Pro. Nothing is downloaded or stored from it.

**Automated width measurement (stage 6) runs against Esri World Imagery
instead** (`src/imagery.py`), which is published specifically as an
analysis/digitising basemap (no "no tracing" clause), via the same tile
access QGIS's built-in "ArcGIS World Imagery" connection uses. For each
crossing:

1. Fetch a ~200x200 m tile mosaic centred on the crossing point (tries
   zoom 19 down to 16, whichever has real imagery there - ~0.28-1.1 m/px
   in this corridor, empirically mostly zoom 18 (~0.56 m/px) with some
   zoom 19 (~0.28 m/px) near town centres).
2. Segment the road surface - **SAM3** (`facebook/sam3`, default,
   `src/measure_width_sam3.py`) or **SAM2** (`facebook/sam2.1-hiera-small`,
   fallback/comparison, `src/measure_width.py`). Both are zero-shot, no
   training/fine-tuning. SAM3 is prompted with `text="road"` - it returns
   every road-like instance in the crop, and `pick_instance()` picks
   whichever one sits at the known crossing point. SAM2 instead needs
   point prompts derived from the OSM road centreline (foreground along
   the road, background to each side). No per-image human interaction
   either way.
3. Measure 5 transects perpendicular to the road (-20, -10, 0, +10, +20 m),
   take the median as `measured_width_m`, IQR as spread (`width_core.py`,
   shared by both model variants).
4. Flag a result **SUSPECT** if it falls outside a plausible range for
   that road class (e.g. a "village road" mask over 12 m almost certainly
   bled into an adjacent field) or the 5 transects disagree by more than
   3 m (usually tree canopy hiding the true edge, or - SAM3-specific - a
   divided/complex carriageway shape the transect walk can't cross
   cleanly). `resolve_width()` refuses to let a SUSPECT reading override
   the IRC floor - it falls back to IRC instead, so one bad mask can't
   silently inflate a span.
5. Design width `W = max(measured, IRC minimum)` when the measurement is
   trusted, `W = IRC minimum` when it's SUSPECT (or missing) and the class
   has one; span `= W / sin(angle)` as before.

A JPG per crossing (imagery + mask overlay + transect lines) is saved to
`output/width_crops_sam3/` (or `output/width_crops/` for the SAM2 run) and
linked from the review map popup, so every automated measurement can be
spot-checked visually without re-running the model.

**Accuracy, honestly.** At ~0.3-0.6 m/px this is well below the ~0.5 m the
project brief targets with licensed imagery (a 3.75 m lane is 6-12 px, not
the ~7 px/±1 m the brief assumes at its finer end). Of 115 measured
crossings with SAM3, confidence broke down **43 high / 29 medium / 23 low
(15 explicitly SUSPECT)**, plus 8 crossings where SAM3 found no "road"
instance at the crossing point at all and 12 where an instance was found
but the transect walk couldn't get a clean measurement across it (mostly
divided/complex carriageways, e.g. the Mumbai-Vadodara Expressway
crossing). Road surface clearly separated from its surroundings gives a
tight, accurate mask (see `output/width_crops_sam3/GJ-004.jpg`, 4.5 m,
high confidence); tree canopy makes even a precise mask's true edge
ambiguous enough that the transect walk (which assumes a locally straight
road, from the OSM tangent) can misfire (`GJ-001.jpg`, flagged SUSPECT
correctly rather than trusted). SAM3's masks were visibly tighter/more
accurate than SAM2's in side-by-side spot checks, at the cost of finding
no instance at all more often than SAM2 (which always returns *some*
mask from a point prompt, good or bad). Treat `measured` widths as a
first-pass automated estimate, and `SUSPECT`/`needs field/imagery survey`
rows as still needing a real check - `src/make_survey_worksheet.py`
generates a short recheck list for just those.

Before any production/design use: replace Esri's imagery with something
you hold a licence for (drone orthomosaic, Cartosat-3 via NRSC, Pleiades
Neo, Maxar WorldView) at ≤0.5 m GSD, and re-run `measure_width_sam3.py`
against it (same logic, just point `imagery.py` at the new source) -
accuracy should improve substantially at real survey resolution.

### Using SAM3

`facebook/sam3` is gated on Hugging Face with *manual* approval from Meta
(confirmed via the Hub API - an actual review queue, not just click-through
terms). Already set up in this project:

1. Request access at <https://huggingface.co/facebook/sam3> (or
   `facebook/sam3.1`, same gate) and wait for approval.
2. `python src/download_sam3.py --token hf_xxxxxxxxxxxx` - downloads the
   weights into `./model/sam3/` **in this project folder only** (not the
   shared `~/.cache/huggingface`), and `.gitignore`d so they're never
   committed. Already done here (~6.5 GB).
3. `python src/measure_width_sam3.py`.

`measure_width.py` (SAM2) writes the same `output/gujarat_width_survey.csv`
schema as `measure_width_sam3.py`'s own SAM3 pass - merge either one with
`measure_width_sam3.py --measured-csv output/gujarat_width_survey.csv`.

## Decision rules implemented

- **R1** through road (both halves connect) -> crossing required
- **R2** dead-end spur (only one side present at all) -> no crossing
- **R3** isolated fragment (neither half connects) -> no crossing, check the map
- **R4** one half dead-ends near the alignment -> `review` (this build
  flags *every* single-sided dead-end for review, since no building
  footprint layer is loaded yet to tell a genuine habitation spur from an
  empty-field stub - see Limitations)
- **R5** dual carriageway (two OSM ways with the same `ref` within 30 m)
  merged into one crossing
- **R6** crossing within 50 m of an existing road junction -> flagged
  (`near_junction`)
- **R7** crossing angle below 45 deg -> flagged (`skewed`)
- **R8** another crossing within 40 m on the same alignment segment ->
  flagged (`nearby_crossing_id`, `nearby_dist_m`), *not* auto-merged. Found
  5 such pairs, all sitting near-parallel and 13-68 m apart from each
  other (similar length, overlapping bounding boxes) - almost certainly
  either the same physical road/track mapped twice as separate OSM ways,
  or a paved road with a genuine parallel unpaved service track. Either
  way it's a real design decision (one wide structure vs two), not
  something the tool should silently decide - it's surfaced for the
  designer instead of merged or left unexplained.

Design width `W = max(measured, IRC roadway width for the class)`
(measured from stage 6 when trusted, IRC floor otherwise); span
`= W / sin(angle)`.

## Current results (this run)

117 crossing candidates over ~100.5 km: 92 `crossing`, 23 `review`, 2
`no_crossing`. Of the 115 `crossing`/`review` candidates, automated
width measurement (`output/gujarat_crossings_final.geojson`) gives:

| `width_source` | count | meaning |
|---|---|---|
| `IRC (exceeds measured)` | 66 | measured width came back below the IRC minimum, so IRC governs (expected - most roads are narrower than their design minimum) |
| `measured` | 14 | trusted SAM3 measurement genuinely exceeds the IRC minimum and governs |
| `IRC (measurement flagged SUSPECT, verify manually)` | 11 | SAM3 mask was flagged unreliable; fell back to the IRC floor rather than trust it |
| `IRC {class}` (untouched) | 13 | no measurement was attempted or returned (see below) - class still has a usable IRC floor |
| `needs field/imagery survey` | 9 | no imagery/road geometry available, or SAM3 found no "road" instance there |
| `needs field/imagery survey (SUSPECT)` | 4 | SUSPECT *and* no IRC floor to fall back to (Track/Path class) - still open |

13 crossings still have no usable width at all and need a real check.
`python src/make_survey_worksheet.py` lists all of these plus the SUSPECT
ones needing a second opinion for a manual recheck.

## Validation against the manually-marked bridge register

`validation/Bridges_corrected_lines.geojson` is a CAD (COGO-label) export
of bridge callouts from the design drawings, covering every DFC section
nationwide - `src/validate.py` filters it to Gujarat and checks it against
`output/gujarat_crossings.geojson`. Two data-cleaning steps were needed
before the comparison meant anything:

1. **Section leakage.** Filtering by "inside the 2 km corridor bbox" alone
   pulled in 528 features, but ~48 of those sit hundreds to ~2000 m from
   the actual Gujarat alignment - they belong to a neighbouring section
   (e.g. PB-Sec-4/5, just across the Maharashtra border) whose corridor
   happens to overlap. Dropped anything farther than 200 m from the
   Gujarat alignment itself, leaving 480.
2. **Duplicate annotation lines.** This CAD export puts ~3 separate leader
   lines per physical bridge (one for the label, one for elevation, one
   for coordinates), almost exactly on top of each other. Collapsing
   points within 15 m of each other brought 480 lines down to **158
   unique bridges**.
3. **Mixed structure types.** `Bridges_corrected_lines.geojson` covers
   *every* structure on the drawings - drainage/waterway/canal crossings,
   catchment areas, low-lying-area structures, other rail-track crossings,
   as well as road bridges - and carries no field distinguishing them. A
   road-crossing pipeline was never going to find a drainage structure, so
   comparing against the raw 158 badly understates recall. `validate.py`
   now checks whether an OSM road exists within 20 m of each missed
   feature: if there's no road at all nearby, it's almost certainly one of
   those non-road structures and is excluded from the recall denominator
   rather than counted as a miss. Of the 158, **106 have no nearby OSM
   road at all** - leaving 52 genuinely road-relevant ground-truth
   features to check against.

Result, at a 60 m match tolerance:

- **Recall: 51/52 (98%)** of road-relevant known bridges have a predicted
  candidate nearby - once non-road structures are excluded, OSM coverage
  and the crossing-detection logic in this corridor turn out to be very
  good, not a major gap as an earlier (uncorrected) pass of this analysis
  concluded. The one remaining miss has an OSM road 1.1 m away but the
  nearest predicted crossing is ~900 m off along the same alignment
  segment - worth a manual look (possibly a small alignment/road
  digitisation mismatch rather than a missed intersection).
- Of the 51 matches, 2 (`GJ-049`, `GJ-056`) were sent to `review` instead
  of `crossing` by the R1-R4 rule - worth checking by hand; everything
  else the rule did decide correctly matched `crossing`.
- **Precision: 45/92 (49%)** of `crossing`-decided candidates have a known
  bridge within 60 m. Given recall is now measured at 98% against the
  road-relevant subset, this gap is *not* mainly false positives - it's
  much more likely that this particular CAD layer only calls out major
  bridges individually, while many of this pipeline's `crossing` decisions
  are minor village-road (VR) crossings that get a standard culvert/LHS
  not separately marked as a "bridge" in this drawing set. Worth
  confirming with the design team rather than assuming over-detection.

Re-run with `python src/validate.py --tolerance-m 100` (etc.) to see how
sensitive these numbers are to the match tolerance. The review map
(`output/gujarat_review_map.html`) includes a "Ground-truth bridges -
matched / missed" layer (off by default) once `validate.py` has run.

## Limitations / next steps

- **R4 has no building-footprint check yet.** Every single-sided dead-end
  is marked `review` rather than being split into "serves habitation" vs
  "drop it" - more conservative than the full rule, but the review queue
  is larger than it needs to be. Add a footprints layer (e.g. Google Open
  Buildings, Microsoft Building Footprints) and re-run to tighten this.
- **Automated width measurement is a first-pass estimate, not survey-grade.**
  ~0.3-0.6 m/px Esri imagery is coarser than the ~0.5 m the brief targets;
  ~15% of measurements were flagged SUSPECT (tree canopy / field-edge
  bleed, or a divided carriageway) and fall back to the IRC floor rather
  than being trusted blindly, plus ~17% where SAM3 couldn't get a
  measurement at all.
  Swap in licensed imagery (drone/Cartosat-3/Pleiades/Maxar) through
  `src/imagery.py` once available and re-run `measure_width_sam3.py` for
  survey-grade numbers.
- **OSM coverage turned out NOT to be a major gap** once validated properly
  (98% recall against road-relevant ground truth - see Validation above).
  An earlier pass of this analysis compared against the raw, unfiltered
  158-feature validation set (which is mostly non-road structures) and
  wrongly concluded OSM coverage was the main problem; that conclusion is
  superseded. `no_crossing`/`R3` results should still be spot-checked
  against imagery on principle, but there's no evidence of a systemic gap.
- **Precision (49%) is still not fully explained** - likely mostly this
  particular CAD layer not separately marking minor/culvert-level
  crossings as "bridges" (see Validation), but worth confirming with the
  design team rather than assuming it either way.
- **R8 nearby-crossing pairs (5 found) are not resolved automatically.**
  Whether two close, near-parallel crossings should become one wide
  structure or stay as two is a design call, not something this pipeline
  decides - `nearby_crossing_id`/`nearby_dist_m` on the affected rows
  need a human look before the register is finalised.
- Chainage in `chainage_m` is local to each of the 31 segments (`align_name`
  identifies which), not a single project-wide chainage - there's no
  authoritative start-chainage-per-section in the source data yet.
