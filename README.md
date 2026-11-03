# Railway Alignment Crossing Finder

Finds every point where a road crosses a proposed railway alignment, decides which ones need a grade-separated structure (no level crossings are permitted on new lines per the Railway Board's Aug 2023 directive), measures the road's width from satellite imagery, and decides what structure each one gets — a standard box culvert, a bridge with its span along the track, or no structure because the road can be re-routed.

**This works for any railway alignment, anywhere** — not tied to any specific line or state. Give it a centreline file and (optionally) a boundary polygon for the state/region you care about, and it does the rest. This checkout happens to come with one example alignment and one example state boundary already sitting in `input/` so you can try it immediately with no arguments, but nothing about the tool itself is specific to them.

Three commands do the whole job:

1. **`crossing_identify.py`** — clip the alignment, fetch the road network, find every crossing, decide which need a structure.
2. **`measure_width_sam3.py`** — measure each crossing's road width with SAM3, and write the measured register.
3. **`structure_decide.py`** — merge crossings that are really one road corridor, then decide box culvert / bridge / re-route and size it. Pure rules, no model — rerun it whenever a threshold or the box table changes.

---

## 1. Setup

```powershell
cd D:\Projects\DFC_crossing_identify_1
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` installs a CPU-only `torch` by default. If you have an NVIDIA GPU (strongly recommended — SAM3 is slow on CPU), install the CUDA build instead:

```powershell
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**Get SAM3 access** (one-time, needed before the second command will run):

1. Request access at <https://huggingface.co/facebook/sam3> — this is a manual approval queue from Meta, not instant.
2. Once approved, create a token at <https://huggingface.co/settings/tokens>.
3. Download the weights into this project only:
   ```powershell
   .venv\Scripts\python.exe src\download_sam3.py --token hf_xxxxxxxxxxxxxxxxxxxx
   ```
   This pulls ~6.5GB into `model/sam3/` (kept out of git via `.gitignore`).

---

## 2. Running it

**For your own alignment and region:**

```powershell
.venv\Scripts\python.exe src\crossing_identify.py `
    --alignment my_line.geojson --boundary my_state.geojson --corridor-km 1.5 -o my_crossings.geojson
.venv\Scripts\python.exe src\measure_width_sam3.py `
    --crossings my_crossings.geojson --roads shared\roads_my_line.geojson -o my_crossings_final.geojson
.venv\Scripts\python.exe src\structure_decide.py `
    --register my_crossings_final.geojson --alignment shared\alignment_my_line.geojson -o my_structures.geojson
.venv\Scripts\python.exe src\make_review_map.py
```

`--alignment` is any centreline file geopandas can read; `--boundary` is any polygon to clip it to (a state, a district, a custom AOI). Pass `--boundary ""` to run against the whole alignment input, unclipped. Every command takes arguments for every input file, output path, and rule threshold — run any of them with `--help` for the full list.

**Regions.** Every file is named after the region it's for, so runs for different states never overwrite or get mixed up with each other. `crossing_identify.py --state Maharashtra` writes `output/maharashtra_crossings.geojson`, `shared/roads_maharashtra.geojson`, … with crossing ids `MH-001`, … (the region is the `--state` name, else the `--boundary` file's name, or `--region` to set it yourself). It also remembers that region in `output/last_region.txt`, so the later commands pick up the same files with no arguments; pass `--region gujarat` (or any explicit path) to any of them to work on another one.

```powershell
.venv\Scripts\python.exe src\crossing_identify.py --state Maharashtra
.venv\Scripts\python.exe src\measure_width_sam3.py        # = --region maharashtra
.venv\Scripts\python.exe src\structure_decide.py
.venv\Scripts\python.exe src\make_review_map.py           # output/maharashtra_review_map.html
```

**Another alignment for the same state.** Every file is named after the region, so a second line run as plain `--state Maharashtra` would overwrite the first one's results. Give it its own region with a tag instead — `run_states.ps1` does it for you:

```powershell
.\run_states.ps1 -Alignment input\new_line.geojson -Tag newline -States "Maharashtra","Jharkhand"
```

That writes `output/maharashtra_newline_*` (crossing IDs still `MH-001`…), with its own OSM cache. By hand it's `crossing_identify.py --state Maharashtra --alignment input\new_line.geojson --region maharashtra_newline`, then `--region maharashtra_newline` on the later steps. Any line file geopandas reads works (GeoJSON, KML, shapefile, GeoPackage); without a `feature_type` field every line is used, and without `section_id` the pieces are named `seg1`, `seg2`… `--resume` after the alignment changed reuses only the widths of crossings that are still in the same place. To score it, add the designers' bridges as `validation/Bridges_road_crossings_lines_<region>.geojson` and run `score_validation.py`.

If a file was open in QGIS when a step tried to overwrite it, the step writes a `.tmp` copy beside it instead; the next step reads that newer copy automatically.

**Quick try with no arguments** — every command falls back to the example alignment/boundary already bundled under `input/` in this checkout, so you can run the same commands above with no flags at all and get a working result immediately:

```powershell
.venv\Scripts\python.exe src\crossing_identify.py
.venv\Scripts\python.exe src\measure_width_sam3.py
.venv\Scripts\python.exe src\structure_decide.py
.venv\Scripts\python.exe src\make_review_map.py
```

Open the resulting HTML file in a browser afterward to see the results on a map.

### Optional extras

```powershell
.venv\Scripts\python.exe src\score_validation.py               # TP/FP/FN, precision, recall and F1 of the structure layer against the designers' bridges in validation/ (writes output/<region>_validation_*.csv)
.venv\Scripts\python.exe src\validate.py                     # check the register against a manually-marked ground-truth set, if you have one
.venv\Scripts\python.exe src\make_survey_worksheet.py         # list only the unresolved/SUSPECT crossings, for a manual recheck
.venv\Scripts\python.exe src\measure_width.py                 # SAM2 instead of SAM3 (no gated access needed, kept as a fallback/comparison)
```

---python src/make)

### Web app: Rail-Road Crossing Detection

```powershell
.\run_viewer.ps1          # then open http://127.0.0.1:8000  (help: http://127.0.0.1:8000/help)
```

A local page that runs the same three steps with every option in a form, and shows the run as it happens. **Help** (top of the inputs panel) opens the full documentation (`webapp/help.html`): usage, every input, every rule and its basis. Hovering over any input shows what it does. **Hide** folds the inputs panel away; **Inputs** on the map brings it back.

1. **Alignment** — the default DFCC line, or upload your own (GeoJSON, KML, GeoPackage, zipped shapefile). For a CAD export it lists the layer field so you keep only the alignment line (e.g. `Layer = C-ALIG-DG-LINE`), and for a UTM file it offers the other zones, since CAD exports often carry the wrong one — the dashed preview shows where each puts the line, and which states it crosses. **Start chainage** (m, 3 decimals) is the chainage at the line's first vertex.
2. **Area and run name** — clip to a state or use the whole line, and the run name (`--region`); it warns if that name already has results.
3. **Crossing detection** — corridor, R6, R10, R11, OSM roads.
4. **Nearby and divided-road crossings** — every rule about crossings close together in one place: R5, R8, R9, R12.
5. **Width measurement** — SAM3 options.
6. **Structure decision** — major classes, the S1 minor-road rule, opening buffer and clearances, review-flag thresholds.

Each parameter appears once, labelled with its rule ID. Tick which steps to run (re-running only *Decide structures* with new thresholds takes seconds) and press **Run**. The crossings appear on the map when they're found, with chainage marks every 100 m when zoomed in; each SAM3 crop and width appears as it's measured; the structures replace them coloured by type when they're decided.

The **Crossings** tab under the map is the output table: rows that need someone to look at them (no width, divided road, suspect width, borderline box, road offset, nearby crossing, ...) are highlighted, with the reason in *Needs review*. Right-click a row to zoom to it on the map; right-click a crossing on the map to find its row. **Columns** shows, hides and reorders the columns (or drag a heading); the choice is remembered in the browser. **Downloads** has the structures GeoJSON/CSV, the width layer, the measured register and a zip of everything. **Open a finished run** shows any earlier run without re-running it.

One run at a time — SAM3 needs the GPU to itself. Uploads are kept in `input/uploads/`. The page uses OpenStreetMap and Esri satellite tiles and Leaflet from unpkg, so the browser needs internet; the pipeline itself runs locally.

## 3. How it works, in simple terms

### Step 1 — `crossing_identify.py`: find the crossings

1. **Clip the alignment.** Reads the rail centreline and cuts it down to just the part inside the given boundary polygon.
2. **Fetch the road network.** Loads every road from the local NIC road data (`input/Roads/*.parquet`) within a corridor around that alignment (2km either side, by default), plus **OpenStreetMap roads** (`src/osm_roads.py`, `--no-osm` to skip). OSM comes from Geofabrik's India zone extracts, not the Overpass API: Overpass gives each IP only a couple of query slots and starts refusing after a handful of big queries, while Geofabrik serves each zone as one `.osm.pbf` file (one plain download, refreshed daily, no query limits). The file is downloaded once to `input/osm/` (re-downloaded after 30 days, `--osm-max-age-days`), read locally, and the corridor's roads cached in `shared/osm_roads_<region>.parquet` — delete that to refresh. Drivable roads only (no service roads, footways, paths, construction). An OSM crossing is added only if no NIC crossing on the same alignment piece is within 60 m at a similar angle (±20°) — the buffer that stops one road being counted twice (`--osm-buffer-m`, `--osm-angle-deg`); `road_source` says `nic` or `osm`.
3. **Find every crossing.** Wherever a road physically intersects the alignment, record the point, its chainage (`chainage_m`) and the angle between the road and the track (`angle_deg`). Chainage is measured along the input line in the direction it was drawn, from `--start-chainage-m` at its first vertex (default 0), to 3 decimals; clipping to a state doesn't restart it. Crossing IDs are numbered in chainage order (line by line, in file order), so the number in an ID increases with chainage.
4. **Classify the road.** Each road gets sorted into an IRC category (`NH`/`SH`/`MDR`/`ODR`/`VR`/`TRK`/`PTH`/`EXP`) based on its OpenStreetMap tags. This class **no longer sets the design width** — it's used later only to flag a measurement as suspicious if it doesn't look right for that kind of road.
5. **Decide whether a structure is needed** (rules R1, R2, R5–R8, R10, R11). The road network is joined up at every shared vertex, not just at line ends — NIC road lines run straight through junctions.
   - **R1** — the road is present on both sides of the alignment → **needs a structure**.
   - **R2** — the road just dead-ends at the alignment on one side → **no structure needed**.
   - **R3** — *removed.* It used to drop a crossing as "no structure, likely a mapping error" when neither side connected to anything. In NIC data an unconnected road line is a gap between state datasets, not a missing road, so a road the alignment actually cuts through is never dropped this way — that case now goes to R4.
   - **R4** — *removed.* It used to flag (and, for a minor road, drop) a crossing whose road halves don't reach the wider network. NIC is a merge of state exports with gaps between them, and the designers built at only 17% of those dropped minor roads' crossings but at many of the rest, so a road present on both sides of the alignment is now simply a crossing (R1).
   - **R5** — two road segments right next to each other sharing the same route number (a divided highway, e.g. both `NH48`) → merged into one crossing. Only a real route number counts: most NIC refs are just the class code (`SH`, `ODR`), which every road of that class shares, so those never merge here — divided roads without a route number are merged later from their measured widths (R9 in step 3).
   - **R6** — flagged if within 50m of an existing road junction.
   - **R8** — flagged if another crossing sits within 40m on the same alignment segment — usually the same road mapped twice, or a genuine parallel track; needs a human decision on whether it's one structure or two.
   - **R10** — crossing angle below 15° (`--min-angle-deg`): the road runs along the alignment, not across it → **no crossing**.
   - **R11** — road class in `--drop-classes` (default `TRK`, tracks) → **no crossing**.

   R10 and R11 come from scoring against the designers' bridges: they built at 4% of crossings under 15° and 19% of tracks. Dropped crossings stay in the file with the rule and reason, but aren't measured.

Output: one GeoJSON file, one row per crossing, with a decision but no width yet.

### Step 2 — `measure_width_sam3.py`: measure the road and finish the register

1. **Fetch a satellite image** of each crossing (a small crop, roughly 200×200m) via `src/imagery.py`. The intended, licensed source for this kind of automated analysis is **Esri World Imagery** — that's what the tile URL should point to for a real run. **Currently `imagery.py`'s `TILE_URL` is set to Google's tile server instead**, for an ad hoc manual comparison. Google's terms don't license this kind of automated, bulk use — switch `TILE_URL` back to the Esri endpoint before running this at scale.
2. **Segment the road** using SAM3, prompted with the word "road" — it finds every road-shaped object in the crop, and the one sitting at the known crossing point is picked. Any other road piece within 20 m of it (the other carriageway across a median, a service road) is added to the mask too.
3. **Measure the width** on 5 lines drawn across the road (perpendicular to it, at 5 points along its length). If the exact expected point doesn't land on the road in the image (a common mismatch between OpenStreetMap's road position and the true position visible in the imagery), it searches nearby for the actual road before measuring.
4. **Read the whole road corridor.** Each of those lines keeps walking past the road edge: any further road within 20 m, edge to edge, is recorded too. The result is the layout from one side to the other — road | gap | road — in `road_layout_m`, with `total_width_m` edge to edge and, for exactly two roads, `cw1_width_m` / `median_width_m` / `cw2_width_m`.
5. **Judge the surface — paved or not.** In the same SAM3 pass (the image is encoded once, each extra prompt adds ~0.1 s), the road is also prompted as "asphalt road" and as "dirt road"/"dirt track", at the spot where the width was measured. Asphalt beating dirt by ≥ 0.05 → `paved`, dirt beating asphalt by ≥ 0.15 → `unpaved`, in between → `uncertain` (`surface_sam3`, with the two scores; `--no-surface` to skip). On 54 hand-checked Jharkhand crossings every confident call was right, against 81% for the road data's metalled flag — dusty BT roads and tree cover are what land in "uncertain".
6. **Flag a measurement SUSPECT** if the 5 lines disagree with each other too much, or if the result doesn't look plausible for that road's IRC class (from step 1). A SUSPECT measurement is still used — it's just labelled, not thrown away or replaced with anything else.
7. **`design_width_m`** — the width of the road at the crossing point itself. Always comes from SAM3; there's no other source. The span is not worked out here any more — whether there is a span at all depends on the structure (step 3).

**Speed and stopping safely.** The run is a pipeline, so the GPU isn't left waiting: imagery for the next 16 crossings downloads in 8 background threads, the GPU only runs SAM3, and the width measurement, surface verdict and QA image run in 3 worker threads (`--fetch-workers`, `--prefetch`, `--cpu-workers`). On 40 Jharkhand crossings that took 2 min 20 s instead of 7 min 41 s (about 18 per minute), with identical results. Progress is saved to `<survey-csv>.partial.csv` every 25 crossings (`--checkpoint-every`); if a run is stopped or crashes, `--resume` carries on where it left off. Earlier widths are matched to crossings by location, not crossing ID, so a re-run of step 1 that renumbers the crossings (another start chainage, OSM on or off) never gives a crossing another one's width; progress files from before this change carry no locations and are not reused. Batching several crops through SAM3 at once was tested and left out: on an 8 GB GPU it was no faster beyond 2 crops, ran out of memory at 8, and changed some masks.

Output: the measured register (GeoJSON + CSV), plus a QA image per crossing so every automated measurement can be checked visually (the combined layout is printed on it in cyan).

### Step 3 — `structure_decide.py`: decide and size the structure

First, **R9 — corridor merge.** Crossings on the same alignment segment whose roads are within 20 m of each other, edge to edge, and roughly parallel, are one road corridor: the two carriageways of a divided highway, a service road beside it, or the same road mapped as two lines. They get **one** structure over their combined width; the others are marked `MERGED` with `merged_into` pointing at the one that carries the structure (the highest road class). Widths are combined as the union of where each road sits in the imagery, so a carriageway two crossings both measured is only counted once.

Last, **R12 — thinning.** A minor road within 100 m of a higher-ranked crossing (higher road class, then measured, then wider) is diverted to it: `MERGED`, `merged_into` pointing at it (`--thin-m`, 0 to disable). The designers built at only 16–23% of crossings that close together.

Then, for each structure (the first rule that matches sets `structure_type`; flags add up and never change it):

| Rule | Condition | Result |
|---|---|---|
| S0 | no structure needed (R2) | `NONE` |
| S1 | minor road — village road, street, cart track, path — **narrower than 4 m**, **within 500 m of a major-road crossing** (it's diverted there), or with **no measured width** | `NIL_REROUTE`: re-routed to the nearest structure. Flag `CLASS_CHECK` if it looks too wide for its class |
| | any other minor road | kept (`MINOR_KEPT`) and sized by S4–S6 like a major road, at 4.0 m clearance (`--minor-clearance-m`) |
| S2 | kept road with no width (in practice a major road — every NIC "District Road" counts) | `PENDING_SURVEY` |
| S4 | more than one road (divided per R5, or 2+ roads in the combined layout) | `DUAL_DESIGNER_DECISION`: one box over the whole width, a split twin-cell box (one cell per carriageway, wall in the median) and a bridge are all worked out; `structure_suggested` is the first that fits |
| S5 | one road, opening fits a standard box | `BOX`, smallest standard size — square to the road, so no span |
| S6 | one road, opening wider than any standard box | `BRIDGE`, `span_m = opening ÷ sin(crossing angle)` |

**Why the S1 spacing rule** (`--minor-rule spacing`, the default; `--minor-keep-width-m 4`, `--minor-divert-m 500`): scored against the designers' bridges in Maharashtra, Jharkhand, Chhattisgarh and West Bengal, the surface hardly told which village roads they built for (59% of SAM3-paved roads, 42–45% of unpaved), while width and spacing did: 24% built within 500 m of a major-road crossing, 66% beyond 1 km; 33–38% under 4 m wide, 52–59% at 4–8 m; 17% where we have no width. Switching from the surface rule raised decision accuracy from 0.63 to 0.69 and structure precision from 0.51 to 0.59. A paved village road can now be re-routed. `--minor-rule surface` brings back the earlier rule (keep if paved by SAM3 or the road data; `road_surface` and `surface_source` are still recorded either way). `nearest_major_crossing_m` is the straight-line distance to the nearest major-road crossing in the region.

- `opening_width_m` = combined width + 1 m buffer each side (`--edge-buffer-m`).
- Box sizes come from `config/box_sizes.csv`, taken from the RDSO "DFC Loading" tables (single cell double track, 1–6 m spans; twin cell 4–6 m). A box's height is the smallest standard height clearing 5.5 m (`--clearance-m`, the organisation's major-road clearance), so in practice 6.0 m. The largest standard box is 6 m, so any road with an opening over 6 m gets a bridge.
- Other flags: `SUSPECT_WIDTH`, `BORDERLINE` (opening within 0.5 m of the 6 m box limit), `CORRIDOR_MERGED`, `LAYOUT_APPROX` (some merged crossing has no imagery road edges), `DUAL_MISMATCH` (road data and imagery disagree on divided), `NEAR_JUNCTION` (R6), `NEARBY_CROSSING` (R8, not already merged), `ROAD_OFFSET` (the road in the imagery meets the track more than 10 m from where the road data's line does — check the QA crop).

Output: the structure register (`output/<region>_structures.geojson` + CSV, every attribute) and a geometry layer (`output/<region>_structure_geometry.geojson`) with one line per crossing that has a structure decision of its own (not `MERGED`, not `NONE`): the measured road width, drawn to scale across the road where the road in the imagery meets the track (so it sits at the crossing angle to the track). No span line is drawn: the span is the `span_m` attribute. The layer keeps only what a map needs, copied as they are from the register: `crossing_id`, `measured_width_m`, `Assumed_irc_class`, `road_surface` (paved/unpaved/unknown), `surface_source` (imagery/road data), `skew_angle_deg` (90 − crossing angle, 0 = square), `Assumed_structure_type`, `Assumed_box_size`, `Assumed_opening_width_m`, `Assumed_span_m`, `structure_flags` — everything else is in the register (which also carries the `Assumed_` columns). "Assumed" because they come from imagery and rules, not a survey.

### The review map

`make_review_map.py` builds an interactive HTML page with the alignment, every crossing colour-coded by structure type (or by decision, before `structure_decide.py` has run), and a toggleable Google Satellite layer for visual reference only (never downloaded or processed by any script — just a live map layer for a person to look at, the same as adding it in QGIS by hand).
