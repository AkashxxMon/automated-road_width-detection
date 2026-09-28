# DFC Alignment Crossing Finder — Gujarat

Finds every point where a road crosses the proposed DFC rail alignment within Gujarat, decides which ones need a grade-separated structure (no level crossings are permitted on new lines per the Railway Board's Aug 2023 directive), measures the road's width from satellite imagery, and computes the span the structure needs along the track.

Two commands do the whole job:

1. **`crossing_identify.py`** — clip the alignment, fetch the road network, find every crossing, decide which need a structure.
2. **`measure_width_sam3.py`** — measure each crossing's road width with SAM3, and write the final register.

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

```powershell
.venv\Scripts\python.exe src\crossing_identify.py
.venv\Scripts\python.exe src\measure_width_sam3.py
.venv\Scripts\python.exe src\make_review_map.py
```

That's the whole pipeline. Open `output/gujarat_review_map.html` in a browser afterward to see the results on a map.

Both commands take arguments for everything (input files, output paths, every rule threshold) — run either with `--help` to see the full list. With no arguments, they reproduce this project's Gujarat setup exactly, reading from `input/DFCC_master_all_sections_Lines.geojson` and `input/boundaries/gujarat.geojson`.

To point the pipeline at a different alignment/state entirely:

```powershell
.venv\Scripts\python.exe src\crossing_identify.py `
    --alignment my_line.geojson --boundary my_state.geojson --corridor-km 1.5 -o my_crossings.geojson
.venv\Scripts\python.exe src\measure_width_sam3.py `
    --crossings my_crossings.geojson --roads shared\roads_my_line.geojson -o my_crossings_final.geojson
```

Pass `--boundary ""` to run against the whole alignment input, unclipped.

### Optional extras

```powershell
.venv\Scripts\python.exe src\validate.py                     # check the register against a manually-marked ground-truth set, if you have one
.venv\Scripts\python.exe src\make_survey_worksheet.py         # list only the unresolved/SUSPECT crossings, for a manual recheck
.venv\Scripts\python.exe src\measure_width.py                 # SAM2 instead of SAM3 (no gated access needed, kept as a fallback/comparison)
```

---

## 3. How it works, in simple terms

### Step 1 — `crossing_identify.py`: find the crossings

1. **Clip the alignment.** Reads the rail centreline and cuts it down to just the part inside the given boundary (Gujarat, by default).
2. **Fetch the road network.** Downloads every road from OpenStreetMap within a corridor around that alignment (2km either side, by default).
3. **Find every crossing.** Wherever a road physically intersects the alignment, record the point, how far along the alignment it is (`chainage_m`), and the angle between the road and the track (`angle_deg`).
4. **Classify the road.** Each road gets sorted into an IRC category (`NH`/`SH`/`MDR`/`ODR`/`VR`/`TRK`/`PTH`/`EXP`) based on its OpenStreetMap tags. This class **no longer sets the design width** — it's used later only to flag a measurement as suspicious if it doesn't look right for that kind of road.
5. **Decide whether a structure is needed** (rules R1–R8):
   - **R1** — the road connects to the wider network on both sides → **needs a structure**.
   - **R2** — the road just dead-ends at the alignment on one side → **no structure needed**.
   - **R3** — neither side connects to anything → **no structure**, likely a mapping error.
   - **R4** — one side dead-ends close by, ambiguous → **flagged for review**.
   - **R5** — two OSM road segments right next to each other sharing the same route number (a divided highway) → merged into one crossing.
   - **R6** — flagged if within 50m of an existing road junction.
   - **R7** — flagged if the crossing angle is sharp (below 45°) — the structure will need to be much longer than the road is wide.
   - **R8** — flagged if another crossing sits within 40m on the same alignment segment — usually the same road mapped twice, or a genuine parallel track; needs a human decision on whether it's one structure or two.

Output: `output/gujarat_crossings.geojson` — one row per crossing, with a decision but no width yet.

### Step 2 — `measure_width_sam3.py`: measure the road and finish the register

1. **Fetch a satellite image** of each crossing (a small crop, roughly 200×200m) via `src/imagery.py`. The intended, licensed source for this kind of automated analysis is **Esri World Imagery** — that's what the tile URL should point to for a real run. **Currently `imagery.py`'s `TILE_URL` is set to Google's tile server instead**, for an ad hoc manual comparison. Google's terms don't license this kind of automated, bulk use — switch `TILE_URL` back to the Esri endpoint before running this at scale.
2. **Segment the road** using SAM3, prompted with the word "road" — it finds every road-shaped object in the crop, and the one sitting at the known crossing point is picked. If a divided highway shows up as two separate pieces (split by the median), they're merged into one.
3. **Measure the width** on 5 lines drawn across the road (perpendicular to it, at 5 points along its length). If the exact expected point doesn't land on the road in the image (a common mismatch between OpenStreetMap's road position and the true position visible in the imagery), it searches nearby for the actual road before measuring.
4. **Flag a measurement SUSPECT** if the 5 lines disagree with each other too much, or if the result doesn't look plausible for that road's IRC class (from step 1). A SUSPECT measurement is still used — it's just labelled, not thrown away or replaced with anything else.
5. **Compute the final numbers:**
   - `design_width_m` — the road's measured width. Always comes from SAM3; there's no other source.
   - `span_m` — the length the structure needs **along the railway track**: `design_width_m ÷ sin(crossing angle)`. At a square (90°) crossing this equals the road width; at a shallow angle it can be much longer.

Output: `output/gujarat_crossings_final_gl.geojson` (and `.csv`) — the finished register, plus a QA image per crossing in `output/width_crops_sam3_gl/` so every automated measurement can be checked visually.

### The review map

`make_review_map.py` builds an interactive HTML page with the alignment, every crossing colour-coded by decision, and a toggleable Google Satellite layer for visual reference only (never downloaded or processed by any script — just a live map layer for a person to look at, the same as adding it in QGIS by hand).
