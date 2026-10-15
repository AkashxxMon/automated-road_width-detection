"""One command: measure road width at every crossing using SAM3 segmentation
on Esri World Imagery, and merge the results straight into the crossing
register - populating design_width_m/width_source (plus the combined
layout of every road within 20 m of it - the other carriageway of a
divided road, a service road - where the imagery shows one), saving a QA
image per crossing, and writing the raw measurement data to a CSV.
Box/bridge call, box size and span along track are the next step,
structure_decide.py.

SAM3 (facebook/sam3, "Promptable Concept Segmentation") is prompted with
a text concept (default "road") and returns every matching instance in
the crop; pick_instance() picks whichever one sits at the known crossing
point. The OSM road tangent (from the roads file crossing_identify.py
already wrote) is only used to orient the 5 perpendicular transects, not
for prompting. No per-image human interaction.

Design width comes entirely from this measurement - there is no
IRC minimum floor. The IRC class (from crossing_identify.py) is used only
to flag a measurement as SUSPECT when it's implausible for that class of
road (or the 5 transects disagree too much) - see resolve_width() - so a
reviewer knows to check it, without the pipeline silently substituting or
capping the number.

Needs local weights in ./model/sam3 - see src/download_sam3.py.
facebook/sam3 is gated with manual approval from Meta; weights are kept
in this project folder only, not the shared Hugging Face cache.

Usage:
    python src/measure_width_sam3.py
    python src/measure_width_sam3.py --crossings my_crossings.geojson --roads my_roads.geojson -o my_crossings_final.geojson
    python src/measure_width_sam3.py --measured-csv output/gujarat_width_recheck.csv   # re-merge a manual recheck, skip SAM3
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter

from geo_utils import fill_region_paths, get_utm_epsg, latest, write_csv, write_geojson
from imagery import fetch_crop
from irc_classify import plausible_range
from width_core import (MAX_ROAD_GAP_M, combine_roads, measure_transects_detailed, road_layout_text,
                        road_tangent_bearing, save_qa_image, write_width_geometry)

# every input/output path defaults from the region (geo_utils.region_paths): --region, else the one
# crossing_identify.py last ran for - any path passed explicitly wins
REGION_PATH_ARGS = {"crossings": "crossings", "roads": "roads", "alignment": "alignment", "output": "final",
                    "csv_output": "final_csv", "survey_csv": "survey_csv", "images_dir": "images_dir",
                    "width_geometry_output": "width_geometry"}
DEFAULT_MODEL_DIR = "model/sam3"

DETECT_THRESHOLD = 0.4  # SAM3's own example uses 0.5; slightly looser since a small crop has few distractors

# road surface from SAM3 alone: the confidence of the "asphalt road" instance at the measured road
# against the best "dirt road"/"dirt track" instance there. The dirt prompts also fire on dusty paved
# roads, so the thresholds are lopsided: on 54 hand-checked Jharkhand crossings, asphalt - dirt >= 0.05
# was always paved and <= -0.15 always unpaved (41 of 54 decided, no errors); in between: uncertain
SURFACE_PROMPTS = {"asphalt road": "paved", "dirt road": "unpaved", "dirt track": "unpaved"}
SURFACE_SCORE_THRESHOLD = 0.2  # keep weak instances - a low score is information here


def load_model(model_dir: Path, device: str):
    if not model_dir.exists() or not any(model_dir.glob("*.safetensors")):
        raise SystemExit(
            f"{model_dir} has no weights yet. Run:\n"
            f"  python src/download_sam3.py --token hf_xxxxxxxxxxxx\n"
            f"(needs Meta's manual approval for facebook/sam3 first - see that script's docstring)"
        )
    from transformers import Sam3Model, Sam3Processor
    model = Sam3Model.from_pretrained(str(model_dir)).to(device).eval()
    processor = Sam3Processor.from_pretrained(str(model_dir))
    return model, processor


MERGE_NEARBY_M = MAX_ROAD_GAP_M  # union in any other SAM3 instance within this (20 m) - the other
                                 # carriageway across a median, a service road - so the transect walk
                                 # sees the whole road corridor (width_core.combine_roads)


def _dilate(mask: np.ndarray, kernel_px: int) -> np.ndarray:
    """Morphological dilation via PIL's MaxFilter (no scipy dependency)."""
    if kernel_px < 3:
        return mask
    if kernel_px % 2 == 0:
        kernel_px += 1
    img = Image.fromarray((mask.astype(np.uint8) * 255))
    dilated = img.filter(ImageFilter.MaxFilter(kernel_px))
    return np.asarray(dilated) > 0


def pick_instance(results: dict, center_xy: tuple[float, float], crop_area: int, mpp: float):
    """Among SAM3's detected instances, pick the one at the known crossing
    point - by mask containment first, else by nearest box centre. Then
    union in any OTHER instance whose mask is within MERGE_NEARBY_M of the
    picked one - a divided highway is often detected as 2+ separate
    instances (one per carriageway, split by the median), and using only
    one badly under-measures the true design width."""
    masks = results.get("masks")
    boxes = results.get("boxes")
    scores = results.get("scores")
    if masks is None or len(masks) == 0:
        return None
    cx, cy = center_xy

    def as_bool(m):
        m = m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)
        return m.astype(bool)

    all_masks = [as_bool(masks[i]) for i in range(len(masks))]

    containing = []
    for i, m in enumerate(all_masks):
        if m.shape[0] > cy >= 0 and m.shape[1] > cx >= 0 and m[int(cy), int(cx)]:
            if m.sum() / crop_area <= 0.6:
                containing.append((i, float(scores[i]) if scores is not None else 0.0))

    if containing:
        best_i = max(containing, key=lambda t: t[1])[0]
    elif boxes is not None:
        best_i, best_d = None, float("inf")
        for i in range(len(boxes)):
            x0, y0, x1, y1 = [float(v) for v in boxes[i]]
            bcx, bcy = (x0 + x1) / 2, (y0 + y1) / 2
            d = (bcx - cx) ** 2 + (bcy - cy) ** 2
            if d < best_d:
                best_d, best_i = d, i
    else:
        return None

    combined = all_masks[best_i].copy()
    kernel_px = int(round(MERGE_NEARBY_M / mpp))
    dilated = _dilate(combined, kernel_px)
    for i, m in enumerate(all_masks):
        if i != best_i and (m & dilated).any() and m.sum() / crop_area <= 0.6:
            combined |= m
    return combined


def _to_numpy_results(res: dict) -> dict:
    """SAM3's post-processed instances as plain numpy/python (masks as bool
    arrays) - done on the GPU thread so the CPU workers never touch a tensor."""
    masks, scores, boxes = res.get("masks"), res.get("scores"), res.get("boxes")
    if masks is None or len(masks) == 0:
        return {"masks": [], "scores": [], "boxes": []}
    return {
        "masks": [masks[i].detach().cpu().numpy().astype(bool) for i in range(len(masks))],
        "scores": [float(x) for x in scores],
        "boxes": [[float(v) for v in boxes[i]] for i in range(len(boxes))] if boxes is not None else None,
    }


def _score_near(results: dict, xy: tuple[float, float], r_px: float) -> float:
    """Best score among instances whose mask reaches within r_px of xy, else 0."""
    masks, scores = results.get("masks"), results.get("scores")
    if masks is None or len(masks) == 0:
        return 0.0
    x0, x1 = int(max(xy[0] - r_px, 0)), int(xy[0] + r_px) + 1
    y0, y1 = int(max(xy[1] - r_px, 0)), int(xy[1] + r_px) + 1
    best = 0.0
    for i in range(len(masks)):
        if np.asarray(masks[i])[y0:y1, x0:x1].astype(bool).any():
            best = max(best, float(scores[i]))
    return best


def run_gpu(model, processor, rgb: np.ndarray, args) -> dict:
    """Everything SAM3 has to say about one crop: the road instances, and (if
    asked) the instances for each surface prompt, reusing the crop's vision
    features. Returned as plain numpy so the rest runs off the GPU thread."""
    inputs = processor(images=Image.fromarray(rgb), text=args.text, return_tensors="pt").to(model.device)
    target_sizes = inputs.get("original_sizes").tolist()
    with torch.no_grad():
        vision = model.get_vision_features(pixel_values=inputs["pixel_values"])
        outputs = model(vision_embeds=vision, input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        road = _to_numpy_results(processor.post_process_instance_segmentation(
            outputs, threshold=args.detect_threshold, mask_threshold=0.5, target_sizes=target_sizes)[0])
        surface = {}
        if args.surface:
            for prompt in SURFACE_PROMPTS:
                txt = processor(text=prompt, return_tensors="pt").to(model.device)
                out = model(vision_embeds=vision, input_ids=txt["input_ids"], attention_mask=txt["attention_mask"])
                surface[prompt] = _to_numpy_results(processor.post_process_instance_segmentation(
                    out, threshold=SURFACE_SCORE_THRESHOLD, mask_threshold=0.5, target_sizes=target_sizes)[0])
    return {"road": road, "surface": surface}


def classify_surface(surface_results: dict, xy: tuple[float, float], r_px: float,
                     paved_margin: float, unpaved_margin: float) -> dict:
    """paved / unpaved / uncertain for the road at xy, from SAM3's surface
    prompt instances (run_gpu)."""
    scores = {prompt: _score_near(res, xy, r_px) for prompt, res in surface_results.items()}
    paved = max(v for k, v in scores.items() if SURFACE_PROMPTS[k] == "paved")
    unpaved = max(v for k, v in scores.items() if SURFACE_PROMPTS[k] == "unpaved")
    gap = paved - unpaved
    verdict = "paved" if gap >= paved_margin else "unpaved" if gap <= -unpaved_margin else "uncertain"
    return {"surface_sam3": verdict, "surface_paved_score": round(paved, 3), "surface_unpaved_score": round(unpaved, 3)}


def finish_crossing(job: dict, crop: tuple, gpu: dict, args) -> tuple[dict, str]:
    """The CPU half of one crossing, off the GPU thread: pick the road mask,
    measure the width, combine neighbouring roads, judge the surface from the
    saved surface instances, write the QA image. Returns (survey row, log line)."""
    cid, row, pt_utm, bearing = job["cid"], job["row"], job["pt_utm"], job["bearing"]
    rgb, mpp, zoom = crop
    h, w = rgb.shape[:2]
    center_px = (w / 2.0, h / 2.0)

    mask = pick_instance(gpu["road"], center_px, h * w, mpp)
    detailed = measure_transects_detailed(mask, mpp, bearing, center_px) if mask is not None else []
    widths = [d["width_m"] for d in detailed]
    # other roads within MAX_ROAD_GAP_M (the other carriageway, a service road) - kept alongside
    # the single-road width, not folded into it: structure_decide.py spans the whole layout
    combo = combine_roads(detailed) if widths else None
    combo_label = (f"combined: {road_layout_text(combo['widths_m'], combo['gaps_m'])} = {combo['total_m']:.1f} m"
                   if combo else None)
    surface = {}
    if args.surface:
        # judged where the road actually is: the middle of the transect the width comes from
        road_xy, road_r = center_px, 12.0 / mpp
        if detailed:
            t = min(detailed, key=lambda d: abs(d["width_m"] - float(np.median(widths))))
            road_xy = ((t["p1_px"][0] + t["p2_px"][0]) / 2, (t["p1_px"][1] + t["p2_px"][1]) / 2)
            road_r = max(6.0 / mpp, 2.0)
        surface = classify_surface(gpu["surface"], road_xy, road_r,
                                   args.surface_paved_margin, args.surface_unpaved_margin)
    surface_label = (f"surface: {surface['surface_sam3']} (asphalt {surface['surface_paved_score']:.2f} / "
                     f"dirt {surface['surface_unpaved_score']:.2f})" if surface else None)
    save_qa_image(Path(args.images_dir) / f"{cid}.jpg", rgb, mask, bearing, mpp, center_px, widths,
                  extra_label=" | ".join(x for x in (combo_label, surface_label) if x) or None)

    if widths:
        med = float(np.median(widths))
        iqr = float(np.percentile(widths, 75) - np.percentile(widths, 25)) if len(widths) > 1 else 0.0
        conf = "high" if len(widths) >= 4 and iqr < 1.0 else "medium" if len(widths) >= 3 else "low"
        lo, hi = plausible_range(row["irc_class"])
        suspect = not (lo <= med <= hi) or iqr > 3.0
        if suspect:
            conf = "low"
        flag = " SUSPECT" if suspect else ""
        flag += f" ({combo_label})" if combo else ""
        flag += f" [{surface_label}]" if surface else ""
        log = f"  {cid}: {med:.1f} m (n={len(widths)}, iqr={iqr:.2f}, conf={conf}, zoom={zoom}){flag}"

        def to_utm(px):
            return pt_utm.x + (px[0] - center_px[0]) * mpp, pt_utm.y - (px[1] - center_px[1]) * mpp

        combo_cols = {"n_roads": len(combo["widths_m"]) if combo else 1}
        if combo:
            (ext_p1_x, ext_p1_y), (ext_p2_x, ext_p2_y) = to_utm(combo["e1_px"]), to_utm(combo["e2_px"])
            combo_cols.update({
                "road_layout_m": road_layout_text(combo["widths_m"], combo["gaps_m"]),
                "total_width_m": round(combo["total_m"], 2),
                "ext_p1_x": ext_p1_x, "ext_p1_y": ext_p1_y, "ext_p2_x": ext_p2_x, "ext_p2_y": ext_p2_y,
            })
            if len(combo["widths_m"]) == 2:
                combo_cols.update({"cw1_width_m": round(combo["widths_m"][0], 2),
                                   "median_width_m": round(combo["gaps_m"][0], 2),
                                   "cw2_width_m": round(combo["widths_m"][1], 2)})

        # the transect whose width is closest to the reported median is the one whose
        # edge pixels become the width-line geometry - not necessarily the geographically
        # centre (offset=0) transect, whose own width can disagree with the reported median
        center_transect = min(detailed, key=lambda d: abs(d["width_m"] - med))
        p1x, p1y = center_transect["p1_px"]
        p2x, p2y = center_transect["p2_px"]
        width_p1_x, width_p1_y = to_utm((p1x, p1y))
        width_p2_x, width_p2_y = to_utm((p2x, p2y))
        out = {"crossing_id": cid, "measured_width_m": round(med, 2),
               "measured_by": f"SAM3({args.text})+EsriWorldImagery-auto(z{zoom})",
               "notes": f"n={len(widths)} transects, IQR={iqr:.2f} m, confidence={conf}"
                        + (" - SUSPECT: mask likely over-extended (tree cover/shadow), check QA crop" if suspect else ""),
               "width_iqr_m": round(iqr, 2), "n_transects": len(widths), **combo_cols, **surface,
               "width_p1_x": width_p1_x, "width_p1_y": width_p1_y,
               "width_p2_x": width_p2_x, "width_p2_y": width_p2_y}
    elif mask is None:
        log = f"  {cid}: no '{args.text}' instance detected at the crossing point"
        out = {"crossing_id": cid, "measured_width_m": None, "measured_by": None, **surface,
               "notes": f"SAM3 found no '{args.text}' instance at the crossing point - check QA crop"}
    else:
        log = f"  {cid}: instance found but all transects failed (likely a divided/complex carriageway shape)"
        out = {"crossing_id": cid, "measured_width_m": None, "measured_by": None, **surface,
               "notes": "SAM3 detected a road instance but the transect walk found no clean crossing - "
                        "check QA crop (often a divided highway, median gap in the mask)"}
    return out, log


def measure_all(cross: gpd.GeoDataFrame, roads: gpd.GeoDataFrame, epsg: int, args) -> pd.DataFrame:
    """A three-stage pipeline so the GPU is never waiting on anything else:
      1. imagery for the next crossings downloads in background threads (the network was
         most of the time per crossing, with the GPU idle meanwhile);
      2. this thread feeds the crops to SAM3 one after another - the only GPU work;
      3. the width measurement, surface verdict and QA image of finished crops run in a
         CPU worker pool while the GPU moves on to the next crop.
    Results are kept in crossing order; progress is saved to <survey>.partial.csv as it
    goes, and --resume picks it back up."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading SAM3 from {args.model_dir} on {device} ...")
    model, processor = load_model(Path(args.model_dir), device)

    roads = roads.to_crs(epsg)
    roads_sindex = roads.sindex
    to_measure = cross[cross["decision"] != "no_crossing"].reset_index(drop=True)
    if args.limit:
        to_measure = to_measure.head(args.limit)
    cross_utm = to_measure.to_crs(epsg)

    partial_path = Path(args.survey_csv).with_suffix(".partial.csv")
    done: dict[str, dict] = {}
    if args.resume:
        for src in (partial_path, Path(args.survey_csv)):
            if src.exists():
                prev = pd.read_csv(src)
                done = {r["crossing_id"]: {k: v for k, v in r.items() if pd.notna(v)} for r in prev.to_dict("records")}
                print(f"resuming: {len(done)} crossing(s) already measured in {src.name}")
                break

    jobs = []
    for i, row in to_measure.iterrows():
        cid = row["crossing_id"]
        if cid in done:
            continue
        pt_utm = cross_utm.geometry.iloc[i]
        jobs.append({"i": i, "cid": cid, "row": row, "pt_utm": pt_utm,
                     "bearing": road_tangent_bearing(roads_sindex, roads, pt_utm)})
    print(f"measuring {len(jobs)} crossing(s) with text prompt {args.text!r} "
          f"({args.fetch_workers} download + {args.cpu_workers} measure worker(s), prefetch {args.prefetch}) ...")

    results: dict[str, dict] = dict(done)
    t0 = time.time()
    n_done = 0

    def checkpoint(force: bool = False):
        if force or (len(results) % args.checkpoint_every == 0):
            write_csv(pd.DataFrame(list(results.values())), partial_path)

    def record(job, out, log):
        nonlocal n_done
        results[job["cid"]] = out
        n_done += 1
        rate = n_done / max(time.time() - t0, 1e-6) * 60
        print(f"{log}  [{n_done}/{len(jobs)}, {rate:.0f}/min]", flush=True)
        checkpoint()

    io_pool = ThreadPoolExecutor(max_workers=args.fetch_workers, thread_name_prefix="fetch")
    cpu_pool = ThreadPoolExecutor(max_workers=args.cpu_workers, thread_name_prefix="measure")
    pending_fetch: deque = deque()   # (job, future) in crossing order, at most `prefetch` ahead
    pending_cpu: deque = deque()     # (job, future), oldest first
    it = iter(jobs)

    def top_up():
        while len(pending_fetch) < args.prefetch:
            job = next(it, None)
            if job is None:
                return
            if job["bearing"] is None:
                pending_fetch.append((job, None))
            else:
                pending_fetch.append((job, io_pool.submit(fetch_crop, job["row"].geometry.x, job["row"].geometry.y,
                                                          args.half_extent_m)))

    def drain(limit: int):
        """Collect finished CPU work (oldest first) until at most `limit` is outstanding."""
        while pending_cpu and (len(pending_cpu) > limit or pending_cpu[0][1].done()):
            job, fut = pending_cpu.popleft()
            out, log = fut.result()
            record(job, out, log)

    try:
        top_up()
        while pending_fetch:
            job, fut = pending_fetch.popleft()
            top_up()
            crop = fut.result() if fut is not None else None
            if crop is None:
                record(job, {"crossing_id": job["cid"], "measured_width_m": None, "measured_by": None,
                             "notes": "no imagery or road geometry available"},
                       f"  {job['cid']}: skip (no road geometry or no imagery available)")
                continue
            gpu = run_gpu(model, processor, crop[0], args)
            pending_cpu.append((job, cpu_pool.submit(finish_crossing, job, crop, gpu, args)))
            drain(limit=args.cpu_workers * 3)  # bounds the masks held in memory
        drain(limit=0)
    finally:
        io_pool.shutdown(wait=False, cancel_futures=True)
        cpu_pool.shutdown(wait=True)
        if results:
            checkpoint(force=True)

    order = {cid: n for n, cid in enumerate(to_measure["crossing_id"])}
    df = pd.DataFrame(sorted(results.values(), key=lambda r: order.get(r["crossing_id"], 10**9)))
    elapsed = time.time() - t0
    print(f"\n{len(jobs)} crossing(s) in {elapsed / 60:.1f} min ({len(jobs) / max(elapsed, 1e-6) * 60:.0f}/min)")
    if partial_path.exists():
        partial_path.unlink()
    return df

def resolve_width(row):
    """design_width_m comes from the SAM3 measurement alone - no IRC floor,
    no substitution. A SUSPECT measurement is still used, just labelled for
    review rather than trusted silently; there's no other number to fall
    back to. span_m is left for structure_decide.py, since whether a span
    applies at all depends on the box/bridge call (a box is always square)
    and the skew gate."""
    if pd.isna(row["measured_width_m"]):
        return None, "needs field/imagery survey"
    source = "measured (SUSPECT - verify manually)" if "SUSPECT" in str(row["notes"]) else "measured"
    return row["measured_width_m"], source

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", default=None,
                     help="Region whose files to use (e.g. maharashtra, as crossing_identify.py printed). Default: the "
                          "region crossing_identify.py last ran for. Any path below passed explicitly overrides it")
    ap.add_argument("--crossings", default=None, help="Input crossings GeoJSON. Default: output/<region>_crossings.geojson")
    ap.add_argument("--roads", default=None,
                     help="Roads GeoJSON - used for the transect bearing. Default: shared/roads_<region>.geojson")
    ap.add_argument("--alignment", default=None,
                     help="Clipped alignment GeoJSON - used to draw span lines. Default: shared/alignment_<region>.geojson")
    ap.add_argument("--epsg", type=int, default=None, help="Metric CRS for measurement. Default: auto-detect UTM zone")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Local SAM3 weights directory")
    ap.add_argument("--text", default="road", help="Concept prompt passed to SAM3")
    ap.add_argument("--half-extent-m", type=float, default=100.0, help="Half-width of the imagery crop fetched per crossing")
    ap.add_argument("--detect-threshold", type=float, default=DETECT_THRESHOLD, help="SAM3 instance-detection confidence threshold")
    ap.add_argument("--surface", action=argparse.BooleanOptionalAction, default=True,
                     help="Also judge paved/unpaved from SAM3 ('asphalt road' vs 'dirt road'/'dirt track' at the road)")
    ap.add_argument("--surface-paved-margin", type=float, default=0.05,
                     help="Call paved when the asphalt score beats the dirt score by at least this")
    ap.add_argument("--surface-unpaved-margin", type=float, default=0.15,
                     help="Call unpaved when the dirt score beats the asphalt score by at least this; between -> uncertain")
    ap.add_argument("--limit", type=int, default=None, help="Only measure the first N crossings (for testing)")
    ap.add_argument("--fetch-workers", type=int, default=8,
                     help="Threads downloading imagery ahead of the GPU (lower it if the tile server throttles)")
    ap.add_argument("--prefetch", type=int, default=16, help="How many crops to download ahead of the GPU")
    ap.add_argument("--cpu-workers", type=int, default=3,
                     help="Threads doing the width measurement / QA images while the GPU runs the next crop")
    ap.add_argument("--checkpoint-every", type=int, default=25,
                     help="Save progress to <survey-csv>.partial.csv every N crossings")
    ap.add_argument("--resume", action="store_true",
                     help="Skip crossings already in <survey-csv>.partial.csv (a stopped run) or, failing that, <survey-csv>")
    ap.add_argument("--measured-csv", default=None,
                     help="Skip SAM3 entirely and merge this CSV instead (crossing_id, measured_width_m[, measured_by, notes, "
                          "cw1_width_m, median_width_m, cw2_width_m, total_width_m]) "
                          "- e.g. a manual recheck from make_survey_worksheet.py")
    ap.add_argument("-o", "--output", default=None,
                     help="Output final crossings GeoJSON. Default: output/<region>_crossings_final_gl.geojson")
    ap.add_argument("--csv-output", default=None, help="Output final crossings CSV")
    ap.add_argument("--survey-csv", default=None, help="Where to save the raw SAM3 measurement data")
    ap.add_argument("--images-dir", default=None, help="Directory for the per-crossing QA images")
    ap.add_argument("--width-geometry-output", default=None,
                     help="Output GeoJSON: the measured width lines, drawn to scale (span lines are added by structure_decide.py)")
    args = ap.parse_args()
    fill_region_paths(args, REGION_PATH_ARGS)
    return args

def main() -> None:
    args = parse_args()

    crossings_path = latest(args.crossings)
    if not crossings_path.exists():
        raise SystemExit(f"{args.crossings} not found - run src/crossing_identify.py first "
                         f"(or pass --region / --crossings for the region you ran it for)")
    print(f"crossings: {crossings_path}")
    cross = gpd.read_file(crossings_path)
    centroid = cross.geometry.union_all().centroid
    epsg = args.epsg or get_utm_epsg(centroid.x, centroid.y)

    if args.measured_csv:
        print(f"using {args.measured_csv} directly - skipping SAM3")
        survey = pd.read_csv(args.measured_csv)
    else:
        roads_path = latest(args.roads)
        if not roads_path.exists():
            raise SystemExit(f"{args.roads} not found - run src/crossing_identify.py first")
        print(f"roads: {roads_path}")
        roads = gpd.read_file(roads_path)
        survey = measure_all(cross, roads, epsg, args)
        write_csv(survey, Path(args.survey_csv))
        n_ok = survey["measured_width_m"].notna().sum()
        print(f"\nwrote {args.survey_csv} ({n_ok}/{len(survey)} measured)")
        print(f"QA crops in {args.images_dir}/")

    survey = survey[survey["measured_width_m"].notna()].copy()
    edge_cols = ["width_p1_x", "width_p1_y", "width_p2_x", "width_p2_y",
                 "ext_p1_x", "ext_p1_y", "ext_p2_x", "ext_p2_y"]
    # n_roads/road_layout_m/total/cw*/median: the combined-road reading (width_core.combine_roads) -
    # cw1/median/cw2 are also fillable by hand in a --measured-csv recheck, for a divided road the
    # imagery pass couldn't split
    width_cols = ["measured_width_m", "measured_by", "notes", "width_iqr_m", "n_transects",
                  "n_roads", "road_layout_m", "total_width_m",
                  "cw1_width_m", "median_width_m", "cw2_width_m",
                  "surface_sam3", "surface_paved_score", "surface_unpaved_score"]
    for col in width_cols + edge_cols:
        if col not in survey.columns:
            survey[col] = None
    survey = survey[["crossing_id"] + width_cols + edge_cols]
    print(f"\n{len(survey)} measured row(s) to merge")

    cross = cross.drop(columns=width_cols + edge_cols, errors="ignore")
    cross = cross.merge(survey, on="crossing_id", how="left")
    cross[["design_width_m", "width_source"]] = cross.apply(resolve_width, axis=1, result_type="expand")
    cross["span_m"] = None  # a span from an earlier run no longer matches this width - structure_decide.py sets it

    # edge coordinates stay in the GeoJSON too - structure_decide.py redraws the width lines from them
    write_geojson(cross, Path(args.output))
    write_csv(cross.drop(columns="geometry"), Path(args.csv_output))
    print(f"\nwrote {args.output}")
    print(f"wrote {args.csv_output}")
    print(cross["width_source"].value_counts().to_string())
    still_missing = cross[cross["width_source"].str.startswith("needs field/imagery survey")]
    print(f"\n{len(still_missing)} crossing(s) still have no usable width at all - keep surveying.")

    write_width_geometry(cross, epsg, str(latest(args.alignment)), args.width_geometry_output)
    print(f"\nBox/bridge call, box size and span along track - run next:\n"
          f"  python src/structure_decide.py --region {args.region}")


if __name__ == "__main__":
    main()
