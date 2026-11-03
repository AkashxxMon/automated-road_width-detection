"""Rail-Road Crossing Detection - the local web app for the crossing pipeline.

Upload an alignment (or use the default DFCC one), pick the area and every
rule/threshold the three scripts take, run them, and watch it happen: the
crossings appear on the map when they're found, each SAM3 result (its QA crop
and width) appears as it's measured, and the structures when they're decided.
The output table sits under the map; the finished GeoJSON/CSV files are downloadable.

It runs the existing scripts (src/crossing_identify.py, measure_width_sam3.py,
structure_decide.py) as subprocesses, one run at a time - SAM3 needs the GPU to
itself. It's local only: the GPU, the NIC road data and the SAM3 model are on
this machine. The help page is webapp/help.html, served at /help.

Run it:
    .venv\\Scripts\\python.exe webapp\\server.py          then open http://127.0.0.1:8000
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
from geo_utils import get_utm_epsg, latest, region_paths, region_prefix, region_slug  # noqa: E402

PY = str(ROOT / ".venv" / "Scripts" / "python.exe") if (ROOT / ".venv" / "Scripts" / "python.exe").exists() else sys.executable
UPLOADS = ROOT / "input" / "uploads"
STATES_FILE = ROOT / "input" / "boundaries" / "india_states.geojson"
DEFAULT_ALIGNMENT = "input/DFCC_master_all_sections_Lines.geojson"
INDIA_BBOX = (68.0, 6.0, 98.0, 37.5)
METRIC = 7755  # India LCC, metres

app = FastAPI(title="Rail-Road Crossing Detection")
_states = None


def states() -> gpd.GeoDataFrame:
    global _states
    if _states is None:
        _states = gpd.read_file(STATES_FILE).to_crs(METRIC)
    return _states


# ---------------------------------------------------------------- uploads and preview

def _read(path: Path) -> gpd.GeoDataFrame:
    g = gpd.read_file(path)
    if g.crs is None:
        g = g.set_crs(4326)
    return g[g.geometry.notna() & g.geom_type.str.contains("Line")]


def _in_india(g: gpd.GeoDataFrame) -> bool:
    c = g.to_crs(4326).geometry.union_all().centroid
    return INDIA_BBOX[0] <= c.x <= INDIA_BBOX[2] and INDIA_BBOX[1] <= c.y <= INDIA_BBOX[3]


def _crs_suggestions(g: gpd.GeoDataFrame) -> list[int]:
    """The other UTM zones that would also put this line inside India. A wrong zone label
    can't be told from the coordinates alone - a zone-44 CAD export labelled zone 43 still
    lands in India, 6 degrees west - so the page offers these and the preview shows where
    each one puts the line."""
    epsg = g.crs.to_epsg() if g.crs else None
    if not epsg or not 32642 <= epsg <= 32647:
        return []
    india = states().geometry.union_all()
    out = []
    for epsg in range(32642, 32648):
        try:
            t = g.set_crs(epsg, allow_override=True).to_crs(METRIC)
        except Exception:
            continue
        if epsg != g.crs.to_epsg() and t.intersects(india).mean() > 0.5:
            out.append(epsg)
    return out


def _filter(g: gpd.GeoDataFrame, field: str | None, value: str | None, epsg: int | None) -> gpd.GeoDataFrame:
    if epsg:
        g = g.set_crs(epsg, allow_override=True)
    if field and field in g.columns and value not in (None, ""):
        g = g[g[field].astype(str) == str(value)]
    return g


def _states_crossed(g: gpd.GeoDataFrame) -> list[dict]:
    t = g.to_crs(METRIC)
    out = []
    for _, st in states().iterrows():
        km = t.intersection(st.geometry).length.sum() / 1000
        if km > 0.05:
            out.append({"state": st["NAME_1"], "km": round(km, 1)})
    return sorted(out, key=lambda x: -x["km"])


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    UPLOADS.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", file.filename or "alignment.geojson")
    path = UPLOADS / f"{time.strftime('%Y%m%d_%H%M%S')}_{name}"
    path.write_bytes(await file.read())
    try:
        g = _read(path)
    except Exception as e:
        raise HTTPException(400, f"could not read {file.filename}: {e}")
    if g.empty:
        raise HTTPException(400, "no line features in that file")
    metric = g.to_crs(METRIC) if _in_india(g) else None
    fields = []
    for col in g.columns:
        if col == "geometry" or g[col].dtype.kind not in "OSU" or not 1 < g[col].nunique() <= 60:
            continue
        vals = g.groupby(g[col].astype(str)).size()
        km = metric.groupby(g[col].astype(str)).apply(lambda x: x.length.sum() / 1000) if metric is not None else None
        fields.append({"field": col, "values": [{"value": v, "n": int(n), "km": round(float(km[v]), 2) if km is not None else None}
                                               for v, n in vals.sort_values(ascending=False).items()]})
    return {"file": str(path.relative_to(ROOT)).replace("\\", "/"), "name": file.filename, "n": len(g),
            "crs": g.crs.to_string() if g.crs else None, "in_india": _in_india(g),
            "crs_suggestions": _crs_suggestions(g), "fields": fields}


@app.get("/api/preview")
def preview(file: str, field: str | None = None, value: str | None = None, epsg: int | None = None):
    g = _filter(_read(ROOT / file), field, value, epsg)
    if g.empty:
        raise HTTPException(400, "no lines left with that filter")
    if not _in_india(g):
        return {"in_india": False, "states": [], "km": None, "geojson": None}
    w = g.to_crs(4326)
    w = gpd.GeoDataFrame(geometry=w.geometry.simplify(0.00005), crs=4326)
    return {"in_india": True, "states": _states_crossed(g), "km": round(g.to_crs(METRIC).length.sum() / 1000, 2),
            "n": len(g), "geojson": json.loads(w.to_json())}


# ---------------------------------------------------------------- jobs

STEP_ORDER = ["identify", "measure", "structures"]
MEAS_RE = re.compile(r"^\s{2}([A-Za-z0-9]+-\d+):\s+(.*?)(?:\s+\[(\d+)/(\d+), (\d+)/min\])?\s*$")


class Job:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.region = cfg["region"]
        self.status = "running"
        self.stage = None
        self.stages_done: list[str] = []
        self.log: list[str] = []
        self.measured: list[dict] = []
        self.progress = None
        self.proc: subprocess.Popen | None = None
        self.cancelled = False
        self.error = None
        self.started = time.time()
        self.ended = None

    def emit(self, line: str):
        self.log.append(line)
        if len(self.log) > 20000:
            del self.log[:5000]


JOB: Job | None = None
LOCK = threading.Lock()


def _opt(args: list, flag: str, v):
    if v is None or v == "":
        return
    if isinstance(v, bool):
        args.append(flag if v else "--no-" + flag[2:])
    else:
        args += [flag, str(v)]


def _extra(s: str | None) -> list[str]:
    import shlex
    return shlex.split(s, posix=False) if s else []


def build_steps(cfg: dict) -> list[tuple[str, list[str]]]:
    r = cfg["region"]
    i, m, s = cfg.get("identify", {}), cfg.get("measure", {}), cfg.get("structures", {})
    steps = []
    if "identify" in cfg["steps"]:
        a = ["src/crossing_identify.py", "--alignment", cfg["alignment"], "--region", r, "--id-prefix", cfg["id_prefix"],
             "--start-chainage-m", f"{float(cfg.get('start_chainage_m') or 0):.3f}"]
        a += ["--state", cfg["state"]] if cfg.get("clip") == "state" else ["--boundary", ""]
        if cfg.get("file"):
            a += ["--feature-type-field", ""]  # an upload's line filter was applied when the prepared file was written
        for k in ("corridor_km", "osm_buffer_m", "osm_angle_deg", "min_angle_deg", "dual_merge_tol_m", "junction_flag_m",
                  "nearby_crossing_flag_m", "min_segment_m"):
            _opt(a, "--" + k.replace("_", "-"), i.get(k))
        _opt(a, "--osm", bool(i.get("osm", True)))
        a += ["--drop-classes", ",".join(i.get("drop_classes", ["TRK"]))]
        steps.append(("identify", a + _extra(i.get("extra"))))
    if "measure" in cfg["steps"]:
        a = ["src/measure_width_sam3.py", "--region", r]
        for k in ("text", "half_extent_m", "detect_threshold", "surface_paved_margin", "surface_unpaved_margin", "limit",
                  "fetch_workers", "prefetch", "cpu_workers", "checkpoint_every"):
            _opt(a, "--" + k.replace("_", "-"), m.get(k))
        _opt(a, "--surface", bool(m.get("surface", True)))
        if m.get("resume"):
            a.append("--resume")
        steps.append(("measure", a + _extra(m.get("extra"))))
    if "structures" in cfg["steps"]:
        a = ["src/structure_decide.py", "--region", r, "--major-classes", ",".join(s.get("major_classes", ["EXP", "NH", "SH", "MDR", "ODR"]))]
        for k in ("edge_buffer_m", "clearance_m", "minor_clearance_m", "minor_rule",
                  "minor_keep_width_m", "minor_divert_m", "unknown_surface", "thin_m", "corridor_merge_m",
                  "corridor_angle_deg", "borderline_m", "road_offset_flag_m", "class_check_width_m"):
            _opt(a, "--" + k.replace("_", "-"), s.get(k))
        _opt(a, "--keep-paved-minor", bool(s.get("keep_paved_minor", True)))
        steps.append(("structures", a + _extra(s.get("extra"))))
    return steps


def run_job(job: Job):
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    try:
        for name, args in build_steps(job.cfg):
            if job.cancelled:
                break
            job.stage = name
            job.emit(f"\n=== {name}: python {' '.join(args)}")
            job.proc = subprocess.Popen([PY, "-u", *args], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
            last_loading = None
            for line in job.proc.stdout:
                line = line.rstrip("\n")
                if "Loading weights" in line:  # tqdm redraws - keep only the latest
                    last_loading = line
                    continue
                if last_loading:
                    job.emit(last_loading.strip()[:160])
                    last_loading = None
                job.emit(line)
                if name == "measure":
                    mm = MEAS_RE.match(line)
                    if mm:
                        cid, text, n, total = mm.group(1), mm.group(2), mm.group(3), mm.group(4)
                        ok = bool(re.match(r"[\d.]+ m", text))
                        job.measured.append({"cid": cid, "text": text, "ok": ok, "suspect": "SUSPECT" in text})
                        if n:
                            job.progress = [int(n), int(total)]
                    mt = re.match(r"measuring (\d+) crossing", line)
                    if mt:
                        job.progress = [0, int(mt.group(1))]
            code = job.proc.wait()
            if job.cancelled:
                break
            if code != 0:
                raise RuntimeError(f"{name} failed (exit {code}) - see the log")
            job.stages_done.append(name)
        job.status = "cancelled" if job.cancelled else "done"
    except Exception as e:
        job.status, job.error = "failed", str(e)
        job.emit(f"\n!!! {e}")
    finally:
        job.stage, job.proc, job.ended = None, None, time.time()


def _prepare_alignment(cfg: dict) -> str:
    """The uploaded file with the chosen line filter and CRS fix applied, as GeoJSON."""
    if not cfg.get("file"):
        return DEFAULT_ALIGNMENT
    g = _filter(_read(ROOT / cfg["file"]), cfg.get("field"), cfg.get("value"), cfg.get("epsg"))
    if g.empty:
        raise HTTPException(400, "no lines left with that filter")
    out = UPLOADS / f"{cfg['region']}_alignment.geojson"
    g.to_file(out, driver="GeoJSON")
    return str(out.relative_to(ROOT)).replace("\\", "/")


@app.post("/api/run")
def run(cfg: dict):
    global JOB
    with LOCK:
        if JOB and JOB.status == "running":
            raise HTTPException(409, "a run is already going - wait for it or cancel it")
        cfg["region"] = region_slug(cfg.get("region") or "")
        if not cfg["region"] or cfg["region"] == "region":
            raise HTTPException(400, "give the run a name (region)")
        cfg["steps"] = [s for s in STEP_ORDER if s in cfg.get("steps", STEP_ORDER)]
        cfg["id_prefix"] = (cfg.get("id_prefix") or region_prefix(cfg["region"])).strip()
        if "identify" in cfg["steps"]:
            if cfg.get("clip") == "state" and not cfg.get("state"):
                raise HTTPException(400, "pick a state, or choose 'whole line'")
            cfg["alignment"] = _prepare_alignment(cfg)
        elif not Path(region_paths(cfg["region"])["crossings"]).exists():
            raise HTTPException(400, f"no crossings yet for {cfg['region']} - include the 'find crossings' step")
        JOB = Job(cfg)
        threading.Thread(target=run_job, args=(JOB,), daemon=True).start()
        return {"region": JOB.region, "steps": cfg["steps"], "commands": [" ".join(a) for _, a in build_steps(cfg)]}


@app.post("/api/cancel")
def cancel():
    if not JOB or JOB.status != "running":
        return {"ok": False}
    JOB.cancelled = True
    if JOB.proc:
        JOB.proc.kill()
    return {"ok": True}


@app.get("/api/job")
def job(log_from: int = 0, meas_from: int = 0):
    if not JOB:
        return {"status": "idle"}
    j = JOB
    return {"status": j.status, "region": j.region, "stage": j.stage, "stages_done": j.stages_done,
            "steps": j.cfg["steps"], "progress": j.progress, "error": j.error,
            "elapsed_s": round((j.ended or time.time()) - j.started),
            "log": j.log[log_from:], "log_next": len(j.log),
            "measured": j.measured[meas_from:], "meas_next": len(j.measured)}


# ---------------------------------------------------------------- results

LAYERS = {"alignment": "alignment", "crossings": "crossings", "final": "final", "structures": "structures",
          "geometry": "structure_geometry"}
DOWNLOADS = {"structures.geojson": "structures", "structures.csv": "structures_csv",
             "structure_geometry.geojson": "structure_geometry", "crossings.geojson": "crossings",
             "crossings_final_gl.csv": "final_csv", "width_survey.csv": "survey_csv"}


@app.get("/api/layer/{region}/{kind}")
def layer(region: str, kind: str):
    if kind not in LAYERS:
        raise HTTPException(404)
    path = latest(region_paths(region)[LAYERS[kind]])
    if not path.exists():
        raise HTTPException(404, f"{path} not there yet")
    g = gpd.read_file(path)
    if g.crs and g.crs.to_epsg() != 4326:
        g = g.to_crs(4326)
    keep = [c for c in ("crossing_id", "section_id", "align_name", "chainage_m", "chainage_start_m", "decision", "rule",
                        "reason", "irc_class", "Assumed_irc_class", "name", "ref", "road_source", "angle_deg",
                        "skew_angle_deg", "measured_width_m", "width_source", "n_roads", "combined_layout_m",
                        "combined_width_m", "road_surface", "surface_source", "road_category", "nearest_major_crossing_m",
                        "structure_type", "Assumed_structure_type", "structure_suggested", "structure_rule", "box_size",
                        "Assumed_box_size", "box_split_size", "opening_width_m", "Assumed_opening_width_m", "span_m",
                        "Assumed_span_m", "span_centre_chainage_m", "clearance_m", "structure_flags", "structure_reason",
                        "merged_into", "merged_ids", "near_junction", "nearby_crossing_id", "nearby_dist_m",
                        "dual_carriageway", "measured_by", "notes", "length_m") if c in g.columns]
    g = g[keep + ["geometry"]]
    return Response(g.to_json(na="null", drop_id=True), media_type="application/geo+json")


@app.get("/api/qa/{region}/{cid}")
def qa(region: str, cid: str):
    path = Path(region_paths(region)["images_dir"]) / f"{Path(cid).stem}.jpg"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/download/{region}/{name}")
def download(region: str, name: str):
    paths = region_paths(region)
    if name == "all.zip":
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for fname, key in DOWNLOADS.items():
                p = latest(paths[key])
                if p.exists():
                    z.write(p, f"{region}_{fname}")
        return Response(buf.getvalue(), media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{region}_results.zip"'})
    if name not in DOWNLOADS:
        raise HTTPException(404)
    p = latest(paths[DOWNLOADS[name]])
    if not p.exists():
        raise HTTPException(404, f"{p} not there yet")
    return FileResponse(p, filename=f"{region}_{name}")


@app.get("/api/region/{region}")
def region_status(region: str):
    region = region_slug(region)
    paths = region_paths(region)
    have = {k: latest(paths[v]).exists() for k, v in LAYERS.items()}
    have["downloads"] = [n for n, k in DOWNLOADS.items() if latest(paths[k]).exists()]
    return {"region": region, "id_prefix": region_prefix(region), **have}


@app.get("/api/regions")
def regions():
    return sorted(p.name[: -len("_crossings.geojson")] for p in (ROOT / "output").glob("*_crossings.geojson"))


@app.get("/api/meta")
def meta():
    return {"states": sorted(states()["NAME_1"].dropna().unique().tolist()),
            "classes": ["EXP", "NH", "SH", "MDR", "ODR", "VR", "TRK", "PTH"],
            "busy": bool(JOB and JOB.status == "running")}


@app.get("/api/chainage/{region}")
def chainage(region: str, step: float = 100.0):
    """A point every `step` metres along the run's alignment, labelled with its chainage, for the
    map. Chainage = the part's chainage at its first vertex (chainage_start_m, set from the start
    chainage by crossing_identify.py) + distance along the part, in the direction it was drawn."""
    path = latest(region_paths(region)["alignment"])
    if not path.exists():
        raise HTTPException(404, f"{path} not there yet")
    g = gpd.read_file(path)
    # the CRS crossing_identify.py measured chainage in (else the UTM zone it would have picked):
    # lengths differ between CRSs - India LCC is ~1.5% short of UTM at 18 deg N - so ticks drawn in
    # any other CRS drift away from the crossings' chainages along the line
    if "chainage_epsg" in g.columns and pd.notna(g["chainage_epsg"].iloc[0]):
        epsg = int(g["chainage_epsg"].iloc[0])
    else:
        c = g.to_crs(4326).geometry.union_all().centroid
        epsg = get_utm_epsg(c.x, c.y)
    g = g.to_crs(epsg)
    has_start = "chainage_start_m" in g.columns
    feats = []
    for _, row in g.iterrows():
        line = row.geometry
        c0 = float(row["chainage_start_m"]) if has_start and pd.notna(row["chainage_start_m"]) else 0.0
        ch = math.ceil(c0 / step) * step  # ticks on round chainages
        while ch <= c0 + line.length + 1e-6:
            feats.append({"chainage_m": round(ch, 3), "geometry": line.interpolate(ch - c0)})
            ch += step
    if not feats:
        return Response('{"type": "FeatureCollection", "features": []}', media_type="application/geo+json")
    pts = gpd.GeoDataFrame(feats, geometry="geometry", crs=epsg).to_crs(4326)
    return Response(pts.to_json(drop_id=True), media_type="application/geo+json")


@app.get("/help")
def help_page():
    return HTMLResponse((Path(__file__).parent / "help.html").read_text(encoding="utf-8"))


@app.get("/")
def index():
    return HTMLResponse((Path(__file__).parent / "index.html").read_text(encoding="utf-8"))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0", help="127.0.0.1 = this machine only; 0.0.0.0 to share on the network")
    a = ap.parse_args()
    print(f"open http://{'127.0.0.1' if a.host == '0.0.0.0' else a.host}:{a.port}")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
