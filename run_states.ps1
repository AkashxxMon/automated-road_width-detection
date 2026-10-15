<#
Runs the whole pipeline for each state, one state after another (SAM3 needs the
GPU to itself - two at once run out of memory on an 8 GB card):
  1. crossing_identify.py   find the crossings (NIC + OSM roads)
  2. measure_width_sam3.py  measure the road widths with SAM3
  3. structure_decide.py    decide and size the structures
then score_validation.py once at the end, for every state with a reference file.

Each state's output is also saved to output\logs\<state>.log. If a step fails,
the script stops there; rerun with -Resume to carry on without re-measuring.

Usage (from the project folder):
  .\run_states.ps1                                  # all four states, full re-measure
  .\run_states.ps1 -Resume                          # only measure crossings not measured yet
  .\run_states.ps1 -States "Jharkhand","West Bengal"
  .\run_states.ps1 -SkipIdentify -Resume            # keep the crossings, just measure/decide

Another alignment: give its file and a tag, so its outputs never overwrite this one's -
every file is named after the region, here maharashtra_newline_*, IDs still MH-001...:
  .\run_states.ps1 -Alignment input\new_line.geojson -Tag newline -States "Maharashtra"
  .\run_states.ps1 -Alignment input\new_line.kml -Tag newline -IdentifyArgs "--corridor-km","1.5"
Never use -Resume on a region whose alignment has changed: the saved widths belong to the
old crossings. A new -Tag starts clean.
If PowerShell refuses to run it: powershell -ExecutionPolicy Bypass -File .\run_states.ps1
#>
param(
    [string[]]$States = @("Maharashtra", "Jharkhand", "Chhattisgarh", "West Bengal"),
    [switch]$Resume,        # measure only crossings without a saved width (and continue a stopped run)
    [switch]$SkipIdentify,  # don't rerun crossing_identify.py
    [switch]$SkipMeasure,   # don't rerun measure_width_sam3.py
    [string]$Alignment = "",      # alignment file; default: input\DFCC_master_all_sections_Lines.geojson
    [string]$Tag = "",            # appended to the region name, e.g. newline -> maharashtra_newline_*
    [string[]]$IdentifyArgs = @() # any extra crossing_identify.py arguments
)

# "Continue", not "Stop": Python writes progress and warnings to stderr, which Windows PowerShell would
# otherwise turn into a terminating error - failures are caught from the exit code instead
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$py = ".venv\Scripts\python.exe"
New-Item -ItemType Directory -Force "output\logs" | Out-Null

function Step($label, $log, [string[]]$cmd) {
    Write-Host "`n--- $label" -ForegroundColor Cyan
    & $py @cmd 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { Write-Host "$label failed (exit $LASTEXITCODE) - see $log" -ForegroundColor Red; exit 1 }
}

function Slug($s) { ($s.ToLower() -replace "[^a-z0-9]+", "_").Trim("_") }

$start = Get-Date
$scored = @()
foreach ($state in $States) {
    $region = Slug $state
    if ($Tag) { $region = "${region}_$(Slug $Tag)" }
    $log = "output\logs\$region.log"
    "=== $state  $(Get-Date)" | Out-File $log -Encoding utf8
    Write-Host "`n=================== $state ($region) ===================" -ForegroundColor Yellow

    if (-not $SkipIdentify) {
        $c = @("src\crossing_identify.py", "--state", $state, "--region", $region)
        if ($Alignment) { $c += @("--alignment", $Alignment) }
        Step "$state - crossing_identify" $log ($c + $IdentifyArgs)
    }
    if (-not $SkipMeasure) {
        $m = @("src\measure_width_sam3.py", "--region", $region)
        if ($Resume) { $m += "--resume" }
        Step "$state - measure_width_sam3" $log $m
    }
    Step "$state - structure_decide" $log @("src\structure_decide.py", "--region", $region)
    if (Test-Path "validation\Bridges_road_crossings_lines_$region.geojson") { $scored += $region }
}

# scored only where there are designers' bridges for the region: validation\Bridges_road_crossings_lines_<region>.geojson
if ($scored) { Step "score_validation" "output\logs\score_validation.log" (@("src\score_validation.py", "--region") + $scored) }
else { Write-Host "`nno validation\Bridges_road_crossings_lines_<region>.geojson for these regions - not scored" }
Write-Host "`nDone in $([int]((Get-Date) - $start).TotalMinutes) min. Logs: output\logs\" -ForegroundColor Green
