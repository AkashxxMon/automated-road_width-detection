<#
Starts the web viewer, then open http://127.0.0.1:8000 in a browser.
  .\run_viewer.ps1               # this machine only
  .\run_viewer.ps1 -Share        # also reachable from other machines on the network
Stop it with Ctrl+C.
#>
param([int]$Port = 8000, [switch]$Share)
Set-Location $PSScriptRoot
$hostAddr = if ($Share) { "0.0.0.0" } else { "127.0.0.1" }
& ".venv\Scripts\python.exe" "webapp\server.py" --port $Port --host $hostAddr
