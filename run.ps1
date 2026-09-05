# ChakraShield — one-shot pipeline: data -> train -> evaluate -> bench -> serve
# Usage:  .\run.ps1            (full pipeline, then serves on :8080)
#         .\run.ps1 -SkipTrain (just serve)
param([switch]$SkipTrain, [int]$Port = 8080)
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"          # the reports print ₹ / Δ; Windows consoles default to cp1252
Set-Location $PSScriptRoot
python -m pip install -q -r requirements.txt
if (-not $SkipTrain) {
  python scripts/01_generate_data.py
  python scripts/02_train.py
  python scripts/03_evaluate.py
  python scripts/04_bench_latency.py
  python -m pytest -q
}
Write-Host "`nChakraShield gateway -> http://127.0.0.1:$Port  (Ctrl+C to stop)" -ForegroundColor Cyan
python scripts/serve.py --port $Port
