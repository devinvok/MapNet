# One-time setup for the real test (Windows PowerShell).
#   powershell -ExecutionPolicy Bypass -File realtest\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$Reader = if ($env:MAPNET_MODEL) { $env:MAPNET_MODEL } else { "qwen2.5:3b" }
$Embed  = if ($env:MAPNET_EMBED_MODEL) { $env:MAPNET_EMBED_MODEL } else { "nomic-embed-text" }

Write-Host "==> Python environment"
if (-not (Test-Path .venv)) { python -m venv .venv }
& .\.venv\Scripts\Activate.ps1
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt pytest

Write-Host "==> Ollama"
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "Ollama is not installed. Download it from https://ollama.com/download, then re-run."
    exit 1
}
try { Invoke-RestMethod http://localhost:11434/api/tags | Out-Null }
catch {
    Write-Host "Starting the Ollama server..."
    Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 5
}
ollama pull $Reader
ollama pull $Embed

Write-Host "==> Tests"
python -m pytest -q

Write-Host "==> Kubernetes docs (needs git)"
python -c "from pathlib import Path; from realtest.corpus import fetch_k8s; fetch_k8s(Path('realtest/data'))"

Write-Host "`nReady. Run the test with:`n  .\.venv\Scripts\Activate.ps1`n  python -m realtest.run"
