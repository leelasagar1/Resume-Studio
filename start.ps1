# Start the app on Windows (PowerShell). Run .\setup.ps1 once first.
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
if (-not (Test-Path ".\.venv\Scripts\python.exe")) { Write-Error "No .venv found. Run .\setup.ps1 first." }
if (-not (Test-Path ".env")) { Write-Error "No .env found. Run .\setup.ps1, then put your API key in .env." }
Write-Host "Resume Studio at http://127.0.0.1:8765  (Ctrl+C to stop)"
& .\.venv\Scripts\python.exe run.py
