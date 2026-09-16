# One-time setup on Windows (PowerShell): creates .venv, installs dependencies, creates .env.
# If scripts are blocked:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$py = $null
foreach ($candidate in @("py -3.12", "py -3.13", "py -3.11", "py -3.14", "python")) {
    try {
        $version = & cmd /c "$candidate -c ""import sys; print('%d.%d' % sys.version_info[:2])""" 2>$null
        if ($version -in @("3.11", "3.12", "3.13", "3.14")) { $py = $candidate; break }
    } catch { }
}
if (-not $py) {
    Write-Error "Python 3.11 or newer is required. Install from https://www.python.org/downloads/ (tick 'Add to PATH') and rerun."
}
Write-Host "Using $py"

if (-not (Test-Path ".venv")) {
    & cmd /c "$py -m venv .venv"
}
& .\.venv\Scripts\python.exe -m pip install --upgrade pip | Out-Null
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host ""
    Write-Host "Created .env from .env.example. Open .env and paste your OPENAI_API_KEY."
}

Write-Host ""
Write-Host "Setup complete. Start the app with:  .\start.ps1"
