# Build a single-file Windows .exe for Bol Spectra
# Usage:  pwsh -File scripts/build_onefile_windows.ps1

$ErrorActionPreference = "Stop"

# Ensure venv (optional)
if (-not (Test-Path ".venv")) {
  python -m venv .venv
}
& .\.venv\Scripts\pip.exe install --upgrade pip
& .\.venv\Scripts\pip.exe install -r requirements.txt
& .\.venv\Scripts\pip.exe install pyinstaller

# Convert icon if needed (PNG -> ICO)
if (-not (Test-Path "assets\speaklab.ico") -and (Test-Path "assets\speaklab.png")) {
  & .\.venv\Scripts\python.exe scripts\convert_icon.py assets\speaklab.png assets\speaklab.ico
}

# Build one-file, windowed app
& .\.venv\Scripts\pyinstaller.exe `
  --noconfirm `
  --onefile `
  --windowed `
  --name "speaklab" `
  --icon "assets\speaklab.ico" `
  speaklab_gui.py

Write-Host ""
Write-Host "Build complete. Executable is in .\dist\speaklab.exe"