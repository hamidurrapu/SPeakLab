# SPeakLab - Raman Module

Download the latest stable release at https://drive.google.com/file/d/1AaZIqtiQ87nwisNZrA7lAqpbOr4V071n/view?usp=sharing

This project includes a Tkinter GUI (`speaklab_gui.py`) for Raman fitting. You can package it as a standalone app for Windows, macOS, or Linux using PyInstaller.

## Prerequisites

- Python 3.9–3.12
- Platform build tools:
  - Windows: PowerShell and Visual C++ Redistributable (typically already present)
  - macOS: Xcode command line tools (for optional icon conversion fallback)
- Internet access to install dependencies

## 1) Install dependencies

```bash
python -m venv .venv

# Windows
.venv\Scripts\pip install --upgrade pip
.venv\Scripts\pip install -r requirements.txt

# macOS/Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

```

## 2) Prepare an app icon (optional but recommended)

Save your logo as `assets/speaklab.png` (you can use the image shared in the issue/chat).

Convert it:

```bash

# Windows ICO
python scripts/convert_icon.py assets/speaklab.png assets/speaklab.ico

# macOS ICNS
python scripts/convert_icon.py assets/speaklab.png assets/speaklab.icns
```

## 3) One-file build

- Windows:

```powershell
pwsh -File scripts/build_onefile_windows.ps1
```

- macOS/Linux:

```bash
bash scripts/build_onefile_posix.sh
```

Artifacts appear in `dist/`:
- Windows: `speaklab.exe`
- macOS: `speaklab` (CLI) and/or `speaklab.app` if using the `.spec`
- Linux: `speaklab`

## 4) One-folder build (faster startup, easier debugging)

```Windows Terminal
python -m PyInstaller --noconfirm --onefile --windowed --name "speaklab" --icon assets/speaklab.ico  --add-data "assets/speaklab_header.png:assets"  --add-data "assets/speaklab_icon.ico:assets" speaklab_gui.py
```

This uses the spec to include matplotlib/lmfit data and creates a `.app` bundle on macOS.

## Notes/Troubleshooting

- If you see missing SciPy or lmfit at runtime, ensure SciPy is installed (lmfit depends on SciPy).
- Tkinter is part of the standard library; ensure your Python includes it.
- Matplotlib backends and fonts are bundled via `collect_data_files('matplotlib')` in the spec.
- If you need a console for debugging, switch to `--console` or set `console=True` in the spec.
- ARM macs: build on the same architecture you plan to run on (or use a universal Python).

## Running the app

Double-click the built executable (or `.app` on macOS). In the GUI:
- Choose a working directory with your `.txt` spectra files
- Configure peaks/ranges and options
- Run Single File or Batch
- Save plots/data as needed
