#!/usr/bin/env bash
# Build a single-file executable for macOS/Linux
# Usage:  bash scripts/build_onefile_posix.sh

set -euo pipefail

# Optional venv
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller

# Convert icon if we only have PNG
if [ ! -f assets/speaklab.icns ] && [ -f assets/speaklab.png ]; then
  # Create .icns for macOS; if not on macOS, we'll just use PNG
  python scripts/convert_icon.py assets/speaklab.png assets/speaklab.icns
fi

ICON_ARG=()
if [[ "$OSTYPE" == "darwin"* && -f assets/speaklab.icns ]]; then
  ICON_ARG=(--icon assets/speaklab.icns)
elif [ -f assets/speaklab.png ]; then
  ICON_ARG=(--icon assets/speaklab.png)
fi

pyinstaller \
  --noconfirm \
  --onefile \
  --windowed \
  "${ICON_ARG[@]}" \
  --name "speaklab" \
  speaklab_gui.py

echo
echo "Build complete. Executable is in ./dist/"