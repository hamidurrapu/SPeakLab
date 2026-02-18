# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for building the Bol Spectra GUI as a desktop app

import glob
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hiddenimports = []
hiddenimports += collect_submodules('lmfit')
hiddenimports += collect_submodules('matplotlib')
hiddenimports += collect_submodules('numpy')
hiddenimports += collect_submodules('scipy')

datas = []
datas += collect_data_files('matplotlib')
datas += collect_data_files('lmfit')

# Add every file in assets/ (header PNG, icon PNG/ICO, etc.)
for f in glob.glob("assets/*"):
    datas.append((f, "assets"))

block_cipher = None

a = Analysis(
    ['speaklab_gui.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='speaklab',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/speaklab.ico',
)

# Optional macOS bundle (only used if building on macOS and you have an .icns file)
app = BUNDLE(
    exe,
    name='speaklab.app',
    icon='assets/speaklab.icns',  # supply if you have it
    bundle_identifier='com.example.bolspectra',
)