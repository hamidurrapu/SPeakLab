"""
Convert a source image (PNG recommended) into .ico (Windows) or .icns (macOS).
Usage:
  python scripts/convert_icon.py assets/speaklab.png assets/speaklab.ico
  python scripts/convert_icon.py assets/speaklab.png assets/speaklab.icns
"""
import sys
from pathlib import Path

def to_ico(src: Path, dst: Path):
    from PIL import Image
    img = Image.open(src).convert("RGBA")
    # Create multiple sizes for better scaling
    sizes = [16, 24, 32, 48, 64, 128, 256]
    imgs = [img.resize((s, s), Image.LANCZOS) for s in sizes]
    imgs[0].save(dst, format="ICO", sizes=[(s, s) for s in sizes])

def to_icns(src: Path, dst: Path):
    try:
        # Use Pillow's icns saving if available
        from PIL import Image
        img = Image.open(src).convert("RGBA")
        img.save(dst, format="ICNS")
    except Exception:
        # Fallback: create an .icns-like file via iconset if running on macOS with sips/iconutil
        import subprocess, tempfile, shutil
        tmp = Path(tempfile.mkdtemp())
        try:
            iconset = tmp / "bol.iconset"
            iconset.mkdir()
            for s in (16,32,64,128,256,512,1024):
                out = iconset / f"icon_{s}x{s}.png"
                subprocess.check_call(["sips", "-z", str(s), str(s), str(src), "--out", str(out)])
            subprocess.check_call(["iconutil", "-c", "icns", str(iconset), "-o", str(dst)])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    if not src.exists():
        raise SystemExit(f"Source image not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.suffix.lower() == ".ico":
        to_ico(src, dst)
    elif dst.suffix.lower() == ".icns":
        to_icns(src, dst)
    else:
        raise SystemExit("Destination must end with .ico or .icns")
    print(f"Saved icon: {dst}")

if __name__ == "__main__":
    main()