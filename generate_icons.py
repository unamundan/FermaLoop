#!/usr/bin/env python3
"""
Generates app-icon files for both platforms from a single square source
PNG (1024x1024 recommended):

  - icon.ico       -- Windows, multi-resolution, via Pillow's native
                       ICO writer (no external tools needed)
  - AppIcon.iconset/ -- a folder of correctly-sized/named PNGs following
                       Apple's iconset convention. This script does NOT
                       produce the final .icns itself -- that requires
                       `iconutil`, a macOS-only system tool, so the
                       macOS build job runs `iconutil -c icns
                       AppIcon.iconset -o icon.icns` as a separate step
                       right after calling this script.

Deliberately generates both outputs regardless of which OS this actually
runs on -- it's pure Pillow, no OS-specific calls -- so both CI jobs can
call the exact same script rather than maintaining two versions.

Usage: python generate_icons.py <source.png> <output_dir>
"""
import sys
import os
from PIL import Image


ICO_SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]

ICONSET_SPEC = [
    ("icon_16x16.png", 16),
    ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32),
    ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128),
    ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256),
    ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512),
    ("icon_512x512@2x.png", 1024),
]


def main():
    if len(sys.argv) != 3:
        print("Usage: python generate_icons.py <source.png> <output_dir>")
        sys.exit(1)
    source_path, output_dir = sys.argv[1], sys.argv[2]

    source = Image.open(source_path).convert("RGBA")
    if source.size[0] != source.size[1]:
        print(f"Warning: source image is {source.size}, not square -- "
              f"icons will be stretched/distorted")

    os.makedirs(output_dir, exist_ok=True)

    ico_path = os.path.join(output_dir, "icon.ico")
    source.save(ico_path, sizes=ICO_SIZES)
    print(f"Wrote {ico_path} ({os.path.getsize(ico_path)} bytes, "
          f"{len(ICO_SIZES)} resolutions)")

    iconset_dir = os.path.join(output_dir, "AppIcon.iconset")
    os.makedirs(iconset_dir, exist_ok=True)
    for filename, px in ICONSET_SPEC:
        resized = source.resize((px, px), Image.LANCZOS)
        resized.save(os.path.join(iconset_dir, filename))
    print(f"Wrote {len(ICONSET_SPEC)} PNGs to {iconset_dir}/ "
          f"(run `iconutil -c icns {iconset_dir} -o icon.icns` on macOS to finish)")


if __name__ == "__main__":
    main()
