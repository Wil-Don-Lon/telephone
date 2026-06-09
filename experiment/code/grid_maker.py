"""
grid_maker.py — Composites sample_0001.png through sample_0025.png into a 5x5 grid.

Usage:
    python grid_maker.py /path/to/folder
    python grid_maker.py /path/to/folder --output grid.png
    python grid_maker.py /path/to/folder --output grid.png --cell-size 512 --gap 4 --bg white
"""

import argparse
from pathlib import Path
from PIL import Image


def make_grid(folder: str, output: str, cell_size: int, gap: int, bg: str):
    folder = Path(folder)
    rows, cols = 5, 5

    files = [folder / f"sample_{i:04d}.png" for i in range(1, 26)]
    missing = [f for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} file(s): {', '.join(f.name for f in missing[:5])}"
        )

    grid_w = cols * cell_size + (cols + 1) * gap
    grid_h = rows * cell_size + (rows + 1) * gap
    grid = Image.new("RGB", (grid_w, grid_h), bg)

    for idx, fpath in enumerate(files):
        img = Image.open(fpath).convert("RGB")
        img = img.resize((cell_size, cell_size), Image.LANCZOS)
        r, c = divmod(idx, cols)
        x = gap + c * (cell_size + gap)
        y = gap + r * (cell_size + gap)
        grid.paste(img, (x, y))

    out = Path(output) if output else folder / "grid.png"
    grid.save(out, dpi=(300, 300))
    print(f"Saved {out} ({grid_w}x{grid_h}px)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="5x5 image grid maker")
    parser.add_argument("folder", help="Folder containing sample_0001.png - sample_0025.png")
    parser.add_argument("--output", "-o", default=None, help="Output path (default: folder/grid.png)")
    parser.add_argument("--cell-size", type=int, default=512, help="Size per cell in px (default: 512)")
    parser.add_argument("--gap", type=int, default=4, help="Gap between cells in px (default: 4)")
    parser.add_argument("--bg", default="white", help="Background color (default: white)")
    args = parser.parse_args()
    make_grid(args.folder, args.output, args.cell_size, args.gap, args.bg)
