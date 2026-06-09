"""
seed_grid.py — Create a 3x5 grid of face-centered square crops from seed images.

Usage:
  python seed_grid.py /path/to/seed_images_dir
  python seed_grid.py /path/to/seed_images_dir --output grid.png --size 300
"""

import argparse
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Ordered display names — edit this to change grid order
SEED_ORDER = [
    "JFK", "Donald Trump", "Gandhi", "Fidel Castro", "MLK",
    "Obama", "Bill Clinton", "Queen Elizabeth", "Joseph Stalin", "Kim Jong Un",
    "Adolf Hitler", "Vladimir Putin", "Xi Jinping", "George W. Bush", "Benjamin Netanyahu",
]

# Maps folder/file keywords to display names
SEED_KEYWORDS = {
    "test 1":  "JFK",        "jfk":    "JFK",
    "test 2":  "Donald Trump","trump":  "Donald Trump",
    "test 3":  "Gandhi",     "gandhi": "Gandhi",
    "test 4":  "Fidel Castro","castro": "Fidel Castro",
    "test 5":  "MLK",        "mlk":    "MLK",
    "test 6":  "Obama",      "obama":  "Obama",
    "test 7":  "Bill Clinton","clinton":"Bill Clinton",
    "test 8":  "Queen Elizabeth","queen":"Queen Elizabeth", "elizabeth":"Queen Elizabeth",
    "test 9":  "Joseph Stalin","stalin":"Joseph Stalin",
    "test 10": "Kim Jong Un", "kim":   "Kim Jong Un",
    "test 11": "Adolf Hitler","hitler": "Adolf Hitler",
    "test 12": "Vladimir Putin","putin":"Vladimir Putin",
    "test 13": "Xi Jinping",  "xi":    "Xi Jinping",
    "test 14": "George W. Bush","bush": "George W. Bush",
    "test 15": "Benjamin Netanyahu","netanyahu":"Benjamin Netanyahu",
}


def detect_face_center(img_array):
    """Use InsightFace/ArcFace to find the primary face center. Returns (cx, cy) or None."""
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"],
                           providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        faces = app.get(img_array)
        if faces:
            # Pick largest face by bounding box area
            best = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            cx = (best.bbox[0] + best.bbox[2]) / 2
            cy = (best.bbox[1] + best.bbox[3]) / 2
            return int(cx), int(cy)
    except Exception as e:
        print(f"  Face detection failed: {e}")
    return None


def face_centered_square_crop(img, face_center=None):
    """Crop image to square, centered on face if detected, else center of image."""
    w, h = img.size
    side = min(w, h)

    if face_center:
        cx, cy = face_center
        x0 = max(0, min(cx - side // 2, w - side))
        y0 = max(0, min(cy - side // 2, h - side))
    else:
        x0 = (w - side) // 2
        y0 = (h - side) // 2

    return img.crop((x0, y0, x0 + side, y0 + side))


def identify_seed(path):
    """Try to match a file/folder path to a seed name."""
    name_lower = path.stem.lower().strip()
    parent_lower = path.parent.name.lower().strip()

    # Exact match first (handles "test 1" vs "test 15" ambiguity)
    for key, seed_name in SEED_KEYWORDS.items():
        if key == name_lower or key == parent_lower:
            return seed_name

    # Fuzzy fallback for non-test-N names
    for key, seed_name in SEED_KEYWORDS.items():
        if not key.startswith("test"):
            if key in name_lower or key in parent_lower:
                return seed_name
    return None


def find_seed_images(seed_dir):
    """Find seed images and map them to display names."""
    seed_dir = Path(seed_dir)
    found = {}

    # Check if seeds are in subdirectories (test 1/, test 2/, etc.)
    subdirs = [d for d in seed_dir.iterdir() if d.is_dir()]
    if subdirs:
        for subdir in subdirs:
            seed_name = identify_seed(subdir)
            if seed_name:
                imgs = list(subdir.glob("seed.*")) + list(subdir.glob("seed_*")) + \
                       [f for f in subdir.glob("*") if f.suffix.lower() in (".png",".jpg",".jpeg",".webp")]
                if imgs:
                    found[seed_name] = imgs[0]

    # Also check flat files in the directory
    for f in seed_dir.glob("*"):
        if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp") and f.is_file():
            seed_name = identify_seed(f)
            if seed_name and seed_name not in found:
                found[seed_name] = f

    return found


def main():
    parser = argparse.ArgumentParser(description="Create 3x5 grid of face-centered seed images")
    parser.add_argument("seed_dir", help="Directory containing seed images")
    parser.add_argument("--output", default="seed_grid.png", help="Output filename")
    parser.add_argument("--size", type=int, default=400, help="Size of each cell in pixels")
    parser.add_argument("--gap", type=int, default=4, help="Gap between cells in pixels")
    parser.add_argument("--label", action="store_true", help="Add name labels below each image")
    args = parser.parse_args()

    seed_images = find_seed_images(args.seed_dir)
    print(f"Found {len(seed_images)} seed images:")
    for name, path in sorted(seed_images.items()):
        print(f"  {name}: {path}")

    missing = [s for s in SEED_ORDER if s not in seed_images]
    if missing:
        print(f"\nMissing seeds: {missing}")
        print("Continuing with available images...")

    available = [s for s in SEED_ORDER if s in seed_images]
    if not available:
        print("No seed images found. Check your directory structure.")
        sys.exit(1)

    # Load face detector once
    print("\nLoading face detector...")
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"],
                           providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        has_detector = True
        print("Face detector ready.")
    except Exception as e:
        print(f"Face detector unavailable ({e}), using center crops.")
        has_detector = False
        app = None

    # Process images
    cell_size = args.size
    gap = args.gap
    cols, rows = 3, 5
    label_h = 30 if args.label else 0

    grid_w = cols * cell_size + (cols - 1) * gap
    grid_h = rows * (cell_size + label_h) + (rows - 1) * gap
    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))

    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(grid) if args.label else None
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
    except:
        draw = None
        font = None

    for idx, seed_name in enumerate(available):
        row = idx // cols
        col = idx % cols

        img = Image.open(seed_images[seed_name]).convert("RGB")

        # Detect face
        face_center = None
        if has_detector:
            try:
                img_array = np.array(img)
                # BGR for insightface
                img_bgr = img_array[:, :, ::-1]
                faces = app.get(img_bgr)
                if faces: #false for auto crop
                    best = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                    cx = int((best.bbox[0] + best.bbox[2]) / 2)
                    cy = int((best.bbox[1] + best.bbox[3]) / 2)
                    face_center = (cx, cy)
                    print(f"  {seed_name}: face at ({cx}, {cy})")
                else:
                    print(f"  {seed_name}: no face detected, center crop")
            except Exception as e:
                print(f"  {seed_name}: detection error ({e}), center crop")

        cropped = face_centered_square_crop(img, face_center)
        cropped = cropped.resize((cell_size, cell_size), Image.LANCZOS)

        x = col * (cell_size + gap)
        y = row * (cell_size + label_h + gap)
        grid.paste(cropped, (x, y))

        if draw and args.label:
            text_bbox = draw.textbbox((0, 0), seed_name, font=font)
            tw = text_bbox[2] - text_bbox[0]
            tx = x + (cell_size - tw) // 2
            ty = y + cell_size + 4
            draw.text((tx, ty), seed_name, fill=(0, 0, 0), font=font)

    output_path = Path(args.output)
    grid.save(output_path, quality=95)
    print(f"\nSaved: {output_path} ({grid_w}x{grid_h})")


if __name__ == "__main__":
    main()