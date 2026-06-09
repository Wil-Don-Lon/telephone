"""
group_photo_analysis.py — Positional ArcFace analysis for group photos.

Takes a seed image and a directory of generated images. Detects the 4 most
prominent faces (by bbox area) in each image, sorts them left-to-right, and
compares seed face N to generated face N by position. Produces:

  1. Per-position identity scores (face1 = leftmost ... face4 = rightmost)
  2. Labeled copies of all images (seed + generated) with face1..face4 tags
  3. Master results summary matching seed_sampler.py output format

Usage:
    python group_photo_analysis.py \
        --seed-image ./seed_02.jpg \
        --gen-dir ./samples \
        --output-dir ./group_analysis_output
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont


# ================================
# CONFIG
# ================================

MAX_FACES = 4
THRESHOLD = 0.3
SEED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

# Label colors per face position
FACE_COLORS = [
    (255, 50, 50),     # face1 — red
    (50, 180, 50),     # face2 — green
    (50, 100, 255),    # face3 — blue
    (255, 200, 0),     # face4 — yellow
]
FACE_LABELS = ["face1", "face2", "face3", "face4"]


# ================================
# FACE DETECTION
# ================================

def load_face_model():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def detect_faces(image_path: Path, app, max_faces: int = MAX_FACES) -> list[dict]:
    """
    Detect faces, take the `max_faces` largest by area, sort left-to-right.
    Returns list of dicts with embedding, bbox, x_center, area.
    """
    img = np.array(PILImage.open(image_path).convert("RGB"))
    faces = app.get(img)
    if not faces:
        return []

    # Build face records
    records = []
    for f in faces:
        x0, y0, x1, y1 = f.bbox
        records.append({
            "embedding": f.normed_embedding.astype(np.float32),
            "bbox":      (float(x0), float(y0), float(x1), float(y1)),
            "x_center":  float((x0 + x1) / 2),
            "y_center":  float((y0 + y1) / 2),
            "area":      float((x1 - x0) * (y1 - y0)),
        })

    # Take top N by area
    records.sort(key=lambda d: d["area"], reverse=True)
    records = records[:max_faces]

    # Sort left-to-right by x_center
    records.sort(key=lambda d: d["x_center"])

    return records


# ================================
# LABELING
# ================================

def get_font(size=20):
    """Try to load a monospace font, fall back to default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def draw_labels(image_path: Path, faces: list[dict], output_path: Path):
    """
    Draw bounding boxes and face1..face4 labels on a copy of the image.
    """
    img = PILImage.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    # Scale font size relative to image
    font_size = max(16, min(img.width, img.height) // 30)
    font = get_font(font_size)

    for i, face in enumerate(faces):
        if i >= MAX_FACES:
            break
        color = FACE_COLORS[i]
        label = FACE_LABELS[i]
        x0, y0, x1, y1 = face["bbox"]

        # Draw bbox
        for offset in range(3):  # thicker line
            draw.rectangle(
                [x0 - offset, y0 - offset, x1 + offset, y1 + offset],
                outline=color,
            )

        # Draw label above bbox
        text_bbox = draw.textbbox((0, 0), label, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        tx = x0 + ((x1 - x0) - tw) / 2  # center over face
        ty = y0 - th - 6
        if ty < 0:
            ty = y1 + 4  # put below if no room above

        # Background rectangle for readability
        draw.rectangle([tx - 3, ty - 2, tx + tw + 3, ty + th + 2], fill=(0, 0, 0))
        draw.text((tx, ty), label, fill=color, font=font)

    img.save(output_path)


# ================================
# ANALYSIS
# ================================

def positional_match(seed_faces: list[dict], gen_faces: list[dict],
                     threshold: float = THRESHOLD) -> list[dict]:
    """
    Match face-by-position: seed face i vs gen face i.
    Returns one result per position (up to min(len(seed), len(gen), MAX_FACES)).
    """
    n = min(len(seed_faces), len(gen_faces), MAX_FACES)
    results = []
    for i in range(MAX_FACES):
        if i < n:
            score = float(np.dot(seed_faces[i]["embedding"], gen_faces[i]["embedding"]))
            results.append({
                "position":  i,
                "label":     FACE_LABELS[i],
                "score":     round(score, 4),
                "matched":   score >= threshold,
                "seed_x":    round(seed_faces[i]["x_center"], 1),
                "gen_x":     round(gen_faces[i]["x_center"], 1),
            })
        else:
            results.append({
                "position":  i,
                "label":     FACE_LABELS[i],
                "score":     None,
                "matched":   False,
                "seed_x":    round(seed_faces[i]["x_center"], 1) if i < len(seed_faces) else None,
                "gen_x":     round(gen_faces[i]["x_center"], 1) if i < len(gen_faces) else None,
                "note":      "missing_in_gen" if i >= len(gen_faces) else "missing_in_seed",
            })
    return results


def run_analysis(seed_image: Path, gen_dir: Path, output_dir: Path):
    """Main analysis pipeline."""
    
    # Discover generated images
    gen_files = sorted([
        f for f in gen_dir.iterdir()
        if f.suffix.lower() in SEED_EXTENSIONS
    ])
    n = len(gen_files)
    print(f"Seed image : {seed_image}")
    print(f"Generated  : {n} images in {gen_dir}")
    print(f"Output     : {output_dir}")
    print()

    # Setup output dirs
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(exist_ok=True)

    # Load ArcFace
    print("Loading ArcFace (buffalo_l)...")
    face_app = load_face_model()

    # ---- Detect seed faces ----
    print(f"Detecting faces in seed image...")
    seed_faces = detect_faces(seed_image, face_app)
    n_seed = len(seed_faces)
    print(f"  Seed faces detected: {n_seed} (using top {min(n_seed, MAX_FACES)})")
    for i, sf in enumerate(seed_faces[:MAX_FACES]):
        print(f"    {FACE_LABELS[i]}: x_center={sf['x_center']:.1f}, area={sf['area']:.0f}")

    # Label seed image
    draw_labels(seed_image, seed_faces, labels_dir / f"seed_labeled{seed_image.suffix}")
    print(f"  Labeled seed saved to {labels_dir / f'seed_labeled{seed_image.suffix}'}")
    print()

    # ---- Process generated images ----
    print("Processing generated images...")
    all_gen_faces = []
    per_image_results = []
    per_position_scores = [[] for _ in range(MAX_FACES)]
    per_position_matched = [0 for _ in range(MAX_FACES)]
    n_with_face = 0
    n_no_face = 0
    images_with_any_match = 0

    for idx, gf in enumerate(gen_files):
        gen_faces = detect_faces(gf, face_app)
        all_gen_faces.append(gen_faces)
        n_detected = len(gen_faces)

        if n_detected == 0:
            n_no_face += 1
            per_image_results.append({
                "image":            gf.name,
                "n_faces_detected": 0,
                "positions":        [],
            })
            # Label anyway (no faces to draw)
            draw_labels(gf, [], labels_dir / gf.name)
            continue

        n_with_face += 1

        # Draw labels on generated image
        draw_labels(gf, gen_faces, labels_dir / gf.name)

        # Positional matching
        matches = positional_match(seed_faces, gen_faces)
        per_image_results.append({
            "image":            gf.name,
            "n_faces_detected": n_detected,
            "positions":        matches,
        })

        any_match = False
        for m in matches:
            pos = m["position"]
            if m["score"] is not None:
                per_position_scores[pos].append(m["score"])
                if m["matched"]:
                    per_position_matched[pos] += 1
                    any_match = True
        if any_match:
            images_with_any_match += 1

        if (idx + 1) % 5 == 0 or idx == n - 1:
            print(f"  {idx + 1}/{n} processed")

    # ---- Pairwise consistency (per position) ----
    print("\nComputing pairwise facial consistency per position...")
    per_position_pairwise = []
    for pos in range(MAX_FACES):
        embs = []
        for gen_faces in all_gen_faces:
            if pos < len(gen_faces):
                embs.append(gen_faces[pos]["embedding"])
        if len(embs) >= 2:
            embs_arr = np.array(embs)
            sims = embs_arr @ embs_arr.T
            pw = [float(sims[i, j]) for i in range(len(embs)) for j in range(i + 1, len(embs))]
            per_position_pairwise.append({
                "position":  pos,
                "label":     FACE_LABELS[pos],
                "n_images":  len(embs),
                "mean":      round(float(np.mean(pw)), 4),
                "std":       round(float(np.std(pw)), 4),
            })
        else:
            per_position_pairwise.append({
                "position":  pos,
                "label":     FACE_LABELS[pos],
                "n_images":  len(embs),
                "mean":      None,
                "std":       None,
            })

    # ---- Aggregate pairwise (all faces pooled, matching original metric) ----
    all_embs = []
    for gen_faces in all_gen_faces:
        for gf in gen_faces[:MAX_FACES]:
            all_embs.append(gf["embedding"])
    agg_pairwise_mean = None
    agg_pairwise_std = None
    if len(all_embs) >= 2:
        all_embs_arr = np.array(all_embs)
        sims = all_embs_arr @ all_embs_arr.T
        pw = [float(sims[i, j]) for i in range(len(all_embs)) for j in range(i + 1, len(all_embs))]
        agg_pairwise_mean = round(float(np.mean(pw)), 4)
        agg_pairwise_std = round(float(np.std(pw)), 4)

    # ---- Build results ----
    per_position_summary = []
    for pos in range(MAX_FACES):
        sc = per_position_scores[pos]
        summary = {
            "position":      pos,
            "label":         FACE_LABELS[pos],
            "n_scored":      len(sc),
            "n_matched":     per_position_matched[pos],
            "match_rate":    round(per_position_matched[pos] / n, 4) if n else 0,
            "threshold":     THRESHOLD,
        }
        if sc:
            summary["mean_score"] = round(float(np.mean(sc)), 4)
            summary["std_score"]  = round(float(np.std(sc)), 4)
            summary["max_score"]  = round(float(np.max(sc)), 4)
            summary["min_score"]  = round(float(np.min(sc)), 4)
        else:
            summary["mean_score"] = None
        # Add seed x_center for reference
        if pos < len(seed_faces):
            summary["seed_x_center"] = round(seed_faces[pos]["x_center"], 1)
            summary["seed_area"] = round(seed_faces[pos]["area"], 0)
        per_position_summary.append(summary)

    results = {
        "seed_image":         str(seed_image),
        "n_generated":        n,
        "n_with_face":        n_with_face,
        "n_no_face":          n_no_face,
        "n_seed_faces":       min(n_seed, MAX_FACES),
        "identity_threshold": THRESHOLD,
        "images_with_any_match": images_with_any_match,
        "aggregate_match_rate":  round(images_with_any_match / n, 4) if n else 0,
        "per_position":       per_position_summary,
        "per_position_pairwise": per_position_pairwise,
        "aggregate_pairwise": {
            "mean": agg_pairwise_mean,
            "std":  agg_pairwise_std,
        },
        "per_image":          per_image_results,
        "finished_at":        datetime.now().isoformat(),
    }

    # ---- Write JSON log ----
    log_path = output_dir / "group_analysis_log.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON log saved to {log_path}")

    # ---- Write human-readable results ----
    txt_path = output_dir / "group_analysis_results.txt"
    write_results_txt(results, txt_path)
    print(f"Results saved to {txt_path}")

    return results


def write_results_txt(results: dict, path: Path):
    """Write master-results-style human-readable output."""
    lines = []
    lines.append("GROUP PHOTO ANALYSIS — POSITIONAL ARCFACE")
    lines.append(f"Run completed: {results['finished_at']}")
    lines.append(f"Seed: {results['seed_image']}")
    lines.append(f"Generated images: {results['n_generated']}")
    lines.append(f"Identity threshold: {results['identity_threshold']}")
    lines.append("=" * 64)
    lines.append("")

    lines.append("FACE DETECTION")
    lines.append(f"  Seed faces (top {MAX_FACES})  : {results['n_seed_faces']}")
    lines.append(f"  Gen images with face   : {results['n_with_face']}/{results['n_generated']}")
    lines.append(f"  Gen images no face     : {results['n_no_face']}/{results['n_generated']}")
    lines.append("")

    lines.append("IDENTITY PRESERVATION (positional match)")
    lines.append(f"  Images with any match  : {results['images_with_any_match']}/{results['n_generated']} ({results['aggregate_match_rate']*100:.1f}%)")
    lines.append("")

    for ps in results["per_position"]:
        label = ps["label"]
        nm = ps["n_matched"]
        ns = ps["n_scored"]
        rate = ps["match_rate"]
        mean = ps.get("mean_score")
        seed_x = ps.get("seed_x_center", "N/A")
        lines.append(f"  {label} (seed x={seed_x})")
        lines.append(f"    Matched   : {nm}/{results['n_generated']} ({rate*100:.1f}%)")
        if mean is not None:
            lines.append(f"    Mean score: {mean:.4f}  std: {ps.get('std_score', 0):.4f}")
            lines.append(f"    Range     : [{ps.get('min_score', 0):.4f}, {ps.get('max_score', 0):.4f}]")
        else:
            lines.append(f"    Mean score: N/A (no detections at this position)")
        lines.append("")

    lines.append("FACIAL CONSISTENCY (pairwise, per position)")
    for pp in results["per_position_pairwise"]:
        label = pp["label"]
        if pp["mean"] is not None:
            lines.append(f"  {label}: mean={pp['mean']:.4f}  std={pp['std']:.4f}  (n={pp['n_images']})")
        else:
            lines.append(f"  {label}: insufficient data (n={pp['n_images']})")
    lines.append("")

    agg = results["aggregate_pairwise"]
    if agg["mean"] is not None:
        lines.append(f"  Aggregate (all faces pooled): mean={agg['mean']:.4f}  std={agg['std']:.4f}")
    lines.append("")

    lines.append("PER-IMAGE DETAIL")
    lines.append("-" * 64)
    for img_r in results["per_image"]:
        lines.append(f"  {img_r['image']}  faces={img_r['n_faces_detected']}")
        for p in img_r.get("positions", []):
            score_str = f"{p['score']:.4f}" if p["score"] is not None else "N/A"
            match_str = "MATCH" if p["matched"] else "miss"
            lines.append(f"    {p['label']}: {score_str} [{match_str}]")
    lines.append("")
    lines.append("=" * 64)

    path.write_text("\n".join(lines), encoding="utf-8")


# ================================
# CLI
# ================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Positional ArcFace analysis for group photos")
    parser.add_argument("--seed-image", type=Path, required=True, help="Path to seed image")
    parser.add_argument("--gen-dir", type=Path, required=True, help="Directory of generated images")
    parser.add_argument("--output-dir", type=Path, default=Path("group_analysis_output"),
                        help="Output directory (default: ./group_analysis_output)")
    args = parser.parse_args()

    if not args.seed_image.exists():
        print(f"Error: seed image not found: {args.seed_image}")
        sys.exit(1)
    if not args.gen_dir.is_dir():
        print(f"Error: generated image directory not found: {args.gen_dir}")
        sys.exit(1)

    run_analysis(args.seed_image, args.gen_dir, args.output_dir)
