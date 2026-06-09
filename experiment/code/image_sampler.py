"""
============================================================================================================================================================================================================================================================
Copyright William Donnell-Lonon, 2026. All rights reserved.
The use of this script for further research or development of AI models is permitted under the following conditions:
    1. Attribution: Any use or modification of this script or its derivatives must include clear attribution to the original author, William Donnell-Lonon, and a link to the original source.
    2. Non-Commercial Use: This script may not be used for commercial purposes without explicit permission from the author.
    3. Open Source Derivatives: Any publications that make use of derivative works or modifications of this script must release modified code under the same license and made publicly available on a platform such as GitHub.
    4. No Endorsement: The author does not endorse any use of this script that violates ethical guidelines or promotes harmful content. Users are responsible for ensuring their use of the script complies with all applicable laws and ethical standards.
    5. Reporting Issues: If you discover any issues, bugs, or potential improvements in this script, please report them to the author to help improve the tool for the research community.
    6. Citation: If you use this script in academic research, please cite the original research paper as well.
============================================================================================================================================================================================================================================================

image_sampler.py — Baseline distribution sampler for a single prompt.

Purpose:
    Generate N independent images from a single fixed prompt with no feedback loop.
    Produces a statistical baseline / null-model distribution for comparison against
    telephone.py chain outputs. Used to test whether chain variance is attributable
    to iterative drift or simply reflects the natural stochasticity of the generator.

Features:
- Accepts prompt as CLI argument or interactive input
- Configurable N (number of samples) via --n flag or interactive input
- Checkpoint/resume: safe to interrupt with Ctrl+C and restart
- Atomic image writes (write to .tmp then rename)
- Per-run log.json with all metadata, prompts, outcomes, and CLIP similarities
- Policy violation tracking with count and per-image records
- Persistent terminal UI: scrolling log buffer with pinned stats footer
- Rate-limit and timeout retry logic matching telephone.py
- 5-minute hang timeout per API call
"""

import os
import re
import json
import time
import base64
import signal
import sys
import argparse
import hashlib
import threading
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import requests
from openai import OpenAI

warnings.filterwarnings("ignore")

# ================================
# CONFIGURATION
# ================================

OUTPUT_DIR      = "image_sampler_output"
CHECKPOINT_FILE = "image_sampler_checkpoint.json"

MAX_RETRIES         = 3    # max retries on rate-limit errors
RETRY_DELAY         = 60   # seconds between rate-limit retries
API_TIMEOUT_SECS    = 300  # seconds before a hung API call is a timeout
MAX_TIMEOUT_RETRIES = 3    # consecutive timeouts before aborting

# ---- Image generation ----
IMAGE_MODEL   = "gpt-image-1.5"
IMAGE_SIZE    = "1024x1024"
IMAGE_QUALITY = "medium"   # low | medium | high

# ================================
# CLIENT INIT
# ================================

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY not set. Export it in your environment before running."
    )

client = OpenAI(api_key=API_KEY)

# ================================
# UTILITIES
# ================================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_image_atomic(image_bytes: bytes, dest: Path):
    """
    Writes image bytes to a .tmp file in the same directory, then renames to dest.
    Guarantees dest is either fully written or absent — never a truncated file.
    """
    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(image_bytes)
    tmp.rename(dest)


def fmt_duration(seconds: float) -> str:
    td = timedelta(seconds=int(seconds))
    h, rem = divmod(td.seconds + td.days * 86400, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def slugify(text: str, max_len: int = 40) -> str:
    """Turns a prompt string into a filesystem-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"-+", "-", text)
    return text[:max_len].strip("_-")


# ================================
# HANG TIMEOUT
# ================================

class HangTimeoutError(Exception):
    pass


def call_with_timeout(fn, timeout=API_TIMEOUT_SECS):
    """
    Runs fn() in a thread. Raises HangTimeoutError if it doesn't return
    within `timeout` seconds.
    """
    result    = [None]
    exc       = [None]
    completed = threading.Event()

    def worker():
        try:
            result[0] = fn()
        except Exception as e:
            exc[0] = e
        finally:
            completed.set()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    if not completed.wait(timeout):
        raise HangTimeoutError(f"API call hung for >{timeout}s")
    if exc[0]:
        raise exc[0]
    return result[0]


# ================================
# TERMINAL DISPLAY
# ================================

UI_WIDTH = 78
BAR_WIDTH = 30


def progress_bar(done: int, total: int, width: int = BAR_WIDTH) -> str:
    frac  = done / total if total > 0 else 0
    filled = int(frac * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def separator(char: str = "-", width: int = UI_WIDTH) -> str:
    return char * width


class Display:
    """
    Persistent terminal UI with a scrolling log buffer and a pinned stats footer.
    Matches the telephone.py Display pattern.
    """

    FOOTER_LINES = 7

    def __init__(self):
        self._lock            = threading.Lock()
        self._log_buf         = []
        self._last_term_w     = UI_WIDTH
        self._live            = False
        self._timer           = None
        self._paused_seconds  = 0.0
        self._pause_start     = None
        self._start_time      = datetime.now()

        # Stats updated by caller
        self.run_label:     str = ""
        self.prompt_label:  str = ""
        self.done:          int = 0
        self.total:         int = 0
        self.policy_blocks: int = 0
        self.errors:        int = 0
        self.stage:         str = ""

    # ---- timing ----

    def record_pause(self):
        self._pause_start = time.monotonic()

    def record_resume(self):
        if self._pause_start is not None:
            self._paused_seconds += time.monotonic() - self._pause_start
            self._pause_start = None

    def active_seconds(self) -> float:
        elapsed = (datetime.now() - self._start_time).total_seconds()
        return max(0.0, elapsed - self._paused_seconds)

    # ---- lifecycle ----

    def start_live(self):
        self._live = True
        if hasattr(signal, "SIGWINCH"):
            signal.signal(signal.SIGWINCH, self._on_resize)
        self._tick()

    def _on_resize(self, signum, frame):
        self.redraw()

    def _tick(self):
        if not self._live:
            return
        self.redraw()
        self._timer = threading.Timer(1.0, self._tick)
        self._timer.daemon = True
        self._timer.start()

    def _build_panel(self, w: int = UI_WIDTH) -> list[str]:
        lines = []
        lines.append(separator("═", w))

        # row 1: run label + elapsed
        elapsed = fmt_duration(self.active_seconds())
        left1   = f"  {self.run_label}"
        right1  = f"{elapsed}  "
        gap1    = max(1, w - len(left1) - len(right1))
        lines.append(left1 + " " * gap1 + right1)

        # row 2: progress bar
        bar  = progress_bar(self.done, self.total)
        pct  = f"{self.done}/{self.total}"
        left2 = f"  {bar} {pct}"
        right2 = f"{self.stage}  "
        gap2  = max(1, w - len(left2) - len(right2))
        lines.append(left2 + " " * gap2 + right2)

        lines.append(separator("-", w))

        # row 3: policy blocks | errors
        left3  = f"  policy blocks: {self.policy_blocks}"
        right3 = f"errors: {self.errors}  "
        gap3   = max(1, w - len(left3) - len(right3))
        lines.append(left3 + " " * gap3 + right3)

        # row 4: prompt preview
        prompt_preview = self.prompt_label[:w - 4]
        lines.append(f"  {prompt_preview}")

        lines.append(separator("═", w))
        return lines

    def _term_width(self) -> int:
        try:
            return os.get_terminal_size().columns
        except OSError:
            return UI_WIDTH

    def redraw(self):
        with self._lock:
            w      = self._term_width()
            panel  = self._build_panel(w)
            nfooter = len(panel)

            # how many log lines fit above the footer
            try:
                rows = os.get_terminal_size().lines
            except OSError:
                rows = 24
            log_space = max(0, rows - nfooter - 1)
            visible   = self._log_buf[-log_space:] if log_space > 0 else []

            # clear screen and redraw
            sys.stdout.write("\033[2J\033[H")
            for line in visible:
                sys.stdout.write(line[:w] + "\n")
            # fill blank lines between log and footer
            for _ in range(log_space - len(visible)):
                sys.stdout.write("\n")
            for line in panel:
                sys.stdout.write(line[:w] + "\n")
            sys.stdout.flush()

    def log(self, text: str = ""):
        with self._lock:
            self._log_buf.append(str(text))
        self.redraw()

    def print_plain(self, text: str = ""):
        print(text)

    def input_plain(self, prompt: str) -> str:
        return input(prompt)

    def finalize(self):
        if self._timer:
            self._timer.cancel()
        self._live = False
        elapsed = fmt_duration(self.active_seconds())
        summary = (
            f"\n{'=' * UI_WIDTH}\n"
            f"  DONE  {self.done}/{self.total} images"
            f"  |  policy blocks: {self.policy_blocks}"
            f"  |  errors: {self.errors}"
            f"  |  elapsed: {elapsed}\n"
            f"{'=' * UI_WIDTH}"
        )
        sys.stdout.write("\033[2J\033[H")
        for line in self._log_buf:
            print(line)
        print(summary)


display = Display()


# ================================
# PAUSE HANDLER
# ================================

class PauseHandler:
    def __init__(self):
        self.paused = False
        self.armed  = False

    def arm(self):
        self.armed = True
        signal.signal(signal.SIGINT, self._on_first_interrupt)

    def _on_first_interrupt(self, signum, frame):
        self.paused = True
        display.record_pause()
        display.log("Ctrl+C — pausing after this sample. Ctrl+C again to force quit.")
        signal.signal(signal.SIGINT, self._on_second_interrupt)

    def _on_second_interrupt(self, signum, frame):
        display.log("FORCE QUIT — progress saved.")
        sys.exit(1)

    def check(self):
        if self.paused:
            display.log("Paused. Press Enter to resume or Ctrl+C to quit.")
            try:
                input()
            except KeyboardInterrupt:
                display.log("Quit.")
                sys.exit(0)
            self.paused = False
            self.armed  = True
            display.record_resume()
            signal.signal(signal.SIGINT, self._on_first_interrupt)


pause_handler = PauseHandler()


def get_input(prompt: str) -> str:
    try:
        return input(prompt)
    except KeyboardInterrupt:
        if pause_handler.armed:
            pause_handler.paused = True
            display.record_pause()
            return ""
        print("\nCancelled.")
        sys.exit(0)


# ================================
# IMAGE GENERATION
# ================================

def generate_image(
    prompt: str,
    dest: Path,
    retry_count: int = 0,
    timeout_count: int = 0,
) -> tuple[bool, str | None, str | None]:
    """
    Generates a single image from prompt and writes it atomically to dest.
    Returns (success, error_type, error_message).
    """
    try:
        def call():
            return client.images.generate(
                model=IMAGE_MODEL,
                prompt=prompt,
                size=IMAGE_SIZE,
                quality=IMAGE_QUALITY,
                n=1,
            )

        resp = call_with_timeout(call)
        item = resp.data[0]

        b64 = getattr(item, "b64_json", None)
        if b64:
            write_image_atomic(base64.b64decode(b64), dest)
            return True, None, None

        url = getattr(item, "url", None)
        if url:
            r = requests.get(url, timeout=(10, 120))
            r.raise_for_status()
            write_image_atomic(r.content, dest)
            return True, None, None

        return False, "NO_IMAGE_DATA", "API returned neither b64_json nor url."

    except HangTimeoutError as e:
        timeout_count += 1
        if timeout_count < MAX_TIMEOUT_RETRIES:
            display.log(f"[generate] TIMEOUT ({timeout_count}/{MAX_TIMEOUT_RETRIES}) — retrying...")
            return generate_image(prompt, dest, retry_count, timeout_count)
        return False, "TIMEOUT", str(e)

    except Exception as e:
        msg   = str(e)
        lower = msg.lower()

        if "rate limit" in lower or "429" in lower:
            if retry_count < MAX_RETRIES:
                display.log(
                    f"[generate] rate limit — waiting {RETRY_DELAY}s "
                    f"(retry {retry_count + 1}/{MAX_RETRIES})..."
                )
                time.sleep(RETRY_DELAY)
                return generate_image(prompt, dest, retry_count + 1, timeout_count)
            return False, "RATE_LIMIT_EXCEEDED", msg

        if "insufficient_quota" in lower or "exceeded your current quota" in lower:
            return False, "INSUFFICIENT_QUOTA", msg

        if (
            "content policy" in lower
            or "safety system" in lower
            or "content_policy" in lower
            or "guardrails" in lower
        ):
            return False, "CONTENT_POLICY_VIOLATION", msg

        if retry_count < 1:
            display.log(f"[generate] error: {msg[:80]}... retrying")
            time.sleep(5)
            return generate_image(prompt, dest, 1, timeout_count)

        return False, "UNKNOWN_ERROR", msg


# ================================
# CHECKPOINT
# ================================

def load_checkpoint(run_dir: Path) -> dict | None:
    cp = run_dir / CHECKPOINT_FILE
    if cp.exists():
        try:
            return json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def save_checkpoint(run_dir: Path, data: dict):
    cp = run_dir / CHECKPOINT_FILE
    tmp = cp.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.rename(cp)


# ================================
# MAIN RUN
# ================================

def run_sampler(prompt: str, n: int, run_dir: Path, seed_image: Path | None = None, multi_face: bool = False):
    """
    Core loop: generate n images from the fixed prompt, checkpoint after each.
    """
    images_dir = run_dir / "images"
    ensure_dir(images_dir)

    # load or init checkpoint
    cp = load_checkpoint(run_dir)
    if cp is not None:
        if cp.get("prompt") != prompt or cp.get("n") != n:
            display.log("ERROR: checkpoint exists with different prompt/n. Delete the run folder to start fresh.")
            sys.exit(1)
        completed   = set(cp.get("completed", []))
        policy_hits = cp.get("policy_hits", [])
        errors_log  = cp.get("errors_log", [])
        display.log(f"Resuming: {len(completed)}/{n} already done.")
    else:
        completed   = set()
        policy_hits = []
        errors_log  = []
        save_checkpoint(run_dir, {
            "prompt":      prompt,
            "n":           n,
            "completed":   [],
            "policy_hits": [],
            "errors_log":  [],
            "started_at":  datetime.now().isoformat(),
        })

    # configure display
    display.run_label    = f"image_sampler  n={n}"
    display.prompt_label = prompt[:70] + ("..." if len(prompt) > 70 else "")
    display.total        = n
    display.done         = len(completed)
    display.policy_blocks = len(policy_hits)
    display.errors        = len(errors_log)
    display.start_live()
    pause_handler.arm()

    for i in range(1, n + 1):
        pause_handler.check()

        if i in completed:
            continue

        dest = images_dir / f"sample_{i:04d}.png"
        display.stage = "[generate]"
        display.log(f"  sample {i:04d}/{n} ...")

        success, err_type, err_msg = generate_image(prompt, dest)

        if success:
            completed.add(i)
            display.done = len(completed)
            display.log(f"  ✓ sample {i:04d} saved → {dest.name}")
        else:
            if err_type == "CONTENT_POLICY_VIOLATION":
                policy_hits.append({"sample": i, "error": err_msg})
                display.policy_blocks = len(policy_hits)
                display.log(f"  ✗ sample {i:04d} POLICY BLOCK — {(err_msg or '')[:60]}")
                # still mark as "attempted" so we don't loop forever on a consistently blocked prompt
                completed.add(i)
                display.done = len(completed)
            elif err_type == "INSUFFICIENT_QUOTA":
                display.log("  QUOTA EXHAUSTED — aborting run.")
                break
            else:
                errors_log.append({"sample": i, "error_type": err_type, "error_msg": err_msg})
                display.errors = len(errors_log)
                display.log(f"  ✗ sample {i:04d} ERROR ({err_type}): {(err_msg or '')[:60]}")

        # save checkpoint after every sample
        save_checkpoint(run_dir, {
            "prompt":      prompt,
            "n":           n,
            "completed":   sorted(completed),
            "policy_hits": policy_hits,
            "errors_log":  errors_log,
            "started_at":  cp["started_at"] if cp else datetime.now().isoformat(),
            "updated_at":  datetime.now().isoformat(),
        })

    display.stage = ""
    # ---- write final log ----
    successful_samples = [
        i for i in sorted(completed)
        if not any(ph["sample"] == i for ph in policy_hits)
    ]

    log = {
        "run_id":             run_dir.name,
        "prompt":             prompt,
        "n_requested":        n,
        "n_generated":        len(successful_samples),
        "n_policy_blocked":   len(policy_hits),
        "n_errors":           len(errors_log),
        "completion_rate":    round(len(successful_samples) / n, 4) if n > 0 else 0,
        "policy_block_rate":  round(len(policy_hits) / n, 4) if n > 0 else 0,
        "policy_hits":        policy_hits,
        "errors_log":         errors_log,
        "image_model":        IMAGE_MODEL,
        "image_size":         IMAGE_SIZE,
        "image_quality":      IMAGE_QUALITY,
        "finished_at":        datetime.now().isoformat(),
    }

    log_path = run_dir / "log.json"
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    display.log(f"\nLog written → {log_path}")

    display.finalize()
    print(f"\nOutput directory: {run_dir.resolve()}")
    print(f"Images:           {images_dir.resolve()}")

    # ---- Post-run analysis ----
    run_analysis(run_dir, prompt, policy_hits, seed_image=seed_image, multi_face=multi_face)


# ================================
# POST-RUN ANALYSIS
# ================================

def _load_clip_model(device=None):
    try:
        import open_clip
        import torch
    except ImportError:
        print("\n  [analysis] open_clip_torch / torch not installed — skipping CLIP.")
        print("  Install with:  pip install open_clip_torch torch torchvision")
        return None, None, None

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"  Loading CLIP (ViT-B-32, LAION-2B) on {device}...")
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
    except Exception:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="laion2b_s34b_b88k"
        )
    model = model.to(device).eval()
    return model, preprocess, device


def _clip_embed(image_path: Path, model, preprocess, device):
    import torch
    from PIL import Image as PILImage
    import numpy as np
    img    = PILImage.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy()[0]


def _load_face_model():
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        print("\n  [analysis] insightface not installed — skipping face analysis.")
        print("  Install with:  pip install insightface onnxruntime")
        return None

    print("  Loading ArcFace (buffalo_l)...")
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def _face_embed(image_path: Path, app):
    """Single-face mode: returns embedding of the largest detected face."""
    from PIL import Image as PILImage
    import numpy as np
    try:
        img   = np.array(PILImage.open(image_path).convert("RGB"))
        faces = app.get(img)
        if not faces:
            return None, "no_face"
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        return face.normed_embedding.astype(np.float32), "ok"
    except Exception as e:
        return None, f"error:{str(e)[:60]}"


def _face_embed_all(image_path: Path, app) -> list[dict]:
    """
    Multi-face mode: returns all detected faces sorted left-to-right by
    bounding box x-center. Each entry is a dict with keys:
      embedding, bbox, x_center, area
    Returns empty list if no faces detected or on error.
    """
    from PIL import Image as PILImage
    import numpy as np
    try:
        img   = np.array(PILImage.open(image_path).convert("RGB"))
        faces = app.get(img)
        if not faces:
            return []
        result = []
        for f in faces:
            x0, y0, x1, y1 = f.bbox
            result.append({
                "embedding": f.normed_embedding.astype(np.float32),
                "bbox":      (float(x0), float(y0), float(x1), float(y1)),
                "x_center":  float((x0 + x1) / 2),
                "area":      float((x1 - x0) * (y1 - y0)),
            })
        # Sort left to right by x_center
        result.sort(key=lambda d: d["x_center"])
        return result
    except Exception:
        return []


def _match_faces(seed_faces: list[dict], gen_faces: list[dict],
                 threshold: float = 0.3) -> list[dict]:
    """
    Best-match assignment: for each seed face, find the generated face with
    the highest ArcFace cosine similarity. Each generated face can only be
    claimed once (greedy, sorted by best score descending).

    Returns a list of match dicts (one per seed face):
      seed_idx, gen_idx, score, matched (bool — score > threshold)
    """
    import numpy as np

    if not seed_faces or not gen_faces:
        return [{"seed_idx": i, "gen_idx": None, "score": None, "matched": False}
                for i in range(len(seed_faces))]

    n_seed = len(seed_faces)
    n_gen  = len(gen_faces)

    # Build score matrix (n_seed x n_gen)
    seed_embs = np.array([f["embedding"] for f in seed_faces])
    gen_embs  = np.array([f["embedding"] for f in gen_faces])
    scores    = seed_embs @ gen_embs.T  # (n_seed, n_gen)

    # Greedy best-match: take highest score pair, remove both, repeat
    claimed_gen = set()
    matches = [None] * n_seed

    # Flatten all (score, seed_idx, gen_idx) tuples, sort descending
    candidates = sorted(
        [(float(scores[s, g]), s, g) for s in range(n_seed) for g in range(n_gen)],
        reverse=True,
    )

    for score, s_idx, g_idx in candidates:
        if matches[s_idx] is not None:
            continue  # seed face already matched
        if g_idx in claimed_gen:
            continue  # gen face already claimed
        matches[s_idx] = {
            "seed_idx": s_idx,
            "gen_idx":  g_idx,
            "score":    round(score, 4),
            "matched":  score >= threshold,
        }
        claimed_gen.add(g_idx)

    # Fill any unmatched seed faces (more seed faces than gen faces)
    for i in range(n_seed):
        if matches[i] is None:
            matches[i] = {"seed_idx": i, "gen_idx": None, "score": None, "matched": False}

    return matches


def run_analysis(run_dir: Path, prompt: str, policy_hits: list,
                 seed_image: Path | None = None,
                 multi_face: bool = False):
    """
    Runs CLIP and ArcFace analysis over all successfully generated images.

    Two modes depending on whether a seed_image is provided:

      seed_image=None  — variance-only mode
        CLIP: all pairwise cosine similarities among generated images.
        Face: all pairwise ArcFace similarities among generated images.
        Answers: "how spread out is this distribution?"

      seed_image=Path  — ground-truth mode (adds vs-seed metrics on top)
        CLIP: pairwise variance + per-image similarity to seed.
        Face: pairwise variance + per-image face similarity to seed face.
        Answers: "how spread out AND how close to the original?"

    Results are printed and appended to log.json under "analysis".
    """
    import numpy as np

    images_dir = run_dir / "images"
    image_files = sorted(
        f for f in images_dir.glob("sample_*.png") if f.stat().st_size > 0
    )
    blocked_samples = {ph["sample"] for ph in policy_hits}
    image_files = [
        f for f in image_files
        if int(f.stem.split("_")[1]) not in blocked_samples
    ]

    if not image_files:
        print("\n  [analysis] No images to analyse.")
        return {}

    n = len(image_files)
    if seed_image and multi_face:
        mode_label = f"multi-face ground-truth  (seed: {seed_image.name})"
    elif seed_image:
        mode_label = f"ground-truth mode  (seed: {seed_image.name})"
    else:
        mode_label = "variance-only mode"
    print(f"\n{'='*UI_WIDTH}")
    print(f"  POST-RUN ANALYSIS  ({n} images)  —  {mode_label}")
    print(f"{'='*UI_WIDTH}")

    # ================================
    # CLIP
    # ================================
    clip_model, clip_preprocess, clip_device = _load_clip_model()
    clip_result = None

    if clip_model is not None:
        try:
            # Embed all generated images
            print(f"  Embedding {n} images with CLIP...")
            embeddings = []
            for i, f in enumerate(image_files, 1):
                print(f"    {i}/{n}...", end="\r")
                embeddings.append(_clip_embed(f, clip_model, clip_preprocess, clip_device))
            print()
            embeddings = np.array(embeddings)  # (n, d)

            # Pairwise variance (always computed)
            sims      = embeddings @ embeddings.T
            pw_scores = [float(sims[i, j]) for i in range(n) for j in range(i+1, n)]
            pw_mean   = float(np.mean(pw_scores))
            pw_std    = float(np.std(pw_scores))
            pw_min    = float(np.min(pw_scores))
            pw_max    = float(np.max(pw_scores))

            print(f"\n  CLIP — pairwise variance ({len(pw_scores)} pairs):")
            print(f"    mean  = {pw_mean:.4f}")
            print(f"    std   = {pw_std:.4f}")
            print(f"    min   = {pw_min:.4f}")
            print(f"    max   = {pw_max:.4f}")

            clip_result = {
                "pairwise": {
                    "n_pairs": len(pw_scores),
                    "mean": round(pw_mean, 4),
                    "std":  round(pw_std,  4),
                    "min":  round(pw_min,  4),
                    "max":  round(pw_max,  4),
                }
            }

            # Ground-truth: similarity of each image to seed
            if seed_image is not None:
                if not seed_image.exists():
                    print(f"  [CLIP] WARNING: seed image not found at {seed_image} — skipping vs-seed.")
                else:
                    print(f"  CLIP — scoring each image vs seed...")
                    seed_emb    = _clip_embed(seed_image, clip_model, clip_preprocess, clip_device)
                    vs_seed     = [float(np.dot(seed_emb, e)) for e in embeddings]
                    vs_mean     = float(np.mean(vs_seed))
                    vs_std      = float(np.std(vs_seed))
                    vs_min      = float(np.min(vs_seed))
                    vs_max      = float(np.max(vs_seed))

                    print(f"\n  CLIP — vs seed ({seed_image.name}, n={n}):")
                    print(f"    mean  = {vs_mean:.4f}")
                    print(f"    std   = {vs_std:.4f}")
                    print(f"    min   = {vs_min:.4f}")
                    print(f"    max   = {vs_max:.4f}")

                    clip_result["vs_seed"] = {
                        "seed_image": str(seed_image),
                        "n":    n,
                        "mean": round(vs_mean, 4),
                        "std":  round(vs_std,  4),
                        "min":  round(vs_min,  4),
                        "max":  round(vs_max,  4),
                    }

        except Exception as e:
            print(f"  [CLIP] Error: {e}")

    # ================================
    # ArcFace
    # ================================
    face_app    = _load_face_model()
    face_result = None

    if face_app is not None:
        try:
            THRESHOLD = 0.3

            # ---- Embed all generated images ----
            print(f"\n  Embedding {n} images with ArcFace...")
            # In multi-face mode we collect all faces per image.
            # In single-face mode we just take the largest face.
            all_gen_faces = []   # list of lists (one list per image)
            no_face = errors = 0
            for i, f in enumerate(image_files, 1):
                print(f"    {i}/{n}...", end="\r")
                if multi_face:
                    faces = _face_embed_all(f, face_app)
                    all_gen_faces.append(faces)
                    if not faces:
                        no_face += 1
                else:
                    emb, status = _face_embed(f, face_app)
                    if status == "no_face":
                        no_face += 1
                        all_gen_faces.append([])
                    elif status != "ok":
                        errors += 1
                        all_gen_faces.append([])
                    else:
                        all_gen_faces.append([{"embedding": emb}])
            print()

            # Flatten to single-face list for pairwise variance
            valid_embs = np.array([
                faces[0]["embedding"]
                for faces in all_gen_faces if faces
            ])
            n_valid = len(valid_embs)

            face_result = {
                "mode":             "multi_face" if multi_face else "single_face",
                "n_images":         n,
                "n_with_face":      n_valid,
                "n_no_face":        no_face,
                "n_errors":         errors,
            }

            # Pairwise variance (primary face per image)
            if n_valid >= 2:
                sims_f    = valid_embs @ valid_embs.T
                pw_f      = [float(sims_f[i, j])
                             for i in range(n_valid) for j in range(i+1, n_valid)]
                pw_f_mean = float(np.mean(pw_f))
                pw_f_std  = float(np.std(pw_f))
                pw_f_min  = float(np.min(pw_f))
                pw_f_max  = float(np.max(pw_f))
                above     = sum(1 for s in pw_f if s > THRESHOLD)

                print(f"\n  ArcFace — pairwise variance ({len(pw_f)} pairs, {n_valid}/{n} faces detected):")
                print(f"    mean  = {pw_f_mean:.4f}")
                print(f"    std   = {pw_f_std:.4f}")
                print(f"    min   = {pw_f_min:.4f}")
                print(f"    max   = {pw_f_max:.4f}")
                print(f"    pairs above identity threshold ({THRESHOLD}): "
                      f"{above}/{len(pw_f)} ({100*above/len(pw_f):.1f}%)")

                face_result["pairwise"] = {
                    "n_pairs":               len(pw_f),
                    "mean":                  round(pw_f_mean, 4),
                    "std":                   round(pw_f_std,  4),
                    "min":                   round(pw_f_min,  4),
                    "max":                   round(pw_f_max,  4),
                    "pairs_above_threshold": above,
                    "identity_threshold":    THRESHOLD,
                }
            else:
                print(f"  [ArcFace] Not enough faces detected ({n_valid}/{n}) for pairwise stats.")

            # ---- vs-seed scoring ----
            if seed_image is not None and seed_image.exists():

                if multi_face:
                    # ---- MULTI-FACE MODE ----
                    seed_faces = _face_embed_all(seed_image, face_app)
                    if not seed_faces:
                        print(f"  [ArcFace] WARNING: no faces detected in seed image.")
                    else:
                        n_seed_faces = len(seed_faces)
                        print(f"\n  ArcFace — multi-face matching")
                        print(f"  Seed has {n_seed_faces} face(s) (left→right: face 0..{n_seed_faces-1})")
                        print(f"  Matching strategy: best-match with threshold {THRESHOLD}\n")

                        # Per seed-face: collect match scores across all generated images
                        per_face_scores  = [[] for _ in range(n_seed_faces)]
                        per_face_matched = [0  for _ in range(n_seed_faces)]
                        per_image_results = []

                        for img_idx, gen_faces in enumerate(all_gen_faces):
                            if not gen_faces:
                                per_image_results.append({
                                    "image":   image_files[img_idx].name,
                                    "n_faces_detected": 0,
                                    "matches": [],
                                })
                                continue

                            matches = _match_faces(seed_faces, gen_faces, THRESHOLD)
                            img_result = {
                                "image":            image_files[img_idx].name,
                                "n_faces_detected": len(gen_faces),
                                "matches":          matches,
                            }
                            per_image_results.append(img_result)

                            for m in matches:
                                s_idx = m["seed_idx"]
                                if m["score"] is not None:
                                    per_face_scores[s_idx].append(m["score"])
                                if m["matched"]:
                                    per_face_matched[s_idx] += 1

                        # Print per-face summary
                        per_face_summary = []
                        for fi in range(n_seed_faces):
                            scores_fi = per_face_scores[fi]
                            n_matched = per_face_matched[fi]
                            if scores_fi:
                                mean_s = float(np.mean(scores_fi))
                                std_s  = float(np.std(scores_fi))
                                max_s  = float(np.max(scores_fi))
                            else:
                                mean_s = std_s = max_s = None

                            label = f"seed face {fi}"
                            match_pct = 100 * n_matched / n if n > 0 else 0
                            if mean_s is not None:
                                print(f"  [{label}]  mean={mean_s:.4f}  std={std_s:.4f}  "
                                      f"max={max_s:.4f}  "
                                      f"matched: {n_matched}/{n} ({match_pct:.1f}%)")
                            else:
                                print(f"  [{label}]  no matches found  (0/{n})")

                            per_face_summary.append({
                                "seed_face_idx":  fi,
                                "seed_x_center":  round(seed_faces[fi]["x_center"], 1),
                                "n_matched":      n_matched,
                                "match_rate":     round(n_matched / n, 4) if n > 0 else 0,
                                "mean_score":     round(mean_s, 4) if mean_s is not None else None,
                                "std_score":      round(std_s,  4) if std_s  is not None else None,
                                "max_score":      round(max_s,  4) if max_s  is not None else None,
                                "identity_threshold": THRESHOLD,
                            })

                        face_result["vs_seed_multi"] = {
                            "seed_image":       str(seed_image),
                            "n_seed_faces":     n_seed_faces,
                            "per_face":         per_face_summary,
                            "per_image":        per_image_results,
                        }

                else:
                    # ---- SINGLE-FACE MODE (original behaviour) ----
                    seed_face_emb, seed_face_status = _face_embed(seed_image, face_app)
                    if seed_face_status != "ok":
                        print(f"  [ArcFace] WARNING: no face detected in seed image ({seed_face_status}).")
                    elif n_valid >= 1:
                        vs_f      = [float(np.dot(seed_face_emb, e)) for e in valid_embs]
                        vs_f_mean = float(np.mean(vs_f))
                        vs_f_std  = float(np.std(vs_f))
                        vs_f_min  = float(np.min(vs_f))
                        vs_f_max  = float(np.max(vs_f))
                        vs_above  = sum(1 for s in vs_f if s > THRESHOLD)

                        print(f"\n  ArcFace — vs seed face ({seed_image.name}, n={n_valid}):")
                        print(f"    mean  = {vs_f_mean:.4f}")
                        print(f"    std   = {vs_f_std:.4f}")
                        print(f"    min   = {vs_f_min:.4f}")
                        print(f"    max   = {vs_f_max:.4f}")
                        print(f"    above identity threshold ({THRESHOLD}): "
                              f"{vs_above}/{n_valid} ({100*vs_above/n_valid:.1f}%)")

                        face_result["vs_seed"] = {
                            "seed_image":        str(seed_image),
                            "n":                 n_valid,
                            "mean":              round(vs_f_mean, 4),
                            "std":               round(vs_f_std,  4),
                            "min":               round(vs_f_min,  4),
                            "max":               round(vs_f_max,  4),
                            "above_threshold":   vs_above,
                            "identity_threshold": THRESHOLD,
                        }

        except Exception as e:
            print(f"  [ArcFace] Error: {e}")

    # ---- Assemble and persist ----
    analysis = {
        "n_analysed":  n,
        "mode":        "ground_truth" if seed_image else "variance_only",
        "seed_image":  str(seed_image) if seed_image else None,
        "clip":        clip_result,
        "face":        face_result,
    }

    log_path = run_dir / "log.json"
    if log_path.exists():
        log = json.loads(log_path.read_text(encoding="utf-8"))
        log["analysis"] = analysis
        log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
        print(f"\n  Analysis results appended to log.json")

    print(f"{'='*UI_WIDTH}\n")
    return analysis


# ================================
# ENTRYPOINT
# ================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate N independent images from a single fixed prompt (no feedback loop)."
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default=None,
        help="The prompt to sample. If omitted, you'll be asked interactively.",
    )
    parser.add_argument(
        "--n", "-n",
        type=int,
        default=None,
        help="Number of images to generate. If omitted, you'll be asked interactively.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=OUTPUT_DIR,
        help=f"Root output directory (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--seed-image", "-s",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a seed image for ground-truth comparison. "
             "If omitted, only pairwise variance is computed.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="RUN_ID",
        help="Resume a specific run by its run ID (folder name inside output dir).",
    )
    parser.add_argument(
        "--multi-face",
        action="store_true",
        default=False,
        help="Enable multi-face mode for group seed images. Detects all faces "
             "in each generated image and matches them to seed faces using "
             "best-match ArcFace scoring with identity threshold.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        metavar="NAME",
        help="Human-readable name for this run (e.g. 'jfk_iter1'). "
             "Used instead of the auto-generated prompt slug. "
             "Timestamp is always prepended.",
    )
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    ensure_dir(out_root)

    # ---- resume path ----
    if args.resume:
        run_dir = out_root / args.resume
        if not run_dir.exists():
            print(f"ERROR: run folder not found: {run_dir}")
            sys.exit(1)
        cp = load_checkpoint(run_dir)
        if cp is None:
            print(f"ERROR: no checkpoint found in {run_dir}")
            sys.exit(1)
        prompt = cp["prompt"]
        n      = cp["n"]
        # seed_image can be re-specified on resume (or passed via --seed-image)
        seed_image = Path(args.seed_image) if args.seed_image else None
        if seed_image and not seed_image.exists():
            print(f"  WARNING: seed image not found at {seed_image} — skipping ground-truth scoring.")
            seed_image = None
        print(f"Resuming run: {args.resume}")
        print(f"  Prompt     : {prompt}")
        print(f"  N          : {n}")
        print(f"  Done       : {len(cp.get('completed', []))}/{n}")
        print(f"  Seed image : {seed_image if seed_image else '(none — variance only)'}")
    else:
        # ---- interactive / CLI prompt collection ----
        if args.prompt:
            prompt = args.prompt.strip()
        else:
            print("\n" + "=" * UI_WIDTH)
            print("  image_sampler — baseline distribution generator")
            print("=" * UI_WIDTH)
            print("  Paste your prompt (multi-line captions are fine).")
            print("  When done, type END on its own line and press Enter.")
            print()
            print("  Prompt (END to finish):")
            lines = []
            while True:
                try:
                    line = input()
                except KeyboardInterrupt:
                    print("\nCancelled.")
                    sys.exit(0)
                if line.strip().upper() == "END":
                    break
                lines.append(line)
            prompt = "\n".join(lines).strip()
            if not prompt:
                print("No prompt entered. Exiting.")
                sys.exit(0)

        if args.n:
            n = args.n
        else:
            print()
            raw = get_input("  Number of samples (N): ").strip()
            try:
                n = int(raw)
                if n < 1:
                    raise ValueError
            except ValueError:
                print("Invalid number. Exiting.")
                sys.exit(1)

        # ---- build run directory ----
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.name:
            run_id = re.sub(r"[^a-z0-9_-]", "_", args.name.strip().lower())[:50]
        else:
            print()
            raw_name = get_input("  Run name [blank for auto]: ").strip()
            if raw_name:
                run_id = re.sub(r"[^a-z0-9_-]", "_", raw_name.lower())[:50]
            else:
                run_id = f"{ts}_{slugify(prompt)}"
        run_dir = out_root / run_id
        ensure_dir(run_dir)

        # ---- seed image (optional) ----
        if args.seed_image:
            seed_image = Path(args.seed_image)
        else:
            print()
            print("  Seed image path (optional).")
            print("  If provided, each generated image will also be scored against it.")
            print("  Leave blank to run variance-only analysis.")
            raw_seed = get_input("  Seed image path [blank to skip]: ").strip()
            seed_image = Path(raw_seed) if raw_seed else None

        if seed_image is not None and not seed_image.exists():
            print(f"  WARNING: seed image not found at {seed_image} — will skip ground-truth scoring.")
            seed_image = None

        # ---- confirm ----
        print()
        print("=" * UI_WIDTH)
        print(f"  Run ID     : {run_id}")
        print(f"  Prompt     : {prompt}")
        print(f"  N          : {n}")
        print(f"  Model      : {IMAGE_MODEL}  {IMAGE_SIZE}  quality={IMAGE_QUALITY}")
        print(f"  Seed image : {seed_image if seed_image else '(none — variance only)'}")
        print(f"  Output     : {run_dir.resolve()}")
        print("=" * UI_WIDTH)
        confirm = get_input("  Start? [Y/n]: ").strip().lower()
        if confirm not in ("", "y", "yes"):
            print("Aborted.")
            sys.exit(0)

    run_sampler(prompt, n, run_dir, seed_image=seed_image, multi_face=args.multi_face)


if __name__ == "__main__":
    main()