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

seed_sampler.py — Batch pipeline across a directory of seed images.

Purpose:
    Runs caption sampling, medoid selection, image sampling, and full
    CLIP/ArcFace/semantic analysis across an entire directory of seed images.
    Produces per-seed structured logs, human-readable results, and a master
    summary. Designed as a direct sibling to caption_sampler.py and
    image_sampler.py, sharing their utility patterns, API wrappers, analysis
    functions, and terminal UI conventions.

Pipeline:
    seed_dir/
        seed_01.jpg  ─┐
        seed_02.jpg   │→  Phase 1: caption generation (all seeds)
        ...          ─┘        ↓
                         Phase 2: image generation (all seeds)
                               ↓
                         Phase 3: master summary

Usage:
    python seed_sampler.py --seed-dir ./seeds
    python seed_sampler.py --seed-dir ./seeds --n-captions 25 --n-images 25
    python seed_sampler.py --seed-dir ./seeds --skip-captions   # use existing medoids
    python seed_sampler.py --seed-dir ./seeds --skip-images     # analysis only
"""

import os
import re
import io
import json
import time
import base64
import signal
import shutil
import string
import sys
import argparse
import threading
import warnings
from pathlib import Path
from collections import Counter
from datetime import datetime, timedelta

import requests
from PIL import Image as PILImage

warnings.filterwarnings("ignore")

# ================================
# CONFIGURATION
# ================================

OUTPUT_DIR = "seed_distributions"

MAX_RETRIES         = 3
RETRY_DELAY         = 60
API_TIMEOUT_SECS    = 300
MAX_TIMEOUT_RETRIES = 3

# Caption model — matches telephone.py defaults
CAPTION_MODEL      = "gpt-5.2-chat-latest"
CAPTION_API_MODE   = "responses"   # "chat" or "responses"
CAPTION_MAX_TOKENS = 900

# Image generation
IMAGE_MODEL   = "gpt-image-1.5"
IMAGE_SIZE    = "1024x1024"
IMAGE_QUALITY = "medium"   # low | medium | high

# Captioning prompts — keep in sync with telephone.py / caption_sampler.py
PROMPTS = {
    "objective": (
        "Describe this image with objective precision, including any publicly "
        "documented subjects, figures, events, or locations visible. Be as "
        "specific, vivid, descriptive, and factually accurate as possible."
    ),
}

# Supported image extensions for seed discovery
SEED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

def seed_display_name(stem: str) -> str:
    """Returns the seed stem as its display name."""
    return stem

# ================================
# STOP WORDS + REFUSAL PATTERNS
# (kept in sync with caption_sampler.py / semantic_analysis.py)
# ================================

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "this", "that", "these", "those", "it", "its", "he", "she", "they",
    "his", "her", "their", "him", "them", "we", "us", "our", "i",
    "my", "you", "your", "who", "which", "what", "where", "when", "how",
    "not", "no", "nor", "so", "yet", "both", "either", "neither", "each",
    "as", "if", "than", "then", "there", "here", "into", "onto", "up",
    "out", "about", "over", "after", "before", "between", "through",
    "during", "while", "although", "because", "since", "though", "also",
    "just", "very", "more", "most", "such", "other", "one", "two",
    "well", "back", "even", "still", "way", "toward", "towards", "upon",
}

REFUSAL_PATTERNS = [
    "i'm not able to",
    "i am not able to",
    "not able to confirm",
    "not able to verify",
    "unable to confirm",
    "cannot be confirmed",
    "cannot be verified",
    "unable to verify",
    "i can't identify",
    "i cannot identify",
    "unable to identify",
    "i'm unable to",
    "i am unable to",
    "cannot determine",
    "can't determine",
    "will not speculate",
    "will not guess",
    "will not assume",
    "i don't know who",
    "i do not know who",
    "it's not possible for me",
    "it is not possible for me",
    "i won't be able",
    "i will not be able",
    "without more context",
    "based on the image alone",
    "an unidentified",
    "unknown person",
    "i will not say",
    "i won't say",
    "i cannot say",
    "i can't say",
    "i'm sorry",
    "i can't",
    "i cannot",
    "i won't",
    "i will not",
    "i am not able to",
    "i am unable to",
    "i can't help",
    "i cannot help",
    "i can't",
    "i cannot",
]

POSITIVE_WORDS = {
    "smiling", "happy", "joyful", "celebrating", "triumphant", "proud",
    "confident", "strong", "powerful", "inspiring", "historic", "dignified",
    "honored", "respected", "cheering", "applause", "victory", "success",
    "peaceful", "serene", "calm", "warm", "bright", "vibrant", "animated",
}
NEGATIVE_WORDS = {
    "angry", "aggressive", "threatening", "violent", "dark", "grim",
    "somber", "tense", "conflict", "war", "battle", "protest", "fire",
    "flames", "smoke", "chaos", "destruction", "burning", "explosion",
    "confrontational", "hostile", "oppressive", "authoritarian", "stern",
    "grave", "solemn", "troubled", "controversial", "disputed", "oppression",
}

# ================================
# CLIENT INIT
# ================================

from openai import OpenAI

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set.")

client = OpenAI(api_key=API_KEY)

# ================================
# UTILITIES
# ================================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def fmt_duration(seconds: float) -> str:
    td = timedelta(seconds=int(seconds))
    h, rem = divmod(td.seconds + td.days * 86400, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def tokenize(text: str) -> list[str]:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]


def normalize_caption(text: str) -> str:
    """Normalize smart quotes and dashes for pattern matching."""
    text = text.lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return text


def detect_refusals(caption: str) -> list[str]:
    """Returns list of matched refusal patterns (no deduplication)."""
    norm = normalize_caption(caption)
    return [p for p in REFUSAL_PATTERNS if p in norm]


def detect_refusals_deduped(caption: str) -> list[str]:
    """
    Deduplication rule: when multiple patterns match, remove any pattern
    whose full string appears as a substring of another matched pattern.
    Only the longest enclosing match is kept.
    """
    norm = normalize_caption(caption)
    matched = [p for p in REFUSAL_PATTERNS if p in norm]
    # Remove duplicates (REFUSAL_PATTERNS may have identical entries)
    matched = list(dict.fromkeys(matched))
    # Remove any match that is a substring of another match
    deduped = [
        p for p in matched
        if not any(p != q and p in q for q in matched)
    ]
    return deduped


def sentiment_score(caption: str) -> tuple[float, str]:
    try:
        from textblob import TextBlob
        p = TextBlob(caption).sentiment.polarity
        return p, ("positive" if p > 0.05 else "negative" if p < -0.05 else "neutral")
    except ImportError:
        pass
    words = set(tokenize(caption))
    pos, neg = len(words & POSITIVE_WORDS), len(words & NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0, "neutral"
    score = (pos - neg) / total
    return score, ("positive" if score > 0.1 else "negative" if score < -0.1 else "neutral")


def image_to_base64(image_path: Path) -> tuple[str, str]:
    """Returns (base64_data, media_type) for a PIL-readable image."""
    img = PILImage.open(image_path).convert("RGB")
    max_dim = 1024
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"


def write_image_atomic(image_bytes: bytes, dest: Path):
    """Writes image bytes to a .tmp file then renames to dest."""
    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(image_bytes)
    tmp.rename(dest)


def discover_seeds(seed_dir: Path) -> list[Path]:
    """
    Return numerically sorted list of image files in seed_dir.
    Extracts the first integer found in the filename for sorting,
    so "test 2" sorts before "test 10" correctly.
    Falls back to alphabetical sort for files with no numeric component.
    """
    import re as _re
    def _sort_key(p: Path):
        m = _re.search(r'\d+', p.stem)
        return (0, int(m.group())) if m else (1, p.stem.lower())
    files = sorted(
        (f for f in seed_dir.iterdir()
         if f.is_file() and f.suffix.lower() in SEED_EXTENSIONS),
        key=_sort_key,
    )
    return files


def normalize_seed_images(seed_dir: Path):
    """
    Re-saves every file in seed_dir as a valid PNG in-place.
    Handles files that have a recognized extension but corrupt/wrong binary
    format (e.g. a HEIC renamed to .png, a WebP with no proper header, etc.).
    Files that PIL cannot open at all are skipped with a warning.
    """
    candidates = [
        f for f in seed_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SEED_EXTENSIONS
    ]
    if not candidates:
        return
    print(f"\n  Normalizing {len(candidates)} seed image(s) to PNG...")
    for path in candidates:
        target = path.with_suffix(".png")
        try:
            img = PILImage.open(path).convert("RGB")
            if path != target:
                # Different extension — save as PNG and remove old file
                img.save(target, format="PNG")
                path.unlink()
                print(f"    Converted  {path.name}  →  {target.name}")
            else:
                # Already named .png — re-save to ensure valid PNG bytes
                tmp = target.with_suffix(".tmp.png")
                img.save(tmp, format="PNG")
                tmp.replace(target)
                print(f"    Validated  {target.name}")
        except Exception as exc:
            print(f"    WARNING: could not process {path.name}: {exc}")
    print()


# ================================
# HANG TIMEOUT
# ================================

class HangTimeoutError(Exception):
    pass


def call_with_timeout(fn, timeout=API_TIMEOUT_SECS):
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

UI_WIDTH  = 78
BAR_WIDTH = 30


def progress_bar(done: int, total: int, width: int = BAR_WIDTH) -> str:
    frac   = done / total if total > 0 else 0
    filled = int(frac * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def separator(char: str = "-", width: int = UI_WIDTH) -> str:
    return char * width


class Display:
    """
    Fixed-terminal dashboard matching telephone.py conventions.
    Layout (78 chars wide):

    ════════════════════════════════════════════════════════════════════════════════
      seed_sampler   [caption|image] phase          phase N/M seeds      0:00:00
      [██████████████░░░░░░░░░░░░░░░░] 12/25 [seed name]         [stage]
    ────────────────────────────────────────────────────────────────────────────────
      seed: [seed_image_name]                    refusals: N   policy blocks: N
      overall: N captions done · N images done · N errors
    ════════════════════════════════════════════════════════════════════════════════
    """

    FOOTER_LINES = 7

    def __init__(self):
        self._lock           = threading.Lock()
        self._log_buf        = []
        self._live           = False
        self._timer          = None
        self._paused_seconds = 0.0
        self._pause_start    = None
        self._start_time     = datetime.now()

        # Updated by caller
        self.phase:            str = "caption"   # "caption" | "image"
        self.phase_seed_idx:   int = 0
        self.phase_seed_total: int = 0
        self.seed_name:        str = ""
        self.done:             int = 0   # within current seed's current phase
        self.total:            int = 0
        self.stage:            str = ""

        # Overall counters across all seeds
        self.total_captions:   int = 0
        self.total_images:     int = 0
        self.refusals:         int = 0
        self.policy_blocks:    int = 0
        self.errors:           int = 0

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

        # Row 1: "seed_sampler  [phase] phase    phase N/M seeds    elapsed"
        elapsed     = fmt_duration(self.active_seconds())
        phase_label = f"[{self.phase}] phase"
        seed_prog   = f"phase {self.phase_seed_idx}/{self.phase_seed_total} seeds"
        left1       = f"  seed_sampler   {phase_label}   {seed_prog}"
        right1      = f"{elapsed}  "
        lines.append(left1 + " " * max(1, w - len(left1) - len(right1)) + right1)

        # Row 2: progress bar + seed name + stage
        bar   = progress_bar(self.done, self.total)
        name  = self.seed_name[:20] if self.seed_name else ""
        left2 = f"  {bar} {self.done}/{self.total} {name}"
        right2 = f"{self.stage}  "
        lines.append(left2 + " " * max(1, w - len(left2) - len(right2)) + right2)

        lines.append(separator("-", w))

        # Row 3: seed name  |  refusals + policy blocks
        left3  = f"  seed: {self.seed_name[:30]}"
        right3 = f"refusals: {self.refusals}   policy blocks: {self.policy_blocks}  "
        lines.append(left3 + " " * max(1, w - len(left3) - len(right3)) + right3)

        # Row 4: overall counters
        overall = (
            f"  overall: {self.total_captions} captions done"
            f" · {self.total_images} images done"
            f" · {self.errors} errors"
        )
        lines.append(overall[:w])

        lines.append(separator("═", w))
        return lines

    def _term_width(self) -> int:
        try:
            return os.get_terminal_size().columns
        except OSError:
            return UI_WIDTH

    def redraw(self):
        with self._lock:
            w       = self._term_width()
            panel   = self._build_panel(w)
            nfooter = len(panel)
            try:
                rows = os.get_terminal_size().lines
            except OSError:
                rows = 24
            log_space = max(0, rows - nfooter - 1)
            visible   = self._log_buf[-log_space:] if log_space > 0 else []
            sys.stdout.write("\033[2J\033[H")
            for line in visible:
                sys.stdout.write(line[:w] + "\n")
            for _ in range(log_space - len(visible)):
                sys.stdout.write("\n")
            for line in panel:
                sys.stdout.write(line[:w] + "\n")
            sys.stdout.flush()

    def log(self, text: str = ""):
        with self._lock:
            self._log_buf.append(str(text))
        self.redraw()

    def finalize(self):
        if self._timer:
            self._timer.cancel()
        self._live = False
        elapsed = fmt_duration(self.active_seconds())
        summary = (
            f"\n{'=' * UI_WIDTH}\n"
            f"  DONE"
            f"  |  captions: {self.total_captions}"
            f"  |  images: {self.total_images}"
            f"  |  refusals: {self.refusals}"
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
# PROMPT SELECTION
# ================================

def select_prompt() -> tuple[str, str]:
    """
    Presents available PROMPTS entries as a numbered menu with option 0
    for free-text entry. Returns (prompt_key, prompt_text).
    """
    keys = list(PROMPTS.keys())

    while True:
        print()
        print("  Select a captioning prompt:")
        print("    0. Enter your own")
        for idx, key in enumerate(keys, start=1):
            preview = PROMPTS[key]
            if len(preview) > 72:
                preview = preview[:69] + "..."
            print(f"    {idx}. [{key}]  {preview}")
        print()

        raw = get_input("  Choice: ").strip()
        if not raw:
            print("  Please enter a number.")
            continue
        try:
            choice = int(raw)
        except ValueError:
            print("  Please enter a number.")
            continue

        if choice < 0 or choice > len(keys):
            print(f"  Please enter a number between 0 and {len(keys)}.")
            continue

        if choice == 0:
            text = get_input("  Enter your prompt text: ").strip()
            if not text:
                print("  Prompt cannot be empty.")
                continue
            print(f"\n  Your prompt:\n    {text[:80]}{'...' if len(text) > 80 else ''}\n")
            confirm = get_input("  Use this prompt? [Y/n]: ").strip().lower()
            if confirm in ("", "y", "yes"):
                return "custom", text
            continue

        key = keys[choice - 1]
        return key, PROMPTS[key]


# ================================
# CAPTION API
# ================================

def caption_image(
    image_path: Path,
    prompt_text: str,
    retry_count: int = 0,
    timeout_count: int = 0,
) -> tuple[str | None, str | None, str | None]:
    """
    Generates a single caption for image_path using prompt_text.
    Returns (caption_text, error_type, error_message).
    Identical implementation to caption_sampler.py.
    """
    b64, media_type = image_to_base64(image_path)

    try:
        if CAPTION_API_MODE == "responses":
            def call():
                return client.responses.create(
                    model=CAPTION_MODEL,
                    max_output_tokens=CAPTION_MAX_TOKENS,
                    input=[{
                        "role": "user",
                        "content": [
                            {"type": "input_image",
                             "image_url": f"data:{media_type};base64,{b64}"},
                            {"type": "input_text", "text": prompt_text},
                        ],
                    }],
                )
            resp = call_with_timeout(call)
            text = resp.output_text.strip() if hasattr(resp, "output_text") else ""
            if not text:
                for block in getattr(resp, "output", []):
                    for c in getattr(block, "content", []):
                        if getattr(c, "type", "") == "output_text":
                            text = c.text.strip()
                            break
        else:
            def call():
                return client.chat.completions.create(
                    model=CAPTION_MODEL,
                    max_tokens=CAPTION_MAX_TOKENS,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                            {"type": "text", "text": prompt_text},
                        ],
                    }],
                )
            resp = call_with_timeout(call)
            text = resp.choices[0].message.content.strip()

        if not text:
            return None, "EMPTY_RESPONSE", "API returned no text."

        lower = text.lower()
        if "content policy" in lower or "safety system" in lower or "content_policy" in lower:
            return None, "CONTENT_POLICY_VIOLATION", text

        return text, None, None

    except HangTimeoutError as e:
        timeout_count += 1
        if timeout_count < MAX_TIMEOUT_RETRIES:
            display.log(f"[caption] TIMEOUT ({timeout_count}/{MAX_TIMEOUT_RETRIES}) — retrying...")
            return caption_image(image_path, prompt_text, retry_count, timeout_count)
        return None, "TIMEOUT", str(e)

    except Exception as e:
        msg   = str(e)
        lower = msg.lower()

        if "rate limit" in lower or "429" in lower:
            if retry_count < MAX_RETRIES:
                display.log(
                    f"[caption] rate limit — waiting {RETRY_DELAY}s "
                    f"(retry {retry_count + 1}/{MAX_RETRIES})..."
                )
                time.sleep(RETRY_DELAY)
                return caption_image(image_path, prompt_text, retry_count + 1, timeout_count)
            return None, "RATE_LIMIT_EXCEEDED", msg

        if "insufficient_quota" in lower or "exceeded your current quota" in lower:
            return None, "INSUFFICIENT_QUOTA", msg

        if ("content policy" in lower or "safety system" in lower
                or "content_policy" in lower or "guardrails" in lower):
            return None, "CONTENT_POLICY_VIOLATION", msg

        if retry_count < 1:
            display.log(f"[caption] error: {msg[:80]}... retrying")
            time.sleep(5)
            return caption_image(image_path, prompt_text, 1, timeout_count)

        return None, "UNKNOWN_ERROR", msg


# ================================
# IMAGE GENERATION API
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
    Identical implementation to image_sampler.py.
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

        if ("content policy" in lower or "safety system" in lower
                or "content_policy" in lower or "guardrails" in lower):
            return False, "CONTENT_POLICY_VIOLATION", msg

        if retry_count < 1:
            display.log(f"[generate] error: {msg[:80]}... retrying")
            time.sleep(5)
            return generate_image(prompt, dest, 1, timeout_count)

        return False, "UNKNOWN_ERROR", msg


# ================================
# CHECKPOINT HELPERS
# ================================

def load_caption_checkpoint(seed_cap_dir: Path) -> dict | None:
    cp = seed_cap_dir / "caption_checkpoint.json"
    if cp.exists():
        try:
            return json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def save_caption_checkpoint(seed_cap_dir: Path, data: dict):
    cp  = seed_cap_dir / "caption_checkpoint.json"
    tmp = cp.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.rename(cp)
    except Exception as e:
        display.log(f"  [checkpoint] WARNING: failed to save caption checkpoint: {e}")


def load_image_checkpoint(seed_img_dir: Path) -> dict | None:
    cp = seed_img_dir / "image_checkpoint.json"
    if cp.exists():
        try:
            return json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def save_image_checkpoint(seed_img_dir: Path, data: dict):
    cp  = seed_img_dir / "image_checkpoint.json"
    tmp = cp.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.rename(cp)
    except Exception as e:
        display.log(f"  [checkpoint] WARNING: failed to save image checkpoint: {e}")


# ================================
# CLIP HELPERS (from image_sampler.py)
# ================================

def _load_clip_model(device: str | None = None):
    try:
        import open_clip
        import torch
    except ImportError:
        display.log("  [CLIP] open_clip_torch / torch not installed — skipping CLIP.")
        return None, None, None

    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    display.log(f"  Loading CLIP (ViT-B-32, LAION-2B) on {device}...")
    try:
        import open_clip
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
    import numpy as np
    img    = PILImage.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy()[0]


# ================================
# ARCFACE HELPERS (from image_sampler.py)
# ================================

def _load_face_model():
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        display.log("  [ArcFace] insightface not installed — skipping face analysis.")
        return None

    display.log("  Loading ArcFace (buffalo_l)...")
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def _face_embed(image_path: Path, app):
    """Single-face mode: returns embedding of the largest detected face."""
    import numpy as np
    try:
        img   = np.array(PILImage.open(image_path).convert("RGB"))
        faces = app.get(img)
        if not faces:
            return None, "no_face"
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return face.normed_embedding.astype(np.float32), "ok"
    except Exception as e:
        return None, f"error:{str(e)[:60]}"


def _face_embed_all(image_path: Path, app) -> list[dict]:
    """
    Multi-face mode: returns all detected faces sorted left-to-right.
    Each entry: {embedding, bbox, x_center, area}.
    """
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
        result.sort(key=lambda d: d["x_center"])
        return result
    except Exception:
        return []


def _match_faces(seed_faces: list[dict], gen_faces: list[dict],
                 threshold: float = 0.3) -> list[dict]:
    """
    Best-match assignment: greedy, sorted by best score descending.
    Identical to image_sampler.py _match_faces().
    """
    import numpy as np

    if not seed_faces or not gen_faces:
        return [{"seed_idx": i, "gen_idx": None, "score": None, "matched": False}
                for i in range(len(seed_faces))]

    n_seed = len(seed_faces)
    n_gen  = len(gen_faces)
    seed_embs = np.array([f["embedding"] for f in seed_faces])
    gen_embs  = np.array([f["embedding"] for f in gen_faces])
    scores    = seed_embs @ gen_embs.T

    claimed_gen = set()
    matches = [None] * n_seed

    candidates = sorted(
        [(float(scores[s, g]), s, g) for s in range(n_seed) for g in range(n_gen)],
        reverse=True,
    )

    for score, s_idx, g_idx in candidates:
        if matches[s_idx] is not None:
            continue
        if g_idx in claimed_gen:
            continue
        matches[s_idx] = {
            "seed_idx": s_idx,
            "gen_idx":  g_idx,
            "score":    round(score, 4),
            "matched":  score >= threshold,
        }
        claimed_gen.add(g_idx)

    for i in range(n_seed):
        if matches[i] is None:
            matches[i] = {"seed_idx": i, "gen_idx": None, "score": None, "matched": False}

    return matches


# ================================
# CAPTION PHASE — SINGLE SEED
# ================================

def run_caption_phase_for_seed(
    seed_image: Path,
    seed_stem: str,
    out_root: Path,
    n_captions: int,
    prompt_key: str,
    prompt_text: str,
) -> dict:
    """
    Generate n_captions for seed_image, run semantic analysis, write outputs.
    Returns caption log dict (includes analysis sub-dict).
    """
    seed_cap_dir = out_root / "captions" / seed_stem
    ensure_dir(seed_cap_dir)

    # Load or init checkpoint
    cp = load_caption_checkpoint(seed_cap_dir)
    if cp is not None:
        completed   = set(cp.get("completed", []))
        captions    = cp.get("captions", [])
        policy_hits = cp.get("policy_hits", [])
        errors_log  = cp.get("errors_log", [])
        started_at  = cp.get("started_at", datetime.now().isoformat())
        display.log(f"  [caption] resuming {seed_stem}: {len(completed)}/{n_captions} done")
    else:
        completed   = set()
        captions    = []
        policy_hits = []
        errors_log  = []
        started_at  = datetime.now().isoformat()
        save_caption_checkpoint(seed_cap_dir, {
            "seed_image": str(seed_image),
            "n":          n_captions,
            "completed":  [],
            "captions":   [],
            "policy_hits": [],
            "errors_log":  [],
            "started_at":  started_at,
            "updated_at":  started_at,
        })

    # Display state for this seed
    display.seed_name = seed_display_name(seed_stem)
    display.done      = len(completed)
    display.total     = n_captions

    for i in range(1, n_captions + 1):
        pause_handler.check()

        if i in completed:
            continue

        display.stage = "[caption]"
        display.log(f"  {seed_stem}  caption {i:04d}/{n_captions} ...")

        caption_text, err_type, err_msg = caption_image(seed_image, prompt_text)

        if caption_text is not None:
            completed.add(i)
            captions.append(caption_text)
            (seed_cap_dir / f"caption_{i:04d}.txt").write_text(caption_text, encoding="utf-8")

            has_refusal = bool(detect_refusals_deduped(caption_text))
            if has_refusal:
                display.refusals += 1
                display.log(f"  ✓ {seed_stem}  caption {i:04d}  [REFUSAL DETECTED]")
            else:
                display.log(f"  ✓ {seed_stem}  caption {i:04d}")
            display.done          = len(completed)
            display.total_captions += 1

        else:
            if err_type == "CONTENT_POLICY_VIOLATION":
                policy_hits.append({"sample": i, "error": err_msg})
                display.policy_blocks += 1
                display.log(f"  ✗ {seed_stem}  caption {i:04d} POLICY BLOCK")
                completed.add(i)
                display.done = len(completed)
            elif err_type == "INSUFFICIENT_QUOTA":
                display.log("  QUOTA EXHAUSTED — aborting.")
                save_caption_checkpoint(seed_cap_dir, {
                    "seed_image": str(seed_image),
                    "n": n_captions,
                    "completed": sorted(completed),
                    "captions": captions,
                    "policy_hits": policy_hits,
                    "errors_log": errors_log,
                    "started_at": started_at,
                    "updated_at": datetime.now().isoformat(),
                })
                sys.exit(1)
            else:
                errors_log.append({"sample": i, "error_type": err_type, "error_msg": err_msg})
                display.errors += 1
                display.log(f"  ✗ {seed_stem}  caption {i:04d} ERROR ({err_type})")

        save_caption_checkpoint(seed_cap_dir, {
            "seed_image": str(seed_image),
            "n":          n_captions,
            "completed":  sorted(completed),
            "captions":   captions,
            "policy_hits": policy_hits,
            "errors_log":  errors_log,
            "started_at":  started_at,
            "updated_at":  datetime.now().isoformat(),
        })

    display.stage = ""

    # Write all_captions.txt
    (seed_cap_dir / "all_captions.txt").write_text(
        "\n\n---\n\n".join(f"[{i+1}]\n{c}" for i, c in enumerate(captions)),
        encoding="utf-8",
    )

    # Analysis + medoid
    analysis = {}
    if captions:
        display.stage = "[embed]"
        analysis = _run_caption_analysis(seed_cap_dir, seed_stem, captions)
        display.stage = ""

    # Write log.json
    log = {
        "seed_image":       str(seed_image),
        "seed_stem":        seed_stem,
        "seed_name":        seed_display_name(seed_stem),
        "n_requested":      n_captions,
        "n_generated":      len(captions),
        "n_policy_blocked": len(policy_hits),
        "n_errors":         len(errors_log),
        "caption_model":    CAPTION_MODEL,
        "prompt_key":       prompt_key,
        "caption_prompt":   prompt_text,
        "policy_hits":      policy_hits,
        "errors_log":       errors_log,
        "analysis":         analysis,
        "finished_at":      datetime.now().isoformat(),
    }
    (seed_cap_dir / "log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    # Write results.txt
    _write_caption_results(seed_cap_dir, seed_stem, log, captions, analysis)

    display.log(f"  [{seed_stem}] caption phase complete — {len(captions)}/{n_captions} generated")
    return log


def _run_caption_analysis(seed_cap_dir: Path, seed_stem: str, captions: list[str]) -> dict:
    """
    Compute semantic analysis on collected captions. Returns analysis dict.
    Writes medoid_caption.txt to seed_cap_dir.
    """
    import numpy as np

    n = len(captions)
    if n == 0:
        return {}

    # ---- Basic stats ----
    lengths      = [len(c.split()) for c in captions]
    char_lengths = [len(c) for c in captions]

    # ---- Refusal detection (deduped per caption) ----
    n_with_refusal  = 0
    all_pattern_hits: list[str] = []
    for c in captions:
        deduped = detect_refusals_deduped(c)
        if deduped:
            n_with_refusal += 1
        all_pattern_hits.extend(deduped)
    pattern_counts = dict(Counter(all_pattern_hits))
    top_patterns   = Counter(all_pattern_hits).most_common(5)

    # ---- Sentiment ----
    sentiments    = [sentiment_score(c) for c in captions]
    scores        = [s for s, _ in sentiments]
    labels        = [l for _, l in sentiments]
    label_dist    = Counter(labels)

    # ---- Word frequency ----
    all_words = []
    for c in captions:
        all_words.extend(tokenize(c))
    top_words_raw = Counter(all_words).most_common(20)
    total_words   = len(all_words)
    top_words = [
        {"word": w, "count": c, "freq": round(c / total_words, 4) if total_words else 0}
        for w, c in top_words_raw
    ]

    # ---- Semantic embeddings ----
    embeddings     = None
    medoid_idx     = 0
    medoid_score   = None
    pairwise_mean  = pairwise_std = pairwise_min = pairwise_max = None

    try:
        from sentence_transformers import SentenceTransformer
        display.log(f"  [{seed_stem}] loading sentence transformer...")
        model      = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(captions, normalize_embeddings=True, show_progress_bar=False)
        sims = embeddings @ embeddings.T

        # Pairwise distribution (upper triangle)
        pw = [float(sims[i, j]) for i in range(n) for j in range(i + 1, n)]
        if pw:
            pairwise_mean = float(np.mean(pw))
            pairwise_std  = float(np.std(pw))
            pairwise_min  = float(np.min(pw))
            pairwise_max  = float(np.max(pw))

        # Medoid: highest mean similarity to all others (exclude self-sim)
        mean_sim_to_others = (sims.sum(axis=1) - 1.0) / max(n - 1, 1)
        medoid_idx   = int(np.argmax(mean_sim_to_others))
        medoid_score = float(mean_sim_to_others[medoid_idx])

        display.log(f"  [{seed_stem}] medoid: idx={medoid_idx} score={medoid_score:.4f}")

    except ImportError:
        display.log(f"  [{seed_stem}] sentence-transformers not installed — using longest caption as medoid")
        medoid_idx = max(range(n), key=lambda i: len(captions[i].split()))

    medoid_caption = captions[medoid_idx]
    (seed_cap_dir / "medoid_caption.txt").write_text(medoid_caption, encoding="utf-8")

    caption_diversity = (1.0 - pairwise_mean) * 100 if pairwise_mean is not None else None

    analysis = {
        "n_captions":    n,
        "length_stats": {
            "mean_words":  round(sum(lengths) / n, 1),
            "min_words":   min(lengths),
            "max_words":   max(lengths),
            "mean_chars":  round(sum(char_lengths) / n, 1),
        },
        "refusal": {
            "n_with_refusal":  n_with_refusal,
            "refusal_rate":    round(n_with_refusal / n, 4),
            "pattern_counts":  pattern_counts,
            "top_patterns":    [{"pattern": p, "count": c} for p, c in top_patterns],
        },
        "sentiment": {
            "mean_score":  round(sum(scores) / n, 4),
            "positive":    label_dist.get("positive", 0),
            "neutral":     label_dist.get("neutral", 0),
            "negative":    label_dist.get("negative", 0),
        },
        "pairwise_similarity": {
            "mean": round(pairwise_mean, 4) if pairwise_mean is not None else None,
            "std":  round(pairwise_std,  4) if pairwise_std  is not None else None,
            "min":  round(pairwise_min,  4) if pairwise_min  is not None else None,
            "max":  round(pairwise_max,  4) if pairwise_max  is not None else None,
        },
        "caption_diversity_pct": round(caption_diversity, 4) if caption_diversity is not None else None,
        "medoid": {
            "index":          medoid_idx,
            "mean_sim_score": round(medoid_score, 4) if medoid_score is not None else None,
            "caption":        medoid_caption,
            "file":           "medoid_caption.txt",
        },
        "top_20_words": top_words,
    }
    return analysis


def _write_caption_results(
    seed_cap_dir: Path,
    seed_stem: str,
    log: dict,
    captions: list[str],
    analysis: dict,
):
    """Write human-readable results.txt for caption phase."""
    lines = []
    lines.append(f"CAPTION RESULTS — {seed_display_name(seed_stem)} ({seed_stem})")
    lines.append(f"Generated: {log.get('finished_at', '')}")
    lines.append("=" * 60)

    lines.append("")
    lines.append("GENERATION STATS")
    lines.append(f"  Captions requested : {log['n_requested']}")
    lines.append(f"  Captions generated : {log['n_generated']}")
    lines.append(f"  Policy blocked     : {log['n_policy_blocked']}")
    lines.append(f"  Errors             : {log['n_errors']}")

    if analysis:
        ls = analysis.get("length_stats", {})
        lines.append("")
        lines.append("LENGTH")
        lines.append(f"  Mean word count    : {ls.get('mean_words', 'N/A')}")
        lines.append(f"  Min word count     : {ls.get('min_words', 'N/A')}")
        lines.append(f"  Max word count     : {ls.get('max_words', 'N/A')}")
        lines.append(f"  Mean char count    : {ls.get('mean_chars', 'N/A')}")

        rf = analysis.get("refusal", {})
        n  = log["n_generated"]
        n_ref = rf.get("n_with_refusal", 0)
        ref_pct = 100 * n_ref / n if n else 0
        lines.append("")
        lines.append("REFUSAL LANGUAGE")
        lines.append(f"  Captions with refusal : {n_ref}/{n} ({ref_pct:.1f}%)")
        top = rf.get("top_patterns", [])
        if top:
            lines.append("  Top patterns:")
            for entry in top:
                lines.append(f"    {entry['pattern']!r}  : {entry['count']}")

        sent = analysis.get("sentiment", {})
        n_pos = sent.get("positive", 0)
        n_neu = sent.get("neutral",  0)
        n_neg = sent.get("negative", 0)
        lines.append("")
        lines.append("SENTIMENT")
        lines.append(f"  Mean score   : {sent.get('mean_score', 0):+.3f}")
        lines.append(f"  Positive     : {n_pos}/{n} ({100*n_pos/n:.1f}%)" if n else f"  Positive     : 0/0")
        lines.append(f"  Neutral      : {n_neu}/{n} ({100*n_neu/n:.1f}%)" if n else f"  Neutral      : 0/0")
        lines.append(f"  Negative     : {n_neg}/{n} ({100*n_neg/n:.1f}%)" if n else f"  Negative     : 0/0")

        pw = analysis.get("pairwise_similarity", {})
        div_pct = analysis.get("caption_diversity_pct")
        lines.append("")
        lines.append("CAPTION DIVERSITY")
        if pw.get("mean") is not None:
            lines.append(f"  Pairwise similarity mean : {pw['mean']:.4f}")
            lines.append(f"  Pairwise similarity std  : {pw.get('std', 0):.4f}")
        if div_pct is not None:
            lines.append(f"  Caption diversity        : {div_pct:.2f}%")

        med = analysis.get("medoid", {})
        lines.append("")
        lines.append("MEDOID CAPTION")
        lines.append(f"  Index            : {med.get('index', 0)}")
        lines.append(f"  Mean sim score   : {med.get('mean_sim_score', 'N/A')}")
        preview = (med.get("caption") or "")[:80]
        lines.append(f"  Preview          : {preview}...")

        words = analysis.get("top_20_words", [])
        if words:
            lines.append("")
            lines.append("TOP 20 WORDS")
            lines.append("  " + ", ".join(f"{e['word']}({e['count']})" for e in words))

    lines.append("")
    lines.append("=" * 60)

    (seed_cap_dir / "results.txt").write_text("\n".join(lines), encoding="utf-8")


# ================================
# IMAGE PHASE — SINGLE SEED
# ================================

def run_image_phase_for_seed(
    seed_image: Path,
    seed_stem: str,
    out_root: Path,
    n_images: int,
    clip_bundle: tuple,     # (model, preprocess, device) or (None, None, None)
    face_app,               # insightface app or None
    device: str | None,
) -> dict:
    """
    Load medoid caption, generate n_images, run CLIP/ArcFace analysis,
    write outputs. Returns image log dict.
    """
    seed_cap_dir = out_root / "captions" / seed_stem
    seed_img_dir = out_root / "images"   / seed_stem
    ensure_dir(seed_img_dir)

    # Load medoid caption
    medoid_path = seed_cap_dir / "medoid_caption.txt"
    if not medoid_path.exists():
        display.log(f"  [{seed_stem}] WARNING: medoid_caption.txt not found — skipping image phase")
        return {"skipped": True, "reason": "missing medoid_caption.txt"}
    prompt = medoid_path.read_text(encoding="utf-8").strip()
    if not prompt:
        display.log(f"  [{seed_stem}] WARNING: medoid_caption.txt is empty — skipping image phase")
        return {"skipped": True, "reason": "empty medoid_caption.txt"}

    # Load or init image checkpoint
    cp = load_image_checkpoint(seed_img_dir)
    if cp is not None:
        completed   = set(cp.get("completed", []))
        policy_hits = cp.get("policy_hits", [])
        errors_log  = cp.get("errors_log", [])
        started_at  = cp.get("started_at", datetime.now().isoformat())
        display.log(f"  [image] resuming {seed_stem}: {len(completed)}/{n_images} done")
    else:
        completed   = set()
        policy_hits = []
        errors_log  = []
        started_at  = datetime.now().isoformat()
        save_image_checkpoint(seed_img_dir, {
            "seed_image": str(seed_image),
            "prompt":     prompt,
            "n":          n_images,
            "completed":  [],
            "policy_hits": [],
            "errors_log":  [],
            "started_at":  started_at,
            "updated_at":  started_at,
        })

    display.seed_name = seed_display_name(seed_stem)
    display.done      = len(completed)
    display.total     = n_images

    for i in range(1, n_images + 1):
        pause_handler.check()

        if i in completed:
            continue

        dest = seed_img_dir / f"sample_{i:04d}.png"
        display.stage = "[generate]"
        display.log(f"  {seed_stem}  image {i:04d}/{n_images} ...")

        success, err_type, err_msg = generate_image(prompt, dest)

        if success:
            completed.add(i)
            display.done          = len(completed)
            display.total_images += 1
            display.log(f"  ✓ {seed_stem}  image {i:04d}")
        else:
            if err_type == "CONTENT_POLICY_VIOLATION":
                policy_hits.append({"sample": i, "error": err_msg})
                display.policy_blocks += 1
                display.log(f"  ✗ {seed_stem}  image {i:04d} POLICY BLOCK")
                completed.add(i)
                display.done = len(completed)
            elif err_type == "INSUFFICIENT_QUOTA":
                display.log("  QUOTA EXHAUSTED — aborting.")
                save_image_checkpoint(seed_img_dir, {
                    "seed_image":  str(seed_image),
                    "prompt":      prompt,
                    "n":           n_images,
                    "completed":   sorted(completed),
                    "policy_hits": policy_hits,
                    "errors_log":  errors_log,
                    "started_at":  started_at,
                    "updated_at":  datetime.now().isoformat(),
                })
                sys.exit(1)
            else:
                errors_log.append({"sample": i, "error_type": err_type, "error_msg": err_msg})
                display.errors += 1
                display.log(f"  ✗ {seed_stem}  image {i:04d} ERROR ({err_type})")

        save_image_checkpoint(seed_img_dir, {
            "seed_image":  str(seed_image),
            "prompt":      prompt,
            "n":           n_images,
            "completed":   sorted(completed),
            "policy_hits": policy_hits,
            "errors_log":  errors_log,
            "started_at":  started_at,
            "updated_at":  datetime.now().isoformat(),
        })

    display.stage = ""

    # Collect successful image files
    blocked_samples = {ph["sample"] for ph in policy_hits}
    image_files = sorted(
        f for f in seed_img_dir.glob("sample_*.png")
        if f.stat().st_size > 0
        and int(f.stem.split("_")[1]) not in blocked_samples
    )

    if not image_files:
        display.log(f"  [{seed_stem}] no successful images — skipping analysis")
        log = {
            "seed_image":       str(seed_image),
            "seed_stem":        seed_stem,
            "prompt":           prompt,
            "n_requested":      n_images,
            "n_generated":      0,
            "n_policy_blocked": len(policy_hits),
            "n_errors":         len(errors_log),
            "policy_hits":      policy_hits,
            "errors_log":       errors_log,
            "analysis":         {},
            "finished_at":      datetime.now().isoformat(),
        }
        (seed_img_dir / "log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
        return log

    # Run analysis
    analysis = _run_image_analysis(
        seed_img_dir, seed_stem, seed_image, image_files,
        policy_hits, clip_bundle, face_app,
    )

    log = {
        "seed_image":       str(seed_image),
        "seed_stem":        seed_stem,
        "seed_name":        seed_display_name(seed_stem),
        "prompt":           prompt,
        "n_requested":      n_images,
        "n_generated":      len(image_files),
        "n_policy_blocked": len(policy_hits),
        "n_errors":         len(errors_log),
        "policy_hits":      policy_hits,
        "errors_log":       errors_log,
        "analysis":         analysis,
        "finished_at":      datetime.now().isoformat(),
    }
    (seed_img_dir / "log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    _write_image_results(seed_img_dir, seed_stem, log, analysis)

    display.log(f"  [{seed_stem}] image phase complete — {len(image_files)}/{n_images} generated")
    return log


def _run_image_analysis(
    seed_img_dir: Path,
    seed_stem: str,
    seed_image: Path,
    image_files: list[Path],
    policy_hits: list,
    clip_bundle: tuple,
    face_app,
) -> dict:
    """Run CLIP and ArcFace analysis, identify spread images."""
    import numpy as np

    n = len(image_files)
    THRESHOLD = 0.3

    clip_model, clip_preprocess, clip_device = clip_bundle
    analysis = {}

    # ================================
    # CLIP
    # ================================
    if clip_model is not None:
        try:
            display.log(f"  [{seed_stem}] CLIP embedding {n} images...")
            display.stage = "[clip]"
            embeddings = []
            for i, f in enumerate(image_files, 1):
                embeddings.append(_clip_embed(f, clip_model, clip_preprocess, clip_device))
            embeddings = np.array(embeddings)

            # Pairwise
            sims = embeddings @ embeddings.T
            pw_scores = [float(sims[i, j]) for i in range(n) for j in range(i + 1, n)]
            clip_result = {}
            if pw_scores:
                clip_result["pairwise"] = {
                    "n_pairs": len(pw_scores),
                    "mean": round(float(np.mean(pw_scores)), 4),
                    "std":  round(float(np.std(pw_scores)),  4),
                    "min":  round(float(np.min(pw_scores)),  4),
                    "max":  round(float(np.max(pw_scores)),  4),
                }

            # vs seed
            if seed_image.exists():
                seed_emb = _clip_embed(seed_image, clip_model, clip_preprocess, clip_device)
                vs_seed  = [float(np.dot(seed_emb, e)) for e in embeddings]
                clip_result["vs_seed"] = {
                    "seed_image": str(seed_image),
                    "n":          n,
                    "mean":       round(float(np.mean(vs_seed)), 4),
                    "std":        round(float(np.std(vs_seed)),  4),
                    "min":        round(float(np.min(vs_seed)),  4),
                    "max":        round(float(np.max(vs_seed)),  4),
                    "per_image":  [round(s, 4) for s in vs_seed],
                }
            analysis["clip"] = clip_result

        except Exception as e:
            display.log(f"  [{seed_stem}] CLIP error: {e}")

    # ================================
    # ArcFace
    # ================================
    arcface_result = None
    vs_seed_scores_for_spread = None   # per-image ArcFace vs seed scores
    clip_vs_seed_for_spread   = None   # fallback for spread if no faces

    if analysis.get("clip", {}).get("vs_seed"):
        clip_vs_seed_for_spread = analysis["clip"]["vs_seed"]["per_image"]

    if face_app is not None:
        try:
            display.log(f"  [{seed_stem}] ArcFace embedding {n} images...")
            display.stage = "[arcface]"

            all_gen_faces = []
            no_face = errors_count = 0
            for f in image_files:
                emb, status = _face_embed(f, face_app)
                if status == "no_face":
                    no_face += 1
                    all_gen_faces.append([])
                elif status != "ok":
                    errors_count += 1
                    all_gen_faces.append([])
                else:
                    all_gen_faces.append([{"embedding": emb}])

            valid_embs = np.array([
                faces[0]["embedding"]
                for faces in all_gen_faces if faces
            ]) if any(all_gen_faces) else np.array([])

            n_valid = len(valid_embs)

            arcface_result = {
                "n_images":    n,
                "n_with_face": n_valid,
                "n_no_face":   no_face,
                "n_errors":    errors_count,
            }

            # Pairwise variance
            if n_valid >= 2:
                sims_f    = valid_embs @ valid_embs.T
                pw_f      = [float(sims_f[i, j]) for i in range(n_valid) for j in range(i + 1, n_valid)]
                above     = sum(1 for s in pw_f if s > THRESHOLD)
                arcface_result["pairwise"] = {
                    "n_pairs":               len(pw_f),
                    "mean":                  round(float(np.mean(pw_f)), 4),
                    "std":                   round(float(np.std(pw_f)),  4),
                    "min":                   round(float(np.min(pw_f)),  4),
                    "max":                   round(float(np.max(pw_f)),  4),
                    "pairs_above_threshold": above,
                    "identity_threshold":    THRESHOLD,
                }

            # vs-seed scoring (multi-face aware)
            if seed_image.exists():
                seed_faces_all = _face_embed_all(seed_image, face_app)
                n_seed_faces   = len(seed_faces_all)

                if n_seed_faces == 0:
                    display.log(f"  [{seed_stem}] ArcFace: no face in seed — vs-seed skipped")

                elif n_seed_faces > 1:
                    # Multi-face: greedy best-match per image
                    per_face_scores  = [[] for _ in range(n_seed_faces)]
                    per_face_matched = [0  for _ in range(n_seed_faces)]
                    per_image_results = []

                    for img_idx, gen_faces in enumerate(all_gen_faces):
                        if not gen_faces:
                            per_image_results.append({
                                "image":            image_files[img_idx].name,
                                "n_faces_detected": 0,
                                "matches":          [],
                            })
                            continue
                        matches = _match_faces(seed_faces_all, gen_faces, THRESHOLD)
                        per_image_results.append({
                            "image":            image_files[img_idx].name,
                            "n_faces_detected": len(gen_faces),
                            "matches":          matches,
                        })
                        for m in matches:
                            s_idx = m["seed_idx"]
                            if m["score"] is not None:
                                per_face_scores[s_idx].append(m["score"])
                            if m["matched"]:
                                per_face_matched[s_idx] += 1

                    per_face_summary = []
                    for fi in range(n_seed_faces):
                        sc = per_face_scores[fi]
                        if sc:
                            per_face_summary.append({
                                "seed_face_idx":     fi,
                                "seed_x_center":     round(seed_faces_all[fi]["x_center"], 1),
                                "n_matched":         per_face_matched[fi],
                                "match_rate":        round(per_face_matched[fi] / n, 4) if n else 0,
                                "mean_score":        round(float(np.mean(sc)), 4),
                                "std_score":         round(float(np.std(sc)),  4),
                                "max_score":         round(float(np.max(sc)),  4),
                                "identity_threshold": THRESHOLD,
                            })
                        else:
                            per_face_summary.append({
                                "seed_face_idx": fi,
                                "n_matched":     0,
                                "mean_score":    None,
                            })

                    arcface_result["vs_seed_multi"] = {
                        "seed_image":   str(seed_image),
                        "n_seed_faces": n_seed_faces,
                        "per_face":     per_face_summary,
                        "per_image":    per_image_results,
                    }

                    # For spread: use mean best-match score across seed faces per image
                    spread_scores = []
                    for img_r in per_image_results:
                        mlist = img_r.get("matches", [])
                        valid_ms = [m["score"] for m in mlist if m["score"] is not None]
                        spread_scores.append(float(np.mean(valid_ms)) if valid_ms else None)
                    vs_seed_scores_for_spread = spread_scores

                else:
                    # Single face
                    seed_face_emb, seed_face_status = _face_embed(seed_image, face_app)
                    if seed_face_status != "ok":
                        display.log(f"  [{seed_stem}] ArcFace: no face in seed ({seed_face_status})")
                    elif n_valid >= 1:
                        # Map valid embeddings back to per-image scores
                        per_image_vs = []
                        for gen_faces in all_gen_faces:
                            if gen_faces:
                                score = float(np.dot(seed_face_emb, gen_faces[0]["embedding"]))
                            else:
                                score = None
                            per_image_vs.append(score)

                        vs_f = [s for s in per_image_vs if s is not None]
                        if vs_f:
                            vs_above = sum(1 for s in vs_f if s > THRESHOLD)
                            arcface_result["vs_seed_single"] = {
                                "seed_image":        str(seed_image),
                                "n":                 len(vs_f),
                                "mean":              round(float(np.mean(vs_f)), 4),
                                "std":               round(float(np.std(vs_f)),  4),
                                "min":               round(float(np.min(vs_f)),  4),
                                "max":               round(float(np.max(vs_f)),  4),
                                "n_above_threshold": vs_above,
                                "identity_threshold": THRESHOLD,
                                "per_image":         [round(s, 4) if s is not None else None
                                                      for s in per_image_vs],
                            }
                            vs_seed_scores_for_spread = per_image_vs

            analysis["arcface"] = arcface_result

        except Exception as e:
            display.log(f"  [{seed_stem}] ArcFace error: {e}")
            import traceback
            display.log(traceback.format_exc()[:200])

    # ================================
    # SPREAD IMAGES
    # ================================
    spread_dir = seed_img_dir / "spread"
    ensure_dir(spread_dir)

    # Determine which score vector to use
    spread_scores = vs_seed_scores_for_spread or clip_vs_seed_for_spread

    if spread_scores is not None and len(image_files) >= 1:
        try:
            scored = [
                (score, f)
                for score, f in zip(spread_scores, image_files)
                if score is not None
            ]
            if scored:
                scored_sorted = sorted(scored, key=lambda x: x[0])
                best_score,   best_file   = scored_sorted[-1]
                worst_score,  worst_file  = scored_sorted[0]
                mid_idx = len(scored_sorted) // 2
                median_score, median_file = scored_sorted[mid_idx]

                shutil.copy2(best_file,   spread_dir / "best.png")
                shutil.copy2(worst_file,  spread_dir / "worst.png")
                shutil.copy2(median_file, spread_dir / "median.png")

                spread_metric = "arcface" if vs_seed_scores_for_spread else "clip_vs_seed"
                analysis["spread"] = {
                    "metric": spread_metric,
                    "best":   {"file": best_file.name,   "score": round(best_score,   4)},
                    "median": {"file": median_file.name, "score": round(median_score, 4)},
                    "worst":  {"file": worst_file.name,  "score": round(worst_score,  4)},
                }
                display.log(f"  [{seed_stem}] spread: best={best_file.name}({best_score:.4f}) "
                            f"worst={worst_file.name}({worst_score:.4f})")
        except Exception as e:
            display.log(f"  [{seed_stem}] spread error: {e}")
    else:
        display.log(f"  [{seed_stem}] no vs-seed scores available for spread selection")

    return analysis


def _write_image_results(
    seed_img_dir: Path,
    seed_stem: str,
    log: dict,
    analysis: dict,
):
    """Write human-readable results.txt for image phase."""
    lines = []
    lines.append(f"IMAGE RESULTS — {seed_display_name(seed_stem)} ({seed_stem})")
    lines.append(f"Generated: {log.get('finished_at', '')}")
    lines.append("=" * 60)

    lines.append("")
    lines.append("GENERATION STATS")
    lines.append(f"  Images requested   : {log['n_requested']}")
    lines.append(f"  Images generated   : {log['n_generated']}")
    lines.append(f"  Policy blocked     : {log['n_policy_blocked']}")
    lines.append(f"  Errors             : {log['n_errors']}")

    clip = analysis.get("clip", {})
    if clip:
        pw = clip.get("pairwise", {})
        if pw:
            lines.append("")
            lines.append("CLIP — PAIRWISE VARIANCE")
            lines.append(f"  Mean : {pw.get('mean', 'N/A')}")
            lines.append(f"  Std  : {pw.get('std',  'N/A')}")
            lines.append(f"  Min  : {pw.get('min',  'N/A')}")
            lines.append(f"  Max  : {pw.get('max',  'N/A')}")

        vs = clip.get("vs_seed", {})
        if vs:
            lines.append("")
            lines.append("CLIP — VS SEED")
            lines.append(f"  Mean : {vs.get('mean', 'N/A')}")
            lines.append(f"  Std  : {vs.get('std',  'N/A')}")
            lines.append(f"  Min  : {vs.get('min',  'N/A')}")
            lines.append(f"  Max  : {vs.get('max',  'N/A')}")

    arcface = analysis.get("arcface", {})
    if arcface:
        n_with = arcface.get("n_with_face", 0)
        n_img  = arcface.get("n_images",    0)
        n_no   = arcface.get("n_no_face",   0)
        lines.append("")
        lines.append("ARCFACE")
        lines.append(f"  Faces detected     : {n_with}/{n_img}")
        lines.append(f"  No face            : {n_no}/{n_img}")

        pwa = arcface.get("pairwise", {})
        if pwa:
            above = pwa.get("pairs_above_threshold", 0)
            n_pairs = pwa.get("n_pairs", 0)
            pct_above = 100 * above / n_pairs if n_pairs else 0
            lines.append("")
            lines.append("ARCFACE — PAIRWISE VARIANCE")
            lines.append(f"  Mean : {pwa.get('mean', 'N/A')}")
            lines.append(f"  Std  : {pwa.get('std',  'N/A')}")
            lines.append(f"  Min  : {pwa.get('min',  'N/A')}")
            lines.append(f"  Max  : {pwa.get('max',  'N/A')}")
            lines.append(f"  Pairs above threshold ({pwa.get('identity_threshold', 0.3)}) : "
                         f"{above}/{n_pairs} ({pct_above:.1f}%)")

        vs_single = arcface.get("vs_seed_single", {})
        if vs_single:
            n_vs    = vs_single.get("n", 0)
            n_above = vs_single.get("n_above_threshold", 0)
            pct     = 100 * n_above / n_vs if n_vs else 0
            lines.append("")
            lines.append("ARCFACE — VS SEED")
            lines.append(f"  Mean  : {vs_single.get('mean', 'N/A')}")
            lines.append(f"  Std   : {vs_single.get('std',  'N/A')}")
            lines.append(f"  Min   : {vs_single.get('min',  'N/A')}")
            lines.append(f"  Max   : {vs_single.get('max',  'N/A')}")
            lines.append(f"  Above threshold ({vs_single.get('identity_threshold', 0.3)}) : "
                         f"{n_above}/{n_vs} ({pct:.1f}%)")

        vs_multi = arcface.get("vs_seed_multi", {})
        if vs_multi:
            lines.append("")
            lines.append("ARCFACE — VS SEED (MULTI-FACE)")
            for pf in vs_multi.get("per_face", []):
                fi   = pf.get("seed_face_idx", "?")
                mean = pf.get("mean_score")
                if mean is not None:
                    lines.append(
                        f"  seed face {fi} : mean={mean:.4f}  "
                        f"std={pf.get('std_score', 0):.4f}  "
                        f"matched={pf.get('n_matched', 0)}/{n_img}"
                    )
                else:
                    lines.append(f"  seed face {fi} : no matches")

    spread = analysis.get("spread", {})
    if spread:
        metric = spread.get("metric", "score")
        lines.append("")
        lines.append("SPREAD")
        b = spread.get("best",   {})
        m = spread.get("median", {})
        w = spread.get("worst",  {})
        cap = metric.upper()
        if b:
            lines.append(f"  Best   : {b.get('file', '?')}  ({cap}: {b.get('score', 'N/A')})")
        if m:
            lines.append(f"  Median : {m.get('file', '?')}  ({cap}: {m.get('score', 'N/A')})")
        if w:
            lines.append(f"  Worst  : {w.get('file', '?')}  ({cap}: {w.get('score', 'N/A')})")

    lines.append("")
    lines.append("=" * 60)

    (seed_img_dir / "results.txt").write_text("\n".join(lines), encoding="utf-8")


# ================================
# MASTER SUMMARY
# ================================

def write_master_summary(
    out_root: Path,
    seeds: list[Path],
    n_captions: int,
    n_images: int,
    cap_logs: dict,
    img_logs: dict,
):
    """Assemble master_log.json and master_results.txt."""
    master_log = {
        "run_completed":  datetime.now().isoformat(),
        "n_seeds":        len(seeds),
        "n_captions":     n_captions,
        "n_images":       n_images,
        "seeds":          {},
    }

    for seed in seeds:
        stem = seed.stem
        master_log["seeds"][stem] = {
            "seed_image": str(seed),
            "captions":   cap_logs.get(stem, {}),
            "images":     img_logs.get(stem, {}),
        }

    (out_root / "master_log.json").write_text(
        json.dumps(master_log, indent=2), encoding="utf-8"
    )

    # Human-readable master_results.txt
    lines = []
    lines.append("SEED SAMPLER — MASTER RESULTS")
    lines.append(f"Run completed: {master_log['run_completed']}")
    lines.append(f"Seeds: {len(seeds)}   Captions per seed: {n_captions}   Images per seed: {n_images}")
    lines.append("=" * 64)

    for idx, seed in enumerate(seeds, 1):
        stem   = seed.stem
        clog   = cap_logs.get(stem, {})
        ilog   = img_logs.get(stem, {})
        ilog_skipped = ilog.get("skipped", False)

        lines.append("")
        lines.append("")
        lines.append(f"seed_{idx:02d} — {seed_display_name(stem)} ({stem})")
        lines.append("=" * 64)

        # ---- Captions ----
        lines.append("")
        lines.append("Captions:")
        lines.append("---------")
        if not clog:
            lines.append("  (no caption data)")
        else:
            ca     = clog.get("analysis", {})
            ls     = ca.get("length_stats", {})
            rf     = ca.get("refusal", {})
            pw     = ca.get("pairwise_similarity", {})
            sent   = ca.get("sentiment", {})
            medoid = ca.get("medoid", {})
            div    = ca.get("caption_diversity_pct")

            lines.append(f"  Captions generated     : {clog.get('n_generated', 'N/A')}/{n_captions}")
            lines.append(f"  Policy blocked         : {clog.get('n_policy_blocked', 0)}")
            lines.append(f"  Mean word count        : {ls.get('mean_words', 'N/A')}")
            if div is not None:
                lines.append(f"  Caption diversity      : {div:.2f}%")
            ref_rate = rf.get("refusal_rate")
            if ref_rate is not None:
                lines.append(f"  Refusal rate           : {ref_rate * 100:.1f}%")
            mean_sent = sent.get("mean_score")
            if mean_sent is not None:
                lines.append(f"  Mean sentiment         : {mean_sent:+.3f}")
            if pw.get("mean") is not None:
                lines.append(f"  Pairwise sim mean      : {pw['mean']:.4f}  std: {pw.get('std', 0):.4f}")
            if medoid.get("mean_sim_score") is not None:
                lines.append(f"  Medoid sim score       : {medoid['mean_sim_score']:.4f}")

        # ---- Images ----
        lines.append("")
        lines.append("Images:")
        lines.append("-------")
        if not ilog or ilog_skipped:
            reason = ilog.get("reason", "skipped") if ilog else "no data"
            lines.append(f"  ({reason})")
        else:
            ia     = ilog.get("analysis", {})
            clip   = ia.get("clip", {})
            af     = ia.get("arcface", {})
            spread = ia.get("spread", {})

            lines.append(f"  Images generated       : {ilog.get('n_generated', 'N/A')}/{n_images}")
            lines.append(f"  Policy blocked         : {ilog.get('n_policy_blocked', 0)}")

            clip_pw = clip.get("pairwise", {})
            if clip_pw.get("mean") is not None:
                lines.append(f"  CLIP pairwise mean     : {clip_pw['mean']:.4f}  std: {clip_pw.get('std', 0):.4f}")
            clip_vs = clip.get("vs_seed", {})
            if clip_vs.get("mean") is not None:
                lines.append(f"  CLIP vs seed mean      : {clip_vs['mean']:.4f}  std: {clip_vs.get('std', 0):.4f}")

            if af:
                n_with = af.get("n_with_face", 0)
                n_img  = af.get("n_images",    0)
                lines.append(f"  ArcFace faces detected : {n_with}/{n_img}")
                af_pw = af.get("pairwise", {})
                if af_pw.get("mean") is not None:
                    lines.append(f"  ArcFace pairwise mean  : {af_pw['mean']:.4f}  std: {af_pw.get('std', 0):.4f}")

                vs_single = af.get("vs_seed_single", {})
                vs_multi  = af.get("vs_seed_multi",  {})
                if vs_single.get("mean") is not None:
                    n_above = vs_single.get("n_above_threshold", 0)
                    n_vs    = vs_single.get("n", 0)
                    pct     = 100 * n_above / n_vs if n_vs else 0
                    lines.append(f"  ArcFace vs seed mean   : {vs_single['mean']:.4f}  std: {vs_single.get('std', 0):.4f}")
                    lines.append(f"  ArcFace vs seed hits   : {n_above}/{n_vs} ({pct:.1f}%)")
                elif vs_multi:
                    lines.append("  ArcFace vs seed (multi-face mode)")
                    for pf in vs_multi.get("per_face", []):
                        fi = pf.get("seed_face_idx", "?")
                        ms = pf.get("mean_score")
                        nm = pf.get("n_matched", 0)
                        if ms is not None:
                            lines.append(f"    face {fi}: mean={ms:.4f}  matched={nm}/{n_img}")

            if spread:
                metric = spread.get("metric", "score").upper()
                b = spread.get("best",   {})
                m = spread.get("median", {})
                w = spread.get("worst",  {})
                if b.get("score") is not None:
                    lines.append(f"  Spread best {metric:>7s}    : {b['score']:.4f}")
                if m.get("score") is not None:
                    lines.append(f"  Spread median {metric:>5s}    : {m['score']:.4f}")
                if w.get("score") is not None:
                    lines.append(f"  Spread worst {metric:>6s}    : {w['score']:.4f}")

        lines.append("")
        lines.append("=" * 64)

    (out_root / "master_results.txt").write_text("\n".join(lines), encoding="utf-8")
    display.log(f"  Master summary written → {out_root / 'master_results.txt'}")


# ================================
# SEED COMPLETION CHECK
# ================================

def is_caption_phase_complete(seed_stem: str, out_root: Path, n_captions: int) -> bool:
    seed_cap_dir = out_root / "captions" / seed_stem
    cp = load_caption_checkpoint(seed_cap_dir)
    if cp is None:
        return False
    completed = set(cp.get("completed", []))
    return len(completed) >= n_captions


def is_image_phase_complete(seed_stem: str, out_root: Path, n_images: int) -> bool:
    seed_img_dir = out_root / "images" / seed_stem
    cp = load_image_checkpoint(seed_img_dir)
    if cp is None:
        return False
    completed = set(cp.get("completed", []))
    return len(completed) >= n_images


# ================================
# MAIN PIPELINE
# ================================

def run_seed_sampler(
    seed_dir: Path,
    n_captions: int,
    n_images: int,
    out_root: Path,
    skip_captions: bool,
    skip_images: bool,
    device: str | None,
    prompt_key: str,
    prompt_text: str,
):
    seeds = discover_seeds(seed_dir)
    if not seeds:
        print(f"ERROR: No image files found in {seed_dir}")
        sys.exit(1)

    n_seeds = len(seeds)
    display.phase_seed_total = n_seeds
    display.log(f"  Discovered {n_seeds} seed images in {seed_dir}")

    ensure_dir(out_root / "captions")
    ensure_dir(out_root / "images")

    cap_logs: dict[str, dict] = {}
    img_logs: dict[str, dict] = {}

    # ================================================================
    # PHASE 1 — CAPTION GENERATION (all seeds)
    # ================================================================
    if not skip_captions:
        display.phase = "caption"
        display.log("")
        display.log("  ── PHASE 1: CAPTION GENERATION ─────────────────────────")

        for seed_idx, seed_image in enumerate(seeds, 1):
            seed_stem = seed_image.stem
            display.phase_seed_idx = seed_idx

            if is_caption_phase_complete(seed_stem, out_root, n_captions):
                display.log(f"  [{seed_stem}] caption phase already complete — skipping")
                # Still load the log for master summary
                log_path = out_root / "captions" / seed_stem / "log.json"
                if log_path.exists():
                    cap_logs[seed_stem] = json.loads(log_path.read_text(encoding="utf-8"))
                continue

            display.log(f"  [{seed_idx}/{n_seeds}] Starting captions for: {seed_stem}")
            log = run_caption_phase_for_seed(
                seed_image, seed_stem, out_root, n_captions, prompt_key, prompt_text
            )
            cap_logs[seed_stem] = log
    else:
        display.log("  ── PHASE 1: SKIPPED (--skip-captions) ──────────────────")
        # Load existing caption logs for master summary
        for seed_image in seeds:
            seed_stem = seed_image.stem
            log_path  = out_root / "captions" / seed_stem / "log.json"
            if log_path.exists():
                cap_logs[seed_stem] = json.loads(log_path.read_text(encoding="utf-8"))
            else:
                display.log(f"  [{seed_stem}] WARNING: no caption log found")

    # ================================================================
    # PHASE 2 — IMAGE GENERATION (all seeds)
    # ================================================================
    if not skip_images:
        display.phase = "image"
        display.log("")
        display.log("  ── PHASE 2: IMAGE GENERATION ───────────────────────────")

        # Load models once for the whole phase
        display.log("  Loading CLIP model...")
        clip_bundle = _load_clip_model(device)
        display.log("  Loading ArcFace model...")
        face_app = _load_face_model()

        for seed_idx, seed_image in enumerate(seeds, 1):
            seed_stem = seed_image.stem
            display.phase_seed_idx = seed_idx

            if is_image_phase_complete(seed_stem, out_root, n_images):
                display.log(f"  [{seed_stem}] image phase already complete — skipping")
                log_path = out_root / "images" / seed_stem / "log.json"
                if log_path.exists():
                    img_logs[seed_stem] = json.loads(log_path.read_text(encoding="utf-8"))
                continue

            display.log(f"  [{seed_idx}/{n_seeds}] Starting images for: {seed_stem}")
            log = run_image_phase_for_seed(
                seed_image, seed_stem, out_root, n_images,
                clip_bundle, face_app, device,
            )
            img_logs[seed_stem] = log
    else:
        display.log("  ── PHASE 2: SKIPPED (--skip-images) ────────────────────")
        for seed_image in seeds:
            seed_stem = seed_image.stem
            log_path  = out_root / "images" / seed_stem / "log.json"
            if log_path.exists():
                img_logs[seed_stem] = json.loads(log_path.read_text(encoding="utf-8"))

    # ================================================================
    # PHASE 3 — MASTER SUMMARY
    # ================================================================
    display.log("")
    display.log("  ── PHASE 3: MASTER SUMMARY ─────────────────────────────")
    write_master_summary(out_root, seeds, n_captions, n_images, cap_logs, img_logs)

    display.finalize()
    print(f"\nOutput directory: {out_root.resolve()}")
    print(f"Master results:   {(out_root / 'master_results.txt').resolve()}")
    print(f"Master log:       {(out_root / 'master_log.json').resolve()}")


# ================================
# ENTRYPOINT
# ================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Batch pipeline: caption sampling + image sampling + CLIP/ArcFace "
            "analysis across an entire directory of seed images."
        )
    )
    parser.add_argument(
        "--seed-dir", "-d",
        type=str, required=True,
        help="Path to directory containing seed images.",
    )
    parser.add_argument(
        "--n-captions", "-c",
        type=int, default=25,
        help="Number of independent captions per seed (default: 25).",
    )
    parser.add_argument(
        "--n-images", "-i",
        type=int, default=25,
        help="Number of independent images per seed (default: 25).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str, default=OUTPUT_DIR,
        help=f"Root output directory (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--skip-captions",
        action="store_true", default=False,
        help="Skip captioning phase; use existing medoid captions.",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true", default=False,
        help="Skip image generation phase; run analysis only.",
    )
    parser.add_argument(
        "--device",
        type=str, default=None,
        help="Force CLIP device: 'cpu' or 'cuda'. Auto-detected if omitted.",
    )
    args = parser.parse_args()

    seed_dir = Path(args.seed_dir)
    if not seed_dir.exists() or not seed_dir.is_dir():
        print(f"ERROR: seed directory not found: {seed_dir}")
        sys.exit(1)

    out_root = Path(args.output_dir)
    ensure_dir(out_root)

    # Normalize all seed images to valid PNGs before anything else runs
    normalize_seed_images(seed_dir)

    # Discover seeds early so we can show count in the confirmation screen
    seeds = discover_seeds(seed_dir)
    if not seeds:
        print(f"ERROR: No image files ({', '.join(SEED_EXTENSIONS)}) found in {seed_dir}")
        sys.exit(1)

    # Prompt selection (once, applied to all seeds)
    if not args.skip_captions:
        print()
        print("=" * UI_WIDTH)
        print("  seed_sampler — batch seed image pipeline")
        print("=" * UI_WIDTH)
        print(f"  Seed directory : {seed_dir.resolve()}")
        print(f"  Seeds found    : {len(seeds)}")
        print(f"  Captions/seed  : {args.n_captions}")
        print(f"  Images/seed    : {args.n_images}")
        print(f"  Output         : {out_root.resolve()}")
        print()

        prompt_key, prompt_text = select_prompt()
    else:
        prompt_key  = "objective"
        prompt_text = PROMPTS["objective"]

    print()
    print("=" * UI_WIDTH)
    print(f"  Seed directory : {seed_dir.resolve()}")
    print(f"  Seeds found    : {len(seeds)}")
    for i, s in enumerate(seeds, 1):
        print(f"    {i:3d}. {s.name}")
    print(f"  Captions/seed  : {args.n_captions}")
    print(f"  Images/seed    : {args.n_images}")
    print(f"  Prompt         : [{prompt_key}]  {prompt_text[:55]}{'...' if len(prompt_text) > 55 else ''}")
    print(f"  Caption model  : {CAPTION_MODEL}  ({CAPTION_API_MODE} API)")
    print(f"  Image model    : {IMAGE_MODEL}  {IMAGE_SIZE}  quality={IMAGE_QUALITY}")
    print(f"  Skip captions  : {args.skip_captions}")
    print(f"  Skip images    : {args.skip_images}")
    print(f"  Output         : {out_root.resolve()}")
    print("=" * UI_WIDTH)

    confirm = get_input("  Start? [Y/n]: ").strip().lower()
    if confirm not in ("", "y", "yes"):
        print("Aborted.")
        sys.exit(0)

    display.phase_seed_total = len(seeds)
    display.start_live()
    pause_handler.arm()

    run_seed_sampler(
        seed_dir     = seed_dir,
        n_captions   = args.n_captions,
        n_images     = args.n_images,
        out_root     = out_root,
        skip_captions= args.skip_captions,
        skip_images  = args.skip_images,
        device       = args.device,
        prompt_key   = prompt_key,
        prompt_text  = prompt_text,
    )


if __name__ == "__main__":
    main()