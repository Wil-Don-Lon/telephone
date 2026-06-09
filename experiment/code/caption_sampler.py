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

caption_sampler.py — Seed image caption distribution sampler.

Purpose:
    Generate N independent captions from a single seed image with no feedback
    loop. Runs semantic analysis on the resulting caption distribution and
    identifies the "most average" caption (embedding-space medoid) for use as
    a stable, representative prompt in prompt_sampler.py.

    This audits the caption model's own stochasticity before any generation
    occurs, answering:
      - How consistent is GPT-4o/5 when describing the same image?
      - Does it produce refusal language, and how often?
      - Does sentiment or vocabulary vary meaningfully across runs?
      - What is the single most representative caption to use downstream?

Pipeline:
    seed image  →  caption_sampler.py  →  captions + analysis + medoid caption
                                                    ↓
                                           prompt_sampler.py  (feed medoid as prompt)

Features:
    - Configurable N (number of captions) via --n or interactive input
    - Supports both chat completions and responses API (matches telephone.py)
    - Checkpoint/resume: safe to interrupt and restart
    - Semantic analysis on completion:
        - Pairwise cosine similarity distribution (how consistent is the model?)
        - Medoid caption (most representative by embedding distance)
        - Word frequency and top terms
        - Refusal language detection
        - Sentiment distribution
        - Caption length stats
    - Medoid caption written to medoid_caption.txt for direct use in prompt_sampler.py
    - Full results appended to log.json

Dependencies:
    pip install openai requests pillow sentence-transformers nltk textblob

Usage:
    python caption_sampler.py                                    # fully interactive
    python caption_sampler.py --image seed_images/jfk.jpg --n 25
    python caption_sampler.py --resume 20260407_143200_jfk
"""

import os
import re
import json
import time
import base64
import signal
import sys
import argparse
import string
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

OUTPUT_DIR      = "caption_sampler_output"
CHECKPOINT_FILE = "caption_checkpoint.json"

MAX_RETRIES         = 3
RETRY_DELAY         = 60
API_TIMEOUT_SECS    = 300
MAX_TIMEOUT_RETRIES = 3

# Caption model — matches telephone.py defaults
CAPTION_MODEL      = "gpt-5.2-chat-latest"
CAPTION_API_MODE   = "responses"   # "chat" or "responses"
CAPTION_MAX_TOKENS = 900

# Captioning prompts — keep in sync with telephone.py PROMPTS dict.
# Add new entries here to extend the selection menu.
PROMPTS = {
    "objective": (
        "Describe this image with objective precision, including any publicly "
        "documented subjects, figures, events, or locations visible. Be as "
        "specific, vivid, descriptive, and factually accurate as possible."
    ),
}

# ================================
# STOP WORDS + REFUSAL PATTERNS
# (kept in sync with semantic_analysis.py)
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


def slugify(text: str, max_len: int = 40) -> str:
    text = Path(text).stem.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "_", text)
    return text[:max_len].strip("_-")


def fmt_duration(seconds: float) -> str:
    td = timedelta(seconds=int(seconds))
    h, rem = divmod(td.seconds + td.days * 86400, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def tokenize(text: str) -> list[str]:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]


def detect_refusals(caption: str) -> list[str]:
    lower = caption.lower()
    lower = lower.replace("\u2019", "'").replace("\u2018", "'")
    lower = lower.replace("\u201c", '"').replace("\u201d", '"')
    lower = lower.replace("\u2013", "-").replace("\u2014", "-")
    return [p for p in REFUSAL_PATTERNS if p in lower]


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
    # Resize if very large to keep token cost reasonable
    max_dim = 1024
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), PILImage.LANCZOS)
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"


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

UI_WIDTH = 78
BAR_WIDTH = 30


def progress_bar(done: int, total: int, width: int = BAR_WIDTH) -> str:
    frac   = done / total if total > 0 else 0
    filled = int(frac * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def separator(char: str = "-", width: int = UI_WIDTH) -> str:
    return char * width


class Display:
    FOOTER_LINES = 7

    def __init__(self):
        self._lock           = threading.Lock()
        self._log_buf        = []
        self._live           = False
        self._timer          = None
        self._paused_seconds = 0.0
        self._pause_start    = None
        self._start_time     = datetime.now()

        self.run_label:     str = ""
        self.image_label:   str = ""
        self.done:          int = 0
        self.total:         int = 0
        self.refusals:      int = 0
        self.errors:        int = 0
        self.stage:         str = ""

    def record_pause(self):
        self._pause_start = time.monotonic()

    def record_resume(self):
        if self._pause_start is not None:
            self._paused_seconds += time.monotonic() - self._pause_start
            self._pause_start = None

    def active_seconds(self) -> float:
        elapsed = (datetime.now() - self._start_time).total_seconds()
        return max(0.0, elapsed - self._paused_seconds)

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

        elapsed = fmt_duration(self.active_seconds())
        left1   = f"  {self.run_label}"
        right1  = f"{elapsed}  "
        lines.append(left1 + " " * max(1, w - len(left1) - len(right1)) + right1)

        bar   = progress_bar(self.done, self.total)
        left2 = f"  {bar} {self.done}/{self.total}"
        right2 = f"{self.stage}  "
        lines.append(left2 + " " * max(1, w - len(left2) - len(right2)) + right2)

        lines.append(separator("-", w))

        left3  = f"  refusals detected: {self.refusals}"
        right3 = f"errors: {self.errors}  "
        lines.append(left3 + " " * max(1, w - len(left3) - len(right3)) + right3)

        lines.append(f"  {self.image_label[:w-4]}")
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
            f"  DONE  {self.done}/{self.total} captions"
            f"  |  refusals: {self.refusals}"
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
        display.log("Ctrl+C — pausing after this caption. Ctrl+C again to force quit.")
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
# CAPTIONING
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
        if ("content policy" in lower or "safety system" in lower
                or "content_policy" in lower):
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
                display.log(f"[caption] rate limit — waiting {RETRY_DELAY}s "
                            f"(retry {retry_count + 1}/{MAX_RETRIES})...")
                time.sleep(RETRY_DELAY)
                return caption_image(image_path, retry_count + 1, timeout_count)
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
    cp  = run_dir / CHECKPOINT_FILE
    tmp = cp.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.rename(cp)


# ================================
# ANALYSIS
# ================================

def run_analysis(run_dir: Path, captions: list[str]) -> dict:
    """
    Runs semantic analysis on the collected captions.
    Returns analysis dict and writes medoid_caption.txt.
    """
    import numpy as np

    n = len(captions)
    if n == 0:
        print("\n  [analysis] No captions to analyse.")
        return {}

    print(f"\n{'=' * UI_WIDTH}")
    print(f"  CAPTION ANALYSIS  ({n} captions)")
    print(f"{'=' * UI_WIDTH}")

    # ---- Basic stats ----
    lengths      = [len(c.split()) for c in captions]
    char_lengths = [len(c) for c in captions]
    print(f"\n  Caption length (words):  mean={sum(lengths)/n:.1f}  "
          f"min={min(lengths)}  max={max(lengths)}")

    # ---- Refusal detection ----
    refusal_counts = [len(detect_refusals(c)) for c in captions]
    n_with_refusal = sum(1 for r in refusal_counts if r > 0)
    all_patterns   = []
    for c in captions:
        all_patterns.extend(detect_refusals(c))
    pattern_freq = Counter(all_patterns).most_common(5)

    print(f"\n  Refusal language:")
    print(f"    captions with refusal: {n_with_refusal}/{n} ({100*n_with_refusal/n:.1f}%)")
    if pattern_freq:
        print(f"    top patterns: {', '.join(f'{p!r} ({c}x)' for p, c in pattern_freq)}")

    # ---- Sentiment ----
    sentiments = [sentiment_score(c) for c in captions]
    scores     = [s for s, _ in sentiments]
    labels     = [l for _, l in sentiments]
    label_dist = Counter(labels)
    print(f"\n  Sentiment distribution:")
    print(f"    mean score = {sum(scores)/n:+.3f}")
    for label in ("positive", "neutral", "negative"):
        ct = label_dist.get(label, 0)
        print(f"    {label:10s}: {ct}/{n} ({100*ct/n:.1f}%)")

    # ---- Word frequency ----
    all_words = []
    for c in captions:
        all_words.extend(tokenize(c))
    top_words = Counter(all_words).most_common(20)
    print(f"\n  Top 20 words across all captions:")
    print(f"    {', '.join(f'{w}({c})' for w, c in top_words)}")

    # ---- Semantic similarity (sentence transformers) ----
    embeddings   = None
    medoid_idx   = 0
    medoid_score = None
    pairwise_mean = pairwise_std = pairwise_min = pairwise_max = None

    try:
        from sentence_transformers import SentenceTransformer
        print(f"\n  Loading sentence transformer (all-MiniLM-L6-v2)...")
        model      = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(captions, normalize_embeddings=True,
                                  show_progress_bar=False)
        print(f"  Computing pairwise similarities ({n*(n-1)//2} pairs)...")

        sims = embeddings @ embeddings.T  # (n, n)

        # Pairwise distribution (upper triangle)
        pw = [float(sims[i, j]) for i in range(n) for j in range(i+1, n)]
        pairwise_mean = float(np.mean(pw))
        pairwise_std  = float(np.std(pw))
        pairwise_min  = float(np.min(pw))
        pairwise_max  = float(np.max(pw))

        print(f"\n  Pairwise caption similarity:")
        print(f"    mean = {pairwise_mean:.4f}  (1.0 = model always says same thing)")
        print(f"    std  = {pairwise_std:.4f}")
        print(f"    min  = {pairwise_min:.4f}")
        print(f"    max  = {pairwise_max:.4f}")

        # Medoid: caption with highest mean similarity to all others
        # Exclude self-similarity (diagonal = 1.0) from the mean
        mean_sim_to_others = (sims.sum(axis=1) - 1.0) / (n - 1)
        medoid_idx   = int(np.argmax(mean_sim_to_others))
        medoid_score = float(mean_sim_to_others[medoid_idx])

        print(f"\n  Medoid caption (most representative):")
        print(f"    index = {medoid_idx + 1}  (mean sim to others = {medoid_score:.4f})")
        print(f"    preview: {captions[medoid_idx][:120]}...")

    except ImportError:
        print("\n  [analysis] sentence-transformers not installed — skipping embedding analysis.")
        print("  Install with:  pip install sentence-transformers")
        # Fall back to longest caption as a crude proxy for medoid
        medoid_idx = max(range(n), key=lambda i: len(captions[i].split()))
        print(f"  Fallback: using longest caption as medoid (index {medoid_idx + 1})")

    medoid_caption = captions[medoid_idx]

    # Write medoid to file
    medoid_path = run_dir / "medoid_caption.txt"
    medoid_path.write_text(medoid_caption, encoding="utf-8")
    print(f"\n  Medoid caption written → {medoid_path}")
    print(f"  (Use this as the prompt for prompt_sampler.py)")

    # ---- Assemble result dict ----
    analysis = {
        "n_captions": n,
        "length_stats": {
            "mean_words":  round(sum(lengths) / n, 1),
            "min_words":   min(lengths),
            "max_words":   max(lengths),
            "mean_chars":  round(sum(char_lengths) / n, 1),
        },
        "refusal": {
            "n_with_refusal":  n_with_refusal,
            "refusal_rate":    round(n_with_refusal / n, 4),
            "top_patterns":    [{"pattern": p, "count": c} for p, c in pattern_freq],
        },
        "sentiment": {
            "mean_score":  round(sum(scores) / n, 4),
            "positive":    label_dist.get("positive", 0),
            "neutral":     label_dist.get("neutral", 0),
            "negative":    label_dist.get("negative", 0),
        },
        "top_words": [{"word": w, "count": c} for w, c in top_words],
        "pairwise_similarity": {
            "mean": round(pairwise_mean, 4) if pairwise_mean is not None else None,
            "std":  round(pairwise_std,  4) if pairwise_std  is not None else None,
            "min":  round(pairwise_min,  4) if pairwise_min  is not None else None,
            "max":  round(pairwise_max,  4) if pairwise_max  is not None else None,
        },
        "medoid": {
            "index":          medoid_idx,
            "mean_sim_score": round(medoid_score, 4) if medoid_score is not None else None,
            "caption":        medoid_caption,
            "file":           "medoid_caption.txt",
        },
    }

    print(f"\n{'=' * UI_WIDTH}\n")
    return analysis


# ================================
# PROMPT SELECTION
# ================================

def select_prompt() -> tuple[str, str]:
    """
    Presents the available PROMPTS entries as a numbered menu with option 0
    for free-text entry. Loops until the user confirms their selection.
    Returns (prompt_key, prompt_text).
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
            print(f"\n  Your prompt:\n    {text[:80]}{'...' if len(text)>80 else ''}\n")
            confirm = get_input("  Use this prompt? [Y/n]: ").strip().lower()
            if confirm in ("", "y", "yes"):
                return "custom", text
            continue

        key = keys[choice - 1]
        return key, PROMPTS[key]


# ================================
# MAIN RUN
# ================================

def run_sampler(image_path: Path, n: int, run_dir: Path, prompt_key: str = "objective", prompt_text: str = ""):
    captions_dir = run_dir / "captions"
    ensure_dir(captions_dir)

    # Load or init checkpoint
    cp = load_checkpoint(run_dir)
    if cp is not None:
        if cp.get("image") != str(image_path) or cp.get("n") != n:
            print("ERROR: checkpoint mismatch. Delete the run folder to start fresh.")
            sys.exit(1)
        completed   = set(cp.get("completed", []))
        captions    = cp.get("captions", [])
        policy_hits = cp.get("policy_hits", [])
        errors_log  = cp.get("errors_log", [])
        display.log(f"Resuming: {len(completed)}/{n} already done.")
    else:
        completed   = set()
        captions    = []
        policy_hits = []
        errors_log  = []
        save_checkpoint(run_dir, {
            "image":       str(image_path),
            "n":           n,
            "prompt_key":  prompt_key,
            "prompt_text": prompt_text,
            "completed":   [],
            "captions":    [],
            "policy_hits": [],
            "errors_log":  [],
            "started_at":  datetime.now().isoformat(),
        })

    display.run_label   = f"caption_sampler  n={n}"
    display.image_label = f"image: {image_path.name}"
    display.total       = n
    display.done        = len(completed)
    display.refusals    = sum(1 for c in captions if detect_refusals(c))
    display.errors      = len(errors_log)
    display.start_live()
    pause_handler.arm()

    for i in range(1, n + 1):
        pause_handler.check()

        if i in completed:
            continue

        display.stage = "[caption]"
        display.log(f"  caption {i:04d}/{n} ...")

        caption_text, err_type, err_msg = caption_image(image_path, prompt_text)

        if caption_text is not None:
            completed.add(i)
            captions.append(caption_text)

            # Write individual caption file
            (captions_dir / f"caption_{i:04d}.txt").write_text(
                caption_text, encoding="utf-8"
            )

            has_refusal = bool(detect_refusals(caption_text))
            if has_refusal:
                display.refusals += 1
                display.log(f"  ✓ caption {i:04d}  [REFUSAL LANGUAGE DETECTED]")
            else:
                display.log(f"  ✓ caption {i:04d}")
            display.done = len(completed)

        else:
            if err_type == "CONTENT_POLICY_VIOLATION":
                policy_hits.append({"sample": i, "error": err_msg})
                display.log(f"  ✗ caption {i:04d} POLICY BLOCK")
                completed.add(i)
                display.done = len(completed)
            elif err_type == "INSUFFICIENT_QUOTA":
                display.log("  QUOTA EXHAUSTED — aborting.")
                break
            else:
                errors_log.append({"sample": i, "error_type": err_type, "error_msg": err_msg})
                display.errors = len(errors_log)
                display.log(f"  ✗ caption {i:04d} ERROR ({err_type})")

        save_checkpoint(run_dir, {
            "image":       str(image_path),
            "n":           n,
            "completed":   sorted(completed),
            "captions":    captions,
            "policy_hits": policy_hits,
            "errors_log":  errors_log,
            "started_at":  cp["started_at"] if cp else datetime.now().isoformat(),
            "updated_at":  datetime.now().isoformat(),
        })

    display.stage = ""

    # Write all captions to single file
    all_captions_path = run_dir / "all_captions.txt"
    all_captions_path.write_text(
        "\n\n---\n\n".join(f"[{i+1}]\n{c}" for i, c in enumerate(captions)),
        encoding="utf-8",
    )

    # Write final log
    log = {
        "run_id":           run_dir.name,
        "image":            str(image_path),
        "n_requested":      n,
        "n_generated":      len(captions),
        "n_policy_blocked": len(policy_hits),
        "n_errors":         len(errors_log),
        "caption_model":    CAPTION_MODEL,
        "prompt_key":       prompt_key,
        "caption_prompt":   prompt_text,
        "policy_hits":      policy_hits,
        "errors_log":       errors_log,
        "finished_at":      datetime.now().isoformat(),
    }
    log_path = run_dir / "log.json"
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")

    display.finalize()
    print(f"\nOutput directory: {run_dir.resolve()}")
    print(f"Captions:         {captions_dir.resolve()}")

    # Post-run analysis
    if captions:
        analysis = run_analysis(run_dir, captions)
        log["analysis"] = analysis
        log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
        print(f"Log updated with analysis → {log_path}")
    else:
        print("\n  No successful captions to analyse.")


# ================================
# ENTRYPOINT
# ================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate N captions from a seed image and analyse the distribution."
    )
    parser.add_argument(
        "--image", "-i",
        type=str, default=None,
        help="Path to seed image. If omitted, asked interactively.",
    )
    parser.add_argument(
        "--n", "-n",
        type=int, default=None,
        help="Number of captions to generate. If omitted, asked interactively.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str, default=OUTPUT_DIR,
        help=f"Root output directory (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--resume",
        type=str, default=None, metavar="RUN_ID",
        help="Resume a specific run by its run ID.",
    )
    parser.add_argument(
        "--name",
        type=str, default=None, metavar="NAME",
        help="Human-readable name for this run (e.g. 'jfk'). "
             "Used instead of the auto-generated image slug. "
             "Timestamp is always prepended.",
    )
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    ensure_dir(out_root)

    if args.resume:
        run_dir = out_root / args.resume
        if not run_dir.exists():
            print(f"ERROR: run folder not found: {run_dir}")
            sys.exit(1)
        cp = load_checkpoint(run_dir)
        if cp is None:
            print(f"ERROR: no checkpoint found in {run_dir}")
            sys.exit(1)
        image_path  = Path(cp["image"])
        n           = cp["n"]
        prompt_key  = cp.get("prompt_key", "objective")
        prompt_text = cp.get("prompt_text", list(PROMPTS.values())[0])
        print(f"Resuming: {args.resume}")
        print(f"  Image  : {image_path}")
        print(f"  N      : {n}")
        print(f"  Prompt : [{prompt_key}]")
        print(f"  Done   : {len(cp.get('completed', []))}/{n}")
    else:
        # Interactive / CLI collection
        if args.image:
            image_path = Path(args.image)
        else:
            print("\n" + "=" * UI_WIDTH)
            print("  caption_sampler — seed image caption distribution sampler")
            print("=" * UI_WIDTH)
            print("  Generates N independent captions from one image and analyses")
            print("  the distribution to find the most representative (medoid) caption.")
            print()
            raw = get_input("  Seed image path: ").strip()
            if not raw:
                print("No path entered. Exiting.")
                sys.exit(0)
            image_path = Path(raw)

        if not image_path.exists():
            print(f"ERROR: image not found: {image_path}")
            sys.exit(1)

        if args.n:
            n = args.n
        else:
            print()
            raw = get_input("  Number of captions (N) [25]: ").strip()
            if not raw:
                n = 25
            else:
                try:
                    n = int(raw)
                    if n < 1:
                        raise ValueError
                except ValueError:
                    print("Invalid number. Exiting.")
                    sys.exit(1)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.name:
            run_id = re.sub(r"[^a-z0-9_-]", "_", args.name.strip().lower())[:50]
        else:
            print()
            raw_name = get_input("  Run name [blank for auto]: ").strip()
            if raw_name:
                run_id = re.sub(r"[^a-z0-9_-]", "_", raw_name.lower())[:50]
            else:
                run_id = f"{ts}_{slugify(str(image_path))}"
        run_dir = out_root / run_id
        ensure_dir(run_dir)

        # Prompt selection
        prompt_key, prompt_text = select_prompt()

        print()
        print("=" * UI_WIDTH)
        print(f"  Run ID : {run_id}")
        print(f"  Image  : {image_path}")
        print(f"  N      : {n}")
        print(f"  Prompt : [{prompt_key}]  {prompt_text[:60]}{'...' if len(prompt_text)>60 else ''}")
        print(f"  Model  : {CAPTION_MODEL}  ({CAPTION_API_MODE} API)")
        print(f"  Output : {run_dir.resolve()}")
        print("=" * UI_WIDTH)
        confirm = get_input("  Start? [Y/n]: ").strip().lower()
        if confirm not in ("", "y", "yes"):
            print("Aborted.")
            sys.exit(0)

    run_sampler(image_path, n, run_dir, prompt_key=prompt_key, prompt_text=prompt_text)


if __name__ == "__main__":
    main()
