# Generative Telephone — A Framework for Studying Model Bias Through Multimodal Feedback Loops

This repository provides a **reproducible framework** for investigating how bias,
distortion, and identity loss accumulate when generative AI models are placed in a closed
loop. It plays a game of **"telephone"** between a vision–language captioner and a
text-to-image generator: an image is described, the description is re-rendered into a new
image, that image is described again, and so on. Iterating this `caption → image →
caption` cycle exposes the priors and biases baked into the underlying models — small
distortions compound with every pass.

The framework was built for an honors thesis, but it is **model-agnostic and
dataset-agnostic**: point it at your own seed images, swap in whichever caption/image
models you have access to, and run the same pipeline to measure drift in your own
experiments. The thesis write-up (`Thesis Final Draft.pdf`) is included as a reference
for methodology and findings.

> Copyright © William Donnell-Lonon, 2026. See the license header in
> `experiment/code/telephone.py` for usage and attribution terms.

---

## The core idea

```
  seed image ──► [caption model] ──► caption ──► [image model] ──► new image
                       ▲                                                │
                       └────────────────────────────────────────────────┘
                                    repeat for N iterations
```

Each pass through the loop is one **iteration**. Several independent **chains** run from
each **seed image** so drift can be measured as a distribution rather than a single
trajectory. Because nothing external corrects the loop, any bias in either model — toward
a demographic, a composition, a stereotype — compounds with each pass. The analysis
scripts then quantify that drift along three axes:

- **Visual / compositional drift** — CLIP cosine similarity between each generated image and its seed.
- **Identity drift** — ArcFace face-embedding similarity, tracking how long a person's identity survives.
- **Semantic drift** — sentence-embedding distance between successive captions, plus sentiment and vocabulary change.

Content-policy **refusals** (when a model declines to caption or generate) are logged and
analyzed as a form of bias in their own right.

---

## Installation

```bash
git clone <this-repo>
cd <repo>/experiment/code      # all scripts live and run here
pip install -r ../../requirements.txt
python -m nltk.downloader vader_lexicon   # one-time, for sentiment analysis
# ffmpeg must also be installed (used for animation/video output)

export OPENAI_API_KEY="sk-..."            # required by the generation scripts
```

Requirements: Python 3.10+, an OpenAI API key (or another provider if you adapt the
client code), and optionally a CUDA GPU (CLIP/ArcFace fall back to CPU automatically).

---

## How to use this framework

All scripts live in `experiment/code/` and are designed to be **run from inside that
directory** — they read and write relative paths (e.g. `seed_images/`, `output/`) right
where they run. The typical workflow:

1. **Generate** — drop your seed images in a folder and run `telephone.py` to produce the feedback-loop chains.
2. **Analyze** — run `clip_analysis.py`, `face_analysis.py`, and `semantic_analysis.py` over the results.
3. **Visualize** — run `meta_analysis.py` / `results_viz.py` / `spiral_viz.py` to produce figures, tables, and posters.
4. *(Optional)* **Baseline** — use the `*_sampler.py` scripts to generate no-feedback control distributions for comparison.

### Adapting it to your own dataset

A few analysis/visualization scripts (`clip_analysis.py`, `face_analysis.py`,
`semantic_analysis.py`, `spiral_viz.py`, `meta_analysis*.py`) contain a `SEED_METADATA` /
`SEED_DISPLAY_NAMES` / `CATEGORY_COLORS` block near the top that maps **specific seed
image names to display labels and categories**. These are placeholders from the original
study — edit them to match your own seed set (or the category groupings will simply not
apply). Everything else keys off your seed filenames automatically.

---

## Script reference

### Stage 1 — Generation

#### `telephone.py` — the feedback-loop engine

- **Purpose:** Runs the core `caption → image → caption` loop across every seed image, producing multiple independent chains per seed.
- **Configuration** (top-of-file `CONFIGURATION` block): `SEED_IMAGES_DIR` (default `seed_images`), `OUTPUT_DIR` (`output`), `ITERATIONS` (5), `CHAINS_PER_SEED` (4), the caption model + API mode (`gpt-5.2-chat-latest`, `responses`), the image model (`gpt-image-1.5`), size/quality, and the `PROMPTS` dictionary of captioning styles. Retry, timeout, and post-mortem settings are here too.
- **Usage:** `python telephone.py` (interactive — prompts for prompt style, chains, iterations, and resume/new-batch). Flags: `--iterations N`, `--chains N`, `--prompt <name>`.
- **Behavior:** Checkpoint/resume safe (`Ctrl+C` to pause, re-run to continue); atomic image writes; rate-limit/timeout retries; content-policy "post-mortem" investigation on refusals; live terminal UI; per-chain token accounting.
- **Outputs:** `output/<seed>/<prompt_type>/chain_<N>/` containing each iteration's image, a per-chain `log.json` + `summary.txt`, plus `violations/`, `post_mortem/`, and a `master_log.json` rebuilt across the full run.

### Stage 2 — Analysis

#### `clip_analysis.py` — compositional / visual drift

- **Purpose:** Measures how much each generated image still resembles its seed using CLIP cosine similarity, quantifying compositional decay across iterations.
- **Configuration:** `SEED_METADATA`, `CATEGORY_COLORS`, `FIGURE_DPI`/`FIGURE_STYLE` (top of file).
- **Usage:** `python clip_analysis.py --data-dir output --output-dir clip_analysis_output` (`--log` points to `master_log.json`; `--no-clip` reuses cached scores; `--device cpu|cuda`; `--scores-csv` to import precomputed scores).
- **Models:** CLIP (ViT-B-32, LAION-2B) via `open_clip`.
- **Outputs:** `clip_analysis_output/` — per-image score CSVs, decay-curve and heatmap figures, preservation tables, and regression/ANOVA summaries.

#### `face_analysis.py` — identity drift

- **Purpose:** Tracks facial-identity preservation per chain and detects the iteration at which a face disappears entirely.
- **Configuration:** `SEED_METADATA`, `CATEGORY_COLORS`, figure settings.
- **Usage:** `python face_analysis.py --data-dir output --output-dir face_analysis_output` (`--clip-scores` to overlay CLIP for comparison; `--scores-csv` to import precomputed face scores).
- **Models:** ArcFace via `insightface` (`buffalo_l`).
- **Outputs:** `face_analysis_output/` — per-chain identity-match trajectories, CSVs, and CLIP-vs-identity comparison figures.

#### `semantic_analysis.py` — caption / semantic drift

- **Purpose:** Measures how captions evolve — absolute drift vs. iteration 1, relative drift between consecutive captions, sentiment trajectories, refusal detection, and vocabulary divergence.
- **Configuration:** `SEED_METADATA`, `STOP_WORDS`, `REFUSAL_PATTERNS`, `POSITIVE/NEGATIVE_WORDS`, figure settings.
- **Usage:** `python semantic_analysis.py --data-dir output --output-dir semantic_output --epoch-size 3` (`--captions-csv` to import precomputed embeddings).
- **Models/libs:** sentence-transformers, TextBlob, NLTK VADER.
- **Outputs:** `semantic_output/` — drift CSVs, sentiment heatmaps, word-bump charts, and statistical summaries.

#### `group_photo_analysis.py` — multi-person identity matching

- **Purpose:** For group/multi-face seeds, detects the largest faces left-to-right and matches each by position to the seed's faces, scoring identity preservation per position.
- **Configuration:** `MAX_FACES` (4), `THRESHOLD` (0.3), `FACE_COLORS`/`FACE_LABELS`.
- **Usage:** `python group_photo_analysis.py --seed-image <path> --gen-dir <dir> --output-dir group_analysis_output` (both `--seed-image` and `--gen-dir` required).
- **Models:** ArcFace via `insightface`.
- **Outputs:** `group_analysis_output/` — per-position identity scores and labeled copies of each generated image.

### Optional — Baseline / control sampling

These generate **no-feedback** distributions, so you can separate "drift caused by the loop" from "raw model variance."

#### `caption_sampler.py` — caption distribution + medoid

- **Purpose:** Generates N independent captions of one image, embeds them, and selects the **medoid** (most representative) caption. Also reports refusals, sentiment, and word frequency.
- **Usage:** `python caption_sampler.py --image <path> --n 25` (interactive if flags omitted; `--output-dir`, `--name`, `--resume <run_id>`).
- **Outputs:** `caption_sampler_output/<run>/` — all captions, the `medoid_caption.txt`, and distribution stats.

#### `image_sampler.py` — image distribution + variance

- **Purpose:** Generates N images from a fixed prompt (no loop) and measures variance via pairwise CLIP and ArcFace similarity, optionally scored against a ground-truth seed.
- **Usage:** `python image_sampler.py --prompt "<text>" --n 25` (`--seed-image` for ground-truth scoring; `--multi-face` for group seeds; `--output-dir`, `--name`, `--resume`).
- **Models:** GPT Image, CLIP, ArcFace.
- **Outputs:** `image_sampler_output/<run>/` — the images plus variance/ground-truth score CSVs.

#### `seed_sampler.py` — batch sampling pipeline

- **Purpose:** Runs caption sampling, medoid selection, and image sampling across **many seeds at once**, with shared checkpoint/resume and aggregate logging.
- **Usage:** `python seed_sampler.py --seed-dir <dir> --n-captions 25 --n-images 25` (`--skip-captions`, `--skip-images`, `--output-dir`, `--device`).
- **Outputs:** `seed_distributions/` — per-seed sampling results and a master summary.

### Visualization & reporting

#### `meta_analysis.py` / `meta_analysis_vertical.py` — main thesis figures

- **Purpose:** Builds the headline figures: CLIP-vs-ArcFace scatters (aggregate, by iteration, by seed), identity-match trajectories, sentiment heatmaps, per-seed overview tables, and **animated progression videos**. `_vertical.py` is a portrait-orientation variant for posters/slides.
- **Configuration:** `IDENTITY_THRESHOLD` (0.3), `FIGURE_DPI` (200), `VIDEO_FPS` (5), `SEED_DISPLAY_NAMES`.
- **Usage:** `python meta_analysis.py` (interactive — prompts for the telephone output and baseline directories).
- **Outputs:** PNG figures, overview CSVs, and MP4 videos (requires ffmpeg).

#### `results_viz.py` — simplified figure suite

- **Purpose:** A lighter-weight version of `meta_analysis.py`: scatters, iteration/seed grids, validation comparison, sentiment facets, an overview table, and optional per-seed chain animations.
- **Configuration:** `IDENTITY_THRESHOLD`, `FIGURE_DPI`, `VIDEO_FPS`.
- **Usage:** `python results_viz.py` (interactive).
- **Outputs:** PNG figures and optional MP4 videos.

#### `spiral_viz.py` — golden-ratio poster

- **Purpose:** Arranges every chain image around its center seed on a Fermat spiral (golden angle) to create a high-resolution poster of a chain's evolution.
- **Configuration:** `SHAPE` (`full`/`square`/`circle`), `CANVAS_SIZE`, `CELL_GAP`, `SEED_BOOST`, `ROTATE`, and `SEED_METADATA`.
- **Usage:** `python spiral_viz.py --seed "<name>" --shape circle --size 6000` (`--gap`, `--seed-boost`, `--rotate/--no-rotate`, `--no-label`, `--list` to show available seeds).
- **Outputs:** A single high-resolution poster PNG.

#### `seed_grid.py` — face-centered seed grid

- **Purpose:** Builds a 3×5 grid of face-centered square crops from your seed images (auto-detects face centers).
- **Usage:** `python seed_grid.py <seed_dir> --size 400 --gap 4 --label` (`--output` for filename).
- **Models:** `insightface` for face detection.
- **Outputs:** A single grid PNG.

#### `grid_maker.py` — simple sample grid

- **Purpose:** Composites `sample_0001.png … sample_0025.png` from a folder into a 5×5 grid.
- **Usage:** `python grid_maker.py <folder> --cell-size 512 --gap 4 --bg white` (`--output` for filename).
- **Outputs:** A single grid PNG.

### Utilities

#### `rebuild_master_log.py` — repair / regenerate the run log

- **Purpose:** Rebuilds `master_log.json` by scanning all per-chain `log.json` files — aggregating policy blocks, token usage, early stops, and completion stats. Always produces a complete log, even across resumed runs.
- **Usage:** `python rebuild_master_log.py --output-dir output`.
- **Outputs:** A freshly written `master_log.json`.

---

## Datasets

The seed images and study sets used in the thesis are a sample. The **full datasets are
available upon request** — email **williamjlonon@gmail.com**.

---

## Citation

If you use this framework in academic work, please cite the accompanying thesis
(`Thesis Final Draft.pdf`) and credit William Donnell-Lonon, per the license terms in
`experiment/code/telephone.py`.
