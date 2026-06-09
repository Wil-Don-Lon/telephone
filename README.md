# Generative Telephone: Investigating Model Bias Through Multimodal Feedback Loops

An honors thesis that probes how bias, distortion, and identity loss accumulate when
generative AI models are made to talk to one another in a closed loop. The experiment
plays a game of **"telephone"** between a vision–language captioner and a text-to-image
generator: an image is described, the description is re-rendered into a new image, the
new image is described again, and so on. By iterating this caption → image → caption
cycle, the project measures how quickly visual composition, facial identity, and
semantic meaning drift away from a real seed image — and what that drift reveals about
the priors and biases baked into the underlying models.

> Copyright © William Donnell-Lonon, 2026. All rights reserved. See the license header
> in `experiment/code/telephone.py` for usage and attribution terms.

---

## The core idea

```
  seed image ──► [caption model] ──► caption ──► [image model] ──► new image
                       ▲                                                │
                       └────────────────────────────────────────────────┘
                                    repeat for N iterations
```

Each pass through the loop is one **iteration**. Several independent **chains** are run
from each **seed image** so that drift can be measured as a distribution rather than a
single trajectory. Because nothing external corrects the loop, any small bias in either
model — toward a demographic, a composition, a stereotype — compounds with every pass.
The downstream analysis quantifies that compounding along three axes:

- **Visual / compositional drift** — CLIP cosine similarity between each generated image and its seed.
- **Identity drift** — ArcFace face-embedding similarity, tracking how long a person's identity survives.
- **Semantic drift** — sentence-embedding distance between successive captions, plus sentiment and vocabulary change.

The project also instruments **content-policy refusals**: when a model declines to
caption or generate, a "post-mortem" prompt investigates why, and the event is logged so
refusals can be analyzed as a form of bias in their own right.

---

## Repository layout

```
Thesis/
├── README.md                  ← you are here
├── experiment/
│   ├── code/                  ← all experiment + analysis source (see below)
│   ├── seed_images/           ← real source images that seed each chain (test 1..15)
│   ├── output/                ← per-chain results, logs, and master_log.json
│   │   └── <seed>/<prompt_type>/chain_<N>/   ← images + log.json per chain
│   ├── *_output/              ← analysis artifacts (clip, face, semantic, group, etc.)
│   ├── seed_images/, spirals/ ← generated figures and poster assets
│   └── checkpoint.json        ← resume state for long runs
├── data/                      ← study image sets (world leaders, sub-study, etc.)
├── drafts/                    ← thesis writing drafts
├── printing/                  ← poster / print-ready assets
└── Thesis Final Draft.{docx,pdf}, Poster Final Draft.pdf, *.pptx
```

---

## The code (`experiment/code/`)

### The experiment engine

| Script | What it does |
|---|---|
| **`telephone.py`** | The heart of the project. Runs the caption → image → caption feedback loop across all seed images. Supports multiple chains per seed, checkpoint/resume (Ctrl+C safe), atomic image writes, rate-limit and timeout retries, content-policy post-mortems, per-chain `log.json` + summaries, token accounting, and a live terminal UI. Captioning via OpenAI Chat **or** Responses API (e.g. `gpt-5.2-chat-latest`); generation via GPT Image (`gpt-image-1.5`). |

### Sampling (baseline / control generation)

These produce the *no-feedback* baselines used to separate "drift from the loop" from "raw model variance."

| Script | What it does |
|---|---|
| **`caption_sampler.py`** | Generates N independent captions of one seed image, embeds them with sentence-transformers, and selects the **medoid** (most representative) caption. Also flags refusals, sentiment, and word frequency. |
| **`image_sampler.py`** | Generates N images from a fixed prompt (no loop) and measures variance via pairwise CLIP and ArcFace similarity, optionally scored against a ground-truth seed. Supports single- and multi-face matching. |
| **`seed_sampler.py`** | Batch pipeline that runs caption + image sampling across many seeds in parallel, with shared checkpoint/resume and aggregate logging. |

### Analysis

| Script | Measures | Models used |
|---|---|---|
| **`clip_analysis.py`** | Compositional preservation/decay of each image vs. its seed; decay curves, heatmaps, regression + ANOVA. | CLIP (ViT-B-32, LAION-2B) |
| **`face_analysis.py`** | Facial-identity preservation per chain, including when faces disappear entirely; compared against CLIP. | ArcFace (insightface `buffalo_l`) |
| **`group_photo_analysis.py`** | Positional face matching for multi-person images (detects up to 4 faces left-to-right and matches by position). | ArcFace (insightface) |
| **`semantic_analysis.py`** | Caption drift (absolute vs. iteration 1, and consecutive), refusal detection, sentiment trajectories, vocabulary divergence, word-bump charts. | sentence-transformers, TextBlob/VADER |

### Visualization & reporting

| Script | What it produces |
|---|---|
| **`meta_analysis.py`** / **`meta_analysis_vertical.py`** | The main thesis figures: CLIP-vs-ArcFace scatters (aggregate, by iteration, by seed), identity-match trajectories, sentiment heatmaps, per-seed overview tables, and animated progression videos (matplotlib + ffmpeg). |
| **`results_viz.py`** | A simplified figure suite: scatters, iteration/seed grids, validation comparison, sentiment facets, and optional per-seed chain animations. |
| **`spiral_viz.py`** | A golden-ratio Fermat-spiral poster arranging every chain image around its center seed image (high-res, configurable cropping/rotation). |
| **`seed_grid.py`** | A face-centered 3×5 grid of seed crops (auto-detects face centers with insightface). |
| **`grid_maker.py`** | A simple N×N grid composite of sample images. |

### Utilities

| Script | What it does |
|---|---|
| **`rebuild_master_log.py`** | Rebuilds `master_log.json` by scanning all per-chain logs — aggregating policy blocks, token usage, early stops, and completion stats. Always produces a complete log even across resumed runs. |

---

## Getting started

### Requirements

- Python 3.10+
- An OpenAI API key with access to the caption and image models
- Python packages — install them all with:

```bash
pip install -r requirements.txt
python -m nltk.downloader vader_lexicon   # one-time, for sentiment analysis
# ffmpeg must also be installed (used for the animation/video outputs)
```

> Note: model identifiers (`gpt-5.2-chat-latest`, `gpt-image-1.5`) are set at the top of
> `telephone.py` in the `CONFIGURATION` block. Adjust them, the iteration count, and the
> chains-per-seed there to match your access and budget.

### 1. Set your API key

```bash
export OPENAI_API_KEY="sk-..."
```

### 2. Run the feedback loop

```bash
cd experiment
python code/telephone.py
```

Drop seed images into `experiment/seed_images/`. Results are written to
`experiment/output/<seed>/<prompt_type>/chain_<N>/`. The run is resumable — press
`Ctrl+C` to pause and re-run the same command to continue from the last checkpoint.
Completed and policy-violated chains are checkpointed so they never re-run.

### 3. Analyze the results

```bash
cd experiment
python code/clip_analysis.py     --data-dir output --output-dir clip_analysis_output
python code/face_analysis.py     --data-dir output
python code/semantic_analysis.py --data-dir output
```

### 4. Build figures and the poster

```bash
python code/meta_analysis.py     # main thesis figures + videos (interactive prompts)
python code/results_viz.py       # simplified figure suite
python code/spiral_viz.py --seed "test 1" --shape circle --size 6000
```

(Most analysis scripts accept `--help` for the full set of options.)

---

## Outputs

- **`experiment/output/master_log.json`** — the canonical run record (per-chain results, tokens, policy blocks, completion stats), rebuilt from per-chain `log.json` files.
- **`experiment/*_output/`** — per-analysis CSVs, statistical summaries, and PNG figures.
- **`experiment/spirals/`, `seed_grid.png`** — poster and grid visualizations.

---

## Datasets

The seed images and study sets included here are a sample. The **full datasets are
available upon request** — email **williamjlonon@gmail.com**.

---

## Thesis documents

The full written thesis, poster, and presentation live in the project root and `drafts/`:
`Thesis Final Draft.pdf` / `.docx`, `Poster Final Draft.pdf`, and the
`thesis_presentation_final_draft.pptx` deck.
