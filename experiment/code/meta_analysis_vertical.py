"""
meta_analysis.py — Combined visualization and summary table for telephone game thesis.

Generates:
  TELEPHONE outputs (telephone/ subdirectory):
    1. CLIP vs ArcFace scatter (aggregate, plain + ID-highlighted)
    2. CLIP vs ArcFace scatter grid by iteration (15 panels, plain + highlighted)
    3. CLIP vs ArcFace scatter grid by seed (14 panels, plain + highlighted)
    4. Telephone vs one-shot comparison scatter
    5. Sentiment trajectory per seed
    6. Per-seed overview CSV table
    7. Per-seed animated videos (chain-by-chain)
    8. Aggregate animated video (iteration-by-iteration, all seeds)
    9. Identity match trajectory line graph

  DISTRIBUTIONS outputs (distributions/ subdirectory):
    10. CLIP vs ArcFace scatter (aggregate, plain + ID-highlighted)
    11. CLIP vs ArcFace scatter grid by seed (per-seed panels)

Usage:
  python meta_analysis.py

Dependencies:
  pip install pandas matplotlib scipy numpy
  For video: ffmpeg (falls back to GIF via pillow if unavailable)
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import stats


IDENTITY_THRESHOLD = 0.3
FIGURE_DPI = 200
VIDEO_FPS = 5
CHAIN_PAUSE_FRAMES = 3

# Seed folder name → display name mapping (for distributions dataset)
SEED_DISPLAY_NAMES = {
    "test 1":  "JFK",
    "test 2":  "Donald Trump",
    "test 3":  "Gandhi",
    "test 4":  "Fidel Castro",
    "test 5":  "MLK",
    "test 6":  "Obama",
    "test 7":  "Bill Clinton",
    "test 8":  "Queen Elizabeth",
    "test 9":  "Joseph Stalin",
    "test 10": "Kim Jong Un",
    "test 11": "Adolf Hitler",
    "test 12": "Vladimir Putin",
    "test 13": "Xi Jinping",
    "test 14": "George W. Bush",
    "test 15": "Benjamin Netanyahu",
}


# ================================
# DATA LOADING
# ================================

def load_telephone_data(clip_dir: Path, face_dir: Path):
    clip_csv = clip_dir / "data" / "clip_scores.csv"
    face_csv = face_dir / "data" / "face_scores.csv"
    if not clip_csv.exists():
        print(f"ERROR: {clip_csv} not found"); sys.exit(1)
    if not face_csv.exists():
        print(f"ERROR: {face_csv} not found"); sys.exit(1)
    clip_df = pd.read_csv(clip_csv)
    face_df = pd.read_csv(face_csv)
    merged = clip_df.merge(
        face_df[["seed_name", "chain_num", "iteration", "face_score", "face_detected", "gen_status"]],
        on=["seed_name", "chain_num", "iteration"], how="inner"
    )
    return clip_df, face_df, merged


def load_validation_data(val_dir: Path):
    images_dir = val_dir / "images"
    if not images_dir.exists():
        print(f"WARNING: {images_dir} not found, skipping validation data")
        return None
    rows = []
    for seed_dir in sorted(images_dir.iterdir()):
        if not seed_dir.is_dir():
            continue
        log_path = seed_dir / "log.json"
        if not log_path.exists():
            continue
        log = json.loads(log_path.read_text(encoding="utf-8"))
        seed_name = seed_dir.name
        analysis = log.get("analysis", {})
        clip_vs = analysis.get("clip", {}).get("vs_seed", {}).get("per_image", [])
        face_vs_single = analysis.get("arcface", {}).get("vs_seed_single", {}).get("per_image", [])
        face_vs_multi = []
        if not face_vs_single:
            multi = analysis.get("arcface", {}).get("vs_seed_multi", {})
            for img_r in multi.get("per_image", []):
                matches = img_r.get("matches", [])
                valid = [m["score"] for m in matches if m.get("score") is not None]
                face_vs_multi.append(float(np.mean(valid)) if valid else None)
        face_vs = face_vs_single if face_vs_single else face_vs_multi
        n = min(len(clip_vs), len(face_vs)) if face_vs else len(clip_vs)
        for i in range(n):
            c = clip_vs[i] if i < len(clip_vs) else None
            f = face_vs[i] if i < len(face_vs) else None
            rows.append({"seed_name": seed_name, "clip_score": c, "face_score": f, "source": "validation"})
    if not rows:
        print("WARNING: No validation data found")
        return None
    return pd.DataFrame(rows)


def load_semantic_data(sem_dir: Path):
    result = {}
    for name in ["iteration_scores", "epoch_scores", "refusal_events",
                  "insertion_events", "vocabulary_divergence"]:
        csv_path = sem_dir / "data" / f"{name}.csv"
        if csv_path.exists():
            result[name] = pd.read_csv(csv_path)
    return result


# ================================
# SCATTER HELPER
# ================================

def _scatter_with_identity(ax, clip_scores, face_scores, threshold=IDENTITY_THRESHOLD,
                            bg_clip=None, bg_face=None):
    if bg_clip is not None and bg_face is not None:
        ax.scatter(bg_clip, bg_face, c="#e0e0e0", alpha=0.15, s=8, edgecolors="none")
    mask_id = face_scores > threshold
    mask_no = ~mask_id
    ax.scatter(clip_scores[mask_no], face_scores[mask_no],
               c="#999999", alpha=0.3, s=18, edgecolors="none")
    ax.scatter(clip_scores[mask_id], face_scores[mask_id],
               c="#e63946", alpha=0.7, s=25, edgecolors="white", linewidths=0.3)


# ================================
# FIGURE 0: ID MATCH TRAJECTORY
# ================================

def fig_id_match_trajectory(merged, out_dir):
    """Plot total number of successful identity matches per iteration (aggregate only)."""
    plot_df = merged.dropna(subset=["clip_score", "face_score"])

    fig, ax = plt.subplots(figsize=(12, 7))

    # Aggregate line
    agg = plot_df.groupby("iteration").apply(
        lambda g: (g["face_score"] > IDENTITY_THRESHOLD).sum()
    )
    ax.plot(agg.index, agg.values, color="black", linewidth=2.5, alpha=0.9)

    # Faint vertical lines at each iteration
    for it in range(1, 16):
        ax.axvline(it, color="#dddddd", linewidth=0.5, zorder=0)

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Successful Identity Matches", fontsize=12)
    ax.set_title("Identity Preservation Across Iterations", fontsize=13, fontweight="bold")
    ax.set_xlim(1, 15)
    ax.set_ylim(0, None)
    ax.set_xticks([5, 10, 15])
    ax.set_xticklabels(["5", "10", "15"])

    plt.tight_layout()
    plt.savefig(out_dir / "id_match_trajectory.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: id_match_trajectory.png")


# ================================
# FIGURE 1: AGGREGATE SCATTER
# ================================

def fig_aggregate_scatter(merged, out_dir):
    plot_df = merged.dropna(subset=["clip_score", "face_score"])
    r, p = stats.pearsonr(plot_df["clip_score"], plot_df["face_score"])

    # Plain version (standard title, no highlighting)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(plot_df["clip_score"], plot_df["face_score"],
               c="#333333", alpha=0.35, s=20, edgecolors="none")
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.text(0.05, 0.95, f"Pearson r = {r:.3f}\np < 0.001",
            transform=ax.transAxes, fontsize=11, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    ax.set_xlabel("CLIP Cosine Similarity (Compositional)", fontsize=12)
    ax.set_ylabel("ArcFace Cosine Similarity (Face Identity)", fontsize=12)
    ax.set_title("Compositional Preservation vs. Face Identity Preservation",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0.2, 1.0); ax.set_ylim(-0.3, 1.0)
    plt.tight_layout()
    plt.savefig(out_dir / "clip_vs_arcface_aggregate.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: clip_vs_arcface_aggregate.png")

    # Highlighted version (identity matches in red, threshold line)
    fig, ax = plt.subplots(figsize=(10, 8))
    _scatter_with_identity(ax, plot_df["clip_score"].values, plot_df["face_score"].values)
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.axhline(IDENTITY_THRESHOLD, color="#e63946", linestyle="--", linewidth=0.6, alpha=0.4)
    n_above = (plot_df["face_score"] > IDENTITY_THRESHOLD).sum()
    ax.text(0.05, 0.95,
            f"Pearson r = {r:.3f}\np < 0.001\n"
            f"Identity matches: {n_above}/{len(plot_df)} ({100*n_above/len(plot_df):.0f}%)",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    ax.set_xlabel("CLIP Cosine Similarity (Compositional)", fontsize=12)
    ax.set_ylabel("ArcFace Cosine Similarity (Face Identity)", fontsize=12)
    ax.set_title("Compositional Preservation vs. Face Identity Preservation (Successful IDs Highlighted)",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0.2, 1.0); ax.set_ylim(-0.3, 1.0)
    plt.tight_layout()
    plt.savefig(out_dir / "clip_vs_arcface_aggregate_highlighted.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: clip_vs_arcface_aggregate_highlighted.png")


# ================================
# FIGURE 2: BY ITERATION
# ================================

def fig_by_iteration(merged, out_dir):
    plot_df = merged.dropna(subset=["clip_score", "face_score"])

    for mode in ["plain", "highlighted"]:
        fig, axes = plt.subplots(5, 3, figsize=(13, 22))
        suffix = "" if mode == "plain" else "_highlighted"
        title_extra = "" if mode == "plain" else " (Successful IDs Highlighted)"
        fig.suptitle(f"CLIP vs ArcFace by Iteration{title_extra}", fontsize=16, fontweight="bold", y=0.98)

        for idx, it in enumerate(range(1, 16)):
            ax = axes[idx // 3][idx % 3]
            it_df = plot_df[plot_df["iteration"] == it]

            # Background
            ax.scatter(plot_df["clip_score"], plot_df["face_score"],
                       c="#e0e0e0", alpha=0.15, s=8, edgecolors="none")

            if mode == "highlighted":
                _scatter_with_identity(ax, it_df["clip_score"].values, it_df["face_score"].values)
                n_above = (it_df["face_score"] > IDENTITY_THRESHOLD).sum()
                ax.set_title(f"Iter {it}  ({n_above} ID matches)", fontsize=9, fontweight="bold")
            else:
                ax.scatter(it_df["clip_score"], it_df["face_score"],
                           c="#333333", alpha=0.5, s=15, edgecolors="none")
                ax.set_title(f"Iteration {it}", fontsize=9, fontweight="bold")

            ax.set_xlim(0.2, 1.0); ax.set_ylim(-0.3, 1.0)
            if idx % 3 == 0: ax.set_ylabel("ArcFace", fontsize=9)
            if idx // 3 == 4: ax.set_xlabel("CLIP", fontsize=9)
            ax.tick_params(labelsize=7)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(out_dir / f"clip_vs_arcface_by_iteration{suffix}.png", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close()
        print(f"  Saved: clip_vs_arcface_by_iteration{suffix}.png")


# ================================
# FIGURE 3: BY SEED
# ================================

def fig_by_seed(merged, out_dir):
    plot_df = merged.dropna(subset=["clip_score", "face_score"])
    seeds_ordered = plot_df.groupby("seed_name")["face_score"].mean().sort_values(ascending=False).index.tolist()
    n_seeds = len(seeds_ordered)
    ncols, nrows = 3, 5

    for mode in ["plain", "highlighted"]:
        fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.3 * nrows))
        suffix = "" if mode == "plain" else "_highlighted"
        title_extra = "" if mode == "plain" else " (Successful IDs Highlighted)"
        fig.suptitle(f"CLIP vs ArcFace by Seed{title_extra}", fontsize=16, fontweight="bold", y=0.98)
        if nrows == 1: axes = [axes]

        for idx, seed in enumerate(seeds_ordered):
            ax = axes[idx // ncols][idx % ncols]
            s_df = plot_df[plot_df["seed_name"] == seed]

            # Background
            ax.scatter(plot_df["clip_score"], plot_df["face_score"],
                       c="#e0e0e0", alpha=0.15, s=8, edgecolors="none")

            if mode == "highlighted":
                _scatter_with_identity(ax, s_df["clip_score"].values, s_df["face_score"].values)
                n_above = (s_df["face_score"] > IDENTITY_THRESHOLD).sum()
                ax.set_title(f"{seed}  ({n_above} ID)", fontsize=9, fontweight="bold")
            else:
                ax.scatter(s_df["clip_score"], s_df["face_score"],
                           c="#333333", alpha=0.5, s=15, edgecolors="none")
                ax.set_title(seed, fontsize=9, fontweight="bold")

            ax.set_xlim(0.2, 1.0); ax.set_ylim(-0.3, 1.0)
            if idx % ncols == 0: ax.set_ylabel("ArcFace", fontsize=9)
            if idx // ncols == nrows - 1: ax.set_xlabel("CLIP", fontsize=9)
            ax.tick_params(labelsize=7)

        for idx in range(n_seeds, nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(out_dir / f"clip_vs_arcface_by_seed{suffix}.png", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close()
        print(f"  Saved: clip_vs_arcface_by_seed{suffix}.png")


# ================================
# FIGURE 4: VALIDATION COMPARISON
# ================================

def fig_validation_comparison(merged, val_df, out_dir):
    tel_df = merged.dropna(subset=["clip_score", "face_score"])
    val_plot = val_df.dropna(subset=["clip_score", "face_score"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("Telephone (Iterative) vs. Validation (One-Shot)", fontsize=14, fontweight="bold", y=0.98)
    r_tel, _ = stats.pearsonr(tel_df["clip_score"], tel_df["face_score"])
    n_tel_id = (tel_df["face_score"] > IDENTITY_THRESHOLD).sum()
    _scatter_with_identity(ax1, tel_df["clip_score"].values, tel_df["face_score"].values)
    ax1.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=0.8, alpha=0.4)
    ax1.text(0.05, 0.95, f"r = {r_tel:.3f}\nn = {len(tel_df)}\n{n_tel_id} ID matches ({100*n_tel_id/len(tel_df):.0f}%)",
             transform=ax1.transAxes, fontsize=11, va="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    ax1.set_xlabel("CLIP", fontsize=11); ax1.set_ylabel("ArcFace", fontsize=11)
    ax1.set_title("Telephone (Iterative Drift)", fontsize=12)
    ax1.set_xlim(0.2, 1.0); ax1.set_ylim(-0.3, 1.0)
    r_val, _ = stats.pearsonr(val_plot["clip_score"], val_plot["face_score"])
    n_val_id = (val_plot["face_score"] > IDENTITY_THRESHOLD).sum()
    _scatter_with_identity(ax2, val_plot["clip_score"].values, val_plot["face_score"].values)
    ax2.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=0.8, alpha=0.4)
    ax2.text(0.05, 0.95, f"r = {r_val:.3f}\nn = {len(val_plot)}\n{n_val_id} ID matches ({100*n_val_id/len(val_plot):.0f}%)",
             transform=ax2.transAxes, fontsize=11, va="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    ax2.set_xlabel("CLIP", fontsize=11); ax2.set_ylabel("ArcFace", fontsize=11)
    ax2.set_title("Validation (One-Shot from Medoid Caption)", fontsize=12)
    ax2.set_xlim(0.2, 1.0); ax2.set_ylim(-0.3, 1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_dir / "telephone_vs_validation.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: telephone_vs_validation.png")


# ================================
# FIGURE 5: SENTIMENT TRAJECTORY
# ================================

def fig_sentiment_trajectory(sem_data, out_dir):
    if "iteration_scores" not in sem_data:
        print("  Skipping sentiment trajectory (no iteration_scores.csv)")
        return
    iter_df = sem_data["iteration_scores"]
    if "sentiment_score" not in iter_df.columns:
        print("  Skipping sentiment trajectory (no sentiment_score column)")
        return
    seeds = sorted(iter_df["seed_name"].unique())
    n_seeds = len(seeds)
    ncols, nrows = 3, 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.5 * nrows))
    fig.suptitle("Sentiment Trajectory by Seed", fontsize=16, fontweight="bold", y=0.98)
    if nrows == 1: axes = [axes]
    for idx, seed in enumerate(seeds):
        ax = axes[idx // ncols][idx % ncols]
        s_df = iter_df[iter_df["seed_name"] == seed]
        for chain, c_df in s_df.groupby("chain_num"):
            c_df = c_df.sort_values("iteration")
            ax.plot(c_df["iteration"], c_df["sentiment_score"], alpha=0.3, linewidth=0.8, color="#666666")
        mean_sent = s_df.groupby("iteration")["sentiment_score"].mean()
        ax.plot(mean_sent.index, mean_sent.values, color="#2563eb", linewidth=2, alpha=0.9)
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_xlim(1, 15); ax.set_ylim(-1, 1)
        ax.set_title(seed, fontsize=9, fontweight="bold")
        if idx % ncols == 0: ax.set_ylabel("Sentiment", fontsize=8)
        if idx // ncols == nrows - 1: ax.set_xlabel("Iteration", fontsize=8)
        ax.tick_params(labelsize=7)
    for idx in range(n_seeds, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_dir / "sentiment_trajectory.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: sentiment_trajectory.png")


# ================================
# TABLE: PER-SEED OVERVIEW
# ================================

def build_overview_table(clip_df, face_df, merged, sem_data, out_dir):
    rows = []
    seeds = sorted(clip_df["seed_name"].unique())
    iter_scores = sem_data.get("iteration_scores", pd.DataFrame())
    has_semantic = len(iter_scores) > 0

    for seed in seeds:
        s_clip = clip_df[clip_df["seed_name"] == seed]
        s_face = face_df[face_df["seed_name"] == seed]
        s_merged = merged[merged["seed_name"] == seed].dropna(subset=["clip_score", "face_score"])

        max_possible = 75
        completed = len(s_clip)
        completion_str = f"{completed}/{max_possible} ({100 * completed / max_possible:.0f}%)"

        chain_lengths = s_clip.groupby("chain_num")["iteration"].max()
        blocks = sum(1 for v in chain_lengths if v < 15)

        id_hits = (s_face["face_score"] > IDENTITY_THRESHOLD).sum()
        total_scored = s_face["face_score"].notna().sum()
        id_hits_str = f"{id_hits}/{total_scored}"

        above = s_face[s_face["face_score"] > IDENTITY_THRESHOLD]
        if len(above) > 0:
            last_hit = above["iteration"].max()
            loss_str = "None (preserved)" if len(above[above["iteration"] == 15]) > 0 else f"After iter {last_hit}"
        else:
            loss_str = "Iter 1 (never matched)"

        r2_str = "N/A"
        if len(s_merged) >= 3:
            r, _ = stats.pearsonr(s_merged["clip_score"], s_merged["face_score"])
            r2_str = f"{r**2:.3f}"

        refusal_str = "N/A"
        drift_str = "N/A"
        if has_semantic and "refusal_detected" in iter_scores.columns:
            s_sem = iter_scores[iter_scores["seed_name"] == seed]
            if len(s_sem) > 0:
                refusal_str = f"{s_sem['refusal_detected'].mean():.3f}"
                if "absolute_drift" in s_sem.columns:
                    max_iter = s_sem["iteration"].max()
                    final = s_sem[s_sem["iteration"] == max_iter]
                    if len(final) > 0:
                        drift_str = f"{final['absolute_drift'].mean():.3f}"

        rows.append({
            "Seed": seed, "Completion": completion_str, "Blocks": blocks,
            "Identity Hits": id_hits_str, "Identity Loss Point": loss_str,
            "CLIP/ArcFace r2": r2_str, "Refusal Rate": refusal_str,
            "Semantic Drift (final)": drift_str,
        })

    df = pd.DataFrame(rows)
    path = out_dir / "seed_overview_table.csv"
    df.to_csv(path, index=False)
    print(f"  Saved: {path.name}")
    print()
    print(df.to_string(index=False))
    return df


# ================================
# VIDEOS: PER-SEED CHAIN ANIMATION
# ================================

def generate_seed_videos(merged, out_dir):
    video_dir = out_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    plot_df = merged.dropna(subset=["clip_score", "face_score"])
    seeds = sorted(plot_df["seed_name"].unique())
    print(f"  Generating {len(seeds)} seed videos...")

    FINAL_HOLD_SECONDS = 3
    final_hold_frames = FINAL_HOLD_SECONDS * VIDEO_FPS

    for seed in seeds:
        s_df = plot_df[plot_df["seed_name"] == seed].sort_values(["chain_num", "iteration"])
        chains = sorted(s_df["chain_num"].unique())

        # Build frame data structure
        # Each frame stores:
        #   - "past": list of (clip, face) from completed chains (always grey)
        #   - "current": list of (clip, face, is_id) from active chain so far
        #   - "chain_label": current chain number
        #   - "is_final": whether this is the final reveal frame

        frame_list = []
        past_points = []  # all points from completed chains

        for chain in chains:
            c_df = s_df[s_df["chain_num"] == chain].sort_values("iteration")
            current_chain = []

            for _, row in c_df.iterrows():
                is_id = row["face_score"] > IDENTITY_THRESHOLD
                current_chain.append((row["clip_score"], row["face_score"], is_id))
                frame_list.append({
                    "past": list(past_points),
                    "current": list(current_chain),
                    "chain_label": chain,
                    "is_final": False,
                })

            # Pause frames at end of chain
            for _ in range(CHAIN_PAUSE_FRAMES):
                frame_list.append({
                    "past": list(past_points),
                    "current": list(current_chain),
                    "chain_label": chain,
                    "is_final": False,
                })

            # Move current chain points to past (muted)
            past_points.extend([(c, f) for c, f, _ in current_chain])

        # Final reveal frames: all points, identity highlighted, held for 3 seconds
        all_points = []
        for _, row in s_df.iterrows():
            all_points.append((row["clip_score"], row["face_score"],
                               row["face_score"] > IDENTITY_THRESHOLD))

        for _ in range(final_hold_frames):
            frame_list.append({
                "past": [],
                "current": all_points,
                "chain_label": "ALL",
                "is_final": True,
            })

        if not frame_list:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))

        def animate(frame_idx):
            ax.clear()

            # Background: all data in light grey
            ax.scatter(plot_df["clip_score"], plot_df["face_score"],
                       c="#e8e8e8", alpha=0.12, s=8, edgecolors="none")

            frame = frame_list[frame_idx]

            # Past chains: all grey, muted
            if frame["past"]:
                ax.scatter([p[0] for p in frame["past"]],
                           [p[1] for p in frame["past"]],
                           c="#bbbbbb", alpha=0.4, s=20, edgecolors="none")

            # Current chain or final reveal
            if frame["current"]:
                if frame["is_final"]:
                    # Final: highlight identity matches
                    no_id = [(c, f) for c, f, i in frame["current"] if not i]
                    yes_id = [(c, f) for c, f, i in frame["current"] if i]
                    if no_id:
                        ax.scatter([x[0] for x in no_id], [x[1] for x in no_id],
                                   c="#666666", alpha=0.5, s=25, edgecolors="none")
                    if yes_id:
                        ax.scatter([x[0] for x in yes_id], [x[1] for x in yes_id],
                                   c="#e63946", alpha=0.9, s=45, edgecolors="white",
                                   linewidths=0.6, zorder=5)
                else:
                    # Active chain: highlight ID matches in red
                    no_id = [(c, f) for c, f, i in frame["current"] if not i]
                    yes_id = [(c, f) for c, f, i in frame["current"] if i]
                    if no_id:
                        ax.scatter([x[0] for x in no_id], [x[1] for x in no_id],
                                   c="#333333", alpha=0.7, s=30, edgecolors="none")
                    if yes_id:
                        ax.scatter([x[0] for x in yes_id], [x[1] for x in yes_id],
                                   c="#e63946", alpha=0.85, s=40, edgecolors="white",
                                   linewidths=0.5, zorder=5)

            ax.set_xlim(0.2, 1.0); ax.set_ylim(-0.3, 1.0)
            ax.set_xlabel("CLIP", fontsize=10)
            ax.set_ylabel("ArcFace", fontsize=10)
            ax.set_title(seed, fontsize=11, fontweight="bold")

            label = "All Chains" if frame["is_final"] else f"Chain {frame['chain_label']}"
            ax.text(0.95, 0.95, label,
                    transform=ax.transAxes, fontsize=10, va="top", ha="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        anim = animation.FuncAnimation(fig, animate, frames=len(frame_list),
                                        interval=1000 // VIDEO_FPS, blit=False)

        video_path = video_dir / f"{seed.replace(' ', '_').lower()}.mp4"
        try:
            anim.save(str(video_path), writer="ffmpeg", fps=VIDEO_FPS, dpi=150)
            print(f"    {seed}: {video_path.name} ({len(frame_list)} frames)")
        except Exception:
            gif_path = video_dir / f"{seed.replace(' ', '_').lower()}.gif"
            try:
                anim.save(str(gif_path), writer="pillow", fps=VIDEO_FPS, dpi=100)
                print(f"    {seed}: {gif_path.name} (GIF fallback)")
            except Exception as e2:
                print(f"    {seed}: FAILED ({e2})")
        plt.close(fig)


# ================================
# VIDEO: AGGREGATE ITERATION-BY-ITERATION
# ================================

def generate_iteration_video(merged, out_dir):
    """Animate all seeds together, one iteration at a time.
    Each frame shows all points for that iteration. Previous iterations
    fade to lighter grey so you can see the progression.
    Final frame shows all data at once, held for 3 seconds."""
    video_dir = out_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    plot_df = merged.dropna(subset=["clip_score", "face_score"])
    max_iter = int(plot_df["iteration"].max())

    HOLD_SECONDS = 3

    # Build frames: one per iteration, then "all" frame held for 3 seconds
    # frame_list entries: int = iteration number, "all" = final reveal
    frame_list = list(range(1, max_iter + 1))
    frame_list.extend(["all"] * HOLD_SECONDS)

    fig, ax = plt.subplots(figsize=(10, 8))

    def animate(frame_idx):
        ax.clear()
        current = frame_list[frame_idx]

        if current == "all":
            # Final frame: all data plotted at once
            mask_id = plot_df["face_score"] > IDENTITY_THRESHOLD
            mask_no = ~mask_id
            ax.scatter(plot_df.loc[mask_no, "clip_score"], plot_df.loc[mask_no, "face_score"],
                       c="#999999", alpha=0.3, s=18, edgecolors="none")
            ax.scatter(plot_df.loc[mask_id, "clip_score"], plot_df.loc[mask_id, "face_score"],
                       c="#e63946", alpha=0.7, s=25, edgecolors="white", linewidths=0.3, zorder=5)
            n_id = mask_id.sum()
            ax.set_title("CLIP vs ArcFace — All Iterations", fontsize=13, fontweight="bold")
            ax.text(0.95, 0.95, f"All iterations\n{n_id}/{len(plot_df)} ID matches ({100*n_id/len(plot_df):.0f}%)",
                    transform=ax.transAxes, fontsize=11, va="top", ha="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
        else:
            current_iter = current
            # Draw previous iterations with increasing transparency
            for prev_iter in range(1, current_iter):
                prev_df = plot_df[plot_df["iteration"] == prev_iter]
                age = current_iter - prev_iter
                alpha = max(0.03, 0.25 - age * 0.015)
                ax.scatter(prev_df["clip_score"], prev_df["face_score"],
                           c="#cccccc", alpha=alpha, s=12, edgecolors="none")

            # Draw current iteration prominently
            curr_df = plot_df[plot_df["iteration"] == current_iter]
            mask_id = curr_df["face_score"] > IDENTITY_THRESHOLD
            mask_no = ~mask_id
            ax.scatter(curr_df.loc[mask_no, "clip_score"], curr_df.loc[mask_no, "face_score"],
                       c="#333333", alpha=0.7, s=25, edgecolors="none")
            ax.scatter(curr_df.loc[mask_id, "clip_score"], curr_df.loc[mask_id, "face_score"],
                       c="#e63946", alpha=0.9, s=35, edgecolors="white", linewidths=0.4, zorder=5)
            n_id = mask_id.sum()
            ax.set_title(f"CLIP vs ArcFace — Iteration {current_iter}",
                         fontsize=13, fontweight="bold")
            n_total = len(curr_df)
            ax.text(0.95, 0.95, f"{n_id}/{n_total} ID matches ({100*n_id/n_total:.0f}%)" if n_total > 0 else "n = 0",
                    transform=ax.transAxes, fontsize=11, va="top", ha="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

        ax.set_xlim(0.2, 1.0)
        ax.set_ylim(-0.3, 1.0)
        ax.set_xlabel("CLIP Cosine Similarity (Compositional)", fontsize=11)
        ax.set_ylabel("ArcFace Cosine Similarity (Face Identity)", fontsize=11)

    anim = animation.FuncAnimation(fig, animate, frames=len(frame_list),
                                    interval=1000, blit=False)

    video_path = video_dir / "aggregate_by_iteration.mp4"
    try:
        anim.save(str(video_path), writer="ffmpeg", fps=1, dpi=150)
        print(f"  Saved: {video_path.name} ({len(frame_list)} frames)")
    except Exception:
        gif_path = video_dir / "aggregate_by_iteration.gif"
        try:
            anim.save(str(gif_path), writer="pillow", fps=1, dpi=100)
            print(f"  Saved: {gif_path.name} (GIF fallback)")
        except Exception as e2:
            print(f"  FAILED: aggregate iteration video ({e2})")
    plt.close(fig)


# ================================
# VIDEO: BASELINE vs TELEPHONE PER SEED
# ================================

def generate_comparison_video(merged, val_df, out_dir):
    """Animate seed-by-seed comparison: baseline distribution then telephone data.
    Each seed gets two frames (1.5s each): one for baseline, one for telephone.
    All data from other seeds shown as light background."""
    video_dir = out_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    tel_df = merged.dropna(subset=["clip_score", "face_score"])
    dist_df = val_df.dropna(subset=["clip_score", "face_score"]).copy()
    dist_df["display_name"] = dist_df["seed_name"].map(SEED_DISPLAY_NAMES).fillna(dist_df["seed_name"])

    # Get seeds present in both datasets, ordered by telephone face score
    tel_seeds = set(tel_df["seed_name"].unique())
    dist_display_names = set(dist_df["display_name"].unique())
    # Match telephone seed_name to distribution display_name
    common_seeds = sorted(
        [s for s in tel_df["seed_name"].unique() if s in dist_display_names],
        key=lambda s: tel_df[tel_df["seed_name"] == s]["face_score"].mean(),
        reverse=True
    )

    if not common_seeds:
        print("  Skipping comparison video (no overlapping seeds)")
        return

    HOLD_SECONDS = 3
    FPS = 2  # 1.5s per frame = not a clean FPS, so use 2 FPS with frame duplication

    # Build frame list: (seed_name, "baseline"|"telephone") pairs
    # Each held for 1.5s. At 2 FPS, 1.5s = 3 frames
    FRAMES_PER_SLIDE = 3
    frame_list = []
    for seed in common_seeds:
        for _ in range(FRAMES_PER_SLIDE):
            frame_list.append((seed, "baseline"))
        for _ in range(FRAMES_PER_SLIDE):
            frame_list.append((seed, "telephone"))

    # Hold on last frame for 3 seconds
    last = frame_list[-1]
    frame_list.extend([last] * (HOLD_SECONDS * FPS))

    fig, ax = plt.subplots(figsize=(10, 8))

    def animate(frame_idx):
        ax.clear()
        seed_name, source = frame_list[frame_idx]

        # Background: all data from both datasets in light grey
        ax.scatter(tel_df["clip_score"], tel_df["face_score"],
                   c="#e8e8e8", alpha=0.08, s=8, edgecolors="none")
        ax.scatter(dist_df["clip_score"], dist_df["face_score"],
                   c="#e8e8e8", alpha=0.08, s=8, edgecolors="none")

        if source == "baseline":
            s_df = dist_df[dist_df["display_name"] == seed_name]
            source_label = "Baseline Distribution"
            color_main = "#333333"
            color_id = "#e63946"
        else:
            s_df = tel_df[tel_df["seed_name"] == seed_name]
            source_label = "Telephone (Iterative)"
            color_main = "#333333"
            color_id = "#e63946"

        if len(s_df) > 0:
            mask_id = s_df["face_score"] > IDENTITY_THRESHOLD
            mask_no = ~mask_id
            ax.scatter(s_df.loc[mask_no, "clip_score"], s_df.loc[mask_no, "face_score"],
                       c=color_main, alpha=0.5, s=25, edgecolors="none")
            ax.scatter(s_df.loc[mask_id, "clip_score"], s_df.loc[mask_id, "face_score"],
                       c=color_id, alpha=0.9, s=35, edgecolors="white", linewidths=0.4, zorder=5)
            n_id = mask_id.sum()
            n_total = len(s_df)
        else:
            n_id = 0
            n_total = 0

        ax.set_xlim(0.2, 1.0)
        ax.set_ylim(-0.3, 1.0)
        ax.set_xlabel("CLIP Cosine Similarity (Compositional)", fontsize=11)
        ax.set_ylabel("ArcFace Cosine Similarity (Face Identity)", fontsize=11)
        ax.set_title(f"{seed_name} — {source_label}", fontsize=13, fontweight="bold")
        ax.text(0.95, 0.95, f"n = {n_total}\n{n_id} ID matches ({100*n_id/n_total:.0f}%)" if n_total > 0 else "n = 0",
                transform=ax.transAxes, fontsize=11, va="top", ha="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    anim = animation.FuncAnimation(fig, animate, frames=len(frame_list),
                                    interval=500, blit=False)  # 500ms at 2 FPS = frames play correctly

    video_path = video_dir / "baseline_vs_telephone_by_seed.mp4"
    try:
        anim.save(str(video_path), writer="ffmpeg", fps=FPS, dpi=150)
        print(f"  Saved: {video_path.name} ({len(common_seeds)} seeds, {len(frame_list)} frames)")
    except Exception:
        gif_path = video_dir / "baseline_vs_telephone_by_seed.gif"
        try:
            anim.save(str(gif_path), writer="pillow", fps=FPS, dpi=100)
            print(f"  Saved: {gif_path.name} (GIF fallback)")
        except Exception as e2:
            print(f"  FAILED: comparison video ({e2})")
    plt.close(fig)


# ================================
# DISTRIBUTIONS: AGGREGATE SCATTER
# ================================

def fig_dist_aggregate_scatter(val_df, out_dir):
    """Aggregate CLIP vs ArcFace scatter for the baseline distributions dataset."""
    plot_df = val_df.dropna(subset=["clip_score", "face_score"])
    if len(plot_df) < 3:
        print("  Skipping distributions aggregate scatter (insufficient data)")
        return

    r, p = stats.pearsonr(plot_df["clip_score"], plot_df["face_score"])

    # Plain version
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(plot_df["clip_score"], plot_df["face_score"],
               c="#333333", alpha=0.35, s=20, edgecolors="none")
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.text(0.05, 0.95, f"Pearson r = {r:.3f}\nn = {len(plot_df)}",
            transform=ax.transAxes, fontsize=11, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    ax.set_xlabel("CLIP Cosine Similarity (Compositional)", fontsize=12)
    ax.set_ylabel("ArcFace Cosine Similarity (Face Identity)", fontsize=12)
    ax.set_title("Baseline Distributions: Compositional vs. Face Identity Preservation",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0.2, 1.0); ax.set_ylim(-0.3, 1.0)
    plt.tight_layout()
    plt.savefig(out_dir / "dist_clip_vs_arcface_aggregate.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: dist_clip_vs_arcface_aggregate.png")

    # Highlighted version
    fig, ax = plt.subplots(figsize=(10, 8))
    _scatter_with_identity(ax, plot_df["clip_score"].values, plot_df["face_score"].values)
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.axhline(IDENTITY_THRESHOLD, color="#e63946", linestyle="--", linewidth=0.6, alpha=0.4)
    n_above = (plot_df["face_score"] > IDENTITY_THRESHOLD).sum()
    ax.text(0.05, 0.95,
            f"Pearson r = {r:.3f}\nn = {len(plot_df)}\n"
            f"Identity matches: {n_above}/{len(plot_df)} ({100*n_above/len(plot_df):.0f}%)",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    ax.set_xlabel("CLIP Cosine Similarity (Compositional)", fontsize=12)
    ax.set_ylabel("ArcFace Cosine Similarity (Face Identity)", fontsize=12)
    ax.set_title("Baseline Distributions: Compositional vs. Face Identity (Successful IDs Highlighted)",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0.2, 1.0); ax.set_ylim(-0.3, 1.0)
    plt.tight_layout()
    plt.savefig(out_dir / "dist_clip_vs_arcface_aggregate_highlighted.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: dist_clip_vs_arcface_aggregate_highlighted.png")


# ================================
# DISTRIBUTIONS: BY SEED SCATTER
# ================================

def fig_dist_by_seed(val_df, out_dir):
    """Per-seed CLIP vs ArcFace scatter grid for the baseline distributions dataset."""
    plot_df = val_df.dropna(subset=["clip_score", "face_score"])
    if len(plot_df) < 3:
        print("  Skipping distributions by-seed scatter (insufficient data)")
        return

    # Map folder names to display names
    plot_df = plot_df.copy()
    plot_df["display_name"] = plot_df["seed_name"].map(SEED_DISPLAY_NAMES).fillna(plot_df["seed_name"])

    seeds_ordered = plot_df.groupby("display_name")["face_score"].mean().sort_values(ascending=False).index.tolist()
    n_seeds = len(seeds_ordered)
    ncols, nrows = 3, 5

    for mode in ["plain", "highlighted"]:
        fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.3 * nrows))
        suffix = "" if mode == "plain" else "_highlighted"
        title_extra = "" if mode == "plain" else " (Successful IDs Highlighted)"
        fig.suptitle(f"Baseline Distributions: CLIP vs ArcFace by Seed{title_extra}",
                     fontsize=16, fontweight="bold", y=0.98)
        if nrows == 1: axes = [axes]

        for idx, seed in enumerate(seeds_ordered):
            ax = axes[idx // ncols][idx % ncols]
            s_df = plot_df[plot_df["display_name"] == seed]

            # Background: all data in light grey
            ax.scatter(plot_df["clip_score"], plot_df["face_score"],
                       c="#e0e0e0", alpha=0.15, s=8, edgecolors="none")

            if mode == "highlighted":
                _scatter_with_identity(ax, s_df["clip_score"].values, s_df["face_score"].values)
                n_above = (s_df["face_score"] > IDENTITY_THRESHOLD).sum()
                ax.set_title(f"{seed}  ({n_above} ID)", fontsize=9, fontweight="bold")
            else:
                ax.scatter(s_df["clip_score"], s_df["face_score"],
                           c="#333333", alpha=0.5, s=15, edgecolors="none")
                ax.set_title(seed, fontsize=9, fontweight="bold")

            ax.set_xlim(0.2, 1.0); ax.set_ylim(-0.3, 1.0)
            if idx % ncols == 0: ax.set_ylabel("ArcFace", fontsize=9)
            if idx // ncols == nrows - 1: ax.set_xlabel("CLIP", fontsize=9)
            ax.tick_params(labelsize=7)

        for idx in range(n_seeds, nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(out_dir / f"dist_clip_vs_arcface_by_seed{suffix}.png", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close()
        print(f"  Saved: dist_clip_vs_arcface_by_seed{suffix}.png")


# ================================
# MAIN
# ================================

def main():
    print()
    print("=" * 60)
    print("  meta_analysis.py")
    print("  Combined visualization + summary table + videos")
    print("=" * 60)
    print()

    clip_dir = Path(input("  CLIP analysis output dir: ").strip())
    face_dir = Path(input("  Face analysis output dir: ").strip())
    sem_dir  = Path(input("  Semantic analysis output dir: ").strip())
    val_dir  = Path(input("  Validation dataset output dir (seed_sampler): ").strip())
    out_root = Path(input("  Output directory [meta_analysis_output]: ").strip() or "meta_analysis_output")
    make_videos = input("  Generate videos? [y/N]: ").strip().lower() in ("y", "yes")
    print()

    for name, d in [("CLIP", clip_dir), ("Face", face_dir), ("Semantic", sem_dir)]:
        if not d.exists():
            print(f"ERROR: {name} directory not found: {d}"); sys.exit(1)

    tel_dir = out_root / "telephone"
    dist_dir = out_root / "distributions"
    tel_dir.mkdir(parents=True, exist_ok=True)
    dist_dir.mkdir(parents=True, exist_ok=True)

    # --- Load telephone data ---
    print("Loading telephone data...")
    clip_df, face_df, merged = load_telephone_data(clip_dir, face_dir)
    plot_df = merged.dropna(subset=["clip_score", "face_score"])
    print(f"  {len(plot_df)} scored image-caption pairs")

    # --- Load validation/distributions data ---
    val_df = None
    if val_dir.exists():
        print("Loading distributions (baseline) data...")
        val_df = load_validation_data(val_dir)
        if val_df is not None:
            print(f"  {len(val_df.dropna(subset=['clip_score','face_score']))} scored baseline images")

    # --- Load semantic data ---
    print("Loading semantic data...")
    sem_data = load_semantic_data(sem_dir)
    if sem_data:
        print(f"  Loaded tables: {', '.join(sem_data.keys())}")

    # --- Telephone figures (telephone/ subdirectory) ---
    print("\n--- Telephone Figures ---")
    fig_id_match_trajectory(merged, tel_dir)
    fig_aggregate_scatter(merged, tel_dir)
    fig_by_iteration(merged, tel_dir)
    fig_by_seed(merged, tel_dir)
    if val_df is not None:
        fig_validation_comparison(merged, val_df, tel_dir)
    fig_sentiment_trajectory(sem_data, tel_dir)

    print("\nBuilding telephone overview table...")
    build_overview_table(clip_df, face_df, merged, sem_data, tel_dir)

    # --- Distributions figures (distributions/ subdirectory) ---
    if val_df is not None:
        print("\n--- Distributions (Baseline) Figures ---")
        fig_dist_aggregate_scatter(val_df, dist_dir)
        fig_dist_by_seed(val_df, dist_dir)
    else:
        print("\nSkipping distributions figures (no validation data loaded)")

    # --- Videos ---
    if make_videos:
        print("\n--- Videos ---")
        print("Generating per-seed chain videos...")
        generate_seed_videos(merged, tel_dir)
        print("Generating aggregate iteration video...")
        generate_iteration_video(merged, tel_dir)
        if val_df is not None:
            print("Generating baseline vs telephone comparison video...")
            generate_comparison_video(merged, val_df, tel_dir)

    print(f"\nAll outputs saved to: {out_root.resolve()}")
    print(f"  Telephone:     {tel_dir.resolve()}")
    print(f"  Distributions: {dist_dir.resolve()}")
    print()


if __name__ == "__main__":
    main()
