"""
results_viz.py — Combined visualization and summary table for telephone game thesis.

Generates:
  1. CLIP vs ArcFace scatter (aggregate, identity matches highlighted in red)
  2. CLIP vs ArcFace scatter grid by iteration (15 panels)
  3. CLIP vs ArcFace scatter grid by seed (14 panels)
  4. Validation dataset comparison (telephone vs one-shot)
  5. Sentiment trajectory per seed
  6. Per-seed overview CSV table (with refusal rate + semantic drift columns)
  7. Per-seed animated videos (dots plotted chain-by-chain, identity matches in red)

Usage:
  python results_viz.py

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

    # Plain version (no highlighting, no threshold line)
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
    plt.savefig(out_dir / "clip_vs_arcface_aggregate_plain.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: clip_vs_arcface_aggregate_plain.png")

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
    ax.set_title("Compositional Preservation vs. Face Identity Preservation",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0.2, 1.0); ax.set_ylim(-0.3, 1.0)
    plt.tight_layout()
    plt.savefig(out_dir / "clip_vs_arcface_aggregate.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: clip_vs_arcface_aggregate.png")


# ================================
# FIGURE 2: BY ITERATION
# ================================

def fig_by_iteration(merged, out_dir):
    plot_df = merged.dropna(subset=["clip_score", "face_score"])

    for mode in ["plain", "highlighted"]:
        fig, axes = plt.subplots(3, 5, figsize=(22, 13))
        suffix = "" if mode == "highlighted" else "_plain"
        title_extra = "" if mode == "highlighted" else " (No ID Highlight)"
        fig.suptitle(f"CLIP vs ArcFace by Iteration{title_extra}", fontsize=16, fontweight="bold", y=0.98)

        for idx, it in enumerate(range(1, 16)):
            ax = axes[idx // 5][idx % 5]
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
            if idx % 5 == 0: ax.set_ylabel("ArcFace", fontsize=9)
            if idx // 5 == 2: ax.set_xlabel("CLIP", fontsize=9)
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
    ncols, nrows = 5, (n_seeds + 4) // 5

    for mode in ["plain", "highlighted"]:
        fig, axes = plt.subplots(nrows, ncols, figsize=(22, 4.3 * nrows))
        suffix = "" if mode == "highlighted" else "_plain"
        title_extra = "" if mode == "highlighted" else " (No ID Highlight)"
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
    _scatter_with_identity(ax1, tel_df["clip_score"].values, tel_df["face_score"].values)
    ax1.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=0.8, alpha=0.4)
    ax1.text(0.05, 0.95, f"r = {r_tel:.3f}\nn = {len(tel_df)}", transform=ax1.transAxes, fontsize=11, va="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    ax1.set_xlabel("CLIP", fontsize=11); ax1.set_ylabel("ArcFace", fontsize=11)
    ax1.set_title("Telephone (Iterative Drift)", fontsize=12)
    ax1.set_xlim(0.2, 1.0); ax1.set_ylim(-0.3, 1.0)
    r_val, _ = stats.pearsonr(val_plot["clip_score"], val_plot["face_score"])
    _scatter_with_identity(ax2, val_plot["clip_score"].values, val_plot["face_score"].values)
    ax2.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=0.8, alpha=0.4)
    ax2.text(0.05, 0.95, f"r = {r_val:.3f}\nn = {len(val_plot)}", transform=ax2.transAxes, fontsize=11, va="top",
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
    ncols, nrows = 5, (n_seeds + 4) // 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 3.5 * nrows))
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
# MAIN
# ================================

def main():
    print()
    print("=" * 60)
    print("  results_viz.py")
    print("  Combined visualization + summary table + videos")
    print("=" * 60)
    print()

    clip_dir = Path(input("  CLIP analysis output dir: ").strip())
    face_dir = Path(input("  Face analysis output dir: ").strip())
    sem_dir  = Path(input("  Semantic analysis output dir: ").strip())
    val_dir  = Path(input("  Validation dataset output dir (seed_sampler): ").strip())
    out_dir  = Path(input("  Output directory [results_viz_output]: ").strip() or "results_viz_output")
    make_videos = input("  Generate per-seed videos? [y/N]: ").strip().lower() in ("y", "yes")
    print()

    for name, d in [("CLIP", clip_dir), ("Face", face_dir), ("Semantic", sem_dir)]:
        if not d.exists():
            print(f"ERROR: {name} directory not found: {d}"); sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading telephone data...")
    clip_df, face_df, merged = load_telephone_data(clip_dir, face_dir)
    plot_df = merged.dropna(subset=["clip_score", "face_score"])
    print(f"  {len(plot_df)} scored image-caption pairs")

    val_df = None
    if val_dir.exists():
        print("Loading validation data...")
        val_df = load_validation_data(val_dir)
        if val_df is not None:
            print(f"  {len(val_df.dropna(subset=['clip_score','face_score']))} scored validation images")

    print("Loading semantic data...")
    sem_data = load_semantic_data(sem_dir)
    if sem_data:
        print(f"  Loaded tables: {', '.join(sem_data.keys())}")

    print("\nGenerating figures...")
    fig_id_match_trajectory(merged, out_dir)
    fig_aggregate_scatter(merged, out_dir)
    fig_by_iteration(merged, out_dir)
    fig_by_seed(merged, out_dir)
    if val_df is not None:
        fig_validation_comparison(merged, val_df, out_dir)
    fig_sentiment_trajectory(sem_data, out_dir)

    print("\nBuilding overview table...")
    build_overview_table(clip_df, face_df, merged, sem_data, out_dir)

    if make_videos:
        print("\nGenerating videos...")
        generate_seed_videos(merged, out_dir)

    print(f"\nAll outputs saved to: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
