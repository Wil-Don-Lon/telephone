"""
rebuild_master_log.py — Rebuilds master_log.json from per-chain log.json files,
skipping duplicate logs in violations/ and post_mortem/.

Usage:
  python rebuild_master_log.py                          # default: output/
  python rebuild_master_log.py --output-dir /path/output
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


def rebuild(output_dir: str = "output"):
    root = Path(output_dir)
    master_path = root / "master_log.json"

    # Read existing master_log to preserve run_stats (timing, pause events, etc.)
    old_master = {}
    if master_path.exists():
        old_master = json.loads(master_path.read_text(encoding="utf-8"))

    # Collect per-chain logs, skipping violations/ and post_mortem/ copies
    skip = {"violations", "post_mortem"}
    chains = []
    for log_file in sorted(root.rglob("log.json")):
        if skip & set(log_file.relative_to(root).parts):
            continue
        try:
            chains.append(json.loads(log_file.read_text(encoding="utf-8")))
        except Exception:
            continue

    # Aggregate
    total_tokens     = sum(c.get("total_tokens_used", 0) for c in chains)
    total_policy     = sum(c.get("policy_violations", {}).get("total_blocks", 0) for c in chains)
    caption_blocks   = sum(c.get("policy_violations", {}).get("caption_blocks", 0) for c in chains)
    gen_blocks       = sum(c.get("policy_violations", {}).get("generation_blocks", 0) for c in chains)
    early_stops      = sum(1 for c in chains if c.get("chain_terminated_early"))
    completed_chains = sum(1 for c in chains if not c.get("chain_terminated_early"))
    hang_events      = sum(len(c.get("hang_events", [])) for c in chains)
    total_iters_done = sum(c.get("completed_iterations", 0) for c in chains)

    stop_reasons = {}
    for c in chains:
        if c.get("chain_terminated_early"):
            reason = c.get("termination_reason", "UNKNOWN")
            stop_reasons[reason] = stop_reasons.get(reason, 0) + 1

    # Preserve run_stats from old master if available
    run_stats = old_master.get("run_stats", {})

    # Preserve old executive_summary fields we can't recompute (timing, models)
    old_exec = old_master.get("executive_summary", {})

    executive_summary = {
        "built_at":               datetime.now().isoformat(),
        "run_start":              old_exec.get("run_start"),
        "run_end":                old_exec.get("run_end"),
        "active_duration":        old_exec.get("active_duration"),
        "active_seconds":         old_exec.get("active_seconds"),
        "total_chains":           len(chains),
        "total_iterations":       total_iters_done,
        "prompt_key":             chains[0].get("prompt_type", "unknown") if chains else "unknown",
        "chains_completed":       completed_chains,
        "chains_early_stop":      early_stops,
        "early_stop_reasons":     stop_reasons,
        "policy_blocks_total":    total_policy,
        "policy_blocks_caption":  caption_blocks,
        "policy_blocks_generate": gen_blocks,
        "total_tokens_used":      total_tokens,
        "avg_tokens_per_chain":   round(total_tokens / len(chains), 1) if chains else 0,
        "hang_events_total":      hang_events,
        "pause_count":            old_exec.get("pause_count", 0),
        "pause_resume_events":    old_exec.get("pause_resume_events", []),
        "caption_model":          old_exec.get("caption_model"),
        "caption_api_mode":       old_exec.get("caption_api_mode"),
        "image_model":            old_exec.get("image_model"),
        "image_size":             old_exec.get("image_size"),
        "image_quality":          old_exec.get("image_quality"),
        "post_mortem_model":      old_exec.get("post_mortem_model"),
    }

    master = {
        "executive_summary": executive_summary,
        "run_stats":         run_stats,
        "total_chains":      len(chains),
        "total_tokens_used": total_tokens,
        "chains":            chains,
    }

    # Back up old master log
    if master_path.exists():
        backup = root / "master_log_OVERCOUNTED.json"
        master_path.rename(backup)
        print(f"Backed up old master log to: {backup}")

    master_path.write_text(json.dumps(master, indent=2), encoding="utf-8")

    # Report
    old_policy = old_master.get("executive_summary", {}).get("policy_blocks_total", "?")
    old_chains = old_master.get("total_chains", "?")

    print(f"\nRebuilt: {master_path}")
    print(f"  Chains:        {len(chains)}  (was {old_chains})")
    print(f"  Policy blocks: {total_policy}  (was {old_policy})")
    print(f"    caption:     {caption_blocks}")
    print(f"    generation:  {gen_blocks}")
    print(f"  Tokens:        {total_tokens}")
    print(f"  Early stops:   {early_stops}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild master_log.json without violation duplicates")
    parser.add_argument("--output-dir", default="output", help="Path to telephone.py output dir")
    args = parser.parse_args()
    rebuild(args.output_dir)
