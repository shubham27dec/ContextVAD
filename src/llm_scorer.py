#!/usr/bin/env python3
"""
llm_scorer.py — VideoLLaMA2 anomaly scoring for video segments.

Scores each video segment using VideoLLaMA2.1-7B-16F with the P2+sumCtx prompt:
  - P2 base prompt asks for activity description + anomaly score [0.0-1.0]
  - sumCtx adds previous segment's activity summary as context

Three context variants:
  - noCtx:  No context (each segment scored independently)
  - sumCtx: Previous segment's activity summary as context (recommended)
  - rawCtx: Previous segment's full raw response as context

Usage (standalone):
  python llm_scorer.py \
    --manifest /path/to/segment_manifest.txt \
    --output_csv /path/to/scores.csv \
    --context sumCtx
"""

import os
import re
import csv
import sys
import gc
import argparse
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

# ─── Config (hardcoded) ─────────────────────────────────────────────────────
MODEL_NAME = "DAMO-NLP-SG/VideoLLaMA2.1-7B-16F"

P2_BASE = (
    "First, explain in one sentence what activity is shown in this video. "
    "Then rate how anomalous it is. "
    "If the activity is completely normal, the score must be 0.0. "
    "If the activity is anomalous in any way, score it between 0.1 and 1.0 "
    "where higher means more anomalous. "
    "Format:\nACTIVITY: <one sentence>\nANOMALY SCORE: <number>"
)


# ─── Helper functions ────────────────────────────────────────────────────────

def extract_activity(raw_text):
    """Extract activity description from LLM response."""
    for line in raw_text.split("\n"):
        stripped = line.strip()
        if stripped.upper().startswith("ACTIVITY:"):
            activity = stripped.split(":", 1)[1].strip()
            if activity:
                return activity
    first = raw_text.split(".")[0].strip()
    return first[:200] if first else ""


def make_summary_ctx(base, activity_summary):
    """Create prompt with previous segment's activity summary as context."""
    return f'[Previous segment showed: "{activity_summary}"]\n\n{base}'


def make_raw_ctx(base, raw_prev):
    """Create prompt with previous segment's full raw response as context."""
    trimmed = raw_prev[:500]
    return f"[Previous segment analysis: {trimmed}]\n\n{base}"


def parse_score(text):
    """Extract anomaly score from LLM response text.

    Returns the last valid float in [0.0, 1.0], or -1.0 if none found.
    """
    matches = re.findall(r"\d+\.?\d*", text)
    valid = [float(m) for m in matches if 0.0 <= float(m) <= 1.0]
    return valid[-1] if valid else -1.0


def load_model():
    """Load VideoLLaMA2 model, processor, and tokenizer.

    Requires VideoLLaMA2 to be cloned and on sys.path.
    See notebook for setup instructions.
    """
    try:
        from videollama2 import model_init, mm_infer
        from videollama2.utils import disable_torch_init
    except ImportError:
        print("ERROR: VideoLLaMA2 not found on sys.path.")
        print("Clone it: git clone https://github.com/DAMO-NLP-SG/VideoLLaMA2.git")
        print("Then: sys.path.insert(0, '/path/to/VideoLLaMA2')")
        sys.exit(1)

    disable_torch_init()
    print("Loading model...", flush=True)
    model, processor, tokenizer = model_init(MODEL_NAME, device_map="auto")
    model = model.eval()
    dev = next(model.parameters()).device
    print(f"Model on {dev}", flush=True)
    return model, processor, tokenizer


def score_segment(video_path, prompt, model, processor, tokenizer):
    """Score a single video segment using VideoLLaMA2.

    Args:
        video_path: Path to segment .mp4 file.
        prompt: The scoring prompt (with or without context).
        model: VideoLLaMA2 model.
        processor: VideoLLaMA2 processor.
        tokenizer: VideoLLaMA2 tokenizer.

    Returns:
        (score, raw_response) tuple.
    """
    from videollama2 import mm_infer

    try:
        raw = mm_infer(
            processor["video"](video_path),
            prompt,
            model=model,
            tokenizer=tokenizer,
            modal="video",
            do_sample=False,
        )
    except Exception as e:
        return -1.0, f"ERROR: {e}"

    score = parse_score(raw)
    return score, raw


def score_manifest(manifest_path, output_csv, context_mode="sumCtx",
                   model=None, processor=None, tokenizer=None):
    """Score all segments in a manifest file.

    Args:
        manifest_path: Path to segment_manifest.txt.
        output_csv: Path to output CSV file.
        context_mode: One of 'noCtx', 'sumCtx', 'rawCtx'.
        model, processor, tokenizer: Pre-loaded model (loads if None).
    """
    if model is None:
        model, processor, tokenizer = load_model()

    # Read manifest
    lines = []
    with open(manifest_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                lines.append({"path": parts[0], "start": parts[1], "end": parts[2]})

    # Group by video
    vlines = defaultdict(list)
    for entry in lines:
        vid = entry["path"].split("/")[-2].replace("_segments", "")
        vlines[vid].append(entry)

    print(f"Manifest: {len(lines)} segments from {len(vlines)} videos", flush=True)
    print(f"Context mode: {context_mode}", flush=True)

    # Resume support
    done = set()
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        with open(output_csv) as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    done.add(row[0])
        print(f"Resuming: {len(done)} already scored", flush=True)

    score_col = f"score_p2_{context_mode.lower()}"
    raw_col = f"raw_p2_{context_mode.lower()}"

    if not done:
        with open(output_csv, "w", newline="") as f:
            csv.writer(f).writerow([
                "path", "start", "end", score_col, raw_col, "status"
            ])

    scored = len(done)
    total = len(lines)

    for vid_name in sorted(vlines.keys()):
        segs = vlines[vid_name]
        prev_activity = ""
        prev_raw = ""

        for seg in segs:
            if seg["path"] in done:
                continue

            # Build prompt based on context mode
            if context_mode == "noCtx" or not prev_activity:
                prompt = P2_BASE
            elif context_mode == "sumCtx":
                prompt = make_summary_ctx(P2_BASE, prev_activity)
            elif context_mode == "rawCtx":
                prompt = make_raw_ctx(P2_BASE, prev_raw)
            else:
                prompt = P2_BASE

            score, raw = score_segment(seg["path"], prompt, model, processor, tokenizer)

            # Update context for next segment
            prev_activity = extract_activity(raw)
            prev_raw = raw

            # Write result
            with open(output_csv, "a", newline="") as f:
                csv.writer(f).writerow([
                    seg["path"], seg["start"], seg["end"],
                    f"{score:.4f}" if score >= 0 else "-1.0",
                    raw.replace("\n", " "),
                    "ok" if score >= 0 else "parse_fail",
                ])

            scored += 1
            if scored % 10 == 0:
                print(f"  [{scored}/{total}] {vid_name}", flush=True)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nDone: {scored} segments scored → {output_csv}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VideoLLaMA2 anomaly scoring for video segments")
    parser.add_argument("--manifest", required=True, help="Path to segment_manifest.txt")
    parser.add_argument("--output_csv", required=True, help="Output CSV path")
    parser.add_argument("--context", default="sumCtx", choices=["noCtx", "sumCtx", "rawCtx"],
                        help="Context mode (default: sumCtx)")
    args = parser.parse_args()

    score_manifest(args.manifest, args.output_csv, args.context)


if __name__ == "__main__":
    main()