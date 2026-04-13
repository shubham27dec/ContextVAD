#!/usr/bin/env python3
"""
semantic_spikes.py — Compute semantic spike signals from CLIP embeddings.

For each frame, computes 1 - cosine_similarity(frame, rolling_average_of_last_k_frames).
High spikes indicate semantic scene changes (potential segment boundaries).

Usage (standalone):
  python semantic_spikes.py --embed_dir /path/to/embeddings --output_dir /path/to/spikes
"""

import os
import argparse

import numpy as np
from tqdm import tqdm

# ─── Config (hardcoded) ─────────────────────────────────────────────────────
CONTEXT_WINDOW = 5  # Number of previous frames for rolling average


# ─── Core functions ──────────────────────────────────────────────────────────

def compute_semantic_spikes(embeddings: np.ndarray, k: int = CONTEXT_WINDOW) -> np.ndarray:
    """Compute semantic spike signal from CLIP embeddings.

    For each frame i, spike[i] = 1 - cos_sim(embedding[i], mean(embeddings[i-k:i])).
    Uses an efficient running sum for O(n) computation.

    Args:
        embeddings: (n_frames, embed_dim) L2-normalized CLIP embeddings.
        k: Context window size (number of previous frames).

    Returns:
        np.ndarray of shape (n_frames,) with spike values in [0, 2].
    """
    n, d = embeddings.shape
    spikes = np.zeros(n, dtype=np.float32)
    running_sum = np.zeros(d, dtype=np.float32)

    for i in range(1, n):
        running_sum += embeddings[i - 1]
        if i > k:
            running_sum -= embeddings[i - k - 1]

        window = min(i, k)
        context = running_sum / window
        context = context / (np.linalg.norm(context) + 1e-8)

        cos_sim = np.dot(embeddings[i], context)
        spikes[i] = 1 - cos_sim

    return spikes


def process_semantic_spikes(embed_dir: str, output_dir: str, context_window: int = CONTEXT_WINDOW):
    """Process all embedding files and save spike signals.

    Args:
        embed_dir: Directory containing .npy embedding files.
        output_dir: Directory to save .npy spike files.
        context_window: Number of previous frames for rolling average.
    """
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(embed_dir) if f.endswith(".npy"))
    print(f"Processing {len(files)} embedding files (context_window={context_window})")

    for fname in tqdm(files, desc="Spikes"):
        in_path = os.path.join(embed_dir, fname)
        out_path = os.path.join(output_dir, fname)

        # Resumable
        if os.path.exists(out_path):
            continue

        embeddings = np.load(in_path).astype(np.float32)
        spikes = compute_semantic_spikes(embeddings, context_window)
        np.save(out_path, spikes)

    print(f"Done: {len(files)} spike files saved to {output_dir}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute semantic spikes from CLIP embeddings")
    parser.add_argument("--embed_dir", required=True, help="Directory containing .npy CLIP embeddings")
    parser.add_argument("--output_dir", required=True, help="Directory to save .npy spike files")
    parser.add_argument("--context_window", type=int, default=CONTEXT_WINDOW, help=f"Context window size (default: {CONTEXT_WINDOW})")
    args = parser.parse_args()

    process_semantic_spikes(args.embed_dir, args.output_dir, args.context_window)


if __name__ == "__main__":
    main()