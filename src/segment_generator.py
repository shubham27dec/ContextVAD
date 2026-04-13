#!/usr/bin/env python3
"""
segment_generator.py — Spike-based video segmentation + segment video cutting.

Splits videos at semantic spike points exceeding mean + std_factor * std.
Produces: segment boundary .npy files, cut segment .mp4 videos, and a manifest.

Usage (standalone):
  python segment_generator.py \
    --spike_dir /path/to/spikes \
    --clip_dir /path/to/embeddings \
    --video_dir /path/to/videos \
    --output_dir /path/to/output
"""

import os
import glob
import argparse

import cv2
import numpy as np
from tqdm import tqdm

# ─── Config (hardcoded) ─────────────────────────────────────────────────────
SPIKE_STD_FACTOR = 2.0
MIN_SPIKE_THRESHOLD = 0.05
MIN_SEGMENT_LENGTH = 5
MAX_RESOLUTION = (1280, 720)


# ─── Core functions ──────────────────────────────────────────────────────────

def create_segments(spikes: np.ndarray, std_factor: float = SPIKE_STD_FACTOR,
                    min_threshold: float = MIN_SPIKE_THRESHOLD,
                    min_length: int = MIN_SEGMENT_LENGTH):
    """Split a spike signal into segments at boundary points.

    A boundary is created at frame i if:
      - spikes[i] > max(mean + std_factor * std, min_threshold)
      - (i - segment_start) >= min_length

    Args:
        spikes: 1D array of spike values per frame.
        std_factor: Multiplier for standard deviation threshold.
        min_threshold: Minimum spike threshold.
        min_length: Minimum segment length in frames.

    Returns:
        segments: np.ndarray of shape (n_segments, 3) with [start, end, beta].
        threshold: The computed threshold value.
    """
    mean = spikes.mean()
    std = spikes.std()
    threshold = max(mean + std_factor * std, min_threshold)

    segments = []
    start = 0
    current_beta = 0.0

    for i in range(1, len(spikes)):
        if spikes[i] > threshold and (i - start) >= min_length:
            segments.append([start, i, current_beta])
            start = i
            current_beta = spikes[i]

    # Final segment
    segments.append([start, len(spikes), current_beta])

    return np.array(segments), threshold


def cut_segments(video_path, seg_out_dir, segments, fps):
    """Cut a video into segment clips based on frame boundaries.

    Args:
        video_path: Path to source video.
        seg_out_dir: Output directory for segment .mp4 files.
        segments: Array of [start_frame, end_frame, beta] per segment.
        fps: Video frame rate.

    Returns:
        List of manifest lines: "{abs_path} {start_frame} {end_frame}"
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ratio = min(MAX_RESOLUTION[0] / w, MAX_RESOLUTION[1] / h)
    ns = (int(w * ratio) // 2 * 2, int(h * ratio) // 2 * 2) if ratio < 1 else (w, h)

    os.makedirs(seg_out_dir, exist_ok=True)
    manifest_lines = []

    for si, (sf, ef, beta) in enumerate(segments):
        sf, ef = int(sf), int(ef)
        out_path = os.path.join(seg_out_dir, f"segment_{si:04d}.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, ns)
        if not writer.isOpened():
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
        for _ in range(ef - sf):
            ret, frame = cap.read()
            if not ret:
                break
            if ratio < 1:
                frame = cv2.resize(frame, ns)
            writer.write(frame)

        writer.release()
        manifest_lines.append(f"{os.path.abspath(out_path)} {sf} {ef}")

    cap.release()
    return manifest_lines


def process_segments(spike_dir, video_dir, output_dir):
    """Process all videos: compute segments from spikes, cut video segments.

    Args:
        spike_dir: Directory containing .npy spike files.
        video_dir: Directory containing source .mp4 videos.
        output_dir: Base output directory (creates segments/, segment_videos/, manifest).
    """
    segment_npy_dir = os.path.join(output_dir, "segments")
    segment_vid_dir = os.path.join(output_dir, "segment_videos")
    manifest_path = os.path.join(output_dir, "segment_manifest.txt")

    os.makedirs(segment_npy_dir, exist_ok=True)
    os.makedirs(segment_vid_dir, exist_ok=True)

    spike_files = sorted(glob.glob(os.path.join(spike_dir, "*.npy")))
    print(f"Found {len(spike_files)} spike files")

    all_manifest = []

    for spike_path in tqdm(spike_files, desc="Segmenting"):
        basename = os.path.basename(spike_path)
        # Handle naming: "Abuse028_x264.mp4.npy" or "Abuse028_x264.npy"
        stem = basename.replace(".mp4.npy", "").replace(".npy", "")

        spikes = np.load(spike_path)

        # Compute segments
        segments, threshold = create_segments(spikes)
        np.save(os.path.join(segment_npy_dir, basename), segments)

        # Find source video
        vid_path = os.path.join(video_dir, f"{stem}.mp4")
        if not os.path.exists(vid_path):
            vid_path = os.path.join(video_dir, basename.replace(".npy", ""))
        if not os.path.exists(vid_path):
            tqdm.write(f"[WARN] No video for {stem}")
            continue

        # Get FPS and cut segments
        cap_tmp = cv2.VideoCapture(vid_path)
        fps = cap_tmp.get(cv2.CAP_PROP_FPS)
        cap_tmp.release()

        seg_vid_out = os.path.join(segment_vid_dir, f"{stem}_segments")
        lines = cut_segments(vid_path, seg_vid_out, segments, fps)
        all_manifest.extend(lines)

        tqdm.write(f"[OK] {stem}: {len(segments)} segs, {len(spikes)} frames")

    # Write manifest
    with open(manifest_path, "w") as f:
        for line in all_manifest:
            f.write(line + "\n")

    # Summary
    seg_count = len(glob.glob(os.path.join(segment_npy_dir, "*.npy")))
    vid_count = len(glob.glob(os.path.join(segment_vid_dir, "*_segments", "segment_*.mp4")))
    print(f"\nDone!")
    print(f"  Segment NPYs: {seg_count} files in {segment_npy_dir}")
    print(f"  Segment Videos: {vid_count} clips in {segment_vid_dir}")
    print(f"  Manifest: {len(all_manifest)} entries in {manifest_path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Spike-based video segmentation + cutting")
    parser.add_argument("--spike_dir", required=True, help="Directory containing .npy spike files")
    parser.add_argument("--video_dir", required=True, help="Directory containing source .mp4 videos")
    parser.add_argument("--output_dir", required=True, help="Base output directory")
    args = parser.parse_args()

    process_segments(args.spike_dir, args.video_dir, args.output_dir)


if __name__ == "__main__":
    main()