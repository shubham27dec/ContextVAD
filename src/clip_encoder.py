#!/usr/bin/env python3
"""
clip_encoder.py — CLIP ViT-L/14 every-frame feature extraction.

Extracts per-frame CLIP embeddings from videos using ViT-L/14 (768-dim).
Supports batch processing and multi-GPU parallelism.

Usage (standalone):
  python clip_encoder.py --video_dir /path/to/videos --output_dir /path/to/embeddings
  python clip_encoder.py --video_dir /path/to/videos --output_dir /path/to/embeddings --num_gpus 2
"""

import os
import sys
import argparse

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm

try:
    import clip
except ImportError:
    print("ERROR: pip install git+https://github.com/openai/CLIP.git")
    sys.exit(1)

# ─── Config (hardcoded) ─────────────────────────────────────────────────────
CLIP_MODEL = "ViT-L/14"
EMBED_DIM = 768
BATCH_SIZE = 64
MAX_RESOLUTION = (1280, 720)


# ─── Core functions ──────────────────────────────────────────────────────────

def load_clip_model(device: str):
    """Load CLIP ViT-L/14 model in eval mode."""
    model, preprocess = clip.load(CLIP_MODEL, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, preprocess


def process_batch(frames, model, preprocess, device):
    """Encode a batch of RGB frames into L2-normalized CLIP embeddings."""
    with torch.no_grad():
        processed = torch.stack(
            [preprocess(Image.fromarray(f)) for f in frames]
        ).to(device)
        feats = model.encode_image(processed)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        result = feats.cpu().numpy()
    del processed, feats
    torch.cuda.empty_cache()
    return result


def extract_single_video(video_path, model, preprocess, device):
    """Extract per-frame CLIP embeddings from a single video.

    Returns:
        np.ndarray of shape (n_frames, 768) or None if video can't be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ratio = min(MAX_RESOLUTION[0] / width, MAX_RESOLUTION[1] / height)
    if ratio < 1:
        new_size = (int(width * ratio) // 2 * 2, int(height * ratio) // 2 * 2)
    else:
        new_size = (width, height)

    all_features = []
    batch_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), new_size)
        batch_frames.append(frame_rgb)

        if len(batch_frames) == BATCH_SIZE:
            feats = process_batch(batch_frames, model, preprocess, device)
            all_features.append(feats)
            batch_frames = []

    if batch_frames:
        feats = process_batch(batch_frames, model, preprocess, device)
        all_features.append(feats)

    cap.release()

    if not all_features:
        return None
    return np.concatenate(all_features).astype(np.float32)


def worker(video_list, device, video_dir, output_dir):
    """Worker function for multi-GPU parallel extraction."""
    import warnings
    warnings.filterwarnings("ignore")
    import logging
    logging.disable(logging.WARNING)

    model, preprocess = load_clip_model(device)
    tag = device.replace("cuda:", "GPU")

    for fname in tqdm(video_list, desc=tag):
        out_path = os.path.join(output_dir, f"{fname}.npy")

        # Resumable: skip if already extracted with correct dim
        if os.path.exists(out_path):
            try:
                existing = np.load(out_path)
                if existing.ndim == 2 and existing.shape[1] == EMBED_DIM:
                    continue
            except Exception:
                pass

        video_path = os.path.join(video_dir, fname)
        try:
            feats = extract_single_video(video_path, model, preprocess, device)
            if feats is not None:
                np.save(out_path, feats)
        except Exception as e:
            print(f"ERR {fname}: {e}")


def process_clip_embeddings(video_dir, output_dir, num_gpus=None):
    """Process all videos in a directory, extracting CLIP ViT-L/14 embeddings.

    Args:
        video_dir: Directory containing .mp4 video files.
        output_dir: Directory to save .npy embedding files.
        num_gpus: Number of GPUs to use (auto-detect if None).
    """
    os.makedirs(output_dir, exist_ok=True)

    video_files = sorted([f for f in os.listdir(video_dir) if f.lower().endswith(".mp4")])
    print(f"Total videos: {len(video_files)}")

    if num_gpus is None:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if num_gpus >= 2:
        print(f"Using {num_gpus} GPUs in parallel")
        mp.set_start_method("spawn", force=True)
        chunk_size = len(video_files) // num_gpus
        processes = []
        for i in range(num_gpus):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_gpus - 1 else len(video_files)
            p = mp.Process(
                target=worker,
                args=(video_files[start_idx:end_idx], f"cuda:{i}", video_dir, output_dir),
            )
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
    elif num_gpus == 1:
        print("Using 1 GPU")
        worker(video_files, "cuda:0", video_dir, output_dir)
    else:
        print("No GPU available, using CPU (will be slow)")
        worker(video_files, "cpu", video_dir, output_dir)

    done = [f for f in os.listdir(output_dir) if f.endswith(".npy")]
    print(f"\nCompleted: {len(done)}/{len(video_files)} videos")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CLIP ViT-L/14 every-frame feature extraction")
    parser.add_argument("--video_dir", required=True, help="Directory containing .mp4 video files")
    parser.add_argument("--output_dir", required=True, help="Directory to save .npy embedding files")
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs (auto-detect if omitted)")
    args = parser.parse_args()

    process_clip_embeddings(args.video_dir, args.output_dir, args.num_gpus)


if __name__ == "__main__":
    main()