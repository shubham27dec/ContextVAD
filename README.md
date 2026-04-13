# ContextVAD: Training-Free Video Anomaly Detection using Semantic Segmentation and Video-LLMs

A training-free pipeline for video anomaly detection that combines CLIP-based semantic scene segmentation with VideoLLaMA2 scoring and FAISS KNN post-processing.

## Architecture

```
Video → CLIP ViT-L/14 → Semantic Spikes → Segmentation → VideoLLaMA2 Scoring → Post-Processing → Evaluation
```

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `clip_encoder.py` | Per-frame CLIP ViT-L/14 embeddings (768-dim) |
| 2a | `semantic_spikes.py` | Scene change detection via cosine similarity drops |
| 2b | `segment_generator.py` | Split videos at spike boundaries, cut segment clips |
| 3 | `llm_scorer.py` | VideoLLaMA2.1-7B-16F anomaly scoring (P2+sumCtx prompt) |
| 4 | `postprocessing.py` | AnomMax + FAISS KNN score refinement + Gaussian smoothing |
| 5 | `evaluate.py` | Frame-level AUC-ROC + AP evaluation |

## Repository Structure

```
ContextVAD/
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    ├── __init__.py
    ├── clip_encoder.py
    ├── semantic_spikes.py
    ├── segment_generator.py
    ├── llm_scorer.py
    ├── postprocessing.py
    └── evaluate.py
```

## Datasets

### UCF-Crime
- 1900 long untrimmed surveillance videos, 13 anomaly categories
- Test split: 290 videos (140 anomaly + 150 normal)
- [Download](https://www.crcv.ucf.edu/projects/real-world/)
- Place videos in `datasets/ucf_crime/videos/`, labels in `datasets/ucf_crime/labels/`

### XD-Violence
- 4754 untrimmed videos from movies, YouTube, surveillance, etc.
- Test split: 800 videos, 6 violence categories + normal
- [Download](https://roc-ng.github.io/XD-Violence/)
- Place videos in `datasets/xd-violence/videos/`, annotations in `datasets/xd-violence/annotations.txt`

## Usage

### Stages 1–3: Feature Extraction + Scoring (GPU required)

Stages 1–3 run on GPU (tested on Kaggle with 2× T4). This produces CLIP embeddings, segment boundaries, and a scoring CSV with per-segment anomaly scores.

```bash
# Stage 1: CLIP encoding
python src/clip_encoder.py --video_dir datasets/ucf_crime/videos/ --output_dir outputs/ucf_crime/embeddings/ --num_gpus 2

# Stage 2a: Semantic spikes
python src/semantic_spikes.py --embed_dir outputs/ucf_crime/embeddings/ --output_dir outputs/ucf_crime/spikes/

# Stage 2b: Segmentation
python src/segment_generator.py --spike_dir outputs/ucf_crime/spikes/ --video_dir datasets/ucf_crime/videos/ --output_dir outputs/ucf_crime/segments/

# Stage 3: LLM scoring
python src/llm_scorer.py --manifest outputs/ucf_crime/segments/segment_manifest.txt --output_csv outputs/ucf_crime/scores.csv
```

### Stage 4: Post-Processing (CPU, no GPU needed)

```bash
# Full pipeline: AnomMax + KNN sweep
python src/postprocessing.py \
  --base_csv outputs/ucf_crime/scores.csv \
  --embed_dir outputs/ucf_crime/embeddings/ \
  --labels datasets/ucf_crime/labels/ucf-crime_test_labels.txt \
  --output_dir outputs/ucf_crime/

# Best fixed params (no sweep)
python src/postprocessing.py \
  --base_csv outputs/ucf_crime/scores.csv \
  --embed_dir outputs/ucf_crime/embeddings/ \
  --labels datasets/ucf_crime/labels/ucf-crime_test_labels.txt \
  --output_dir outputs/ucf_crime/ \
  --step best
```

For XD-Violence, add `--video_dir` to discover normal videos:
```bash
python src/postprocessing.py \
  --base_csv outputs/xd_violence/scores.csv \
  --embed_dir outputs/xd_violence/embeddings/ \
  --labels datasets/xd-violence/annotations.txt \
  --video_dir datasets/xd-violence/videos/ \
  --output_dir outputs/xd_violence/
```

### Stage 5: Evaluation (CPU)

```bash
# Single evaluation
python src/evaluate.py --csv outputs/ucf_crime/scores.csv --labels datasets/ucf_crime/labels/ucf-crime_test_labels.txt --smooth 700

# Smoothing sweep
python src/evaluate.py --csv outputs/ucf_crime/scores.csv --labels datasets/ucf_crime/labels/ucf-crime_test_labels.txt --sweep

# Per-video breakdown
python src/evaluate.py --csv outputs/ucf_crime/scores.csv --labels datasets/ucf_crime/labels/ucf-crime_test_labels.txt --smooth 700 --per-video

# XD-Violence (pass --video_dir for normal video discovery)
python src/evaluate.py --csv outputs/xd_violence/scores.csv --labels datasets/xd-violence/annotations.txt --video_dir datasets/xd-violence/videos/ --sweep
```

## Requirements

```bash
pip install -r requirements.txt
```

Stages 1–3 additionally require [VideoLLaMA2](https://github.com/DAMO-NLP-SG/VideoLLaMA2) (cloned separately on GPU machine).