# ContextVAD — Training-Free Video Anomaly Detection
#
# Pipeline modules:
#   clip_encoder.py      — CLIP ViT-L/14 per-frame embedding extraction
#   semantic_spikes.py   — Semantic spike computation from embeddings
#   segment_generator.py — Spike-based segmentation + video cutting
#   llm_scorer.py        — VideoLLaMA2 anomaly scoring (P2+sumCtx)
#   postprocessing.py    — AnomMax + FAISS KNN score refinement
#   evaluate.py          — Frame-level AUC-ROC evaluation