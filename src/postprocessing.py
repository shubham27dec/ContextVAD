#!/usr/bin/env python3
"""
postprocessing.py — ContextVAD Post-Processing Pipeline.

Dataset-agnostic: auto-detects UCF-Crime vs XD-Violence label format.

Two steps:
  1. AnomMax: If LLM raw response contains "anomalous", extract max float ≤1.0
     from entire raw text and upgrade the score if higher.
  2. FAISS KNN Refinement: For each segment, find K nearest neighbors by CLIP
     embedding similarity, weighted-average their scores, blend with original.
     Combined with Gaussian smoothing sweep to find optimal σ.

Usage:
  python postprocessing.py --base_csv scores.csv --embed_dir embeddings/ --labels labels.txt
  python postprocessing.py --base_csv scores.csv --embed_dir embeddings/ --labels labels.txt --video_dir videos/
  python postprocessing.py --base_csv scores.csv --embed_dir embeddings/ --labels labels.txt --step best
"""

import argparse
import csv
import os
import re
import time

import numpy as np
import pandas as pd

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score, average_precision_score

# ─── Config (hardcoded) ─────────────────────────────────────────────────────
SCORE_COL_BASE = "score_p2_sumctx"
RAW_COL = "raw_p2_sumctx"
SCORE_COL_ANOMMAX = "score_anommax"

K_ALL = [5, 10, 15, 20, 25, 30, 35, 40]
K_CROSS = [5, 10, 15, 20]
ALPHA_VALUES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
SIGMA_VALUES = list(range(0, 1250, 50))  # 0, 50, 100, ..., 1200


# ─── Label parsing (auto-detect, shared with evaluate.py) ───────────────────

def parse_labels(path, video_dir=None):
    """Auto-detect label format and parse.
    
    UCF-Crime:    name category s1 e1 s2 e2  (always 6 fields)
    XD-Violence:  name s1 e1 [s2 e2 ...]     (variable fields, no category)
    """
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                if parts[1].lstrip('-').isdigit():
                    return _parse_xd_violence(path, video_dir)
                else:
                    return _parse_ucf_crime(path)
    return {}


def _parse_ucf_crime(path):
    labels = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            name = parts[0]
            s1, e1, s2, e2 = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
            intervals = []
            if s1 != -1 and e1 != -1:
                intervals.append((s1, e1))
            if s2 != -1 and e2 != -1:
                intervals.append((s2, e2))
            labels[normalize_name(name)] = intervals
    return labels


def _parse_xd_violence(path, video_dir=None):
    labels = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            name = parts[0]
            if name.endswith('.mp4'):
                name = name[:-4]
            nums = [int(x) for x in parts[1:]]
            intervals = [(nums[i], nums[i+1]) for i in range(0, len(nums), 2)]
            labels[normalize_name(name)] = intervals
    
    if video_dir and os.path.isdir(video_dir):
        for f in os.listdir(video_dir):
            if f.endswith('.mp4'):
                stem = normalize_name(f[:-4])
                if stem not in labels:
                    labels[stem] = []
    
    return labels


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: AnomMax
# ═══════════════════════════════════════════════════════════════════════════════

def apply_anommax(df, score_col, raw_col):
    """AnomMax: If raw contains 'anomalous', extract ALL floats ≤1.0 from the
    entire raw text, take the max, and assign as score if higher than current."""
    new_scores = df[score_col].copy().astype(float)
    mask = df[raw_col].str.contains("anomalous", case=False, na=False)

    upgraded = 0
    for idx in df[mask].index:
        raw = str(df.loc[idx, raw_col])
        floats = [float(x) for x in re.findall(r"\d+\.?\d*", raw) if float(x) <= 1.0]
        if floats:
            mx = max(floats)
            if mx > new_scores[idx]:
                new_scores[idx] = mx
                upgraded += 1

    new_scores = new_scores.clip(0, 1)
    print(f"  AnomMax: {mask.sum()} anomalous segments, {upgraded} upgraded")
    return new_scores


def run_anommax(base_csv, output_csv, labels_path, video_dir=None):
    """Step 1: Generate AnomMax CSV from base scores."""
    print("\n" + "=" * 80)
    print("STEP 1: AnomMax")
    print("=" * 80)

    df = pd.read_csv(base_csv)
    print(f"  Input: {base_csv} ({len(df)} segments)")

    mask_anom = df[RAW_COL].str.contains("anomalous", case=False, na=False)
    mask_normal = df[RAW_COL].str.contains("normal", case=False, na=False) & ~mask_anom
    print(f"  Keywords: {mask_anom.sum()} anomalous, {mask_normal.sum()} normal, "
          f"{(~mask_anom & ~mask_normal).sum()} neither")

    df[SCORE_COL_ANOMMAX] = apply_anommax(df, SCORE_COL_BASE, RAW_COL)
    df.to_csv(output_csv, index=False)
    print(f"  Output: {output_csv}")

    # Quick baseline eval
    labels = parse_labels(labels_path, video_dir)
    rows, header = load_csv(output_csv)
    orig_map = build_score_map(rows, SCORE_COL_ANOMMAX)
    best_auc, best_ap, best_sig = 0, 0, 0
    for sigma in SIGMA_VALUES:
        auc, ap = evaluate_metrics(rows, orig_map, labels, sigma)
        if auc > best_auc:
            best_auc = auc
            best_ap_at_best_auc = ap
            best_sig = sigma
    print(f"  Baseline (AnomMax, no KNN): σ={best_sig}, AUC={best_auc:.4f}, AP={best_ap_at_best_auc:.4f}")

    return output_csv


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: FAISS KNN Refinement + Gaussian Smoothing Sweep
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_name(name):
    """Normalize video name for matching: strip '=' (Kaggle mangles v= to v-)."""
    return name.replace("=", "")


def extract_video_stem(path_str):
    parent = os.path.basename(os.path.dirname(path_str))
    return normalize_name(parent.replace("_segments", ""))


def load_csv(csv_path):
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        for row in reader:
            rows.append(row)
    return rows, header


def load_embedding(embed_dir, video_stem):
    for suffix in [".npy", ".mp4.npy", "_x264.npy"]:
        p = os.path.join(embed_dir, f"{video_stem}{suffix}")
        if os.path.exists(p):
            return np.load(p).astype(np.float32)
    return None


def build_segment_embeddings(rows, score_col, embed_dir):
    embeddings_list, scores_list, video_ids, row_indices = [], [], [], []
    emb_cache, missing = {}, set()

    for i, row in enumerate(rows):
        vstem = extract_video_stem(row["path"])
        if vstem not in emb_cache and vstem not in missing:
            fe = load_embedding(embed_dir, vstem)
            if fe is not None:
                emb_cache[vstem] = fe
            else:
                missing.add(vstem)
        if vstem in missing:
            continue

        try:
            score = max(0.0, float(row.get(score_col, "0")))
        except (ValueError, TypeError):
            score = 0.0
        try:
            start, end = int(row["start"]), int(row["end"])
        except (ValueError, KeyError):
            continue

        fe = emb_cache[vstem]
        nf = fe.shape[0]
        s, e = max(0, min(start, nf - 1)), max(1, min(end, nf))
        mean_emb = fe[s:e].mean(axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb /= norm

        embeddings_list.append(mean_emb)
        scores_list.append(score)
        video_ids.append(vstem)
        row_indices.append(i)

    if missing:
        print(f"  Warning: {len(missing)} videos missing embeddings")

    return (np.array(embeddings_list, dtype=np.float32),
            np.array(scores_list, dtype=np.float32),
            video_ids, row_indices)


def build_score_map(rows, score_col):
    score_map = {}
    for i, row in enumerate(rows):
        try:
            score_map[i] = max(0.0, float(row.get(score_col, "0")))
        except (ValueError, TypeError):
            score_map[i] = 0.0
    return score_map


def evaluate_metrics(rows, score_map, labels, sigma):
    """Compute both AUC-ROC and AP."""
    video_data = {}
    for i, row in enumerate(rows):
        score = score_map.get(i, 0.0)
        try:
            start, end = int(row["start"]), int(row["end"])
        except (ValueError, KeyError):
            continue
        vname = extract_video_stem(row["path"])
        # Try with .mp4 for UCF-Crime compat
        if vname not in labels and vname + ".mp4" in labels:
            vname = vname + ".mp4"
        if vname not in labels:
            continue
        if vname not in video_data:
            video_data[vname] = {"segs": [], "mf": 0}
        video_data[vname]["segs"].append((start, end, score))
        video_data[vname]["mf"] = max(video_data[vname]["mf"], end)

    all_scores, all_labels = [], []
    for vname in sorted(video_data.keys()):
        d = video_data[vname]
        mf = d["mf"]
        if mf == 0:
            continue
        fs = np.zeros(mf, dtype=np.float64)
        for s, e, sc in d["segs"]:
            ef = min(e, mf)
            if s < ef:
                fs[s:ef] = sc
        if sigma > 0:
            fs = gaussian_filter1d(fs, sigma=sigma)
        gt = np.zeros(mf, dtype=int)
        for sa, ea in labels[vname]:
            ea_adj = min(ea, mf - 1)
            if sa <= ea_adj:
                gt[sa:ea_adj + 1] = 1
        all_scores.extend(fs.tolist())
        all_labels.extend(gt.tolist())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    if len(np.unique(all_labels)) < 2:
        return 0.0, 0.0
    auc = roc_auc_score(all_labels, all_scores)
    ap = average_precision_score(all_labels, all_scores)
    return auc, ap


def faiss_knn_refine(embeddings, scores, video_ids, k, cross_video_only):
    """FAISS KNN score refinement."""
    n, d = embeddings.shape
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    search_k = min(k * 3 if cross_video_only else k + 1, n)
    distances, indices = index.search(embeddings, search_k)

    refined = np.zeros(n, dtype=np.float32)
    for i in range(n):
        ns, sims = [], []
        for j in range(search_k):
            nn = indices[i][j]
            if nn == i:
                continue
            if cross_video_only and video_ids[nn] == video_ids[i]:
                continue
            ns.append(scores[nn])
            sims.append(distances[i][j])
            if len(ns) >= k:
                break
        if not ns:
            refined[i] = scores[i]
            continue
        ns, sims = np.array(ns), np.array(sims)
        shifted = sims - sims.max()
        w = np.exp(shifted)
        w /= w.sum()
        refined[i] = np.dot(w, ns)
    return refined


def write_output_csv(rows, header, row_indices, final_scores, output_path, col_name):
    rm = {idx: sc for idx, sc in zip(row_indices, final_scores)}
    out_header = list(header) + [col_name] if col_name not in header else list(header)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_header)
        writer.writeheader()
        for i, row in enumerate(rows):
            out = dict(row)
            out[col_name] = f"{rm[i]:.4f}" if i in rm else "0.0"
            writer.writerow(out)


def run_knn_sweep(anommax_csv, embed_dir, labels_path, output_dir, video_dir=None, modes=None):
    """Step 2: FAISS KNN sweep + Gaussian smoothing."""
    if not HAS_FAISS:
        print("ERROR: pip install faiss-cpu")
        return

    print("\n" + "=" * 80)
    print("STEP 2: FAISS KNN Refinement + Gaussian Smoothing Sweep")
    print("=" * 80)

    rows, header = load_csv(anommax_csv)
    print(f"  Input: {anommax_csv} ({len(rows)} segments)")

    embeddings, scores, video_ids, row_indices = build_segment_embeddings(
        rows, SCORE_COL_ANOMMAX, embed_dir
    )
    n = len(scores)
    n_videos = len(set(video_ids))
    print(f"  Valid: {n} segments, {n_videos} videos, dim={embeddings.shape[1]}")

    labels = parse_labels(labels_path, video_dir)
    print(f"  Labels: {len(labels)} videos")

    # Baseline
    orig_map = {idx: s for idx, s in zip(row_indices, scores)}
    best_bl_auc, best_bl_ap, best_bl_sig = 0, 0, 0
    for sigma in SIGMA_VALUES:
        auc, ap = evaluate_metrics(rows, orig_map, labels, sigma)
        if auc > best_bl_auc:
            best_bl_auc = auc
            best_bl_ap = ap
            best_bl_sig = sigma
    print(f"\n  Baseline (AnomMax, no KNN): σ={best_bl_sig}, AUC={best_bl_auc:.4f}, AP={best_bl_ap:.4f}")

    # Determine modes
    if modes is None:
        mode_configs = [("all", False, K_ALL), ("cross", True, K_CROSS)]
    else:
        mode_configs = []
        if "all" in modes:
            mode_configs.append(("all", False, K_ALL))
        if "cross" in modes:
            mode_configs.append(("cross", True, K_CROSS))

    all_results = []

    for mode_name, cross_video, k_values in mode_configs:
        print(f"\n{'='*60}")
        print(f"MODE: {'Cross-video only' if cross_video else 'All neighbors'}")
        print(f"{'='*60}")

        for k in k_values:
            print(f"  Computing K={k}...", flush=True)
            t0 = time.time()
            refined = faiss_knn_refine(embeddings, scores, video_ids, k, cross_video)

            best_auc, best_ap, best_alpha, best_sigma = 0, 0, 0, 0
            for alpha in ALPHA_VALUES:
                final = alpha * scores + (1.0 - alpha) * refined
                score_map = {idx: sc for idx, sc in zip(row_indices, final)}
                for sigma in SIGMA_VALUES:
                    auc, ap = evaluate_metrics(rows, score_map, labels, sigma)
                    if auc > best_auc:
                        best_auc = auc
                        best_ap = ap
                        best_alpha = alpha
                        best_sigma = sigma

            dt = time.time() - t0
            all_results.append({
                "mode": mode_name, "k": k, "alpha": best_alpha,
                "sigma": best_sigma, "auc": best_auc, "ap": best_ap
            })
            print(f"  K={k:<4}  best: α={best_alpha:.2f}, σ={best_sigma}, AUC={best_auc:.4f}, AP={best_ap:.4f}  ({dt:.1f}s)",
                  flush=True)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for mode_name, label in [("all", "All neighbors"), ("cross", "Cross-video only")]:
        mode_results = [r for r in all_results if r["mode"] == mode_name]
        if not mode_results:
            continue
        mode_results.sort(key=lambda x: x["auc"], reverse=True)
        best = mode_results[0]
        print(f"\n  {label}:")
        print(f"  {'K':<5} {'α':<6} {'σ':<6} {'AUC':>8} {'AP':>8}")
        print(f"  {'-'*36}")
        for r in mode_results:
            star = " ⭐" if r["auc"] == best["auc"] else ""
            print(f"  {r['k']:<5} {r['alpha']:<6.2f} {r['sigma']:<6} {r['auc']:>8.4f} {r['ap']:>8.4f}{star}")
        print(f"\n  Best: K={best['k']}, α={best['alpha']:.2f}, σ={best['sigma']} → AUC={best['auc']:.4f}, AP={best['ap']:.4f}")

    # Generate CSVs for best of each mode
    os.makedirs(output_dir, exist_ok=True)
    for mode_name in ["all", "cross"]:
        mode_results = [r for r in all_results if r["mode"] == mode_name]
        if not mode_results:
            continue
        mode_results.sort(key=lambda x: x["auc"], reverse=True)
        best = mode_results[0]

        refined = faiss_knn_refine(embeddings, scores, video_ids, best["k"], mode_name == "cross")
        final = best["alpha"] * scores + (1.0 - best["alpha"]) * refined

        out_name = f"scores_knn_{mode_name}_k{best['k']}.csv"
        out_path = os.path.join(output_dir, out_name)
        write_output_csv(rows, header, row_indices, final, out_path, "score_knn")
        print(f"\n  CSV: {out_path}")
        print(f"  Eval: python evaluate.py --csv {out_path} --scores score_knn --labels {labels_path} --sigma {best['sigma']}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Run Best (fixed params, no sweep)
# ═══════════════════════════════════════════════════════════════════════════════

# Best configs from ContextVAD sweep (UCF-Crime full 290)
BEST_CONFIGS = [
    {"name": "cross", "k": 15, "alpha": 0.65, "sigma": 700, "cross_video": True},
    {"name": "all",   "k": 35, "alpha": 0.70, "sigma": 350, "cross_video": False},
]


def run_best(base_csv, embed_dir, labels_path, output_dir, video_dir=None, configs=None):
    """Apply AnomMax + KNN with fixed best params. No sweep."""
    if not HAS_FAISS:
        print("ERROR: pip install faiss-cpu")
        return

    if configs is None:
        configs = BEST_CONFIGS

    print("\n" + "=" * 80)
    print("RUN BEST: AnomMax + KNN (fixed params)")
    print("=" * 80)

    # Step 1: AnomMax
    df = pd.read_csv(base_csv)
    print(f"  Input: {base_csv} ({len(df)} segments)")
    am_scores = apply_anommax(df, SCORE_COL_BASE, RAW_COL)
    df[SCORE_COL_ANOMMAX] = am_scores

    # Build embeddings
    rows = df.to_dict("records")
    embeddings, scores, video_ids, row_indices = build_segment_embeddings(
        rows, SCORE_COL_ANOMMAX, embed_dir
    )
    n_videos = len(set(video_ids))
    print(f"  Valid: {len(scores)} segments, {n_videos} videos")

    labels = parse_labels(labels_path, video_dir)

    for cfg in configs:
        print(f"\n  --- {cfg['name']}: K={cfg['k']}, α={cfg['alpha']}, σ={cfg['sigma']} ---")
        t0 = time.time()
        refined = faiss_knn_refine(embeddings, scores, video_ids, cfg["k"], cfg["cross_video"])
        final = cfg["alpha"] * scores + (1.0 - cfg["alpha"]) * refined
        dt = time.time() - t0

        score_map = {idx: sc for idx, sc in zip(row_indices, final)}
        auc, ap = evaluate_metrics(rows, score_map, labels, cfg["sigma"])
        print(f"  AUC: {auc:.4f}, AP: {ap:.4f}  ({dt:.1f}s)")

        # Save CSV
        os.makedirs(output_dir, exist_ok=True)
        out_name = f"scores_best_{cfg['name']}_k{cfg['k']}.csv"
        out_path = os.path.join(output_dir, out_name)
        header = list(df.columns)
        write_output_csv(rows, header, row_indices, final, out_path, f"score_knn_{cfg['name']}")
        print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ContextVAD Post-Processing: AnomMax + FAISS KNN Refinement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base_csv", required=True, help="Base scoring CSV (from llm_scorer.py)")
    parser.add_argument("--embed_dir", required=True, help="Directory with CLIP embedding .npy files")
    parser.add_argument("--labels", required=True, help="Path to labels/annotations file")
    parser.add_argument("--video_dir", default=None, help="Video directory (for discovering normal videos in XD-Violence)")
    parser.add_argument("--output_dir", default=".", help="Output directory for result CSVs")
    parser.add_argument("--step", default="all", choices=["all", "anommax", "knn", "best"],
                        help="Which step to run (default: all)")
    parser.add_argument("--mode", default=None,
                        help="KNN neighbor mode: 'all', 'cross', or 'all,cross' (default: both)")
    args = parser.parse_args()

    anommax_csv = os.path.join(args.output_dir, "scores_anommax.csv")

    if args.step == "best":
        run_best(args.base_csv, args.embed_dir, args.labels, args.output_dir, args.video_dir)
        return

    if args.step in ("all", "anommax"):
        run_anommax(args.base_csv, anommax_csv, args.labels, args.video_dir)

    if args.step in ("all", "knn"):
        if not os.path.exists(anommax_csv):
            print(f"AnomMax CSV not found, generating first...")
            run_anommax(args.base_csv, anommax_csv, args.labels, args.video_dir)
        modes = args.mode.split(",") if args.mode else None
        run_knn_sweep(anommax_csv, args.embed_dir, args.labels, args.output_dir, args.video_dir, modes)


if __name__ == "__main__":
    main()