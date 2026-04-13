#!/usr/bin/env python3
"""
evaluate.py — Frame-level AUC-ROC + AP evaluation for ContextVAD scoring CSVs.

Dataset-agnostic: auto-detects UCF-Crime vs XD-Violence label format.

Modes:
  1. Single CSV:      --csv scores.csv [--smooth 1000] [--normalize]
  2. Batch directory:  --dir output/ucf_crime/
  3. Smoothing sweep:  --csv scores.csv --sweep
  4. Per-video:        --csv scores.csv --smooth 1000 --per-video

Examples:
  python evaluate.py --csv scores.csv --labels labels.txt
  python evaluate.py --csv scores.csv --labels annotations.txt --video_dir videos/
  python evaluate.py --csv scores.csv --labels labels.txt --sweep
  python evaluate.py --dir output/ --labels labels.txt --sweep
"""
import argparse
import csv
import os
import sys

import numpy as np

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except ImportError:
    print("ERROR: pip install scikit-learn")
    sys.exit(1)

try:
    from scipy.ndimage import gaussian_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ─── Config ──────────────────────────────────────────────────────────────────
SWEEP_SIGMAS = list(range(0, 1250, 50))


# ─── Label parsing (auto-detect format) ─────────────────────────────────────

def parse_labels(path, video_dir=None):
    """Auto-detect label format and parse.
    
    UCF-Crime:    name category s1 e1 s2 e2  (always 6 fields)
    XD-Violence:  name s1 e1 [s2 e2 ...]     (variable fields, no category)
    
    Auto-detect: if field[1] of first data line is a digit → XD-Violence.
    """
    # Peek at first non-empty line to detect format
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                if parts[1].lstrip('-').isdigit():
                    # XD-Violence format
                    return _parse_xd_violence(path, video_dir)
                else:
                    # UCF-Crime format
                    return _parse_ucf_crime(path)
    return {}


def _parse_ucf_crime(path):
    """Parse UCF-Crime labels: name category s1 e1 s2 e2"""
    labels = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            name = parts[0]
            cat = parts[1]
            s1, e1, s2, e2 = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
            intervals = []
            if s1 != -1 and e1 != -1:
                intervals.append((s1, e1))
            if s2 != -1 and e2 != -1:
                intervals.append((s2, e2))
            labels[normalize_name(name)] = {"category": cat, "intervals": intervals}
    return labels


def _parse_xd_violence(path, video_dir=None):
    """Parse XD-Violence labels: name s1 e1 [s2 e2 ...]
    
    - Variable number of interval pairs
    - Some names have spurious .mp4 suffix → strip it
    - Normal videos not in annotations → discover from video_dir
    - All keys normalized (strip '=') for robust matching
    """
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
            labels[normalize_name(name)] = {"category": "anomalous", "intervals": intervals}
    
    # Discover normal videos from video_dir (not in annotations)
    if video_dir and os.path.isdir(video_dir):
        for f in os.listdir(video_dir):
            if f.endswith('.mp4'):
                stem = normalize_name(f[:-4])
                if stem not in labels:
                    labels[stem] = {"category": "normal", "intervals": []}
    
    return labels


def normalize_name(name):
    """Normalize video name for matching: strip '=' (Kaggle mangles v= to v-)."""
    return name.replace("=", "")


def extract_video_name(path_str):
    """Extract video stem from segment path like .../Abuse028_x264_segments/segment_0000.mp4
    
    Returns the normalized stem WITHOUT .mp4 extension for matching against labels dict.
    """
    parent = os.path.basename(os.path.dirname(path_str))
    stem = parent.replace("_segments", "")
    return normalize_name(stem)


# ─── Core evaluation ─────────────────────────────────────────────────────────

def evaluate(rows, annotations, score_col, smooth_sigma=None, normalize=False):
    """Build frame-level score/label arrays and compute AUC-ROC + AP."""
    video_data = {}
    for row in rows:
        raw = row.get(score_col, "")
        try:
            score = float(raw)
        except (ValueError, TypeError):
            score = 0.0
        if score < 0:
            score = 0.0
        try:
            start = int(row["start"])
            end = int(row["end"])
        except (ValueError, KeyError):
            continue
        
        vname = extract_video_name(row["path"])
        # Try both with and without .mp4 for UCF-Crime compatibility
        if vname not in annotations and vname + ".mp4" in annotations:
            vname = vname + ".mp4"
        if vname not in annotations:
            continue
        
        if vname not in video_data:
            video_data[vname] = {"segments": [], "max_frame": 0}
        video_data[vname]["segments"].append((start, end, score))
        video_data[vname]["max_frame"] = max(video_data[vname]["max_frame"], end)

    all_scores = []
    all_labels = []

    for vname in sorted(video_data.keys()):
        data = video_data[vname]
        mf = data["max_frame"]
        if mf == 0:
            continue
        scores = np.zeros(mf, dtype=np.float64)
        for s, e, sc in data["segments"]:
            end_f = min(e, mf)
            if s < end_f:
                scores[s:end_f] = sc
        if smooth_sigma and smooth_sigma > 0 and HAS_SCIPY:
            scores = gaussian_filter1d(scores, sigma=smooth_sigma)
        if normalize:
            smin, smax = scores.min(), scores.max()
            if smax > smin:
                scores = (scores - smin) / (smax - smin)
        gt = np.zeros(mf, dtype=int)
        for s_ann, e_ann in annotations[vname]["intervals"]:
            e_adj = min(e_ann, mf - 1)
            if s_ann <= e_adj:
                gt[s_ann : e_adj + 1] = 1
        all_scores.extend(scores.tolist())
        all_labels.extend(gt.tolist())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    if len(np.unique(all_labels)) < 2:
        return None, None, len(all_scores)
    auc = roc_auc_score(all_labels, all_scores)
    ap = average_precision_score(all_labels, all_scores)
    return auc, ap, len(all_scores)


def evaluate_per_video(rows, annotations, score_col, smooth_sigma=None, normalize=False):
    """Return per-video metrics as a dict: {vname: (auc, ap, n_frames, category, mean_score)}."""
    video_data = {}
    for row in rows:
        raw = row.get(score_col, "")
        try:
            score = float(raw)
        except (ValueError, TypeError):
            score = 0.0
        if score < 0:
            score = 0.0
        try:
            start = int(row["start"])
            end = int(row["end"])
        except (ValueError, KeyError):
            continue
        vname = extract_video_name(row["path"])
        if vname not in annotations and vname + ".mp4" in annotations:
            vname = vname + ".mp4"
        if vname not in annotations:
            continue
        if vname not in video_data:
            video_data[vname] = {"segments": [], "max_frame": 0}
        video_data[vname]["segments"].append((start, end, score))
        video_data[vname]["max_frame"] = max(video_data[vname]["max_frame"], end)

    results = {}
    for vname in sorted(video_data.keys()):
        data = video_data[vname]
        mf = data["max_frame"]
        if mf == 0:
            continue
        scores = np.zeros(mf, dtype=np.float64)
        for s, e, sc in data["segments"]:
            end_f = min(e, mf)
            if s < end_f:
                scores[s:end_f] = sc
        if smooth_sigma and smooth_sigma > 0 and HAS_SCIPY:
            scores = gaussian_filter1d(scores, sigma=smooth_sigma)
        if normalize:
            smin, smax = scores.min(), scores.max()
            if smax > smin:
                scores = (scores - smin) / (smax - smin)
        gt = np.zeros(mf, dtype=int)
        for s_ann, e_ann in annotations[vname]["intervals"]:
            e_adj = min(e_ann, mf - 1)
            if s_ann <= e_adj:
                gt[s_ann : e_adj + 1] = 1

        cat = annotations[vname]["category"]
        n_unique = len(np.unique(gt))
        if n_unique < 2:
            mean_score = float(scores.mean())
            results[vname] = (None, None, mf, cat, mean_score)
        else:
            auc = roc_auc_score(gt, scores)
            ap = average_precision_score(gt, scores)
            results[vname] = (auc, ap, mf, cat, float(scores.mean()))
    return results


# ─── CSV loading helpers ─────────────────────────────────────────────────────

def load_csv(path):
    """Load CSV, deduplicate by (path, start, end), return (rows, header)."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        for row in reader:
            rows.append(row)
    seen = set()
    unique = []
    for row in rows:
        key = (row.get("path", ""), row.get("start", ""), row.get("end", ""))
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique, header


def auto_detect_score_cols(header):
    """Return all column names starting with 'score'."""
    return [c for c in header if c.startswith("score")]


# ─── Sweep ────────────────────────────────────────────────────────────────────

def sweep(rows, annotations, score_col, normalize=False):
    """Sweep through fixed sigma list, return list of (sigma, auc, ap)."""
    results = []
    for sigma in SWEEP_SIGMAS:
        auc, ap, _ = evaluate(rows, annotations, score_col, sigma, normalize)
        if auc is None:
            continue
        results.append((sigma, auc, ap))
    return results


# ─── Display helpers ──────────────────────────────────────────────────────────

def print_sweep_table(results):
    """Print sweep results as a table with best markers."""
    print(f"\n{'σ':<12} {'AUC-ROC':>10} {'AP':>10}")
    print("-" * 34)
    best_auc = max(r[1] for r in results if r[1] is not None)
    best_ap = max(r[2] for r in results if r[2] is not None)
    for sigma, auc, ap in results:
        auc_mark = " ⭐" if auc == best_auc else ""
        ap_mark = " ⭐" if ap == best_ap else ""
        print(f"{sigma:<12} {auc:>10.4f}{auc_mark} {ap:>10.4f}{ap_mark}")
    print()


def print_per_video(pv_results):
    """Print per-video results sorted by AUC (worst first)."""
    with_auc = [(v, a, ap, nf, cat, ms) for v, (a, ap, nf, cat, ms) in pv_results.items() if a is not None]
    no_auc = [(v, a, ap, nf, cat, ms) for v, (a, ap, nf, cat, ms) in pv_results.items() if a is None]

    with_auc.sort(key=lambda x: x[1])

    print(f"\n{'Video':<45} {'Category':<12} {'AUC':>8} {'AP':>8} {'MeanScore':>10} {'Frames':>8}")
    print("-" * 95)

    for vname, auc, ap, nf, cat, ms in with_auc:
        ap_s = f"{ap:.4f}" if ap is not None else "N/A"
        print(f"{vname:<45} {cat:<12} {auc:>8.4f} {ap_s:>8} {ms:>10.4f} {nf:>8}")

    if no_auc:
        print(f"\n--- Normal-only videos (no anomaly frames, AUC=N/A) ---")
        no_auc.sort(key=lambda x: x[5], reverse=True)
        for vname, _, _, nf, cat, ms in no_auc:
            print(f"{vname:<45} {cat:<12} {'N/A':>8} {'N/A':>8} {ms:>10.4f} {nf:>8}")

    print()


# ─── Mode: single CSV ────────────────────────────────────────────────────────

def mode_single(args, annotations):
    """Evaluate a single CSV."""
    if not os.path.exists(args.csv):
        print(f"ERROR: CSV not found: {args.csv}")
        sys.exit(1)

    rows, header = load_csv(args.csv)

    if args.scores:
        score_cols = [s.strip() for s in args.scores.split(",")]
    else:
        score_cols = auto_detect_score_cols(header)

    if not score_cols:
        print(f"ERROR: No score columns found. Header: {header}")
        sys.exit(1)

    n_vids = len(set(extract_video_name(r["path"]) for r in rows))
    csv_name = os.path.basename(args.csv)
    print(f"CSV:        {csv_name}")
    print(f"Segments:   {len(rows)}")
    print(f"Videos:     {n_vids}")
    print(f"Columns:    {score_cols}")
    if args.smooth:
        print(f"Smoothing:  σ={args.smooth}")
    if args.normalize:
        print(f"Normalize:  per-video min-max")

    # Per-video mode
    if args.per_video:
        for col in score_cols:
            if col not in header:
                print(f"\n{col}: MISSING")
                continue
            print(f"\n=== {col} (σ={args.smooth or 0}) ===")
            pv = evaluate_per_video(rows, annotations, col, args.smooth, args.normalize)
            print_per_video(pv)
        return

    # Sweep mode
    if args.sweep:
        for col in score_cols:
            if col not in header:
                print(f"\n{col}: MISSING")
                continue
            print(f"\n=== Sweep: {col} ===")
            results = sweep(rows, annotations, col, args.normalize)
            print_sweep_table(results)
            best_auc_r = max(results, key=lambda r: r[1])
            best_ap_r = max(results, key=lambda r: r[2])
            print(f"  Best AUC: σ={best_auc_r[0]} → AUC={best_auc_r[1]:.4f}")
            print(f"  Best AP:  σ={best_ap_r[0]} → AP={best_ap_r[2]:.4f}")
        return

    # Default: single evaluation
    print()
    print(f"{'Column':<25} {'AUC-ROC':>10} {'AP':>10} {'Frames':>10}")
    print("-" * 60)
    for col in score_cols:
        if col not in header:
            print(f"{col:<25} {'MISSING':>10}")
            continue
        auc, ap, nf = evaluate(rows, annotations, col, args.smooth, args.normalize)
        auc_s = f"{auc:.4f}" if auc is not None else "N/A"
        ap_s = f"{ap:.4f}" if ap is not None else "N/A"
        print(f"{col:<25} {auc_s:>10} {ap_s:>10} {nf:>10}")
    print()


# ─── Mode: batch directory ───────────────────────────────────────────────────

def mode_batch(args, annotations):
    """Evaluate all CSVs in a directory."""
    if not os.path.isdir(args.dir):
        print(f"ERROR: Directory not found: {args.dir}")
        sys.exit(1)

    csv_files = sorted([f for f in os.listdir(args.dir) if f.endswith(".csv")])
    if not csv_files:
        print(f"No CSV files found in {args.dir}")
        return

    print(f"Found {len(csv_files)} CSVs in {args.dir}\n")

    all_results = []

    for fname in csv_files:
        fpath = os.path.join(args.dir, fname)
        rows, header = load_csv(fpath)
        score_cols = auto_detect_score_cols(header)
        n_vids = len(set(extract_video_name(r["path"]) for r in rows))

        print(f"=== {fname} ({len(rows)} segs, {n_vids} vids, cols: {score_cols}) ===")

        for col in score_cols:
            if args.sweep:
                results = sweep(rows, annotations, col, args.normalize)
                best_auc = max(results, key=lambda r: r[1])
                best_ap = max(results, key=lambda r: r[2])
                print(f"  {col:<22} best AUC: σ={best_auc[0]:<6} AUC={best_auc[1]:.4f}  |  best AP: σ={best_ap[0]:<6} AP={best_ap[2]:.4f}")
                all_results.append({
                    "file": fname, "col": col,
                    "sigma_auc": best_auc[0], "auc": best_auc[1],
                    "sigma_ap": best_ap[0], "ap": best_ap[2],
                    "n_vids": n_vids,
                })
            else:
                sigma = args.smooth or 0
                auc, ap, nf = evaluate(rows, annotations, col, sigma, args.normalize)
                auc_s = f"{auc:.4f}" if auc is not None else "N/A"
                ap_s = f"{ap:.4f}" if ap is not None else "N/A"
                tag = f"σ={sigma}" if sigma else "raw"
                print(f"  {col:<22} {tag:<10} AUC={auc_s} AP={ap_s}")
                if auc is not None:
                    all_results.append({
                        "file": fname, "col": col,
                        "sigma_auc": sigma, "auc": auc,
                        "sigma_ap": sigma, "ap": ap,
                        "n_vids": n_vids,
                    })
        print()

    # Consolidated ranking
    if all_results:
        all_results.sort(key=lambda x: x["auc"], reverse=True)
        print("=" * 100)
        print("TOP RESULTS BY AUC-ROC")
        print("=" * 100)
        print(f"{'Rank':<5} {'File':<45} {'Column':<22} {'σ':<8} {'AUC':>8} {'AP':>8}")
        print("-" * 100)
        for i, r in enumerate(all_results[:30]):
            print(f"{i+1:<5} {r['file']:<45} {r['col']:<22} {r['sigma_auc']:<8} {r['auc']:>8.4f} {r['ap']:>8.4f}")

        print()
        all_results.sort(key=lambda x: x["ap"], reverse=True)
        print("=" * 100)
        print("TOP RESULTS BY AP")
        print("=" * 100)
        print(f"{'Rank':<5} {'File':<45} {'Column':<22} {'σ':<8} {'AUC':>8} {'AP':>8}")
        print("-" * 100)
        for i, r in enumerate(all_results[:30]):
            print(f"{i+1:<5} {r['file']:<45} {r['col']:<22} {r['sigma_ap']:<8} {r['auc']:>8.4f} {r['ap']:>8.4f}")
        print()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Frame-level AUC-ROC + AP evaluation for ContextVAD scoring CSVs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  --csv FILE                Evaluate a single CSV
  --dir DIR                 Batch-evaluate all CSVs in directory
  --sweep                   Smoothing sigma sweep (0 to 1200, step 50)
  --per-video               Per-video AUC/AP breakdown (requires --csv)

Examples:
  python evaluate.py --csv scores.csv --labels labels.txt
  python evaluate.py --csv scores.csv --labels annotations.txt --video_dir videos/
  python evaluate.py --csv scores.csv --labels labels.txt --sweep
  python evaluate.py --dir output/ --labels labels.txt --sweep
        """,
    )
    parser.add_argument("--csv", help="Path to a single scoring CSV file")
    parser.add_argument("--dir", help="Path to directory of scoring CSVs (batch mode)")
    parser.add_argument("--scores", default=None, help="Comma-separated score column names (auto-detect if omitted)")
    parser.add_argument("--smooth", type=float, default=None, help="Fixed Gaussian smoothing sigma in frames")
    parser.add_argument("--normalize", action="store_true", help="Per-video min-max normalization")
    parser.add_argument("--sweep", action="store_true", help="Smoothing sigma sweep")
    parser.add_argument("--per-video", action="store_true", help="Per-video AUC/AP breakdown")
    parser.add_argument("--labels", required=True, help="Path to labels/annotations file")
    parser.add_argument("--video_dir", default=None, help="Video directory (for discovering normal videos in XD-Violence)")
    parser.add_argument("--sigma", type=float, default=None, help="Alias for --smooth")
    args = parser.parse_args()

    # Allow --sigma as alias for --smooth
    if args.sigma and not args.smooth:
        args.smooth = args.sigma

    if not args.csv and not args.dir:
        parser.print_help()
        print("\nERROR: Must specify --csv or --dir")
        sys.exit(1)
    if args.csv and args.dir:
        print("ERROR: Cannot use both --csv and --dir")
        sys.exit(1)
    if args.per_video and not args.csv:
        print("ERROR: --per-video requires --csv")
        sys.exit(1)
    if (args.sweep or args.smooth) and not HAS_SCIPY:
        print("ERROR: --sweep/--smooth requires scipy. pip install scipy")
        sys.exit(1)
    if not os.path.exists(args.labels):
        print(f"ERROR: Labels not found: {args.labels}")
        sys.exit(1)

    annotations = parse_labels(args.labels, args.video_dir)
    n_anomalous = sum(1 for v in annotations.values() if v["intervals"])
    n_normal = sum(1 for v in annotations.values() if not v["intervals"])
    print(f"Labels:     {len(annotations)} videos ({n_anomalous} anomalous, {n_normal} normal)")

    if args.csv:
        mode_single(args, annotations)
    elif args.dir:
        mode_batch(args, annotations)


if __name__ == "__main__":
    main()