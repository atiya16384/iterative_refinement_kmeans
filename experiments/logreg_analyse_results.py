# LOGREG_stats_tests.py
# Paired statistical tests across approaches for the logreg_precision experiments

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy import stats

# ---------------- config ----------------
IN_FILES = [
    "../Results/uniform_results.csv",
    "../Results/gaussian_results.csv",
    "../Results/blobs_results.csv",
    "../Results/susy_results.csv",
]
OUTDIR = Path("../Results/SUMMARY_LOGPREC_STATS")

# Approach labels as produced by your runner
APP_SINGLE = "single(f32)"
APP_DOUBLE = "double(f64)"
APP_HYBRID = "hybrid(f32→f64)"

# Metrics to look for (only those present in the CSVs will be processed)
METRIC_CANDIDATES = ["time_sec", "roc_auc", "pr_auc", "logloss"]

# ---------------- helpers ----------------
def _ensure_outdir():
    OUTDIR.mkdir(parents=True, exist_ok=True)

def _read_frames(paths):
    frames = []
    for p in paths:
        p = Path(p)
        if p.exists() and p.stat().st_size > 0:
            df = pd.read_csv(p)
            if not df.empty:
                frames.append(df)
    if not frames:
        raise FileNotFoundError("No non-empty results CSVs found.")
    return frames

def _present_metrics(df):
    return [m for m in METRIC_CANDIDATES if m in df.columns]

def _agg_over_repeats(df, metrics):
    """
    Aggregate per (dataset, approach, *params*) over repeats to get one row per config.
    """
    param_cols = [c for c in [
        "penalty","alpha","lambda","solver","max_iter","tol","max_iter_single","C"
    ] if c in df.columns]

    group_cols = ["dataset", "approach"] + param_cols
    agg = df.groupby(group_cols, as_index=False)[metrics].mean()
    return agg, param_cols

def _align_pairs(agg, param_cols, metric, baseline_label):
    """
    Create aligned vectors (x_hybrid, y_baseline) for a given metric and baseline.
    One entry per hyper-parameter config that exists for BOTH approaches.
    """
    # pivot to columns by approach for the metric
    pivot_cols = ["dataset"] + param_cols
    piv = agg.pivot_table(index=pivot_cols, columns="approach", values=metric, aggfunc="mean")

    # we need both hybrid and the baseline
    needed = [APP_HYBRID, baseline_label]
    if not all(col in piv.columns for col in needed):
        return pd.DataFrame(columns=pivot_cols + ["hybrid", "baseline"])

    sub = piv.dropna(subset=needed).reset_index()
    sub = sub.rename(columns={APP_HYBRID: "hybrid", baseline_label: "baseline"})
    return sub

def _paired_effects(x, y):
    """
    x, y: matched arrays (hybrid, baseline)
    Returns dict with paired t-test, Wilcoxon, Cohen's d for paired samples,
    mean/median deltas and counts.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = x - y
    n = d.size

    # Paired t-test
    t_stat, t_p = stats.ttest_rel(x, y, nan_policy="omit")

    # Wilcoxon signed-rank (use 'pratt' so zeros are handled)
    try:
        w_stat, w_p = stats.wilcoxon(x, y, zero_method="pratt", alternative="two-sided")
    except ValueError:
        # Not enough nonzero differences; fallback
        w_stat, w_p = np.nan, np.nan

    # Cohen's d for paired (uses sd of differences)
    sd_d = np.std(d, ddof=1)
    cohen_d = np.mean(d) / sd_d if sd_d > 0 else np.nan

    return {
        "n_pairs": int(n),
        "mean_hybrid": float(np.nanmean(x)),
        "mean_baseline": float(np.nanmean(y)),
        "mean_delta": float(np.nanmean(d)),
        "median_delta": float(np.nanmedian(d)),
        "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
        "t_pvalue": float(t_p) if np.isfinite(t_p) else np.nan,
        "wilcoxon_stat": float(w_stat) if np.isfinite(w_stat) else np.nan,
        "wilcoxon_pvalue": float(w_p) if np.isfinite(w_p) else np.nan,
        "cohens_d_paired": float(cohen_d) if np.isfinite(cohen_d) else np.nan,
    }

def _summarize_for_dataset(df_ds, metrics, param_cols):
    """
    For one dataset: run tests for each metric, comparing (hybrid vs single) and (hybrid vs double).
    Return a DataFrame summary.
    """
    rows = []
    for metric in metrics:
        # Hybrid vs single
        tab_s = _align_pairs(df_ds, param_cols, metric, APP_SINGLE)
        if not tab_s.empty:
            eff = _paired_effects(tab_s["hybrid"].values, tab_s["baseline"].values)
            rows.append({
                "dataset": df_ds["dataset"].iloc[0],
                "metric": metric,
                "comparison": "hybrid_vs_single",
                **eff
            })

        # Hybrid vs double
        tab_d = _align_pairs(df_ds, param_cols, metric, APP_DOUBLE)
        if not tab_d.empty:
            eff = _paired_effects(tab_d["hybrid"].values, tab_d["baseline"].values)
            rows.append({
                "dataset": df_ds["dataset"].iloc[0],
                "metric": metric,
                "comparison": "hybrid_vs_double",
                **eff
            })
    return pd.DataFrame(rows)

def main(paths):
    _ensure_outdir()
    frames = _read_frames(paths)
    df = pd.concat(frames, ignore_index=True)

    # keep successful runs only
    if "approach" in df.columns:
        df = df[df["approach"] != "ERROR"].copy()

    metrics = _present_metrics(df)
    if not metrics:
        raise RuntimeError("No known metric columns present in data.")

    # Aggregate over repeats for stable paired comparisons
    agg, param_cols = _agg_over_repeats(df, metrics)

    all_summaries = []
    for ds, ds_df in agg.groupby("dataset", as_index=False):
        summary = _summarize_for_dataset(ds_df, metrics, param_cols)
        if not summary.empty:
            all_summaries.append(summary)
            # save per-dataset CSV
            out = OUTDIR / f"{ds}_stats_summary.csv"
            summary.to_csv(out, index=False)
            print(f"Wrote: {out}")

    if all_summaries:
        big = pd.concat(all_summaries, ignore_index=True)
        out_all = OUTDIR / "ALL_DATASETS_stats_summary.csv"
        big.to_csv(out_all, index=False)
        print(f"Wrote: {out_all}")
    else:
        print("No overlapping (hybrid vs baseline) pairs found to test.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paired stats for hybrid vs baselines on logreg results.")
    parser.add_argument("--inputs", nargs="*", default=IN_FILES,
                        help="List of CSV result files to include.")
    args = parser.parse_args()
    main(args.inputs)
