# kmeans_analyse_results.py
from pathlib import Path
import numpy as np
import pandas as pd

# Stats
try:
    from scipy.stats import ttest_rel, ttest_ind, wilcoxon
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# =========================================================
# Metric aliasing (logical -> actual CSV column candidates)
# =========================================================
METRIC_ALIASES = {
    "PeakMB":  ["PeakMB", "Memory_MB", "Peak_MB", "RSS_Peak_MB"],
    "Time":    ["Time"],
    "Inertia": ["Inertia"],
    # TrafficRel is derived (no alias)
}

def _resolve_metrics_present(df: pd.DataFrame, desired=("Time", "Inertia", "PeakMB")):
    """
    Return (present_logical_metrics, alias_map).
    alias_map maps logical metric name -> actual df column used.
    """
    present = []
    alias_map = {}
    for logical in desired:
        for cand in METRIC_ALIASES.get(logical, [logical]):
            if cand in df.columns:
                present.append(logical)
                alias_map[logical] = cand
                break
    return tuple(present), alias_map


# ==========================================
# Derived metric: estimated memory traffic
# ==========================================
def _maybe_add_trafficrel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add TrafficRel (estimated memory traffic relative to Double) if we have
    the needed columns. Model:
        traffic_abs ≈ TotalIter - 0.5 * ItersSingle
    because a float32 iteration moves ~half as many bytes as float64.
    We normalize by the cohort's Double TotalIter (median).

    Cohort keys used for normalization:
        (DatasetName, NumClusters, Mode [, tolerance_single])
    """
    need = {"ItersSingle", "ItersDouble", "Suite", "DatasetName", "NumClusters", "Mode"}
    if not need.issubset(df.columns):
        return df

    d = df.copy()
    d["ItersSingle"] = d["ItersSingle"].fillna(0.0).astype(float)
    d["ItersDouble"] = d["ItersDouble"].fillna(0.0).astype(float)
    d["TotalIter"]   = d["ItersSingle"] + d["ItersDouble"]

    cohort_keys = ["DatasetName", "NumClusters", "Mode"]
    if "tolerance_single" in d.columns:
        cohort_keys.append("tolerance_single")

    base = (
        d[d["Suite"] == "Double"]
        .groupby(cohort_keys, as_index=False)["TotalIter"]
        .median()
        .rename(columns={"TotalIter": "Tdouble"})
    )
    d = d.merge(base, on=cohort_keys, how="left")

    # Double-equivalent traffic for each row
    d["TrafficAbs"] = d["TotalIter"] - 0.5 * d["ItersSingle"]
    # Relative to the *matching* Double cohort (can be NaN for non-matching cohorts)
    d["TrafficRel"] = d["TrafficAbs"] / d["Tdouble"]

    return d


# ======================
# Markdown export helper
# ======================
def _export_md(per_df: pd.DataFrame, outstem: Path) -> None:
    outstem.parent.mkdir(parents=True, exist_ok=True)
    with open(outstem.with_suffix(".md"), "w", encoding="utf-8") as f:
        f.write(per_df.to_markdown(index=False))


# =========================
# Schema / Peek utilities
# =========================
def peek_csv_fields(
    csv_file: str,
    expected_metrics=("Time", "Inertia", "PeakMB"),  # logical names
    candidate_compare=("Hybrid", "Adaptive", "MiniBatch+Full", "MixedPerCluster"),
    candidate_baselines=("Double", "Single"),
    sample_rows=None,
):
    """
    Print & return schema info; also derives TrafficRel if possible.
    """
    kw = {}
    if sample_rows is not None:
        kw["nrows"] = int(sample_rows)

    df = pd.read_csv(csv_file, **kw)
    df = _maybe_add_trafficrel(df)

    metrics_present, alias_map = _resolve_metrics_present(df, expected_metrics)
    if "TrafficRel" in df.columns:
        metrics_present = tuple(list(metrics_present) + ["TrafficRel"])

    cols = list(df.columns)

    print(f"\n[SCHEMA] {csv_file}")
    print(f"  #rows={len(df):,}, #cols={len(cols)}")
    print("  columns:", cols)

    key_candidates = ["DatasetName", "DatasetSize", "NumClusters"]
    present_keys = [k for k in key_candidates if k in df.columns]
    print("  group keys present:", present_keys or "(none)")

    suites_present = list(df["Suite"].unique()) if "Suite" in df.columns else []
    print("  Suite present:", suites_present or "(Suite column missing)")

    present_compares = [s for s in candidate_compare if s in suites_present]
    present_baselines = [s for s in candidate_baselines if s in suites_present]
    print("  compare candidates found:", present_compares)
    print("  baselines found:", present_baselines)

    print("  metrics found (logical):", metrics_present or "(none of expected)")
    if metrics_present and "Suite" in df.columns:
        alias_for_print = dict(alias_map)
        alias_for_print["TrafficRel"] = "TrafficRel"
        for logical in metrics_present:
            col = alias_for_print.get(logical, logical)
            cnt = df.groupby("Suite")[col].apply(lambda s: s.notna().sum()).to_dict()
            print(f"  non-null by Suite for {logical}[{col}]:", cnt)

    return {
        "df": df,  # include for reuse if you want
        "columns": cols,
        "present_keys": present_keys,
        "suites_present": suites_present,
        "present_compares": present_compares,
        "present_baselines": present_baselines,
        "metrics_present": metrics_present,
    }


# ===========================================
# Utility: find a repeat column for pairing
# ===========================================
_REPEAT_CANDIDATES = ["Repeat", "repeat", "rep", "seed", "random_state", "RandomState"]

def _find_repeat_col(df: pd.DataFrame):
    for c in _repeat_candidates_with_index(df):
        if c in df.columns:
            return c
    return None

def _repeat_candidates_with_index(df: pd.DataFrame):
    # Also allow "index within cohort" pairing if a column named "RunIdx" exists
    return _REPEAT_CANDIDATES + ["RunIdx"]


# =======================================================
# Core analysis (comparisons + statistical validation)
# =======================================================
def analyze_experiment_per_dataset(
    csv_file: str,
    metrics=("Time", "Inertia", "PeakMB", "TrafficRel"),  # logical names
    compare_suite=("Hybrid", "Adaptive", "MiniBatch+Full", "MixedPerCluster"),
    baselines=("Double", "Single"),
    outdir="../Results/SUMMARY",
    label=None,
):
    """
    For each dataset (DatasetName/Size, NumClusters):
      - average repeats per Suite for the "comparisons" table,
      - compute statistical tests on paired repeats when possible:
          * Paired t-test (fallback to Welch t-test if not pairable)
          * Wilcoxon signed-rank (only if paired and >= 5 pairs)
          * Cohen's d (paired; mean(diff)/std(diff))
    Returns dict with written paths.
    """
    df_raw = pd.read_csv(csv_file)
    if df_raw.empty:
        return {"_note": "empty file"}

    # Derive TrafficRel if possible
    df = _maybe_add_trafficrel(df_raw)

    # Group keys (prefer DatasetName; fall back to DatasetSize)
    base_keys = ["DatasetName", "NumClusters"]
    if not set(base_keys).issubset(df.columns):
        base_keys = [c for c in ["DatasetSize", "NumClusters"] if c in df.columns]
        if not base_keys:
            return {"_note": "no suitable grouping columns (DatasetName/DatasetSize, NumClusters)"}

    # Resolve metrics present
    desired_no_tr = [m for m in metrics if m != "TrafficRel"]
    metrics_present, alias_map = _resolve_metrics_present(df, desired_no_tr)
    if "TrafficRel" in df.columns and "TrafficRel" in metrics:
        metrics_present = tuple(list(metrics_present) + ["TrafficRel"])
    if not metrics_present:
        return {"_note": "no desired metrics present"}

    # Suites present
    compare_suites = [compare_suite] if isinstance(compare_suite, str) else list(compare_suite)
    suites_in_file = set(df["Suite"].unique().tolist()) if "Suite" in df.columns else set()
    suites_present = [s for s in compare_suites if s in suites_in_file]
    if not suites_present:
        return {"_note": f"none of {compare_suites} found in file"}

    outdir = Path(outdir)
    stem_label = label or Path(csv_file).stem

    # ---------- Build the "comparisons" table (mean per suite) ----------
    comp_rows = []
    for ds_keys, sub in df.groupby(base_keys, as_index=False):
        ds_row = ds_keys.iloc[0].to_dict() if isinstance(ds_keys, pd.DataFrame) else dict(zip(base_keys, ds_keys))
        suites_here = set(sub["Suite"].unique().tolist())
        if not (set(suites_present) & suites_here):
            continue

        for logical_metric in metrics_present:
            col = logical_metric if logical_metric == "TrafficRel" else alias_map.get(logical_metric, logical_metric)
            if col not in sub.columns:
                continue

            per_suite = (
                sub.groupby(["Suite"], as_index=False)[col]
                   .mean()
                   .set_index("Suite")[col]
            )

            for baseline in baselines:
                if baseline not in suites_here or baseline not in per_suite.index:
                    continue
                for cs in suites_present:
                    if cs == baseline or cs not in suites_here or cs not in per_suite.index:
                        continue

                    b_val = float(per_suite.loc[baseline])
                    c_val = float(per_suite.loc[cs])
                    if not np.isfinite(b_val) or not np.isfinite(c_val) or b_val == 0:
                        continue

                    rel = c_val / b_val
                    imp = (b_val - c_val) / b_val * 100.0

                    comp_rows.append({
                        **ds_row,
                        "metric": logical_metric,
                        "baseline": baseline,
                        "compare_suite": cs,
                        "baseline_value": b_val,
                        "compare_value": c_val,
                        "Rel": rel,
                        "Improvement_%": imp,
                        "n_pairs": int(sub[sub["Suite"] == baseline][col].shape[0]),
                    })

    if comp_rows:
        comp_df = pd.DataFrame(comp_rows)
        _export_md(comp_df, outdir / f"{stem_label}__all_comparisons")
    else:
        comp_df = pd.DataFrame()

    # ---------- Statistical validation table ----------
    stats_rows = []
    # Cohort granularity for pairing: include optional columns if present
    optional_keys = []
    for cand in ["Mode", "Cap", "tolerance_single"]:
        if cand in df.columns:
            optional_keys.append(cand)
    cohort_keys = base_keys + optional_keys

    repeat_col = _find_repeat_col(df)

    for ds_keys, sub in df.groupby(cohort_keys, as_index=False):
        if isinstance(ds_keys, pd.DataFrame):
            ds_row_base = ds_keys.iloc[0].to_dict()
        else:
            ds_row_base = dict(zip(cohort_keys, ds_keys))

        suites_here = set(sub["Suite"].unique().tolist())
        for baseline in baselines:
            if baseline not in suites_here:
                continue
            for cs in suites_present:
                if cs == baseline or cs not in suites_here:
                    continue

                for logical_metric in metrics_present:
                    col = logical_metric if logical_metric == "TrafficRel" else alias_map.get(logical_metric, logical_metric)
                    if col not in sub.columns:
                        continue

                    a = sub[sub["Suite"] == baseline][[col] + ([repeat_col] if repeat_col else [])].dropna(subset=[col]).copy()
                    b = sub[sub["Suite"] == cs][[col] + ([repeat_col] if repeat_col else [])].dropna(subset=[col]).copy()

                    if a.empty or b.empty:
                        continue

                    # Pair runs if a repeat column exists; else fall back to unpaired
                    paired = False
                    x = y = None
                    if repeat_col and repeat_col in a.columns and repeat_col in b.columns:
                        merged = pd.merge(a, b, on=repeat_col, how="inner", suffixes=("_base", "_cmp"))
                        x = merged[f"{col}_base"].to_numpy()
                        y = merged[f"{col}_cmp"].to_numpy()
                        # keep only finite
                        mask = np.isfinite(x) & np.isfinite(y)
                        x, y = x[mask], y[mask]
                        paired = (len(x) >= 2)   # need >=2 for stable stats and effect size
                    else:
                        # align by order if counts match and >1 (soft-pair); else unpaired
                        if len(a) == len(b) and len(a) >= 2:
                            x = a[col].to_numpy()
                            y = b[col].to_numpy()
                            mask = np.isfinite(x) & np.isfinite(y)
                            x, y = x[mask], y[mask]
                            paired = (len(x) >= 2)
                        else:
                            paired = False

                    # Compute summary %
                    # lower is better for all our metrics (Time, PeakMB, TrafficRel, Inertia in practice)
                    mean_base = float(np.nanmean(a[col])) if len(a) else np.nan
                    mean_cmp  = float(np.nanmean(b[col])) if len(b) else np.nan
                    rel = (mean_cmp / mean_base) if (np.isfinite(mean_base) and mean_base != 0) else np.nan
                    impr = ((mean_base - mean_cmp) / mean_base * 100.0) if (np.isfinite(mean_base) and mean_base != 0) else np.nan

                    # Stats
                    t_stat = np.nan
                    t_p    = np.nan
                    w_stat = np.nan
                    w_p    = np.nan
                    cohend = np.nan
                    test_used = "paired" if paired else "unpaired"

                    if _HAVE_SCIPY:
                        try:
                            if paired:
                                # paired t-test
                                res = ttest_rel(x, y, nan_policy="omit")
                                t_stat, t_p = float(res.statistic), float(res.pvalue)

                                # Wilcoxon (requires at least 5 pairs and non-all-equal diff)
                                diffs = y - x
                                if len(diffs) >= 5 and not np.allclose(diffs, diffs[0]):
                                    try:
                                        w = wilcoxon(diffs, zero_method="wilcox", correction=False, alternative="two-sided", mode="auto")
                                        w_stat, w_p = float(w.statistic), float(w.pvalue)
                                    except Exception:
                                        w_stat, w_p = np.nan, np.nan

                                # Cohen's d for paired (mean of differences / std of differences)
                                if len(diffs) >= 2 and np.nanstd(diffs, ddof=1) > 0:
                                    cohend = float(np.nanmean(diffs) / np.nanstd(diffs, ddof=1))
                            else:
                                # Welch's t-test (unpaired)
                                res = ttest_ind(
                                    a[col].to_numpy(), b[col].to_numpy(),
                                    equal_var=False, nan_policy="omit"
                                )
                                t_stat, t_p = float(res.statistic), float(res.pvalue)
                                test_used = "welch"
                        except Exception:
                            pass

                    stats_rows.append({
                        **ds_row_base,
                        "metric": logical_metric,
                        "baseline": baseline,
                        "compare_suite": cs,
                        "mean_baseline": mean_base,
                        "mean_compare":  mean_cmp,
                        "Rel": rel,
                        "Improvement_%": impr,
                        "paired_mode": test_used,
                        "n_base": int(len(a)),
                        "n_cmp":  int(len(b)),
                        "t_stat": t_stat,
                        "t_p":    t_p,
                        "wilcoxon_stat": w_stat,
                        "wilcoxon_p":    w_p,
                        "cohens_d": cohend,
                    })

    if stats_rows:
        stats_df = pd.DataFrame(stats_rows)

        # Order columns for readability
        lead = [
            *cohort_keys, "metric", "baseline", "compare_suite",
            "mean_baseline", "mean_compare", "Rel", "Improvement_%",
            "paired_mode", "n_base", "n_cmp",
            "t_stat", "t_p", "wilcoxon_stat", "wilcoxon_p", "cohens_d",
        ]
        lead = [c for c in lead if c in stats_df.columns] + [c for c in stats_df.columns if c not in lead]

        stats_df = stats_df[lead]
        _export_md(stats_df, outdir / f"{stem_label}__stats")
    else:
        stats_df = pd.DataFrame()

    written = []
    if not comp_df.empty:
        written.append(str(outdir / f"{stem_label}__all_comparisons.md"))
    if not stats_df.empty:
        written.append(str(outdir / f"{stem_label}__stats.md"))

    if not written:
        return {"_note": "nothing written (no comparable suites/metrics)"}
    return {"written": written}


# ================
# Run all CSVs
# ================
if __name__ == "__main__":
    csv_files = [
        "../Results/hybrid_kmeans_Results_expA.csv",
        "../Results/hybrid_kmeans_Results_expB.csv",
        "../Results/hybrid_kmeans_Results_expC.csv",
        "../Results/hybrid_kmeans_Results_expD.csv",
        "../Results/hybrid_kmeans_Results_expE.csv",
        "../Results/hybrid_kmeans_Results_expF.csv",
    ]

    desired_compares = ("Hybrid", "Adaptive", "MiniBatch+Full", "MixedPerCluster")
    desired_baselines = ("Double", "Single")
    desired_metrics = ("Time", "Inertia", "PeakMB")  # TrafficRel added if derivable

    for f in csv_files:
        p = Path(f)
        if not p.exists() or p.stat().st_size == 0:
            print(f"[skip] {f} missing or empty")
            continue

        info = peek_csv_fields(
            f,
            expected_metrics=desired_metrics,
            candidate_compare=desired_compares,
            candidate_baselines=desired_baselines,
            sample_rows=None,
        )

        if not info["metrics_present"]:
            print(f"[skip] {f}: none of expected metrics {desired_metrics} (or TrafficRel) found")
            continue
        if not info["present_compares"]:
            print(f"[skip] {f}: none of desired compare suites {desired_compares} found in Suite column")
            continue
        if not info["present_baselines"]:
            print(f"[skip] {f}: none of desired baselines {desired_baselines} found in Suite column")
            continue

        print(f"\n==== ANALYZE {f} ====")
        res = analyze_experiment_per_dataset(
            f,
            metrics=tuple(info["metrics_present"]),           # includes TrafficRel if derivable
            compare_suite=tuple(info["present_compares"]),
            baselines=tuple(info["present_baselines"]),
            outdir="../Results/SUMMARY",
            label=Path(f).stem,
        )
        if "_note" in res:
            print(res["_note"])
        else:
            for path in res["written"]:
                print("  wrote:", path)

