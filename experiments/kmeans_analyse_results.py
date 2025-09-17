# kmeans_analyse_results.py
from pathlib import Path
import itertools
import numpy as np
import pandas as pd

# Optional libs
try:
    from scipy.stats import (
        ttest_rel, ttest_ind, wilcoxon,
        mannwhitneyu, ks_2samp,
    )
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

try:
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False

try:
    from sklearn.metrics import silhouette_score
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False


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
    Normalize by cohort Double median(TotalIter).
    Cohort keys: (DatasetName, NumClusters, Mode [, tolerance_single])
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

    d["TrafficAbs"] = d["TotalIter"] - 0.5 * d["ItersSingle"]
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
        "df": df,
        "columns": cols,
        "present_keys": present_keys,
        "suites_present": suites_present,
        "present_compares": present_compares,
        "present_baselines": present_baselines,
        "metrics_present": metrics_present,
    }


# ===========================================
# Repeat/Pairing helpers
# ===========================================
_REPEAT_CANDIDATES = ["Repeat", "repeat", "rep", "seed", "random_state", "RandomState"]

def _repeat_candidates_with_index(df: pd.DataFrame):
    return _REPEAT_CANDIDATES + ["RunIdx"]

def _find_repeat_col(df: pd.DataFrame):
    for c in _repeat_candidates_with_index(df):
        if c in df.columns:
            return c
    return None


# ===========================================
# Non-parametric effect size & bootstrap CI
# ===========================================
def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cliff's delta (non-parametric effect size). Positive if y > x.
    """
    x = np.asarray(x); y = np.asarray(y)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    n, m = len(x), len(y)
    # Efficient-ish: sort and use ranks
    xv = np.sort(x)
    yv = np.sort(y)
    i = j = gt = lt = 0
    while i < n and j < m:
        if xv[i] < yv[j]:
            lt += (m - j)
            i += 1
        elif xv[i] > yv[j]:
            gt += (n - i)
            j += 1
        else:
            # equal; advance both
            # Count equals as half (no contribution)
            i += 1; j += 1
    delta = (gt - lt) / (n * m)
    return float(delta)

def bootstrap_ci_relative_improvement(base, cmp_, n_boot=5000, alpha=0.05, paired=False, rng=None):
    """
    Bootstrap CI for % improvement = (base - cmp)/base * 100.
    If paired=True and sizes match -> sample indices jointly.
    """
    base = np.asarray(base); cmp_ = np.asarray(cmp_)
    maskb = np.isfinite(base); maskc = np.isfinite(cmp_)
    if paired and len(base) == len(cmp_):
        mask = maskb & maskc
        b = base[mask]; c = cmp_[mask]
        n = len(b)
        if n < 2:
            return (np.nan, np.nan)
        rng = np.random.default_rng(rng)
        stats = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            bb = b[idx]; cc = c[idx]
            denom = np.nanmean(bb)
            if not np.isfinite(denom) or denom == 0:
                stats.append(np.nan)
            else:
                stats.append((denom - np.nanmean(cc)) / denom * 100.0)
        lo, hi = np.nanpercentile(stats, [100*alpha/2, 100*(1-alpha/2)])
        return float(lo), float(hi)
    else:
        b = base[maskb]; c = cmp_[maskc]
        nb, nc = len(b), len(c)
        if nb < 2 or nc < 2:
            return (np.nan, np.nan)
        rng = np.random.default_rng(rng)
        stats = []
        for _ in range(n_boot):
            bb = b[rng.integers(0, nb, size=nb)]
            cc = c[rng.integers(0, nc, size=nc)]
            denom = np.nanmean(bb)
            if not np.isfinite(denom) or denom == 0:
                stats.append(np.nan)
            else:
                stats.append((denom - np.nanmean(cc)) / denom * 100.0)
        lo, hi = np.nanpercentile(stats, [100*alpha/2, 100*(1-alpha/2)])
        return float(lo), float(hi)


# =======================================================
# Core analysis (comparisons + statistical validation)
# =======================================================
def analyze_experiment_per_dataset(
    csv_file: str,
    metrics=("Time", "Inertia", "PeakMB", "TrafficRel"),
    compare_suite=("Hybrid", "Adaptive", "MiniBatch+Full", "MixedPerCluster"),
    baselines=("Double", "Single"),
    outdir="../Results/SUMMARY",
    label=None,
):
    """
    For each dataset (DatasetName/Size, NumClusters):
      - average repeats per Suite for the "comparisons" table,
      - statistical tests on paired repeats when possible:
          * Paired t-test (fallback to Welch)
          * Wilcoxon signed-rank / Mann-Whitney U
          * KS test
          * Cliff's delta (non-parametric effect size)
          * Bootstrap CI for % improvement
    """
    df_raw = pd.read_csv(csv_file)
    if df_raw.empty:
        return {"_note": "empty file"}

    df = _maybe_add_trafficrel(df_raw)

    base_keys = ["DatasetName", "NumClusters"]
    if not set(base_keys).issubset(df.columns):
        base_keys = [c for c in ["DatasetSize", "NumClusters"] if c in df.columns]
        if not base_keys:
            return {"_note": "no suitable grouping columns (DatasetName/DatasetSize, NumClusters)"}

    desired_no_tr = [m for m in metrics if m != "TrafficRel"]
    metrics_present, alias_map = _resolve_metrics_present(df, desired_no_tr)
    if "TrafficRel" in df.columns and "TrafficRel" in metrics:
        metrics_present = tuple(list(metrics_present) + ["TrafficRel"])
    if not metrics_present:
        return {"_note": "no desired metrics present"}

    compare_suites = [compare_suite] if isinstance(compare_suite, str) else list(compare_suite)
    suites_in_file = set(df["Suite"].unique().tolist()) if "Suite" in df.columns else set()
    suites_present = [s for s in compare_suites if s in suites_in_file]
    if not suites_present:
        return {"_note": f"none of {compare_suites} found in file"}

    outdir = Path(outdir)
    stem_label = label or Path(csv_file).stem

    # ---------- "comparisons" table (means) ----------
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

    # ---------- Statistical validation ---------- 
    stats_rows = []
    optional_keys = [c for c in ["Mode", "Cap", "tolerance_single"] if c in df.columns]
    cohort_keys = base_keys + optional_keys
    repeat_col = _find_repeat_col(df)

    for ds_keys, sub in df.groupby(cohort_keys, as_index=False):
        ds_row_base = ds_keys.iloc[0].to_dict() if isinstance(ds_keys, pd.DataFrame) else dict(zip(cohort_keys, ds_keys))
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

                    # Pair where possible
                    paired = False
                    x = y = None
                    if repeat_col and repeat_col in a.columns and repeat_col in b.columns:
                        merged = pd.merge(a, b, on=repeat_col, how="inner", suffixes=("_base", "_cmp"))
                        x = merged[f"{col}_base"].to_numpy()
                        y = merged[f"{col}_cmp"].to_numpy()
                        mask = np.isfinite(x) & np.isfinite(y)
                        x, y = x[mask], y[mask]
                        paired = (len(x) >= 2)
                    elif len(a) == len(b) and len(a) >= 2:
                        x = a[col].to_numpy()
                        y = b[col].to_numpy()
                        mask = np.isfinite(x) & np.isfinite(y)
                        x, y = x[mask], y[mask]
                        paired = (len(x) >= 2)
                    else:
                        x = a[col].to_numpy()
                        y = b[col].to_numpy()

                    mean_base = float(np.nanmean(a[col])) if len(a) else np.nan
                    mean_cmp  = float(np.nanmean(b[col])) if len(b) else np.nan
                    rel = (mean_cmp / mean_base) if (np.isfinite(mean_base) and mean_base != 0) else np.nan
                    impr = ((mean_base - mean_cmp) / mean_base * 100.0) if (np.isfinite(mean_base) and mean_base != 0) else np.nan

                    # Stats defaults
                    t_stat = t_p = w_stat = w_p = mw_stat = mw_p = ks_stat = ks_p = np.nan
                    cohend = np.nan
                    cliffs = np.nan
                    ci_lo = ci_hi = np.nan
                    test_used = "paired" if paired else "welch"

                    if _HAVE_SCIPY:
                        try:
                            if paired:
                                res = ttest_rel(x, y, nan_policy="omit")
                                t_stat, t_p = float(res.statistic), float(res.pvalue)

                                diffs = y - x
                                if len(diffs) >= 5 and not np.allclose(diffs, diffs[0]):
                                    try:
                                        w = wilcoxon(diffs, zero_method="wilcox", correction=False,
                                                     alternative="two-sided", mode="auto")
                                        w_stat, w_p = float(w.statistic), float(w.pvalue)
                                    except Exception:
                                        pass
                                # Cohen's d for paired
                                if len(diffs) >= 2 and np.nanstd(diffs, ddof=1) > 0:
                                    cohend = float(np.nanmean(diffs) / np.nanstd(diffs, ddof=1))

                                # Mann-Whitney/KS also useful even when paired
                                if len(x) >= 2 and len(y) >= 2:
                                    try:
                                        mw = mannwhitneyu(x, y, alternative="two-sided")
                                        mw_stat, mw_p = float(mw.statistic), float(mw.pvalue)
                                        ks = ks_2samp(x, y, alternative="two-sided", method="auto")
                                        ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
                                    except Exception:
                                        pass
                            else:
                                res = ttest_ind(a[col].to_numpy(), b[col].to_numpy(),
                                                equal_var=False, nan_policy="omit")
                                t_stat, t_p = float(res.statistic), float(res.pvalue)
                                test_used = "welch"

                                if len(a) >= 2 and len(b) >= 2:
                                    try:
                                        mw = mannwhitneyu(a[col].to_numpy(), b[col].to_numpy(), alternative="two-sided")
                                        mw_stat, mw_p = float(mw.statistic), float(mw.pvalue)
                                        ks = ks_2samp(a[col].to_numpy(), b[col].to_numpy(), alternative="two-sided", method="auto")
                                        ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
                                    except Exception:
                                        pass
                        except Exception:
                            pass

                    # Effect size (non-parametric)
                    cliffs = cliffs_delta(a[col].to_numpy(), b[col].to_numpy())

                    # Bootstrap CI for % improvement
                    ci_lo, ci_hi = bootstrap_ci_relative_improvement(
                        a[col].to_numpy(), b[col].to_numpy(),
                        n_boot=3000, alpha=0.05, paired=paired
                    )

                    stats_rows.append({
                        **ds_row_base,
                        "metric": logical_metric,
                        "baseline": baseline,
                        "compare_suite": cs,
                        "mean_baseline": mean_base,
                        "mean_compare":  mean_cmp,
                        "Rel": rel,
                        "Improvement_%": impr,
                        "Improvement_%_CI95_lo": ci_lo,
                        "Improvement_%_CI95_hi": ci_hi,
                        "paired_mode": test_used,
                        "n_base": int(len(a)),
                        "n_cmp":  int(len(b)),
                        "t_stat": t_stat, "t_p": t_p,
                        "wilcoxon_stat": w_stat, "wilcoxon_p": w_p,
                        "mannwhitney_stat": mw_stat, "mannwhitney_p": mw_p,
                        "ks_stat": ks_stat, "ks_p": ks_p,
                        "cohens_d": cohend,
                        "cliffs_delta": cliffs,
                    })

    if stats_rows:
        stats_df = pd.DataFrame(stats_rows)
        lead = [
            *cohort_keys, "metric", "baseline", "compare_suite",
            "mean_baseline", "mean_compare", "Rel", "Improvement_%",
            "Improvement_%_CI95_lo", "Improvement_%_CI95_hi",
            "paired_mode", "n_base", "n_cmp",
            "t_stat", "t_p", "wilcoxon_stat", "wilcoxon_p",
            "mannwhitney_stat", "mannwhitney_p", "ks_stat", "ks_p",
            "cohens_d", "cliffs_delta",
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
    return {"written": written} if written else {"_note": "nothing written"}


# =======================================================
# Correlation matrices (Pearson / Spearman)
# =======================================================
def _corr_heatmap(df_corr: pd.DataFrame, outpath: Path, title: str):
    if not _HAVE_MPL or df_corr.empty:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=150)
    im = ax.imshow(df_corr.values, vmin=-1, vmax=1)
    ax.set_xticks(range(df_corr.shape[1])); ax.set_yticks(range(df_corr.shape[0]))
    ax.set_xticklabels(df_corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(df_corr.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)

def build_metric_correlations(df: pd.DataFrame, alias_map: dict, outstem: Path, title_prefix=""):
    """
    Build correlations between logical metrics using per-(Suite, cohort) means.
    """
    logical_cols = []
    for logical, actual in alias_map.items():
        if actual in df.columns:
            logical_cols.append((logical, actual))
    # include TrafficRel if present
    if "TrafficRel" in df.columns:
        logical_cols.append(("TrafficRel", "TrafficRel"))

    # collapse to (Suite, DatasetName, NumClusters, [optional]) means per metric
    group_keys = [c for c in ["Suite", "DatasetName", "DatasetSize", "NumClusters", "Mode"] if c in df.columns]
    agg = df.groupby(group_keys, as_index=False)[[a for _, a in logical_cols]].mean()

    if agg.empty or len(logical_cols) < 2:
        return

    # rename to logical names for correlation readability
    ren = {actual: logical for logical, actual in logical_cols}
    agg = agg.rename(columns=ren)

    pear = agg[[logical for logical, _ in logical_cols]].corr(method="pearson")
    spear = agg[[logical for logical, _ in logical_cols]].corr(method="spearman")

    # write
    _export_md(pear.reset_index().rename(columns={"index": ""}), outstem.with_name(outstem.name + "_corr_pearson"))
    _export_md(spear.reset_index().rename(columns={"index": ""}), outstem.with_name(outstem.name + "_corr_spearman"))
    pear.to_csv(outstem.with_name(outstem.name + "_corr_pearson.csv"), index=True)
    spear.to_csv(outstem.with_name(outstem.name + "_corr_spearman.csv"), index=True)

    # optional heatmaps
    _corr_heatmap(pear, outstem.with_name(outstem.name + "_corr_pearson.png"),
                  title=f"{title_prefix} Pearson corr")
    _corr_heatmap(spear, outstem.with_name(outstem.name + "_corr_spearman.png"),
                  title=f"{title_prefix} Spearman corr")


# =======================================================
# OPTIONAL: cluster diagnostics (if you have label files)
# =======================================================
def analyze_cluster_labels(glob_pattern: str, outdir="../Results/SUMMARY/CLUSTERS"):
    """
    Expect CSVs with at least:
      DatasetName, NumClusters, Suite, RunIdx (optional), SampleId, ClusterId
    Produces per-(DatasetName, NumClusters, Suite) cluster size tables.
    If you later provide features X to this function, we can add silhouette_score.
    """
    paths = list(Path().glob(glob_pattern))
    if not paths:
        print(f"[clusters] no files match {glob_pattern}")
        return

    rows = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        need = {"DatasetName", "NumClusters", "Suite", "ClusterId"}
        if not need.issubset(df.columns):
            print(f"[clusters] skip {p} (needs {need})")
            continue
        grp = (df.groupby(["DatasetName", "NumClusters", "Suite", "ClusterId"])
                 .size().reset_index(name="Count"))
        rows.append(grp)
    if not rows:
        print("[clusters] nothing to summarize")
        return
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    sizes = pd.concat(rows, ignore_index=True)
    _export_md(sizes, outdir / "cluster_sizes")

    # If later you have X (features) aligned to SampleId per file, you can
    # compute silhouette_score and extend this helper.

    print(f"[clusters] wrote: {outdir/'cluster_sizes.md'}")


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

    all_alias_maps = []
    all_dfs_for_corr = []

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

        # Correlations per file
        df_full = _maybe_add_trafficrel(info["df"])
        present, alias_map = _resolve_metrics_present(df_full, desired_metrics)
        all_alias_maps.append(alias_map)
        build_metric_correlations(
            df_full, alias_map,
            Path("../Results/SUMMARY") / (Path(f).stem),
            title_prefix=Path(f).stem
        )
        all_dfs_for_corr.append(df_full.rename(columns={v: k for k, v in alias_map.items()}))

    # Global correlations across all files
    if all_dfs_for_corr:
        merged = pd.concat(all_dfs_for_corr, ignore_index=True)
        # Rebuild alias on merged (now renamed to logical already)
        merged_alias = {m: m for m in ["Time", "Inertia", "PeakMB"] if m in merged.columns}
        if "TrafficRel" in merged.columns:
            merged_alias["TrafficRel"] = "TrafficRel"
        build_metric_correlations(
            merged, merged_alias,
            Path("../Results/SUMMARY/GLOBAL"),
            title_prefix="GLOBAL"
        )

    # OPTIONAL: enable if you have cluster label CSVs
    # analyze_cluster_labels("../ClusterOutputs/*.csv", outdir="../Results/SUMMARY/CLUSTERS")
