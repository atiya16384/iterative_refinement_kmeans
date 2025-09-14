# kmeans_analyse_results.py
from pathlib import Path
import pandas as pd

# =========================
# Helpers: metric aliasing
# =========================

# Logical -> acceptable column names in CSVs
METRIC_ALIASES = {
    "PeakMB":  ["PeakMB", "Memory_MB", "Peak_MB", "RSS_Peak_MB"],
    "Time":    ["Time"],
    "Inertia": ["Inertia"],
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


# =============================
# Helpers: traffic derivation
# =============================

def _maybe_add_trafficrel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add TrafficRel (estimated memory traffic relative to Double) if we have
    the needed columns. Model:
        traffic_abs ≈ TotalIter - 0.5 * ItersSingle
    because a float32 iteration moves ~half as many bytes as float64.
    We normalize by the cohort's Double TotalIter.

    Cohort keys: (DatasetName, NumClusters, Mode [, tolerance_single]).
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
    # Relative to the *matching* Double cohort
    d["TrafficRel"] = d["TrafficAbs"] / d["Tdouble"]

    return d


# ======================
# Markdown export helper
# ======================

def _export_simple(per_ds_df: pd.DataFrame, outstem: Path) -> None:
    """
    Write one tidy (long) table to <outstem>.md
    """
    outstem.parent.mkdir(parents=True, exist_ok=True)

    # Preferred column order (use only those present)
    lead = [
        "DatasetName", "DatasetSize", "NumClusters",
        "metric", "baseline", "compare_suite",
        "baseline_value", "compare_value",
        "Rel", "Improvement_%", "n_pairs"
    ]
    lead = [c for c in lead if c in per_ds_df.columns]
    rest = [c for c in per_ds_df.columns if c not in lead]
    tbl = per_ds_df[lead + rest]

    with open(outstem.with_suffix(".md"), "w", encoding="utf-8") as f:
        f.write(tbl.to_markdown(index=False))


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
    Print & return schema info so we only analyze what's present.
    Also derives TrafficRel if possible (non-destructive if not).
    """
    kw = {}
    if sample_rows is not None:
        kw["nrows"] = int(sample_rows)

    df = pd.read_csv(csv_file, **kw)
    df = _maybe_add_trafficrel(df)

    metrics_present, alias_map = _resolve_metrics_present(df, expected_metrics)
    # If TrafficRel is available, include it
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
        for logical in metrics_present:
            col = logical if logical == "TrafficRel" else alias_map.get(logical, logical)
            cnt = df.groupby("Suite")[col].apply(lambda s: s.notna().sum()).to_dict()
            print(f"  non-null by Suite for {logical}[{col}]:", cnt)

    return {
        "columns": cols,
        "present_keys": present_keys,
        "suites_present": suites_present,
        "present_compares": present_compares,
        "present_baselines": present_baselines,
        "metrics_present": metrics_present,
    }


# ===========================
# Core analysis (per-dataset)
# ===========================

def analyze_experiment_per_dataset(
    csv_file: str,
    metrics=("Time", "Inertia", "PeakMB", "TrafficRel"),  # logical names
    compare_suite=("Hybrid", "Adaptive", "MiniBatch+Full", "MixedPerCluster"),
    baselines=("Double", "Single"),
    outdir="../Results/SUMMARY",
    label=None,
):
    """
    For each dataset (DatasetName/Szie, NumClusters):
      - average repeats within the dataset per Suite,
      - compare each baseline vs each compare_suite for every metric present,
      - return a tidy table with generic value columns to avoid NaNs.

    Notes:
      - Handles PeakMB vs Memory_MB via METRIC_ALIASES.
      - Derives TrafficRel if ItersSingle/ItersDouble are available.
      - Baseline comparisons are cohort-matched implicitly by grouping per dataset.
    """
    df = pd.read_csv(csv_file)
    if df.empty:
        return {"_note": "empty file"}

    # Derive TrafficRel if possible
    df = _maybe_add_trafficrel(df)

    # Group keys (prefer DatasetName; fall back to DatasetSize)
    base_keys = ["DatasetName", "NumClusters"]
    if not set(base_keys).issubset(df.columns):
        base_keys = [c for c in ["DatasetSize", "NumClusters"] if c in df.columns]
        if not base_keys:
            return {"_note": "no suitable grouping columns (DatasetName/DatasetSize, NumClusters)"}

    # Resolve which desired metrics exist *in this file*
    desired_no_traffic = [m for m in metrics if m != "TrafficRel"]
    metrics_present, alias_map = _resolve_metrics_present(df, desired_no_traffic)

    # Add TrafficRel if present and requested
    if "TrafficRel" in df.columns and "TrafficRel" in metrics:
        metrics_present = tuple(list(metrics_present) + ["TrafficRel"])

    if not metrics_present:
        return {"_note": "no desired metrics present"}

    # Suites to compare (only those that actually appear)
    compare_suites = [compare_suite] if isinstance(compare_suite, str) else list(compare_suite)
    suites_in_file = set(df["Suite"].unique().tolist()) if "Suite" in df.columns else set()
    suites_present = [s for s in compare_suites if s in suites_in_file]
    if not suites_present:
        return {"_note": f"none of {compare_suites} found in file"}

    outdir = Path(outdir)
    stem_label = label or Path(csv_file).stem

    all_rows = []

    # Work per dataset
    for ds_keys, sub in df.groupby(base_keys, as_index=False):
        ds_row = ds_keys.iloc[0].to_dict() if isinstance(ds_keys, pd.DataFrame) else dict(zip(base_keys, ds_keys))
        suites_here = set(sub["Suite"].unique().tolist())
        if not (set(suites_present) & suites_here):
            continue

        for logical_metric in metrics_present:
            col = logical_metric if logical_metric == "TrafficRel" else alias_map.get(logical_metric, logical_metric)
            if col not in sub.columns:
                continue

            # average repeats within this dataset per Suite
            per_suite = (
                sub.groupby(["Suite"], as_index=False)[col]
                   .mean()
                   .set_index("Suite")[col]
            )

            for baseline in baselines:
                if baseline not in suites_here:
                    continue

                for cs in suites_present:
                    if cs == baseline or cs not in suites_here:
                        continue
                    if baseline not in per_suite.index or cs not in per_suite.index:
                        continue

                    b_val = float(per_suite.loc[baseline])
                    c_val = float(per_suite.loc[cs])
                    if not pd.notna(b_val) or not pd.notna(c_val) or b_val == 0:
                        continue

                    rel = c_val / b_val
                    imp = (b_val - c_val) / b_val * 100.0

                    row = {
                        **ds_row,
                        "metric": logical_metric,        # logical name (PeakMB/Time/Inertia/TrafficRel)
                        "baseline": baseline,
                        "compare_suite": cs,
                        "baseline_value": b_val,
                        "compare_value": c_val,
                        "Rel": rel,
                        "Improvement_%": imp,
                        # approx. pair count: number of baseline rows for this dataset & metric
                        "n_pairs": int(sub[sub["Suite"] == baseline][col].shape[0]),
                    }
                    all_rows.append(row)

    if not all_rows:
        return {"_note": "nothing written (no comparable suites/metrics)"}

    out_df = pd.DataFrame(all_rows)
    outstem = outdir / f"{stem_label}__all_comparisons"
    _export_simple(out_df, outstem)

    return {"written": [str(outstem)]}


# =================
# Run all CSVs
# =================

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
    desired_metrics = ("Time", "Inertia", "PeakMB")  # TrafficRel will be added if derivable

    for f in csv_files:
        p = Path(f)
        if not p.exists() or p.stat().st_size == 0:
            print(f"[skip] {f} missing or empty")
            continue

        # 1) Inspect fields present in this CSV
        info = peek_csv_fields(
            f,
            expected_metrics=desired_metrics,                 # NOTE: PeakMB (not Memory_MB)
            candidate_compare=desired_compares,
            candidate_baselines=desired_baselines,
            sample_rows=None,  # set an int if files are huge
        )

        # 2) Skip if essentials are missing
        if not info["metrics_present"]:
            print(f"[skip] {f}: none of expected metrics {desired_metrics} (or TrafficRel) found")
            continue
        if not info["present_compares"]:
            print(f"[skip] {f}: none of desired compare suites {desired_compares} found in Suite column")
            continue
        if not info["present_baselines"]:
            print(f"[skip] {f}: none of desired baselines {desired_baselines} found in Suite column")
            continue

        # 3) Analyze using ONLY what's present in this file
        print(f"\n==== ANALYZE {f} ====")
        res = analyze_experiment_per_dataset(
            f,
            metrics=tuple(info["metrics_present"]),           # includes TrafficRel if derivable
            compare_suite=tuple(info["present_compares"]),
            baselines=tuple(info["present_baselines"]),
            outdir="../Results/SUMMARY",
            label=Path(f).stem,
        )
        note = res.get("_note")
        if note:
            print(note)
        else:
            for path in res["written"]:
                print("  wrote:", path + ".md")


