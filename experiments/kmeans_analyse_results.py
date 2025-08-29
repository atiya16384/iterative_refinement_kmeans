# kmeans_analyse_results.py
from pathlib import Path
import pandas as pd

# ---------------- helpers ----------------

def _export_simple(per_ds_df: pd.DataFrame, outstem: Path) -> None:
    """
    Write one tidy (long) table to <outstem>.md
    """
    outstem.parent.mkdir(parents=True, exist_ok=True)

    # Preferred column order (only those that exist are used)
    lead = [
        "DatasetName", "DatasetSize", "NumClusters",
        "metric", "baseline", "compare_suite",
        "baseline_value", "compare_value",
        "Rel", "Improvement_%", "n_pairs"
    ]
    lead = [c for c in lead if c in per_ds_df.columns]
    rest = [c for c in per_ds_df.columns if c not in lead]
    per_ds_df = per_ds_df[lead + rest]

    with open(outstem.with_suffix(".md"), "w", encoding="utf-8") as f:
        f.write(per_ds_df.to_markdown(index=False))


def peek_csv_fields(
    csv_file: str,
    expected_metrics=("Time", "Inertia", "Memory_MB"),
    candidate_compare=("Hybrid", "Adaptive", "MiniBatch+Full", "MixedPerCluster"),
    candidate_baselines=("Double", "Single"),
    sample_rows=None,
):
    """
    Print & return schema info so we only analyze what's truly present.
    """
    kw = {}
    if sample_rows is not None:
        kw["nrows"] = int(sample_rows)

    df = pd.read_csv(csv_file, **kw)
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

    metrics_present = [m for m in expected_metrics if m in df.columns]
    print("  metrics found:", metrics_present or "(none of expected)")

    # quick non-null counts per metric × suite
    if metrics_present and "Suite" in df.columns:
        for m in metrics_present:
            cnt = df.groupby("Suite")[m].apply(lambda s: s.notna().sum()).to_dict()
            print(f"  non-null by Suite for {m}:", cnt)

    return {
        "columns": cols,
        "present_keys": present_keys,
        "suites_present": suites_present,
        "present_compares": present_compares,
        "present_baselines": present_baselines,
        "metrics_present": metrics_present,
    }


# ---------------- core analysis (per-dataset) ----------------

def analyze_experiment_per_dataset(
    csv_file: str,
    metrics=("Time", "Inertia", "Memory_MB"),
    compare_suite=("Hybrid", "Adaptive", "MiniBatch+Full", "MixedPerCluster"),
    baselines=("Double", "Single"),
    outdir="../Results/SUMMARY",
    label=None,
):
    """
    For each dataset (DatasetName/Size, NumClusters):
      - average repeats within the dataset per Suite,
      - compare each baseline vs each compare_suite for every metric present,
      - return a single tidy table with generic value columns to avoid NaNs.
    """
    df = pd.read_csv(csv_file)
    if df.empty:
        return {"_note": "empty file"}

    # Group keys (older runs might have DatasetSize instead of DatasetName)
    base_keys = ["DatasetName", "NumClusters"]
    if not set(base_keys).issubset(df.columns):
        base_keys = [c for c in ["DatasetSize", "NumClusters"] if c in df.columns]
        if not base_keys:
            return {"_note": "no suitable grouping columns (DatasetName/DatasetSize, NumClusters)"}

    # Normalize compare suites to a list
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
        # ds_keys comes as a DataFrame row when as_index=False; handle both cases
        if isinstance(ds_keys, pd.DataFrame):
            ds_row = ds_keys.iloc[0].to_dict()
        else:
            ds_row = dict(zip(base_keys, ds_keys))

        suites_here = set(sub["Suite"].unique().tolist())
        if not (set(suites_present) & suites_here):
            continue

        for metric in metrics:
            if metric not in sub.columns:
                continue

            # average repeats within this dataset per Suite
            per_suite = (
                sub.groupby(["Suite"], as_index=False)[metric]
                   .mean()
                   .set_index("Suite")[metric]
            )

            for baseline in baselines:
                if baseline not in suites_here:
                    continue

                for cs in compare_suites:
                    if cs == baseline or cs not in suites_here:
                        continue
                    if baseline not in per_suite.index or cs not in per_suite.index:
                        continue

                    b_val = float(per_suite.loc[baseline])
                    c_val = float(per_suite.loc[cs])
                    if b_val == 0 or not pd.notna(b_val) or not pd.notna(c_val):
                        continue

                    row = {
                        **ds_row,
                        "metric": metric,
                        "baseline": baseline,
                        "compare_suite": cs,
                        "baseline_value": b_val,
                        "compare_value": c_val,
                        "Rel": c_val / b_val,
                        "Improvement_%": (b_val - c_val) / b_val * 100.0,
                        "n_pairs": int(sub[sub["Suite"] == baseline][metric].shape[0]),
                    }
                    all_rows.append(row)

    if not all_rows:
        return {"_note": "nothing written (no comparable suites/metrics)"}

    out_df = pd.DataFrame(all_rows)
    outstem = outdir / f"{stem_label}__all_comparisons"
    _export_simple(out_df, outstem)

    return {"written": [str(outstem)]}


# ---------------- run all ----------------

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
    desired_metrics = ("Time", "Inertia", "Memory_MB")

    for f in csv_files:
        p = Path(f)
        if not p.exists() or p.stat().st_size == 0:
            print(f"[skip] {f} missing or empty")
            continue

        # 1) Inspect fields actually present in this CSV
        info = peek_csv_fields(
            f,
            expected_metrics=desired_metrics,
            candidate_compare=desired_compares,
            candidate_baselines=desired_baselines,
            sample_rows=None,  # set to an int if the files are huge
        )

        # 2) Skip politely if essentials are missing
        if not info["metrics_present"]:
            print(f"[skip] {f}: none of expected metrics {desired_metrics} found")
            continue
        if not info["present_compares"]:
            print(f"[skip] {f}: none of desired compare suites {desired_compares} found in Suite column")
            continue
        if not info["present_baselines"]:
            print(f"[skip] {f}: none of desired baselines {desired_baselines} found in Suite column")
            continue

        # 3) Analyze using ONLY what exists in this file
        print(f"\n==== ANALYZE {f} ====")
        res = analyze_experiment_per_dataset(
            f,
            metrics=tuple(info["metrics_present"]),
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


