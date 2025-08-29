# kmeans_analyse_results.py
from pathlib import Path
import pandas as pd

# ---------------- helpers ----------------

def _export_simple(per_ds_df: pd.DataFrame, outstem: Path) -> None:
    """
    Write a per-dataset table (no global summary) to:
      - <outstem>.csv
      - <outstem>.md
      - <outstem>.tex
    """
    outstem.parent.mkdir(parents=True, exist_ok=True)

    # Stable column order if present
    lead = [c for c in ["DatasetName", "DatasetSize", "NumClusters"] if c in per_ds_df.columns]
    rest = [c for c in per_ds_df.columns if c not in lead]
    per_ds_df = per_ds_df[lead + rest]

    per_ds_df.to_csv(outstem.with_suffix(".csv"), index=False)

    with open(outstem.with_suffix(".md"), "w", encoding="utf-8") as f:
        f.write(per_ds_df.to_markdown(index=False))

    with open(outstem.with_suffix(".tex"), "w", encoding="utf-8") as f:
        f.write(per_ds_df.to_latex(index=False, float_format="%.6g"))


# ---------------- core analysis (per-dataset) ----------------

def analyze_experiment_per_dataset(
    csv_file: str,
    metrics=("Time", "Inertia", "Memory_MB"),
    compare_suite=("Hybrid", "Adaptive", "MiniBatch+Full", "MixedPerCluster"),
    baselines=("Double", "Single"),
    outdir="../Results/SUMMARY",
    label=None,
):
    df = pd.read_csv(csv_file)
    if df.empty:
        return {"_note": "empty file"}

    base_keys = ["DatasetName", "NumClusters"]
    if not set(base_keys).issubset(df.columns):
        base_keys = [c for c in ["DatasetSize", "NumClusters"] if c in df.columns]
        if not base_keys:
            return {"_note": "no suitable grouping columns"}

    # normalize suites
    compare_suites = [compare_suite] if isinstance(compare_suite, str) else list(compare_suite)
    suites_in_file = set(df["Suite"].unique().tolist())
    suites_present = [s for s in compare_suites if s in suites_in_file]
    if not suites_present:
        return {"_note": f"none of {compare_suites} found in file"}

    outdir = Path(outdir)
    stem_label = label or Path(csv_file).stem

    all_rows = []   # collect all comparisons here

    for ds_keys, sub in df.groupby(base_keys, as_index=False):
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

                    rel = c_val / b_val
                    improve_pct = (b_val - c_val) / b_val * 100.0

                    row = {
                        **ds_row,
                        "metric": metric,
                        "baseline": baseline,
                        "compare_suite": cs,
                        f"{baseline}_{metric}": b_val,
                        f"{cs}_{metric}": c_val,
                        f"Rel_{metric}": rel,
                        "Improvement_%": improve_pct,
                        "n_pairs": int(sub[sub["Suite"] == baseline][metric].shape[0]),
                    }
                    all_rows.append(row)

    if not all_rows:
        return {"_note": "nothing written"}

    # make one big dataframe
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

    for f in csv_files:
        p = Path(f)
        if not p.exists() or p.stat().st_size == 0:
            print(f"[skip] {f} missing or empty")
            continue

        print(f"\n==== {f} ====")
        res = analyze_experiment_per_dataset(
            f,
            metrics=("Time", "Inertia", "Memory_MB"),
            compare_suite=("Hybrid", "Adaptive", "MiniBatch+Full", "MixedPerCluster"),
            baselines=("Double", "Single"),
            outdir="../Results/SUMMARY",
            label=Path(f).stem,
        )
        note = res.get("_note")
        if note:
            print(note)
        else:
            for path in res["written"]:
                print("  wrote:", path + ".csv / .md / .tex")



