# kmeans_analyse_results.py
import math
from pathlib import Path
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

# ---------- helpers for table export ----------

def _export_tables(per_ds_df: pd.DataFrame, summary_row: pd.Series, outstem: Path) -> None:
    """
    Write per-dataset table and a one-row summary to:
      - <outstem>.csv
      - <outstem>.md
      - <outstem>.tex
    """
    outstem.parent.mkdir(parents=True, exist_ok=True)

    # CSV
    per_ds_df.to_csv(outstem.with_suffix(".csv"), index=False)

    # Markdown (nice for quick copy into docs)
    with open(outstem.with_suffix(".md"), "w", encoding="utf-8") as f:
        f.write(per_ds_df.to_markdown(index=False))
        f.write("\n\n**Summary:**\n\n")
        f.write(pd.DataFrame([summary_row]).to_markdown(index=False))

    # LaTeX (thesis-ready)
    with open(outstem.with_suffix(".tex"), "w", encoding="utf-8") as f:
        f.write(per_ds_df.to_latex(index=False, float_format="%.6g"))
        f.write("\n% Summary\n")
        f.write(pd.DataFrame([summary_row]).to_latex(index=False, float_format="%.6g"))


# ---------- core analysis ----------

def analyze_experiment(csv_file, metrics=("Time", "Inertia"), outdir="../Results/SUMMARY", label=None):
    """
    Analyze one experiment CSV.
    - Aggregates per (DatasetName, NumClusters) across any sweep columns.
    - Compares Double vs the first non-Double suite found (e.g., Hybrid).
    - Returns dict and writes per-dataset + summary tables to outdir.
    """
    df = pd.read_csv(csv_file)
    if df.empty:
        return {"_note": "empty file"}

    # Base index across A..F (older files might have DatasetSize instead of DatasetName)
    base_index = ["DatasetName", "NumClusters"]
    if not set(base_index).issubset(df.columns):
        base_index = [c for c in ["DatasetSize", "NumClusters"] if c in df.columns]
        if not base_index:
            return {"_note": "no suitable grouping columns (DatasetName/DatasetSize, NumClusters)"}

    suites = sorted(df["Suite"].unique().tolist())
    if "Double" not in suites:
        return {"_note": "no Double rows in file"}

    # Pick the comparison suite automatically (first non-Double)
    hybrid_candidates = [s for s in suites if s != "Double"]
    if not hybrid_candidates:
        return {"_note": "no non-Double rows to compare against"}
    hybrid_name = hybrid_candidates[0]

    out = {}
    outdir = Path(outdir)
    stem_label = label or Path(csv_file).stem

    for metric in metrics:
        if metric not in df.columns:
            continue

        # Aggregate to (dataset, k, suite) first to avoid overweighting any sweep setting
        agg = (
            df.groupby(base_index + ["Suite"], as_index=False)[metric]
              .mean()
        )

        # Pivot to compare Double vs chosen suite
        pivoted = (
            agg.pivot_table(index=base_index, columns="Suite", values=metric, aggfunc="mean")
               .filter(items=["Double", hybrid_name])
               .dropna()
        )

        if pivoted.empty or {"Double", hybrid_name} - set(pivoted.columns):
            continue

        d = pivoted["Double"].to_numpy()
        h = pivoted[hybrid_name].to_numpy()

        improvement_pct = (d - h) / d * 100.0
        diff = d - h

        # stats (safe guards for size 1)
        if len(d) >= 2:
            t_stat, t_p = ttest_rel(d, h)
            try:
                w_stat, w_p = wilcoxon(d, h)
            except ValueError:
                w_stat, w_p = float("nan"), float("nan")
            cohens_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else float("nan")
        else:
            t_stat = t_p = w_stat = w_p = cohens_d = float("nan")

        # Build per-dataset table (absolute values + relative/improvement)
        per_ds_df = (
            pivoted.reset_index()
                   .rename(columns={
                       "Double": f"Double_{metric}",
                       hybrid_name: f"{hybrid_name}_{metric}"
                   })
        )
        per_ds_df[f"Rel_{metric}"] = per_ds_df[f"{hybrid_name}_{metric}"] / per_ds_df[f"Double_{metric}"]
        if metric == "Time":
            per_ds_df["Improvement_%"] = (per_ds_df[f"Double_{metric}"] - per_ds_df[f"{hybrid_name}_{metric}"]) \
                                          / per_ds_df[f"Double_{metric}"] * 100.0

        # One-row summary for the thesis
        summary_row = pd.Series({
            "n_pairs": len(d),
            "mean_double": d.mean(),
            "mean_other": h.mean(),
            "mean_improvement_%": improvement_pct.mean() if metric == "Time" else float("nan"),
            "t_test_stat": t_stat, "t_test_p": t_p,
            "wilcoxon_stat": w_stat, "wilcoxon_p": w_p,
            "cohens_d": cohens_d,
        })

        # Emit tables
        outstem = outdir / f"{stem_label}__{metric}__double_vs_{hybrid_name.lower()}"
        _export_tables(per_ds_df, summary_row, outstem)

        out[metric] = {
            "hybrid_label": hybrid_name,
            "per_dataset": per_ds_df,
            "summary": dict(summary_row),
            "outstem": str(outstem)
        }

    if not out:
        return {"_note": "no comparable metrics present"}
    return out


# ---------- CLI ----------

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
        res = analyze_experiment(f, outdir="../Results/SUMMARY", label=Path(f).stem)
        note = res.get("_note")
        if note:
            print(note)
            continue

        for metric, r in res.items():
            print(f"\n[{metric}] vs '{r['hybrid_label']}'")
            for k, v in r["summary"].items():
                print(f"  {k:>20}: {v}")
            print(f"  wrote: {r['outstem']}.csv / .md / .tex")


