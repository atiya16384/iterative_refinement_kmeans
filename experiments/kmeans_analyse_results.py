# kmeans_analyse_results.py
from pathlib import Path
import math
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

# ---------------- helpers ----------------

def _export_tables(per_ds_df: pd.DataFrame, summary_row: pd.Series, outstem: Path) -> None:
    """
    Write per-dataset table and a one-row summary to:
      - <outstem>.csv
      - <outstem>.md
      - <outstem>.tex
    """
    outstem.parent.mkdir(parents=True, exist_ok=True)

    # Stable column order if present
    order_cols = [c for c in ["DatasetName", "DatasetSize", "NumClusters"] if c in per_ds_df.columns]
    rest_cols = [c for c in per_ds_df.columns if c not in order_cols]
    per_ds_df = per_ds_df[order_cols + rest_cols]

    per_ds_df.to_csv(outstem.with_suffix(".csv"), index=False)

    with open(outstem.with_suffix(".md"), "w", encoding="utf-8") as f:
        f.write(per_ds_df.to_markdown(index=False))
        f.write("\n\n**Summary**\n\n")
        f.write(pd.DataFrame([summary_row]).to_markdown(index=False))

    with open(outstem.with_suffix(".tex"), "w", encoding="utf-8") as f:
        f.write(per_ds_df.to_latex(index=False, float_format="%.6g"))
        f.write("\n% Summary\n")
        f.write(pd.DataFrame([summary_row]).to_latex(index=False, float_format="%.6g"))


# ---------------- core analysis ----------------

def analyze_experiment(
    csv_file: str,
    metrics=("Time", "Inertia", "Memory_MB"),
    compare_suite="Hybrid",
    baselines=("Double", "Single"),   # run both if present
    outdir="../Results/SUMMARY",
    label=None,
):
    """
    Analyze one experiment CSV.
    - Aggregates per (DatasetName, NumClusters) across any sweep columns.
    - Compares <compare_suite> (default: Hybrid) to each baseline in `baselines`
      if that baseline exists in the CSV (Double and/or Single).
    - Emits per-dataset tables + one-row summaries to CSV/MD/TeX.
    - Returns dict with results for each (metric, baseline) present.
    """
    df = pd.read_csv(csv_file)
    if df.empty:
        return {"_note": "empty file"}

    # Group keys (older runs might have DatasetSize instead of DatasetName)
    base_index = ["DatasetName", "NumClusters"]
    if not set(base_index).issubset(df.columns):
        base_index = [c for c in ["DatasetSize", "NumClusters"] if c in df.columns]
        if not base_index:
            return {"_note": "no suitable grouping columns (DatasetName/DatasetSize, NumClusters)"}

    suites = sorted(df["Suite"].unique().tolist())
    if compare_suite not in suites:
        return {"_note": f"no '{compare_suite}' rows in file"}

    out = {}
    outdir = Path(outdir)
    stem_label = label or Path(csv_file).stem

    for metric in metrics:
        if metric not in df.columns:
            continue

        # Aggregate to avoid overweighting sweep settings
        agg = df.groupby(base_index + ["Suite"], as_index=False)[metric].mean()

        for baseline in baselines:
            if baseline not in suites or baseline == compare_suite:
                continue  # skip if baseline absent or same as compare_suite

            # Pivot to [baseline, compare_suite]
            pivoted = (
                agg.pivot_table(index=base_index, columns="Suite", values=metric, aggfunc="mean")
                   .filter(items=[baseline, compare_suite])
                   .dropna()
            )
            if pivoted.empty or {baseline, compare_suite} - set(pivoted.columns):
                continue

            b = pivoted[baseline].to_numpy()
            c = pivoted[compare_suite].to_numpy()
            diff = b - c  # positive means compare_suite is better (smaller) when lower is better

            # % improvement (lower is better for Time/Memory_MB; for Inertia we still show % change vs baseline)
            lower_is_better = metric in ("Time", "Memory_MB")
            improvement_pct = (b - c) / b * 100.0  # +% = compare_suite lower than baseline

            # paired stats when we have ≥2 pairs
            if len(b) >= 2:
                t_stat, t_p = ttest_rel(b, c)
                try:
                    w_stat, w_p = wilcoxon(b, c)
                except ValueError:
                    w_stat, w_p = float("nan"), float("nan")
                cohens_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else float("nan")
            else:
                t_stat = t_p = w_stat = w_p = cohens_d = float("nan")

            # per-dataset table
            per_ds_df = (
                pivoted.reset_index()
                       .rename(columns={
                           baseline: f"{baseline}_{metric}",
                           compare_suite: f"{compare_suite}_{metric}",
                       })
            )
            per_ds_df[f"Rel_{metric}"] = per_ds_df[f"{compare_suite}_{metric}"] / per_ds_df[f"{baseline}_{metric}"]
            per_ds_df["Improvement_%"] = (
                (per_ds_df[f"{baseline}_{metric}"] - per_ds_df[f"{compare_suite}_{metric}"])
                / per_ds_df[f"{baseline}_{metric}"] * 100.0
            )

            # summary row
            summary_row = pd.Series({
                "baseline": baseline,
                "compare_suite": compare_suite,
                "metric": metric,
                "n_pairs": len(b),
                "mean_baseline": b.mean(),
                "mean_compare": c.mean(),
                "mean_rel": (c / b).mean(),
                "mean_improvement_%": improvement_pct.mean(),
                "t_test_stat": t_stat, "t_test_p": t_p,
                "wilcoxon_stat": w_stat, "wilcoxon_p": w_p,
                "cohens_d": cohens_d,
            })

            # write tables
            outstem = outdir / f"{stem_label}__{metric}__{compare_suite.lower()}_vs_{baseline.lower()}"
            _export_tables(per_ds_df, summary_row, outstem)

            out[(metric, baseline)] = {
                "baseline": baseline,
                "compare_suite": compare_suite,
                "per_dataset": per_ds_df,
                "summary": dict(summary_row),
                "outstem": str(outstem),
            }

    if not out:
        return {"_note": "no comparable metrics present"}
    return out


# ---------------- run all & emit combined summary ----------------

if __name__ == "__main__":
    csv_files = [
        "../Results/hybrid_kmeans_Results_expA.csv",
        "../Results/hybrid_kmeans_Results_expB.csv",
        "../Results/hybrid_kmeans_Results_expC.csv",
        "../Results/hybrid_kmeans_Results_expD.csv",
        "../Results/hybrid_kmeans_Results_expE.csv",
        "../Results/hybrid_kmeans_Results_expF.csv",
    ]

    all_rows = []  # one row per (experiment × metric × baseline)
    for f in csv_files:
        p = Path(f)
        if not p.exists() or p.stat().st_size == 0:
            print(f"[skip] {f} missing or empty")
            continue

        print(f"\n==== {f} ====")
        res = analyze_experiment(
            f,
            metrics=("Time", "Inertia", "Memory_MB"),
            compare_suite="Hybrid",
            baselines=("Double", "Single"),
            outdir="../Results/SUMMARY",
            label=Path(f).stem,
        )
        note = res.get("_note")
        if note:
            print(note)
            continue

        for (metric, baseline), r in res.items():
            print(f"\n[{metric}] {r['compare_suite']} vs {baseline}")
            for k, v in r["summary"].items():
                print(f"  {k:>22}: {v}")
            print(f"  wrote: {r['outstem']}.csv / .md / .tex")

            all_rows.append({
                "Experiment": Path(f).stem,
                "Metric": metric,
                "Baseline": baseline,
                **r["summary"],
            })

    # write combined summary across experiments/metrics/baselines
    if all_rows:
        outdir = Path("../Results/SUMMARY")
        outdir.mkdir(parents=True, exist_ok=True)
        all_df = pd.DataFrame(all_rows)
        stem = outdir / "_ALL_experiments_summary"
        all_df.to_csv(stem.with_suffix(".csv"), index=False)
        with open(stem.with_suffix(".md"), "w", encoding="utf-8") as fh:
            fh.write(all_df.to_markdown(index=False))
        with open(stem.with_suffix(".tex"), "w", encoding="utf-8") as fh:
            fh.write(all_df.to_latex(index=False, float_format="%.6g"))
        print(f"\nWrote combined summary: {stem}.csv / .md / .tex")



