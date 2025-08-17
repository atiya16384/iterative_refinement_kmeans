# kmeans_analyse_results.py
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from pathlib import Path

def analyze_experiment(csv_file, metrics=("Time", "Inertia")):
    df = pd.read_csv(csv_file)
    if df.empty:
        return {"_note": "empty file"}

    # We aggregate per dataset & K, ignoring sweep columns
    # (works for A..F even if they have different extra columns).
    base_index = ["DatasetName", "NumClusters"]
    if not set(base_index).issubset(df.columns):
        # Older files may have DatasetSize but not DatasetName
        base_index = [c for c in ["DatasetSize", "NumClusters"] if c in df.columns]

    suites = sorted(df["Suite"].unique().tolist())
    if "Double" not in suites:
        return {"_note": "no Double rows in file"}

    # Pick the “hybrid/other” suite automatically.
    # For E this will be MiniBatch+Full, for F MixedPerCluster, for D Hybrid, etc.
    hybrid_candidates = [s for s in suites if s != "Double"]
    if not hybrid_candidates:
        return {"_note": "no non-Double rows to compare against"}
    hybrid_name = hybrid_candidates[0]  # if more than one, compare with the first

    out = {}

    for metric in metrics:
        if metric not in df.columns:
            continue

        # pivot to Double vs Hybrid-like suite
        pivoted = (
            df.pivot_table(index=base_index, columns="Suite", values=metric, aggfunc="mean")
              .filter(items=["Double", hybrid_name])  # keep only the 2 suites
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

        out[metric] = {
            "hybrid_label": hybrid_name,
            "per_dataset": pivoted.assign(Improvement_pct=improvement_pct),
            "summary": {
                "n_pairs": len(d),
                "mean_double": d.mean(),
                "mean_other": h.mean(),
                "mean_improvement_%": improvement_pct.mean(),
                "t_test_stat": t_stat, "t_test_p": t_p,
                "wilcoxon_stat": w_stat, "wilcoxon_p": w_p,
                "cohens_d": cohens_d,
            },
        }

    return out

if __name__ == "__main__":
    csv_files = [
        "Results/hybrid_kmeans_Results_expA.csv",
        "Results/hybrid_kmeans_Results_expB.csv",
        "Results/hybrid_kmeans_Results_expC.csv",
        "Results/hybrid_kmeans_Results_expD.csv",
        "Results/hybrid_kmeans_Results_expE.csv",
        "Results/hybrid_kmeans_Results_expF.csv",
    ]

    for f in csv_files:
        p = Path(f)
        if not p.exists() or p.stat().st_size == 0:
            print(f"[skip] {f} missing or empty")
            continue

        print(f"\n==== {f} ====")
        res = analyze_experiment(f)
        note = res.get("_note")
        if note:
            print(note)
            continue

        for metric, r in res.items():
            print(f"\n[{metric}] vs '{r['hybrid_label']}'")
            for k, v in r["summary"].items():
                print(f"  {k:>20}: {v}")

