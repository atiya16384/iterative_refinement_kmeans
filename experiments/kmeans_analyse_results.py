import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

def analyze_experiment(csv_file, metrics=("Time","Inertia"), group_index=("DatasetSize","NumClusters")):
    df = pd.read_csv(csv_file)

    # quick sanity: see what the file actually has
    # print(df[["Suite","Mode"]].drop_duplicates().head())

    results = {}
    for metric in metrics:
        # group per dataset/cluster (average over any sweep params)
        pivoted = (
            df.pivot_table(
                index=list(group_index),
                columns="Mode",             # <<--- THIS is the key change
                values=metric,
                aggfunc="mean"
            )
            .dropna(subset=["Double","Hybrid"], how="any")
        )

        if pivoted.empty:
            continue

        times_double = pivoted["Double"].to_numpy()
        times_hybrid = pivoted["Hybrid"].to_numpy()

        improvement = (times_double - times_hybrid) / times_double * 100
        diff = times_double - times_hybrid

        t_stat, t_p = ttest_rel(times_double, times_hybrid)
        w_stat, w_p = wilcoxon(times_double, times_hybrid, zero_method="wilcox")

        results[metric] = {
            "n_pairs": len(pivoted),
            "per_dataset": pivoted.assign(Improvement_pct=improvement),
            "summary": {
                "mean_double": float(times_double.mean()),
                "mean_hybrid": float(times_hybrid.mean()),
                "mean_improvement_%": float(improvement.mean()),
                "t_test_stat": float(t_stat),
                "t_test_p": float(t_p),
                "wilcoxon_stat": float(w_stat),
                "wilcoxon_p": float(w_p),
                "cohens_d": float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) else 0.0,
            }
        }
    return results

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
        try:
            out = analyze_experiment(f)
            print(f"\n==== {f} ====")
            if not out:
                print("No Double/Hybrid pairs found (check that A/B/C were run or the column names).")
            for metric, res in out.items():
                print(f"[{metric}] pairs={res['n_pairs']}")
                print(res["summary"])
        except FileNotFoundError:
            print(f"[skip] {f} not found")

