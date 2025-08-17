import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

# --- your function, unchanged ---
def analyze_experiment(csv_file, metrics=("Time","Inertia")):
    df = pd.read_csv(csv_file)

    results = {}
    for metric in metrics:
        pivoted = df.pivot_table(
            index=["DatasetSize","NumClusters"],
            columns="Suite",
            values=metric,
            aggfunc="mean"
        ).dropna()

        if "Double" not in pivoted.columns or "Hybrid" not in pivoted.columns:
            continue  # skip if missing Suite

        times_double = pivoted["Double"].values
        times_hybrid = pivoted["Hybrid"].values

        # relative improvement (%)
        improvement = (times_double - times_hybrid) / times_double * 100
        diff = times_double - times_hybrid

        t_stat, t_p = ttest_rel(times_double, times_hybrid)
        w_stat, w_p = wilcoxon(times_double, times_hybrid)
        
        results[metric] = {
            "per_dataset": pivoted.assign(Improvement_pct=improvement),
            "summary": {
                "mean_double": times_double.mean(),
                "mean_hybrid": times_hybrid.mean(),
                "mean_improvement_%": improvement.mean(),
                "t_test_stat": t_stat,
                "t_test_p": t_p,
                "wilcoxon_stat": w_stat,
                "wilcoxon_p": w_p,
                "cohens_d": diff.mean() / diff.std(ddof=1),
            }
        }
    return results


# --- Just call it for each experiment CSV ---
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
        try:
            results = analyze_experiment(f)
            print(f"\n==== {f} ====")
            for metric, res in results.items():
                print(f"[{metric}] Summary:")
                print(res["summary"])
        except FileNotFoundError:
            print(f"[skip] {f} not found")
