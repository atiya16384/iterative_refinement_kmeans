import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class LOGREVisualiser:

    def plot_results(csv_path="../Results/results_all.csv"):
        df = pd.read_csv(csv_path)

        # --- Runtime vs AUC ---
        plt.figure(figsize=(8,6))
        sns.scatterplot(data=df, x="time_sec", y="roc_auc", hue="approach", style="penalty", s=80)
        plt.title("Runtime vs AUC")
        plt.xlabel("Runtime (sec)")
        plt.ylabel("ROC AUC")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # --- Barplot: Average logloss by approach ---
        plt.figure(figsize=(8,6))
        sns.barplot(data=df, x="approach", y="logloss")
        plt.title("Average Logloss by Approach")
        plt.tight_layout()
        plt.show()

        # --- Line plot: iterations vs AUC (for each approach) ---
        plt.figure(figsize=(8,6))
        sns.lineplot(data=df, x="iters", y="roc_auc", hue="approach", marker="o")
        plt.title("Convergence (iters vs AUC)")
        plt.tight_layout()
        plt.show()

        # --- Tradeoff curves: time vs logloss ---
        plt.figure(figsize=(8,6))
        sns.scatterplot(data=df, x="time_sec", y="logloss", hue="approach")
        plt.title("Runtime vs Logloss")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8,6))
        sns.lineplot(data=df, x="lambda", y="roc_auc", hue="approach", marker="o")
        plt.xscale("log")
        plt.title("Effect of λ on ROC AUC")
        plt.show()
        
        plt.figure(figsize=(8,6))
        sns.lineplot(data=df, x="lambda", y="time_sec", hue="approach", marker="o")
        plt.xscale("log")
        plt.title("Effect of λ on Runtime")
        plt.show()

        plt.figure(figsize=(8,6))
        sns.boxplot(data=df, x="penalty", y="roc_auc", hue="approach")
        plt.title("Penalty vs ROC AUC")
        plt.show()
        
        plt.figure(figsize=(8,6))
        sns.boxplot(data=df, x="penalty", y="time_sec", hue="approach")
        plt.title("Penalty vs Runtime")
        plt.show()

        plt.figure(figsize=(8,6))
        sns.scatterplot(data=df, x="time_sec", y="roc_auc", hue="solver", style="approach", s=80)
        plt.title("Solver Comparison")
        plt.show()
        
        plt.figure(figsize=(8,6))
        sns.boxplot(data=df, x="tol", y="roc_auc", hue="approach")
        plt.title("Tolerance vs ROC AUC")
        plt.show()
        
        plt.figure(figsize=(8,6))
        sns.boxplot(data=df, x="tol", y="time_sec", hue="approach")
        plt.title("Tolerance vs Runtime")
        plt.show()
        
        plt.figure(figsize=(8,6))
        sns.scatterplot(data=df, x="time_sec", y="roc_auc", hue="approach", style="penalty")
        for name, grp in df.groupby("approach"):
            best = grp.sort_values("time_sec").iloc[-1]
            plt.text(best["time_sec"], best["roc_auc"], name)
        plt.title("Pareto Frontier: Runtime vs ROC AUC")
        plt.show()

        def plot_chunks(history, title="Chunk Dynamics"):
            hist_df = pd.DataFrame(history, columns=["kind","chunk","prec","iters","delta","loss"])
            plt.figure(figsize=(8,6))
            sns.lineplot(data=hist_df, x="chunk", y="loss", hue="prec", marker="o")
            plt.title(title)
            plt.ylabel("Loss")
            plt.show()

        pivot = df.pivot_table(index="lambda", columns="alpha", values="roc_auc", aggfunc="mean")
        sns.heatmap(pivot, annot=True, cmap="viridis")
        plt.title("ROC AUC across λ and α")
        plt.show()

        sns.catplot(data=df, x="approach", y="roc_auc", kind="box")
        plt.title("Stability across folds")
        plt.show()

                

if __name__ in '__main__':
    logreg_vis = LOGREVisualiser()


