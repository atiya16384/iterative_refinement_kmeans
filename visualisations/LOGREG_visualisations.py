# visualisations/LOGREG_visualisations.py
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import pandas as pd

RESULTS_DIR = pathlib.Path("Results")
RESULTS_DIR.mkdir(exist_ok=True)

class LogisticVisualizer:
    def __init__(self, out_dir=RESULTS_DIR):
        self.out_dir = out_dir

    def normalize_against_double(self, df, y_col):
        """
        Divide values of y_col by the corresponding Double baseline.
        Works per dataset, classes, and sweep variable (Cap or tolerance).
        """
        normed = []
        for keys, group in df.groupby(["DatasetName", "NumClasses"]):
            # Find double baseline rows
            double_rows = group[group["Mode"] == "Double"]
            if double_rows.empty:
                continue
            for _, drow in double_rows.iterrows():
                if "Cap" in group.columns:
                    sub = group[group["Cap"] == drow["Cap"]].copy()
                else:
                    sub = group[group["tolerance_single"] == drow["tolerance_single"]].copy()
                sub[y_col] = sub[y_col] / drow[y_col]  # normalize
                normed.append(sub)
        return pd.concat(normed, ignore_index=True)

    def plot_cap_vs_time(self, df_A):
        df_norm = self.normalize_against_double(df_A, "Time")
        plt.figure(figsize=(8, 5))
        sns.lineplot(
            data=df_norm[df_norm["Mode"] == "Hybrid"],
            x="Cap", y="Time", marker="o", ci="sd"
        )
        plt.axhline(1.0, ls="--", color="red", label="Double baseline")
        plt.title("Experiment A: Cap vs Runtime (normalized)")
        plt.xlabel("Cap (fp32 iterations)")
        plt.ylabel("Relative Time (Hybrid / Double)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.out_dir / "logreg_cap_vs_time.png")
        plt.close()

    def plot_cap_vs_accuracy(self, df_A):
        df_norm = self.normalize_against_double(df_A, "Accuracy")
        plt.figure(figsize=(8, 5))
        sns.lineplot(
            data=df_norm[df_norm["Mode"] == "Hybrid"],
            x="Cap", y="Accuracy", marker="o", ci="sd"
        )
        plt.axhline(1.0, ls="--", color="red", label="Double baseline")
        plt.title("Experiment A: Cap vs Accuracy (normalized)")
        plt.xlabel("Cap (fp32 iterations)")
        plt.ylabel("Relative Accuracy (Hybrid / Double)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.out_dir / "logreg_cap_vs_accuracy.png")
        plt.close()

    def plot_tolerance_vs_time(self, df_B):
        df_norm = self.normalize_against_double(df_B, "Time")
        plt.figure(figsize=(8, 5))
        sns.lineplot(
            data=df_norm[df_norm["Mode"] == "Hybrid"],
            x="tolerance_single", y="Time", marker="o", ci="sd"
        )
        plt.axhline(1.0, ls="--", color="red", label="Double baseline")
        plt.title("Experiment B: Tolerance vs Runtime (normalized)")
        plt.xlabel("fp32 Tolerance")
        plt.ylabel("Relative Time (Hybrid / Double)")
        plt.xscale("log")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.out_dir / "logreg_tol_vs_time.png")
        plt.close()

    def plot_tolerance_vs_accuracy(self, df_B):
        df_norm = self.normalize_against_double(df_B, "Accuracy")
        plt.figure(figsize=(8, 5))
        sns.lineplot(
            data=df_norm[df_norm["Mode"] == "Hybrid"],
            x="tolerance_single", y="Accuracy", marker="o", ci="sd"
        )
        plt.axhline(1.0, ls="--", color="red", label="Double baseline")
        plt.title("Experiment B: Tolerance vs Accuracy (normalized)")
        plt.xlabel("fp32 Tolerance")
        plt.ylabel("Relative Accuracy (Hybrid / Double)")
        plt.xscale("log")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.out_dir / "logreg_tol_vs_accuracy.png")
        plt.close()


if __name__ == '__main__':
    # Load results
    df_A = pd.read_csv("Results/logistic_results_expA.csv")
    df_B = pd.read_csv("Results/logistic_results_expB.csv")

    log_vis = LogisticVisualizer(out_dir=RESULTS_DIR)

    # Experiment A plots
    log_vis.plot_cap_vs_time(df_A)
    log_vis.plot_cap_vs_accuracy(df_A)

    # Experiment B plots
    log_vis.plot_tolerance_vs_time(df_B)
    log_vis.plot_tolerance_vs_accuracy(df_B)

