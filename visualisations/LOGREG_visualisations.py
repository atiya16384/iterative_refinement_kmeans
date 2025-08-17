# visualisations/LOGREG_visualisations.py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pathlib

RESULTS_DIR = pathlib.Path("Results")
RESULTS_DIR.mkdir(exist_ok=True)

class LogisticVisualizer:
    def __init__(self, out_dir=RESULTS_DIR):
        self.out_dir = out_dir

    def plot_cap_vs_time(self, df_A):
        plt.figure(figsize=(8, 5))
        sns.lineplot(
            data=df_A, x="Cap", y="Time", hue="Mode",
            marker="o", ci="sd"
        )
        plt.title("Experiment A: Cap vs Runtime")
        plt.xlabel("Cap (fp32 iterations)")
        plt.ylabel("Time (s)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.out_dir / "logreg_cap_vs_time.png")
        plt.close()

    def plot_cap_vs_accuracy(self, df_A):
        plt.figure(figsize=(8, 5))
        sns.lineplot(
            data=df_A, x="Cap", y="Accuracy", hue="Mode",
            marker="o", ci="sd"
        )
        plt.title("Experiment A: Cap vs Accuracy")
        plt.xlabel("Cap (fp32 iterations)")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.out_dir / "logreg_cap_vs_accuracy.png")
        plt.close()

    def plot_tolerance_vs_time(self, df_B):
        plt.figure(figsize=(8, 5))
        sns.lineplot(
            data=df_B, x="tolerance_single", y="Time", hue="Mode",
            marker="o", ci="sd"
        )
        plt.title("Experiment B: Tolerance vs Runtime")
        plt.xlabel("fp32 Tolerance")
        plt.ylabel("Time (s)")
        plt.xscale("log")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.out_dir / "logreg_tol_vs_time.png")
        plt.close()

    def plot_tolerance_vs_accuracy(self, df_B):
        plt.figure(figsize=(8, 5))
        sns.lineplot(
            data=df_B, x="tolerance_single", y="Accuracy", hue="Mode",
            marker="o", ci="sd"
        )
        plt.title("Experiment B: Tolerance vs Accuracy")
        plt.xlabel("fp32 Tolerance")
        plt.ylabel("Accuracy")
        plt.xscale("log")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.out_dir / "logreg_tol_vs_accuracy.png")
        plt.close()

if __name__ == '__main__':
    # After created df_A and df_B

    df_A = pd.read_csv("../Results/logistic_results_expA.csv")
    df_B = pd.read_csv("../Results/logistic_results_expB.csv")

    log_vis = LogisticVisualizer()

    # Experiment A plots
    log_vis.plot_cap_vs_time(df_A)
    log_vis.plot_cap_vs_accuracy(df_A)

    # Experiment B plots
    log_vis.plot_tolerance_vs_time(df_B)
    log_vis.plot_tolerance_vs_accuracy(df_B)
