# visualisations/LOGREG_visualisations.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib

RESULTS_DIR = pathlib.Path("../Results")
RESULTS_DIR.mkdir(exist_ok=True)

class LogisticVisualizer:
    
    def __init__(self, out_dir=RESULTS_DIR):
        self.out_dir = pathlib.Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # Experiment A: Cap sweep
    # =========================
    def plot_cap_vs_time(self, df_A: pd.DataFrame):
        if df_A.empty:
            print("[WARN] plot_cap_vs_time: df_A is empty.")
            return
        rel = _relative_hybrid_vs_double(df_A, xcol="Cap", ycol="Time")
        classes_col = _classes_col(df_A)

        plt.figure(figsize=(8, 5))
        for (ds, k), grp in rel.groupby(["DatasetName", classes_col]):
            plt.plot(grp["Cap"], grp["rel"], marker="o", label=f"{ds}-C{k}")
        plt.axhline(1.0, linestyle="--", color="gray", linewidth=1, label="Double baseline")
        plt.title("Experiment A: Cap vs Runtime (Hybrid / Double)")
        plt.xlabel("Cap (fp32 iteration cap)")
        plt.ylabel("Relative Time (Hybrid / Double)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.out_dir / "logreg_cap_vs_time.png")
        plt.close()

    def plot_cap_vs_accuracy(self, df_A: pd.DataFrame):
        if df_A.empty:
            print("[WARN] plot_cap_vs_accuracy: df_A is empty.")
            return
        rel = _relative_hybrid_vs_double(df_A, xcol="Cap", ycol="Accuracy")
        classes_col = _classes_col(df_A)

        plt.figure(figsize=(8, 5))
        for (ds, k), grp in rel.groupby(["DatasetName", classes_col]):
            plt.plot(grp["Cap"], grp["rel"], marker="o", label=f"{ds}-C{k}")
        plt.axhline(1.0, linestyle="--", color="gray", linewidth=1, label="Double baseline")
        plt.title("Experiment A: Cap vs Accuracy (Hybrid / Double)")
        plt.xlabel("Cap (fp32 iteration cap)")
        plt.ylabel("Relative Accuracy (Hybrid / Double)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.out_dir / "logreg_cap_vs_accuracy.png")
        plt.close()

    # =========================
    # Experiment B: Tolerance sweep
    # =========================
    def plot_tolerance_vs_time(self, df_B: pd.DataFrame):
        if df_B.empty:
            print("[WARN] plot_tolerance_vs_time: df_B is empty.")
            return
        rel = _relative_hybrid_vs_double(df_B, xcol="tolerance_single", ycol="Time")
        classes_col = _classes_col(df_B)

        plt.figure(figsize=(8, 5))
        for (ds, k), grp in rel.groupby(["DatasetName", classes_col]):
            plt.plot(grp["tolerance_single"], grp["rel"], marker="o", label=f"{ds}-C{k}")
        plt.xscale("log")
        plt.axhline(1.0, linestyle="--", color="gray", linewidth=1, label="Double baseline")
        plt.title("Experiment B: Tolerance vs Runtime (Hybrid / Double)")
        plt.xlabel("Single-precision Tolerance (log)")
        plt.ylabel("Relative Time (Hybrid / Double)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.out_dir / "logreg_tol_vs_time.png")
        plt.close()

    def plot_tolerance_vs_accuracy(self, df_B: pd.DataFrame):
        if df_B.empty:
            print("[WARN] plot_tolerance_vs_accuracy: df_B is empty.")
            return
        rel = _relative_hybrid_vs_double(df_B, xcol="tolerance_single", ycol="Accuracy")
        classes_col = _classes_col(df_B)

        plt.figure(figsize=(8, 5))
        for (ds, k), grp in rel.groupby(["DatasetName", classes_col]):
            plt.plot(grp["tolerance_single"], grp["rel"], marker="o", label=f"{ds}-C{k}")
        plt.xscale("log")
        plt.axhline(1.0, linestyle="--", color="gray", linewidth=1, label="Double baseline")
        plt.title("Experiment B: Tolerance vs Accuracy (Hybrid / Double)")
        plt.xlabel("Single-precision Tolerance (log)")
        plt.ylabel("Relative Accuracy (Hybrid / Double)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.out_dir / "logreg_tol_vs_accuracy.png")
        plt.close()


if __name__ == '__main__':
    # Load results
    df_A = pd.read_csv("../Results/logistic_results_expA.csv")
    df_B = pd.read_csv("../Results/logistic_results_expB.csv")

    vis = LogisticVisualizer(out_dir=RESULTS_DIR)
    vis.plot_cap_vs_time(df_A)
    vis.plot_cap_vs_accuracy(df_A)
    vis.plot_tolerance_vs_time(df_B)
    vis.plot_tolerance_vs_accuracy(df_B)

