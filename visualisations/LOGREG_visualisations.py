# visualisations/logistic_vis.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "Results"
os.makedirs(RESULTS_DIR, exist_ok=True)
sns.set(style="whitegrid", font_scale=1.2)

def _make_relative(df: pd.DataFrame, metric: str, key_cols: list) -> pd.DataFrame:
    """
    Return a tidy DataFrame with columns key_cols + [metric] where
    metric = mean(Hybrid metric) / mean(Double metric) per key.
    """
    # group means for each mode
    g = df.groupby(key_cols + ["Mode"])[metric].mean().unstack("Mode")
    # Guard: only keep keys that have BOTH modes
    g = g.dropna(subset=["Hybrid", "Double"], how="any")
    rel = (g["Hybrid"] / g["Double"]).reset_index()
    rel.rename(columns={0: metric}, inplace=True)
    return rel

class LogisticVisualizer:
    # ===== EXPERIMENT A =====
    def plot_cap_vs_time(self, df_A: pd.DataFrame):
        # keys for A: by Cap (and dataset identity in case multiple datasets)
        keys = [c for c in ["DatasetName","DatasetSize","NumClasses","Cap","tolerance_single"] if c in df_A.columns]

        rel = _make_relative(df_A, "Time", keys)

        # numeric and clean
        if "Cap" in rel.columns:
            rel["Cap"] = pd.to_numeric(rel["Cap"], errors="coerce")
        rel = rel.dropna(subset=["Cap", "Time"])

        plt.figure(figsize=(8, 6))
        sns.lineplot(data=rel, x="Cap", y="Time", marker="o", errorbar=None, label="Hybrid")
        plt.axhline(1.0, linestyle="--", color="black", label="Double baseline")
        plt.title("Cap vs Time (Hybrid / Double)")
        plt.xlabel("Cap (Single-Precision Iteration Limit)")
        plt.ylabel("Relative Time to Double")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "logreg_A_cap_vs_time_relative.png"))
        plt.close()

    def plot_cap_vs_accuracy(self, df_A: pd.DataFrame):
        keys = [c for c in ["DatasetName","DatasetSize","NumClasses","Cap","tolerance_single"] if c in df_A.columns]

        rel = _make_relative(df_A, "Accuracy", keys)
        if "Cap" in rel.columns:
            rel["Cap"] = pd.to_numeric(rel["Cap"], errors="coerce")
        rel = rel.dropna(subset=["Cap", "Accuracy"])

        plt.figure(figsize=(8, 6))
        sns.lineplot(data=rel, x="Cap", y="Accuracy", marker="o", errorbar=None, label="Hybrid")
        plt.axhline(1.0, linestyle="--", color="black", label="Double baseline")
        plt.title("Cap vs Accuracy (Hybrid / Double)")
        plt.xlabel("Cap (Single-Precision Iteration Limit)")
        plt.ylabel("Relative Accuracy to Double")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "logreg_A_cap_vs_accuracy_relative.png"))
        plt.close()

    # ===== EXPERIMENT B =====
    def plot_tolerance_vs_time(self, df_B: pd.DataFrame):
        keys = [c for c in ["DatasetName","DatasetSize","NumClasses","tolerance_single"] if c in df_B.columns]

        # coerce tolerance to numeric > 0 for log-scale
        df_B = df_B.copy()
        df_B["tolerance_single"] = pd.to_numeric(df_B["tolerance_single"], errors="coerce")
        df_B = df_B[df_B["tolerance_single"] > 0]

        rel = _make_relative(df_B, "Time", keys)

        plt.figure(figsize=(8, 6))
        ax = sns.lineplot(data=rel, x="tolerance_single", y="Time", marker="o", errorbar=None, label="Hybrid")
        ax.set_xscale("log")
        plt.axhline(1.0, linestyle="--", color="black", label="Double baseline")
        plt.title("Tolerance vs Time (Hybrid / Double)")
        plt.xlabel("Single-Precision Tolerance (log scale)")
        plt.ylabel("Relative Time to Double")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "logreg_B_tolerance_vs_time_relative.png"))
        plt.close()

    def plot_tolerance_vs_accuracy(self, df_B: pd.DataFrame):
        keys = [c for c in ["DatasetName","DatasetSize","NumClasses","tolerance_single"] if c in df_B.columns]

        df_B = df_B.copy()
        df_B["tolerance_single"] = pd.to_numeric(df_B["tolerance_single"], errors="coerce")
        df_B = df_B[df_B["tolerance_single"] > 0]

        rel = _make_relative(df_B, "Accuracy", keys)

        plt.figure(figsize=(8, 6))
        ax = sns.lineplot(data=rel, x="tolerance_single", y="Accuracy", marker="o", errorbar=None, label="Hybrid")
        ax.set_xscale("log")
        plt.axhline(1.0, linestyle="--", color="black", label="Double baseline")
        plt.title("Tolerance vs Accuracy (Hybrid / Double)")
        plt.xlabel("Single-Precision Tolerance (log scale)")
        plt.ylabel("Relative Accuracy to Double")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "logreg_B_tolerance_vs_accuracy_relative.png"))
        plt.close()


