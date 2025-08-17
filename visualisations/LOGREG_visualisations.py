# visualisations/LOGREG_visualisations.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "Results"
os.makedirs(RESULTS_DIR, exist_ok=True)
sns.set(style="whitegrid", font_scale=1.2)

def _make_relative(df: pd.DataFrame, metric: str, key_cols: list) -> pd.DataFrame:
    g = df.groupby(key_cols + ["Mode"], dropna=False)[metric].mean().unstack("Mode")
    if not {"Hybrid", "Double"}.issubset(g.columns):
        return pd.DataFrame(columns=key_cols+[metric])
    g = g.dropna(subset=["Hybrid", "Double"], how="any")
    rel = (g["Hybrid"] / g["Double"]).reset_index()
    rel.rename(columns={0: metric}, inplace=True)
    return rel

class LogisticVisualizer:
    def plot_cap_vs_time(self, df_A):
        df = df_A.copy()
        # sanitize dtypes before grouping
        for c in ("Cap", "DatasetSize", "NumClasses", "tolerance_single"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        keys = [c for c in ["DatasetName", "Cap"] if c in df.columns]
        rel = _make_relative(df, "Time", keys).dropna(subset=["Cap", "Time"])

        if rel.empty:
            print("[WARN] plot_cap_vs_time: nothing to plot (no Hybrid/Double pairs).")
            return

        plt.figure(figsize=(8,6))
        sns.lineplot(data=rel.sort_values("Cap"), x="Cap", y="Time", marker="o", errorbar=None, label="Hybrid")
        plt.axhline(1.0, linestyle="--", color="black", label="Double baseline")
        plt.title("Cap vs Time (Hybrid / Double)")
        plt.xlabel("Cap (Single-Precision Iteration Limit)")
        plt.ylabel("Relative Time to Double")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "logreg_A_cap_vs_time_relative.png"))
        plt.close()

    def plot_cap_vs_accuracy(self, df_A):
        df = df_A.copy()
        for c in ("Cap", "DatasetSize", "NumClasses", "tolerance_single"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        keys = [c for c in ["DatasetName", "Cap"] if c in df.columns]
        rel = _make_relative(df, "Accuracy", keys).dropna(subset=["Cap", "Accuracy"])

        if rel.empty:
            print("[WARN] plot_cap_vs_accuracy: nothing to plot.")
            return

        plt.figure(figsize=(8,6))
        sns.lineplot(data=rel.sort_values("Cap"), x="Cap", y="Accuracy", marker="o", errorbar=None, label="Hybrid")
        plt.axhline(1.0, linestyle="--", color="black", label="Double baseline")
        plt.title("Cap vs Accuracy (Hybrid / Double)")
        plt.xlabel("Cap (Single-Precision Iteration Limit)")
        plt.ylabel("Relative Accuracy to Double")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "logreg_A_cap_vs_accuracy_relative.png"))
        plt.close()

    def plot_tolerance_vs_time(self, df_B):
        df = df_B.copy()
        if "tolerance_single" in df.columns:
            df["tolerance_single"] = pd.to_numeric(df["tolerance_single"], errors="coerce")
            df = df[df["tolerance_single"] > 0]
        keys = [c for c in ["DatasetName", "tolerance_single"] if c in df.columns]
        rel = _make_relative(df, "Time", keys)

        if rel.empty:
            print("[WARN] plot_tolerance_vs_time: nothing to plot.")
            return

        plt.figure(figsize=(8,6))
        ax = sns.lineplot(data=rel.sort_values("tolerance_single"),
                          x="tolerance_single", y="Time", marker="o", errorbar=None, label="Hybrid")
        ax.set_xscale("log")
        plt.axhline(1.0, linestyle="--", color="black", label="Double baseline")
        plt.title("Tolerance vs Time (Hybrid / Double)")
        plt.xlabel("Single-Precision Tolerance (log scale)")
        plt.ylabel("Relative Time to Double")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "logreg_B_tolerance_vs_time_relative.png"))
        plt.close()

    def plot_tolerance_vs_accuracy(self, df_B):
        df = df_B.copy()
        if "tolerance_single" in df.columns:
            df["tolerance_single"] = pd.to_numeric(df["tolerance_single"], errors="coerce")
            df = df[df["tolerance_single"] > 0]
        keys = [c for c in ["DatasetName", "tolerance_single"] if c in df.columns]
        rel = _make_relative(df, "Accuracy", keys)

        if rel.empty:
            print("[WARN] plot_tolerance_vs_accuracy: nothing to plot.")
            return

        plt.figure(figsize=(8,6))
        ax = sns.lineplot(data=rel.sort_values("tolerance_single"),
                          x="tolerance_single", y="Accuracy", marker="o", errorbar=None, label="Hybrid")
        ax.set_xscale("log")
        plt.axhline(1.0, linestyle="--", color="black", label="Double baseline")
        plt.title("Tolerance vs Accuracy (Hybrid / Double)")
        plt.xlabel("Single-Precision Tolerance (log scale)")
        plt.ylabel("Relative Accuracy to Double")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "logreg_B_tolerance_vs_accuracy_relative.png"))
        plt.close()


