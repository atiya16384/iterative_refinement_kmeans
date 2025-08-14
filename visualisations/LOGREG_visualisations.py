import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

RESULTS_DIR = "Results"
os.makedirs(RESULTS_DIR, exist_ok=True)

class LogisticVisualizer:
    def __init__(self):
        sns.set(style="whitegrid", font_scale=1.2)

    def _relative_to_double(self, df, metric_col):
        """Returns df with Hybrid metric divided by Double metric per group."""
        # group by dataset params and match Hybrid to Double
        rel_df = []
        group_cols = [c for c in df.columns if c not in [metric_col, 'Mode', 'Accuracy', 'Time']]
        for _, group in df.groupby(group_cols):
            double_val = group.loc[group['Mode'] == 'Double', metric_col].mean()
            hybrid_rows = group[group['Mode'] == 'Hybrid'].copy()
            hybrid_rows[metric_col] = hybrid_rows[metric_col] / double_val if double_val != 0 else float('nan')
            rel_df.append(hybrid_rows)
        return pd.concat(rel_df, ignore_index=True)

    # ===== EXPERIMENT A =====
    def plot_cap_vs_time(self, df_A):
        df_rel = self._relative_to_double(df_A, "Time")
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=df_rel, x="Cap", y="Time", marker="o", ci=None, label="Hybrid")
        plt.axhline(1.0, color="black", linestyle="--", label="Double Baseline")
        plt.title("Cap vs Time (Hybrid relative to Double)")
        plt.xlabel("Cap (Single-Precision Iteration Limit)")
        plt.ylabel("Relative Time to Double")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "logreg_A_cap_vs_time_relative.png"))
        plt.close()

    def plot_cap_vs_accuracy(self, df_A):
        df_rel = self._relative_to_double(df_A, "Accuracy")
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=df_rel, x="Cap", y="Accuracy", marker="o", ci=None, label="Hybrid")
        plt.axhline(1.0, color="black", linestyle="--", label="Double Baseline")
        plt.title("Cap vs Accuracy (Hybrid relative to Double)")
        plt.xlabel("Cap (Single-Precision Iteration Limit)")
        plt.ylabel("Relative Accuracy to Double")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "logreg_A_cap_vs_accuracy_relative.png"))
        plt.close()

    # ===== EXPERIMENT B =====
    def plot_tolerance_vs_time(self, df_B):
        df_rel = self._relative_to_double(df_B, "Time")
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=df_rel, x="tolerance_single", y="Time", marker="o", ci=None, label="Hybrid")
        plt.axhline(1.0, color="black", linestyle="--", label="Double Baseline")
        plt.xscale("log")
        plt.title("Tolerance vs Time (Hybrid relative to Double)")
        plt.xlabel("Single-Precision Tolerance")
        plt.ylabel("Relative Time to Double")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "logreg_B_tolerance_vs_time_relative.png"))
        plt.close()

    def plot_tolerance_vs_accuracy(self, df_B):
        df_rel = self._relative_to_double(df_B, "Accuracy")
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=df_rel, x="tolerance_single", y="Accuracy", marker="o", ci=None, label="Hybrid")
        plt.axhline(1.0, color="black", linestyle="--", label="Double Baseline")
        plt.xscale("log")
        plt.title("Tolerance vs Accuracy (Hybrid relative to Double)")
        plt.xlabel("Single-Precision Tolerance")
        plt.ylabel("Relative Accuracy to Double")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "logreg_B_tolerance_vs_accuracy_relative.png"))
        plt.close()


