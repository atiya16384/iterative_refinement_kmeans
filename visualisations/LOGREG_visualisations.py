import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

RESULTS_DIR = "Results"
os.makedirs(RESULTS_DIR, exist_ok=True)

class LogisticVisualizer:
    def __init__(self):
        sns.set(style="whitegrid", font_scale=1.2)

    # ===== EXPERIMENT A =====
    def plot_cap_vs_time(self, df_A):
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=df_A, x="Cap", y="Time", hue="Mode", marker="o", ci=None)
        plt.title("Cap vs Time (Experiment A)")
        plt.xlabel("Cap (Single-Precision Iteration Limit)")
        plt.ylabel("Time (s)")
        plt.legend(title="Mode")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "logreg_A_cap_vs_time.png"))
        plt.close()

    def plot_cap_vs_accuracy(self, df_A):
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=df_A, x="Cap", y="Accuracy", hue="Mode", marker="o", ci=None)
        plt.title("Cap vs Accuracy (Experiment A)")
        plt.xlabel("Cap (Single-Precision Iteration Limit)")
        plt.ylabel("Accuracy")
        plt.legend(title="Mode")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "logreg_A_cap_vs_accuracy.png"))
        plt.close()

    # ===== EXPERIMENT B =====
    def plot_tolerance_vs_time(self, df_B):
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=df_B, x="tolerance_single", y="Time", hue="Mode", marker="o", ci=None)
        plt.title("Tolerance vs Time (Experiment B)")
        plt.xlabel("Single-Precision Tolerance")
        plt.ylabel("Time (s)")
        plt.xscale("log")
        plt.legend(title="Mode")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "logreg_B_tolerance_vs_time.png"))
        plt.close()

    def plot_tolerance_vs_accuracy(self, df_B):
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=df_B, x="tolerance_single", y="Accuracy", hue="Mode", marker="o", ci=None)
        plt.title("Tolerance vs Accuracy (Experiment B)")
        plt.xlabel("Single-Precision Tolerance")
        plt.ylabel("Accuracy")
        plt.xscale("log")
        plt.legend(title="Mode")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "logreg_B_tolerance_vs_accuracy.png"))
        plt.close()

