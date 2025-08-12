# visualisations/SVM_visualisations.py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class SVMVisualizer:
    def __init__(self, output_dir="Results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # visualisations/SVM_visualisations.py
    def plot_cap_vs_accuracy(self, df):
        if df.empty: return
        base_acc = df[df["Suite"] == "Double"]["Accuracy"].mean()
        if not np.isfinite(base_acc) or base_acc == 0: return
    
        hyb = (df[df["Suite"] == "Hybrid"]
               .groupby("Cap")[["Accuracy"]].mean()
               .reset_index()
               .sort_values("Cap"))
        if hyb.empty: return
    
        hyb["Accuracy_norm"] = hyb["Accuracy"] / base_acc
        plt.plot(hyb["Cap"], hyb["Accuracy_norm"], marker="o")
        plt.axhline(1.0, ls="--", lw=1)
        plt.title("Cap vs Accuracy (SVM Hybrid)")
        plt.xlabel("Single Precision Iteration Cap")
        plt.ylabel("Accuracy (relative to double)")
        plt.grid(True); plt.tight_layout()
        plt.savefig(self.output_dir / "svm_cap_vs_accuracy.png"); plt.close()
    
    def plot_cap_vs_time(self, df):
        if df.empty: return
        base_time = df[df["Suite"] == "Double"]["Time"].mean()
        if not np.isfinite(base_time) or base_time == 0: return
    
        hyb = (df[df["Suite"] == "Hybrid"]
               .groupby("Cap")[["Time"]].mean()
               .reset_index()
               .sort_values("Cap"))
        if hyb.empty: return
    
        hyb["Time_norm"] = hyb["Time"] / base_time
        plt.plot(hyb["Cap"], hyb["Time_norm"], marker="o")
        plt.axhline(1.0, ls="--", lw=1)
        plt.title("Cap vs Time (SVM Hybrid / Double)")
        plt.xlabel("Single Precision Iteration Cap")
        plt.ylabel("Time (relative to double)")
        plt.grid(True); plt.tight_layout()
        plt.savefig(self.output_dir / "svm_cap_vs_time.png"); plt.close()

    def plot_tolerance_vs_accuracy(self, df):
        if df.empty: return
        # baseline is the single Double run(s) at fixed tol_double_B
        base_acc = df[df["Suite"] == "Double"]["Accuracy"].mean()
        if not np.isfinite(base_acc) or base_acc == 0: return
        hyb = (df[df["Suite"] == "Hybrid"]
               .groupby("Tolerance")[["Accuracy"]].mean()
               .reset_index()
               .sort_values("Tolerance"))
        if hyb.empty: return

        hyb["Accuracy_norm"] = hyb["Accuracy"] / base_acc
        plt.plot(hyb["Tolerance"], hyb["Accuracy_norm"], marker="o")
        plt.axhline(1.0, linestyle="--", linewidth=1)  # dotted baseline
        plt.xscale("log")
        plt.title("Tolerance vs Accuracy (SVM Hybrid)")
        plt.xlabel("Single Precision Tolerance")
        plt.ylabel("Accuracy (relative to double)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / "svm_tolerance_vs_accuracy.png")
        plt.close()

    def plot_tolerance_vs_time(self, df):
        if df.empty: return
        base_time = df[df["Suite"] == "Double"]["Time"].mean()
        if not np.isfinite(base_time) or base_time == 0: return
        hyb = (df[df["Suite"] == "Hybrid"]
               .groupby("Tolerance")[["Time"]].mean()
               .reset_index()
               .sort_values("Tolerance"))
        if hyb.empty: return

        hyb["Time_norm"] = hyb["Time"] / base_time
        plt.plot(hyb["Tolerance"], hyb["Time_norm"], marker="o")
        plt.axhline(1.0, linestyle="--", linewidth=1)  # dotted baseline
        plt.xscale("log")
        plt.title("Tolerance vs Time (SVM Hybrid)")
        plt.xlabel("Single Precision Tolerance")
        plt.ylabel("Time (relative to double)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / "svm_tolerance_vs_time.png")
        plt.close()
