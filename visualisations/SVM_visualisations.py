import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class SVMVisualizer:
    def __init__(self, output_dir="Results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def plot_cap_vs_accuracy(self, df):
        df = df[df["Mode"] == "Hybrid"]
        grouped = df.groupby("Cap")["Accuracy"].mean().reset_index()
        plt.plot(grouped["Cap"], grouped["Accuracy"], marker='o')
        plt.title("Cap vs Accuracy (SVM Hybrid)")
        plt.xlabel("Single Precision Iteration Cap")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.tight_layout()
       
        plt.savefig(self.output_dir / "svm_cap_vs_accuracy")
        plt.close()
    
    def plot_cap_vs_time(self, df):
        df = df[df["Mode"] == "Hybrid"]
        grouped = df.groupby("Cap")["Time"].mean().reset_index()
        plt.plot(grouped["Cap"], grouped["Time"], marker='o')
        plt.title("Cap vs Time (SVM Hybrid)")
        plt.xlabel("Single Precision Iteration Cap")
        plt.ylabel("Total Time (s)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / "svm_cap_vs_time")
        plt.close()
    
    def plot_tolerance_vs_accuracy(self, df):
        df = df[df["Mode"] == "Hybrid"]
        grouped = df.groupby("Tolerance")["Accuracy"].mean().reset_index()
        plt.plot(grouped["Tolerance"], grouped["Accuracy"], marker='o')
        plt.xscale("log")
        plt.title("Tolerance vs Accuracy (SVM Hybrid)")
        plt.xlabel("Single Precision Tolerance")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / "svm_tolerance_vs_accuracy")
        plt.close()
    
    def plot_tolerance_vs_time(self, df):
        df = df[df["Mode"] == "Hybrid"]
        grouped = df.groupby("Tolerance")["Time"].mean().reset_index()
        plt.plot(grouped["Tolerance"], grouped["Time"], marker='o')
        plt.xscale("log")
        plt.title("Tolerance vs Time (SVM Hybrid)")
        plt.xlabel("Single Precision Tolerance")
        plt.ylabel("Total Time (s)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / "svm_tolerance_vs_time")
        plt.close()
