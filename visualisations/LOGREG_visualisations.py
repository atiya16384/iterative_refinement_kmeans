# visualisations/LOGREG_visualisations.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib

# Save one level up, as requested
RESULTS_DIR = pathlib.Path("../Results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def _mode_col(df: pd.DataFrame) -> str:
    # Accept either "Mode" (LR) or "Suite" (k-means style)
    if "Mode" in df.columns:
        return "Mode"
    if "Suite" in df.columns:
        return "Suite"
    raise KeyError("Neither 'Mode' nor 'Suite' column found.")

def _classes_col(df: pd.DataFrame) -> str:
    # LR uses NumClasses; KMeans used NumClusters
    if "NumClasses" in df.columns:
        return "NumClasses"
    if "NumClusters" in df.columns:
        return "NumClusters"
    raise KeyError("Neither 'NumClasses' nor 'NumClusters' column found.")

def _ensure_numeric(df: pd.DataFrame, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def _relative_hybrid_vs_double(
    df: pd.DataFrame,
    xcol: str,          # "Cap" or "tolerance_single"
    ycol: str           # "Time" or "Accuracy"
) -> pd.DataFrame:
    """
    Returns a tidy DataFrame with columns:
      DatasetName, ClassesCol, xcol, rel (Hybrid/Double)
    The mean over n_repeats is baked in via groupby(...).mean().
    """
    mode_col = _mode_col(df)
    classes_col = _classes_col(df)

    # Coerce numeric axes
    df = _ensure_numeric(df, [xcol, ycol, classes_col, "DatasetSize"])

    # Average over repeats for stability
    keys = ["DatasetName", classes_col, mode_col, xcol]
    g = df.groupby(keys, dropna=False)[ycol].mean().reset_index()

    # Pivot to align Hybrid & Double per (dataset, classes, x)
    pivot = g.pivot_table(
        index=["DatasetName", classes_col, xcol],
        columns=mode_col,
        values=ycol,
        aggfunc="mean"
    )

    # Keep only pairs with both modes
    needed = {"Hybrid", "Double"}
    if not needed.issubset(pivot.columns):
        # If some are missing, fill with NaN and we’ll drop them
        for c in needed:
            if c not in pivot.columns:
                pivot[c] = np.nan
    pivot = pivot.dropna(subset=["Hybrid", "Double"], how="any")

    # Compute relative = Hybrid / Double
    rel = (pivot["Hybrid"] / pivot["Double"]).reset_index()
    rel.rename(columns={0: "rel"}, inplace=True)
    rel["rel"] = pd.to_numeric(rel["rel"], errors="coerce")
    # Sort by xcol
    rel = rel.sort_values([ "DatasetName", classes_col, xcol ])
    return rel

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


if __name__ == "__main__":
    # Optional CLI usage
    df_A = pd.read_csv("../Results/logistic_results_expA.csv")
    df_B = pd.read_csv("../Results/logistic_results_expB.csv")

    vis = LogisticVisualizer(out_dir=RESULTS_DIR)
    vis.plot_cap_vs_time(df_A)
    vis.plot_cap_vs_accuracy(df_A)
    vis.plot_tolerance_vs_time(df_B)
    vis.plot_tolerance_vs_accuracy(df_B)

