# visualisations/ENET_visualisations.py
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Default output folder (one level up, same as your other plots)
DEFAULT_OUT = pathlib.Path("../Results")
DEFAULT_OUT.mkdir(parents=True, exist_ok=True)

class ENetVisualizer:
    def __init__(self, out_dir: pathlib.Path = DEFAULT_OUT):
        self.out_dir = pathlib.Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- helpers ----------
    @staticmethod
    def _normalize_by_double(df: pd.DataFrame, sweep_col: str, y_col: str) -> pd.DataFrame:
        """
        For each (DatasetName, NumFeatures, <sweep_col>) pair,
        divide y_col by the corresponding Double baseline value.
        Returns rows for BOTH modes with y_col normalized;
        call-side will filter Mode == 'Hybrid' when plotting.
        """
        req_cols = {"DatasetName", "NumFeatures", "Mode", sweep_col, y_col}
        missing = req_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns for normalization: {sorted(missing)}")

        parts = []
        for (ds, d), g in df.groupby(["DatasetName", "NumFeatures"]):
            doubles = g[g["Mode"] == "Double"][[sweep_col, y_col]].dropna()
            if doubles.empty:
                continue
            # map sweep value -> double metric
            base = dict(zip(doubles[sweep_col].values, doubles[y_col].values))

            # only keep rows whose sweep value exists in the double baseline
            sub = g[g[sweep_col].isin(base.keys())].copy()
            sub[y_col] = sub.apply(lambda r: r[y_col] / base[r[sweep_col]], axis=1)
            parts.append(sub)

        if not parts:
            return pd.DataFrame(columns=list(df.columns))
        return pd.concat(parts, ignore_index=True)

    def _lineplot_norm(self, df_norm: pd.DataFrame, x: str, y: str, title: str, fname: str, logx=False):
        if df_norm.empty:
            print(f"[warn] nothing to plot for {fname} (empty frame after normalization)")
            return
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=df_norm[df_norm["Mode"] == "Hybrid"],
                     x=x, y=y, hue="DatasetName", style="NumFeatures",
                     marker="o", ci="sd")
        if logx:
            plt.xscale("log")
        plt.axhline(1.0, ls="--", color="gray", linewidth=1, label="Double baseline")
        plt.title(title)
        plt.ylabel(f"Relative {y} (Hybrid / Double)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out = self.out_dir / fname
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved: {out}")

    # ---------- Experiment A (cap sweep) ----------
    def plot_cap_vs_time(self, df_A: pd.DataFrame):
        df_norm = self._normalize_by_double(df_A, "Cap", "Time")
        self._lineplot_norm(df_norm, "Cap", "Time",
                            "Experiment A: Cap vs Runtime (Hybrid / Double)",
                            "enet_cap_vs_time.png", logx=False)

    def plot_cap_vs_r2(self, df_A: pd.DataFrame):
        df_norm = self._normalize_by_double(df_A, "Cap", "R2")
        self._lineplot_norm(df_norm, "Cap", "R2",
                            "Experiment A: Cap vs R² (Hybrid / Double)",
                            "enet_cap_vs_r2.png", logx=False)

    def plot_cap_vs_mse(self, df_A: pd.DataFrame):
        # For MSE, <1 is better than baseline
        df_norm = self._normalize_by_double(df_A, "Cap", "MSE")
        self._lineplot_norm(df_norm, "Cap", "MSE",
                            "Experiment A: Cap vs MSE (Hybrid / Double)",
                            "enet_cap_vs_mse.png", logx=False)

    # ---------- Experiment B (tolerance sweep) ----------
    def plot_tol_vs_time(self, df_B: pd.DataFrame):
        df_norm = self._normalize_by_double(df_B, "tolerance_single", "Time")
        self._lineplot_norm(df_norm, "tolerance_single", "Time",
                            "Experiment B: Tolerance vs Runtime (Hybrid / Double)",
                            "enet_tol_vs_time.png", logx=True)

    def plot_tol_vs_r2(self, df_B: pd.DataFrame):
        df_norm = self._normalize_by_double(df_B, "tolerance_single", "R2")
        self._lineplot_norm(df_norm, "tolerance_single", "R2",
                            "Experiment B: Tolerance vs R² (Hybrid / Double)",
                            "enet_tol_vs_r2.png", logx=True)

    def plot_tol_vs_mse(self, df_B: pd.DataFrame):
        df_norm = self._normalize_by_double(df_B, "tolerance_single", "MSE")
        self._lineplot_norm(df_norm, "tolerance_single", "MSE",
                            "Experiment B: Tolerance vs MSE (Hybrid / Double)",
                            "enet_tol_vs_mse.png", logx=True)


if __name__ == "__main__":
    # Load CSVs produced by ENET_main.py
    results_dir = pathlib.Path("Results")  # where ENET_main writes CSVs
    df_A = pd.read_csv(results_dir / "enet_results_expA.csv")
    df_B = pd.read_csv(results_dir / "enet_results_expB.csv")

    vis = ENetVisualizer(out_dir=DEFAULT_OUT)

    # Exp A
    vis.plot_cap_vs_time(df_A)
    vis.plot_cap_vs_r2(df_A)
    vis.plot_cap_vs_mse(df_A)

    # Exp B
    vis.plot_tol_vs_time(df_B)
    vis.plot_tol_vs_r2(df_B)
    vis.plot_tol_vs_mse(df_B)
    # Exp B
    vis.plot_tol_vs_time(df_B)
    vis.plot_tol_vs_r2(df_B)
    vis.plot_tol_vs_mse(df_B)
