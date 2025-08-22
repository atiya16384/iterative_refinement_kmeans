
# Run: python3 logistic_visualisations.py


import pathlib
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------- CONFIG -----------------
CSV_PATH = "../Results/results_all.csv"
OUTDIR   = "../Results/Figures"

APPROACHES = ["hybrid(f32→f64)", "multistage-IR", "adaptive-precision"]   # lines we plot
BASELINES  = ("single(f32)", "double(f64)")                               # for normalisation only

PARAMS  = ["lambda", "tol", "max_iter", "max_iter_single"]
METRICS = ["logloss", "time_sec", "roc_auc", "pr_auc"]

SLICE_KEYS = ["dataset", "penalty", "alpha", "solver"]  # solver gets its own folder
PALETTE = "tab10"
# ------------------------------------------


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _canon_x(x: pd.Series, name: str) -> pd.Series:
    """
    Snap x-values to stable bins:
      - lambda/tol: round in log10 space
      - max_iter/*: cast to int
      - else: round to 12 sig figs
    Returns a float suitable for grouping/merging and plotting.
    """
    x = _safe_num(x)
    if name in {"lambda", "tol"}:
        with np.errstate(divide="ignore"):
            lx = np.log10(x.replace(0, np.nan))
        lx = np.round(lx, 10)
        out = (10.0 ** lx).fillna(0.0)
        return out.astype(float)
    elif name in {"max_iter", "max_iter_single"}:
        return x.fillna(0).astype(np.int64).astype(float)
    else:
        return np.round(x, 12)


class LogisticVisualizer:
    def __init__(self, csv_path: str = CSV_PATH, outdir: str = OUTDIR):
        self.csv_path = pathlib.Path(csv_path)
        self.outdir   = pathlib.Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.df = self._load_and_prepare()

    # ---------- load & average repeats ----------
    def _load_and_prepare(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)

        # keep only approaches we need (variants + baselines for ratios)
        keep = set(APPROACHES) | set(BASELINES)
        df = df[df["approach"].isin(keep)].copy()

        # numeric params
        for c in PARAMS:
            if c in df.columns:
                df[c] = _safe_num(df[c])

        # strict average over repeats by FULL key
        full_key = [
            "dataset", "penalty", "alpha", "solver", "approach",
            "lambda", "tol", "max_iter", "max_iter_single"
        ]
        full_key = [k for k in full_key if k in df.columns]
        agg_cols = [c for c in METRICS + ["iters", "iters_single", "iters_double"] if c in df.columns]

        if agg_cols:
            df = (
                df.groupby(full_key, dropna=False)[agg_cols]
                  .mean()
                  .reset_index()
            )

        # Pre-compute canonical x columns for each PARAM for robust grouping
        for xp in PARAMS:
            if xp in df.columns:
                df[f"{xp}__bin"] = _canon_x(df[xp], xp)

        return df

    # ---------- split per solver ----------
    def _per_solver(self) -> List[Tuple[str, pd.DataFrame, pathlib.Path]]:
        if "solver" not in self.df.columns:
            return [("unspecified", self.df.copy(), self.outdir / "solver=unspecified")]
        solvers = self.df["solver"].dropna().unique().tolist() or ["unspecified"]
        out = []
        for s in solvers:
            sdf = self.df if s == "unspecified" else self.df[self.df["solver"] == s]
            if sdf.empty: 
                continue
            root = self.outdir / (f"solver={s}" if s != "unspecified" else "solver=unspecified")
            root.mkdir(parents=True, exist_ok=True)
            out.append((str(s), sdf.copy(), root))
        return out

    # ---------- aggregation helpers ----------
    @staticmethod
    def _aggregate_abs(df: pd.DataFrame, xp: str, metric: str) -> pd.DataFrame:
        """
        Build ONE point per x (xp__bin), per slice+approach:
            keys = [dataset, penalty, alpha, solver, approach, xp__bin]
            value = mean(metric)  (averaging across ALL other knobs)
        """
        xp_bin = f"{xp}__bin"
        keys = [k for k in ["dataset", "penalty", "alpha", "solver", "approach", xp_bin] if k in df.columns]
        if xp_bin not in df.columns or metric not in df.columns:
            return pd.DataFrame(columns=keys + [metric, xp])

        g = (
            df.groupby(keys, dropna=False)[metric]
              .mean()
              .reset_index()
        )
        g[xp] = g[xp_bin].astype(float)
        return g

    @staticmethod
    def _aggregate_rel(df: pd.DataFrame, xp: str, metric: str, baseline: str) -> pd.DataFrame:
        """
        Relative = (mean metric per x for approach) / (mean metric per x for baseline),
        aligned by slice and xp__bin.
        Returns rows only for APPROACHES (baselines excluded from output).
        """
        xp_bin = f"{xp}__bin"
        base_keys = [k for k in ["dataset", "penalty", "alpha", "solver", xp_bin] if k in df.columns]

        # mean per-x for each approach
        abs_df = LogisticVisualizer._aggregate_abs(df, xp, metric)
        if abs_df.empty:
            return abs_df

        # split variant vs baseline
        base_df = abs_df[abs_df["approach"] == baseline].rename(columns={metric: "BASE"})
        var_df  = abs_df[abs_df["approach"].isin(APPROACHES)].copy()

        # merge 1:1 on slice + xp_bin
        m = pd.merge(
            var_df, 
            base_df[base_keys + ["BASE"]], 
            on=base_keys, how="inner", validate="many_to_one"
        )
        if m.empty:
            return pd.DataFrame(columns=list(var_df.columns) + [f"{metric}_rel"])

        m[f"{metric}_rel"] = m[metric] / m["BASE"]
        return m

    # ---------- plotting primitives ----------
    @staticmethod
    def _apply_axes_style(ax, xp: str, ylab: str, title: str, relative: bool = False):
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(xp)
        ax.set_ylabel(ylab)
        if xp in {"lambda", "tol"}:
            ax.set_xscale("log")
        if relative:
            ax.axhline(1.0, ls="--", c="gray", lw=1.0, label="baseline")
        ax.grid(True, ls="--", alpha=0.4)

    # ---------- high-level plotters ----------
    def plot_absolute(self, xp: str, metric: str):
        ylab = f"{metric} (seconds)" if metric == "time_sec" else metric
        for _, sdf, root in self._per_solver():
            # slice by dataset/penalty/alpha so each figure is coherent
            slice_cols = [c for c in ["dataset", "penalty", "alpha"] if c in sdf.columns]
            groups = sdf.groupby(slice_cols, dropna=False) if slice_cols else [({}, sdf)]

            for keys, df_slice in groups:
                title_prefix = " | ".join([f"{slice_cols[i]}={keys[i]}" for i in range(len(slice_cols))]) if slice_cols else "All"
                out_dir = root / "absolute"; out_dir.mkdir(parents=True, exist_ok=True)

                # aggregate once for the slice
                abs_df = self._aggregate_abs(df_slice, xp, metric)
                if abs_df.empty: 
                    continue

                # limit to the three approaches
                plot_df = abs_df[abs_df["approach"].isin(APPROACHES)].copy()
                if plot_df.empty: 
                    continue

                plt.figure(figsize=(8, 5))
                sns.lineplot(
                    data=plot_df.sort_values(xp),
                    x=xp, y=metric, hue="approach", style="approach",
                    marker="o", errorbar=("ci", 95), palette=PALETTE
                )
                self._apply_axes_style(
                    plt.gca(), xp, ylab, f"{title_prefix}\n{ylab} vs {xp}", relative=False
                )
                plt.legend(title=None)
                fname = f"{title_prefix.replace(' | ', '__').replace('=','-')}__{metric}_vs_{xp}.png".replace(" ", "")
                plt.tight_layout()
                plt.savefig(out_dir / "png" / fname, dpi=180)
                plt.close()

    def plot_relative(self, xp: str, metric: str, baseline: str):
        ylab = f"{metric} / {baseline}"
        for _, sdf, root in self._per_solver():
            slice_cols = [c for c in ["dataset", "penalty", "alpha"] if c in sdf.columns]
            groups = sdf.groupby(slice_cols, dropna=False) if slice_cols else [({}, sdf)]
            sub_folder = "relative_single_baseline" if baseline.startswith("single") else "relative_double_baseline"

            for keys, df_slice in groups:
                title_prefix = " | ".join([f"{slice_cols[i]}={keys[i]}" for i in range(len(slice_cols))]) if slice_cols else "All"
                out_dir = root / sub_folder; out_dir.mkdir(parents=True, exist_ok=True)

                rel_df = self._aggregate_rel(df_slice, xp, metric, baseline)
                if rel_df.empty: 
                    continue

                rel_df = rel_df[rel_df["approach"].isin(APPROACHES)].copy()
                if rel_df.empty:
                    continue

                ycol = f"{metric}_rel"
                plt.figure(figsize=(8, 5))
                sns.lineplot(
                    data=rel_df.sort_values(xp),
                    x=xp, y=ycol, hue="approach", style="approach",
                    marker="o", errorbar=("ci", 95), palette=PALETTE
                )
                self._apply_axes_style(
                    plt.gca(), xp, ylab, f"{title_prefix}\n{ylab} vs {xp}", relative=True
                )
                plt.legend(title=None)
                fname = f"{title_prefix.replace(' | ', '__').replace('=','-')}__{metric}_vs_{xp}__rel_{baseline.replace('(','').replace(')','')}.png".replace(" ", "")
                plt.tight_layout()
                (out_dir / "png").mkdir(parents=True, exist_ok=True)
                plt.savefig(out_dir / "png" / fname, dpi=180)
                plt.close()

    # ---------- convenience: produce the suite ----------
    def make_all(self):
        # absolute
        for xp in PARAMS:
            for metric in METRICS:
                self.plot_absolute(xp, metric)

        # relative (two baselines)
        for xp in PARAMS:
            for metric in METRICS:
                for base in BASELINES:
                    self.plot_relative(xp, metric, baseline=base)


# ----------------- RUN -----------------
if __name__ == "__main__":
    sns.set_context("talk")
    sns.set_style("whitegrid")
    viz = LogisticVisualizer()
    viz.make_all()
    print(f"Figures saved under {OUTDIR} / solver=<solver> / (absolute|relative_*) / png")


