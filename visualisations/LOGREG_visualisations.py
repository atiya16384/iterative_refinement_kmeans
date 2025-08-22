
# Run: python3 logistic_visualisations.py

import pathlib
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------- CONFIG -----------------
CSV_PATH = "../Results/results_all.csv"
OUTDIR   = "./Figures"

VARIANTS  = ["hybrid(f32→f64)", "multistage-IR", "adaptive-precision"]
BASELINES = ("single(f32)", "double(f64)")

PARAMS  = ["lambda", "tol", "max_iter", "max_iter_single"]
METRICS = ["logloss", "time_sec", "roc_auc", "pr_auc"]

SLICE_KEYS = ["dataset", "penalty", "alpha", "solver"]  # we split folders by solver
# ------------------------------------------


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _canon_x(x: pd.Series, name: str) -> pd.Series:
    """Snap x to stable bins to avoid float drift."""
    x = _safe_num(x)
    if name in {"lambda", "tol"}:
        with np.errstate(divide="ignore"):
            lx = np.log10(x.replace(0, np.nan))
        lx = np.round(lx, 10)
        return (10.0 ** lx).fillna(0.0).astype(float)
    elif name in {"max_iter", "max_iter_single"}:
        return x.fillna(0).astype(np.int64).astype(float)
    else:
        return np.round(x, 12)


class LogisticVisualizerKM:
    def __init__(self, csv_path=CSV_PATH, outdir=OUTDIR):
        self.csv_path = pathlib.Path(csv_path)
        self.outdir   = pathlib.Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.df = self._load_avg_repeats()
        # precompute canonical bins for all PARAMS
        for xp in PARAMS:
            if xp in self.df.columns:
                self.df[f"{xp}__bin"] = _canon_x(self.df[xp], xp)

    # ------- load + average repeats by FULL key -------
    def _load_avg_repeats(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)

        keep = set(VARIANTS) | set(BASELINES)
        df = df[df["approach"].isin(keep)].copy()

        for c in PARAMS:
            if c in df.columns:
                df[c] = _safe_num(df[c])

        full_key = [
            "dataset", "penalty", "alpha", "solver", "approach",
            "lambda", "tol", "max_iter", "max_iter_single"
        ]
        full_key = [k for k in full_key if k in df.columns]
        agg_cols = [c for c in METRICS + ["iters", "iters_single", "iters_double"] if c in df.columns]

        if agg_cols:
            df = df.groupby(full_key, dropna=False)[agg_cols].mean().reset_index()

        return df

    # ------- helpers to define true per-line slices -------
    @staticmethod
    def _line_group_cols(xp: str) -> List[str]:
        """For a swept x param, lines must hold all other params fixed."""
        others = [p for p in PARAMS if p != xp]
        # use the canonical bins to avoid float drift in grouping
        others_bin = [f"{p}__bin" for p in others]
        return ["dataset", "penalty", "alpha", "solver", "approach"] + others_bin

    # ------- ABSOLUTE: faint lines + bold median -------
    def _plot_absolute(self, df_solver: pd.DataFrame, xp: str, metric: str, out_dir: pathlib.Path, title_prefix: str):
        if xp not in df_solver.columns or metric not in df_solver.columns:
            return
        xp_bin = f"{xp}__bin"
        if xp_bin not in df_solver.columns:
            return

        sdf = df_solver[df_solver["approach"].isin(VARIANTS)].copy()
        if sdf.empty:
            return

        # one per-line slice = hold all other params fixed (use bins)
        gcols = [c for c in self._line_group_cols(xp) if c in sdf.columns]

        # Build plot frame with canonical x
        plot_df = sdf.copy()
        plot_df[xp] = plot_df[xp_bin]

        if not gcols:
            return

        # ensure at least two unique xs in total
        if plot_df[xp].nunique(dropna=True) < 2:
            return

        fig, ax = plt.subplots(figsize=(7.5, 5.0))

        # faint lines per true slice
        for _, g in plot_df.groupby(gcols, dropna=False):
            # need >=2 x to draw a line
            gg = g.sort_values(xp)
            if gg[xp].nunique() < 2:
                continue
            ax.plot(gg[xp], gg[metric], marker="o", alpha=0.30)

        # bold median across slices at each x
        med = (
            plot_df.groupby(xp, dropna=False)[metric]
                   .median()
                   .reset_index().sort_values(xp)
        )
        ax.plot(med[xp], med[metric], marker="o", lw=2.2, label="Median")

        if xp in {"lambda", "tol"}:
            ax.set_xscale("log")

        ax.set_title(f"{title_prefix}\n{metric} vs {xp}")
        ax.set_xlabel(xp)
        ax.set_ylabel("time (sec)" if metric == "time_sec" else metric)
        ax.grid(True, ls="--", alpha=0.45)
        ax.legend()

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "png").mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_dir / "png" / f"{metric}_vs_{xp}.png", dpi=180)
        plt.close(fig)

    # ------- RELATIVE core (per-slice ratio) -------
    @staticmethod
    def _relative_df(df: pd.DataFrame, xp: str, metric: str, baseline: str) -> pd.DataFrame:
        xp_bin = f"{xp}__bin"

        # keys to enforce "same slice" (others fixed) + same x
        others_bin = [f"{p}__bin" for p in PARAMS if p != xp]
        keys = ["dataset", "penalty", "alpha", "solver"] + others_bin + [xp_bin]

        # variant and baseline reduced to (keys, metric)
        var = (df[df["approach"].isin(VARIANTS)]
               .groupby(keys + ["approach"], dropna=False)[metric].mean()
               .reset_index())
        base = (df[df["approach"] == baseline]
                .groupby(keys, dropna=False)[metric].mean()
                .reset_index()
                .rename(columns={metric: "BASE"}))

        m = var.merge(base, on=keys, how="inner")
        if m.empty:
            return m
        m["Rel"] = m[metric] / m["BASE"]
        # use canonical x for plotting
        m[xp] = m[xp_bin]
        return m

    # ------- RELATIVE: faint lines + bold median -------
    def _plot_relative(self, df_solver: pd.DataFrame, xp: str, metric: str,
                       baseline: str, out_dir: pathlib.Path, title_prefix: str):
        if xp not in df_solver.columns or metric not in df_solver.columns:
            return
        xp_bin = f"{xp}__bin"
        if xp_bin not in df_solver.columns:
            return

        rel = self._relative_df(df_solver, xp, metric, baseline)
        if rel.empty or rel[xp].nunique(dropna=True) < 2:
            return

        # group columns for true per-line slices (include approach)
        gcols = [c for c in (["dataset","penalty","alpha","solver","approach"]
                             + [f"{p}__bin" for p in PARAMS if p != xp]) if c in rel.columns]

        fig, ax = plt.subplots(figsize=(7.5, 5.0))

        # faint lines per slice
        for _, g in rel.groupby(gcols, dropna=False):
            gg = g.sort_values(xp)
            if gg[xp].nunique() < 2:
                continue
            ax.plot(gg[xp], gg["Rel"], marker="o", alpha=0.30)

        # bold median across slices at each x
        med = (
            rel.groupby(xp, dropna=False)["Rel"]
               .median()
               .reset_index().sort_values(xp)
        )
        ax.plot(med[xp], med["Rel"], marker="o", lw=2.2, label="Median")

        if xp in {"lambda", "tol"}:
            ax.set_xscale("log")

        ax.axhline(1.0, ls="--", c="gray", lw=1.0, label=f"{baseline} (y=1)")
        ax.set_title(f"{title_prefix}\n{metric} / {baseline} vs {xp}")
        ax.set_xlabel(xp)
        ax.set_ylabel(f"{metric} / {baseline}")
        ax.grid(True, ls="--", alpha=0.45)
        ax.legend()

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "png").mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_dir / "png" / f"{metric}_vs_{xp}__rel.png", dpi=180)
        plt.close(fig)

    # ------- main driver -------
    def make_all(self):
        # split output by solver
        solvers = self.df["solver"].dropna().unique().tolist() if "solver" in self.df.columns else ["unspecified"]
        for solver_name in solvers:
            if solver_name == "unspecified":
                df_solver = self.df.copy()
            else:
                df_solver = self.df[self.df["solver"] == solver_name].copy()
            root = self.outdir / f"solver={solver_name if pd.notna(solver_name) else 'unspecified'}"

            # title prefix (dataset/penalty/alpha vary inside figures; we aggregate)
            title_prefix = f"solver={solver_name}"

            # ABSOLUTE
            abs_dir = root / "absolute"
            for xp in PARAMS:
                if xp not in df_solver.columns: 
                    continue
                for metric in METRICS:
                    if metric not in df_solver.columns:
                        continue
                    self._plot_absolute(df_solver, xp, metric, abs_dir, title_prefix)

            # RELATIVE (single / double)
            for base in BASELINES:
                rel_dir = root / ("relative_single_baseline" if base.startswith("single") else "relative_double_baseline")
                for xp in PARAMS:
                    if xp not in df_solver.columns: 
                        continue
                    for metric in METRICS:
                        if metric not in df_solver.columns:
                            continue
                        self._plot_relative(df_solver, xp, metric, base, rel_dir, title_prefix)


# ----------------- RUN -----------------
if __name__ == "__main__":
    viz = LogisticVisualizerKM()
    viz.make_all()
    print(f" Figures saved under {OUTDIR}/solver=<solver>/(absolute|relative_*)/png")


