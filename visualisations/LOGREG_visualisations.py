# logistic_visualisations.py
# Run: python3 logistic_visualisations.py

import pathlib
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------- CONFIG -----------------
CSV_PATH = "../Results/results_all.csv"
OUTDIR   = "../Results/Figures"

# show only these variants (curves) and use these baselines (for the ratios)
VARIANTS  = ["hybrid(f32→f64)", "multistage-IR", "adaptive-precision"]
BASELINES = ("single(f32)", "double(f64)")

# x-axes we might sweep and y metrics we can show
PARAMS  = ["lambda", "tol", "max_iter", "max_iter_single"]
METRICS = ["logloss", "time_sec", "roc_auc", "pr_auc"]

# we create a folder per solver
SLICE_KEYS = ["dataset", "penalty", "alpha", "solver"]
# ------------------------------------------


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _canon_x(x: pd.Series, name: str) -> pd.Series:
    """Snap x to stable bins to avoid float drift; returned as float."""
    x = _safe_num(x)
    if name in {"lambda", "tol"}:
        with np.errstate(divide="ignore"):
            lx = np.log10(x.replace(0, np.nan))
        lx = np.round(lx, 12)              # kill tiny noise
        return (10.0 ** lx).fillna(0.0).astype(float)
    elif name in {"max_iter", "max_iter_single"}:
        return x.fillna(0).astype(np.int64).astype(float)
    else:
        return np.round(x, 12)


class LogisticVisualizer:
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

        # keep only the three variants + the two baselines
        keep = set(VARIANTS) | set(BASELINES)
        df = df[df["approach"].isin(keep)].copy()

        # make numeric
        for c in PARAMS:
            if c in df.columns:
                df[c] = _safe_num(df[c])

        # strict averaging across repeats: full param key
        full_key = [
            "dataset", "penalty", "alpha", "solver", "approach",
            "lambda", "tol", "max_iter", "max_iter_single"
        ]
        full_key = [k for k in full_key if k in df.columns]
        agg_cols = [c for c in METRICS + ["iters", "iters_single", "iters_double"] if c in df.columns]
        if agg_cols:
            df = df.groupby(full_key, dropna=False)[agg_cols].mean().reset_index()

        return df

    # ------- helpers: per-line grouping (others fixed) -------
    @staticmethod
    def _line_group_cols(xp: str) -> List[str]:
        """For a swept x, keep all other params fixed (using binned versions)."""
        others_bin = [f"{p}__bin" for p in PARAMS if p != xp]
        return ["dataset", "penalty", "alpha", "solver"] + others_bin

    # ------- build relative dataframe to a baseline -------
    @staticmethod
    def _relative_df(df: pd.DataFrame, xp: str, metric: str, approach: str, baseline: str) -> pd.DataFrame:
        xp_bin = f"{xp}__bin"
        if xp not in df.columns or xp_bin not in df.columns or metric not in df.columns:
            return pd.DataFrame()

        # keys to enforce SAME slice (others fixed) + SAME x
        keys = LogisticVisualizer._line_group_cols(xp) + [xp_bin]

        # average within slice (handles repeats), then join variant ↔ baseline
        var = (df[df["approach"] == approach]
               .groupby(keys, dropna=False)[metric].mean()
               .reset_index()
               .rename(columns={metric: "VAR"}))

        base = (df[df["approach"] == baseline]
                .groupby(keys, dropna=False)[metric].mean()
                .reset_index()
                .rename(columns={metric: "BASE"}))

        m = var.merge(base, on=keys, how="inner")
        if m.empty:
            return m
        m["Rel"] = m["VAR"] / m["BASE"]
        m[xp] = m[xp_bin]
        return m

    # ------- plotting (relative only) -------
    def _plot_relative(self, df_solver: pd.DataFrame, xp: str, metric: str,
                       baseline: str, approach: str, out_dir: pathlib.Path):
        rel = self._relative_df(df_solver, xp, metric, approach, baseline)
        if rel.empty:
            return

        # we draw a line per DATASET (keeping other params fixed)
        gcols = ["dataset"] + [c for c in self._line_group_cols(xp) if c != "dataset"]
        fig, ax = plt.subplots(figsize=(8, 5.2))

        drew_any = False
        for (ds, *_), g in rel.groupby(gcols, dropna=False):
            gg = g.sort_values(xp)
            if gg[xp].nunique() < 2:
                continue
            label = str(ds)
            ax.plot(gg[xp], gg["Rel"], marker="o", lw=1.8, alpha=0.85, label=label)
            drew_any = True

        if not drew_any:
            plt.close(fig)
            return

        # axis cosmetics
        if xp in {"lambda", "tol"}:
            ax.set_xscale("log")
        ax.axhline(1.0, ls="--", c="gray", lw=1.0, label=f"{baseline} (y=1)")

        # titles/labels
        solver_val = rel["solver"].dropna().iat[0] if "solver" in rel.columns and not rel["solver"].dropna().empty else "unspecified"
        ax.set_title(f"solver={solver_val} | approach={approach}\n{metric} / {baseline} vs {xp}", fontsize=12)
        ax.set_xlabel(xp)
        if metric == "time_sec":
            ax.set_ylabel(f"time speed ratio ( {approach} / {baseline} )")
        else:
            ax.set_ylabel(f"{metric} / {baseline}")
        ax.grid(True, ls="--", alpha=0.45)
        # shrink legend if many datasets
        n_legend = len(ax.lines)
        ax.legend(loc="best", fontsize=8 if n_legend > 8 else 9, frameon=True)

        # save
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "png").mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fname = f"{approach}__{metric}_vs_{xp}__rel.png"
        fig.savefig(out_dir / "png" / fname, dpi=200)
        plt.close(fig)

    # ------- main driver -------
    def make_all(self):
        # split output by solver
        solvers = self.df["solver"].dropna().unique().tolist() if "solver" in self.df.columns else ["unspecified"]
        if not solvers:
            solvers = ["unspecified"]

        for solver_name in solvers:
            if solver_name == "unspecified":
                df_solver = self.df.copy()
            else:
                df_solver = self.df[self.df["solver"] == solver_name].copy()

            root = self.outdir / f"solver={solver_name if pd.notna(solver_name) else 'unspecified'}"

            # RELATIVE (single / double); no absolute plots
            for base in BASELINES:
                base_dir = root / ("relative_single_baseline" if base.startswith("single") else "relative_double_baseline")
                for xp in PARAMS:
                    if xp not in df_solver.columns:
                        continue
                    # need at least 2 x-values overall, otherwise skip
                    if df_solver[xp].nunique(dropna=True) < 2:
                        continue
                    for metric in METRICS:
                        if metric not in df_solver.columns:
                            continue
                        for approach in VARIANTS:
                            self._plot_relative(df_solver, xp, metric, base, approach, base_dir)


# ----------------- RUN -----------------
if __name__ == "__main__":
    viz = LogisticVisualizer()
    viz.make_all()
    print(f"✅ Relative figures saved under: {OUTDIR}/solver=<solver>/relative_(single|double)_baseline/approach=.../png")


