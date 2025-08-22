# logistic_visualisations.py
# ------------------------------------------------------------
# KMeans-style visualisations for AOCL-DA logistic experiments
# - No CI, no boxplots
# - Faint per-slice lines + bold median (like your _clean_line)
# - Mean across repeats
# - Only plot: hybrid(f32→f64), multistage-IR, adaptive-precision
# - Single(f32)/Double(f64) are used only as baselines (dotted y=1)
# - Canonicalise lambda/tol in log10-space to avoid float drift
#
# Run: python3 logistic_visualisations.py
# ------------------------------------------------------------

import pathlib
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
CSV_PATH = "../Results/results_all.csv"
OUTDIR   = "../Results/Figures"

# We only draw these approaches (baselines are not drawn; they’re used for ratios)
VARIANTS  = ["hybrid(f32→f64)", "multistage-IR", "adaptive-precision"]
BASELINES = ("single(f32)", "double(f64)")

# X & Y grids to produce
PARAMS  = ["lambda", "tol", "max_iter", "max_iter_single"]
METRICS = ["logloss", "time_sec", "roc_auc", "pr_auc"]

# We facet per solver into folders; within a figure we faint-plot each slice
SLICE_KEYS = ["dataset", "penalty", "alpha", "solver"]
# --------------------------------------


# ----------- small utils -----------
def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _canon_x(x: pd.Series, name: str) -> pd.Series:
    """
    Snap x-values to stable bins so one clean line per x.
    - lambda/tol: round in log10 space
    - max_iter, max_iter_single: cast to int
    - else: round to 12 decimals
    """
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


# ----------- visualiser -----------
class LogisticVisualizerKMStyle:
    def __init__(self, csv_path=CSV_PATH, outdir=OUTDIR):
        self.csv_path = pathlib.Path(csv_path)
        self.outdir   = pathlib.Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.df = self._load_avg_repeats()
        # precompute canonical bins for each x-param once
        for xp in PARAMS:
            if xp in self.df.columns:
                self.df[f"{xp}__bin"] = _canon_x(self.df[xp], xp)

    # --- load + average repeats by full key ---
    def _load_avg_repeats(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)

        # keep only approaches we need (variants + baselines)
        keep = set(VARIANTS) | set(BASELINES)
        df = df[df["approach"].isin(keep)].copy()

        # ensure numeric params
        for c in PARAMS:
            if c in df.columns:
                df[c] = _safe_num(df[c])

        # average across repeats by full hyper-parameter key
        full_key = [
            "dataset", "penalty", "alpha", "solver", "approach",
            "lambda", "tol", "max_iter", "max_iter_single"
        ]
        full_key = [k for k in full_key if k in df.columns]
        metric_cols = [c for c in METRICS + ["iters", "iters_single", "iters_double"] if c in df.columns]

        if metric_cols:
            df = (
                df.groupby(full_key, dropna=False)[metric_cols]
                  .mean()
                  .reset_index()
            )
        return df

    # --- relative aggregator (like your _rel) ---
    @staticmethod
    def _rel(df: pd.DataFrame, keys: List[str], metric: str, baseline_label: str) -> pd.DataFrame:
        """
        Build a dataframe keyed by `keys` with:
           [*keys, metric, BASE, Rel]
        where Rel = variant metric / baseline metric.
        We only keep rows where approach in VARIANTS.
        """
        base = (
            df[df["approach"] == baseline_label]
            .groupby(keys, as_index=False)[metric].mean()
            .rename(columns={metric: "BASE"})
        )
        var = (
            df[df["approach"].isin(VARIANTS)]
            .groupby(keys + ["approach"], as_index=False)[metric].mean()
        )
        out = var.merge(base, on=keys, how="inner")
        if out.empty:
            return out
        out["Rel"] = out[metric] / out["BASE"]
        return out

    # --- plotting primitive (exactly like your _clean_line) ---
    @staticmethod
    def _clean_line(
        rel_df: pd.DataFrame,
        xcol: str,
        title: str,
        ylabel: str,
        outpath: pathlib.Path,
        logx: bool = False,
    ) -> None:
        """
        Plot faint per-slice curves and a bold median curve.
        x is sorted; optional log-x.
        """
        fig, ax = plt.subplots(figsize=(7, 5))

        # faint per-(dataset, penalty, alpha) *per approach*
        for (ds, pen, a, app), g in rel_df.groupby(["dataset", "penalty", "alpha", "approach"]):
            g = g.sort_values(xcol)
            ax.plot(g[xcol], g["Rel"], marker="o", alpha=0.30)

        # bold median (across all slices + approaches) per x
        agg = (
            rel_df.groupby([xcol])["Rel"]
                  .median()
                  .reset_index()
                  .sort_values(xcol)
        )
        ax.plot(agg[xcol], agg["Rel"], marker="o", lw=2.2, label="Median")

        if logx:
            ax.set_xscale("log")

        ax.axhline(1.0, ls="--", c="gray", lw=1.0, label="Baseline (y=1)")
        ax.set_title(title)
        ax.set_xlabel(xcol)
        ax.set_ylabel(ylabel)
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=200)
        plt.close(fig)

    # --- one generator for relative plots vs a chosen baseline ---
    def _make_relative_suite_for_baseline(self, baseline_label: str):
        sub_folder = "relative_single_baseline" if baseline_label.startswith("single") else "relative_double_baseline"

        # split per solver (folder)
        if "solver" in self.df.columns:
            solver_groups = self.df.groupby("solver", dropna=False)
        else:
            solver_groups = [( "unspecified", self.df )]

        for solver_name, df_solver in solver_groups:
            root = self.outdir / f"solver={solver_name if pd.notna(solver_name) else 'unspecified'}" / sub_folder
            root.mkdir(parents=True, exist_ok=True)

            # within each solver, we’ll build per-x curves using canonical bins
            for xp in [p for p in PARAMS if p in df_solver.columns]:
                xp_bin = f"{xp}__bin"
                if xp_bin not in df_solver.columns:  # safety
                    continue

                for metric in [m for m in METRICS if m in df_solver.columns]:
                    # Build relative dataframe keyed by slice + x
                    # keys = dataset, penalty, alpha, xp_bin
                    keys = [k for k in ["dataset", "penalty", "alpha", xp_bin] if k in df_solver.columns]
                    if not keys:
                        continue

                    # Prepare a tiny df with just what we need:
                    use_cols = list(set(keys + ["approach", metric, xp]))
                    sdf = df_solver[use_cols].copy()
                    # ensure the x plotting column equals the canonical bin:
                    sdf[xp] = sdf[xp_bin] if xp_bin in sdf.columns else sdf[xp]

                    rel_df = self._rel(sdf, keys, metric, baseline_label)
                    if rel_df.empty:
                        continue

                    # use the real xcol = xp_bin replaced as xp
                    rel_df = rel_df.rename(columns={xp_bin: xp})

                    # title & y-label
                    ylab = f"{metric} / {baseline_label}"
                    ttl  = f"{metric} vs {xp}  | baseline={baseline_label}  | solver={solver_name}"

                    # decide log x for lambda/tol
                    logx = xp in {"lambda", "tol"}

                    # path
                    fname = f"{metric}_vs_{xp}__{sub_folder}.png"
                    outpath = root / fname

                    self._clean_line(rel_df, xp, ttl, ylab, outpath, logx=logx)

    # --- public API ---
    def make_all(self):
        # produce relative plots vs both baselines, for every (xp, metric)
        for base in BASELINES:
            self._make_relative_suite_for_baseline(base)


# -------------- RUN --------------
if __name__ == "__main__":
    viz = LogisticVisualizerKMStyle()
    viz.make_all()
    print(f"Figures saved under {OUTDIR}/solver=<solver>/relative_*/*.png")

