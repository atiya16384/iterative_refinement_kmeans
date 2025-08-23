
# Run: python3 logistic_visualisations.py
# ------------------------------------------------------------

import pathlib
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
CSV_PATH = "../Results/results_all.csv"
OUTDIR   = "../Results/Figures"

VARIANTS  = ["hybrid(f32→f64)", "multistage-IR", "adaptive-precision"]
BASELINES = ("single(f32)", "double(f64)")

PARAMS  = ["lambda", "tol", "max_iter", "max_iter_single"]
METRICS = ["logloss", "time_sec", "roc_auc", "pr_auc"]
# ----------------------------


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _canon_x(x: pd.Series, name: str) -> pd.Series:
    """Snap x to stable bins so tiny float noise doesn't break grouping."""
    x = _safe_num(x)
    if name in {"lambda", "tol"}:
        with np.errstate(divide="ignore"):
            lx = np.log10(x.replace(0, np.nan))
        lx = np.round(lx, 12)
        return (10.0**lx).fillna(0.0).astype(float)
    if name in {"max_iter", "max_iter_single"}:
        return x.fillna(0).astype(np.int64).astype(float)
    return np.round(x, 12)


class Visualiser:
    def __init__(self, csv_path=CSV_PATH, outdir=OUTDIR):
        self.outdir = pathlib.Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.df = self._load_avg(csv_path)

        # precompute “binned” (canonical) columns for all PARAMS
        for p in PARAMS:
            if p in self.df.columns:
                self.df[f"{p}__bin"] = _canon_x(self.df[p], p)

    @staticmethod
    def _load_avg(csv_path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)

        keep = set(VARIANTS) | set(BASELINES)
        df = df[df["approach"].isin(keep)].copy()

        for p in PARAMS:
            if p in df.columns:
                df[p] = _safe_num(df[p])

        # strict mean across repeats for the exact same param key
        key = [
            "dataset", "penalty", "alpha", "solver", "approach",
            "lambda", "tol", "max_iter", "max_iter_single"
        ]
        key = [k for k in key if k in df.columns]
        vals = [c for c in METRICS + ["iters", "iters_single", "iters_double"] if c in df.columns]
        if vals:
            df = df.groupby(key, dropna=False)[vals].mean().reset_index()
        return df

    @staticmethod
    def _slice_keys_for(xp: str) -> List[str]:
        """To draw lines, everything except xp must be fixed; use *_bin columns."""
        others = [f"{p}__bin" for p in PARAMS if p != xp]
        return ["dataset", "penalty", "alpha", "solver"] + others + [f"{xp}__bin"]

    @staticmethod
    def _relative(df: pd.DataFrame, xp: str, metric: str, approach: str, baseline: str) -> pd.DataFrame:
        """Return per-slice relative values:
           - For time_sec: SPEEDUP = BASE/VAR  (↑ better)
           - For other metrics: RATIO = VAR/BASE
        """
        xp_bin = f"{xp}__bin"
        if any(c not in df.columns for c in [xp, xp_bin, metric, "approach"]):
            return pd.DataFrame()

        keys = Visualiser._slice_keys_for(xp)

        var = (df[df["approach"] == approach]
               .groupby(keys, dropna=False)[metric].mean()
               .reset_index()
               .rename(columns={metric: "VAR"}))
        base = (df[df["approach"] == baseline]
                .groupby(keys[:-1] + [xp_bin], dropna=False)[metric].mean()
                .reset_index()
                .rename(columns={metric: "BASE"}))

        m = var.merge(base, on=keys, how="inner")
        if m.empty:
            return m

        if metric == "time_sec":
            m["Y"] = m["BASE"] / m["VAR"]       # speedup  ↑ better
            m["ylabel"] = f"speedup vs {baseline} (↑ better)"
            m["hline"] = 1.0
        else:
            m["Y"] = m["VAR"] / m["BASE"]       # ratio    ~1 good
            m["ylabel"] = f"{metric} / {baseline}"
            m["hline"] = 1.0

        m[xp] = m[xp_bin]
        return m

    def _plot_rel(self, rel: pd.DataFrame, xp: str, approach: str, metric: str, baseline: str, outdir: pathlib.Path):
        if rel.empty or rel[xp].nunique(dropna=True) < 2:
            return

        fig, ax = plt.subplots(figsize=(8.4, 5.2))

        # one line per DATASET (with other params fixed)
        group_cols = ["dataset"] + [c for c in self._slice_keys_for(xp) if c not in {"dataset", f"{xp}__bin"}]
        drew = False
        for (ds, *_), g in rel.groupby(group_cols, dropna=False):
            gg = g.sort_values(xp)
            if gg[xp].nunique() < 2:
                continue
            ax.plot(gg[xp], gg["Y"], marker="o", lw=1.8, alpha=0.9, label=str(ds))
            drew = True
        if not drew:
            plt.close(fig)
            return

        # scales, reference, labels
        if xp in {"lambda", "tol"}:
            ax.set_xscale("log")
        ax.axhline(rel["hline"].iat[0], ls="--", c="gray", lw=1.0, label=f"{baseline} (y=1)")

        # Solver in title (all rows share it inside this plot)
        solver = rel["solver"].dropna().iat[0] if "solver" in rel.columns and not rel["solver"].dropna().empty \
                 else "unspecified"
        ax.set_title(f"solver={solver} | approach={approach}\n{('time_sec' if metric=='time_sec' else metric)} vs {xp} (relative to {baseline})", fontsize=12)
        ax.set_xlabel(xp)
        ax.set_ylabel(rel["ylabel"].iat[0])

        # keep y-range sensible for readability (optional)
        if metric == "time_sec":
            ax.set_ylim(bottom=max(0.7, ax.get_ylim()[0]))  # don't let it squash below 0.7
        ax.grid(True, ls="--", alpha=0.45)

        # tidy legend
        n = len(ax.lines)
        ax.legend(loc="best", fontsize=8 if n > 10 else 9, frameon=True)

        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "png").mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(outdir / "png" / f"{approach}__{metric}_vs_{xp}__rel.png", dpi=200)
        plt.close(fig)

    def make_all(self):
        # split by solver for folder layout
        solvers = self.df["solver"].dropna().unique().tolist() if "solver" in self.df.columns else ["unspecified"]
        if not solvers:
            solvers = ["unspecified"]

        for solver_name in solvers:
            df_s = self.df[self.df["solver"] == solver_name] if solver_name != "unspecified" else self.df.copy()
            solver_root = self.outdir / f"solver={solver_name}"

            for baseline in BASELINES:
                base_root = solver_root / ( "relative_single_baseline" if baseline.startswith("single") else "relative_double_baseline" )
                for approach in VARIANTS:
                    app_root = base_root / f"approach={approach.replace('→','to')}"
                    for xp in PARAMS:
                        if xp not in df_s.columns or df_s[xp].nunique(dropna=True) < 2:
                            continue
                        for metric in METRICS:
                            if metric not in df_s.columns:
                                continue
                            rel = self._relative(df_s, xp, metric, approach, baseline)
                            self._plot_rel(rel, xp, approach, metric, baseline, app_root)


if __name__ == "__main__":
    Visualiser().make_all()
    print(f"Relative plots written to {OUTDIR}/solver=<solver>/relative_(single|double)_baseline/approach=.../png/")

