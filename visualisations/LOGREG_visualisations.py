# logistic_visualisations.py
# ------------------------------------------------------------
# Strict, repeat-aware visualisations for AOCL-DA logistic runs.
# - Averages across repeats by the FULL hyper-parameter key.
# - For a chosen x (lambda/tol/max_iter/max_iter_single), holds ALL
#   other hyper-params fixed and draws a curve per configuration.
# - Relative plots pair rows 1:1 with the SAME config (exact match)
#   and divide by SINGLE or DOUBLE baseline.
# - Only plots approaches_of_interest; baselines appear only as y=1 line.
#
# Run: python3 logistic_visualisations.py
# Output: ./Figures/solver=<solver>/(absolute|relative_*)/png/*.png
# ------------------------------------------------------------

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from typing import Dict, List, Tuple

# ------------- CONFIG -------------
CSV_PATH = "../Results/results_all.csv"
OUTDIR   = "./Figures"
FORMATS  = ["png"]

# Only draw these approaches (use baselines only for normalization)
APPROACHES_OF_INTEREST = ["hybrid(f32→f64)", "multistage-IR", "adaptive-precision"]
BASELINES = ("single(f32)", "double(f64)")

PARAMS  = ["lambda", "tol", "max_iter", "max_iter_single"]
METRICS = ["logloss", "time_sec", "roc_auc", "pr_auc"]

SLICE_KEYS = ["dataset", "penalty", "alpha", "solver"]  # solver gets its own folder
# -----------------------------------


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _style_axes(ax, x_param: str, y_label: str, title: str, relative: bool):
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=11)
    if x_param in {"lambda", "tol"}:
        ax.set_xscale("log")
    if relative:
        ax.axhline(1.0, ls=":", c="gray", lw=1.4, label="baseline")
    sf = ScalarFormatter(useMathText=False); sf.set_scientific(False); sf.set_useOffset(False)
    ax.yaxis.set_major_formatter(sf)
    ax.grid(True, ls="--", alpha=0.35)
    ax.legend(loc="best", frameon=True, fontsize=9)


class LogisticVisualizer:
    def __init__(self, csv_path: str, outdir: str):
        self.csv_path = pathlib.Path(csv_path)
        self.outdir = pathlib.Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.df = self._load_and_average()

    # ---------- data prep (average repeats by full key) ----------
    def _load_and_average(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)

        for c in ["lambda", "tol", "max_iter", "max_iter_single"]:
            if c in df.columns: df[c] = _safe_num(df[c])

        # Keep approaches we draw + baselines we need for ratios
        keep = set(APPROACHES_OF_INTEREST) | set(BASELINES)
        if "approach" in df.columns:
            df = df[df["approach"].isin(keep)].copy()

        # Average across repeats STRICTLY by full key
        group_keys = [
            "dataset", "penalty", "alpha", "solver", "approach",
            "lambda", "tol", "max_iter", "max_iter_single"
        ]
        group_keys = [k for k in group_keys if k in df.columns]

        agg_cols = [c for c in METRICS + ["iters", "iters_single", "iters_double"] if c in df.columns]
        if agg_cols:
            g = df.groupby(group_keys, dropna=False)[agg_cols].mean().reset_index()
            return g
        return df

    # ---------- per-solver split ----------
    def _per_solver(self) -> List[Tuple[str, pd.DataFrame, pathlib.Path]]:
        if "solver" not in self.df.columns:
            return [("unspecified", self.df.copy(), self.outdir / "solver=unspecified")]
        solvers = [s for s in self.df["solver"].dropna().unique().tolist()] or [None]
        out = []
        for s in solvers:
            sdf = self.df.copy() if s is None else self.df[self.df["solver"] == s].copy()
            if sdf.empty: continue
            od = self.outdir / (f"solver={s}" if s is not None else "solver=unspecified")
            od.mkdir(parents=True, exist_ok=True)
            out.append((str(s) if s is not None else "unspecified", sdf, od))
        return out

    # ---------- strict pairing helpers ----------
    @staticmethod
    def _group_cols_for_x(x_param: str) -> List[str]:
        """Return the columns that define a fixed-config curve when sweeping x_param."""
        others = [p for p in PARAMS if p != x_param]
        # solver split is outside; keep dataset/penalty/alpha here
        return [c for c in ["dataset", "penalty", "alpha"] if c] + others

    @staticmethod
    def _curve_abs(df: pd.DataFrame, x_param: str, metric: str, approach: str) -> pd.DataFrame:
        """Within a fixed-config group (handled outside), one point per x_param."""
        sub = df[df["approach"] == approach]
        if sub.empty or x_param not in sub.columns:
            return pd.DataFrame(columns=[x_param, metric])
        # already averaged over repeats; no further aggregation needed
        cur = sub[[x_param, metric]].dropna(subset=[x_param]).copy()
        # multiple rows at same x can exist (rare); average them
        cur = cur.groupby(x_param, dropna=False)[metric].mean().reset_index()
        return cur.sort_values(x_param)

    @staticmethod
    def _curve_rel(df: pd.DataFrame, x_param: str, metric: str, approach: str, baseline: str) -> pd.DataFrame:
        """
        Pair rows 1:1 by EXACT config (group is fixed outside) and SAME x value.
        ratio = metric(approach) / metric(baseline)
        """
        var = df[df["approach"] == approach][[x_param, metric]].rename(columns={metric: "var"}).copy()
        base = df[df["approach"] == baseline][[x_param, metric]].rename(columns={metric: "base"}).copy()
        if var.empty or base.empty: return pd.DataFrame(columns=[x_param, f"{metric}_rel"])
        m = pd.merge(var, base, on=x_param, how="inner")
        if m.empty: return pd.DataFrame(columns=[x_param, f"{metric}_rel"])
        m[f"{metric}_rel"] = m["var"] / m["base"]
        out = m[[x_param, f"{metric}_rel"]].groupby(x_param, dropna=False).mean().reset_index()
        return out.sort_values(x_param)

    # ---------- plotting per fixed-config group ----------
    def _plot_group_abs(self, gdf: pd.DataFrame, x_param: str, metric: str,
                        save_dir: pathlib.Path, title_prefix: str, fname_prefix: str):
        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        y_label = f"{metric} (seconds)" if metric == "time_sec" else metric

        drew = False
        for approach in APPROACHES_OF_INTEREST:
            c = self._curve_abs(gdf, x_param, metric, approach)
            if c.empty or c[x_param].nunique(dropna=True) <= 1: continue
            ax.plot(c[x_param], c[metric], marker="o", lw=1.8, label=approach)
            drew = True
        if not drew:
            plt.close(fig); return

        # build a short suffix describing the fixed config
        fixed_cols = [p for p in PARAMS if p != x_param and p in gdf.columns]
        fixed_vals = ", ".join([f"{c}={gdf.iloc[0][c]}" for c in fixed_cols])
        full_title = f"{title_prefix}" + (f"\n(fixed: {fixed_vals})" if fixed_vals else "")
        _style_axes(ax, x_param, y_label, f"{full_title}\n{y_label} vs {x_param}", relative=False)

        for fmt in FORMATS:
            out = save_dir / fmt / f"{fname_prefix}__{metric}_vs_{x_param}__{fixed_vals.replace(', ','__').replace('=','-')}.{fmt}"
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout(); fig.savefig(out, dpi=180)
        plt.close(fig)

    def _plot_group_rel(self, gdf: pd.DataFrame, x_param: str, metric: str, baseline: str,
                        save_dir: pathlib.Path, title_prefix: str, fname_prefix: str):
        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        y_label = f"{metric} / {baseline}"

        drew = False
        for approach in APPROACHES_OF_INTEREST:
            c = self._curve_rel(gdf, x_param, metric, approach, baseline)
            if c.empty or c[x_param].nunique(dropna=True) <= 1: continue
            ax.plot(c[x_param], c[f"{metric}_rel"], marker="o", lw=1.8, label=approach)
            drew = True
        if not drew:
            plt.close(fig); return

        fixed_cols = [p for p in PARAMS if p != x_param and p in gdf.columns]
        fixed_vals = ", ".join([f"{c}={gdf.iloc[0][c]}" for c in fixed_cols])
        full_title = f"{title_prefix}" + (f"\n(fixed: {fixed_vals})" if fixed_vals else "")
        _style_axes(ax, x_param, y_label, f"{full_title}\n{y_label} vs {x_param}", relative=True)

        for fmt in FORMATS:
            out = save_dir / fmt / f"{fname_prefix}__{metric}_vs_{x_param}__rel__{fixed_vals.replace(', ','__').replace('=','-')}.{fmt}"
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout(); fig.savefig(out, dpi=180)
        plt.close(fig)

    # ---------- orchestrator ----------
    def make_all(self):
        for _, df_solver, root in self._per_solver():
            abs_dir  = root / "absolute"
            rel_sgl  = root / "relative_single_baseline"
            rel_dbl  = root / "relative_double_baseline"
            for d in [abs_dir, rel_sgl, rel_dbl]:
                for fmt in FORMATS: (d / fmt).mkdir(parents=True, exist_ok=True)

            # slice by dataset/penalty/alpha (solver already isolated)
            slice_cols = [c for c in ["dataset", "penalty", "alpha"] if c in df_solver.columns]
            slice_groups = df_solver.groupby(slice_cols, dropna=False) if slice_cols else [({}, df_solver)]

            for s_keys, sdf in slice_groups:
                if isinstance(s_keys, tuple):
                    sdict = {slice_cols[i]: s_keys[i] for i in range(len(slice_cols))}
                else:
                    sdict = {}
                title_prefix = " | ".join([f"{k}={sdict[k]}" for k in ["dataset", "penalty", "alpha"] if k in sdict]) or "All Experiments"
                fname_prefix = "__".join([f"{k}-{sdict[k]}" for k in ["dataset", "penalty", "alpha"] if k in sdict]) or "All"

                # for each x, group by ALL other params (strict fixed-config curves)
                for xp in [p for p in PARAMS if p in sdf.columns]:
                    other_cols = [p for p in PARAMS if p != xp and p in sdf.columns]
                    if xp not in sdf.columns: continue
                    if sdf[xp].nunique(dropna=True) <= 1: continue

                    cfg_groups = sdf.groupby(other_cols, dropna=False) if other_cols else [({}, sdf)]
                    for _, gdf in cfg_groups:
                        # need at least two distinct x values AND at least one approach of interest present
                        if gdf[xp].nunique(dropna=True) <= 1: continue
                        if not any(gdf["approach"].eq(a).any() for a in APPROACHES_OF_INTEREST): continue

                        # absolute plots
                        self._plot_group_abs(
                            gdf, x_param=xp, metric="logloss",
                            save_dir=abs_dir, title_prefix=title_prefix, fname_prefix=fname_prefix
                        )
                        self._plot_group_abs(
                            gdf, x_param=xp, metric="time_sec",
                            save_dir=abs_dir, title_prefix=title_prefix, fname_prefix=fname_prefix
                        )
                        self._plot_group_abs(
                            gdf, x_param=xp, metric="roc_auc",
                            save_dir=abs_dir, title_prefix=title_prefix, fname_prefix=fname_prefix
                        )
                        self._plot_group_abs(
                            gdf, x_param=xp, metric="pr_auc",
                            save_dir=abs_dir, title_prefix=title_prefix, fname_prefix=fname_prefix
                        )

                        # relative plots (single & double) — only if baseline rows exist
                        for baseline, rdir in [(BASELINES[0], rel_sgl), (BASELINES[1], rel_dbl)]:
                            if not gdf["approach"].eq(baseline).any(): 
                                continue
                            self._plot_group_rel(
                                gdf, x_param=xp, metric="logloss", baseline=baseline,
                                save_dir=rdir, title_prefix=title_prefix, fname_prefix=fname_prefix
                            )
                            self._plot_group_rel(
                                gdf, x_param=xp, metric="time_sec", baseline=baseline,
                                save_dir=rdir, title_prefix=title_prefix, fname_prefix=fname_prefix
                            )
                            self._plot_group_rel(
                                gdf, x_param=xp, metric="roc_auc", baseline=baseline,
                                save_dir=rdir, title_prefix=title_prefix, fname_prefix=fname_prefix
                            )
                            self._plot_group_rel(
                                gdf, x_param=xp, metric="pr_auc", baseline=baseline,
                                save_dir=rdir, title_prefix=title_prefix, fname_prefix=fname_prefix
                            )


# ---------------- RUN ----------------
if __name__ == "__main__":
    viz = LogisticVisualizer(csv_path=CSV_PATH, outdir=OUTDIR)
    viz.make_all()
    print(f" Figures saved under {OUTDIR} (per solver).")
