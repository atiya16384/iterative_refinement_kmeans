# logistic_visualisations.py
# ------------------------------------------------------------
# Repeat-aware, de-jagged visualisations for AOCL-DA logistic runs.
# Curves are MEAN per (approach, x_param), independent of other knobs.
# Relative curves divide by SINGLE or DOUBLE baseline mean per x.
#
# Run: python3 logistic_visualisations.py
# ------------------------------------------------------------

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from typing import Dict, List, Tuple

# ----------------- config -----------------
CSV_PATH = "../Results/results_all.csv"
OUTDIR   = "./Figures"
FORMATS  = ["png"]

# Only plot these; singles/doubles are baselines only
PLOT_APPROACHES = ["hybrid(f32→f64)", "multistage-IR", "adaptive-precision"]
BASELINES = ("single(f32)", "double(f64)")

# Axes
PARAMS  = ["lambda", "tol", "max_iter", "max_iter_single"]
METRICS = ["logloss", "time_sec", "roc_auc", "pr_auc"]

# Slice keys (solver gets its own folder)
SLICE_KEYS = ["dataset", "penalty", "alpha", "solver"]
# -----------------------------------------


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
    sf = ScalarFormatter(useMathText=False)
    sf.set_scientific(False)
    sf.set_useOffset(False)
    ax.yaxis.set_major_formatter(sf)
    ax.grid(True, ls="--", alpha=0.35)
    ax.legend(loc="best", frameon=True, fontsize=9)


class LogisticVisualizer:
    def __init__(self, csv_path: str, outdir: str):
        self.csv_path = pathlib.Path(csv_path)
        self.outdir = pathlib.Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.df = self._load_and_average()

    # ---------------- data prep ----------------
    def _load_and_average(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)

        # Coerce param columns to numeric for reliable grouping/sorting
        for c in ["lambda", "tol", "max_iter", "max_iter_single"]:
            if c in df.columns:
                df[c] = _safe_num(df[c])

        # Keep the approaches we plot + baselines we need
        keep = set(PLOT_APPROACHES) | set(BASELINES)
        df = df[df["approach"].isin(keep)].copy()

        # --- average across repeats (if present) ---
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

    # -------------- per-solver split --------------
    def _per_solver(self) -> List[Tuple[str, pd.DataFrame, pathlib.Path]]:
        if "solver" not in self.df.columns:
            return [("unspecified", self.df.copy(), self.outdir / "solver=unspecified")]
        solvers = [s for s in self.df["solver"].dropna().unique().tolist()]
        if not solvers:
            return [("unspecified", self.df.copy(), self.outdir / "solver=unspecified")]
        out = []
        for s in solvers:
            sdf = self.df[self.df["solver"] == s].copy()
            od  = self.outdir / f"solver={s}"
            od.mkdir(parents=True, exist_ok=True)
            out.append((str(s), sdf, od))
        return out

    # -------------- curve builders (de-jagged) --------------
    @staticmethod
    def _mean_curve_by_x(df: pd.DataFrame, x_param: str, metric: str, approach: str) -> pd.DataFrame:
        """
        Return a clean curve: one row per x, y = mean(metric) over ALL rows
        with the given approach (ignoring other knobs). This kills 'towers'.
        """
        sub = df[df["approach"] == approach]
        if sub.empty or x_param not in sub.columns:
            return pd.DataFrame(columns=[x_param, metric, "n"])
        g = (sub.groupby(x_param, dropna=False)[metric]
                  .agg(["mean", "count"])
                  .reset_index()
                  .rename(columns={"mean": metric, "count": "n"}))
        return g.sort_values(x_param)

    @staticmethod
    def _relative_curve_by_x(df: pd.DataFrame, x_param: str, metric: str,
                             approach: str, baseline: str) -> pd.DataFrame:
        """
        Compute mean(metric) per x for 'approach' and mean(metric) per x for 'baseline',
        then take their ratio → exactly one point per x.
        """
        var_curve  = LogisticVisualizer._mean_curve_by_x(df, x_param, metric, approach)
        base_curve = LogisticVisualizer._mean_curve_by_x(df, x_param, metric, baseline)
        if var_curve.empty or base_curve.empty:
            return pd.DataFrame(columns=[x_param, f"{metric}_rel"])

        merged = pd.merge(var_curve[[x_param, metric]],
                          base_curve[[x_param, metric]].rename(columns={metric: f"{metric}_base"}),
                          on=x_param, how="inner")
        if merged.empty:
            return pd.DataFrame(columns=[x_param, f"{metric}_rel"])

        merged[f"{metric}_rel"] = merged[metric] / merged[f"{metric}_base"]
        return merged[[x_param, f"{metric}_rel"]].sort_values(x_param)

    # -------------- plotting --------------
    def _plot_abs(self, df: pd.DataFrame, x_param: str, metric: str, save_to: pathlib.Path, title_prefix: str):
        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        y_label = f"{metric} (seconds)" if metric == "time_sec" else metric

        drew_any = False
        for approach in PLOT_APPROACHES:
            curve = self._mean_curve_by_x(df, x_param, metric, approach)
            if curve.empty or curve[x_param].nunique(dropna=True) <= 1:
                continue
            ax.plot(curve[x_param], curve[metric], marker="o", lw=1.8, label=approach)
            drew_any = True

        if not drew_any:
            plt.close(fig); return

        _style_axes(ax, x_param, y_label, f"{title_prefix}\n{y_label} vs {x_param}", relative=False)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_to, dpi=180)
        plt.close(fig)

    def _plot_rel(self, df: pd.DataFrame, x_param: str, metric: str, baseline: str,
                  save_to: pathlib.Path, title_prefix: str):
        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        y_label = f"{metric} / {baseline}"

        drew_any = False
        for approach in PLOT_APPROACHES:
            curve = self._relative_curve_by_x(df, x_param, metric, approach, baseline)
            if curve.empty or curve[x_param].nunique(dropna=True) <= 1:
                continue
            ax.plot(curve[x_param], curve[f"{metric}_rel"], marker="o", lw=1.8, label=approach)
            drew_any = True

        if not drew_any:
            plt.close(fig); return

        _style_axes(ax, x_param, y_label, f"{title_prefix}\n{y_label} vs {x_param}", relative=True)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_to, dpi=180)
        plt.close(fig)

    # -------------- orchestrator --------------
    def make_all(self):
        for solver_name, df_s, root_out in self._per_solver():
            # per-solver dirs
            abs_dir  = root_out / "absolute"
            rel_sgl  = root_out / "relative_single_baseline"
            rel_dbl  = root_out / "relative_double_baseline"
            for d in [abs_dir, rel_sgl, rel_dbl]:
                for fmt in FORMATS:
                    (d / fmt).mkdir(parents=True, exist_ok=True)

            # per-slice (dataset, penalty, alpha), solver already split
            slice_cols = [c for c in SLICE_KEYS if c in df_s.columns]
            groups = df_s.groupby([c for c in slice_cols if c != "solver"], dropna=False)

            for keys, sdf in groups:
                if not isinstance(keys, tuple):
                    keys = (keys,)
                sdict = {col: keys[i] for i, col in enumerate([c for c in slice_cols if c != "solver"])}

                title_prefix = " | ".join([f"{k}={sdict[k]}" for k in ["dataset", "penalty", "alpha"] if k in sdict]) or "All Experiments"
                base_fname   = "__".join([f"{k}-{sdict[k]}" for k in ["dataset", "penalty", "alpha"] if k in sdict]) or "All"

                for xp in [p for p in PARAMS if p in sdf.columns]:
                    if sdf[xp].nunique(dropna=True) <= 1:
                        continue
                    for metric in [m for m in METRICS if m in sdf.columns]:
                        # absolute
                        for fmt in FORMATS:
                            self._plot_abs(
                                sdf, xp, metric,
                                save_to=abs_dir / fmt / f"{base_fname}__{metric}_vs_{xp}.{fmt}",
                                title_prefix=title_prefix
                            )
                        # relative
                        for baseline, rdir in [(BASELINES[0], rel_sgl), (BASELINES[1], rel_dbl)]:
                            for fmt in FORMATS:
                                self._plot_rel(
                                    sdf, xp, metric, baseline=baseline,
                                    save_to=rdir / fmt / f"{base_fname}__{metric}_vs_{xp}__rel.{fmt}",
                                    title_prefix=title_prefix
                                )


# ---------------- run ----------------
if __name__ == "__main__":
    viz = LogisticVisualizer(csv_path=CSV_PATH, outdir=OUTDIR)
    viz.make_all()
    print(f"✅ Figures saved to {OUTDIR} (per solver).")
