# logistic_visualisations.py
# ------------------------------------------------------------
# Averages over repeats, then plots absolute and relative curves.
# Relative curves divide by SINGLE or DOUBLE baseline per-slice+X.
# Only plots: hybrid(f32→f64), multistage-IR, adaptive-precision.
# Baselines appear only as the dotted horizontal line at y=1.
#
# Run: python3 logistic_visualisations.py
# ------------------------------------------------------------

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from typing import List, Dict, Tuple

# ----------------- config you can tweak -----------------
CSV_PATH = "../Results/results_all.csv"
OUTDIR   = "../Results/Figures"
FORMATS  = ["png"]
SHOW     = False

PLOT_APPROACHES = ["hybrid(f32→f64)", "multistage-IR", "adaptive-precision"]
BASELINES = ("single(f32)", "double(f64)")

PARAMS  = ["lambda", "tol", "max_iter", "max_iter_single"]
METRICS = ["logloss", "time_sec", "roc_auc", "pr_auc"]

SLICE_KEYS = ["dataset", "penalty", "alpha", "solver"]   # solver is split into folders
# --------------------------------------------------------


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _most_common_combo(df: pd.DataFrame, cols: List[str]) -> Dict[str, object]:
    """Pick a single configuration for 'other params' to avoid mixing."""
    if not cols:
        return {}
    vc = (
        df[cols].astype(object)
        .value_counts(dropna=False)
        .reset_index(name="cnt")
        .sort_values("cnt", ascending=False)
    )
    if vc.empty:
        return {}
    return {c: vc.iloc[0][c] for c in cols}


def _filter_to_combo(df: pd.DataFrame, combo: Dict[str, object]) -> pd.DataFrame:
    if not combo:
        return df
    out = df
    for c, v in combo.items():
        out = out[out[c] == v]
    return out


class LogisticVisualizer:
    def __init__(self, csv_path: str, outdir: str):
        self.csv_path = pathlib.Path(csv_path)
        self.outdir = pathlib.Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.df = self._load_and_average(self.csv_path)

    # ---------------- data prep ----------------
    def _load_and_average(self, csv_path: pathlib.Path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)

        # Coerce param columns to numeric
        for c in ["lambda", "tol", "max_iter", "max_iter_single"]:
            if c in df.columns:
                df[c] = _safe_num(df[c])

        # Keep the approaches we care about + baselines (needed to compute relatives)
        keep = set(PLOT_APPROACHES) | set(BASELINES)
        if "approach" in df.columns:
            df = df[df["approach"].isin(keep)].copy()

        # --- average over repeats (mean; keep std as *_std for error bars) ---
        group_keys = [
            "dataset", "penalty", "alpha", "solver", "approach",
            "lambda", "tol", "max_iter", "max_iter_single"
        ]
        group_keys = [k for k in group_keys if k in df.columns]

        agg_cols = [c for c in METRICS + ["iters", "iters_single", "iters_double"] if c in df.columns]
        agg = {c: ["mean", "std"] for c in agg_cols}

        g = df.groupby(group_keys, dropna=False).agg(agg)
        g.columns = ["__".join(c).strip() for c in g.columns]
        g = g.reset_index()

        # flatten: keep *mean as the main value, retain *std
        for c in agg_cols:
            mcol, scol = f"{c}__mean", f"{c}__std"
            if mcol in g.columns:
                g[c] = g[mcol]
                g.drop(columns=[mcol], inplace=True)
            if scol not in g.columns:
                g[scol] = np.nan

        return g

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
            od = self.outdir / f"solver={s}"
            od.mkdir(parents=True, exist_ok=True)
            out.append((str(s), sdf, od))
        return out

    # -------------- baseline math --------------
    def _relative_to_baseline(self, df: pd.DataFrame, baseline: str, x_param: str) -> pd.DataFrame:
        """
        Divide each metric by the baseline measured on the SAME (dataset, penalty, alpha, solver)
        and SAME x_param value. We lock other params to a single combo (most common) before plotting,
        so here we only join on (slice + x).
        """
        slice_cols = [c for c in SLICE_KEYS if c in df.columns]
        join_cols  = slice_cols + [x_param]

        metrics_here = [m for m in METRICS if m in df.columns]

        bdf = df[df["approach"] == baseline].copy()
        if bdf.empty:
            return pd.DataFrame(columns=df.columns)

        bdf = (
            bdf[join_cols + metrics_here]
            .groupby(join_cols, dropna=False, as_index=False)
            .mean()
            .rename(columns={m: f"{m}_base" for m in metrics_here})
        )

        ldf = df[join_cols + ["approach"] + metrics_here].copy()
        merged = pd.merge(ldf, bdf, on=join_cols, how="inner")

        for m in metrics_here:
            merged[f"{m}_rel"] = merged[m] / merged[f"{m}_base"]
        return merged

    # -------------- plotting helpers --------------
    def _style_axes(self, ax, x_param: str, y_label: str, title: str, relative: bool):
        ax.set_xlabel(x_param)
        ax.set_ylabel(y_label)
        ax.set_title(title, fontsize=11)

        if x_param in {"lambda", "tol"}:
            ax.set_xscale("log")

        if relative:
            ax.axhline(1.0, ls=":", c="gray", lw=1.4, label="baseline")

        # Force plain decimal ticks (avoid 10^0 notation)
        sf = ScalarFormatter(useMathText=False)
        sf.set_scientific(False)
        sf.set_useOffset(False)
        ax.yaxis.set_major_formatter(sf)

        ax.grid(True, ls="--", alpha=0.4)
        ax.legend(loc="best", frameon=True, fontsize=9)

    def _aggregate_per_x(self, df: pd.DataFrame, x_param: str, y_col: str) -> pd.DataFrame:
        """One row per (approach, x)."""
        g = (
            df.groupby(["approach", x_param], dropna=False)[y_col]
              .agg(["mean", "std", "count"])
              .reset_index()
              .rename(columns={"mean": y_col, "std": f"{y_col}_std", "count": "n"})
        )
        return g

    # -------------- plotters --------------
    def _plot_abs(self, sdf: pd.DataFrame, x_param: str, metric: str, save_to: pathlib.Path, title_prefix: str):
        plot_df = sdf[sdf["approach"].isin(PLOT_APPROACHES)].copy()
        if plot_df.empty or plot_df[x_param].nunique(dropna=True) <= 1:
            return

        # lock other params so we only vary x_param
        other_params = [p for p in PARAMS if p in plot_df.columns and p != x_param]
        combo = _most_common_combo(plot_df, other_params)
        plot_df = _filter_to_combo(plot_df, combo)

        y_label = f"{metric} (seconds)" if metric == "time_sec" else metric

        agg = self._aggregate_per_x(plot_df, x_param, metric)

        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        for approach, g in agg.groupby("approach"):
            g = g.sort_values(x_param)
            ax.errorbar(g[x_param], g[metric], yerr=g.get(f"{metric}_std", None),
                        marker="o", lw=1.7, capsize=3, label=approach)

        self._style_axes(ax, x_param, y_label, f"{title_prefix}\n{y_label} vs {x_param}", relative=False)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_to, dpi=180)
        plt.close(fig)

    def _plot_rel(self, sdf: pd.DataFrame, x_param: str, metric: str, baseline: str,
                  save_to: pathlib.Path, title_prefix: str):
        # lock other params first (most common combo)
        other_params = [p for p in PARAMS if p in sdf.columns and p != x_param]
        combo = _most_common_combo(sdf, other_params)
        sdf = _filter_to_combo(sdf, combo)

        rel = self._relative_to_baseline(sdf, baseline=baseline, x_param=x_param)
        if rel.empty:
            return

        rel = rel[rel["approach"].isin(PLOT_APPROACHES)].copy()
        y_col = f"{metric}_rel"
        agg = self._aggregate_per_x(rel, x_param, y_col)
        if agg.empty or agg[x_param].nunique(dropna=True) <= 1:
            return

        y_label = f"{metric} / {baseline}"

        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        for approach, g in agg.groupby("approach"):
            g = g.sort_values(x_param)
            ax.errorbar(g[x_param], g[y_col], yerr=g.get(f"{y_col}_std", None),
                        marker="o", lw=1.7, capsize=3, label=approach)

        self._style_axes(ax, x_param, y_label, f"{title_prefix}\n{y_label} vs {x_param}", relative=True)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_to, dpi=180)
        plt.close(fig)

    # -------------- orchestrator --------------
    def make_all(self):
        for solver_name, df_s, root_out in self._per_solver():
            abs_dir  = root_out / "absolute"
            rel_sgl  = root_out / "relative_single_baseline"
            rel_dbl  = root_out / "relative_double_baseline"
            for d in [abs_dir, rel_sgl, rel_dbl]:
                for fmt in FORMATS:
                    (d / fmt).mkdir(parents=True, exist_ok=True)

            # group by dataset/penalty/alpha (solver already separated)
            slice_cols = [c for c in SLICE_KEYS if c in df_s.columns]
            if slice_cols:
                groups = df_s.groupby(slice_cols, dropna=False)
            else:
                groups = [({}, df_s)]

            for keys, sdf in groups:
                if isinstance(keys, tuple):
                    sdict = {slice_cols[i]: keys[i] for i in range(len(slice_cols))}
                elif isinstance(keys, dict):
                    sdict = keys
                else:
                    sdict = {}

                title_prefix = " | ".join([f"{k}={sdict[k]}" for k in ["dataset", "penalty", "alpha"] if k in sdict])
                base_fname   = "__".join([f"{k}-{sdict[k]}" for k in ["dataset", "penalty", "alpha"] if k in sdict]) or "All"

                for xp in [p for p in PARAMS if p in sdf.columns]:
                    if sdf[xp].nunique(dropna=True) <= 1:
                        continue
                    for metric in METRICS:
                        if metric not in sdf.columns:
                            continue

                        # absolute
                        for fmt in FORMATS:
                            self._plot_abs(
                                sdf, xp, metric,
                                save_to=abs_dir / fmt / f"{base_fname}__{metric}_vs_{xp}.{fmt}",
                                title_prefix=title_prefix or "All Experiments"
                            )

                        # relative (single, double)
                        for baseline, rdir in [(BASELINES[0], rel_sgl), (BASELINES[1], rel_dbl)]:
                            for fmt in FORMATS:
                                self._plot_rel(
                                    sdf, xp, metric, baseline=baseline,
                                    save_to=rdir / fmt / f"{base_fname}__{metric}_vs_{xp}__rel.{fmt}",
                                    title_prefix=title_prefix or "All Experiments"
                                )


# ---------------- run ----------------
if __name__ == "__main__":
    viz = LogisticVisualizer(csv_path=CSV_PATH, outdir=OUTDIR)
    viz.make_all()
    print(f"✅ Figures saved to {OUTDIR} (per solver subfolders).")

