
# Run: python3 logistic_visualisations.py
# ------------------------------------------------------------

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from typing import List, Tuple

# ---------- CONFIG ----------
CSV_PATH = "../Results/results_all.csv"
OUTDIR   = "./Figures"
FORMATS  = ["png"]

PLOT_APPROACHES = ["hybrid(f32→f64)", "multistage-IR", "adaptive-precision"]
BASELINES = ("single(f32)", "double(f64)")

PARAMS  = ["lambda", "tol", "max_iter", "max_iter_single"]
METRICS = ["logloss", "time_sec", "roc_auc", "pr_auc"]

SLICE_KEYS = ["dataset", "penalty", "alpha", "solver"]  # we split by solver into folders
# ----------------------------


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
    sf.set_scientific(False); sf.set_useOffset(False)
    ax.yaxis.set_major_formatter(sf)
    ax.grid(True, ls="--", alpha=0.35)
    ax.legend(loc="best", frameon=True, fontsize=9)


class LogisticVisualizer:
    def __init__(self, csv_path: str, outdir: str):
        self.csv_path = pathlib.Path(csv_path)
        self.outdir = pathlib.Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.df = self._load_and_avg_repeats()

    # --------- load + average repeats by FULL key ---------
    def _load_and_avg_repeats(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)

        # numeric params
        for c in ["lambda", "tol", "max_iter", "max_iter_single"]:
            if c in df.columns:
                df[c] = _safe_num(df[c])

        # keep plot approaches + baselines (we need baselines only for ratios)
        keep = set(PLOT_APPROACHES) | set(BASELINES)
        df = df[df["approach"].isin(keep)].copy()

        # strict repeat-averaging: full key
        group_keys = [
            "dataset", "penalty", "alpha", "solver", "approach",
            "lambda", "tol", "max_iter", "max_iter_single"
        ]
        group_keys = [k for k in group_keys if k in df.columns]

        agg_cols = [c for c in METRICS + ["iters", "iters_single", "iters_double"] if c in df.columns]
        if agg_cols:
            df = df.groupby(group_keys, dropna=False)[agg_cols].mean().reset_index()

        return df

    # --------- split per solver ---------
    def _per_solver(self) -> List[Tuple[str, pd.DataFrame, pathlib.Path]]:
        if "solver" not in self.df.columns:
            return [("unspecified", self.df.copy(), self.outdir / "solver=unspecified")]
        solvers = self.df["solver"].dropna().unique().tolist() or ["unspecified"]
        out = []
        for s in solvers:
            if s == "unspecified":
                sdf = self.df.copy()
                root = self.outdir / "solver=unspecified"
            else:
                sdf = self.df[self.df["solver"] == s].copy()
                root = self.outdir / f"solver={s}"
            if sdf.empty:
                continue
            root.mkdir(parents=True, exist_ok=True)
            out.append((str(s), sdf, root))
        return out

    # --------- aggregate ONE point per x (absolute) ---------
    @staticmethod
    def _abs_curve_per_x(sdf: pd.DataFrame, x_param: str, metric: str, approach: str) -> pd.DataFrame:
        sub = sdf[sdf["approach"] == approach]
        if sub.empty or x_param not in sub.columns:
            return pd.DataFrame(columns=[x_param, metric, "n"])
        # mean across ALL other knobs at each x
        g = (sub.groupby(x_param, dropna=False)[metric]
                 .agg(["mean", "std", "count"])
                 .reset_index()
                 .rename(columns={"mean": metric, "std": f"{metric}_std", "count": "n"}))
        return g.sort_values(x_param)

    # --------- aggregate ONE point per x (relative) ---------
    @staticmethod
    def _rel_curve_per_x(sdf: pd.DataFrame, x_param: str, metric: str,
                         approach: str, baseline: str) -> pd.DataFrame:
        var = LogisticVisualizer._abs_curve_per_x(sdf, x_param, metric, approach)
        bas = LogisticVisualizer._abs_curve_per_x(sdf, x_param, metric, baseline)
        if var.empty or bas.empty: 
            return pd.DataFrame(columns=[x_param, f"{metric}_rel", "n"])
        m = pd.merge(
            var[[x_param, metric, "n"]].rename(columns={metric: "var", "n": "n_var"}),
            bas[[x_param, metric, "n"]].rename(columns={metric: "base", "n": "n_base"}),
            on=x_param, how="inner"
        )
        if m.empty:
            return pd.DataFrame(columns=[x_param, f"{metric}_rel", "n"])
        m[f"{metric}_rel"] = m["var"] / m["base"]
        m["n"] = np.minimum(m["n_var"], m["n_base"])
        return m[[x_param, f"{metric}_rel", "n"]].sort_values(x_param)

    # --------- plotters ---------
    def _plot_abs(self, sdf: pd.DataFrame, x_param: str, metric: str, save_dir: pathlib.Path, title_prefix: str):
        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        y_label = f"{metric} (seconds)" if metric == "time_sec" else metric
        drew = False
        for approach in PLOT_APPROACHES:
            c = self._abs_curve_per_x(sdf, x_param, metric, approach)
            if c.empty or c[x_param].nunique(dropna=True) <= 1:
                continue
            ax.plot(c[x_param], c[metric], marker="o", lw=1.8, label=approach)
            drew = True
        if not drew:
            plt.close(fig); return
        _style_axes(ax, x_param, y_label, f"{title_prefix}\n{y_label} vs {x_param}", relative=False)
        for fmt in FORMATS:
            path = save_dir / fmt / f"{metric}_vs_{x_param}.{fmt}"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout(); fig.savefig(path, dpi=180)
        plt.close(fig)

    def _plot_rel(self, sdf: pd.DataFrame, x_param: str, metric: str, baseline: str,
                  save_dir: pathlib.Path, title_prefix: str):
        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        y_label = f"{metric} / {baseline}"
        drew = False
        for approach in PLOT_APPROACHES:
            c = self._rel_curve_per_x(sdf, x_param, metric, approach, baseline)
            if c.empty or c[x_param].nunique(dropna=True) <= 1:
                continue
            ax.plot(c[x_param], c[f"{metric}_rel"], marker="o", lw=1.8, label=approach)
            drew = True
        if not drew:
            plt.close(fig); return
        _style_axes(ax, x_param, y_label, f"{title_prefix}\n{y_label} vs {x_param}", relative=True)
        for fmt in FORMATS:
            path = save_dir / fmt / f"{metric}_vs_{x_param}__rel.{fmt}"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout(); fig.savefig(path, dpi=180)
        plt.close(fig)

    # --------- orchestrator ---------
    def make_all(self):
        for _, df_solver, root in self._per_solver():
            # dirs
            abs_dir  = root / "absolute"
            rel_sgl  = root / "relative_single_baseline"
            rel_dbl  = root / "relative_double_baseline"
            for d in [abs_dir, rel_sgl, rel_dbl]:
                for fmt in FORMATS:
                    (d / fmt).mkdir(parents=True, exist_ok=True)

            # slice by dataset/penalty/alpha (solver already split)
            slice_cols = [c for c in SLICE_KEYS if c in df_solver.columns]
            groups = df_solver.groupby([c for c in slice_cols if c != "solver"], dropna=False) if slice_cols else [({}, df_solver)]

            for keys, sdf in groups:
                if isinstance(keys, tuple):
                    sdict = {([c for c in slice_cols if c != "solver"][i]): keys[i] for i in range(len([c for c in slice_cols if c != "solver"]))}
                else:
                    sdict = {}
                title_prefix = " | ".join([f"{k}={sdict[k]}" for k in ["dataset", "penalty", "alpha"] if k in sdict]) or "All Experiments"

                for xp in [p for p in PARAMS if p in sdf.columns]:
                    if sdf[xp].nunique(dropna=True) <= 1:
                        continue
                    for metric in [m for m in METRICS if m in sdf.columns]:
                        # absolute
                        self._plot_abs(sdf, xp, metric, abs_dir, title_prefix)
                        # relative (single & double)
                        self._plot_rel(sdf, xp, metric, BASELINES[0], rel_sgl, title_prefix)
                        self._plot_rel(sdf, xp, metric, BASELINES[1], rel_dbl, title_prefix)


# ---------------- RUN ----------------
if __name__ == "__main__":
    viz = LogisticVisualizer(CSV_PATH, OUTDIR)
    viz.make_all()
    print(f"✅ Figures saved to {OUTDIR} (per solver).")
