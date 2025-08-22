# logistic_visualisations.py
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple

sns.set_style("whitegrid")

PLOT_APPROACHES = ["hybrid(f32→f64)", "multistage-IR", "adaptive-precision"]
BASELINES = ("single(f32)", "double(f64)")
PARAMS = ["lambda", "tol", "max_iter", "max_iter_single"]
METRICS = ["logloss", "time_sec", "roc_auc", "pr_auc"]


def _isnum(x) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def _safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


class LogisticVisualizer:
    def __init__(self, csv_path: str = "../Results/results_all.csv",
                 outdir: str = "./Figures"):
        self.csv_path = pathlib.Path(csv_path)
        self.outdir = pathlib.Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.df = self._load_and_aggregate(self.csv_path)

    # ---------------- basics ----------------
    def _load_and_aggregate(self, csv_path: pathlib.Path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)

        # ensure types
        for c in ["lambda", "tol", "max_iter", "max_iter_single"]:
            if c in df.columns:
                df[c] = _safe_num(df[c])

        # limit to approaches we care about + baselines (we need baselines to compute relatives)
        keep = set(PLOT_APPROACHES) | set(BASELINES)
        df = df[df["approach"].isin(keep)].copy()

        # average over repeats (if present)
        group_keys = [
            "dataset", "penalty", "alpha", "solver", "approach",
            "lambda", "tol", "max_iter", "max_iter_single"
        ]
        group_keys = [k for k in group_keys if k in df.columns]
        agg_cols = [c for c in METRICS + ["iters_single", "iters_double", "time_sec"] if c in df.columns]
        agg = {c: ["mean", "std"] for c in agg_cols}

        df_g = df.groupby(group_keys, dropna=False).agg(agg)
        df_g.columns = ["__".join(t).strip() for t in df_g.columns]
        df_g = df_g.reset_index()

        # flatten -> keep mean column with original name; std stays as *_std
        for c in agg_cols:
            mcol, scol = f"{c}__mean", f"{c}__std"
            if mcol in df_g.columns:
                df_g[c] = df_g[mcol]
                df_g.drop(columns=[mcol], inplace=True)
            if scol not in df_g.columns:
                df_g[scol] = np.nan

        return df_g

    def _slice_per_solver(self) -> List[Tuple[str, pd.DataFrame, pathlib.Path]]:
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

    # ------------- relative math -------------
    def _relative_to_baseline(self, df: pd.DataFrame, baseline: str, x_param: str,
                              strict: bool = True) -> pd.DataFrame:
        """
        Divide metric columns by baseline for same (dataset, penalty, alpha, [other params], x).
        If strict and no rows, fallback to averaging baseline across other params.
        """
        base_keys = [k for k in ["dataset", "penalty", "alpha"] if k in df.columns]
        other_params = [p for p in PARAMS if (p in df.columns and p != x_param)]
        join = base_keys + [x_param] + (other_params if strict else [])

        metrics_here = [m for m in METRICS if m in df.columns]

        bdf = df[df["approach"] == baseline].copy()
        if bdf.empty:
            return pd.DataFrame(columns=df.columns)

        bdf = (bdf[join + metrics_here]
               .groupby(join, dropna=False, as_index=False)
               .mean()
               .rename(columns={m: f"{m}_base" for m in metrics_here}))

        ldf = df[join + ["approach"] + metrics_here].copy()
        merged = pd.merge(ldf, bdf, on=join, how="inner")

        if merged.empty and strict and len(other_params) > 0:
            return self._relative_to_baseline(df, baseline, x_param, strict=False)

        for m in metrics_here:
            merged[f"{m}_rel"] = merged[m] / merged[f"{m}_base"]
        return merged

    # ------------- plotting core -------------
    def _aggregate_per_x(self, sdf: pd.DataFrame, x_param: str, y_col: str) -> pd.DataFrame:
        """one point per (approach, x) with mean & std for error bars."""
        g = (sdf.groupby(["approach", x_param], dropna=False)[y_col]
                .agg(["mean", "std"])
                .reset_index()
                .rename(columns={"mean": y_col, "std": f"{y_col}_std"}))
        return g

    def _style_axes(self, ax, x_param: str, y_label: str, title: str, relative: bool):
        ax.set_xlabel(x_param)
        ax.set_ylabel(y_label)
        ax.set_title(title, fontsize=11)
        if x_param in {"lambda", "tol"}:
            ax.set_xscale("log")
        if relative:
            ax.axhline(1.0, ls=":", c="gray", lw=1.4, label="baseline")
        ax.grid(True, ls="--", alpha=0.4)
        ax.legend(loc="best", frameon=True, fontsize=9)

    def plot_metric_vs_param_abs(self, df: pd.DataFrame, x_param: str, metric: str,
                                 save_to: pathlib.Path, title_prefix: str):
        """absolute means (no baselines)."""
        plot_df = df[df["approach"].isin(PLOT_APPROACHES)].copy()
        if plot_df.empty or plot_df[x_param].nunique(dropna=True) <= 1:
            return

        y_label = f"{metric} (seconds)" if metric == "time_sec" else metric

        agg = self._aggregate_per_x(plot_df, x_param, metric)
        plt.figure(figsize=(7.6, 5.2))
        for approach, g in agg.groupby("approach"):
            g = g.sort_values(x_param)
            plt.errorbar(g[x_param], g[metric], yerr=g.get(f"{metric}_std", None),
                         marker="o", lw=1.7, capsize=3, label=approach)
        self._style_axes(plt.gca(), x_param, y_label, f"{title_prefix}\n{y_label} vs {x_param}", relative=False)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_to, dpi=180)
        plt.close()

    def plot_metric_vs_param_rel(self, df: pd.DataFrame, x_param: str, metric: str,
                                 baseline: str, save_to: pathlib.Path, title_prefix: str):
        """relative to a baseline (single or double) → clean curve with error bars."""
        rel = self._relative_to_baseline(df, baseline=baseline, x_param=x_param, strict=True)
        if rel.empty:
            return
        rel = rel[rel["approach"].isin(PLOT_APPROACHES)].copy()
        y_col = f"{metric}_rel"
        agg = self._aggregate_per_x(rel, x_param, y_col)
        if agg[x_param].nunique(dropna=True) <= 1:
            return

        y_label = f"{metric} / {baseline}"
        plt.figure(figsize=(7.6, 5.2))
        for approach, g in agg.groupby("approach"):
            g = g.sort_values(x_param)
            plt.errorbar(g[x_param], g[y_col], yerr=g.get(f"{y_col}_std", None),
                         marker="o", lw=1.7, capsize=3, label=approach)
        self._style_axes(plt.gca(), x_param, y_label, f"{title_prefix}\n{y_label} vs {x_param}", relative=True)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_to, dpi=180)
        plt.close()

    # ------------- convenience runners -------------
    def _title_from_slice(self, s: Dict) -> str:
        parts = []
        for k in ["dataset", "penalty", "alpha"]:
            if k in s and pd.notna(s[k]):
                parts.append(f"{k}={s[k]}")
        return " | ".join(parts) if parts else "Slice"

    def _fname_from_slice(self, s: Dict) -> str:
        parts = []
        for k in ["dataset", "penalty", "alpha"]:
            if k in s and pd.notna(s[k]):
                parts.append(f"{k}-{str(s[k]).replace(' ', '')}")
        return "__".join(parts) if parts else "All"

    def make_all_plots(self):
        for solver_name, df_s, root_out in self._slice_per_solver():
            # absolute & relative dirs
            abs_dir = root_out / "absolute" / "png"
            rel_single = root_out / "relative_single_baseline" / "png"
            rel_double = root_out / "relative_double_baseline" / "png"

            # iterate slices by dataset/penalty/alpha so curves don’t mix configs
            slice_cols = [c for c in ["dataset", "penalty", "alpha"] if c in df_s.columns]
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

                ttl = self._title_from_slice(sdict)
                base_fname = self._fname_from_slice(sdict)

                # sweep each x param independently
                for xp in [p for p in PARAMS if p in sdf.columns]:
                    if sdf[xp].nunique(dropna=True) <= 1:
                        continue
                    for metric in METRICS:
                        if metric not in sdf.columns:
                            continue
                        # ABSOLUTE
                        self.plot_metric_vs_param_abs(
                            sdf, xp, metric,
                            save_to=abs_dir / f"{base_fname}__{metric}_vs_{xp}.png",
                            title_prefix=ttl
                        )
                        # RELATIVE (single & double)
                        self.plot_metric_vs_param_rel(
                            sdf, xp, metric, baseline=BASELINES[0],
                            save_to=rel_single / f"{base_fname}__{metric}_vs_{xp}__rel.png",
                            title_prefix=ttl
                        )
                        self.plot_metric_vs_param_rel(
                            sdf, xp, metric, baseline=BASELINES[1],
                            save_to=rel_double / f"{base_fname}__{metric}_vs_{xp}__rel.png",
                            title_prefix=ttl
                        )


# ---------------- run directly ----------------
if __name__ == "__main__":
    viz = LogisticVisualizer(
        csv_path="../Results/results_all.csv",
        outdir="./Figures"
    )
    viz.make_all_plots()
    print("✅ Plots written under ./Figures (per solver).")


