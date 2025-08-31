# visualisations/svm_visualisations.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SVMVisualizer:
    """
    Plots for outputs produced by svm_precision.py (svm_precision_runs.csv).
    Expects columns like:
      dataset, approach, kernel, C, gamma,
      time_sec, accuracy, roc_auc,
      max_iter_single, tol_single, time_stageA, time_stageB, ...
    """
    def __init__(self, output_dir="Results/SUMMARY_SVM"):
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    # ------------- utilities -------------
    def _fix_approach(self, s: pd.Series) -> pd.Series:
        # unify labels a bit
        return (s.replace({
            "double(precise)": "double",
            "hybrid(SV-refit)": "hybrid",
            "adaptive-hybrid(SV-shrink)": "adaptive"
        }).fillna(s))

    def _filter_grid(self, df, dataset=None, kernel=None, C=None, gamma=None):
        q = df.copy()
        if dataset is not None: q = q[q["dataset"] == dataset]
        if kernel  is not None: q = q[q["kernel"]  == kernel]
        if C       is not None: q = q[q["C"]       == C]
        if gamma   is not None: q = q[q["gamma"]   == gamma]
        return q

    def _baseline_time(self, df_base):
        # average time of the double baseline for the fixed (dataset, kernel, C, gamma)
        base = df_base[df_base["approach"] == "double"]
        return np.nan if base.empty else float(base["time_sec"].mean())

    def _relative_curve(self, df, xcol, ycol="time_sec"):
        """
        Returns a tidy dataframe with columns:
            [xcol, 'series', 'value']
        where 'series' is 'hybrid/double' or 'adaptive/double' for time,
        or 'hybrid acc/double', etc. depending on caller.
        """
        need_cols = {xcol, "approach", ycol}
        if not need_cols.issubset(df.columns) or df.empty:
            return pd.DataFrame()

        # take group means per (xcol, approach)
        g = (df.groupby([xcol, "approach"], as_index=False)[[ycol]].mean())

        # get double baseline (constant across xcol)
        dbl = g[g["approach"] == "double"]
        if dbl.empty:
            return pd.DataFrame()

        base_val = float(dbl[ycol].mean())  # single number baseline
        out = []
        for app in ("hybrid", "adaptive"):
            sub = g[g["approach"] == app].copy()
            if sub.empty: 
                continue
            sub = sub.sort_values(xcol)
            if ycol == "time_sec":
                sub["value"] = sub[ycol] / base_val
                label = f"{app}/double (time)"
            else:
                # accuracy or roc_auc -> relative to double (app/double)
                acc_base = float(df[df["approach"] == "double"][ycol].mean())
                if not np.isfinite(acc_base) or acc_base == 0:
                    continue
                sub["value"] = sub[ycol] / acc_base
                label = f"{app}/double ({ycol})"
            out.append(sub[[xcol, "value"]].assign(series=label))
        if not out:
            return pd.DataFrame()
        return pd.concat(out, ignore_index=True)

    # ------------- plots -------------
    def plot_time_vs_maxiter(self, df, *, dataset, kernel, C, gamma):
        """
        hybrid/double and adaptive/double time ratios across max_iter_single.
        """
        d = self._filter_grid(df, dataset, kernel, C, gamma).copy()
        if d.empty or "max_iter_single" not in d.columns: 
            return
        d["approach"] = self._fix_approach(d["approach"])
        rel = self._relative_curve(d, xcol="max_iter_single", ycol="time_sec")
        if rel.empty:
            return

        plt.figure(figsize=(7,5))
        for name, g in rel.groupby("series"):
            plt.plot(g["max_iter_single"], g["value"], marker="o", label=name)
        plt.axhline(1.0, ls="--", c="gray", lw=1, label="parity (×1.0)")
        plt.title(f"{dataset}: time_sec vs max_iter_single — hybrid/adaptive vs double\n"
                  f"kernel={kernel}, C={C}, gamma={gamma}")
        plt.xlabel("max_iter_single (Stage-A budget)")
        plt.ylabel("time ratio to double (↓ is faster)")
        plt.grid(True, ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.out / f"{dataset}__time_by_max_iter_single__vs_double.png")
        plt.close()

    def plot_time_vs_tol(self, df, *, dataset, kernel, C, gamma):
        """
        hybrid/double and adaptive/double time ratios across tol_single.
        """
        d = self._filter_grid(df, dataset, kernel, C, gamma).copy()
        if d.empty or "tol_single" not in d.columns: 
            return
        d["approach"] = self._fix_approach(d["approach"])
        rel = self._relative_curve(d, xcol="tol_single", ycol="time_sec")
        if rel.empty:
            return

        plt.figure(figsize=(7,5))
        for name, g in rel.groupby("series"):
            g = g.sort_values("tol_single")
            plt.plot(g["tol_single"], g["value"], marker="o", label=name)
        plt.xscale("log")
        plt.axhline(1.0, ls="--", c="gray", lw=1, label="parity (×1.0)")
        plt.title(f"{dataset}: time_sec vs tol_single — hybrid/adaptive vs double\n"
                  f"kernel={kernel}, C={C}, gamma={gamma}")
        plt.xlabel("tol_single (Stage-A tolerance, log)")
        plt.ylabel("time ratio to double (↓ is faster)")
        plt.grid(True, ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.out / f"{dataset}__time_by_tol_single__vs_double.png")
        plt.close()

    def plot_accuracy_vs_param(self, df, *, dataset, kernel, C, gamma, param="max_iter_single", metric="accuracy"):
        """
        Relative accuracy (or roc_auc) to double across `param`.
        """
        d = self._filter_grid(df, dataset, kernel, C, gamma).copy()
        if d.empty or param not in d.columns or metric not in d.columns:
            return
        d["approach"] = self._fix_approach(d["approach"])
        rel = self._relative_curve(d, xcol=param, ycol=metric)
        if rel.empty:
            return

        plt.figure(figsize=(7,5))
        for name, g in rel.groupby("series"):
            g = g.sort_values(param)
            plt.plot(g[param], g["value"], marker="o", label=name)
        if param.lower().startswith("tol"):
            plt.xscale("log")
        plt.axhline(1.0, ls="--", c="gray", lw=1, label="double baseline")
        plt.title(f"{dataset}: {metric} vs {param} — relative to double\n"
                  f"kernel={kernel}, C={C}, gamma={gamma}")
        plt.xlabel(param)
        plt.ylabel(f"{metric} / double (≈1.0 means parity)")
        plt.grid(True, ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        fname = f"{dataset}__{metric}_by_{param}__vs_double.png".replace("/", "_")
        plt.savefig(self.out / fname)
        plt.close()

    def plot_hybrid_stage_breakdown(self, df, *, dataset, kernel, C, gamma, param="max_iter_single"):
        """
        For the 'hybrid' approach only: stacked bars (Stage-A vs Stage-B time) vs param.
        """
        d = self._filter_grid(df, dataset, kernel, C, gamma)
        if d.empty or "time_stageA" not in d.columns or "time_stageB" not in d.columns:
            return
        d = d.copy()
        d["approach"] = self._fix_approach(d["approach"])
        d = d[d["approach"] == "hybrid"]
        if d.empty or param not in d.columns: 
            return

        g = d.groupby(param, as_index=False)[["time_stageA","time_stageB"]].mean().sort_values(param)

        plt.figure(figsize=(7,5))
        plt.bar(g[param], g["time_stageA"], label="Stage-A (loose)", width=0.8, align="center")
        plt.bar(g[param], g["time_stageB"], bottom=g["time_stageA"], label="Stage-B (precise)", width=0.8, align="center")
        if str(param).lower().startswith("tol"):
            plt.xscale("log")
        plt.title(f"{dataset}: hybrid stage breakdown vs {param}\n"
                  f"kernel={kernel}, C={C}, gamma={gamma}")
        plt.xlabel(param)
        plt.ylabel("time (seconds)")
        plt.grid(True, axis="y", ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.out / f"{dataset}__hybrid_stage_breakdown_by_{param}.png")
        plt.close()

    def plot_adaptive_shrink_curve(self, df, history_col="history", *, save_csv=False):
        """
        Optional: if you kept the 'history' object, you can expand & plot working-set shrinkage.
        This requires you to have saved 'history' in your CSV (not done by default).
        """
        if history_col not in df.columns or df.empty:
            return
        # This is left as a placeholder – depends on how you export 'history'.
        pass


import pandas as pd
from visualisations.svm_visualisations import SVMVisualizer

# Load the tidy runs CSV written by run_svc_experiments
df = pd.read_csv("svm_precision_runs.csv")

viz = SVMVisualizer(output_dir="Results/SUMMARY_SVM")

# pick one grid setting to slice by (adjust to your grid)
dataset = "circles"
kernel  = "rbf"
C       = 1.0
gamma   = "scale"

viz.plot_time_vs_maxiter(df, dataset=dataset, kernel=kernel, C=C, gamma=gamma)
viz.plot_time_vs_tol(df,      dataset=dataset, kernel=kernel, C=C, gamma=gamma)

# accuracy (or roc_auc) relative to double
viz.plot_accuracy_vs_param(df, dataset=dataset, kernel=kernel, C=C, gamma=gamma,
                           param="max_iter_single", metric="accuracy")
viz.plot_accuracy_vs_param(df, dataset=dataset, kernel=kernel, C=C, gamma=gamma,
                           param="tol_single", metric="roc_auc")

# stage breakdown (hybrid only)
viz.plot_hybrid_stage_breakdown(df, dataset=dataset, kernel=kernel, C=C, gamma=gamma,
                                param="max_iter_single")

