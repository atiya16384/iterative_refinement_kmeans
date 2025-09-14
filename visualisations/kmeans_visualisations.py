import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pathlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter


class KMeansVisualizer:
    def __init__(self, output_dir="Results", cluster_dir="ClusterPlots"):
        self.output_dir = pathlib.Path(output_dir)
        self.cluster_dir = pathlib.Path(cluster_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cluster_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------- small utilities ----------------------
    @staticmethod
    def _rel(df: pd.DataFrame, keys, value_col: str, baseline_suite: str = "Double") -> pd.DataFrame:
        """
        Build a dataframe keyed by `keys` with columns:
            [*keys, value_col, BASE, Rel, Baseline]
        where Rel = variant / baseline (mean within each group).
        Baseline can be "Double" or "Single".
        """
        base = (
            df[df["Suite"] == baseline_suite]
            .groupby(keys, as_index=False)[value_col].mean()
            .rename(columns={value_col: "BASE"})
        )
        var = (
            df[df["Suite"] != baseline_suite]
            .groupby(keys, as_index=False)[value_col].mean()
        )
        out = var.merge(base, on=keys, how="inner")
        out["Rel"] = out[value_col] / out["BASE"]
        out["Baseline"] = baseline_suite
        return out

    # REMOVE the @staticmethod above _clean_line

    def _clean_line(
        self,
        rel_df: pd.DataFrame,
        xcol: str,
        title: str,
        ylabel: str,
        outpath=None,
        logx: bool = False,
        baseline_label: str = "Double",
    ) -> None:
        fig, ax = plt.subplots(figsize=(7, 5))
    
        # one line per (dataset, k)
        for (_, _), g in rel_df.groupby(["DatasetName", "NumClusters"]):
            # robust numeric sort (handles str columns)
            g = g.sort_values(xcol, key=lambda s: pd.to_numeric(s, errors="coerce"))
            # force a line (linestyle='-') + markers
            ax.plot(g[xcol], g["Rel"], linestyle="-", marker="o", linewidth=1.5, markersize=4, alpha=0.8)
    
        if logx:
            ax.set_xscale("log")
        ax.axhline(1.0, ls="--", c="gray", lw=1, label=f"{baseline_label} baseline")
        ax.set_title(title)
        ax.set_xlabel(xcol)
        ax.set_ylabel(ylabel)
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        if outpath is not None:
            fig.savefig(outpath, dpi=200)
        plt.close(fig)
    


    def _baseline_mean(self, df, keys, value_col, baseline_suite):
        """Return baseline means with columns = keys + ['BASE']."""
        return (
            df[df["Suite"] == baseline_suite]
            .groupby(keys, as_index=False)[value_col].mean()
            .rename(columns={value_col: "BASE"})
        )

    # ---------------------- generic plots ----------------------
    def plot_with_ci(self, df, x_col, y_col, hue_col, title, xlabel, ylabel, filename):
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=df, x=x_col, y=y_col, hue=hue_col, errorbar="ci", marker="o")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()
        print(f"Saved CI plot to {self.output_dir / filename}")

    def boxplot_comparison(self, df, x_col, y_col, hue_col, title, xlabel, ylabel, filename):
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x=x_col, y=y_col, hue=hue_col)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()
        print(f"Saved boxplot to {self.output_dir / filename}")

    # ---------------------- A/C: Cap sweeps ----------------------
    def plot_hybrid_cap_vs_inertia(self, df, baseline: str = "Double"):
        df_hybrid = df[df["Suite"] == "Hybrid"]
        group_cols = ["DatasetName", "NumClusters", "Cap"]
        df_grouped = df_hybrid.groupby(group_cols)[["Inertia"]].mean().reset_index()
    
        fig, ax = plt.subplots(figsize=(7, 5))
        base = self._baseline_mean(df, ["DatasetName", "NumClusters"], "Inertia", baseline)
    
        ymins, ymaxs = [], []
        plotted = False
        for (ds, k), group in df_grouped.groupby(["DatasetName", "NumClusters"]):
            group_sorted = group.sort_values("Cap")
            base_val = base[(base["DatasetName"] == ds) & (base["NumClusters"] == k)]["BASE"].mean()
            if not np.isfinite(base_val) or base_val == 0:
                continue
            rel = group_sorted["Inertia"] / base_val
            ax.plot(group_sorted["Cap"], rel, marker="o", label=f"{ds}-C{k}")
            ymins.append(rel.min()); ymaxs.append(rel.max())
            plotted = True
    
        ax.set_title("Cap vs Inertia (Hybrid)")
        ax.set_xlabel("Cap (Single-precision iteration cap)")
        ax.set_ylabel(f"Inertia (Relative to {baseline})")
        ax.axhline(1.0, linestyle="--", color="gray", linewidth=1)
    
        # Friendly y-axis: no offset; fixed decimals; zoom around 1.0
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.5f"))
        if ymins and ymaxs:
            lo, hi = float(min(ymins)), float(max(ymaxs))
            span = max(hi - lo, 1e-6)
            pad = max(0.05 * span, 1e-6)
            # If values are very close to 1, center around 1 for readability
            if 0.9 < lo < 1.1 and 0.9 < hi < 1.1:
                mid_pad = max(abs(1 - lo), abs(hi - 1))
                pad = max(pad, 0.5 * mid_pad)
                ax.set_ylim(1 - (mid_pad + pad), 1 + (mid_pad + pad))
            else:
                ax.set_ylim(lo - pad, hi + pad)
    
        ax.grid(True, ls="--", alpha=0.6)
        if plotted:
            ax.legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / f"cap_vs_inertia_hybrid_vs_{baseline.lower()}.png", dpi=200)
        plt.close(fig)
    
    def plot_cap_vs_time(self, df, baseline: str = "Double"):
        df_hybrid = df[df["Suite"] == "Hybrid"]
        group_cols = ["DatasetName", "NumClusters", "Cap"]
        df_grouped = df_hybrid.groupby(group_cols)[["Time"]].mean().reset_index()
    
        fig, ax = plt.subplots(figsize=(7, 5))
        base = self._baseline_mean(df, ["DatasetName", "NumClusters"], "Time", baseline)
    
        ymins, ymaxs = [], []
        plotted = False
        for (ds, k), group in df_grouped.groupby(["DatasetName", "NumClusters"]):
            base_val = base[(base["DatasetName"] == ds) & (base["NumClusters"] == k)]["BASE"].mean()
            if not np.isfinite(base_val) or base_val == 0:
                continue
            group_sorted = group.sort_values("Cap")
            rel = group_sorted["Time"] / base_val
            ax.plot(group_sorted["Cap"], rel, marker="o", label=f"{ds}-C{k}")
            ymins.append(rel.min()); ymaxs.append(rel.max())
            plotted = True
    
        ax.set_title("Cap vs Time (Hybrid)")
        ax.set_xlabel("Cap (Single-precision iteration cap)")
        ax.set_ylabel(f"Total Time (Relative to {baseline})")
        ax.axhline(1.0, linestyle="--", color="gray", linewidth=1)
    
        # Friendly y-axis formatting (plain numbers; gentle zoom if near 1)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        if ymins and ymaxs:
            lo, hi = float(min(ymins)), float(max(ymaxs))
            span = max(hi - lo, 1e-6)
            pad = max(0.05 * span, 1e-6)
            if 0.9 < lo < 1.1 and 0.9 < hi < 1.1:
                mid_pad = max(abs(1 - lo), abs(hi - 1))
                pad = max(pad, 0.5 * mid_pad)
                ax.set_ylim(1 - (mid_pad + pad), 1 + (mid_pad + pad))
            else:
                ax.set_ylim(lo - pad, hi + pad)
    
        ax.grid(True, ls="--", alpha=0.6)
        if plotted:
            ax.legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / f"cap_vs_time_hybrid_vs_{baseline.lower()}.png", dpi=200)
        plt.close(fig)

    # ---------------------- B: tolerance sweeps ----------------------
    def plot_tolerance_vs_time(self, df, baseline: str = "Double"):
        df_hybrid = df[df["Suite"] == "Hybrid"]
        group_cols = ["DatasetName", "NumClusters", "tolerance_single"]
        df_grouped = df_hybrid.groupby(group_cols)[["Time"]].mean().reset_index()

        plt.figure(figsize=(7, 5))
        base = self._baseline_mean(df, ["DatasetName", "NumClusters"], "Time", baseline)
        for (ds, k), group in df_grouped.groupby(["DatasetName", "NumClusters"]):
            base_val = base[(base["DatasetName"] == ds) & (base["NumClusters"] == k)]["BASE"].mean()
            group_sorted = group.sort_values("tolerance_single")
            group_sorted["Time"] = group_sorted["Time"] / base_val
            plt.plot(group_sorted["tolerance_single"], group_sorted["Time"], marker="o")

        plt.title("Tolerance vs Time (Hybrid)")
        plt.xlabel("Single-precision tolerance")
        plt.xscale("log")
        plt.ylabel(f"Total Time (Relative to {baseline})")
        plt.axhline(1.0, linestyle="--", color="gray", linewidth=1)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / f"tolerance_vs_time_hybrid_vs_{baseline.lower()}.png")
        plt.close()

    def plot_tolerance_vs_inertia(self, df, baseline: str = "Double"):
        df_hybrid = df[df["Suite"] == "Hybrid"]
        group_cols = ["DatasetName", "NumClusters", "tolerance_single"]
        df_grouped = df_hybrid.groupby(group_cols)[["Inertia"]].mean().reset_index()

        plt.figure(figsize=(7, 5))
        base = self._baseline_mean(df, ["DatasetName", "NumClusters"], "Inertia", baseline)
        for (ds, k), group in df_grouped.groupby(["DatasetName", "NumClusters"]):
            base_val = base[(base["DatasetName"] == ds) & (base["NumClusters"] == k)]["BASE"].mean()
            group_sorted = group.sort_values("tolerance_single")
            group_sorted["Inertia"] = group_sorted["Inertia"] / base_val
            plt.plot(group_sorted["tolerance_single"], group_sorted["Inertia"], marker="o")

        plt.title("Tolerance vs Inertia (Hybrid)")
        # plt.ylim(0.9999, 1.0001)
        plt.xlabel("Single-precision tolerance (log)")
        plt.xscale("log")
        plt.ylabel(f"Inertia (Relative to {baseline})")
        plt.axhline(1.0, linestyle="--", color="gray", linewidth=1)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / f"tolerance_vs_inertia_hybrid_vs_{baseline.lower()}.png")
        plt.close()

    def plot_cap_vs_peakmem(self, df, baseline: str = "Double"):
        df_hybrid = df[df["Suite"] == "Hybrid"]
        if df_hybrid.empty or "PeakMB" not in df.columns:
            print("No Hybrid rows or PeakMB column missing; skipping peak-memory plot.")
            return
    
        group_cols = ["DatasetName", "NumClusters", "Cap"]
        df_grouped = df_hybrid.groupby(group_cols)[["PeakMB"]].mean().reset_index()
    
        plt.figure(figsize=(7, 5))
        base = self._baseline_mean(df, ["DatasetName", "NumClusters"], "PeakMB", baseline)
        for (ds, k), group in df_grouped.groupby(["DatasetName", "NumClusters"]):
            base_val = base[(base["DatasetName"] == ds) & (base["NumClusters"] == k)]["BASE"].mean()
            if not np.isfinite(base_val) or base_val == 0:
                continue
            g = group.sort_values("Cap").copy()
            g["PeakMB_rel"] = g["PeakMB"] / base_val
            plt.plot(g["Cap"], g["PeakMB_rel"], marker="o", label=f"{ds}-C{k}")
    
        plt.title("Cap vs Peak Memory (Hybrid)")
        plt.xlabel("Cap (Single-precision iteration cap)")
        plt.ylabel(f"Peak Memory (Relative to {baseline})")
        plt.axhline(1.0, ls="--", c="gray", lw=1, label=f"{baseline} baseline")
        plt.grid(True, ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / f"cap_vs_peakmem_hybrid_vs_{baseline.lower()}.png")
        plt.close()
    def plot_cap_vs_memtraffic(self, df, baseline_double_label: str = "Double"):
    """
        Cap vs Estimated Memory Traffic (Hybrid), relative to Double.
        TrafficRel = (T - 0.5*C) / Tdouble, where:
            C = ItersSingle (float32 iters)
            T = ItersSingle + ItersDouble (total iters actually run)
            Tdouble = median Double TotalIter for the *same cohort*.
        Cohort keys: DatasetName, NumClusters, Mode, tolerance_single (if present).
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
    
        need = {"ItersSingle","ItersDouble","Suite","DatasetName","NumClusters","Cap","Mode"}
        if not need.issubset(df.columns):
            print("Missing columns for memory-traffic plot; need", need)
            return
    
        d = df.copy()
    
        # ---- (1) keep only A/C (or choose what you want explicitly)
        d = d[d["Mode"].isin(["A","C"])].copy()
    
        # ---- (2) total iterations per run
        d["ItersSingle"] = d["ItersSingle"].fillna(0).astype(float)
        d["ItersDouble"] = d["ItersDouble"].fillna(0).astype(float)
        d["TotalIter"]   = d["ItersSingle"] + d["ItersDouble"]
    
        # ---- (3) cohort keys: include tolerance if present (A uses fixed tol but safe)
        cohort_keys = ["DatasetName","NumClusters","Mode"]
        if "tolerance_single" in d.columns:
            cohort_keys.append("tolerance_single")
    
        # ---- (4) Double baseline Tdouble (median per cohort)
        dbl = d[d["Suite"] == baseline_double_label].copy()
        if dbl.empty:
            print("No Double rows; skipping memory-traffic plot.")
            return
        # median across repeats (and across cap duplicates in A/C)
        base = (dbl.groupby(cohort_keys, as_index=False)["TotalIter"]
                   .median()
                   .rename(columns={"TotalIter":"Tdouble"}))
    
        # ---- (5) Hybrid rows; aggregate repeats per (cohort + Cap)
        hyb = d[d["Suite"] == "Hybrid"].copy()
        if hyb.empty:
            print("No Hybrid rows; skipping memory-traffic plot.")
            return
    
        # mean across repeats at each Cap (you can use median if you prefer)
        agg_keys = cohort_keys + ["Cap"]
        hybG = (hyb.groupby(agg_keys, as_index=False)[["ItersSingle","TotalIter"]]
                    .mean()
                    .rename(columns={"ItersSingle":"C","TotalIter":"T"}))
    
        # attach matching Tdouble
        hybM = hybG.merge(base, on=cohort_keys, how="inner")
        hybM = hybM[np.isfinite(hybM["Tdouble"]) & (hybM["Tdouble"] > 0)].copy()
    
        # ---- (6) traffic model: f32 iteration = 0.5 of f64 iteration
        hybM["TrafficRel"] = (hybM["T"] - 0.5 * hybM["C"]) / hybM["Tdouble"]
    
        # ---- (7) plot one line per dataset/k (combine cohorts with same tol/mode implicitly)
        plt.figure(figsize=(7,5))
        for (ds, k), g in hybM.groupby(["DatasetName","NumClusters"]):
            g = g.sort_values("Cap", key=lambda s: pd.to_numeric(s, errors="coerce"))
            plt.plot(g["Cap"], g["TrafficRel"], marker="o", label=f"{ds}-C{k}", alpha=0.9)
    
        plt.title("Cap vs Estimated Memory Traffic (Hybrid)")
        plt.xlabel("Cap (Single-precision iteration cap)")
        plt.ylabel("Traffic (Relative to Double)")
        plt.axhline(1.0, ls="--", c="gray", lw=1, label="Double baseline")
        plt.grid(True, ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "cap_vs_memtraffic_hybrid_vs_double.png", dpi=200)
        plt.close()
    
        # Optional sanity check: if any cap==0 is far from 1.0, print the cohorts
        bad0 = hybM[(hybM["Cap"].astype(float)==0) & (np.abs(hybM["TrafficRel"]-1.0)>0.1)]
        if not bad0.empty:
            print(" Sanity check: cap=0 far from 1.0 for cohorts:")
            print(bad0[cohort_keys + ["TrafficRel"]].drop_duplicates().to_string(index=False))
    

    def _cap_fraction_column(self, df: pd.DataFrame) -> pd.Series:
        """
        Return a Series with the cap fraction in [0,1].
        Supports either:
          - df['Cap'] already being a fraction, OR
          - df has 'single_iter_cap' and 'max_iter' (or 'max_iter_C').
        """
        if "Cap" in df.columns:
            s = df["Cap"].astype(float)
            # if Cap looks like an integer count, try to divide by max_iter if present
            if s.max() > 1.0 and ("max_iter" in df.columns or "max_iter_C" in df.columns):
                denom = df.get("max_iter", df.get("max_iter_C")).astype(float).replace(0, np.nan)
                return (s / denom).clip(0, 1)
            return s.clip(0, 1)
    
        if {"single_iter_cap", "max_iter"}.issubset(df.columns):
            return (df["single_iter_cap"].astype(float) / df["max_iter"].astype(float)).clip(0, 1)
        if {"single_iter_cap", "max_iter_C"}.issubset(df.columns):
            return (df["single_iter_cap"].astype(float) / df["max_iter_C"].astype(float)).clip(0, 1)
    
        raise KeyError("Need a 'Cap' fraction column or ('single_iter_cap' & 'max_iter[_C]') to compute it.")

    # ---------------------- C: cap-as-fraction plots ----------------------
    def plot_cap_percentage_vs_inertia(self, df, baseline: str = "Double"):
        # Hybrid rows only; compute/normalize cap fraction
        df_h = df[df["Suite"] == "Hybrid"].copy()
        if df_h.empty:
            print("No Hybrid rows for Experiment C; skipping inertia plot.")
            return
    
        df_h["CapFrac"] = self._cap_fraction_column(df_h)
    
        # Aggregate repeats
        grp_cols = ["DatasetName", "NumClusters", "CapFrac"]
        dfH = df_h.groupby(grp_cols, as_index=False)[["Inertia"]].mean()
    
        # Baseline (mean over repeats) per dataset/cluster
        base = (
            df[(df["Suite"] == baseline)]
            .groupby(["DatasetName", "NumClusters"], as_index=False)["Inertia"].mean()
            .rename(columns={"Inertia": "BASE"})
        )
    
        # Merge baseline onto hybrid means, compute relative inertia
        dfM = dfH.merge(base, on=["DatasetName", "NumClusters"], how="inner")
        dfM = dfM[np.isfinite(dfM["BASE"]) & (dfM["BASE"] != 0)].copy()
        if dfM.empty:
            print(f"No valid {baseline} baseline to normalize; skipping inertia plot.")
            return
        dfM["RelInertia"] = dfM["Inertia"] / dfM["BASE"]
    
        plt.figure(figsize=(7, 5))
        for (ds, k), g in dfM.groupby(["DatasetName", "NumClusters"]):
            g = g.sort_values("CapFrac")
            plt.plot(g["CapFrac"], g["RelInertia"], marker="o", label=f"{ds}-C{k}", alpha=0.9)
    
    
        plt.title("Cap (fraction) vs Final Inertia (Hybrid)")
        plt.xlabel("Cap (fraction of max_iter)")
        plt.ylabel(f"Inertia (Relative to {baseline})")
        plt.axhline(1.0, linestyle="--", color="gray", linewidth=1, label=f"{baseline} baseline")
        plt.grid(True, ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / f"exp_C_cap_percentage_vs_inertia_vs_{baseline.lower()}.png")
        plt.close()

    def plot_cap_percentage_vs_time(self, df, baseline: str = "Double"):
        df_h = df[df["Suite"] == "Hybrid"].copy()
        if df_h.empty:
            print("No Hybrid rows for Experiment C; skipping time plot.")
            return
    
        df_h["CapFrac"] = self._cap_fraction_column(df_h)
    
        grp_cols = ["DatasetName", "NumClusters", "CapFrac"]
        dfH = df_h.groupby(grp_cols, as_index=False)[["Time"]].mean()
    
        base = (
            df[(df["Suite"] == baseline)]
            .groupby(["DatasetName", "NumClusters"], as_index=False)["Time"].mean()
            .rename(columns={"Time": "BASE"})
        )
    
        dfM = dfH.merge(base, on=["DatasetName", "NumClusters"], how="inner")
        dfM = dfM[np.isfinite(dfM["BASE"]) & (dfM["BASE"] != 0)].copy()
        if dfM.empty:
            print(f"No valid {baseline} baseline to normalize; skipping time plot.")
            return
        dfM["RelTime"] = dfM["Time"] / dfM["BASE"]
    
        plt.figure(figsize=(7, 5))
        for (ds, k), g in dfM.groupby(["DatasetName", "NumClusters"]):
            g = g.sort_values("CapFrac")
            plt.plot(g["CapFrac"], g["RelTime"], marker="o", label=f"{ds}-C{k}", alpha=0.9)
    
        plt.title("Cap (fraction) vs Time (Hybrid)")
        plt.xlabel("Cap (fraction of max_iter)")
        plt.ylabel(f"Time (Relative to {baseline})")
        plt.axhline(1.0, linestyle="--", color="gray", linewidth=1, label=f"{baseline} baseline")
        plt.grid(True, ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / f"exp_C_cap_percentage_vs_norm_time_vs_{baseline.lower()}.png")
        plt.close()

    # ---------------------- D/E/F plots (produce both baselines internally) ----------------------
    def plot_expD(self, df_D: pd.DataFrame) -> None:
        keys = ["DatasetName", "NumClusters", "chunk_single"]
        for base in ("Double", "Single"):
            relT = self._rel(df_D, keys, "Time", baseline_suite=base)
            relJ = self._rel(df_D, keys, "Inertia", baseline_suite=base)
            self._clean_line(
                relT, "chunk_single",
                f"Experiment D: Chunk vs Relative Time (baseline={base})",
                "Time / Baseline",
                self.output_dir / f"expD_chunk_vs_time_vs_{base.lower()}.png",
                baseline_label=base,
            )
            self._clean_line(
                relJ, "chunk_single",
                f"Experiment D: Chunk vs Relative Inertia (baseline={base})",
                "Inertia / Baseline",
                self.output_dir / f"expD_chunk_vs_inertia_vs_{base.lower()}.png",
                baseline_label=base,
            )

    def plot_expE(self, df_E: pd.DataFrame) -> None:
        # Fix batch size (if present) to the most common value; fix RefineIter to mode
        if "MB_Batch" in df_E.columns and not df_E["MB_Batch"].empty:
            batch_fix = df_E["MB_Batch"].mode().iat[0]
            df_E = df_E[df_E["MB_Batch"] == batch_fix].copy()
        else:
            batch_fix = None
        refine_fix = int(df_E["RefineIter"].mode().iat[0])
        df_use = df_E[df_E["RefineIter"] == refine_fix].copy()

        keys = ["DatasetName", "NumClusters", "MB_Iter", "RefineIter"]
        suffix = f"(Refine={refine_fix}" + (f", Batch={batch_fix})" if batch_fix is not None else ")")

        for base in ("Double", "Single"):
            relT = self._rel(df_use, keys, "Time", baseline_suite=base)
            relJ = self._rel(df_use, keys, "Inertia", baseline_suite=base)
            self._clean_line(
                relT, "MB_Iter",
                f"Experiment E: MB_Iter vs Relative Time {suffix} (baseline={base})",
                "Time / Baseline",
                self.output_dir / f"expE_mbiter_vs_time_vs_{base.lower()}.png",
                baseline_label=base,
            )
            self._clean_line(
                relJ, "MB_Iter",
                f"Experiment E: MB_Iter vs Relative Inertia {suffix} (baseline={base})",
                "Inertia / Baseline",
                self.output_dir / f"expE_mbiter_vs_inertia_vs_{base.lower()}.png",
                baseline_label=base,
            )

    def plot_expF(self, df_F: pd.DataFrame, use_log_for_tol: bool = True) -> None:
        tol_fix = float(df_F["tol_single"].mode().iat[0])
        sub_cap = df_F[np.isclose(df_F["tol_single"], tol_fix)].copy()
        cap_fix = int(df_F["single_iter_cap"].mode().iat[0])
        sub_tol = df_F[df_F["single_iter_cap"] == cap_fix].copy()

        for base in ("Double", "Single"):
            # (a) Cap sweep at fixed tol
            keys_cap = ["DatasetName", "NumClusters", "single_iter_cap", "tol_single"]
            relT_cap = self._rel(sub_cap, keys_cap, "Time", baseline_suite=base)
            relJ_cap = self._rel(sub_cap, keys_cap, "Inertia", baseline_suite=base)
            self._clean_line(
                relT_cap, "single_iter_cap",
                f"Experiment F: Cap vs Relative Time (tol={tol_fix:g}, base={base})",
                "Time / Baseline",
                self.output_dir / f"expF_cap_vs_time_vs_{base.lower()}.png",
                baseline_label=base,
            )
            self._clean_line(
                relJ_cap, "single_iter_cap",
                f"Experiment F: Cap vs Relative Inertia (tol={tol_fix:g}, base={base})",
                "Inertia / Baseline",
                self.output_dir / f"expF_cap_vs_inertia_vs_{base.lower()}.png",
                baseline_label=base,
            )

            # (b) tol sweep at fixed cap
            keys_tol = ["DatasetName", "NumClusters", "tol_single", "single_iter_cap"]
            relT_tol = self._rel(sub_tol, keys_tol, "Time", baseline_suite=base)
            relJ_tol = self._rel(sub_tol, keys_tol, "Inertia", baseline_suite=base)
            self._clean_line(
                relT_tol.sort_values("tol_single"),
                "tol_single",
                f"Experiment F: tol_single vs Relative Time (cap={cap_fix}, base={base})",
                "Time / Baseline",
                self.output_dir / f"expF_tol_vs_time_vs_{base.lower()}.png",
                logx=use_log_for_tol,
                baseline_label=base,
            )
            self._clean_line(
                relJ_tol.sort_values("tol_single"),
                "tol_single",
                f"Experiment F: tol_single vs Relative Inertia (cap={cap_fix}, base={base})",
                "Inertia / Baseline",
                self.output_dir / f"expF_tol_vs_inertia_vs_{base.lower()}.png",
                logx=use_log_for_tol,
                baseline_label=base,
            )

    # ---------------------- cluster visual aids ----------------------
    @staticmethod
    def pca_2d_view(X_full, centers_full, resolution=300, random_state=0):
        pca = PCA(n_components=2, random_state=random_state)
        X_vis = pca.fit_transform(X_full)
        centers_vis = pca.transform(centers_full)

        x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
        y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                             np.linspace(y_min, y_max, resolution))
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        labels_grid = pairwise_distances_argmin(grid_points, centers_vis).reshape(xx.shape)
        return X_vis, centers_vis, xx, yy, labels_grid

    @staticmethod
    def plot_clusters(X_vis, labels, centers_vis, xx, yy, labels_grid, title="", filename=""):
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, labels_grid, cmap="Pastel1", alpha=0.2)
        if len(X_vis) > 5000:
            idx = np.random.choice(len(X_vis), size=5000, replace=False)
        else:
            idx = np.arange(len(X_vis))
        cmap = cm.get_cmap("tab20", np.unique(labels).size)
        plt.scatter(X_vis[idx, 0], X_vis[idx, 1], c=labels[idx], s=8, cmap=cmap, alpha=0.7, edgecolors="none")
        plt.scatter(centers_vis[:, 0], centers_vis[:, 1], c="black", marker="x", s=120, linewidths=2, label="Centers")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        cluster_dir = pathlib.Path("ClusterPlots")
        cluster_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(cluster_dir / f"{filename}.png")
        plt.close()


if __name__ == "__main__":
    # Load results
    df_A = pd.read_csv("../Results/hybrid_kmeans_Results_expA.csv")
    df_B = pd.read_csv("../Results/hybrid_kmeans_Results_expB.csv")
    df_C = pd.read_csv("../Results/hybrid_kmeans_Results_expC.csv")
    df_D = pd.read_csv("../Results/hybrid_kmeans_Results_expD.csv")
    df_E = pd.read_csv("../Results/hybrid_kmeans_Results_expE.csv")
    df_F = pd.read_csv("../Results/hybrid_kmeans_Results_expF.csv")

    vis = KMeansVisualizer(output_dir="../Results", cluster_dir="../ClusterPlots")

    # A/C: cap-based
    vis.plot_cap_vs_time(df_A, baseline="Double")
    vis.plot_cap_vs_time(df_A, baseline="Single")
    vis.plot_hybrid_cap_vs_inertia(df_A, baseline="Double")
    vis.plot_hybrid_cap_vs_inertia(df_A, baseline="Single")

    # B: tolerance-based
    vis.plot_tolerance_vs_inertia(df_B, baseline="Double")
    vis.plot_tolerance_vs_inertia(df_B, baseline="Single")
    vis.plot_tolerance_vs_time(df_B, baseline="Double")
    vis.plot_tolerance_vs_time(df_B, baseline="Single")

    # C: cap as fraction (same CSV layout as A if you used Cap fraction there)
    vis.plot_cap_percentage_vs_inertia(df_C, baseline="Double")
    vis.plot_cap_percentage_vs_inertia(df_C, baseline="Single")
    vis.plot_cap_percentage_vs_time(df_C, baseline="Double")
    vis.plot_cap_percentage_vs_time(df_C, baseline="Single")

    # D/E/F: these produce *both* baselines internally
    vis.plot_expD(df_D)
    vis.plot_expE(df_E)
    vis.plot_expF(df_F)
        # Memory plots (A/C style datasets)
    vis.plot_cap_vs_peakmem(df_A, baseline="Double")
    vis.plot_cap_vs_peakmem(df_A, baseline="Single")   # optional
    
    vis.plot_cap_vs_memtraffic(df_A)  # relative to Double baseline
    















