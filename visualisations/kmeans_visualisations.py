# kmeans_visualisations.py
# --------------------------------------------------------------------------------------
# Clean, consolidated visualisation script for Experiments A–F with memory plots.
# Key fixes:
#   • Accepts Memory_MB/mem_MB/PeakMB transparently
#   • Treats any non-{Single, Double} Suite as the "variant" (Hybrid-like) rows
#   • Normalises iter column names (iter_single/iter_double → ItersSingle/ItersDouble)
#   • Ensures Mode is present (A–F) to avoid cohort merge empties
#   • Adds conditional legend to remove "no artists with labels" warnings
#   • Provides peak-memory and estimated memory-traffic plots for D/E/F
# --------------------------------------------------------------------------------------

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

    # ---------------------- tiny helpers ----------------------
    def _legend_if_any(self, ax):
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend()

    def _variant_mask(self, d: pd.DataFrame) -> pd.Series:
        # treat any non-baseline Suite as the "variant" (Hybrid-like) rows
        return d["Suite"].notna() & ~d["Suite"].isin(["Single", "Double"])

    def _peak_col(self, df: pd.DataFrame) -> str | None:
        for c in ("PeakMB", "Memory_MB", "mem_MB"):
            if c in df.columns:
                return c
        return None

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
        if rel_df.empty:
            print("Nothing to plot for:", title)
            plt.close(fig)
            return

        for (_, _), g in rel_df.groupby(["DatasetName", "NumClusters"]):
            g = g.sort_values(xcol, key=lambda s: pd.to_numeric(s, errors="coerce"))
            ax.plot(g[xcol], g["Rel"], linestyle="-", marker="o",
                    linewidth=1.5, markersize=4, alpha=0.9, label=f"{g['DatasetName'].iat[0]}-C{g['NumClusters'].iat[0]}")

        if logx:
            ax.set_xscale("log")
        ax.axhline(1.0, ls="--", c="gray", lw=1, label=f"{baseline_label} baseline")
        ax.set_title(title)
        ax.set_xlabel(xcol)
        ax.set_ylabel(ylabel)
        ax.grid(True, ls="--", alpha=0.5)
        self._legend_if_any(ax)
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

    def _cohort_base(self, df, metric, baseline="Double"):
        return (
            df[df["Suite"] == baseline]
            .groupby(["DatasetName", "NumClusters"], as_index=False)[metric]
            .mean()
            .rename(columns={metric: "BASE"})
        )

    # ---------------------- canonicalisation ----------------------
    def _normalize_iter_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a copy of df that guarantees the presence of:
          - ItersSingle
          - ItersDouble
        If canonical names are missing, try common alternatives.
        """
        d = df.copy()
        alt_map = {
            "ItersSingle": ["iter_single", "iters_single", "it_single", "iters_f32", "IterSingle", "ItersF32"],
            "ItersDouble": ["iter_double", "iters_double", "it_double", "iters_f64", "IterDouble", "ItersF64"],
        }
        for want, alts in alt_map.items():
            if want not in d.columns:
                for a in alts:
                    if a in d.columns:
                        d[want] = d[a]
                        break
        if "ItersSingle" not in d.columns:
            d["ItersSingle"] = 0.0
        if "ItersDouble" not in d.columns:
            d["ItersDouble"] = 0.0
        return d

    def _ensure_mode(self, df: pd.DataFrame, mode_label: str) -> pd.DataFrame:
        """
        Ensure a 'Mode' column exists. If missing, create it with a constant value.
        """
        d = df.copy()
        if "Mode" not in d.columns:
            for alt in ["mode", "Experiment", "Exp", "exp"]:
                if alt in d.columns:
                    d["Mode"] = d[alt]
                    break
        if "Mode" not in d.columns:
            d["Mode"] = mode_label
        return d

    # ---------------------- 'double work' prep ----------------------
    def _prep_hybrid_double_share(
        self,
        df: pd.DataFrame,
        metric: str,
        use_share: bool = True,
        baseline: str = "Double",
    ):
        """
        Return a tidy frame with columns:
          DatasetName, NumClusters, X, Rel
        where X is either ShareDouble (ItersDouble / TotalIter) or raw ItersDouble,
        and Rel = mean(variant metric at this X) / mean(Double baseline metric).
        """
        required = {"Suite", "DatasetName", "NumClusters", "ItersSingle", "ItersDouble", metric}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"Missing required columns: {sorted(missing)}")

        d = df.copy()
        d["ItersSingle"] = d["ItersSingle"].fillna(0).astype(float)
        d["ItersDouble"] = d["ItersDouble"].fillna(0).astype(float)
        if "TotalIter" not in d.columns:
            d["TotalIter"] = d["ItersSingle"] + d["ItersDouble"]

        if use_share:
            d["X"] = d["ItersDouble"] / d["TotalIter"].replace(0, np.nan)
        else:
            d["X"] = d["ItersDouble"]

        hyb = (
            d[self._variant_mask(d)]
            .groupby(["DatasetName", "NumClusters", "X"], as_index=False)[metric]
            .mean()
            .rename(columns={metric: "VAR"})
        )
        if hyb.empty:
            return hyb

        base = self._cohort_base(d, metric, baseline=baseline)

        out = hyb.merge(base, on=["DatasetName", "NumClusters"], how="inner")
        out = out[np.isfinite(out["BASE"]) & (out["BASE"] != 0)].copy()
        if out.empty:
            return out

        out["Rel"] = out["VAR"] / out["BASE"]
        return out[["DatasetName", "NumClusters", "X", "Rel"]]

    # ---------------------- generic plots ----------------------
    def plot_with_ci(self, df, x_col, y_col, hue_col, title, xlabel, ylabel, filename):
        plt.figure(figsize=(8, 6))
        g = sns.lineplot(data=df, x=x_col, y=y_col, hue=hue_col, errorbar="ci", marker="o")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        handles, labels = g.get_legend_handles_labels()
        if labels:
            plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()

    def boxplot_comparison(self, df, x_col, y_col, hue_col, title, xlabel, ylabel, filename):
        plt.figure(figsize=(8, 6))
        g = sns.boxplot(data=df, x=x_col, y=y_col, hue=hue_col)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        handles, labels = g.get_legend_handles_labels()
        if labels:
            plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()

    # ---------------------- A/C: Cap sweeps ----------------------
    def plot_hybrid_cap_vs_inertia(self, df, baseline: str = "Double"):
        df_hybrid = df[self._variant_mask(df)]
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

        ax.set_title("Cap vs Inertia (Variant)")
        ax.set_xlabel("Cap (Single-precision iteration cap)")
        ax.set_ylabel(f"Inertia (Relative to {baseline})")
        ax.axhline(1.0, linestyle="--", color="gray", linewidth=1)

        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.5f"))
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
        self._legend_if_any(ax)
        fig.tight_layout()
        fig.savefig(self.output_dir / f"cap_vs_inertia_variant_vs_{baseline.lower()}.png", dpi=200)
        plt.close(fig)

    def plot_cap_vs_time(self, df, baseline: str = "Double"):
        df_hybrid = df[self._variant_mask(df)]
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

        ax.set_title("Cap vs Time (Variant)")
        ax.set_xlabel("Cap (Single-precision iteration cap)")
        ax.set_ylabel(f"Total Time (Relative to {baseline})")
        ax.axhline(1.0, linestyle="--", color="gray", linewidth=1)

        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        if ymins and ymaxs:
            lo, hi = float(min(ymins)), float(max(ymaxs))
            span = max(hi - lo, 1e-6); pad = max(0.05 * span, 1e-6)
            if 0.9 < lo < 1.1 and 0.9 < hi < 1.1:
                mid_pad = max(abs(1 - lo), abs(hi - 1)); pad = max(pad, 0.5 * mid_pad)
                ax.set_ylim(1 - (mid_pad + pad), 1 + (mid_pad + pad))
            else:
                ax.set_ylim(lo - pad, hi + pad)

        ax.grid(True, ls="--", alpha=0.6)
        self._legend_if_any(ax)
        fig.tight_layout()
        fig.savefig(self.output_dir / f"cap_vs_time_variant_vs_{baseline.lower()}.png", dpi=200)
        plt.close(fig)

    # ---------------------- B: tolerance sweeps ----------------------
    def plot_tolerance_vs_time(self, df, baseline: str = "Double"):
        df_hybrid = df[self._variant_mask(df)]
        group_cols = ["DatasetName", "NumClusters", "tolerance_single"]
        df_grouped = df_hybrid.groupby(group_cols)[["Time"]].mean().reset_index()

        fig, ax = plt.subplots(figsize=(7, 5))
        base = self._baseline_mean(df, ["DatasetName", "NumClusters"], "Time", baseline)
        for (ds, k), group in df_grouped.groupby(["DatasetName", "NumClusters"]):
            base_val = base[(base["DatasetName"] == ds) & (base["NumClusters"] == k)]["BASE"].mean()
            group_sorted = group.sort_values("tolerance_single")
            rel = group_sorted["Time"] / base_val
            ax.plot(group_sorted["tolerance_single"], rel, marker="o", label=f"{ds}-C{k}")

        ax.set_title("Tolerance vs Time (Variant)")
        ax.set_xlabel("Single-precision tolerance (log)")
        ax.set_xscale("log")
        ax.set_ylabel(f"Total Time (Relative to {baseline})")
        ax.axhline(1.0, linestyle="--", color="gray", linewidth=1)
        ax.grid(True)
        self._legend_if_any(ax)
        fig.tight_layout()
        fig.savefig(self.output_dir / f"tolerance_vs_time_variant_vs_{baseline.lower()}.png")
        plt.close()

    def plot_tolerance_vs_inertia(self, df, baseline: str = "Double"):
        df_hybrid = df[self._variant_mask(df)]
        group_cols = ["DatasetName", "NumClusters", "tolerance_single"]
        df_grouped = df_hybrid.groupby(group_cols)[["Inertia"]].mean().reset_index()

        fig, ax = plt.subplots(figsize=(7, 5))
        base = self._baseline_mean(df, ["DatasetName", "NumClusters"], "Inertia", baseline)
        for (ds, k), group in df_grouped.groupby(["DatasetName", "NumClusters"]):
            base_val = base[(base["DatasetName"] == ds) & (base["NumClusters"] == k)]["BASE"].mean()
            group_sorted = group.sort_values("tolerance_single")
            rel = group_sorted["Inertia"] / base_val
            ax.plot(group_sorted["tolerance_single"], rel, marker="o", label=f"{ds}-C{k}")

        ax.set_title("Tolerance vs Inertia (Variant)")
        ax.set_xlabel("Single-precision tolerance (log)")
        ax.set_xscale("log")
        ax.set_ylabel(f"Inertia (Relative to {baseline})")
        ax.axhline(1.0, linestyle="--", color="gray", linewidth=1)
        ax.grid(True)
        self._legend_if_any(ax)
        fig.tight_layout()
        fig.savefig(self.output_dir / f"tolerance_vs_inertia_variant_vs_{baseline.lower()}.png")
        plt.close()

    # ---------------------- Peak memory (A/C generic + D/E/F generalised) ----------------------
    def plot_cap_vs_peakmem(self, df, baseline: str = "Double"):
        peak = self._peak_col(df)
        df_hybrid = df[self._variant_mask(df)]
        if df_hybrid.empty or peak is None:
            print("No variant rows or PeakMB/Memory_MB column missing; skipping peak-memory plot.")
            return

        group_cols = ["DatasetName", "NumClusters", "Cap"]
        df_grouped = df_hybrid.groupby(group_cols)[[peak]].mean().reset_index()

        fig, ax = plt.subplots(figsize=(7, 5))
        base = self._baseline_mean(df, ["DatasetName", "NumClusters"], peak, baseline)
        for (ds, k), group in df_grouped.groupby(["DatasetName", "NumClusters"]):
            base_val = base[(base["DatasetName"] == ds) & (base["NumClusters"] == k)]["BASE"].mean()
            if not np.isfinite(base_val) or base_val == 0:
                continue
            g = group.sort_values("Cap").copy()
            rel = g[peak] / base_val
            ax.plot(g["Cap"], rel, marker="o", label=f"{ds}-C{k}")

        ax.set_title("Cap vs Peak Memory (Variant)")
        ax.set_xlabel("Cap (Single-precision iteration cap)")
        ax.set_ylabel(f"Peak Memory (Relative to {baseline})")
        ax.axhline(1.0, ls="--", c="gray", lw=1, label=f"{baseline} baseline")
        ax.grid(True, ls="--", alpha=0.5)
        self._legend_if_any(ax)
        fig.tight_layout()
        fig.savefig(self.output_dir / f"cap_vs_peakmem_variant_vs_{baseline.lower()}.png")
        plt.close()

    def _plot_rel_peakmem_line(self, df, keys, xcol, baseline_suite: str, title: str, outfile: str):
        """
        Plot memory_peak (variant) / memory_peak (baseline_suite) vs xcol,
        one line per (DatasetName, NumClusters).
        """
        peak = self._peak_col(df)
        need = {"Suite", "DatasetName", "NumClusters", xcol}
        if peak is None or not need.issubset(df.columns):
            print(f"Missing columns for peak-mem plot; need {need} and a memory column (PeakMB/Memory_MB/mem_MB).")
            return

        hyb = (df[self._variant_mask(df)]
               .groupby(keys, as_index=False)[peak].mean()
               .rename(columns={peak: "VAR"}))
        if hyb.empty:
            print("No variant rows for peak-mem plot.")
            return

        base = (df[df["Suite"] == baseline_suite]
                .groupby(["DatasetName", "NumClusters"], as_index=False)[peak]
                .mean().rename(columns={peak: "BASE"}))

        out = hyb.merge(base, on=["DatasetName", "NumClusters"], how="inner")
        out = out[np.isfinite(out["BASE"]) & (out["BASE"] != 0)]
        if out.empty:
            print("No valid baseline for peak-mem plot.")
            return
        out["Rel"] = out["VAR"] / out["BASE"]

        fig, ax = plt.subplots(figsize=(7, 5))
        for (ds, k), g in out.groupby(["DatasetName", "NumClusters"]):
            g = g.sort_values(xcol, key=lambda s: pd.to_numeric(s, errors="coerce"))
            ax.plot(g[xcol], g["Rel"], marker="o", label=f"{ds}-C{k}", alpha=0.9)

        ax.axhline(1.0, ls="--", c="gray", lw=1, label=f"{baseline_suite} baseline")
        ax.set_title(title)
        ax.set_xlabel(xcol)
        ax.set_ylabel(f"Peak Memory / {baseline_suite}")
        ax.grid(True, ls="--", alpha=0.6)
        self._legend_if_any(ax)
        fig.tight_layout()
        fig.savefig(self.output_dir / outfile, dpi=200)
        plt.close(fig)

    # ---------------------- Estimated memory traffic ----------------------
    def _plot_memtraffic_vs_x(self, df, xcol, mode_label: str, cohort_extra_keys=None):
        """
        TrafficRel = (T - 0.5*C) / Tdouble
          C = mean(ItersSingle)
          T = mean(TotalIter) = mean(ItersSingle + ItersDouble)
          Tdouble = median TotalIter for Double in the same cohort
        """
        d = self._normalize_iter_cols(df)
        d = self._ensure_mode(d, mode_label)

        need_basic = {"Suite", "DatasetName", "NumClusters", xcol}
        if not need_basic.issubset(d.columns):
            missing = need_basic - set(d.columns)
            print(f"Missing columns for traffic plot; need {need_basic} (missing: {missing})")
            return

        d["ItersSingle"] = d["ItersSingle"].fillna(0).astype(float)
        d["ItersDouble"] = d["ItersDouble"].fillna(0).astype(float)
        d["TotalIter"] = d["ItersSingle"] + d["ItersDouble"]

        cohort_keys = ["DatasetName", "NumClusters", "Mode"]
        if cohort_extra_keys:
            for k in cohort_extra_keys:
                if k in d.columns and k not in cohort_keys:
                    cohort_keys.append(k)

        dbl = d[(d["Suite"] == "Double") & (d["Mode"] == mode_label)]
        if dbl.empty:
            print("No Double rows for traffic baseline in this cohort; skipping.")
            return
        base = (dbl.groupby(cohort_keys, as_index=False)["TotalIter"]
                  .median().rename(columns={"TotalIter": "Tdouble"}))

        hyb = d[self._variant_mask(d) & (d["Mode"] == mode_label)]
        if hyb.empty:
            print("No variant rows for traffic plot in this cohort; skipping.")
            return
        agg_keys = cohort_keys + [xcol]
        hybG = (hyb.groupby(agg_keys, as_index=False)[["ItersSingle", "TotalIter"]]
                  .mean().rename(columns={"ItersSingle": "C", "TotalIter": "T"}))

        hybM = hybG.merge(base, on=cohort_keys, how="inner")
        hybM = hybM[np.isfinite(hybM["Tdouble"]) & (hybM["Tdouble"] > 0)]
        if hybM.empty:
            print("Traffic table empty after merging baseline; skipping.")
            return

        hybM["TrafficRel"] = (hybM["T"] - 0.5 * hybM["C"]) / hybM["Tdouble"]

        fig, ax = plt.subplots(figsize=(7, 5))
        for (ds, k), g in hybM.groupby(["DatasetName", "NumClusters"]):
            g = g.sort_values(xcol, key=lambda s: pd.to_numeric(s, errors="coerce"))
            ax.plot(g[xcol], g["TrafficRel"], marker="o", label=f"{ds}-C{k}", alpha=0.9)

        ax.axhline(1.0, ls="--", c="gray", lw=1, label="Double baseline")
        ax.set_title(f"Experiment {mode_label}: {xcol} vs Estimated Memory Traffic (Variant)")
        ax.set_xlabel(xcol)
        ax.set_ylabel("Traffic (Relative to Double)")
        ax.grid(True, ls="--", alpha=0.6)
        self._legend_if_any(ax)
        fig.tight_layout()
        fig.savefig(self.output_dir / f"exp{mode_label}_{xcol}_vs_memtraffic_variant_vs_double.png", dpi=200)
        plt.close(fig)

    # ---------------------- Double-work (relative to Single) ----------------------
    def _plot_doublework_generic(self, df, mode, metric, use_share=True, baseline="Single"):
        if df.empty:
            return
        sub = df[df["Mode"] == mode] if "Mode" in df.columns else df.copy()
        if sub.empty:
            return

        tidy = self._prep_hybrid_double_share(sub, metric=metric, use_share=use_share, baseline=baseline)
        if tidy.empty:
            return

        fig, ax = plt.subplots(figsize=(7, 5))
        for (ds, k), g in tidy.groupby(["DatasetName", "NumClusters"]):
            g = g.sort_values("X", key=lambda s: pd.to_numeric(s, errors="coerce"))
            ax.plot(g["X"], g["Rel"], marker="o", label=f"{ds}-C{k}", alpha=0.9)

        ax.axhline(1.0, ls="--", c="gray", lw=1, label=f"{baseline} baseline")
        title_x = "Remaining Double iterations" if not use_share else "Share of iterations in Double"
        ax.set_title(f"Experiment {mode}: {title_x} vs {metric} (Variant)")
        ax.set_xlabel(title_x)
        ax.set_ylabel(f"{metric} (Relative to {baseline})")
        ax.grid(True, ls="--", alpha=0.6)
        self._legend_if_any(ax)
        fig.tight_layout()
        fname = f"exp{mode}_doublework_vs_{metric.lower()}_{'iters' if not use_share else 'share'}.png"
        fig.savefig(self.output_dir / fname, dpi=200)
        plt.close(fig)

    # ---- thin wrappers you may call ----
    def plot_A_doublework_vs_time_vs_single(self, df):
        self._plot_doublework_generic(df, mode="A", metric="Time", use_share=False, baseline="Single")
    def plot_A_doublework_vs_inertia_vs_single(self, df):
        self._plot_doublework_generic(df, mode="A", metric="Inertia", use_share=False, baseline="Single")
    def plot_B_doublework_vs_time_vs_single(self, df):
        self._plot_doublework_generic(df, mode="B", metric="Time", use_share=False, baseline="Single")
    def plot_B_doublework_vs_inertia_vs_single(self, df):
        self._plot_doublework_generic(df, mode="B", metric="Inertia", use_share=False, baseline="Single")
    def plot_C_doublework_vs_time_vs_single(self, df):
        self._plot_doublework_generic(df, mode="C", metric="Time", use_share=False, baseline="Single")
    def plot_C_doublework_vs_inertia_vs_single(self, df):
        self._plot_doublework_generic(df, mode="C", metric="Inertia", use_share=False, baseline="Single")

    # ---------------------- C: cap-as-fraction plots ----------------------
    def _cap_fraction_column(self, df: pd.DataFrame) -> pd.Series:
        if "Cap" in df.columns:
            s = df["Cap"].astype(float)
            if s.max() > 1.0 and ("max_iter" in df.columns or "max_iter_C" in df.columns):
                denom = df.get("max_iter", df.get("max_iter_C")).astype(float).replace(0, np.nan)
                return (s / denom).clip(0, 1)
            return s.clip(0, 1)
        if {"single_iter_cap", "max_iter"}.issubset(df.columns):
            return (df["single_iter_cap"].astype(float) / df["max_iter"].astype(float)).clip(0, 1)
        if {"single_iter_cap", "max_iter_C"}.issubset(df.columns):
            return (df["single_iter_cap"].astype(float) / df["max_iter_C"].astype(float)).clip(0, 1)
        raise KeyError("Need 'Cap' or ('single_iter_cap' & 'max_iter[_C]') to compute fraction.")

    def plot_cap_percentage_vs_inertia(self, df, baseline: str = "Double",
                                       xlim=None, ylim=None, clip_x_at_zero=True, pad_frac: float = 0.05):
        df_h = df[self._variant_mask(df)].copy()
        if df_h.empty:
            print("No variant rows for Experiment C; skipping inertia plot.")
            return
        df_h["CapFrac"] = self._cap_fraction_column(df_h)

        grp_cols = ["DatasetName", "NumClusters", "CapFrac"]
        dfH = df_h.groupby(grp_cols, as_index=False)[["Inertia"]].mean()

        base = (
            df[(df["Suite"] == baseline)]
            .groupby(["DatasetName", "NumClusters"], as_index=False)["Inertia"].mean()
            .rename(columns={"Inertia": "BASE"})
        )

        dfM = dfH.merge(base, on=["DatasetName", "NumClusters"], how="inner")
        dfM = dfM[np.isfinite(dfM["BASE"]) & (dfM["BASE"] != 0)].copy()
        if dfM.empty:
            print(f"No valid {baseline} baseline to normalize; skipping inertia plot.")
            return
        dfM["RelInertia"] = dfM["Inertia"] / dfM["BASE"]

        fig, ax = plt.subplots(figsize=(7, 5))
        for (ds, k), g in dfM.groupby(["DatasetName", "NumClusters"]):
            g = g.sort_values("CapFrac")
            ax.plot(g["CapFrac"], g["RelInertia"], marker="o", label=f"{ds}-C{k}", alpha=0.9)

        ax.set_title("Cap (fraction) vs Final Inertia (Variant)")
        ax.set_xlabel("Cap (fraction of max_iter)")
        ax.set_ylabel(f"Inertia (Relative to {baseline})")
        ax.axhline(1.0, linestyle="--", color="gray", linewidth=1, label=f"{baseline} baseline")
        ax.grid(True, ls="--", alpha=0.5)
        self._legend_if_any(ax)

        if xlim is not None:
            ax.set_xlim(*xlim)
        else:
            xmin = 0.0 if clip_x_at_zero else float(dfM["CapFrac"].min())
            xmax = float(dfM["CapFrac"].max())
            if xmax <= xmin:
                xmax = xmin + 1e-6
            span = xmax - xmin
            ax.set_xlim(xmin, xmax + pad_frac * span)
            ax.set_xmargin(0)

        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            ymin = float(dfM["RelInertia"].min())
            ymax = float(dfM["RelInertia"].max())
            if ymax <= ymin:
                ymax = ymin + 1e-6
            yspan = ymax - ymin
            ax.set_ylim(ymin - pad_frac * yspan, ymax + pad_frac * yspan)

        fig.tight_layout()
        fig.savefig(self.output_dir / f"exp_C_cap_percentage_vs_inertia_vs_{baseline.lower()}.png", dpi=200)
        plt.close(fig)

    def plot_cap_percentage_vs_time(self, df, baseline: str = "Double"):
        df_h = df[self._variant_mask(df)].copy()
        if df_h.empty:
            print("No variant rows for Experiment C; skipping time plot.")
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

        fig, ax = plt.subplots(figsize=(7, 5))
        for (ds, k), g in dfM.groupby(["DatasetName", "NumClusters"]):
            g = g.sort_values("CapFrac")
            ax.plot(g["CapFrac"], g["RelTime"], marker="o", label=f"{ds}-C{k}", alpha=0.9)

        ax.set_title("Cap (fraction) vs Time (Variant)")
        ax.set_xlabel("Cap (fraction of max_iter)")
        ax.set_ylabel(f"Time (Relative to {baseline})")
        ax.axhline(1.0, linestyle="--", color="gray", linewidth=1, label=f"{baseline} baseline")
        ax.grid(True, ls="--", alpha=0.5)
        self._legend_if_any(ax)
        fig.tight_layout()
        fig.savefig(self.output_dir / f"exp_C_cap_percentage_vs_norm_time_vs_{baseline.lower()}.png")
        plt.close()

    # ---------------------- D/E/F plots (with memory graphs) ----------------------
    def plot_expD(self, df_D: pd.DataFrame) -> None:
        df_D = self._ensure_mode(df_D, "D")
        keys = ["DatasetName", "NumClusters", "chunk_single"]

        for base in ("Double", "Single"):
            relT = self._rel(df_D, keys, "Time", baseline_suite=base)
            relJ = self._rel(df_D, keys, "Inertia", baseline_suite=base)
            self._clean_line(relT, "chunk_single",
                             f"Experiment D: Chunk vs Relative Time (baseline={base})",
                             "Time / Baseline",
                             self.output_dir / f"expD_chunk_vs_time_vs_{base.lower()}.png",
                             baseline_label=base)
            self._clean_line(relJ, "chunk_single",
                             f"Experiment D: Chunk vs Relative Inertia (baseline={base})",
                             "Inertia / Baseline",
                             self.output_dir / f"expD_chunk_vs_inertia_vs_{base.lower()}.png",
                             baseline_label=base)

        # Peak memory (Variant / Double)
        self._plot_rel_peakmem_line(
            df_D, keys, "chunk_single", "Double",
            "Experiment D: chunk_single vs Peak Memory (Variant / Double)",
            "expD_chunk_vs_peakmem_vs_double.png"
        )

        # Estimated memory traffic
        self._plot_memtraffic_vs_x(df_D, xcol="chunk_single", mode_label="D", cohort_extra_keys=None)

    def plot_expE(self, df_E: pd.DataFrame) -> None:
        df_E = self._ensure_mode(df_E, "E")

        batch_fix = None
        if "MB_Batch" in df_E.columns and not df_E["MB_Batch"].empty:
            batch_fix = df_E["MB_Batch"].mode().iat[0]
            df_E = df_E[df_E["MB_Batch"] == batch_fix].copy()
        refine_fix = int(df_E["RefineIter"].mode().iat[0]) if "RefineIter" in df_E.columns else None
        df_use = df_E[df_E["RefineIter"] == refine_fix].copy() if refine_fix is not None else df_E.copy()

        keys = ["DatasetName", "NumClusters", "MB_Iter", "RefineIter"] if "RefineIter" in df_use.columns else ["DatasetName", "NumClusters", "MB_Iter"]
        suffix = f"(Refine={refine_fix}" + (f", Batch={batch_fix})" if batch_fix is not None else ")") if refine_fix is not None else f"(Batch={batch_fix})" if batch_fix is not None else ""

        for base in ("Double", "Single"):
            relT = self._rel(df_use, keys, "Time", baseline_suite=base)
            relJ = self._rel(df_use, keys, "Inertia", baseline_suite=base)
            self._clean_line(relT, "MB_Iter",
                             f"Experiment E: MB_Iter vs Relative Time {suffix} (baseline={base})",
                             "Time / Baseline",
                             self.output_dir / f"expE_mbiter_vs_time_vs_{base.lower()}.png",
                             baseline_label=base)
            self._clean_line(relJ, "MB_Iter",
                             f"Experiment E: MB_Iter vs Relative Inertia {suffix} (baseline={base})",
                             "Inertia / Baseline",
                             self.output_dir / f"expE_mbiter_vs_inertia_vs_{base.lower()}.png",
                             baseline_label=base)

        # Peak memory
        self._plot_rel_peakmem_line(
            df_use, keys, "MB_Iter", "Double",
            f"Experiment E: MB_Iter vs Peak Memory {suffix} (Variant / Double)",
            "expE_mbiter_vs_peakmem_vs_double.png",
        )

        # Estimated memory traffic
        extra_keys = []
        if "RefineIter" in keys:
            extra_keys.append("RefineIter")
        if batch_fix is not None and "MB_Batch" in df_use.columns:
            extra_keys.append("MB_Batch")
        self._plot_memtraffic_vs_x(df_use, xcol="MB_Iter", mode_label="E", cohort_extra_keys=extra_keys)

    def plot_expF(self, df_F: pd.DataFrame, use_log_for_tol: bool = True) -> None:
        df_F = self._ensure_mode(df_F, "F")

        tol_fix = float(df_F["tol_single"].mode().iat[0])
        sub_cap = df_F[np.isclose(df_F["tol_single"], tol_fix)].copy()
        cap_fix = int(df_F["single_iter_cap"].mode().iat[0])
        sub_tol = df_F[df_F["single_iter_cap"] == cap_fix].copy()

        for base in ("Double", "Single"):
            keys_cap = ["DatasetName", "NumClusters", "single_iter_cap", "tol_single"]
            relT_cap = self._rel(sub_cap, keys_cap, "Time", baseline_suite=base)
            relJ_cap = self._rel(sub_cap, keys_cap, "Inertia", baseline_suite=base)
            self._clean_line(relT_cap, "single_iter_cap",
                             f"Experiment F: Cap vs Relative Time (tol={tol_fix:g}, base={base})",
                             "Time / Baseline",
                             self.output_dir / f"expF_cap_vs_time_vs_{base.lower()}.png",
                             baseline_label=base)
            self._clean_line(relJ_cap, "single_iter_cap",
                             f"Experiment F: Cap vs Relative Inertia (tol={tol_fix:g}, base={base})",
                             "Inertia / Baseline",
                             self.output_dir / f"expF_cap_vs_inertia_vs_{base.lower()}.png",
                             baseline_label=base)

            keys_tol = ["DatasetName", "NumClusters", "tol_single", "single_iter_cap"]
            relT_tol = self._rel(sub_tol, keys_tol, "Time", baseline_suite=base)
            relJ_tol = self._rel(sub_tol, keys_tol, "Inertia", baseline_suite=base)
            self._clean_line(relT_tol.sort_values("tol_single"), "tol_single",
                             f"Experiment F: tol_single vs Relative Time (cap={cap_fix}, base={base})",
                             "Time / Baseline",
                             self.output_dir / f"expF_tol_vs_time_vs_{base.lower()}.png",
                             logx=use_log_for_tol, baseline_label=base)
            self._clean_line(relJ_tol.sort_values("tol_single"), "tol_single",
                             f"Experiment F: tol_single vs Relative Inertia (cap={cap_fix}, base={base})",
                             "Inertia / Baseline",
                             self.output_dir / f"expF_tol_vs_inertia_vs_{base.lower()}.png",
                             logx=use_log_for_tol, baseline_label=base)

        # Peak memory
        if "PeakMB" in df_F.columns or "Memory_MB" in df_F.columns or "mem_MB" in df_F.columns:
            self._plot_rel_peakmem_line(
                sub_cap, ["DatasetName","NumClusters","single_iter_cap","tol_single"],
                "single_iter_cap", "Double",
                f"Experiment F: Cap vs Peak Memory (tol={tol_fix:g}) (Variant / Double)",
                "expF_cap_vs_peakmem_vs_double.png",
            )
            self._plot_rel_peakmem_line(
                sub_tol, ["DatasetName","NumClusters","tol_single","single_iter_cap"],
                "tol_single", "Double",
                f"Experiment F: tol_single vs Peak Memory (cap={cap_fix}) (Variant / Double)",
                "expF_tol_vs_peakmem_vs_double.png",
            )

        # Estimated memory traffic
        self._plot_memtraffic_vs_x(sub_cap, xcol="single_iter_cap", mode_label="F",
                                   cohort_extra_keys=["tol_single"])
        self._plot_memtraffic_vs_x(sub_tol, xcol="tol_single", mode_label="F",
                                   cohort_extra_keys=["single_iter_cap"])

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


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Load results
    df_A = pd.read_csv("../Results/hybrid_kmeans_Results_expA.csv")
    df_B = pd.read_csv("../Results/hybrid_kmeans_Results_expB.csv")
    df_C = pd.read_csv("../Results/hybrid_kmeans_Results_expC.csv")
    df_D = pd.read_csv("../Results/hybrid_kmeans_Results_expD.csv")
    df_E = pd.read_csv("../Results/hybrid_kmeans_Results_expE.csv")
    df_F = pd.read_csv("../Results/hybrid_kmeans_Results_expF.csv")

    # --- Normalise suites for D/E/F so variants are clearly non-{Single,Double} ---
    df_D["Suite"] = df_D["Suite"].replace({"Adaptive": "Adaptive"})  # already variant
    df_E["Suite"] = df_E["Suite"].replace({"MiniBatch+Full": "MiniBatch+Full"})  # variant
    df_F["Suite"] = df_F["Suite"].replace({"MixedPerCluster": "MixedPerCluster"})  # variant

    # --- Normalise memory column name if needed (so peak plots always find it) ---
    for dfx in (df_A, df_B, df_C, df_D, df_E, df_F):
        if "PeakMB" not in dfx.columns:
            if "Memory_MB" in dfx.columns:
                dfx.rename(columns={"Memory_MB": "PeakMB"}, inplace=True)
            elif "mem_MB" in dfx.columns:
                dfx.rename(columns={"mem_MB": "PeakMB"}, inplace=True)

        # normalise iter names too (so A/C traffic + D/E/F traffic always work)
        if "ItersSingle" not in dfx.columns and "iter_single" in dfx.columns:
            dfx.rename(columns={"iter_single": "ItersSingle"}, inplace=True)
        if "ItersDouble" not in dfx.columns and "iter_double" in dfx.columns:
            dfx.rename(columns={"iter_double": "ItersDouble"}, inplace=True)

    vis = KMeansVisualizer(output_dir="../Results", cluster_dir="../ClusterPlots")

    # Ensure Mode labels where absent, for consistency
    if "Mode" not in df_A.columns: df_A["Mode"] = "A"
    if "Mode" not in df_B.columns: df_B["Mode"] = "B"
    if "Mode" not in df_C.columns: df_C["Mode"] = "C"
    if "Mode" not in df_D.columns: df_D["Mode"] = "D"
    if "Mode" not in df_E.columns: df_E["Mode"] = "E"
    if "Mode" not in df_F.columns: df_F["Mode"] = "F"

    # ---------------- A/C ----------------
    vis.plot_cap_vs_time(df_A, baseline="Double")
    vis.plot_cap_vs_time(df_A, baseline="Single")
    vis.plot_hybrid_cap_vs_inertia(df_A, baseline="Double")
    vis.plot_hybrid_cap_vs_inertia(df_A, baseline="Single")

    vis.plot_cap_percentage_vs_inertia(df_C, baseline="Double")
    vis.plot_cap_percentage_vs_inertia(df_C, baseline="Single")
    vis.plot_cap_percentage_vs_time(df_C, baseline="Double")
    vis.plot_cap_percentage_vs_time(df_C, baseline="Single")

    vis.plot_cap_vs_peakmem(df_A, baseline="Double")
    vis.plot_cap_vs_peakmem(df_A, baseline="Single")   # optional
    # Optional: A/C traffic plot relative to Double
    vis.plot_cap_vs_memtraffic(df_A:=df_A, baseline_double_label="Double") if hasattr(KMeansVisualizer, "plot_cap_vs_memtraffic") else None

    # ---------------- B ----------------
    vis.plot_tolerance_vs_inertia(df_B, baseline="Double")
    vis.plot_tolerance_vs_inertia(df_B, baseline="Single")
    vis.plot_tolerance_vs_time(df_B, baseline="Double")
    vis.plot_tolerance_vs_time(df_B, baseline="Single")

    # ---------------- D/E/F (includes memory graphs) ----------------
    vis.plot_expD(df_D)
    vis.plot_expE(df_E)
    vis.plot_expF(df_F)

    # ---------------- Double-work vs Single (A/B/C) ----------------
    vis.plot_A_doublework_vs_time_vs_single(df_A)
    vis.plot_A_doublework_vs_inertia_vs_single(df_A)

    vis.plot_B_doublework_vs_time_vs_single(df_B)
    vis.plot_B_doublework_vs_inertia_vs_single(df_B)

    vis.plot_C_doublework_vs_time_vs_single(df_C)
    vis.plot_C_doublework_vs_inertia_vs_single(df_C)




