# LOGREG_visualisations.py
# Visualizations for logreg_precision.py experiments (similar to k-means summaries)

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- config ----------------
IN_FILES = [
    "../Results/uniform_results.csv",
    "../Results/gaussian_results.csv",
    "../Results/blobs_results.csv",
    "../Results/susy_results.csv",
    "../Results/3droad_results.csv",
]
OUTDIR = Path("../Results/SUMMARY_LOGPREC")
METRICS = ["time_sec", "iters_single", "iters_double", "roc_auc", "pr_auc", "logloss"]

# parameters we’ll sweep/plot; include only those present in the CSVs
PARAM_CANDIDATES = [
    "penalty", "alpha", "lambda", "solver",
    "max_iter", "tol", "max_iter_single"
]

# ---------- helpers ----------
def _order_vals(vals):
    """Try numeric sort; otherwise keep pleasant semantic order for solvers."""
    try:
        as_float = [float(v) for v in vals]
        return [v for _, v in sorted(zip(as_float, vals))]
    except Exception:
        key = {"coord": 0, "sparse_cg": 1, "lbfgs": 2}
        return sorted(vals, key=lambda v: key.get(str(v), 9999) if isinstance(v, str) else str(v))

def _ensure_outdir():
    OUTDIR.mkdir(parents=True, exist_ok=True)

def _read_existing(files):
    frames = []
    for f in files:
        p = Path(f)
        if p.exists() and p.stat().st_size > 0:
            df = pd.read_csv(p)
            if not df.empty:
                frames.append(df)
    if not frames:
        raise FileNotFoundError("No non-empty results CSVs found.")
    return frames

def _agg_mean_std(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean/std over repeats, keeping group-key names intact.

    Pandas will produce a MultiIndex on columns; we flatten it in a way that
    leaves the group-by keys (e.g., 'dataset', 'approach') unchanged and adds
    '_mean'/'_std' to metric columns.
    """
    # only keep metrics that are present in df
    metrics_present = [m for m in METRICS if m in df.columns]
    if not metrics_present:
        raise ValueError("None of the expected metrics are present in the data.")

    group_cols = ["dataset", "approach"] + [c for c in PARAM_CANDIDATES if c in df.columns]

    # aggregate
    agg = df.groupby(group_cols, as_index=False).agg({m: ["mean", "std"] for m in metrics_present})

    # flatten multiindex columns
    flat_cols = []
    for col in agg.columns:
        if isinstance(col, tuple):
            top, sub = col
            if sub is None or sub == "":
                # group-by key -> keep as-is
                flat_cols.append(str(top))
            else:
                flat_cols.append(f"{top}_{sub}")
        else:
            flat_cols.append(str(col))
    agg.columns = flat_cols
    return agg

def _lineplot_param(df_ds, ds_name, param, metric="time_sec_mean"):
    if param not in df_ds.columns or metric not in df_ds.columns:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))

    # columns we need (std if available)
    cols = ["approach", param, metric]
    std_col = metric.replace("_mean", "_std")
    has_std = std_col in df_ds.columns
    if has_std:
        cols.append(std_col)

    d = df_ds[cols].dropna(subset=[metric]).copy()

    # order x values
    x_vals = _order_vals(d[param].unique().tolist())
    for app, sub in d.groupby("approach"):
        sub = sub.set_index(param).reindex(x_vals).reset_index()
        ax.plot(sub[param].astype(str), sub[metric].values, marker="o", label=app)
        if has_std:
            ax.fill_between(
                sub[param].astype(str),
                (sub[metric] - sub[std_col]).values,
                (sub[metric] + sub[std_col]).values,
                alpha=0.15
            )

    ax.set_xlabel(param)
    ax.set_ylabel(metric.replace("_mean", ""))
    ax.set_title(f"{ds_name}: {metric.replace('_mean','')} vs {param}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.25)
    fname = OUTDIR / f"{ds_name}__{metric}__by_{param}.png"
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname

def _bar_iters(df_ds, ds_name):
    # average (over params) the mean-iteration columns if present
    cols = [c for c in ["iters_single_mean", "iters_double_mean"] if c in df_ds.columns]
    if not cols:
        return None, None

    want = df_ds.groupby("approach", as_index=False)[cols].mean()
    if want.empty:
        return None, None

    f1 = f2 = None
    if "iters_single_mean" in want.columns:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.bar(want["approach"], want["iters_single_mean"])
        ax1.set_ylabel("iters_single")
        ax1.set_title(f"{ds_name}: mean iters_single by approach")
        ax1.grid(True, axis="y", alpha=0.25)
        f1 = OUTDIR / f"{ds_name}__iters_single_bar.png"
        fig1.tight_layout(); fig1.savefig(f1, dpi=150); plt.close(fig1)

    if "iters_double_mean" in want.columns:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(want["approach"], want["iters_double_mean"])
        ax2.set_ylabel("iters_double")
        ax2.set_title(f"{ds_name}: mean iters_double by approach")
        ax2.grid(True, axis="y", alpha=0.25)
        f2 = OUTDIR / f"{ds_name}__iters_double_bar.png"
        fig2.tight_layout(); fig2.savefig(f2, dpi=150); plt.close(fig2)

    return f1, f2

def _heatmap(df_ds, ds_name, metric="time_sec_mean",
             x="tol", y="max_iter_single"):
    if (x not in df_ds.columns) or (y not in df_ds.columns) or (metric not in df_ds.columns):
        return []
    paths = []
    for app, sub in df_ds.groupby("approach"):
        piv = sub.pivot_table(index=y, columns=x, values=metric, aggfunc="mean")
        if piv.empty or piv.shape[0] < 1 or piv.shape[1] < 1:
            continue
        x_order = _order_vals(piv.columns.tolist())
        y_order = _order_vals(piv.index.tolist())
        piv = piv.loc[y_order, x_order]

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        im = ax.imshow(piv.values, aspect="auto")
        ax.set_xticks(range(len(piv.columns)), labels=[str(v) for v in piv.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(piv.index)), labels=[str(v) for v in piv.index])
        ax.set_xlabel(x); ax.set_ylabel(y)
        ax.set_title(f"{ds_name} [{app}]: {metric.replace('_mean','')}")
        fig.colorbar(im, ax=ax, shrink=0.85)
        fig.tight_layout()
        fp = OUTDIR / f"{ds_name}__{app}__{metric}__heatmap_{y}_vs_{x}.png"
        fig.savefig(fp, dpi=150)
        plt.close(fig)
        paths.append(fp)
    return paths

def _write_md(ds_name, tables, figs):
    md = OUTDIR / f"{ds_name}__summary.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write(f"# {ds_name} – logreg precision visualizations\n\n")
        if "best_time" in tables:
            f.write("## Fastest config per approach (lower is better)\n\n")
            f.write(tables["best_time"].to_markdown(index=False))
            f.write("\n\n")

        if "mean_scores" in tables:
            f.write("## Mean metrics by approach\n\n")
            f.write(tables["mean_scores"].to_markdown(index=False))
            f.write("\n\n")

        f.write("## Figures\n\n")
        for title, path in figs:
            f.write(f"**{title}**  \n")
            f.write(f"![{title}]({path.as_posix()})\n\n")
    return md

# ---------- main ----------
def main():
    _ensure_outdir()
    frames = _read_existing(IN_FILES)
    df = pd.concat(frames, ignore_index=True)

    # keep successful runs only
    if "approach" in df.columns:
        df = df[df["approach"] != "ERROR"].copy()

    # aggregate mean/std over repeats
    agg = _agg_mean_std(df)

    # per dataset
    for ds, ds_df in agg.groupby("dataset"):
        ds_df = ds_df.copy()

        # tables
        # 1) fastest config per approach by time_sec_mean
        metric_name = "time_sec_mean" if "time_sec_mean" in ds_df.columns else None
        tables = {}

        cols_for_id = ["approach"] + [c for c in PARAM_CANDIDATES if c in ds_df.columns]
        if metric_name:
            fastest = (
                ds_df.loc[ds_df.groupby("approach")[metric_name].idxmin(), cols_for_id + [metric_name]]
                     .sort_values(metric_name)
                     .rename(columns={metric_name: "best_time_sec"})
            )
            tables["best_time"] = fastest

        # 2) mean metrics by approach (time/iters/aucs)
        mean_cols = [c for c in [
            "time_sec_mean", "iters_single_mean", "iters_double_mean",
            "roc_auc_mean", "pr_auc_mean", "logloss_mean"
        ] if c in ds_df.columns]

        if mean_cols:
            mean_scores = (
                ds_df.groupby("approach", as_index=False)[mean_cols]
                     .mean()
                     .rename(columns=lambda c: re.sub("_mean$", "", c))
            )
            tables["mean_scores"] = mean_scores

        # figures
        figs = []

        # lines: time vs each parameter that varies
        for p in [c for c in PARAM_CANDIDATES if c in ds_df.columns]:
            if metric_name and ds_df[p].nunique() > 1:
                path = _lineplot_param(ds_df, ds, param=p, metric=metric_name)
                if path:
                    figs.append((f"time vs {p}", path))

        # bars: iters
        f1, f2 = _bar_iters(ds_df, ds)
        if f1: figs.append(("iters_single by approach", f1))
        if f2: figs.append(("iters_double by approach", f2))

        # heatmaps for 2-D sweeps (if both axes vary)
        if metric_name and ("tol" in ds_df.columns) and ("max_iter_single" in ds_df.columns):
            if ds_df["tol"].nunique() > 1 and ds_df["max_iter_single"].nunique() > 1:
                for pth in _heatmap(ds_df, ds, metric=metric_name, x="tol", y="max_iter_single"):
                    figs.append(("time heatmap (tol × max_iter_single)", pth))

        md_path = _write_md(ds, tables, figs)
        print(f"wrote: {md_path}")

if __name__ == "__main__":
    main()
