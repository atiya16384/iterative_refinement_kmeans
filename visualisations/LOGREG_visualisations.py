# LOGREG_visualisations.py
# Visualizations for logreg_precision.py experiments (hybrid normalized to single/double)

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

# metrics emitted by logreg_precision.py (we'll normalize these)
METRICS = ["time_sec", "iters_single", "iters_double", "roc_auc", "pr_auc", "logloss"]

# parameters we may sweep; we’ll auto-skip ones not present
PARAM_CANDIDATES = [
    "penalty", "alpha", "lambda", "solver",
    "max_iter", "tol", "max_iter_single"
]

# approach labels as produced by your runner
APP_SINGLE = "single(f32)"
APP_DOUBLE = "double(f64)"
APP_HYBRID = "hybrid(f32→f64)"

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
    """Aggregate mean/std over repeats, flatten MultiIndex nicely."""
    metrics_present = [m for m in METRICS if m in df.columns]
    if not metrics_present:
        raise ValueError("None of the expected metrics are present in the data.")

    group_cols = ["dataset", "approach"] + [c for c in PARAM_CANDIDATES if c in df.columns]
    agg = df.groupby(group_cols, as_index=False).agg({m: ["mean", "std"] for m in metrics_present})

    flat_cols = []
    for col in agg.columns:
        if isinstance(col, tuple):
            top, sub = col
            if sub in (None, ""):
                flat_cols.append(str(top))
            else:
                flat_cols.append(f"{top}_{sub}")
        else:
            flat_cols.append(str(col))
    agg.columns = flat_cols
    return agg

# ---------- plotting ----------
def _plot_hybrid_ratio_one_baseline(df_ds, ds_name, param, metric_mean_col, baseline_label):
    """
    Make a single line plot for (param vs hybrid/baseline) where baseline is APP_SINGLE or APP_DOUBLE.
    We first collapse over all *other* params so we have one value per (approach, param).
    """
    # safety: need param, approach, and metric column
    if param not in df_ds.columns or "approach" not in df_ds.columns or metric_mean_col not in df_ds.columns:
        return None

    # keep just what we need
    d = df_ds[["approach", param, metric_mean_col]].dropna(subset=[metric_mean_col]).copy()
    if d.empty:
        return None

    # average over everything except (approach,param)
    d = d.groupby(["approach", param], as_index=False)[metric_mean_col].mean()

    # pivot to have columns by approach
    piv = d.pivot(index=param, columns="approach", values=metric_mean_col)

    # ensure needed columns
    if APP_HYBRID not in piv.columns or baseline_label not in piv.columns:
        return None

    # compute ratio; drop rows where baseline is 0 or NaN
    base = piv[baseline_label]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = piv[APP_HYBRID] / base
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()

    if ratio.empty:
        return None

    # order x for nicer plotting
    idx_ordered = _order_vals(ratio.index.tolist())
    ratio = ratio.reindex(idx_ordered)

    # plot
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x_labels = [str(v) for v in ratio.index.tolist()]
    ax.plot(x_labels, ratio.values, marker="o", label=f"{APP_HYBRID} / {baseline_label}")

    # grey dotted baseline at 1.0
    ax.axhline(1.0, linestyle=":", linewidth=1.5, color="grey")

    # labels/titles
    y_label_raw = re.sub("_mean$", "", metric_mean_col)
    ax.set_xlabel(param)
    ax.set_ylabel(f"{y_label_raw} (hybrid / baseline)")
    pretty_base = "single(f32)" if baseline_label == APP_SINGLE else "double(f64)"
    ax.set_title(f"{ds_name}: {y_label_raw} vs {param} — hybrid ÷ {pretty_base}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    out = OUTDIR / f"{ds_name}__{y_label_raw}__by_{param}__hybrid_over_{'single' if baseline_label==APP_SINGLE else 'double'}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out

def _make_all_ratio_plots_for_dataset(ds_df, ds_name):
    """For each param that varies and each metric found, emit two plots:
       hybrid/single and hybrid/double. Returns list of (title, path) pairs."""
    paths = []
    # which metric mean columns do we have?
    metric_mean_cols = [f"{m}_mean" for m in METRICS if f"{m}_mean" in ds_df.columns]
    if not metric_mean_cols:
        return paths

    # loop params that actually vary
    varying_params = [p for p in PARAM_CANDIDATES if p in ds_df.columns and ds_df[p].nunique() > 1]
    for param in varying_params:
        for mcol in metric_mean_cols:
            p1 = _plot_hybrid_ratio_one_baseline(ds_df, ds_name, param, mcol, APP_SINGLE)
            if p1 is not None:
                title = f"{ds_name}: {re.sub('_mean$','', mcol)} vs {param} — hybrid ÷ single"
                paths.append((title, p1))
            p2 = _plot_hybrid_ratio_one_baseline(ds_df, ds_name, param, mcol, APP_DOUBLE)
            if p2 is not None:
                title = f"{ds_name}: {re.sub('_mean$','', mcol)} vs {param} — hybrid ÷ double"
                paths.append((title, p2))
    return paths

def _write_md(ds_name, figs):
    md = OUTDIR / f"{ds_name}__summary.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write(f"# {ds_name} – Hybrid speed/quality ratios\n\n")
        f.write("All plots show **hybrid ÷ baseline** with a grey dotted line at 1.0 (baseline).\n\n")
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

    # aggregate mean/std over repeats & configs
    agg = _agg_mean_std(df)

    # per dataset
    for ds, ds_df in agg.groupby("dataset"):
        ds_df = ds_df.copy()

        # make ratio plots for every metric × varying-param
        figs = _make_all_ratio_plots_for_dataset(ds_df, ds)

        # write a simple index markdown
        md_path = _write_md(ds, figs)
        print(f"wrote: {md_path}")

if __name__ == "__main__":
    main()
