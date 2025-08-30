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
METRICS = ["time_sec", "roc_auc", "pr_auc", "logloss"]

# parameters we may sweep; we’ll auto-skip ones not present
PARAM_CANDIDATES = [
    "lambda", 
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
def _plot_hybrid_ratio_both_baselines(df_ds, ds_name, param, metric_mean_col):
    """
    Single chart with two lines:
      - hybrid / single(f32)
      - hybrid / double(f64)
    Uses exact-config pairing (dataset + all params) and then averages ratios per x.
    """
    if param not in df_ds.columns or "approach" not in df_ds.columns or metric_mean_col not in df_ds.columns:
        return None

    # keep all params so we can pair exactly
    cols_keep = ["dataset", "approach", param] + [c for c in PARAM_CANDIDATES if c in df_ds.columns]
    cols_keep = list(dict.fromkeys(cols_keep))
    cols_keep += [metric_mean_col]

    d = df_ds[cols_keep].dropna(subset=[metric_mean_col]).copy()
    if d.empty:
        return None

    hy = d[d["approach"] == APP_HYBRID].copy()
    if hy.empty:
        return None

    def _pair_and_ratio(baseline_label):
        bl = d[d["approach"] == baseline_label].copy()
        if bl.empty:
            return None
        pair_keys = ["dataset"] + [c for c in PARAM_CANDIDATES if c in d.columns]
        merged = hy.merge(bl, on=pair_keys, suffixes=("_hy", "_bl"), how="inner")
        if merged.empty:
            return None
        num = merged[f"{metric_mean_col}_hy"].astype(float)
        den = merged[f"{metric_mean_col}_bl"].astype(float).replace(0.0, np.nan)
        merged["ratio"] = (num / den).replace([np.inf, -np.inf], np.nan)
        merged = merged.dropna(subset=["ratio"])
        if merged.empty:
            return None
        series = merged.groupby(param, as_index=True)["ratio"].mean()
        return series

    r_single = _pair_and_ratio(APP_SINGLE)
    r_double = _pair_and_ratio(APP_DOUBLE)
    if r_single is None and r_double is None:
        return None

    # union of x values we have for either baseline, ordered nicely
    x_all = set()
    if r_single is not None: x_all |= set(r_single.index.tolist())
    if r_double is not None: x_all |= set(r_double.index.tolist())
    x_order = _order_vals(list(x_all))

    # plot
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    # draw baseline parity line
    ax.axhline(1.0, linestyle=":", linewidth=1.5, color="grey", label="baseline parity (×1.0)")

    def _plot_one(series, label, marker):
        s = series.reindex(x_order) if series is not None else None
        if s is None: 
            return
        ax.plot([str(v) for v in s.index], s.values, marker=marker, label=label)

    _plot_one(r_single,  f"{APP_HYBRID} / {APP_SINGLE}  (hybrid ÷ single)", marker="o")
    _plot_one(r_double,  f"{APP_HYBRID} / {APP_DOUBLE}  (hybrid ÷ double)", marker="s")

    y_label_raw = re.sub("_mean$", "", metric_mean_col)
    ax.set_xlabel(param)
    ax.set_ylabel(f"{y_label_raw} (ratio to baseline)")
    ax.set_title(f"{ds_name}: {y_label_raw} vs {param} — hybrid vs single & double")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    out = OUTDIR / f"{ds_name}__{y_label_raw}__by_{param}__hybrid_over_single_and_double.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out

def _make_all_ratio_plots_for_dataset(ds_df, ds_name):
    paths = []
    metric_mean_cols = [f"{m}_mean" for m in METRICS if f"{m}_mean" in ds_df.columns]
    if not metric_mean_cols:
        return paths

    varying_params = [p for p in PARAM_CANDIDATES if p in ds_df.columns and ds_df[p].nunique() > 1]
    for param in varying_params:
        for mcol in metric_mean_cols:
            p_both = _plot_hybrid_ratio_both_baselines(ds_df, ds_name, param, mcol)
            if p_both is not None:
                title = f"{ds_name}: {re.sub('_mean$','', mcol)} vs {param} — hybrid vs single & double"
                paths.append((title, p_both))
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
