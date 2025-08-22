# logistic_visualisations.py
# Visualizations for results produced by log_precision.py (run_experiments).
# Generates absolute and relative-to-baseline plots for multiple metric/param pairs.
#
# Run:
#   python3 logistic_visualisations.py

import os
import itertools
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Config you may tweak
# -----------------------------

# Where your run_experiments CSV lives and where to write figures
CSV_PATH = "../Results/results_all.csv"
OUTDIR = "./Figures"
FORMATS = ["png"]        # e.g., ["png", "pdf", "svg"]
SHOW = False             # True pops windows; False saves quietly
MAKE_ABSOLUTE = True
MAKE_RELATIVE = True

# Only plot THESE approaches (baselines are used for normalization only)
PLOT_APPROACHES = ["hybrid(f32→f64)", "multistage-IR", "adaptive-precision"]

# Baselines (must match the strings produced upstream)
BASELINES = ("single(f32)", "double(f64)")

# Which parameters (x-axes) and which metrics (y-axes) to sweep
PARAMS = ["lambda", "tol", "max_iter", "max_iter_single"]
METRICS = ["logloss", "time_sec", "roc_auc", "pr_auc"]

# Markers / linestyles for consistency
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]

# Slicing dimensions (a “slice” keeps these fixed while sweeping a single x param)
BASE_GROUP_COLS = ["dataset", "penalty", "alpha", "solver"]

# Optional extra in titles/filenames if present
OPTIONAL_META = ["C"]


# -----------------------------
# Helpers
# -----------------------------

def _safe_unique_sorted(series: pd.Series):
    vals = [v for v in series.dropna().unique().tolist()]
    try:
        return sorted(vals)
    except Exception:
        return vals


def _shorten(val):
    if isinstance(val, float):
        if val == int(val):
            return str(int(val))
        if (abs(val) < 1e-3) or (abs(val) >= 1e3):
            return f"{val:.2e}"
        return f"{val:.4g}"
    return str(val)


def _title_from_slice(slice_dict: Dict) -> str:
    parts = []
    for k in BASE_GROUP_COLS + OPTIONAL_META:
        if k in slice_dict:
            parts.append(f"{k}={_shorten(slice_dict[k])}")
    return " | ".join(parts) if parts else "All Experiments"


def _fname_from_slice(slice_dict: Dict) -> str:
    parts = []
    for k in BASE_GROUP_COLS + OPTIONAL_META:
        if k in slice_dict:
            v = str(slice_dict[k]).replace(" ", "").replace("→", "to").replace("/", "_")
            parts.append(f"{k}-{v}")
    return "__".join(parts) if parts else "All"


def _ensure_outdirs(base_outdir: str, rel_tag: str, formats: List[str]) -> Dict[str, str]:
    out_abs = os.path.join(base_outdir, "absolute")
    out_rel = os.path.join(base_outdir, f"relative_{rel_tag}")
    os.makedirs(out_abs, exist_ok=True)
    os.makedirs(out_rel, exist_ok=True)
    for d in (out_abs, out_rel):
        for fmt in formats:
            os.makedirs(os.path.join(d, fmt), exist_ok=True)
    return {"abs": out_abs, "rel": out_rel}


def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected_cols = {
        "dataset", "approach", "penalty", "alpha", "lambda", "solver",
        "max_iter", "max_iter_single", "tol",
        "time_sec", "iters_single", "iters_double",
        "roc_auc", "pr_auc", "logloss"
    }
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        warnings.warn(f"Missing expected columns: {missing}. Proceeding anyway.")

    # Categorical ordering for consistent legends (include baselines so merges work fine)
    cat_order = PLOT_APPROACHES + list(BASELINES)
    df["approach"] = pd.Categorical(df["approach"], categories=cat_order, ordered=True)
    return df


def compute_relative_to_baseline(df: pd.DataFrame, baseline_approach: str, x_param: str) -> pd.DataFrame:
    """
    For each slice and x_param value, divide metric columns by the baseline’s metric
    from the SAME slice and SAME x_param. Adds <metric>_rel. Drops rows without baseline.
    """
    keep_cols = [c for c in (BASE_GROUP_COLS + OPTIONAL_META) if c in df.columns]
    on_cols = keep_cols + [x_param]

    bdf = df[df["approach"] == baseline_approach].copy()
    if bdf.empty:
        return pd.DataFrame(columns=df.columns)

    metric_cols = [m for m in METRICS if m in df.columns]
    bcols = on_cols + metric_cols
    bdf = bdf[bcols].rename(columns={m: f"{m}_baseline" for m in metric_cols})

    merged = pd.merge(df, bdf, on=on_cols, how="inner", validate="many_to_one")

    for m in metric_cols:
        base = f"{m}_baseline"
        rel = f"{m}_rel"
        merged[rel] = np.where(merged[base].astype(float) == 0.0, np.nan,
                               merged[m].astype(float) / merged[base].astype(float))
    return merged


def _iter_slices(df: pd.DataFrame):
    group_cols = [c for c in BASE_GROUP_COLS if c in df.columns]
    opt_cols = [c for c in OPTIONAL_META if c in df.columns]
    all_cols = group_cols + opt_cols

    if not all_cols:
        yield ({}, df)
        return

    for keys, sdf in df.groupby(all_cols, dropna=False):
        if len(all_cols) == 1:
            keys = (keys,)
        slice_dict = {col: keys[i] for i, col in enumerate(all_cols)}
        yield (slice_dict, sdf)


def _plot_xy_lines(
    sdf: pd.DataFrame,
    x_param: str,
    y_metric: str,
    out_dir: str,
    fname_prefix: str,
    title_prefix: str,
    relative: bool,
    formats: List[str],
    show: bool,
    ylog: bool = False,
    baseline_label: str = None
):
    fig, ax = plt.subplots(figsize=(8.2, 5.2))

    y_col = f"{y_metric}_rel" if relative else y_metric
    y_label = f"{y_metric} / {baseline_label}" if relative else y_metric

    # choose approaches to plot (ONLY the three requested)
    present = sdf["approach"].astype(str).unique().tolist()
    approaches = [a for a in PLOT_APPROACHES if a in present]

    mk_cycle = itertools.cycle(MARKERS)
    ls_cycle = itertools.cycle(LINESTYLES)

    for approach in approaches:
        adf = sdf[sdf["approach"] == approach].copy()
        if adf.empty:
            continue
        try:
            adf = adf.sort_values(by=x_param, key=lambda s: s.astype(float))
        except Exception:
            adf = adf.sort_values(by=x_param)

        ax.plot(
            adf[x_param], adf[y_col],
            marker=next(mk_cycle),
            linestyle=next(ls_cycle),
            linewidth=1.7,
            markersize=6,
            label=approach
        )

    if relative:
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.5, label=f"{baseline_label} baseline")

    ax.set_xlabel(x_param)
    ax.set_ylabel(y_label)
    ax.set_title(f"{title_prefix}\n{y_label} vs {x_param}", fontsize=11)
    ax.grid(True, alpha=0.25)

    if x_param in ["lambda", "tol"]:
        try:
            ax.set_xscale("log")
        except Exception:
            pass
    if ylog:
        try:
            ax.set_yscale("log")
        except Exception:
            pass

    ax.legend(loc="best", fontsize=9, frameon=True)

    fname_core = f"{fname_prefix}__{y_metric}_vs_{x_param}{'__rel' if relative else ''}"
    for fmt in formats:
        path = os.path.join(out_dir, fmt, f"{fname_core}.{fmt}")
        fig.savefig(path, bbox_inches="tight", dpi=180)

    if show:
        plt.show()
    else:
        plt.close(fig)


def make_plots_for_csv(
    csv_path: str,
    outdir: str,
    make_absolute: bool = True,
    make_relative: bool = True,
    baselines: Tuple[str, str] = BASELINES,
    formats: List[str] = tuple(FORMATS),
    show: bool = SHOW,
):
    df = load_results(csv_path)

    # Ensure output dirs
    baseline_tags = {
        baselines[0]: "single_baseline",
        baselines[1]: "double_baseline",
    }
    if make_absolute:
        os.makedirs(os.path.join(outdir, "absolute"), exist_ok=True)
        for fmt in formats:
            os.makedirs(os.path.join(outdir, "absolute", fmt), exist_ok=True)

    if make_relative:
        for b, tag in baseline_tags.items():
            _ensure_outdirs(outdir, tag, formats)

    # Iterate over slices and plot
    for slice_dict, sdf in _iter_slices(df):
        title_prefix = _title_from_slice(slice_dict)
        fname_prefix = _fname_from_slice(slice_dict)

        # ABSOLUTE (only the three requested approaches)
        if make_absolute:
            plot_df = sdf[sdf["approach"].astype(str).isin(PLOT_APPROACHES)].copy()
            for x_param in PARAMS:
                if x_param not in plot_df.columns:
                    continue
                if plot_df[x_param].nunique(dropna=True) <= 1:
                    continue
                for y_metric in METRICS:
                    if y_metric not in plot_df.columns:
                        continue
                    ylog = (y_metric == "time_sec")
                    _plot_xy_lines(
                        sdf=plot_df, x_param=x_param, y_metric=y_metric,
                        out_dir=os.path.join(outdir, "absolute"),
                        fname_prefix=fname_prefix,
                        title_prefix=title_prefix,
                        relative=False,
                        formats=formats,
                        show=show,
                        ylog=ylog,
                        baseline_label=None
                    )

        # RELATIVE (two sets: single and double baselines)
        if make_relative:
            for baseline in baselines:
                for xp in [p for p in PARAMS if p in sdf.columns]:
                    rel_df_all = compute_relative_to_baseline(sdf, baseline_approach=baseline, x_param=xp)
                    if rel_df_all.empty or rel_df_all[xp].nunique(dropna=True) <= 1:
                        continue
                    # only plot the three approaches; keep baseline rows out of plots
                    rel_df = rel_df_all[rel_df_all["approach"].astype(str).isin(PLOT_APPROACHES)].copy()
                    if rel_df.empty:
                        continue
                    for y_metric in METRICS:
                        if f"{y_metric}_rel" not in rel_df.columns:
                            continue
                        ylog = (y_metric == "time_sec")
                        _plot_xy_lines(
                            sdf=rel_df,
                            x_param=xp,
                            y_metric=y_metric,
                            out_dir=os.path.join(outdir, f"relative_{'single_baseline' if baseline == baselines[0] else 'double_baseline'}"),
                            fname_prefix=fname_prefix,
                            title_prefix=title_prefix,
                            relative=True,
                            formats=formats,
                            show=show,
                            ylog=ylog,
                            baseline_label=baseline
                        )


# -----------------------------
# Run directly
# -----------------------------
if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    make_plots_for_csv(
        csv_path=CSV_PATH,
        outdir=OUTDIR,
        make_absolute=MAKE_ABSOLUTE,
        make_relative=MAKE_RELATIVE,
        baselines=BASELINES,
        formats=FORMATS,
        show=SHOW,
    )
    print(f"Figures saved under {OUTDIR}")


