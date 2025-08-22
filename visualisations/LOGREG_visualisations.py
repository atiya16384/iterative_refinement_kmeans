# logistic_visualisations.py
# -------------------------------------------------------------------
# Visualizations for results produced by log_precision.py (run_experiments).
# Plots only: hybrid(f32→f64), multistage-IR, adaptive-precision
# Baselines: single(f32), double(f64) for relative plots.
#
# Run:
#   python3 logistic_visualisations.py
#
# Outputs (per solver):
#   ./Figures/solver=<SOLVER>/absolute/<fmt>/*.png
#   ./Figures/solver=<SOLVER>/relative_single_baseline/<fmt>/*.png
#   ./Figures/solver=<SOLVER>/relative_double_baseline/<fmt>/*.png
# -------------------------------------------------------------------

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

CSV_PATH = "../Results/results_all.csv"
OUTDIR = "../Results/Figures"
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

# Base identifiers for a dataset slice. We keep solver at the folder level.
BASE_GROUP_COLS = ["dataset", "penalty", "alpha"]

# Optional extra in titles/filenames if present
OPTIONAL_META = ["C"]

# -----------------------------
# Helpers
# -----------------------------

def _is_nan(x) -> bool:
    try:
        return pd.isna(x)
    except Exception:
        return False

def _safe_unique_sorted(series: pd.Series):
    vals = [v for v in series.dropna().unique().tolist()]
    try:
        return sorted(vals)
    except Exception:
        return vals

def _shorten(val):
    """Compact pretty-print for titles/filenames, robust to NaN/mixed types."""
    if _is_nan(val):
        return "NA"
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, (float, np.floating)):
        if not np.isfinite(val):
            return "NA"
        if abs(val - round(val)) < 1e-12:
            return str(int(round(val)))
        if (abs(val) < 1e-3) or (abs(val) >= 1e3):
            return f"{val:.2e}"
        return f"{val:.4g}"
    return str(val)

def _title_from_slice(slice_dict: Dict) -> str:
    parts = []
    for k in BASE_GROUP_COLS + OPTIONAL_META:
        if k in slice_dict and not _is_nan(slice_dict[k]):
            parts.append(f"{k}={_shorten(slice_dict[k])}")
    return " | ".join(parts) if parts else "All Experiments"

def _fname_from_slice(slice_dict: Dict) -> str:
    parts = []
    for k in BASE_GROUP_COLS + OPTIONAL_META:
        if k in slice_dict and not _is_nan(slice_dict[k]):
            v = str(_shorten(slice_dict[k])).replace(" ", "").replace("→", "to").replace("/", "_")
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

# -----------------------------
# Data load & aggregation
# -----------------------------

def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected_cols = {
        "dataset", "approach", "penalty", "alpha", "lambda", "solver",
        "max_iter", "max_iter_single", "tol",
        "time_sec", "iters_single", "iters_double",
        "roc_auc", "pr_auc", "logloss", "repeat"
    }
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        # repeat may not exist in the saved CSV; that’s fine
        missing = [m for m in missing if m != "repeat"]
    if missing:
        warnings.warn(f"Missing expected columns: {missing}. Proceeding anyway.")

    # Coerce param columns to numeric
    for c in ["lambda", "tol", "max_iter", "max_iter_single"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Average across repeats so we plot the MEAN curve
    has_repeat = "repeat" in df.columns
    group_keys = [
        "dataset", "penalty", "alpha", "solver", "approach",
        "lambda", "tol", "max_iter", "max_iter_single"
    ]
    group_keys = [k for k in group_keys if k in df.columns]

    agg = {}
    for m in METRICS + ["iters_single", "iters_double"]:
        if m in df.columns:
            agg[m] = ["mean", "std"]
    if agg:
        g = df.groupby(group_keys, dropna=False).agg(agg)
        g.columns = ["__".join(col).strip() for col in g.columns.values]
        g = g.reset_index()
        # flatten to ..._mean columns; keep std as ..._std (for possible error bars)
        for m in list(agg.keys()):
            mean_col = f"{m}__mean"
            std_col  = f"{m}__std"
            if mean_col in g.columns:
                g[m] = g[mean_col]
                g.drop(columns=[mean_col], inplace=True)
            if std_col not in g.columns:
                g[std_col] = np.nan
        df = g

    # Categorical ordering for consistent legends (include baselines so merges work)
    cat_order = PLOT_APPROACHES + list(BASELINES)
    df["approach"] = pd.Categorical(df["approach"], categories=cat_order, ordered=True)
    return df

# -----------------------------
# Slicing helpers so other params are FIXED when we sweep one x
# -----------------------------

def _iter_param_slices(df: pd.DataFrame, x_param: str):
    """Yield (slice_info, sub_df) where sub_df varies in x_param only;
       all other PARAMS are held constant."""
    other_params = [p for p in PARAMS if p in df.columns and p != x_param]
    slice_cols = [c for c in BASE_GROUP_COLS if c in df.columns] + other_params + [x_param, "approach"]
    # group by everything BUT x_param and approach to build a slice with fixed others
    group_cols = [c for c in slice_cols if c not in [x_param, "approach"]]
    if not group_cols:
        yield ({}, df)
        return
    for keys, sdf in df.groupby(group_cols, dropna=False):
        if len(group_cols) == 1:
            keys = (keys,)
        slice_dict = {col: keys[i] for i, col in enumerate(group_cols) if col in BASE_GROUP_COLS or col in OPTIONAL_META}
        yield (slice_dict, sdf)

# -----------------------------
# Baseline math
# -----------------------------

def compute_relative_to_baseline(
    df: pd.DataFrame,
    baseline_approach: str,
    x_param: str,
    strict: bool = True,
) -> pd.DataFrame:
    """
    For each (slice with fixed other params) + x, divide metric columns by the baseline metric.
    If strict=True, the join keys include the *other params*, guaranteeing an exact match.
    If that yields no rows (no exact baseline), we fall back to a relaxed join that
    averages the baseline across the other params.
    """
    # Identify join keys
    keep_cols = [c for c in BASE_GROUP_COLS if c in df.columns]
    other_params = [p for p in PARAMS if p in df.columns and p != x_param]
    if strict:
        join_cols = keep_cols + [x_param] + other_params
    else:
        join_cols = keep_cols + [x_param]

    # Build baseline table (unique per join key)
    metric_cols = [m for m in METRICS if m in df.columns]
    bdf = df[df["approach"] == baseline_approach].copy()
    if bdf.empty:
        return pd.DataFrame(columns=df.columns)
    bcols = [c for c in join_cols if c in bdf.columns] + metric_cols
    bdf = (
        bdf[bcols]
        .groupby(join_cols, dropna=False, as_index=False)[metric_cols]
        .mean()
        .rename(columns={m: f"{m}_baseline" for m in metric_cols})
    )

    # Left side (only approaches we care about + baseline columns needed for division)
    lcols = [c for c in join_cols if c in df.columns] + ["approach"] + metric_cols
    ldf = df[lcols].copy()

    merged = pd.merge(ldf, bdf, on=join_cols, how="inner")

    # If strict produced nothing but we have data, try relaxed
    if merged.empty and strict and len(other_params) > 0:
        return compute_relative_to_baseline(df, baseline_approach, x_param, strict=False)

    # Compute ratios
    for m in metric_cols:
        base = f"{m}_baseline"
        rel = f"{m}_rel"
        merged[rel] = np.where(merged[base].astype(float) == 0.0, np.nan,
                               merged[m].astype(float) / merged[base].astype(float))
    return merged

# -----------------------------
# Plotters
# -----------------------------

def _maybe_log_y_for_time(y_values: np.ndarray) -> bool:
    """Use log scale for 'time_sec' only if the spread is huge (>50×)."""
    y_values = y_values[np.isfinite(y_values)]
    if y_values.size == 0:
        return False
    mn, mx = np.nanmin(y_values), np.nanmax(y_values)
    return (mx > 0) and (mn > 0) and (mx / max(mn, 1e-12) > 50.0)

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
    baseline_label: str = None
):
    fig, ax = plt.subplots(figsize=(8.2, 5.2))

    y_col = f"{y_metric}_rel" if relative else y_metric
    y_label = (f"{y_metric} / {baseline_label}") if relative else (
        f"{y_metric} (seconds)" if y_metric == "time_sec" else y_metric
    )

    present = sdf["approach"].astype(str).unique().tolist()
    approaches = [a for a in PLOT_APPROACHES if a in present]

    mk_cycle = itertools.cycle(MARKERS)
    ls_cycle = itertools.cycle(LINESTYLES)

    for approach in approaches:
        adf = sdf[sdf["approach"] == approach].copy()
        if adf.empty:
            continue
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

    # Time in seconds: use linear unless spread is huge
    yvals = sdf[y_col].values.astype(float)
    ylog = (y_metric == "time_sec") and _maybe_log_y_for_time(yvals)
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

# -----------------------------
# Orchestration
# -----------------------------

def make_plots_for_df(
    df: pd.DataFrame,
    outdir: str,
    make_absolute: bool = True,
    make_relative: bool = True,
    baselines: Tuple[str, str] = BASELINES,
    formats: List[str] = tuple(FORMATS),
    show: bool = SHOW,
):
    baseline_tags = {baselines[0]: "single_baseline", baselines[1]: "double_baseline"}

    if make_absolute:
        os.makedirs(os.path.join(outdir, "absolute"), exist_ok=True)
        for fmt in formats:
            os.makedirs(os.path.join(outdir, "absolute", fmt), exist_ok=True)
    if make_relative:
        for _, tag in baseline_tags.items():
            _ensure_outdirs(outdir, tag, formats)

    # For each x_param, we iterate slices where OTHER params are fixed
    for x_param in [p for p in PARAMS if p in df.columns]:
        for slice_dict, sdf in _iter_param_slices(df, x_param):
            # titles/filenames reflect dataset/penalty/alpha (others fixed anyway)
            title_prefix = _title_from_slice(slice_dict)
            fname_prefix = _fname_from_slice(slice_dict)

            # Absolute: just the three approaches
            if make_absolute:
                plot_df = sdf[sdf["approach"].astype(str).isin(PLOT_APPROACHES)].copy()
                if plot_df[x_param].nunique(dropna=True) > 1:
                    for y_metric in METRICS:
                        if y_metric not in plot_df.columns:
                            continue
                        _plot_xy_lines(
                            sdf=plot_df, x_param=x_param, y_metric=y_metric,
                            out_dir=os.path.join(outdir, "absolute"),
                            fname_prefix=fname_prefix,
                            title_prefix=title_prefix,
                            relative=False,
                            formats=formats,
                            show=show,
                            baseline_label=None
                        )

            # Relative: per-baseline
            if make_relative:
                for baseline in baselines:
                    rel_all = compute_relative_to_baseline(sdf, baseline_approach=baseline, x_param=x_param, strict=True)
                    if rel_all.empty or rel_all[x_param].nunique(dropna=True) <= 1:
                        continue
                    rel_df = rel_all[rel_all["approach"].astype(str).isin(PLOT_APPROACHES)].copy()
                    if rel_df.empty:
                        continue
                    _plot_xy_lines(
                        sdf=rel_df,
                        x_param=x_param,
                        y_metric="logloss",
                        out_dir=os.path.join(outdir, f"relative_{baseline_tags[baseline]}"),
                        fname_prefix=fname_prefix,
                        title_prefix=title_prefix,
                        relative=True,
                        formats=formats,
                        show=show,
                        baseline_label=baseline
                    )
                    _plot_xy_lines(
                        sdf=rel_df,
                        x_param=x_param,
                        y_metric="time_sec",
                        out_dir=os.path.join(outdir, f"relative_{baseline_tags[baseline]}"),
                        fname_prefix=fname_prefix,
                        title_prefix=title_prefix,
                        relative=True,
                        formats=formats,
                        show=show,
                        baseline_label=baseline
                    )
                    _plot_xy_lines(
                        sdf=rel_df,
                        x_param=x_param,
                        y_metric="roc_auc",
                        out_dir=os.path.join(outdir, f"relative_{baseline_tags[baseline]}"),
                        fname_prefix=fname_prefix,
                        title_prefix=title_prefix,
                        relative=True,
                        formats=formats,
                        show=show,
                        baseline_label=baseline
                    )
                    _plot_xy_lines(
                        sdf=rel_df,
                        x_param=x_param,
                        y_metric="pr_auc",
                        out_dir=os.path.join(outdir, f"relative_{baseline_tags[baseline]}"),
                        fname_prefix=fname_prefix,
                        title_prefix=title_prefix,
                        relative=True,
                        formats=formats,
                        show=show,
                        baseline_label=baseline
                    )

# -----------------------------
# Run directly (per-solver folders)
# -----------------------------
if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    df_all = load_results(CSV_PATH)

    # Split per solver into separate folders
    if "solver" in df_all.columns:
        solvers = [s for s in df_all["solver"].dropna().unique().tolist()]
        if not solvers:
            solvers = [None]
    else:
        solvers = [None]

    for solver in solvers:
        if solver is None:
            df_solver = df_all.copy()
            outdir_solver = os.path.join(OUTDIR, "solver=unspecified")
        else:
            df_solver = df_all[df_all["solver"] == solver].copy()
            if df_solver.empty:
                continue
            outdir_solver = os.path.join(OUTDIR, f"solver={str(solver)}")

        os.makedirs(outdir_solver, exist_ok=True)
        make_plots_for_df(
            df=df_solver,
            outdir=outdir_solver,
            make_absolute=MAKE_ABSOLUTE,
            make_relative=MAKE_RELATIVE,
            baselines=BASELINES,
            formats=FORMATS,
            show=SHOW,
        )
        print(f"✅ Figures saved under {outdir_solver}")


