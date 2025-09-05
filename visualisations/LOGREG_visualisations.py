# LOGREG_visualisations.py — linear/MSE friendly, hybrid vs single/double
from pathlib import Path
import re
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # save PNGs without a display
import matplotlib.pyplot as plt

# --- INPUT CSVs ---
IN_FILES = [
    "../Results/uniform_results.csv",
    "../Results/gaussian_results.csv",
    "../Results/blobs_results.csv",
    "../Results/susy_results.csv",
    "../Results/3droad_results.csv",
]

# set to e.g. ["gaussian"] if you only want to plot that dataset
DATASET_WHITELIST = None  # or ["gaussian"]

OUTDIR = Path("../Results/SUMMARY_LOGPREC")

# Metrics expected in CSVs
METRICS = ["time_sec", "roc_auc", "pr_auc", "logloss"]

# Parameters we may use on x-axis / for pairing.
# NOTE: use tol_single/tol_double (your CSVs) instead of tol.
PARAM_CANDIDATES = ["lambda", "C", "max_iter", "tol_single", "tol_double", "max_iter_single"]
# Always pair on these when possible (prevents mismatching solvers/penalties)
PAIR_ALWAYS = ["solver", "penalty", "alpha"]

APP_SINGLE = "single(f32)"
APP_DOUBLE = "double(f64)"
APP_HYBRID = "hybrid(f32→f64,budgeted)"

def _norm_approach(s):
    if not isinstance(s, str):
        return s
    t = s.strip().replace("->", "→")
    tl = t.lower()
    if tl == "single(f32)": return APP_SINGLE
    if tl == "double(f64)": return APP_DOUBLE
    if tl == "hybrid(f32→f64)": return APP_HYBRID
    return t

def _order_vals(vals):
    try:
        xs = [float(v) for v in vals]
        return [v for _, v in sorted(zip(xs, vals))]
    except Exception:
        key = {"coord": 0, "sparse_cg": 1, "lbfgs": 2}
        return sorted(vals, key=lambda v: key.get(str(v), 9999))

def _ensure_outdir():
    OUTDIR.mkdir(parents=True, exist_ok=True)

def _read_existing(files):
    frames = []
    print("[viz] CWD:", Path().resolve())
    for f in files:
        p = Path(f)
        if p.exists() and p.stat().st_size > 0:
            print(f"[viz] reading {p} ({p.stat().st_size} bytes)")
            df = pd.read_csv(p)
            if not df.empty:
                frames.append(df)
        else:
            print(f"[viz] missing/empty: {p}")
    if not frames:
        print("[viz] No non-empty results CSVs found.")
    return frames

def _filter_datasets(df: pd.DataFrame) -> pd.DataFrame:
    if DATASET_WHITELIST:
        keep = df["dataset"].isin(DATASET_WHITELIST)
        print(f"[viz] filtering datasets -> {DATASET_WHITELIST}, kept {int(keep.sum())}/{len(df)} rows")
        return df.loc[keep].copy()
    return df

def _agg_mean_std(df: pd.DataFrame) -> pd.DataFrame:
    metrics_present = [m for m in METRICS if m in df.columns]
    if not metrics_present:
        print("[viz] None of the expected metrics in CSVs:", METRICS)
        return pd.DataFrame()

    if "approach" in df.columns:
        df["approach"] = df["approach"].map(_norm_approach)

    group_cols = (
        ["dataset", "approach"]
        + [c for c in PARAM_CANDIDATES if c in df.columns]
        + [c for c in PAIR_ALWAYS if c in df.columns]
    )
    # drop columns that don't exist to avoid KeyErrors
    group_cols = [c for c in group_cols if c in df.columns]
    if not group_cols:
        print("[viz] No grouping columns present. Columns:", list(df.columns))
        return pd.DataFrame()

    agg = df.groupby(group_cols, as_index=False).agg({m: ["mean", "std"] for m in metrics_present})

    # flatten multi-index columns
    flat = []
    for col in agg.columns:
        if isinstance(col, tuple):
            a, b = col
            flat.append(f"{a}_{b}" if b else str(a))
        else:
            flat.append(str(col))
    agg.columns = flat
    return agg

# --- add: small helper to pretty tag names ---
def _mk_tag(penalty, solver):
    pen = str(penalty).lower() if penalty is not None else "none"
    sol = str(solver).lower() if solver is not None else "unspecified"
    return f"penalty={pen}__solver={sol}"

# --- change signature: add tag to title/filename ---
def _plot_hybrid_ratio_both_baselines(df_ds, ds_name, param, metric_mean_col, tag=""):
    ...
    yname = re.sub("_mean$", "", metric_mean_col)
    title_prefix = f"{ds_name}"
    if tag:
        title_prefix = f"{title_prefix} ({tag})"
    _line(r_single, f"{APP_HYBRID} / {APP_SINGLE}", "o")
    _line(r_double, f"{APP_HYBRID} / {APP_DOUBLE}", "s")
    ax.set_xlabel(param)
    ax.set_ylabel(f"{yname} (ratio to baseline)")
    ax.set_title(f"{title_prefix}: {yname} vs {param} — hybrid vs single & double")
    ax.grid(True, alpha=0.25)
    ax.legend()

    safe_tag = f"__{tag}" if tag else ""
    out = OUTDIR / f"{ds_name}{safe_tag}__{yname}__by_{param}__hybrid_over_single_and_double.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("[viz] wrote:", out)
    return out

# --- change: make_all_ratio_plots_for_dataset carries a tag through ---
def _make_all_ratio_plots_for_dataset(ds_df, ds_name, tag=""):
    paths = []
    metric_mean_cols = [f"{m}_mean" for m in METRICS if f"{m}_mean" in ds_df.columns]
    if not metric_mean_cols:
        print(f"[viz] {ds_name} {tag}: no metric means present; cols={list(ds_df.columns)}")
        return paths

    varying = [p for p in PARAM_CANDIDATES if p in ds_df.columns and ds_df[p].nunique() > 1]
    print(f"[viz] {ds_name} {tag}: varying params -> {varying}")
    for param in varying:
        for mcol in metric_mean_cols:
            out = _plot_hybrid_ratio_both_baselines(ds_df, ds_name, param, mcol, tag=tag)
            if out is not None:
                paths.append(out)
    return paths

# --- change main(): split by dataset, penalty, solver ---
def main():
    _ensure_outdir()
    frames = _read_existing(IN_FILES)
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)

    if "approach" not in df.columns:
        print("[viz] No 'approach' column in CSVs; columns:", list(df.columns))
        return

    df["approach"] = df["approach"].map(_norm_approach)
    df = _filter_datasets(df)
    print("[viz] approaches:", df["approach"].value_counts().to_dict())

    agg = _agg_mean_std(df)
    if agg.empty:
        print("[viz] Aggregation empty. Check column names & METRICS.")
        print("[viz] columns seen:", list(df.columns))
        return

    # ensure columns exist even if missing in some CSVs
    for c in ["penalty", "solver"]:
        if c not in agg.columns:
            agg[c] = "unspecified"

    # group by (dataset, penalty, solver) so each figure set is clean
    for (ds, pen, sol), ds_df in agg.groupby(["dataset", "penalty", "solver"], dropna=False):
        tag = _mk_tag(pen, sol)
        print(f"[viz] dataset: {ds}, penalty: {pen}, solver: {sol}, rows: {len(ds_df)}")
        figs = _make_all_ratio_plots_for_dataset(ds_df.copy(), ds, tag=tag)
        if not figs:
            print(f"[viz] {ds} ({tag}): no figures produced.")
        else:
            md = OUTDIR / f"{ds}__{tag}__summary.md"
            with open(md, "w", encoding="utf-8") as f:
                f.write(f"# {ds} – {tag} – Hybrid speed/quality ratios\n\n")
                for fp in figs:
                    f.write(f"![{fp.name}]({fp.as_posix()})\n\n")
            print("[viz] wrote:", md)


if __name__ == "__main__":
    main()


