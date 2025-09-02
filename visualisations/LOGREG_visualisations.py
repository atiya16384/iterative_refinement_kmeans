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
APP_HYBRID = "hybrid(f32→f64)"

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

def _plot_hybrid_ratio_both_baselines(df_ds, ds_name, param, metric_mean_col):
    need = {param, "approach", metric_mean_col}
    # special case: if param not present (e.g. tol_double for single/double), skip
    if param not in df_ds.columns:
        print(f"[viz] {ds_name}: skip param={param} (not in columns)")
        return None
    if metric_mean_col not in df_ds.columns:
        print(f"[viz] {ds_name}: skip metric={metric_mean_col} (not in columns)")
        return None

    # Keys to pair on (everything except the x-axis param)
    pair_keys_all = (
        ["dataset"]
        + [c for c in PARAM_CANDIDATES if c in df_ds.columns]
        + [c for c in PAIR_ALWAYS if c in df_ds.columns]
    )
    pair_keys_all = [k for k in dict.fromkeys(pair_keys_all) if k != param]

    d = df_ds[pair_keys_all + [param, metric_mean_col, "approach"]].dropna(subset=[metric_mean_col]).copy()
    if d.empty:
        print(f"[viz] {ds_name}: no rows after filter for {param}/{metric_mean_col}")
        return None

    hy = d[d["approach"] == APP_HYBRID]
    if hy.empty:
        print(f"[viz] {ds_name}: no HYBRID rows")
        return None

    def _pair(baseline_label):
        bl = d[d["approach"] == baseline_label]
        if bl.empty:
            return None
        m = hy.merge(bl, on=pair_keys_all, suffixes=("_hy", "_bl"), how="inner")
        if m.empty:
            return None

        # x-axis column could have been suffixed during merge if it was in the pair keys (it isn't here),
        # but be defensive anyway:
        xcol = f"{param}_hy" if f"{param}_hy" in m.columns else param

        num = m[f"{metric_mean_col}_hy"].astype(float)
        den = m[f"{metric_mean_col}_bl"].astype(float).replace(0.0, np.nan)
        m["ratio"] = (num / den).replace([np.inf, -np.inf], np.nan)
        m = m.dropna(subset=["ratio"])
        if m.empty:
            return None
        return m.groupby(xcol)["ratio"].mean()

    r_single = _pair(APP_SINGLE)
    r_double = _pair(APP_DOUBLE)

    if r_single is None and r_double is None:
        print(f"[viz] {ds_name}: cannot pair hybrid with single/double on {param}")
        return None

    # x values
    x_all = set()
    if r_single is not None: x_all |= set(r_single.index.tolist())
    if r_double is not None: x_all |= set(r_double.index.tolist())
    xs = _order_vals(list(x_all))

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.axhline(1.0, ls=":", lw=1.5, c="grey", label="baseline parity (×1.0)")

    def _line(s, label, marker):
        if s is None:
            return
        s = s.reindex(xs)
        ax.plot([str(v) for v in s.index], s.values, marker=marker, label=label)

    yname = re.sub("_mean$", "", metric_mean_col)
    _line(r_single, f"{APP_HYBRID} / {APP_SINGLE}", "o")
    _line(r_double, f"{APP_HYBRID} / {APP_DOUBLE}", "s")
    ax.set_xlabel(param)
    ax.set_ylabel(f"{yname} (ratio to baseline)")
    ax.set_title(f"{ds_name}: {yname} vs {param} — hybrid vs single & double")
    ax.grid(True, alpha=0.25)
    ax.legend()

    out = OUTDIR / f"{ds_name}__{yname}__by_{param}__hybrid_over_single_and_double.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("[viz] wrote:", out)
    return out

def _make_all_ratio_plots_for_dataset(ds_df, ds_name):
    paths = []
    metric_mean_cols = [f"{m}_mean" for m in METRICS if f"{m}_mean" in ds_df.columns]
    if not metric_mean_cols:
        print(f"[viz] {ds_name}: no metric means present; cols={list(ds_df.columns)}")
        return paths

    varying = [p for p in PARAM_CANDIDATES if p in ds_df.columns and ds_df[p].nunique() > 1]
    print(f"[viz] {ds_name}: varying params -> {varying}")

    for param in varying:
        for mcol in metric_mean_cols:
            out = _plot_hybrid_ratio_both_baselines(ds_df, ds_name, param, mcol)
            if out is not None:
                paths.append(out)
    return paths

def main():
    _ensure_outdir()
    frames = _read_existing(IN_FILES)
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)

    if "approach" not in df.columns:
        print("[viz] No 'approach' column in CSVs; columns:", list(df.columns))
        return

    # normalize labels + (optionally) keep only chosen datasets
    df["approach"] = df["approach"].map(_norm_approach)
    df = _filter_datasets(df)

    print("[viz] approaches:", df["approach"].value_counts().to_dict())

    agg = _agg_mean_std(df)
    if agg.empty:
        print("[viz] Aggregation empty. Check column names & METRICS.")
        print("[viz] columns seen:", list(df.columns))
        return

    for ds, ds_df in agg.groupby("dataset"):
        print(f"[viz] dataset: {ds}, rows: {len(ds_df)}")
        figs = _make_all_ratio_plots_for_dataset(ds_df.copy(), ds)
        if not figs:
            print(f"[viz] {ds}: no figures produced.")
        else:
            md = OUTDIR / f"{ds}__summary.md"
            with open(md, "w", encoding="utf-8") as f:
                f.write(f"# {ds} – Hybrid speed/quality ratios\n\n")
                for fp in figs:
                    f.write(f"![{fp.name}]({fp.as_posix()})\n\n")
            print("[viz] wrote:", md)

if __name__ == "__main__":
    main()


