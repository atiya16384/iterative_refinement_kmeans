import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score
import time
import pandas as pd
from sklearn.model_selection import train_test_split
import io, re, contextlib

# ----- helpers -----
def _svc(kernel="rbf", C=1.0, gamma="scale", tol=1e-3, max_iter=-1,
         shrinking=True, random_state=0, cache_size=500, verbose=False):
    # Pipeline to standardize features (important for SVMs)
    return make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        SVC(
            kernel=kernel, C=C, gamma=gamma,
            tol=tol, max_iter=max_iter,
            shrinking=shrinking,
            random_state=random_state,
            probability=False,  # use decision_function for metrics
            cache_size=cache_size,
            verbose=verbose,
        )
    )

def _eval_svc(model, X, y):
    # decision_function gives signed distance; good for ROC AUC
    s = model.decision_function(X)
    auc = float(roc_auc_score(y, s))
    acc = float(accuracy_score(y, (s > 0).astype(int)))
    return {"roc_auc": auc, "accuracy": acc}

def _parse_iters_from_libsvm_stdout(s:str):
    m = re.search(r"#iter\s*=\s*(\d+)", s)
                  
    return int(m.group(1)) if m else None


def svc_double(Xtr, ytr, Xte, yte, *,
               kernel="rbf", C=1.0, gamma="scale",
               tol=1e-5, max_iter=-1, random_state=0):
    cap = io.StringIO()
    t0 = time.perf_counter()
    clf = _svc(kernel, C, gamma, tol, max_iter, random_state=random_state, verbose=True)
    with contextlib.redirect_stdout(cap):
        clf.fit(Xtr, ytr)
    t1 = time.perf_counter()
    iters = _parse_iters_from_libsvm_stdout(cap.getvalue())
    m = _eval_svc(clf, Xte, yte)
    n_sv = int(np.sum(clf[-1].n_support_))
    return {"mode": "double(precise)", "time_sec": t1 - t0, "n_sv": n_sv, "iters_single": None, "iters_double": iters, **m, "model": clf}

# ----- hybrid (SV-refit) -----
def svc_hybrid(Xtr, ytr, Xte, yte, *,
               kernel="rbf", C=1.0, gamma="scale",
               tol_single=1e-3, tol_double=1e-5,
               max_iter_single=-1, max_iter_double=-1,
               buffer_frac=0.1, random_state=0):
    """
    Stage A: loose SVC on full data -> collect SVs.
    Stage B: tight SVC ONLY on those SVs (+ optional random buffer).
    """
    # Stage A
    capA = io.StringIO()
    t0 = time.perf_counter()
    loose = _svc(kernel, C, gamma, tol_single, max_iter_single, random_state=random_state, verbose = True)
    with contextlib.redirect_stdout(capA):    
        loose.fit(Xtr, ytr)

    itersA = _parse_iters_from_libsvm_stdout(capA.getvalue())
    tA = time.perf_counter()

    # collect SVs (indices are w.r.t. post-scaling inputs inside the pipe)
    sv_idx = loose[-1].support_
    X_sv, y_sv = Xtr[sv_idx], ytr[sv_idx]

    # optional small buffer from non-SVs to guard against borderline cases
    if buffer_frac > 0.0:
        n_buf = max(0, int(buffer_frac * (len(Xtr) - len(X_sv))))
        if n_buf > 0:
            # pick from the complement deterministically
            mask = np.ones(len(Xtr), dtype=bool)
            mask[sv_idx] = False
            non_sv_idx = np.flatnonzero(mask)
            rng = np.random.default_rng(random_state)
            extra = rng.choice(non_sv_idx, size=min(n_buf, len(non_sv_idx)), replace=False)
            X_sv = np.vstack([X_sv, Xtr[extra]])
            y_sv = np.hstack([y_sv, ytr[extra]])

    capB = io.StringIO()
    # Stage B
    tight = _svc(kernel, C, gamma, tol_double, max_iter_double, random_state=random_state, verbose= True)
    with contextlib.redirect_stdout(capB):
        tight.fit(X_sv, y_sv)
    itersB = _parse_iters_from_libsvm_stdout(capB.getvalue())
    tB = time.perf_counter()

    m = _eval_svc(tight, Xte, yte)
    n_sv_tight = int(np.sum(tight[-1].n_support_))

    return {
        "mode": "hybrid(SV-refit)",
        "time_sec": (tA - t0) + (tB - tA),
        "time_stageA": (tA - t0),
        "time_stageB": (tB - tA),
        "n_sv_stageA": int(np.sum(loose[-1].n_support_)),
        "n_used_stageB": int(len(X_sv)),
        "n_sv": n_sv_tight,
        **m,
        "itersA": itersA,
        "itersB": itersB,
        "model": tight
    }

# ----- adaptive-hybrid (multi-pass SV shrinking) -----
def svc_adaptive_hybrid(Xtr, ytr, Xte, yte, *,
                        kernel="rbf", C=1.0, gamma="scale",
                        tol_schedule=(1e-2, 5e-3, 1e-3, 5e-4, 1e-4),
                        final_tol=1e-5,
                        max_iter=-1,
                        min_rel_drop=0.05,   # promote if SV count drops by >=5%
                        random_state=0):
    """
    Iteratively fit with decreasing tol on the *current* subset;
    each pass keeps only SVs (+ tiny buffer). Stop when SV set stabilizes,
    then do one final tight fit (final_tol). Designed to be faster than one-shot 'double'.
    """
    t0 = time.perf_counter()
    idx = np.arange(len(Xtr))  # current working set
    history = []

    for k, tol in enumerate(tol_schedule, 1):
        cap = io.StringIO()
        clf = _svc(kernel, C, gamma, tol, max_iter, random_state=random_state, verbose = True)
        with contextlib.redirect_stdout(cap):
            clf.fit(Xtr[idx], ytr[idx])
        iters = _parse_iters_from_libsvm_stdout(cap.getvalue())
        n_sv = int(np.sum(clf[-1].n_support_))
        history.append((tol, len(idx), n_sv, iters ))
        sv_local = clf[-1].support_
        idx_new = idx[sv_local]

        history.append((tol, len(idx), n_sv))

        # relative drop in working set size
        rel_drop = 1.0 - len(idx_new) / max(1, len(idx))
        idx = idx_new

        # early stop if little shrinkage
        if rel_drop < min_rel_drop:
            break

    # final precise fit on the shrunken set
    capF = io.StringIO()
    clf_final = _svc(kernel, C, gamma, final_tol, max_iter, random_state=random_state, verbose = True)
    with contextlib.redirect_stdout(capF):
        clf_final.fit(Xtr[idx], ytr[idx])

    itersF = _parse_iters_from_libsvm_stdout(capF.getvalue())    
    t1 = time.perf_counter()

    m = _eval_svc(clf_final, Xte, yte)
    n_sv_final = int(np.sum(clf_final[-1].n_support_))

    return {
        "mode": "adaptive-hybrid(SV-shrink)",
        "time_sec": t1 - t0,
        "n_passes": len(history) + 1,
        "working_set_final": int(len(idx)),
        "n_sv": n_sv_final,
        "history": history,  # list of (tol, |working_set|, n_sv_on_pass)
        **m,
        "iters_single": None,
        "iters_double": itersF,
        "model": clf_final
    }

# =========================
# Synthetic datasets (X: m×n, y: {0,1})
# =========================
import numpy as np

def make_shifted_gaussian(m=4000, n=20, delta=0.8, pos_frac=0.5, seed=0):
    """
    Linear-ish Gaussian blobs:
      X|y=1 ~ N(+delta, I),  X|y=0 ~ N(-delta, I)
    """
    rng = np.random.default_rng(seed)
    m1 = int(m * pos_frac)
    m0 = m - m1
    X1 = rng.normal(loc=+delta, scale=1.0, size=(m1, n))
    X0 = rng.normal(loc=-delta, scale=1.0, size=(m0, n))
    X = np.vstack([X1, X0]).astype(np.float64, copy=False)
    y = np.hstack([np.ones(m1, dtype=np.int32), np.zeros(m0, dtype=np.int32)])
    perm = rng.permutation(m)
    return X[perm], y[perm]

def make_circles_like(m=4000, noise=0.15, factor=0.45, seed=0):
    """
    Nonlinear RBF-friendly data (two noisy concentric circles, many dims = 2).
    """
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 2*np.pi, size=m//2)
    inner = np.stack([np.cos(t)*factor, np.sin(t)*factor], axis=1)
    outer = np.stack([np.cos(t), np.sin(t)], axis=1)
    X = np.vstack([outer, inner])
    y = np.hstack([np.zeros(len(outer), dtype=np.int32), np.ones(len(inner), dtype=np.int32)])
    X += rng.normal(scale=noise, size=X.shape)
    perm = rng.permutation(m)
    return X[perm], y[perm]

# =========================
# Experiment runner for SVC
# =========================

def run_svc_experiments(
    X, y, *, dataset_name="unknown",
    approaches=("single", "double", "hybrid", "adaptive"),
    grid=None, repeats=3, test_size=0.25, random_state=42, stratify=True,
):
    """
    Runs a grid of SVC experiments with repeats. Returns the full results DataFrame
    and a mean/std summary.

    approaches: any subset of {"single", "double", "hybrid", "adaptive"}
    grid: dict of lists. Keys supported:
      - kernel (e.g. ["rbf"])
      - C      (e.g. [0.5, 1, 2, 10])
      - gamma  (e.g. ["scale", 0.1, 0.05])
      - tol_single, tol_double
      - max_iter_single, max_iter_double
      - buffer_frac (for hybrid)
      - tol_schedule (for adaptive)
      - final_tol, min_rel_drop (for adaptive)
    """
    if grid is None:
        grid = {
            "kernel":        ["rbf"],
            "C":             [1.0, 4.0],
            "gamma":         ["scale"],
            "tol_single":    [1e-3],
            "tol_double":    [1e-5],
            "max_iter_single":[-1],
            "max_iter_double":[-1],
            "buffer_frac":   [0.05],                 # for hybrid
            "tol_schedule":  [(2e-2, 5e-3, 1e-3)],   # for adaptive
            "final_tol":     [1e-5],
            "min_rel_drop":  [0.05],
        }

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if stratify else None
    )

    from itertools import product
    keys = list(grid.keys())
    combos = list(product(*[grid[k] for k in keys]))

    rows = []
    for combo in combos:
        params = dict(zip(keys, combo))

        # repeat each setting
        for r in range(1, repeats+1):
            for mode in approaches:
                try:
                    if mode == "double":
                        out = svc_double(
                            Xtr, ytr, Xte, yte,
                            kernel=params["kernel"], C=params["C"], gamma=params["gamma"],
                            tol=params.get("tol_double", 1e-5),
                            max_iter=params.get("max_iter_double", -1),
                            random_state=random_state + r
                        )
                    elif mode == "hybrid":
                        out = svc_hybrid(
                            Xtr, ytr, Xte, yte,
                            kernel=params["kernel"], C=params["C"], gamma=params["gamma"],
                            tol_single=params.get("tol_single", 1e-3),
                            tol_double=params.get("tol_double", 1e-5),
                            max_iter_single=params.get("max_iter_single", -1),
                            max_iter_double=params.get("max_iter_double", -1),
                            buffer_frac=params.get("buffer_frac", 0.05),
                            random_state=random_state + r
                        )
                    elif mode == "adaptive":
                        out = svc_adaptive_hybrid(
                            Xtr, ytr, Xte, yte,
                            kernel=params["kernel"], C=params["C"], gamma=params["gamma"],
                            tol_schedule=params.get("tol_schedule", (2e-2, 5e-3, 1e-3)),
                            final_tol=params.get("final_tol", 1e-5),
                            max_iter=params.get("max_iter_double", -1),
                            min_rel_drop=params.get("min_rel_drop", 0.05),
                            random_state=random_state + r
                        )
                    else:
                        continue

                    # strip model object for the table
                    out_tbl = {k:v for k,v in out.items() if k != "model"}
                    out_tbl.update({
                        "dataset": dataset_name,
                        **{k: params[k] for k in keys}   # record hyperparams used
                    })
                    out_tbl["repeat"] = r
                    rows.append(out_tbl)

                except Exception as e:
                    rows.append({
                        "dataset": dataset_name, "mode": mode, "repeat": r,
                        **{k: params[k] for k in keys},
                        "error": str(e)
                    })

    df  = pd.DataFrame(rows)

         # --- Summarize results ---
         # Add a friendly alias and drop failed rows, if any
    # ---- make 'approach' alias and separate raw/success ----
    if "mode" in df.columns and "approach" not in df.columns:
        df["approach"] = df["mode"]
         
    df_raw = df.copy()
    if "error" in df.columns:
        df = df[df["error"].isna()].copy()
         
         # Guard: if nothing succeeded, show errors and exit cleanly
    if df.empty:
        print("\nNo successful runs. First few errors:")
        cols = [c for c in ["dataset","mode","repeat","error"] if c in df_raw.columns]
        print(df_raw[cols].head(10).to_string(index=False))
        df_raw.to_csv("svm_precision_raw.csv", index=False)
        print("\nSaved: svm_precision_raw.csv (contains only error rows)")
        # Return something sensible
        return df_raw, pd.DataFrame()
         
         # ---- tidy output for CSVs (KEEP 'approach') ----
    keep = [
        "dataset","repeat","approach","mode","kernel","C","gamma",
        "tol_single","tol_double","buffer_frac","tol_schedule","final_tol","min_rel_drop",
        "time_sec","n_sv","roc_auc","accuracy",
        "iters_single","iters_double",
        "time_stageA","time_stageB","n_sv_stageA","n_used_stageB",
        "n_passes","working_set_final"
        ]
    df_out = df[[c for c in keep if c in df.columns]].copy()
         
         # ---- summary (like your k-means example) ----
    group_key = "approach" if "approach" in df_out.columns else ("mode" if "mode" in df_out.columns else None)
    group_cols = ["dataset","kernel","C","gamma"] + ([group_key] if group_key else [])
    metric_cols = [c for c in ["time_sec","n_sv","roc_auc","accuracy", "iters_single", "iters_double"] if c in df_out.columns]
         
    summary = (
        df_out.groupby(group_cols, as_index=False)[metric_cols].mean())
         
    print("\n==== SUMMARY: SVM Precision Experiments ====")
    print(summary.to_string(index=False))
         
         # ---- save CSVs ----
   
    df_out.to_csv("svm_precision_runs.csv", index=False)       # tidy successful runs
    summary.to_csv("svm_precision_summary.csv", index=False)   # grouped mean table
    print("\nSaved: svm_precision_raw.csv, svm_precision_runs.csv, svm_precision_summary.csv")
         
    return df_out, summary


# 1) Nonlinear dataset (RBF-friendly)
X, y = make_circles_like(m=100_0000, noise=0.15, factor=0.45, seed=1)
grid = {
    # "linear", "poly", "sigmoid"

    "kernel": ["rbf", "linear", "poly", "sigmoid"],
    "C": [1.0, 4.0],
    "gamma": ["scale", "auto"],
    "tol_single": [1e-2, 1e-3],
    "tol_double": [1e-4],
    "max_iter_single":[100, 250, 500, 1000, 3000],
    "max_iter_double":[10000],
    "buffer_frac": [0.05],
    "tol_schedule": [(2e-2, 5e-3, 1e-3)],
    "final_tol": [1e-4],
    "min_rel_drop": [0.05],
}
df, df_mean = run_svc_experiments(X, y, dataset_name="circles", grid=grid,
                                  approaches=("single","double","hybrid","adaptive"),
                                  repeats=3)

# 2) Linear-ish dataset (hybrid may help less because 'double' is already quick)
X, y = make_shifted_gaussian(m=1000_000, n=120, delta=0.7, seed=2)
df, df_mean = run_svc_experiments(X, y, dataset_name="gauss", grid=grid,
                                  approaches=("single","double","hybrid","adaptive"),
                                  repeats=3)
