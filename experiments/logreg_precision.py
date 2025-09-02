import pathlib
import time
import itertools
import numpy as np
import pandas as pd
from aoclda.linear_model import linmod
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

# =========================
# Paths
# =========================
DATA_DIR = pathlib.Path("../datasets")

# =========================
# Compatibility / mapping helpers
# =========================
def valid_combo_mse(solver, penalty, reg_alpha, reg_lambda) -> bool:
    s = (solver or "").lower()
    p = (penalty or "").lower()
    if s == "coord":
        # L1/L2/elastic-net allowed; alpha ∈ [0,1]
        return p in ("l1", "l2", "elasticnet") and 0.0 <= float(reg_alpha) <= 1.0
    if s == "sparse_cg":
        # ridge only, lambda > 0
        return p == "l2" and float(reg_alpha) == 0.0 and float(reg_lambda) > 0.0
    return False

def _map_penalty_to_alpha(penalty, alpha=None):
    if penalty is None:
        return 0.0
    p = penalty.lower()
    if p == "l2":
        return 0.0
    if p == "l1":
        return 1.0
    if p == "elasticnet":
        return 0.5 if alpha is None else float(alpha)
    raise ValueError(f"Unknown penalty: {penalty}")

def _map_C_to_lambda(C=None, reg_lambda=None):
    # AOCL-DA takes reg_lambda; allow user to provide either lambda or C (=1/lambda)
    if reg_lambda is not None:
        return float(reg_lambda)
    if C is None:
        return 0.0
    C = float(C)
    if C <= 0:
        raise ValueError("C must be > 0")
    return 1.0 / C

def _now():
    return time.perf_counter()

# =========================
# Synthetic datasets
# =========================
def make_shifted_gaussian(m=2000, n=100, delta=0.5, pos_frac=0.5, seed=0, dtype=np.float64):
    rng = np.random.RandomState(seed)
    m_pos = int(m * pos_frac)
    m_neg = m - m_pos
    X_pos = rng.normal(loc=+delta, scale=1.0, size=(m_pos, n))
    X_neg = rng.normal(loc=-delta, scale=1.0, size=(m_neg, n))
    X = np.vstack([X_pos, X_neg]).astype(dtype, copy=False)
    y = np.hstack([np.ones(m_pos, dtype=np.int32), np.zeros(m_neg, dtype=np.int32)])
    perm = rng.permutation(m)
    return X[perm], y[perm]

def make_uniform_binary(m=2000, n=100, shift=0.25, seed=0, dtype=np.float64):
    rng = np.random.RandomState(seed)
    m_pos = m // 2
    m_neg = m - m_pos
    X_pos = rng.uniform(low=0.5+shift, high=1.0+shift, size=(m_pos, n))
    X_neg = rng.uniform(low=0.0-shift, high=0.5-shift, size=(m_neg, n))
    X = np.vstack([X_pos, X_neg]).astype(dtype, copy=False)
    y = np.hstack([np.ones(m_pos, dtype=np.int32), np.zeros(m_neg, dtype=np.int32)])
    perm = rng.permutation(m)
    return X[perm], y[perm]

def make_blobs_binary(n_samples=5000, n_features=30, n_clusters=2,
                      cluster_std=1.0, random_state=0, dtype=np.float64):
    X, y = make_blobs(
        n_samples=n_samples, n_features=n_features,
        centers=n_clusters, cluster_std=cluster_std, random_state=random_state
    )
    return X.astype(dtype, copy=False), y.astype(np.int32, copy=False)

def load_3d_road(n_rows=1_000_000, dtype=np.float64):
    path = DATA_DIR / "3D_spatial_network.csv"
    X = pd.read_csv(
        path, sep=r"\s+|,", engine="python",
        header=None, usecols=[1, 2, 3], nrows=n_rows, dtype=dtype
    ).to_numpy()
    return X, None

def load_susy(n_rows=1_000_000, dtype=np.float64):
    path = DATA_DIR / "SUSY.csv"
    df = pd.read_csv(path, header=None, nrows=n_rows, dtype=dtype,
                     names=[f"c{i}" for i in range(19)])  # 1 label + 18 features
    y = df.iloc[:, 0].to_numpy().astype(np.int32, copy=False)
    X = df.iloc[:, 1:].to_numpy()
    return X, y

# =========================
# Core model helpers (AOCL-DA linmod with mod="mse")
# =========================
def train_linmod(X, y, *, precision="single", reg_lambda=0.0, reg_alpha=0.0,
                 solver="coord", max_iter=1000, tol=1e-4, scaling="standardize"):
    dt = np.float32 if str(precision).startswith("single") else np.float64
    Xd = X.astype(dt, copy=False)
    yd = y.astype(np.int32, copy=False)
    mdl = linmod(
        mod="mse", solver=solver, precision=precision,
        intercept=True, max_iter=max_iter, scaling=scaling
    )
    mdl.fit(Xd, yd, reg_lambda=float(reg_lambda), reg_alpha=float(reg_alpha), tol=float(tol))
    return mdl

def _margin(model, X):
    X_aug = np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])
    return X_aug @ model.coef.astype(X.dtype)

def evaluate_linear_mse(model, X, y):
    z = _margin(model, X)
    roc = float(roc_auc_score(y, z))
    pr  = float(average_precision_score(y, z))
    p = 1.0 / (1.0 + np.exp(-z))
    p = np.clip(p, 1e-12, 1-1e-12)
    return {
        "roc_auc": roc,
        "pr_auc":  pr,
        "logloss": float(log_loss(y, p)),
        "loss_internal": float(model.loss[0]),
    }

# =========================
# Approaches
# =========================
def approach_single(Xtr, ytr, Xte, yte, *, solver="coord", reg_lambda=0.01, reg_alpha=0.0,
                    max_iter=1000, tol=1e-4):
    t0 = _now()
    mdl = train_linmod(Xtr, ytr, precision="single", solver=solver,
                       reg_lambda=reg_lambda, reg_alpha=reg_alpha,
                       max_iter=max_iter, tol=tol)
    t1 = _now()
    metrics = evaluate_linear_mse(mdl, Xte, yte)
    return {"approach": "single(f32)", "time_sec": t1 - t0, "iters_single": int(mdl.n_iter), **metrics}

def approach_double(Xtr, ytr, Xte, yte, *, solver="coord", reg_lambda=0.01, reg_alpha=0.0,
                    max_iter=1000, tol=1e-4):
    t0 = _now()
    mdl = train_linmod(Xtr, ytr, precision="double", solver=solver,
                       reg_lambda=reg_lambda, reg_alpha=reg_alpha,
                       max_iter=max_iter, tol=tol)
    t1 = _now()
    metrics = evaluate_linear_mse(mdl, Xte, yte)
    return {"approach": "double(f64)", "time_sec": t1 - t0, "iters_single": 0, "iters_double": int(mdl.n_iter), **metrics}

def approach_hybrid_budgeted_fast(
    Xtr32, ytr, Xte32, yte, Xtr64, Xte64, *,
    solver="coord",
    reg_lambda=1e-4, reg_alpha=0.0,
    max_iter_single=800,     # main f32 work
    tol_single=1e-4,
    max_iter_double=None,    # if None -> small fixed budget
    tol_double=1e-6,
    double_budget_frac=0.10  # 10% of single iters, hard capped by floor
):
    t0 = _now()
    y_i32 = ytr.astype(np.int32, copy=False)

    # Stage A: single (fast warm start)
    mdl_f32 = linmod(mod="mse", solver=solver, precision="single",
                     intercept=True, max_iter=int(max_iter_single), scaling="standardize")
    mdl_f32.fit(Xtr32, y_i32,
                reg_lambda=float(reg_lambda),
                reg_alpha=float(reg_alpha),
                tol=float(tol_single))
    iters_single = int(mdl_f32.n_iter)

    # Stage B: tiny, fixed f64 polish
    if max_iter_double is None:
        max_iter_double = max(10, int(round(max_iter_single * double_budget_frac)))
        max_iter_double = min(max_iter_double, 30)  # hard cap keeps it cheap

    x0 = mdl_f32.coef.astype(np.float64, copy=False)
    mdl_f64 = linmod(mod="mse", solver=solver, precision="double",
                     intercept=True, max_iter=int(max_iter_double), scaling="standardize")
    mdl_f64.fit(Xtr64, y_i32,
                reg_lambda=float(reg_lambda),
                reg_alpha=float(reg_alpha),
                x0=x0,
                tol=float(tol_double))
    iters_double = int(mdl_f64.n_iter)

    t1 = _now()
    metrics = evaluate_linear_mse(mdl_f64, Xte64, yte)
    return {"approach": "hybrid(f32→f64,budgeted)",
            "time_sec": t1 - t0,
            "iters_single": iters_single,
            "iters_double": iters_double, **metrics}

# =========================
# Grid runner
# =========================
def run_experiments(X, y,
                    grid=None,
                    test_size=0.25, random_state=42, stratify=True,
                    save_path="../Results/results_all.csv",
                    dataset="",
                    repeats=3):

    if grid is None:
        grid = {
            "dataset": ["gaussian"],
            "penalty": ["l2"],
            "alpha":   [0.0],
            "lambda":  [1e-4, 1e-6],
            "C":       [None],
            "solver":  ["coord", "sparse_cg"],
            "max_iter": [800],                 # also used as f64 budget in hybrid
            "tol":      [1e-4, 1e-6, 1e-8],    # f32 tol (single & hybrid)
            "max_iter_single": [300, 500, 800],
            "tol_double": [1e-6],              # f64 tol for hybrid
            "double_budget_frac": [0.10],      # used only if max_iter_double not passed
            "approaches": ["single", "double", "hybrid"],
        }

    keys = [
        "penalty", "alpha", "lambda", "C",
        "solver", "max_iter", "tol", "max_iter_single",
        "tol_double", "double_budget_frac",
    ]
    combos = list(itertools.product(*[grid.get(k, [None]) for k in keys]))

    approaches = grid.get("approaches", [])
    total_tasks = repeats * len(combos) * len(approaches)
    done_tasks = 0
    t_start = _now()
    PRINT_EVERY = 25

    all_repeats = []

    for rep in range(repeats):
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_size, random_state=random_state + rep,
            stratify=y if stratify else None
        )

        rows = []

        for vals in combos:
            P = dict(zip(keys, vals))
            try:
                reg_alpha  = _map_penalty_to_alpha(P["penalty"], P["alpha"])
                reg_lambda = _map_C_to_lambda(C=P["C"], reg_lambda=P["lambda"])

                if not valid_combo_mse(P["solver"], P["penalty"], reg_alpha, reg_lambda):
                    continue

                for approach in approaches:
                    if approach == "single":
                        res = approach_single(
                            Xtr, ytr, Xte, yte,
                            solver=P["solver"],
                            reg_lambda=reg_lambda, reg_alpha=reg_alpha,
                            max_iter=P["max_iter"], tol=P["tol"]
                        )
                    elif approach == "double":
                        res = approach_double(
                            Xtr, ytr, Xte, yte,
                            solver=P["solver"],
                            reg_lambda=reg_lambda, reg_alpha=reg_alpha,
                            max_iter=P["max_iter"], tol=P["tol"]
                        )
                    elif approach == "hybrid":
                        res = approach_hybrid(
                            Xtr, ytr, Xte, yte,
                            solver=P["solver"],
                            reg_lambda=reg_lambda, reg_alpha=reg_alpha,
                            max_iter_single=P["max_iter_single"],    # f32 work
                            tol_single=P["tol"],                     # f32 tol
                            max_iter_double=P["max_iter"],           # f64 budget (small)
                            tol_double=P["tol_double"],
                            double_budget_frac=P["double_budget_frac"]
                        )
                    else:
                        continue

                    rows.append({
                        "dataset": dataset, "repeat": rep, "approach": res["approach"],
                        "penalty": P["penalty"], "alpha": reg_alpha,
                        "lambda": reg_lambda, "C": P["C"],
                        "solver": P["solver"],
                        "max_iter": P["max_iter"], "max_iter_single": P["max_iter_single"],
                        "tol_single": P["tol"], "tol_double": P["tol_double"],
                        "double_budget_frac": P["double_budget_frac"],
                        "time_sec": res["time_sec"],
                        "iters_single": res.get("iters_single", np.nan),
                        "iters_double": res.get("iters_double", np.nan),
                        "roc_auc": res.get("roc_auc", np.nan),
                        "pr_auc": res.get("pr_auc", np.nan),
                        "logloss": res.get("logloss", np.nan),
                    })

                done_tasks += 1
                if done_tasks % PRINT_EVERY == 0 or done_tasks == total_tasks:
                    elapsed = _now() - t_start
                    rate = done_tasks / max(elapsed, 1e-9)
                    remaining = total_tasks - done_tasks
                    print(f"[{done_tasks}/{total_tasks}] elapsed={elapsed:.1f}s, "
                          f"avg={1.0/rate:.3f}s/task, ETA~{remaining/max(rate,1e-9):.1f}s")

            except Exception as e:
                rows.append({"dataset": dataset, "repeat": rep, "approach": "ERROR", **P, "error": str(e)})

        all_repeats.extend(rows)

    df = pd.DataFrame(all_repeats)

    if save_path is not None:
        df.to_csv(save_path, index=False)
        print(f"\nSaved results to {save_path}")

    if df.empty or "approach" not in df.columns:
        print("No valid runs were produced. Check grid/compatibility.")
        return df, pd.DataFrame()

    df = df[df["approach"] != "ERROR"].copy()
    if df.empty:
        print("All runs errored out after filtering.")
        return df, pd.DataFrame()

    # Summary
    group_cols = [
        "dataset", "penalty", "alpha", "lambda", "solver",
        "max_iter", "tol_single", "max_iter_single", "tol_double",
        "double_budget_frac", "approach"
    ]
    metric_cols = ["time_sec", "iters_single", "iters_double", "roc_auc", "pr_auc", "logloss"]
    df_mean = df.groupby(group_cols, as_index=False)[metric_cols].mean()

    print(
        df_mean.groupby([
            "dataset", "penalty", "alpha", "C", "lambda", "solver",
            "max_iter", "tol_single", "max_iter_single", "tol_double",
            "double_budget_frac", "approach"
        ]) [["time_sec", "iters_single", "iters_double", "roc_auc", "pr_auc", "logloss"]].mean()
    )

    return df, df_mean

# =========================
# Main
# =========================
if __name__ == "__main__":
    datasets = ["gaussian"]  # adjust as needed

    base_grid = {
        "penalty": ["l1", "l2"],
        "alpha":   [0.0, 0.25, 0.75, 1.0],
        "lambda":  [None, 1e-4, 1e-6, 1e-8],
        "C":       [None, 0.01, 0.1, 1.0, 10, 100],
        "solver":  ["coord", "sparse_cg"],
        "max_iter": [800],                   # f64 budget in hybrid
        "tol":      [1e-4, 1e-6, 1e-8],      # f32 tol
        "max_iter_single": [300, 500, 800],  # f32 work
        "tol_double": [1e-6],                # f64 tol
        "double_budget_frac": [0.10],        # used only if you don't pass max_iter_double
        "approaches": ["single", "double", "hybrid"],
    }

    all_df, all_df_mean = [], []
    for dataset in datasets:
        if dataset == "gaussian":
            X, y = make_shifted_gaussian(m=100_000, n=120, delta=0.5, seed=42)
        elif dataset == "uniform":
            X, y = make_uniform_binary(m=100_000, n=120, shift=0.25, seed=42)
        elif dataset == "blobs":
            X, y = make_blobs_binary(n_samples=100_000, n_features=30, cluster_std=1.2, random_state=42)
        elif dataset == "susy":
            X, y = load_susy(n_rows=100_000)
        else:
            raise ValueError("Unknown dataset")

        grid = dict(base_grid)
        grid["dataset"] = [dataset]

        df, df_mean = run_experiments(
            X, y, grid=grid,
            dataset=dataset,
            save_path=f"../Results/{dataset}_results.csv",
            repeats=1
        )
        all_df.append(df)
        all_df_mean.append(df_mean)

