import pathlib
import time
import itertools
import numpy as np
import pandas as pd
from aoclda.linear_model import linmod
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

DATA_DIR = pathlib.Path("../datasets")


# ---- replace the linear guard with a logistic one ----
def valid_combo_mse(solver, penalty, reg_alpha, reg_lambda) -> bool:
    s = (solver or "").lower()
    p = (penalty or "").lower()

    if s == "coord":
        # L1/L2/elastic-net allowed; alpha must be in [0,1]
        return p in ("l1", "l2", "elasticnet") and 0.0 <= float(reg_alpha) <= 1.0
    if s == "sparse_cg":
        # L2 only, and needs lambda > 0
        return p == "l2" and float(reg_alpha) == 0.0 and float(reg_lambda) > 0.0
    return False



def _now():
    import time
    return time.perf_counter()

# Helpers
#update the code so we know the number of single and double iterations printed out
def _map_penalty_to_alpha(penalty, alpha=None):
    """Translate a 'penalty' label to reg_alpha.
       If alpha is provided and penalty is 'elasticnet', we use alpha directly."""
    if penalty is None:
        # treat as unpenalized -> reg_alpha doesn't matter if reg_lambda == 0
        return 0.0
    p = penalty.lower()
    if p == "l2":
        return 0.0
    if p == "l1":
        return 1.0
    if p == "elasticnet":
        if alpha is None:
            return 0.5
        return float(alpha)
    raise ValueError(f"Unknown penalty: {penalty}")

def _map_C_to_lambda(C=None, reg_lambda=None):
    """AOCL-DA uses reg_lambda. If C is given (sklearn-style), use reg_lambda = 1/C."""
    if reg_lambda is not None:
        return float(reg_lambda)
    if C is None:
        return 0.0
    C = float(C)
    if C <= 0:
        raise ValueError("C must be > 0")
    return 1.0 / C

# Synthetic datasets
# m => number of rows, n=> columns
def make_shifted_gaussian(m=2000, n=100, delta=0.5, pos_frac=0.5, seed=0, dtype=np.float64):
    """
    Gaussian synthetic data:
      X | y=1 ~ N(+delta, I)
      X | y=0 ~ N(-delta, I)
    """
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

    # Uniform synthetic data:
    #  y=1 points ~ U(0.5+shift, 1.0+shift)
     # y=0 points ~ U(0.0-shift, 0.5-shift)

    rng = np.random.RandomState(seed)
    m_pos = m // 2
    m_neg = m - m_pos
    X_pos = rng.uniform(low=0.5+shift, high=1.0+shift, size=(m_pos, n))
    X_neg = rng.uniform(low=0.0-shift, high=0.5-shift, size=(m_neg, n))
    X = np.vstack([X_pos, X_neg]).astype(dtype, copy=False)
    y = np.hstack([np.ones(m_pos, dtype=np.int32), np.zeros(m_neg, dtype=np.int32)])
    perm = rng.permutation(m)
    return X[perm], y[perm]

def generate_synthetic_data(n_samples, n_features, n_clusters, random_state):
    X, y_true = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=random_state)
    
    return X.astype(np.float64), y_true

def make_blobs_binary(n_samples=5000, n_features=30, n_clusters=2, cluster_std=1.0,
                      random_state=0, dtype=np.float64):
    """'Normal' synthetic dataset via sklearn.make_blobs (binary)."""
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=random_state
    )
    return X.astype(dtype, copy=False), y.astype(np.int32, copy=False)

def load_3d_road(n_rows=1_000_000, dtype=np.float64):
    """3D Spatial Network (features only). Returns (X, None).
       Columns used: 1,2,3 (as in your screenshot)."""
    path = DATA_DIR / "3D_spatial_network.csv"
    X = pd.read_csv(
        path, sep=r"\s+|,", engine="python",
        header=None, usecols=[1, 2, 3], nrows=n_rows, dtype=dtype
    ).to_numpy()
    return X, None

def load_susy(n_rows=1_000_000, dtype=np.float64):
    """SUSY dataset. Assumes CSV with label in col 0, features in cols 1.."""
    path = DATA_DIR / "SUSY.csv"
    df = pd.read_csv(path, header=None, nrows=n_rows, dtype=dtype,
                     names=[f"c{i}" for i in range(19)])  # 1 label + 18 features
    y = df.iloc[:, 0].to_numpy().astype(np.int32, copy=False)
    X = df.iloc[:, 1:].to_numpy()
    return X, y

# utilities for warm-started chunked fitting
def _fit_chunk(X, y, *, precision, solver, reg_lambda, reg_alpha, max_iter, tol, x0=None):
    dt = np.float32 if str(precision).startswith("single") else np.float64
    X = X.astype(dt, copy=False)
    y = y.astype(np.int32, copy= False)
    x0= None if x0 is None else x0.astype(dt, copy=False)

    mdl = linmod(
        mod="mse",
        solver=solver,
        precision=precision,
        intercept=True,
        max_iter=max_iter,
        scaling="standardize",
    )
    mdl.fit(X, y, reg_lambda=float(reg_lambda), reg_alpha=float(reg_alpha), x0=x0, tol=float(tol))
    return mdl

def _proba_from_coef(model, X):
    X_aug = np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])
    z = X_aug @ model.coef.astype(X.dtype)
    return 1.0 / (1.0 + np.exp(-z))

def _score_model(model, X, y):
    p = _proba_from_coef(model, X)
    p = np.clip(p, 1e-12, 1-1e-12)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc":  float(average_precision_score(y, p)),
        "logloss": float(log_loss(y, p)),
        "loss_internal": float(model.loss[0]),
        "iters": model.n_iter,
        "grad_norm": float(model.nrm_gradient_loss[0]) if hasattr(model, "nrm_gradient_loss") else np.nan,
    }

# Train & evaluate_linear_mse_linear_mse (AOCL-DA)
def train_linmod(X, y, *, precision="single", reg_lambda=0.0, reg_alpha=0.0,
                 solver="lbfgs", max_iter=10000, tol=1e-4, scaling="standardize"):


    dt = np.float32 if str(precision).startswith("single") else np.float64
    X= X.astype(dt, copy=False)
    y= y.astype(np.int32, copy=False)
    # AOCL-DA logistic:
    # - Use scaling='standardize' so 'coord' is valid (variance=1).
    # - For ridge-like only, 'lbfgs' can be used; 'coord' works across penalties.

    mdl = linmod(
        mod="mse",
        solver=solver,
        precision=precision,
        intercept=True,
        max_iter=max_iter,
        scaling=scaling
    )
    mdl.fit(X, y, reg_lambda=float(reg_lambda), reg_alpha=float(reg_alpha), tol=float(tol))
    return mdl

def _margin(model, X):
    X_aug = np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])
    return X_aug @ model.coef.astype(X.dtype)

def evaluate_linear_mse(model, X, y):
    z = _margin(model, X)                 # real-valued marginf
    # AUC / PR take arbitrary scores:
    roc = float(roc_auc_score(y, z))
    pr  = float(average_precision_score(y, z))
    # For log loss, map to [0,1] with a sigmoid:
    p = 1.0 / (1.0 + np.exp(-z))
    p = np.clip(p, 1e-12, 1-1e-12)
    return {
        "roc_auc": roc,
        "pr_auc":  pr,
        "logloss": float(log_loss(y, p)),
        "loss_internal": float(model.loss[0]),
    }



# All approaches
def approach_single(Xtr, ytr, Xte, yte, *, solver="lbfgs", reg_lambda=0.01, reg_alpha=0.0,
                    max_iter=10000, tol=1e-4):
    t0 = time.perf_counter()
    mdl = train_linmod(Xtr, ytr, precision="single", solver=solver,
                       reg_lambda=reg_lambda, reg_alpha=reg_alpha,
                       max_iter=max_iter, tol=tol)
    t1 = time.perf_counter()
    metrics = evaluate_linear_mse(mdl, Xte, yte)
    return {"approach": "single(f32)", "time_sec": t1 - t0, "iters_single": mdl.n_iter, **metrics}

def approach_double(Xtr, ytr, Xte, yte, *, solver="lbfgs", reg_lambda=0.01, reg_alpha=0.0,
                    max_iter=10000, tol=1e-4):
    t0 = time.perf_counter()
    mdl = train_linmod(Xtr, ytr, precision="double", solver=solver,
                       reg_lambda=reg_lambda, reg_alpha=reg_alpha,
                       max_iter=max_iter, tol=tol)
    t1 = time.perf_counter()
    metrics = evaluate_linear_mse(mdl, Xte, yte)
    return {"approach": "double(f64)", "time_sec": t1 - t0, "iters_single" : 0, "iters_double": mdl.n_iter, **metrics}

# def approach_hybrid(Xtr, ytr, Xte, yte, *, solver="lbfgs", reg_lambda=0.01, reg_alpha=0.0,
#                     max_iter_single=200, max_iter_double=10000, tol=1e-4):
#     # Stage A: fast f32 warm start
#     t0 = time.perf_counter()
#     mdl_f32 = train_linmod(Xtr, ytr, precision="single", solver=solver,
#                            reg_lambda=reg_lambda, reg_alpha=reg_alpha,
#                            max_iter=max_iter_single, tol=tol)
#     x0 = mdl_f32.coef.astype(np.float64, copy=False)

#     # Stage B: refine in f64 from warm start
#     mdl_f64 = linmod(mod="mse", solver=solver, precision="double",
#                      intercept=True, max_iter=max_iter_double, scaling="standardize")

#     Xd = Xtr.astype(np.float64, copy=False)
#     yd = ytr.astype(np.int32, copy=False)
#     mdl_f64.fit(Xd, yd, reg_lambda=float(reg_lambda), reg_alpha=float(reg_alpha),  # <-- use Xd, yd
#                 x0=x0, tol=float(tol))
#     t1 = time.perf_counter()

#     metrics = evaluate_linear_mse(mdl_f64, Xte, yte)
#     return {"approach": "hybrid(f32→f64)", "time_sec": t1 - t0, "iters_single": mdl_f32.n_iter, "iters_double": mdl_f64.n_iter, **metrics}



def approach_hybrid(
    Xtr, ytr, Xte, yte, *,
    solver="coord",
    reg_lambda=1e-4, reg_alpha=0.0,
    # --- f32 phase (do most of the work here) ---
    max_iter_single=500,        # chunk size in f32 (re-uses your grid’s 'max_iter_single')
    tol_single=1e-4,            # f32 internal tolerance (re-uses your grid’s 'tol' by default)
    max_chunks_single=20,       # safety cap: total f32 iters ≤ max_iter_single * max_chunks_single
    grad_tol_switch=3e-4,       # promote to f64 if grad-norm ≤ this
    delta_tol_switch=1e-6,      # or if ||Δw||_2 ≤ this
    # --- f64 polish (brief) ---
    max_iter_double=800,        # small budget in f64 (re-uses your grid’s 'max_iter' by default)
    tol_double=1e-6,            # tighter tol for polish
):
    """
    Late-switch hybrid:
      - Loop f32 in chunks until we're 'close enough' (by grad norm or Δw),
      - Then do a short f64 polish from the f32 warm start.
    Returns iters spent in each precision so you can verify most work was f32.
    """
    t0 = time.perf_counter()

    iters_single_total = 0
    x_prev = None
    last_grad = np.inf

    # --- f32 stage in chunks ---
    for _ in range(int(max_chunks_single)):
        mdl32 = _fit_chunk(
            Xtr, ytr,
            precision="single", solver=solver,
            reg_lambda=reg_lambda, reg_alpha=reg_alpha,
            max_iter=int(max_iter_single), tol=float(tol_single), x0=x_prev
        )
        iters_single_total += int(mdl32.n_iter)
        x_curr = mdl32.coef.copy()

        # movement between chunks
        if x_prev is None:
            delta = np.inf
        else:
            delta = float(np.linalg.norm(x_curr - x_prev))

        # grad-norm if exposed by AOCL-DA
        last_grad = float(getattr(mdl32, "nrm_gradient_loss", [np.inf])[0])

        x_prev = x_curr

        # promotion condition
        if (last_grad <= float(grad_tol_switch)) or (delta <= float(delta_tol_switch)):
            break

    # --- short f64 polish ---
    mdl64 = _fit_chunk(
        Xtr, ytr,
        precision="double", solver=solver,
        reg_lambda=reg_lambda, reg_alpha=reg_alpha,
        max_iter=int(max_iter_double), tol=float(tol_double), x0=x_prev
    )

    t1 = time.perf_counter()
    metrics = evaluate_linear_mse(mdl64, Xte, yte)

    return {
        "approach": "hybrid(f32→f64, late-switch)",
        "time_sec": t1 - t0,
        "iters_single": iters_single_total,
        "iters_double": int(mdl64.n_iter),
        "grad32_last": float(last_grad),
        **metrics
    }



# Grid runner
def run_experiments(X, y,
                    grid=None,
                    test_size=0.25, random_state=42, stratify=True,
                    save_path="../Results/results_all.csv",
                    dataset="",
                    repeats=3):


    # for 'mse' and 'coord', we have alpha [0.0, 0.25, 0.5, 0.75, 1.0], and use "l1", "l2"
    # for 'mse' and 'sparse_cg', this only supports l2 ridge and alpha is none
    if grid is None:
        grid = {
            "dataset": ["uniform", "gaussian", "blobs", "susy", "3droad"],
            "penalty": ["l2"],
            "alpha": [0.0, 0.25, 0.5, 0.75, 1.0],
            "lambda": [1e-10, 1e-8, 1e-5, 1e-3, 1e-2],
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "solver": ["coord", "sparse_cg"],
            "max_iter": [3000, 10000],
            "tol": [1e-2, 1e-4, 1e-6, 1e-8],
            "max_iter_single": [150, 300, 500, 1000],
            "approaches": ["single", "double", "hybrid", "multistage-ir", "adaptive-precision"]
        }

    keys = ["penalty", "alpha", "lambda", "C", "solver", "max_iter", "tol", "max_iter_single"]
    combos = list(itertools.product(*[grid.get(k, [None]) for k in keys]))

    # count total tasks for progress display
    approaches = grid.get("approaches", [])
    total_tasks = repeats * len(combos) * len(approaches)
    done_tasks = 0
    t_start = _now()
    PRINT_EVERY = 25  # progress cadence

    all_repeats = []

    for rep in range(repeats):
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_size, random_state=random_state + rep,
            stratify=y if stratify else None
        )

        rows = []

        for vals in combos:
            penalty, alpha, lam, C, solver, max_iter, tol, max_iter_single = vals
            try:
                reg_alpha  = _map_penalty_to_alpha(penalty, alpha)
                reg_lambda = _map_C_to_lambda(C=C, reg_lambda=lam)

                #  linear/MSE compatibility guard
                if not valid_combo_mse(solver, penalty, reg_alpha, reg_lambda):
                    continue

                for approach in approaches:
                    if approach == "single":
                        res = approach_single(Xtr, ytr, Xte, yte,
                                              solver=solver, reg_lambda=reg_lambda, reg_alpha=reg_alpha,
                                              max_iter=max_iter, tol=tol)
                    elif approach == "double":
                        res = approach_double(Xtr, ytr, Xte, yte,
                                              solver=solver, reg_lambda=reg_lambda, reg_alpha=reg_alpha,
                                              max_iter=max_iter, tol=tol)
                    elif approach == "hybrid":
                        res = approach_hybrid(Xtr, ytr, Xte, yte,
                                              solver=solver, reg_lambda=reg_lambda, reg_alpha=reg_alpha,
                                              max_iter_single=max_iter_single, max_iter_double=max_iter, tol=tol)
                    else:
                        continue

                    row = {
                        "dataset": dataset,
                        "repeat": rep,
                        "approach": res["approach"],
                        "penalty": penalty,
                        "alpha": reg_alpha,
                        "lambda": reg_lambda,
                        "solver": solver,
                        "max_iter": max_iter,
                        "max_iter_single": max_iter_single,
                        "tol": tol,
                        "time_sec": res["time_sec"],
                        "iters_single": res.get("iters_single", np.nan),
                        "iters_double": res.get("iters_double", np.nan),
                        "roc_auc": res.get("roc_auc", np.nan),
                        "pr_auc": res.get("pr_auc", np.nan),
                        "logloss": res.get("logloss", np.nan),
                    }
                    rows.append(row)

                    # ---- progress & self-timing ----
                    done_tasks += 1
                    if done_tasks % PRINT_EVERY == 0 or done_tasks == total_tasks:
                        elapsed = _now() - t_start
                        rate = done_tasks / max(elapsed, 1e-9)
                        remaining = total_tasks - done_tasks
                        eta_sec = remaining / max(rate, 1e-9)
                        print(f"[{done_tasks}/{total_tasks}] "
                              f"elapsed={elapsed:.1f}s, "
                              f"avg={1.0/rate:.3f}s/task, "
                              f"ETA~{eta_sec:.1f}s")

            except Exception as e:
                rows.append({
                    "dataset": dataset,
                    "repeat": rep,
                    "approach": "ERROR",
                    "penalty": penalty, "alpha": alpha, "lambda": lam, "C": C,
                    "solver": solver, "max_iter": max_iter, "tol": tol,
                    "error": str(e)
                })

        all_repeats.extend(rows)

    df = pd.DataFrame(all_repeats)

       # Save raw results
    if save_path is not None:
        df.to_csv(save_path, index=False)
        print(f"\nSaved results to {save_path}")
    
    # If nothing made it through, exit gracefully
    if df.empty or "approach" not in df.columns:
        print("No valid runs were produced. Check grid/compatibility.")
        return df, pd.DataFrame()
    
    # Drop errors
    df = df[df["approach"] != "ERROR"].copy()
    if df.empty:
        print("All runs errored out after filtering.")
        return df, pd.DataFrame()

    # === Take mean over repeats ===
    group_cols = ["dataset", "penalty", "alpha", "lambda", "solver", "max_iter", "tol", "max_iter_single", "approach"]
    metric_cols = ["time_sec", "iters_single", "iters_double", "roc_auc", "pr_auc", "logloss"]

    df_mean = df.groupby(group_cols, as_index=False)[metric_cols].mean()

    print(df_mean.groupby(["dataset",  "penalty", "alpha", "C", "lambda", "solver", "max_iter", "tol", "max_iter_single", "approach"])
                          [["time_sec", "iters_single", "iters_double", "roc_auc", "pr_auc", "logloss"]].mean())

    return df, df_mean

if __name__ == "__main__":
    # pick multiple datasets instead of just one
    # blobs and susy gaussian, "uniform", "blobs"
    datasets = ["gaussian"]  # add/remove any you want

    # Linear/MSE with both solvers
    base_grid = {
        "penalty": ["l1", "l2"],                 # coord supports both; sparse_cg -> L2 only (guarded above)
        "alpha":   [0.0, 0.25, 0.75, 1.0],         # only used by coord (elastic-net family)
        "lambda":  [None, 1e-4, 1e-6, 1e-8],           # NOTE: sparse_cg requires > 0
        "C":       [None, 0.01, 0.1, 1.0, 10, 100],
        "solver":  ["coord", "sparse_cg"],       #  both in the same sweep
        "max_iter": [800],
        "tol":      [ 1e-4, 1e-6, 1e-8],
        "max_iter_single": [300, 500, 800],
        "approaches": ["single", "double" ,"hybrid"]
        
        # --- NEW knobs (optional to sweep) ---
        "max_chunks_single": [20],             # cap on number of f32 chunks
        "grad_tol_switch":  [3e-4],            # promote if grad-norm below this
        "delta_tol_switch": [1e-6],            # or if parameter change is tiny
        "tol_double":       [1e-6],            # polish tolerance
        "approaches": ["single", "double", "hybrid"],
            
        
    }

    all_df, all_df_mean = [], []

    for dataset in datasets:
        # load each dataset (100k rows)
        if dataset == "gaussian":
            X, y = make_shifted_gaussian(m=100_000, n=120, delta=0.5, seed=42)
        elif dataset == "uniform":
            X, y = make_uniform_binary(m=100_00, n=120, shift=0.25, seed=42)
        elif dataset == "blobs":
            X, y = make_blobs_binary(n_samples=100_00, n_features=30,
                                     cluster_std=1.2, random_state=42)
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
