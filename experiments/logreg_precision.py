import time
import itertools
import numpy as np
import pandas as pd
from aoclda.linear_model import linmod
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.model_selection import StratifiedKFold

# -------------------------
# Helpers
# -------------------------
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

# -------------------------
# Synthetic datasets
# -------------------------


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
    """
    Uniform synthetic data:
      y=1 points ~ U(0.5+shift, 1.0+shift)
      y=0 points ~ U(0.0-shift, 0.5-shift)
    """
    rng = np.random.RandomState(seed)
    m_pos = m // 2
    m_neg = m - m_pos
    X_pos = rng.uniform(low=0.5+shift, high=1.0+shift, size=(m_pos, n))
    X_neg = rng.uniform(low=0.0-shift, high=0.5-shift, size=(m_neg, n))
    X = np.vstack([X_pos, X_neg]).astype(dtype, copy=False)
    y = np.hstack([np.ones(m_pos, dtype=np.int32), np.zeros(m_neg, dtype=np.int32)])
    perm = rng.permutation(m)
    return X[perm], y[perm]

# ==== utilities for warm-started chunked fitting ====
def _fit_chunk(X, y, *, precision, solver, reg_lambda, reg_alpha, max_iter, tol, x0=None):
    mdl = linmod(
        mod="logistic",
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
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc":  float(average_precision_score(y, p)),
        "logloss": float(log_loss(y, p, eps=1e-12)),
        "loss_internal": float(model.loss[0]),
        "iters": model.n_iter,
        "grad_norm": float(model.nrm_gradient_loss[0]) if hasattr(model, "nrm_gradient_loss") else np.nan,
    }

# -------------------------
# Train & evaluate (AOCL-DA)
# -------------------------
def train_linmod(X, y, *, precision="single", reg_lambda=0.0, reg_alpha=0.0,
                 solver="coord", max_iter=10000, tol=1e-4, scaling="standardize"):
    """
    AOCL-DA logistic:
    - Use scaling='standardize' so 'coord' is valid (variance=1).
    - For ridge-like only, 'lbfgs' can be used; 'coord' works across penalties.
    """
    mdl = linmod(
        mod="logistic",
        solver=solver,
        precision=precision,
        intercept=True,
        max_iter=max_iter,
        scaling=scaling
    )
    mdl.fit(X, y, reg_lambda=float(reg_lambda), reg_alpha=float(reg_alpha), tol=float(tol))
    return mdl

def evaluate(model, X, y):
    # Build logits via [X | 1] @ coef, then sigmoid to get P(y=1)
    X_aug = np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])
    z = X_aug @ model.coef.astype(X.dtype)
    p = 1.0 / (1.0 + np.exp(-z))

    metrics = {
        "roc_auc": float(roc_auc_score(y, p)),           # AUC (ROC)
        "pr_auc":  float(average_precision_score(y, p)), # AUC (PR)
        "logloss": float(log_loss(y, p, eps=1e-12)),
        "loss_internal": float(model.loss[0]),
    }
    return metrics

# -------------------------
# Your three approaches
# -------------------------
def approach_single(Xtr, ytr, Xte, yte, *, solver="coord", reg_lambda=0.01, reg_alpha=0.0,
                    max_iter=10000, tol=1e-4):
    t0 = time.perf_counter()
    mdl = train_linmod(Xtr, ytr, precision="single", solver=solver,
                       reg_lambda=reg_lambda, reg_alpha=reg_alpha,
                       max_iter=max_iter, tol=tol)
    t1 = time.perf_counter()
    metrics = evaluate(mdl, Xte, yte)
    return {"approach": "single(f32)", "time_sec": t1 - t0, "iters": mdl.n_iter, **metrics}

def approach_double(Xtr, ytr, Xte, yte, *, solver="coord", reg_lambda=0.01, reg_alpha=0.0,
                    max_iter=10000, tol=1e-4):
    t0 = time.perf_counter()
    mdl = train_linmod(Xtr, ytr, precision="double", solver=solver,
                       reg_lambda=reg_lambda, reg_alpha=reg_alpha,
                       max_iter=max_iter, tol=tol)
    t1 = time.perf_counter()
    metrics = evaluate(mdl, Xte, yte)
    return {"approach": "double(f64)", "time_sec": t1 - t0, "iters": mdl.n_iter, **metrics}

def approach_hybrid(Xtr, ytr, Xte, yte, *, solver="coord", reg_lambda=0.01, reg_alpha=0.0,
                    max_iter_single=200, max_iter_double=10000, tol=1e-4):
    # Stage A: fast f32 warm start
    t0 = time.perf_counter()
    mdl_f32 = train_linmod(Xtr, ytr, precision="single", solver=solver,
                           reg_lambda=reg_lambda, reg_alpha=reg_alpha,
                           max_iter=max_iter_single, tol=tol)
    x0 = mdl_f32.coef

    # Stage B: refine in f64 from warm start
    mdl_f64 = linmod(mod="logistic", solver=solver, precision="double",
                     intercept=True, max_iter=max_iter_double, scaling="standardize")
    mdl_f64.fit(Xtr, ytr, reg_lambda=float(reg_lambda), reg_alpha=float(reg_alpha),
                x0=x0, tol=float(tol))
    t1 = time.perf_counter()

    metrics = evaluate(mdl_f64, Xte, yte)
    return {"approach": "hybrid(f32→f64)", "time_sec": t1 - t0, "iters": mdl_f64.n_iter, **metrics}

def approach_multistage_ir(
    Xtr, ytr, Xte, yte, *,
    # schedule = list of (precision, chunk_iters)
    schedule=(("single", 200), ("double", 800)),
    solver="coord", reg_lambda=1e-2, reg_alpha=0.0,
    tol=1e-6, max_chunks=10,
    stop_delta=1e-7  # stop if ||coef_new - coef_old||_2 < stop_delta
):
    """
    Example schedules:
      [("single", 200), ("double", 800)]
      [("single", 100), ("double", 200), ("single", 100), ("double", 200)]
    """
    t0 = time.perf_counter()
    x0 = None
    coef_prev = None
    history = []
    chunk_id = 0

    for _ in range(max_chunks):
        for prec, iters in schedule:
            chunk_id += 1
            mdl = _fit_chunk(
                Xtr, ytr,
                precision="single" if prec.startswith("single") else "double",
                solver=solver, reg_lambda=reg_lambda, reg_alpha=reg_alpha,
                max_iter=int(iters), tol=tol, x0=x0
            )
            # convergence check
            coef = mdl.coef.copy()
            if coef_prev is not None:
                delta = float(np.linalg.norm(coef - coef_prev))
                history.append(("chunk", chunk_id, prec, iters, delta, float(mdl.loss[0])))
                if delta < stop_delta:
                    t1 = time.perf_counter()
                    metrics = _score_model(mdl, Xte, yte)
                    return {"approach": "multistage-IR", "time_sec": t1 - t0, **metrics, "chunks": history}
            else:
                history.append(("chunk", chunk_id, prec, iters, np.nan, float(mdl.loss[0])))

            coef_prev = coef
            x0 = coef  # warm-start next chunk

    # finished all chunks
    t1 = time.perf_counter()
    metrics = _score_model(mdl, Xte, yte)
    return {"approach": "multistage-IR", "time_sec": t1 - t0, **metrics, "chunks": history}


def approach_adaptive_precision(
    Xtr, ytr, Xte, yte, *,
    solver="coord", reg_lambda=1e-2, reg_alpha=0.0,
    chunk_iters=100, tol=1e-6,
    promote_patience=2,       # how many "weak-improvement" chunks before promoting precision
    improve_thresh=1e-3,      # relative loss improvement threshold per chunk
    max_chunks=50,
    allow_demote=False        # set True if you want to switch back to single after good progress
):
    """
    Start in single precision. If relative loss improvement per chunk < improve_thresh
    for 'promote_patience' consecutive chunks, switch to double. Optionally demote back.
    """
    t0 = time.perf_counter()
    prec = "single"
    x0 = None
    best_loss = np.inf
    weak_streak = 0
    history = []

    for k in range(1, max_chunks + 1):
        mdl = _fit_chunk(
            Xtr, ytr, precision=prec, solver=solver,
            reg_lambda=reg_lambda, reg_alpha=reg_alpha,
            max_iter=int(chunk_iters), tol=tol, x0=x0
        )
        loss = float(mdl.loss[0])
        rel_improve = (best_loss - loss) / max(abs(best_loss), 1e-12)
        best_loss = min(best_loss, loss)

        history.append(("chunk", k, prec, chunk_iters, loss, rel_improve, float(mdl.nrm_gradient_loss[0]) if hasattr(mdl,"nrm_gradient_loss") else np.nan))

        if rel_improve < improve_thresh:
            weak_streak += 1
        else:
            weak_streak = 0

        # precision switching logic
        if prec == "single" and weak_streak >= promote_patience:
            prec = "double"; weak_streak = 0  # promote
        elif allow_demote and prec == "double" and rel_improve >= 5 * improve_thresh:
            prec = "single"  # (optional) demote if double suddenly makes big progress

        # stopping by gradient norm or tiny movement
        if hasattr(mdl, "nrm_gradient_loss") and float(mdl.nrm_gradient_loss[0]) < tol:
            break

        x0 = mdl.coef  # warm start

    t1 = time.perf_counter()
    metrics = _score_model(mdl, Xte, yte)
    return {"approach": "adaptive-precision", "time_sec": t1 - t0, **metrics, "chunks": history}


def kfold_eval(X, y, approach_fn, approach_kwargs, k=5, seed=42):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    rows = []
    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]
        res = approach_fn(Xtr, ytr, Xva, yva, **approach_kwargs)
        rows.append({"fold": fold, **{k: v for k, v in res.items() if k != "chunks"}})
    df = pd.DataFrame(rows)
    summary = df.drop(columns=["fold"]).mean(numeric_only=True).to_dict()
    summary.update({"approach": rows[0]["approach"], "kfold": k})
    return df, summary


# -------------------------
# Grid runner
# -------------------------
def run_experiments(X, y,
                    grid=None,
                    test_size=0.25, random_state=42, stratify=True):
    """
    grid keys (values are lists):
      - penalty: ["l2","l1","elasticnet"]  (maps to reg_alpha)
      - alpha:   [None, 0.25, 0.5, 0.75]   (only used when penalty='elasticnet')
      - lambda:  [1e-3, 1e-2, 1e-1]        (AOCL reg_lambda); OR use 'C' instead
      - C:       [0.1, 1.0, 10.0]          (ignored if lambda is present)
      - solver:  ["coord","lbfgs"]         (coord works for logistic + L1/EN; lbfgs best for ridge-like)
      - max_iter:     [2000, 10000]
      - tol:          [1e-4, 1e-6]
      - max_iter_single (hybrid warm start): [100, 300]
      - approaches: ["single","double","hybrid"]  (which to run)
    """
    if grid is None:
        grid = {
            "penalty":   ["l2", "l1", "elasticnet"],
            "alpha":     [None, 0.5],          # used when elasticnet
            "lambda":    [1e-3, 1e-2, 1e-1],
            "C":         [None],               # set non-None to use C mapping instead of lambda
            "solver":    ["coord"],            # safest for logistic + L1/EN
            "max_iter":  [10000],
            "tol":       [1e-4, 1e-6],
            "max_iter_single": [200],          # hybrid warm-start
            "approaches": ["single","double","hybrid"]
        }

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if stratify else None
    )

    keys = ["penalty", "alpha", "lambda", "C", "solver", "max_iter", "tol", "max_iter_single"]
    combos = list(itertools.product(*[grid.get(k, [None]) for k in keys]))

    rows = []

    for vals in combos:
        penalty, alpha, lam, C, solver, max_iter, tol, max_iter_single = vals

        try:
            reg_alpha = _map_penalty_to_alpha(penalty, alpha)
            reg_lambda = _map_C_to_lambda(C=C, reg_lambda=lam)

            # guard: if using 'lbfgs', prefer ridge-like (alpha ~ 0). For L1/EN, coord is recommended.
            if solver == "lbfgs" and reg_alpha != 0.0:
                # Skip incompatible combo (lbfgs not suitable for L1/EN in AOCL-DA docs)
                continue

            for approach in grid.get("approaches", ["single","double","hybrid"]):
                if approach == "single":
                    res = approach_single(
                        Xtr, ytr, Xte, yte,
                        solver=solver, reg_lambda=reg_lambda, reg_alpha=reg_alpha,
                        max_iter=max_iter, tol=tol
                    )
                elif approach == "double":
                    res = approach_double(
                        Xtr, ytr, Xte, yte,
                        solver=solver, reg_lambda=reg_lambda, reg_alpha=reg_alpha,
                        max_iter=max_iter, tol=tol
                    )
                elif approach == "hybrid":
                    res = approach_hybrid(
                        Xtr, ytr, Xte, yte,
                        solver=solver, reg_lambda=reg_lambda, reg_alpha=reg_alpha,
                        max_iter_single=max_iter_single, max_iter_double=max_iter, tol=tol
                    )
                else:
                    continue

                row = {
                    "approach": res["approach"],
                    "penalty": penalty,
                    "alpha": reg_alpha,
                    "lambda": reg_lambda,
                    "C": None if reg_lambda == 0 else 1.0 / reg_lambda,
                    "solver": solver,
                    "max_iter": max_iter,
                    "tol": tol,
                    "time_sec": res["time_sec"],
                    "iters": res["iters"],
                    "auc_test": res["auc"],
                    "logloss_test": res["logloss"],
                    "loss_internal": res["loss_internal"]
                }
                if approach == "hybrid":
                    row["max_iter_single"] = max_iter_single
                rows.append(row)

        except Exception as e:
            # record failures so you can audit incompatible settings
            rows.append({
                "approach": "ERROR",
                "penalty": penalty, "alpha": alpha, "lambda": lam, "C": C,
                "solver": solver, "max_iter": max_iter, "tol": tol,
                "error": str(e)
            })

    df = pd.DataFrame(rows)
    # Order by approach, then descending AUC, then ascending time
    if not df.empty and "auc_test" in df:
        df = df.sort_values(["approach","auc_test","time_sec"], ascending=[True, False, True]).reset_index(drop=True)
    return df

# -------------------------
# Demo
# -------------------------
if __name__ == "__main__":
    dataset = "gaussian"   # options: "gaussian", "uniform", "breast_cancer"

    if dataset == "gaussian":
        X, y = make_shifted_gaussian(m=5000, n=200, delta=0.5, seed=42)
    elif dataset == "uniform":
        X, y = make_uniform_binary(m=5000, n=200, shift=0.25, seed=42)
    elif dataset == "breast_cancer":
        data = load_breast_cancer()
        X = data.data.astype(np.float64)
        y = data.target.astype(np.int32)
    else:
        raise ValueError("Unknown dataset")

    grid = {
        "penalty": ["l2", "l1", "elasticnet"],
        "alpha":   [None, 0.3, 0.7],
        "lambda":  [1e-3, 1e-2, 1e-1],
        "C":       [None],
        "solver":  ["coord", "lbfgs"],
        "max_iter": [3000, 10000],
        "tol":      [1e-4, 1e-6],
        "max_iter_single": [150, 300],
        "approaches": ["single","double","hybrid"]
    }

    df = run_experiments(X, y, grid=grid)
    with pd.option_context("display.max_columns", None, "display.width", 140):
        print(df.groupby("approach").head(5))
