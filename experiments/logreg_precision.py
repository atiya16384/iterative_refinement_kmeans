# experiments/logreg_precision.py
import time
import numpy as np
from aoclda.linear_model import linmod
from sklearn.metrics import accuracy_score

# experiments/logreg_precision.py

def _iter_scalar(n_iter_attr) -> int:
    """scikit can return int (binary) or array (multinomial); normalize to int."""
    arr = np.asarray(n_iter_attr)
    return int(arr.max()) if arr.ndim else int(arr)

# --- Pure double precision run (baseline) ---
def run_full_double(X, y, n_classes, max_iter, tol):
    X64 = np.asarray(X, dtype=np.float64)
    t0 = time.perf_counter()
    clf = linmod("logistic",
        solver="lbfgs", 
        max_iter=int(max_iter), precision='double'
    )
    clf.fit(X64, y, tol=tol)
    elapsed = time.perf_counter() - t0

    it_double = _iter_scalar(getattr(clf, "n_iter_", 0))
    acc = float(accuracy_score(y, clf.predict(X64)))
    mem_MB = X64.nbytes / 1e6

    # Return tuple your experiments expect: (model, it_single, it_double, time, mem, acc)
    return clf, 0, it_double, elapsed, mem_MB, acc

# need to set it up as a classification task
# --- Hybrid: fp32 probe (capped) → fp64 fresh run (remaining budget) ---
def run_hybrid(X, y, n_classes, max_iter_total, tol_single, tol_double, single_iter_cap):
    X32 = np.asarray(X, dtype=np.float32)
    X64 = np.asarray(X, dtype=np.float64)

    # cap can be None; clamp to [0, max_iter_total]
    if single_iter_cap is None:
        cap = int(max_iter_total)
    else:
        cap = int(max(0, min(int(single_iter_cap), int(max_iter_total))))

    # --- Stage 1: fp32 probe (no warm_start; AOCL-safe) ---
    t0 = time.perf_counter()
    clf_fp32 = linmod("logistic",
        solver="lbfgs", 
        max_iter=cap, precision='single'
    )
    clf_fp32.fit(X32, y, tol=tol_single)
    t_single = time.perf_counter() - t0
    it_single = _iter_scalar(getattr(clf_fp32, "n_iter_", 0))

    # --- Stage 2: fp64 from scratch with remaining budget ---
    remaining = int(max(1, int(max_iter_total) - it_single))
    t1 = time.perf_counter()
    clf_fp64 = linmod("logistic",
        solver="lbfgs", 
        max_iter=remaining, precision='double'
    )
    clf_fp64.fit(X64, y, tol_double)
    t_double = time.perf_counter() - t1
    it_double = _iter_scalar(getattr(clf_fp64, "n_iter_", 0))

    # Evaluate
    acc = float(accuracy_score(y, clf_fp64.predict(X64)))
    mem_MB = max(X32.nbytes, X64.nbytes) / 1e6

    # Return the 5-tuple your experiment code writes into CSV
    return it_single, it_double, (t_single + t_double), mem_MB, acc


# def adaptive_mixed_precision_lr(X, y, switch_tol=1e-3, max_iter=1000):
#     """AOCL-compatible adaptive mixed precision logistic regression"""
#     # Precision conversion
#     X_fp32 = np.asarray(X, dtype=np.float32, order='C')
#     X_fp64 = X.astype(np.float64, copy=False) if X.dtype != np.float64 else X
    
#     # --- FP32 Phase ---
#     clf_fp32 = LogisticRegression(
#         solver='lbfgs',
#         max_iter=max_iter,
#         warm_start=False  # Explicitly disabled for AOCL
#     )
#     t_fp32_start = time.perf_counter()
#     clf_fp32.fit(X_fp32, y)
#     t_fp32 = time.perf_counter() - t_fp32_start
    
#     # --- Switching Logic ---
#     grad_norm = np.linalg.norm(clf_fp32.coef_)
#     needs_refinement = grad_norm > switch_tol
    
#     if not needs_refinement:
#         # Early return if FP32 is sufficient
#         clf_fp32._used_mixed_precision = False
#         clf_fp32._time_fp32 = t_fp32
#         clf_fp32._time_fp64 = 0
#         clf_fp32._final_precision = 'fp32'
#         return clf_fp32
    
#     # --- FP64 Refinement Phase ---
#     t_fp64_start = time.perf_counter()
#     clf_fp64 = LogisticRegression(
#         solver='lbfgs',
#         max_iter=max_iter,
#         warm_start=False,
#         # AOCL-compatible initialization
#         coef_init=clf_fp32.coef_.astype(np.float64),
#         intercept_init=clf_fp32.intercept_.astype(np.float64)
#     )
#     clf_fp64.fit(X_fp64, y)
#     t_fp64 = time.perf_counter() - t_fp64_start
    
#     # --- Instrumentation ---
#     clf_fp64._used_mixed_precision = True
#     clf_fp64._time_fp32 = t_fp32
#     clf_fp64._time_fp64 = t_fp64
#     clf_fp64._final_precision = 'mixed'
    
#     return clf_fp64
