# experiments/logreg_precision.py
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- Pure double precision run ---
def run_full_double(X, y, n_classes, max_iter, tol):
    X64 = np.asarray(X, dtype=np.float64)
    t0 = time.perf_counter()
    clf = LogisticRegression(
        max_iter=max_iter, tol=tol,
        solver="lbfgs", multi_class="auto"
    )
    clf.fit(X64, y)
    elapsed = time.perf_counter() - t0
    y_pred = clf.predict(X64)
    acc = accuracy_score(y, y_pred)
    mem_MB = X64.nbytes / 1e6
    return clf, 0, _iter_scalar(clf.n_iter_), elapsed, mem_MB, acc


def _iter_scalar(n_iter_attr):
    arr = np.asarray(n_iter_attr)
    return int(arr.max()) if arr.ndim else int(arr)

def run_hybrid(X, y, n_classes, max_iter_total, tol_single, tol_double, single_iter_cap):
    X32 = np.asarray(X, dtype=np.float32)
    X64 = np.asarray(X, dtype=np.float64)

    cap = int(min(single_iter_cap, max_iter_total))

    # --- Stage 1: fp32 (probe) ---
    t0 = time.perf_counter()
    clf_fp32 = LogisticRegression(
        solver="lbfgs", multi_class="auto",
        max_iter=cap, tol=tol_single
    )
    clf_fp32.fit(X32, y)
    t_single = time.perf_counter() - t0

    # --- Stage 2: fp64 (refine from scratch, AOCL-accelerated) ---
    remaining = max_iter_total - clf_fp32.n_iter_.max()
    t1 = time.perf_counter()
    clf_fp64 = LogisticRegression(
        solver="lbfgs", multi_class="auto",
        max_iter=remaining, tol=tol_double
    )
    clf_fp64.fit(X64, y)
    t_double = time.perf_counter() - t1

    # --- Final evaluation ---
    y_pred = clf_fp64.predict(X64)
    acc = accuracy_score(y, y_pred)
    return {
        "time_fp32": t_single,
        "time_fp64": t_double,
        "iters_fp32": int(np.max(clf_fp32.n_iter_)),
        "iters_fp64": int(np.max(clf_fp64.n_iter_)),
        "acc": acc,
    }

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
