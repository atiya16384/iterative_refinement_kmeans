# experiments/logreg_precision.py
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def _iter_scalar(n_iter_attr):
    arr = np.asarray(n_iter_attr)
    return int(arr.max()) if arr.ndim else int(arr)

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

def run_hybrid(
    X, y, n_classes,
    max_iter_total,
    tol_single,
    tol_double,
    single_iter_cap,
):
    """AOCL-compatible hybrid precision logistic regression"""
    # Single allocation with memory reuse
    X32 = np.asarray(X, dtype=np.float32, order="C")
    X64 = X.astype(np.float64, copy=False) if X.dtype != np.float64 else X
    
    # Cap handling
    cap = min(int(single_iter_cap), int(max_iter_total))
    
    # --- Phase 1: FP32 ---
    t0 = time.perf_counter()
    clf_s = LogisticRegression(
        solver="lbfgs",
        max_iter=cap,
        tol=tol_single,
        warm_start=False
    ).fit(X32, y)
    t_single = time.perf_counter() - t0
    it_single = _iter_scalar(clf_s.n_iter_)
    
    # --- Phase 2: FP64 ---
    t1 = time.perf_counter()
    
    # AOCL-compatible initialization
    clf_d = LogisticRegression(
        solver="lbfgs",
        max_iter=max_iter_total - it_single,
        tol=tol_double,
        warm_start=False,
        # Initialize with FP32 solution through parameters
        coef_init=clf_s.coef_.astype(np.float64),
        intercept_init=clf_s.intercept_.astype(np.float64)
    ).fit(X64, y)
    
    t_double = time.perf_counter() - t1
    it_double = _iter_scalar(clf_d.n_iter_)
    
    # Metrics
    acc = accuracy_score(y, clf_d.predict(X64))
    return (
        it_single,
        it_double,
        t_single + t_double,
        max(X32.nbytes, X64.nbytes) / 1e6,
        acc
    )

def adaptive_mixed_precision_lr(X, y, switch_tol=1e-3, max_iter=1000):
    """AOCL-compatible adaptive mixed precision logistic regression"""
    # Precision conversion
    X_fp32 = np.asarray(X, dtype=np.float32, order='C')
    X_fp64 = X.astype(np.float64, copy=False) if X.dtype != np.float64 else X
    
    # --- FP32 Phase ---
    clf_fp32 = LogisticRegression(
        solver='lbfgs',
        max_iter=max_iter,
        warm_start=False  # Explicitly disabled for AOCL
    )
    t_fp32_start = time.perf_counter()
    clf_fp32.fit(X_fp32, y)
    t_fp32 = time.perf_counter() - t_fp32_start
    
    # --- Switching Logic ---
    grad_norm = np.linalg.norm(clf_fp32.coef_)
    needs_refinement = grad_norm > switch_tol
    
    if not needs_refinement:
        # Early return if FP32 is sufficient
        clf_fp32._used_mixed_precision = False
        clf_fp32._time_fp32 = t_fp32
        clf_fp32._time_fp64 = 0
        clf_fp32._final_precision = 'fp32'
        return clf_fp32
    
    # --- FP64 Refinement Phase ---
    t_fp64_start = time.perf_counter()
    clf_fp64 = LogisticRegression(
        solver='lbfgs',
        max_iter=max_iter,
        warm_start=False,
        # AOCL-compatible initialization
        coef_init=clf_fp32.coef_.astype(np.float64),
        intercept_init=clf_fp32.intercept_.astype(np.float64)
    )
    clf_fp64.fit(X_fp64, y)
    t_fp64 = time.perf_counter() - t_fp64_start
    
    # --- Instrumentation ---
    clf_fp64._used_mixed_precision = True
    clf_fp64._time_fp32 = t_fp32
    clf_fp64._time_fp64 = t_fp64
    clf_fp64._final_precision = 'mixed'
    
    return clf_fp64
