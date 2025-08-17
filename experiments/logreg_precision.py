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

def run_hybrid_optimized(
    X, y, n_classes,
    max_iter_total,
    tol_single,
    tol_double,
    single_iter_cap,
):
    """Optimized non-adaptive hybrid logistic regression"""
    # Single allocation with memory reuse
    X32 = np.asarray(X, dtype=np.float32, order="C")
    X64 = X.astype(np.float64, copy=False) if X.dtype != np.float64 else X
    
    # Cap handling - ensure we don't exceed max_iter_total
    cap = min(int(single_iter_cap), int(max_iter_total))
    
    # --- Phase 1: FP32 with early stopping check ---
    t0 = time.perf_counter()
    clf_s = LogisticRegression(
        solver="lbfgs",
        max_iter=cap,
        tol=tol_single,
        warm_start=False,
        random_state=42  # For reproducibility
    )
    clf_s.fit(X32, y)
    t_single = time.perf_counter() - t0
    it_single = _iter_scalar(clf_s.n_iter_)
    
    # --- Phase 2: FP64 with manual warm start ---
    remaining_iters = max(1, max_iter_total - it_single)
    t1 = time.perf_counter()
    
    # Create new model and manually inject FP32 solution
    clf_d = LogisticRegression(
        solver="lbfgs",
        max_iter=remaining_iters,
        tol=tol_double,
        warm_start=False  # Explicitly disabled for AOCL
    )
    
    # Direct attribute injection (more reliable than warm_start)
    clf_d.coef_ = clf_s.coef_.astype(np.float64, copy=False)
    clf_d.intercept_ = clf_s.intercept_.astype(np.float64, copy=False)
    clf_d.classes_ = clf_s.classes_.copy()
    clf_d.n_iter_ = [0]  # Reset iteration counter
    
    clf_d.fit(X64, y)
    t_double = time.perf_counter() - t1
    it_double = _iter_scalar(clf_d.n_iter_)
    
    # --- Metrics ---
    y_pred = clf_d.predict(X64)
    acc = accuracy_score(y, y_pred)
    total_time = t_single + t_double
    mem_MB = max(X32.nbytes, X64.nbytes) / 1e6
    
    return it_single, it_double, total_time, mem_MB, acc

def adaptive_mixed_precision_lr(X, y, switch_tol=1e-3, max_iter=1000):
    """Instrumented version with tracking"""
    # Precision conversion
    X_fp32 = np.asarray(X, dtype=np.float32, order='C')
    X_fp64 = X.astype(np.float64, copy=False) if X.dtype != np.float64 else X
    
    # FP32 phase
    clf = LogisticRegression(solver='lbfgs', max_iter=max_iter, warm_start=True)
    t_fp32_start = perf_counter()
    clf.fit(X_fp32, y)
    t_fp32 = perf_counter() - t_fp32_start
    
    # Switching logic
    grad_norm = np.linalg.norm(clf.coef_)
    needs_refinement = grad_norm > switch_tol
    
    # FP64 phase if needed
    t_fp64 = 0
    if needs_refinement:
        clf.coef_ = clf.coef_.astype(np.float64)
        clf.intercept_ = clf.intercept_.astype(np.float64)
        
        t_fp64_start = perf_counter()
        clf.fit(X_fp64, y)
        t_fp64 = perf_counter() - t_fp64_start
        
    # Store instrumentation data
    clf._used_mixed_precision = needs_refinement
    clf._time_fp32 = t_fp32
    clf._time_fp64 = t_fp64
    clf._final_precision = 'mixed' if needs_refinement else 'fp32'
    
    return clf
