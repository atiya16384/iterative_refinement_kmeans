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
    max_iter_total=100,
    tol_single=1e-4,
    tol_double=1e-8,
    switch_criteria="gradient_norm",
    grad_norm_threshold=1e-3
):
    """Optimized hybrid precision logistic regression"""
    # Single allocation for data (avoid copies)
    X32 = np.asarray(X, dtype=np.float32, order="C")
    X64 = X.astype(np.float64, copy=False) if X.dtype != np.float64 else X
    
    # Stage 1: fp32
    t0 = time.perf_counter()
    clf_s = LogisticRegression(
        solver="lbfgs",
        max_iter=max_iter_total,
        tol=tol_single,
        warm_start=False  # AOCL may ignore anyway
    )
    clf_s.fit(X32, y)
    t_single = time.perf_counter() - t0
    it_single = _iter_scalar(clf_s.n_iter_)
    
    # Early exit if single precision already converged
    if clf_s.n_iter_ < max_iter_total:
        y_pred = clf_s.predict(X32)
        acc = accuracy_score(y, y_pred)
        return it_single, 0, t_single, X32.nbytes/1e6, acc
    
    # Stage 2: fp64 with better initialization
    t1 = time.perf_counter()
    
    # Create new model with proper initialization
    clf_d = LogisticRegression(
        solver="lbfgs",
        max_iter=max_iter_total - it_single,
        tol=tol_double,
        warm_start=False
    )
    
    # Manually initialize with fp32 solution (more reliable than warm_start)
    if hasattr(clf_d, 'coef_'):
        clf_d.coef_ = clf_s.coef_.astype(np.float64)
        clf_d.intercept_ = clf_s.intercept_.astype(np.float64)
        clf_d.classes_ = clf_s.classes_
    
    clf_d.fit(X64, y)
    t_double = time.perf_counter() - t1
    it_double = _iter_scalar(clf_d.n_iter_)
    
    # Calculate metrics
    acc = accuracy_score(y, clf_d.predict(X64))
    total_time = t_single + t_double
    mem_MB = max(X32.nbytes, X64.nbytes) / 1e6
    
    return it_single, it_double, total_time, mem_MB, acc
