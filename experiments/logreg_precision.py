# logreg_precision.py
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def evaluate_metrics(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def _iter_scalar(n_iter_attr):
    arr = np.asarray(n_iter_attr)
    return int(arr.max()) if arr.ndim else int(arr)

# --- Full double-precision run ---
def run_full_double(X, y, n_classes, max_iter, tol):
    X64 = np.asarray(X, dtype=np.float64)
    t0 = time.perf_counter()
    clf = LogisticRegression(
        max_iter=max_iter, tol=tol,
        solver="lbfgs", multi_class="auto", n_jobs=1
    )
    clf.fit(X64, y)
    elapsed = time.perf_counter() - t0
    y_pred = clf.predict(X64)
    acc = evaluate_metrics(y, y_pred)
    mem_MB = X64.nbytes / 1e6
    return clf, 0, _iter_scalar(clf.n_iter_), elapsed, mem_MB, acc

# --- Hybrid with warm-start from fp32 to fp64 ---
def run_hybrid(X, y, n_classes, max_iter_total, tol_single, tol_double, single_iter_cap):
    X32 = np.asarray(X, dtype=np.float32)
    X64 = np.asarray(X, dtype=np.float64)

    # No cap -> let single use entire budget (still warm-start later)
    if single_iter_cap is None:
        single_iter_cap = max_iter_total

    cap = int(max(0, min(single_iter_cap, max_iter_total)))

    # Pure double if cap==0
    if cap == 0:
        t0 = time.perf_counter()
        clf_d = LogisticRegression(
            max_iter=max(1, int(max_iter_total)), tol=tol_double,
            solver="lbfgs", multi_class="auto", n_jobs=1
        )
        clf_d.fit(X64, y)
        t = time.perf_counter() - t0
        y_pred = clf_d.predict(X64)
        acc = accuracy_score(y, y_pred)
        # Report peak footprint rather than sum (more realistic)
        mem_MB = max(X32.nbytes, X64.nbytes) / 1e6
        return 0, _iter_scalar(clf_d.n_iter_), t, mem_MB, acc

    # ----- fp32 phase -----
    t0 = time.perf_counter()
    clf_s = LogisticRegression(
        max_iter=cap, tol=tol_single,
        solver="lbfgs", multi_class="auto", n_jobs=1
    )
    clf_s.fit(X32, y)
    t_single = time.perf_counter() - t0
    it_single = _iter_scalar(clf_s.n_iter_)

    # ----- fp64 refinement (WARM-START) -----
    remaining = max(1, int(max_iter_total) - it_single)
    clf_d = LogisticRegression(
        max_iter=remaining, tol=tol_double,
        solver="lbfgs", multi_class="auto",
        warm_start=True, n_jobs=1
    )
    # transplant parameters & classes (cast to float64)
    clf_d.classes_   = clf_s.classes_
    clf_d.coef_      = clf_s.coef_.astype(np.float64, copy=True)
    clf_d.intercept_ = clf_s.intercept_.astype(np.float64, copy=True)

    t1 = time.perf_counter()
    clf_d.fit(X64, y)   # continues from fp32 solution
    t_double = time.perf_counter() - t1

    y_pred = clf_d.predict(X64)
    acc = accuracy_score(y, y_pred)
    mem_MB = max(X32.nbytes, X64.nbytes) / 1e6
    it_double = _iter_scalar(clf_d.n_iter_)

    return it_single, it_double, (t_single + t_double), mem_MB, acc

