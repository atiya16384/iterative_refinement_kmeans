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

# --- Hybrid: run single (fp32) then double (fp64) ---
def run_hybrid(X, y, n_classes, max_iter_total, tol_single, tol_double, single_iter_cap):
    X32 = np.asarray(X, dtype=np.float32)
    X64 = np.asarray(X, dtype=np.float64)

    cap = max_iter_total if single_iter_cap is None else int(single_iter_cap)
    cap = int(max(0, min(cap, int(max_iter_total))))

    # fp32 stage
    t0 = time.perf_counter()
    clf_s = LogisticRegression(
        max_iter=cap, tol=tol_single,
        solver="lbfgs", multi_class="auto"
    )
    clf_s.fit(X32, y)
    t_single = time.perf_counter() - t0
    it_single = _iter_scalar(clf_s.n_iter_)

    # fp64 refinement (fresh run, AOCL-friendly)
    remaining = max(1, int(max_iter_total) - it_single)
    t1 = time.perf_counter()
    clf_d = LogisticRegression(
        max_iter=remaining, tol=tol_double,
        solver="lbfgs", multi_class="auto"
    )
    clf_d.fit(X64, y)
    t_double = time.perf_counter() - t1
    it_double = _iter_scalar(clf_d.n_iter_)

    y_pred = clf_d.predict(X64)
    acc = accuracy_score(y, y_pred)
    mem_MB = max(X32.nbytes, X64.nbytes) / 1e6

    return it_single, it_double, (t_single + t_double), mem_MB, acc

