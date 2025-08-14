import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def evaluate_metrics(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

# --- Full double-precision run ---
def run_full_double(X, y, n_classes, max_iter, tol):
    X64 = X.astype(np.float64, copy=False)
    t0 = time.perf_counter()
    clf = LogisticRegression(max_iter=max_iter, tol=tol,
                             solver='lbfgs', multi_class='auto')
    clf.fit(X64, y)
    elapsed = time.perf_counter() - t0
    y_pred = clf.predict(X64)
    acc = evaluate_metrics(y, y_pred)
    mem_MB = X64.nbytes / 1e6
    # n_iter_ can be array (per class); use max for a scalar count
    return clf, 0, int(np.asarray(clf.n_iter_).max()), elapsed, mem_MB, acc

def _iter_scalar(n_iter_attr):
    # n_iter_ can be array-like (per class); use the max as a scalar
    return int(np.asarray(n_iter_attr).max()) if np.ndim(n_iter_attr) >= 1 else int(n_iter_attr)

import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def _iter_scalar(n_iter_attr):
    # n_iter_ can be array-like per class (OvR). Use max as a scalar.
    arr = np.asarray(n_iter_attr)
    return int(arr.max()) if arr.ndim else int(arr)

def run_hybrid(X, y, n_classes, max_iter_total, tol_single, tol_double, single_iter_cap):
    """
    AOCL-compatible hybrid (NO warm_start / NO manual coef assignment).

    Flow:
      - Cast once: keep X32 and X64.
      - If cap==0: run pure fp64 baseline.
      - Else: run fp32 phase for 'cap' iters (measure time), then start a *fresh*
        fp64 LogisticRegression with remaining iteration budget.
    """
    # Cast once, reuse
    X32 = np.asarray(X, dtype=np.float32)
    X64 = np.asarray(X, dtype=np.float64)

    # Cap sanitize
    if single_iter_cap is None:
        single_iter_cap = max_iter_total
    cap = int(max(0, min(single_iter_cap, max_iter_total)))

    # === cap == 0 → pure double baseline ===
    if cap == 0:
        t_single = 0.0
        iter_single = 0

        remaining_iter = max(1, int(max_iter_total))
        t1 = time.perf_counter()
        clf_double = LogisticRegression(
            max_iter=remaining_iter, tol=tol_double,
            solver='lbfgs', multi_class='auto'
        )
        clf_double.fit(X64, y)
        t_double = time.perf_counter() - t1

        y_pred = clf_double.predict(X64)
        acc = accuracy_score(y, y_pred)
        mem_MB_total = (X32.nbytes + X64.nbytes) / 1e6
        iter_double = _iter_scalar(clf_double.n_iter_)
        return iter_single, iter_double, (t_single + t_double), mem_MB_total, acc

    # === fp32 phase (capped) ===
    t0 = time.perf_counter()
    clf_single = LogisticRegression(
        max_iter=cap, tol=tol_single,
        solver='lbfgs', multi_class='auto'
    )
    clf_single.fit(X32, y)
    t_single = time.perf_counter() - t0
    iter_single = _iter_scalar(clf_single.n_iter_)

    # === fp64 refinement (fresh model; AOCL-friendly) ===
    remaining_iter = max(1, int(max_iter_total) - iter_single)
    t1 = time.perf_counter()
    clf_double = LogisticRegression(
        max_iter=remaining_iter, tol=tol_double,
        solver='lbfgs', multi_class='auto'
    )
    clf_double.fit(X64, y)
    t_double = time.perf_counter() - t1

    y_pred = clf_double.predict(X64)
    acc = accuracy_score(y, y_pred)
    mem_MB_total = (X32.nbytes + X64.nbytes) / 1e6
    iter_double = _iter_scalar(clf_double.n_iter_)

    return iter_single, iter_double, (t_single + t_double), mem_MB_total, acc
