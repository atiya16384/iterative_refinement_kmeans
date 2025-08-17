# experiments/logreg_precision.py
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

def evaluate_metrics(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def _iter_scalar(n_iter_attr):
    arr = np.asarray(n_iter_attr)
    return int(arr.max()) if arr.ndim else int(arr)

def run_full_double(X, y, n_classes, max_iter, tol):
    X64 = np.asarray(X, dtype=np.float64)
    t0 = time.perf_counter()
    clf = LogisticRegression(
        max_iter=max_iter, tol=tol,
        solver="lbfgs", multi_class="auto"  # AOCL ignores n_jobs
    )
    clf.fit(X64, y)
    elapsed = time.perf_counter() - t0
    y_pred = clf.predict(X64)
    acc = evaluate_metrics(y, y_pred)
    mem_MB = X64.nbytes / 1e6
    return clf, 0, _iter_scalar(clf.n_iter_), elapsed, mem_MB, acc

def run_hybrid(
    X, y, n_classes,
    max_iter_total,
    tol_single,
    tol_double,
    single_iter_cap,
    min_acc_to_skip=None,   # e.g., 0.98 to skip fp64 if fp32 is already that good
):
    """
    AOCL-compatible hybrid:
      1) run fp32 for up to 'cap' iterations (lbfgs)
      2) if fp32 converged early (did not hit cap) and satisfies 'min_acc_to_skip' (if set),
         SKIP fp64 and return
      3) else run a fresh fp64 model for the REMAINING budget (no warm_start)
    Notes:
      - No warm_start and no pre-fit attribute injection (AOCL forbids/ignores).
      - Memory is reported as peak of {fp32, fp64} arrays.
    """
    X32 = np.asarray(X, dtype=np.float32)
    X64 = np.asarray(X, dtype=np.float64)

    # sanitize cap
    cap = max_iter_total if single_iter_cap is None else int(single_iter_cap)
    cap = int(max(0, min(cap, int(max_iter_total))))

    # === fp32 phase ===
    t0 = time.perf_counter()
    clf_s = LogisticRegression(
        max_iter=cap, tol=tol_single,
        solver="lbfgs", multi_class="auto"
    )
    clf_s.fit(X32, y)
    t_single = time.perf_counter() - t0
    it_single = _iter_scalar(clf_s.n_iter_)

    # training accuracy (cheap proxy)
    acc_single = accuracy_score(y, clf_s.predict(X32))

    # If fp32 converged BEFORE hitting the cap, sklearn sets n_iter_ < cap
    converged_fp32 = (it_single < cap)

    # === SKIP rule (AOCL-friendly) ===
    if converged_fp32 and (min_acc_to_skip is None or acc_single >= float(min_acc_to_skip)):
        mem_MB = max(X32.nbytes, X64.nbytes) / 1e6
        return it_single, 0, t_single, mem_MB, acc_single

    # === fp64 refinement (fresh, no warm-start) ===
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

