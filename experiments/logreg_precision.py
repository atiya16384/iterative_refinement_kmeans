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

# --- Hybrid run with warm start (fp32 -> fp64) ---
def run_hybrid(X, y, n_classes, max_iter_total, tol_single, tol_double, single_iter_cap=None):
    # If no cap provided (Experiment B), treat single phase as unconstrained by cap
    if single_iter_cap is None:
        single_iter_cap = max_iter_total

    # --- fp32 single phase ---
    X32 = X.astype(np.float32, copy=False)
    t0 = time.perf_counter()
    clf_single = LogisticRegression(max_iter=single_iter_cap, tol=tol_single,
                                    solver='lbfgs', multi_class='auto')
    clf_single.fit(X32, y)
    t_single = time.perf_counter() - t0
    iter_single = int(np.asarray(clf_single.n_iter_).max())

    # Transfer coefficients to double
    coefs64 = clf_single.coef_.astype(np.float64, copy=False)
    intercept64 = clf_single.intercept_.astype(np.float64, copy=False)

    # --- fp64 refinement ---
    remaining_iter = max(1, max_iter_total - iter_single)
    X64 = X.astype(np.float64, copy=False)
    t1 = time.perf_counter()
    clf_double = LogisticRegression(max_iter=remaining_iter, tol=tol_double,
                                    solver='lbfgs', multi_class='auto', warm_start=True)
    clf_double.classes_ = np.unique(y)
    clf_double.coef_ = coefs64
    clf_double.intercept_ = intercept64
    clf_double.fit(X64, y)
    t_double = time.perf_counter() - t1

    y_pred = clf_double.predict(X64)
    acc = evaluate_metrics(y, y_pred)
    mem_MB_total = X64.nbytes / 1e6 + X32.nbytes / 1e6
    iter_double = int(np.asarray(clf_double.n_iter_).max())

    return iter_single, iter_double, (t_single + t_double), mem_MB_total, acc

