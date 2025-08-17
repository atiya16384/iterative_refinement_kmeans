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
    """
    Stage 1: fp32 LogisticRegression (lbfgs) for up to 'cap' iterations.
    Stage 2: fp64 LogisticRegression, warm-started from fp32 weights.
      - If AOCL blocks attribute injection or warm_start, we fall back to a fresh fp64 fit.
    Returns: it_single, it_double, total_time_sec, peak_mem_MB, accuracy
    """
    # Prepare views once (avoid repeated allocations)
    X32 = np.asarray(X, dtype=np.float32, order="C")
    X64 = np.asarray(X, dtype=np.float64, order="C")

    # Cap handling
    cap = max_iter_total if single_iter_cap is None else int(single_iter_cap)
    cap = int(max(0, min(cap, int(max_iter_total))))

    # ---------- fp32 stage ----------
    t0 = time.perf_counter()
    clf_s = LogisticRegression(
        solver="lbfgs",                # AOCL supports lbfgs
        multi_class="auto",
        max_iter=max(1, cap),
        tol=float(tol_single),
        # n_jobs / warm_start are ignored/unsupported in AOCL; omit them here
    )
    clf_s.fit(X32, y)
    t_single = time.perf_counter() - t0
    it_single = _iter_scalar(clf_s.n_iter_)

    # ---------- fp64 stage (warm-start if possible) ----------
    remaining = max(1, int(max_iter_total) - it_single)

    # Start with a safe default: fresh fit
    do_warm = True
    it_double = 0
    t_double = 0.0
    acc = 0.0

    try:
        # Build fp64 model that *can* warm-start
        clf_d = LogisticRegression(
            solver="lbfgs",
            multi_class="auto",
            max_iter=remaining,
            tol=float(tol_double),
            warm_start=True,          # may be ignored by AOCL, we handle that
        )

        # Inject fp32 solution (cast to fp64). AOCL may block this; hence try/except.
        clf_d.classes_   = np.asarray(clf_s.classes_, dtype=clf_s.classes_.dtype, order="C")
        clf_d.coef_      = clf_s.coef_.astype(np.float64, copy=True)
        clf_d.intercept_ = clf_s.intercept_.astype(np.float64, copy=True)

        t1 = time.perf_counter()
        clf_d.fit(X64, y)            # continue from injected weights
        t_double = time.perf_counter() - t1
        it_double = _iter_scalar(clf_d.n_iter_)
        acc = accuracy_score(y, clf_d.predict(X64))
    except Exception:
        # AOCL-patched estimator may refuse manual state; fall back to fresh fp64 run
        do_warm = False
        clf_d = LogisticRegression(
            solver="lbfgs",
            multi_class="auto",
            max_iter=remaining,
            tol=float(tol_double),
        )
        t1 = time.perf_counter()
        clf_d.fit(X64, y)
        t_double = time.perf_counter() - t1
        it_double = _iter_scalar(clf_d.n_iter_)
        acc = accuracy_score(y, clf_d.predict(X64))

    # Peak memory: keep the bigger array; (plus tiny model state, negligible)
    mem_MB = max(X32.nbytes, X64.nbytes) / 1e6
    total_time = t_single + t_double

    # Optional: if you want to know whether warm start actually happened, you can
    # return `do_warm` too (but keep signature/CSV consistent if you log it).
    return it_single, it_double, total_time, mem_MB, acc
