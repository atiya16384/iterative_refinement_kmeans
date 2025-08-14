# experiments/logreg_precision.py
from aoclda.sklearn import skpatch; skpatch()
import time, psutil, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ----------------------------
# small helpers
# ----------------------------
def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024**2)

def _iters_scalar(n_iter_attr) -> int:
    """LogReg n_iter_ can be an array (OVR). Normalize to an int."""
    if n_iter_attr is None: return 0
    arr = np.atleast_1d(n_iter_attr)
    return int(arr.max()) if arr.size else 0


# =============================================================================
# Logistic Regression — DOUBLE (baseline)
# =============================================================================
def logreg_double_precision(
    tag, X, y, *,
    max_iter,                 # total iteration budget (all in float64)
    tol,
    cap=0,                    # kept for schema compatibility / plotting
    C=1.0,
    solver="lbfgs",
    test_size=0.2,
    seed=0,
):
    """
    Baseline: single float64 fit with (max_iter, tol).
    Returns a row tuple:
      (DatasetName, DatasetSize, ModelDim, Tolerance, Cap,
       IterSingle, IterDouble, Suite, Time, Memory_MB, Accuracy)
    """
    # 1) split + scale once (fairness)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    sc = StandardScaler().fit(Xtr)
    Xtr = sc.transform(Xtr).astype(np.float64, copy=False)
    Xte = sc.transform(Xte).astype(np.float64, copy=False)

    # 2) fit in double
    m0 = _rss_mb(); t0 = time.perf_counter()
    clf = LogisticRegression(
        C=float(C), solver=solver, tol=float(tol),
        max_iter=int(max_iter), warm_start=True, random_state=seed
    )
    clf.fit(Xtr, ytr)
    elapsed = time.perf_counter() - t0; m1 = _rss_mb()

    # 3) metrics
    acc = accuracy_score(yte, clf.predict(Xte))
    it_d = _iters_scalar(getattr(clf, "n_iter_", 0))

    return (tag, len(X), int(np.unique(y).size), float(tol), int(cap),
            0, it_d, "Double", elapsed, max(0.0, m1 - m0), float(acc))


# =============================================================================
# Logistic Regression — HYBRID (float32 -> float64 warm-start)
# =============================================================================
def logreg_hybrid_precision(
    tag, X, y, *,
    max_iter_total,          # total iteration budget (stage1 + stage2)
    tol_single,              # stage-1 tol (float32)
    tol_double,              # stage-2 tol (float64)
    single_iter_cap,         # stage-1 iters
    C=1.0,
    solver="lbfgs",
    test_size=0.2,
    seed=0,
):
    """
    Two-stage warm-start:
      Stage-1: fit on float32 with (single_iter_cap, tol_single)
      Stage-2: continue on float64 with (max_iter_total - cap, tol_double)
    """
    # 1) split + scale (once)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    sc = StandardScaler().fit(Xtr)
    Xtr32 = sc.transform(Xtr).astype(np.float32, copy=False)  # fast stage
    Xtr64 = Xtr32.astype(np.float64, copy=False)              # polish stage
    Xte64 = sc.transform(Xte).astype(np.float64, copy=False)

    cap = int(max(0, min(int(single_iter_cap), int(max_iter_total))))
    rem = int(max(0, int(max_iter_total) - cap))

    # 2) Stage-1 (float32)
    m0 = _rss_mb(); t0 = time.perf_counter()
    clf = LogisticRegression(
        C=float(C), solver=solver, tol=float(tol_single),
        max_iter=int(cap), warm_start=True, random_state=seed
    )
    clf.fit(Xtr32, ytr)
    it_s = _iters_scalar(getattr(clf, "n_iter_", 0))

    # 3) Stage-2 (float64) — continue from same weights
    clf.set_params(max_iter=int(rem), tol=float(tol_double))
    clf.fit(Xtr64, ytr)
    it_d = _iters_scalar(getattr(clf, "n_iter_", 0))

    elapsed = time.perf_counter() - t0; m1 = _rss_mb()
    acc = accuracy_score(yte, clf.predict(Xte64))

    return (tag, len(X), int(np.unique(y).size), float(tol_single), int(cap),
            int(it_s), int(it_d), "Hybrid", elapsed, max(0.0, m1 - m0), float(acc))
