# experiments/svm_precision.py
from aoclda.sklearn import skpatch; skpatch()
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np, time, psutil

def _iters_scalar(n_iter_attr) -> int:
    if n_iter_attr is None: return 0
    arr = np.atleast_1d(n_iter_attr)
    return int(arr.max()) if arr.size else 0

def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024**2)

def _top2_margin(scores: np.ndarray) -> np.ndarray:
    # scores: (n_samples, n_classes) for OVR; large gap => confident
    part = np.partition(scores, -2, axis=1)[:, -2:]
    top1 = part.max(axis=1); top2 = part.min(axis=1)
    return top1 - top2

def svm_double_precision(tag, X, y, *, max_iter, tol, cap=0,
                         C=1.0, kernel='rbf', test_size=0.2, seed=0):
    """Pure double-precision SVC (baseline). Returns 11-tuple."""
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    # scale BOTH baseline and hybrid fairly
    scaler = StandardScaler(copy=True).fit(Xtr)
    Xtr = scaler.transform(Xtr); Xte = scaler.transform(Xte)

    m0 = _rss_mb(); t0 = time.perf_counter()
    clf = SVC(C=C, kernel=kernel, gamma='scale',
              tol=float(tol), max_iter=int(max_iter),
              cache_size=1024, random_state=seed)
    clf.fit(Xtr, ytr)
    elapsed = time.perf_counter() - t0; m1 = _rss_mb()
    acc = accuracy_score(yte, clf.predict(Xte))
    return (
        tag, len(X), int(np.unique(y).size), float(tol), int(cap),
        0, _iters_scalar(getattr(clf, "n_iter_", 0)),
        "Double", elapsed, max(0.0, m1 - m0), float(acc)
    )

def svm_hybrid_precision(tag, X, y, *,
                         max_iter_total,
                         tol_single, tol_double,
                         single_iter_cap,            # interpreted as % subset for stage-1 (e.g., 1,2,5,10)
                         C=1.0, kernel='rbf',
                         test_size=0.2, seed=0):
    """
    Two-stage hybrid:
      1) Train SVC on a small random subset (single_iter_cap %) with tol_single
      2) Use that model to find 'hard' samples (small decision margins) and
         train final SVC on the reduced set with tol_double.
    Returns 11-tuple with the same schema as baseline.
    """
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler(copy=True).fit(Xtr)
    Xtr = scaler.transform(Xtr); Xte = scaler.transform(Xte)

    # subset fraction from cap (accept 1..100 or 0..1)
    cap = float(single_iter_cap)
    cap_frac = cap/100.0 if cap > 1.0 else cap
    cap_frac = float(np.clip(cap_frac, 0.01, 0.20))  # keep Stage-1 small (1–20%)

    rng = np.random.RandomState(seed)
    n = Xtr.shape[0]
    m = max(1, int(round(cap_frac * n)))
    idx_sub = rng.choice(n, size=m, replace=False)

    mem0 = _rss_mb()

    # ---- Stage 1: probe on subset ----
    t1 = time.perf_counter()
    s1 = SVC(C=C, kernel=kernel, gamma='scale',
             tol=float(tol_single),
             max_iter=max(1, int(max_iter_total // 5)),
             cache_size=1024, random_state=seed)
    s1.fit(Xtr[idx_sub], ytr[idx_sub])
    time1 = time.perf_counter() - t1
    it1 = _iters_scalar(getattr(s1, "n_iter_", 0))

    # ---- Filter hard samples from full train ----
    scores = s1.decision_function(Xtr)
    if scores.ndim == 1:                # binary
        margin = np.abs(scores)
    else:                               # multiclass OVR
        margin = _top2_margin(scores)
    keep_frac = 0.25                    # keep hardest 25% (works well in practice)
    thresh = np.percentile(margin, keep_frac * 100.0)
    keep = (margin <= thresh)
    X_red, y_red = Xtr[keep], ytr[keep]
    if X_red.shape[0] == 0:
        X_red, y_red = Xtr, ytr   # fallback

    # ---- Stage 2: final SVC on reduced set ----
    t2 = time.perf_counter()
    s2 = SVC(C=C, kernel=kernel, gamma='scale',
             tol=float(tol_double),
             max_iter=int(max_iter_total),
             cache_size=1024, random_state=seed)
    s2.fit(X_red, y_red)
    time2 = time.perf_counter() - t2
    it2 = _iters_scalar(getattr(s2, "n_iter_", 0))

    acc = accuracy_score(yte, s2.predict(Xte))
    mem1 = _rss_mb()

    return (
        tag, len(X), int(np.unique(y).size), float(tol_single), int(single_iter_cap),
        it1, it2, "Hybrid", time1 + time2, max(0.0, mem1 - mem0), float(acc)
    )
