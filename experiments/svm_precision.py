from aoclda.sklearn import skpatch; skpatch()
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np, time, psutil

def _iters_scalar(n_iter_attr) -> int:
    if n_iter_attr is None: return 0
    arr = np.atleast_1d(n_iter_attr)
    return int(arr.max()) if arr.size else 0

def _rss_mb():
    return psutil.Process().memory_info().rss / (1024**2)

def _top2_margin(scores: np.ndarray) -> np.ndarray:
    # scores shape: (n_samples, n_classes) for OVR; larger gap => more confident
    part = np.partition(scores, -2, axis=1)[:, -2:]   # top2 per row, unordered
    top1 = part.max(axis=1)
    top2 = part.min(axis=1)
    return top1 - top2

def svm_double_precision(tag, X, y, *, tol, max_iter, C=1.0, kernel='rbf',
                         test_size=0.2, seed=0):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    m0 = _rss_mb(); t0 = time.time()
    clf = SVC(C=C, kernel=kernel, gamma='scale', tol=tol, max_iter=int(max_iter), random_state=seed)
    clf.fit(Xtr, ytr)
    elapsed = time.time() - t0; m1 = _rss_mb()
    acc = accuracy_score(yte, clf.predict(Xte))
    return (tag, len(X), 0, tol, 0, 0, _iters_scalar(getattr(clf, "n_iter_", 0)),
            "Double", elapsed, max(0.0, m1 - m0), acc)

def svm_hybrid_subset_filter(tag, X, y, *,
    cap_frac,                     # e.g., 0.02 => 2% subset for Stage-1
    tol_stage1, tol_stage2,       # stage tolerances
    max_iter_stage1, max_iter_stage2,
    margin_thresh=1.0,            # keep samples with margin <= this
    C=1.0, kernel='rbf', test_size=0.2, seed=0):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    rng = np.random.RandomState(seed)
    n = len(Xtr)
    m = max(1, int(round(cap_frac * n)))
    idx_sub = rng.choice(n, size=m, replace=False)

    mem0 = _rss_mb()
    # ---- Stage 1: subset SVC ----
    t1 = time.time()
    s1 = SVC(C=C, kernel=kernel, gamma='scale', tol=tol_stage1, max_iter=int(max_iter_stage1), random_state=seed)
    s1.fit(Xtr[idx_sub], ytr[idx_sub])
    time1 = time.time() - t1
    it1 = _iters_scalar(getattr(s1, "n_iter_", 0))

    # ---- Filter: keep near-boundary samples ----
    scores = s1.decision_function(Xtr)
    if scores.ndim == 1:          # binary
        margin = np.abs(scores)
    else:                          # multiclass OVR
        margin = _top2_margin(scores)
    keep = (margin <= margin_thresh)
    X_red = Xtr[keep]; y_red = ytr[keep]
    if X_red.shape[0] == 0:       # fallback: if filter too aggressive
        X_red, y_red = Xtr, ytr

    # ---- Stage 2: final SVC on reduced set ----
    t2 = time.time()
    s2 = SVC(C=C, kernel=kernel, gamma='scale', tol=tol_stage2, max_iter=int(max_iter_stage2), random_state=seed)
    s2.fit(X_red, y_red)
    time2 = time.time() - t2
    it2 = _iters_scalar(getattr(s2, "n_iter_", 0))

    acc = accuracy_score(yte, s2.predict(Xte))
    mem1 = _rss_mb()

    return (tag, len(X), 0, tol_stage1, cap_frac, it1, it2,
            "Hybrid", time1 + time2, max(0.0, mem1 - mem0), acc,
            X_tr_kept := X_red.shape[0] / Xtr.shape[0])

