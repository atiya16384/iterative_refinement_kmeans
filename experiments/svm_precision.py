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
    cap_frac,                      # subset fraction for the probe (1–5% is typical)
    tol_stage1, tol_stage2,
    max_iter_stage1, max_iter_stage2,
    target_keep_frac=0.2,          # <-- new: keep hardest 20%
    probe_gamma=None,              # <-- new: e.g., 0.25 * 'scale'
    cache_size_mb=1000,            # <-- new: bigger kernel cache
    scale_X=True,                  # <-- new: scale once for both stages
    C=1.0, kernel='rbf', test_size=0.2, seed=0):

    # split
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)

    # optional scaling
    if scale_X:
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(Xtr)
        Xtr = sc.transform(Xtr); Xte = sc.transform(Xte)

    rng = np.random.RandomState(seed)
    n = len(Xtr)
    m = max(1, int(round(cap_frac * n)))
    idx_sub = rng.choice(n, size=m, replace=False)

    mem0 = _rss_mb()

    # ---- Stage 1 (cheap probe) ----
    g1 = 'scale' if probe_gamma is None else probe_gamma
    t1 = time.time()
    s1 = SVC(C=C, kernel=kernel, gamma=g1, tol=tol_stage1,
             max_iter=int(max_iter_stage1), cache_size=cache_size_mb, random_state=seed)
    s1.fit(Xtr[idx_sub], ytr[idx_sub])
    time1 = time.time() - t1
    it1 = _iters_scalar(getattr(s1, "n_iter_", 0))

    # ---- Filter by percentile of margins ----
    scores = s1.decision_function(Xtr)
    if scores.ndim == 1:
        margin = np.abs(scores)
    else:
        margin = _top2_margin(scores)
    th = np.percentile(margin, int(round(target_keep_frac * 100)))
    keep = (margin <= th)
    X_red, y_red = Xtr[keep], ytr[keep]
    if X_red.shape[0] == 0:
        X_red, y_red = Xtr, ytr

    # ---- Stage 2 (final) ----
    t2 = time.time()
    s2 = SVC(C=C, kernel=kernel, gamma='scale', tol=tol_stage2,
             max_iter=int(max_iter_stage2), cache_size=cache_size_mb, random_state=seed)
    s2.fit(X_red, y_red)
    time2 = time.time() - t2
    it2 = _iters_scalar(getattr(s2, "n_iter_", 0))

    acc = accuracy_score(yte, s2.predict(Xte))
    mem1 = _rss_mb()

    frac_kept = X_red.shape[0] / Xtr.shape[0]
    return (tag, len(X), 0, tol_stage1, cap_frac, it1, it2,
            "Hybrid", time1 + time2, max(0.0, mem1 - mem0), acc, frac_kept)
