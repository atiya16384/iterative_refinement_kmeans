# experiments/svm_precision.py
import numpy as np, time, psutil
from sklearn.svm import SVC, NuSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------- helpers ----------
def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024**2)

def _iters_scalar(n_iter_attr) -> int:
    # scikit's SVC exposes n_iter_ only in some cases; make it robust
    if n_iter_attr is None: return 0
    arr = np.atleast_1d(n_iter_attr)
    return int(arr.max()) if arr.size else 0

def _top2_margin(scores: np.ndarray) -> np.ndarray:
    """
    For OVR decision_function scores with shape (n_samples, n_classes):
    margin = gap between top-1 and top-2 class scores.
    Smaller margin => "harder" sample.
    """
    if scores.ndim == 1:  # binary OVR returns (n,)
        return np.abs(scores)
    part = np.partition(scores, -2, axis=1)[:, -2:]
    return part.max(axis=1) - part.min(axis=1)

def _stratified_subset(y, m, rng):
    """Pick ~m samples keeping class balance."""
    y = np.asarray(y)
    idx_out = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        k = max(1, int(round(m * (cls_idx.size / y.size))))
        pick = rng.choice(cls_idx, size=min(k, cls_idx.size), replace=False)
        idx_out.append(pick)
    return np.concatenate(idx_out)

# ---------- baselines ----------
def svm_double_precision(
    tag, X, y, *,
    tol, max_iter,
    cap=0,                        # kept for schema compatibility (unused)
    C=1.0, kernel='rbf', gamma='scale',
    test_size=0.2, seed=0, cache_mb=1024
):
    """
    Baseline: train one SVC on the full training set (double precision).
    Returns the standard 11-tuple your plotting code expects.
    """
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr).astype(np.float64, copy=False)
    Xte = scaler.transform(Xte).astype(np.float64, copy=False)

    m0 = _rss_mb(); t0 = time.perf_counter()
    clf = SVC(C=C, kernel=kernel, gamma=gamma, tol=float(tol),
              max_iter=int(max_iter), cache_size=float(cache_mb), random_state=seed)
    clf.fit(Xtr, ytr)
    elapsed = time.perf_counter() - t0; m1 = _rss_mb()

    acc = accuracy_score(yte, clf.predict(Xte))
    it = _iters_scalar(getattr(clf, "n_iter_", 0))
    # (Dataset tag, N, n_classes, tol, cap, it_single, it_double, Suite, Time, Mem, Acc)
    return (tag, len(X), int(np.unique(y).size), float(tol), int(cap),
            0, it, "Double", elapsed, max(0.0, m1 - m0), float(acc))

# experiments/svm_precision.py
def svm_hybrid_precision(
    tag, X, y, *,
    max_iter_total,
    tol_single, tol_double,
    single_iter_cap,                  # percent (e.g. 5) or fraction (e.g. 0.05)
    C=1.0, kernel='rbf', gamma='scale',
    keep_frac=0.10,                   # fraction of a small probe to add
    probe_max=100_000,                # max rows to probe for margins
    test_size=0.2, seed=0, cache_mb=512
):
    """
    Two-stage 'data hybrid' (subset -> refine).  When cap<=0 this becomes
    an exact baseline fit (one SVC in double) but the row is labelled 'Hybrid'
    so plots join the first points fairly.
    """
    # ---- identical split + scaling for fairness ----
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xte = scaler.transform(Xte)

    rng = np.random.RandomState(seed)
    n = Xtr.shape[0]

    # interpret CAP as % if >=1, else as fraction; clamp to [0,1]
    cap = float(single_iter_cap)
    cap_frac = cap/100.0 if cap >= 1.0 else cap
    cap_frac = float(np.clip(cap_frac, 0.0, 1.0))
    m = int(round(cap_frac * n))

    # ---------- special case: cap==0 -> match baseline exactly ----------
    if m <= 0:
        mem0 = _rss_mb(); t0 = time.perf_counter()
        s2 = SVC(C=C, kernel=kernel, gamma=gamma, tol=float(tol_double),
                 max_iter=int(max_iter_total), cache_size=float(cache_mb),
                 random_state=seed)
        s2.fit(Xtr.astype(np.float64, copy=False), ytr)
        it2 = _iters_scalar(getattr(s2, "n_iter_", 0))
        elapsed = time.perf_counter() - t0; mem1 = _rss_mb()
        acc = accuracy_score(yte, s2.predict(Xte.astype(np.float64, copy=False)))
        # suite='Hybrid' so the visualiser compares Hybrid vs Double at Cap=0
        return (tag, len(X), int(np.unique(y).size), float(tol_single), int(single_iter_cap),
                0, int(it2), "Hybrid", elapsed, max(0.0, mem1 - mem0), float(acc))

    # ---------- Stage 1: small stratified subset (float32 OK) ----------
    idx_sub = _stratified_subset(ytr, m, rng)
    s1 = SVC(C=C, kernel=kernel, gamma=gamma, tol=float(tol_single),
             max_iter=int(max_iter_total // 3), cache_size=float(cache_mb),
             random_state=seed)
    s1.fit(Xtr[idx_sub].astype(np.float32, copy=False), ytr[idx_sub])
    it1 = _iters_scalar(getattr(s1, "n_iter_", 0))

    # map subset SVs back to global indices
    idx_sv = idx_sub[np.asarray(s1.support_, dtype=int)]

    # ---------- Build reduced set via a small probe ----------
    pool = np.setdiff1d(np.arange(n, dtype=int), idx_sv, assume_unique=False)
    probe_size = min(int(probe_max), int(pool.size))
    if probe_size > 0:
        probe = rng.choice(pool, size=probe_size, replace=False)
        scores = s1.decision_function(Xtr[probe].astype(np.float32, copy=False))
        margins = _top2_margin(scores)
        k = max(1, int(round(float(np.clip(keep_frac, 0.0, 1.0)) * probe_size)))
        add = probe[np.argpartition(margins, k-1)[:k]]
        idx_final = np.unique(np.concatenate([idx_sv, add]))
    else:
        idx_final = idx_sv  # no probe; just the SVs we already found

    X_red = Xtr[idx_final].astype(np.float64, copy=False)
    y_red = ytr[idx_final]

    # ---------- Stage 2: final SVC (double) on reduced set ----------
    mem0 = _rss_mb(); t0 = time.perf_counter()
    s2 = SVC(C=C, kernel=kernel, gamma=gamma, tol=float(tol_double),
             max_iter=int(max_iter_total), cache_size=float(cache_mb),
             random_state=seed)
    s2.fit(X_red, y_red)
    it2 = _iters_scalar(getattr(s2, "n_iter_", 0))
    elapsed = time.perf_counter() - t0; mem1 = _rss_mb()
    acc = accuracy_score(yte, s2.predict(Xte.astype(np.float64, copy=False)))

    return (tag, len(X), int(np.unique(y).size), float(tol_single), int(single_iter_cap),
            int(it1), int(it2), "Hybrid", elapsed, max(0.0, mem1 - mem0), float(acc))
