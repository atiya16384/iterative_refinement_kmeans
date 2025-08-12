from aoclda.sklearn import skpatch; skpatch()
import numpy as np, time, psutil
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024**2)

def _epochs_fit(clf: SGDClassifier, X, y, epochs, batch_size, classes, dtype, rng_seed=0):
    """
    Mini-epoch partial_fit loop.
    rng_seed ensures identical minibatch order across Double vs Hybrid.
    """
    if epochs <= 0:
        return 0
    rng = np.random.RandomState(rng_seed)   # <-- same order for both suites
    n = X.shape[0]
    bs = max(1, int(batch_size))
    first = True
    iters = 0
    for _ in range(int(epochs)):
        idx = rng.permutation(n)
        for i in range(0, n, bs):
            sl = idx[i:i+bs]
            if sl.size == 0:
                continue
            Xb = X[sl].astype(dtype, copy=False)
            yb = y[sl]
            if first:
                clf.partial_fit(Xb, yb, classes=classes)
                first = False
            else:
                clf.partial_fit(Xb, yb)
        iters += 1
    return iters

# -----------------------
# Baseline: all double
# -----------------------
def svm_double_precision(tag, X, y, *, max_iter, tol, cap=0, alpha=1e-4,
                         batch_size=2048, test_size=0.2, seed=0):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr); Xte = scaler.transform(Xte)
    classes = np.unique(ytr)

    clf = SGDClassifier(loss='hinge', alpha=float(alpha), learning_rate='optimal',
                        tol=float(tol), warm_start=True, random_state=seed)

    m0 = _rss_mb(); t0 = time.perf_counter()
    it_d = _epochs_fit(clf, Xtr, ytr,
                       epochs=int(max_iter),
                       batch_size=int(batch_size),
                       classes=classes,
                       dtype=np.float64,
                       rng_seed=seed)   # <-- same batch order
    elapsed = time.perf_counter() - t0; m1 = _rss_mb()
    acc = accuracy_score(yte, clf.predict(Xte))
    return (tag, len(X), int(classes.size), float(tol), int(cap),
            0, int(it_d), "Double", elapsed, max(0.0, m1 - m0), float(acc))

# -----------------------
# Hybrid IR: float32 -> float64
# -----------------------
def svm_hybrid_precision(tag, X, y, *, max_iter_total, tol_single, tol_double,
                         single_iter_cap, alpha=1e-4, batch_size=2048,
                         test_size=0.2, seed=0):
    """
    Iterative refinement with equal total work:
      - Stage-1: float32 for `single_iter_cap` epochs (fast)
      - Stage-2: float64 for `(max_iter_total - single_iter_cap)` epochs (polish)
      If `single_iter_cap == 0`, we do exactly the same as the Double baseline
      (so the ratio at cap=0 is ~1.0).
    """
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr); Xte = scaler.transform(Xte)
    classes = np.unique(ytr)

    cap = int(max(0, min(int(single_iter_cap), int(max_iter_total))))
    rem = int(max(0, int(max_iter_total) - cap))

    # --- Special case: cap == 0  -> identical path as Double (but Suite="Hybrid")
    if cap == 0:
        clf = SGDClassifier(loss='hinge', alpha=float(alpha), learning_rate='optimal',
                            tol=float(tol_double), warm_start=True, random_state=seed)
        m0 = _rss_mb(); t0 = time.perf_counter()
        it_d = _epochs_fit(clf, Xtr, ytr,
                           epochs=int(max_iter_total),
                           batch_size=int(batch_size),
                           classes=classes,
                           dtype=np.float64,
                           rng_seed=seed)
        elapsed = time.perf_counter() - t0; m1 = _rss_mb()
        acc = accuracy_score(yte, clf.predict(Xte))
        return (tag, len(X), int(classes.size), float(tol_single), 0,
                0, int(it_d), "Hybrid", elapsed, max(0.0, m1 - m0), float(acc))

    # --- Stage 1: float32
    clf32 = SGDClassifier(loss='hinge', alpha=float(alpha), learning_rate='optimal',
                          tol=float(tol_single), warm_start=True, random_state=seed)
    m0 = _rss_mb(); t0 = time.perf_counter()
    it_s = _epochs_fit(clf32, Xtr, ytr,
                       epochs=cap,
                       batch_size=int(batch_size),
                       classes=classes,
                       dtype=np.float32,
                       rng_seed=seed)

    # Transfer weights to float64 model
    coef64 = clf32.coef_.astype(np.float64, copy=True)
    inter64 = clf32.intercept_.astype(np.float64, copy=True)

    clf64 = SGDClassifier(loss='hinge', alpha=float(alpha), learning_rate='optimal',
                          tol=float(tol_double), warm_start=True, random_state=seed)
    # init shapes
    init_bs = min(32, Xtr.shape[0])
    clf64.partial_fit(Xtr[:init_bs].astype(np.float64), ytr[:init_bs], classes=classes)
    clf64.coef_[:] = coef64
    clf64.intercept_[:] = inter64
    if hasattr(clf64, "t_") and hasattr(clf32, "t_"):
        clf64.t_ = clf32.t_

    # --- Stage 2: float64
    it_d = _epochs_fit(clf64, Xtr, ytr,
                       epochs=rem,
                       batch_size=int(batch_size),
                       classes=classes,
                       dtype=np.float64,
                       rng_seed=seed)
    elapsed = time.perf_counter() - t0; m1 = _rss_mb()
    acc = accuracy_score(yte, clf64.predict(Xte))
    return (tag, len(X), int(classes.size), float(tol_single), int(cap),
            int(it_s), int(it_d), "Hybrid", elapsed, max(0.0, m1 - m0), float(acc))

