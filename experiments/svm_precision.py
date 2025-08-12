# experiments/svm_precision.py
from aoclda.sklearn import skpatch; skpatch()
import numpy as np, time, psutil
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# -----------------------
# helpers
# -----------------------
def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024**2)

def _epochs_fit(clf: SGDClassifier, X, y, epochs, batch_size, classes, dtype):
    """
    Mini-epoch training loop using partial_fit so we can warm-start.
    dtype controls precision: np.float32 for fast stage, np.float64 to polish.
    Returns 'epochs' consumed (for logging).
    """
    if epochs <= 0:
        return 0
    rng = np.random.RandomState(0)
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
def svm_double_precision(
    tag, X, y, *,
    max_iter,            # interpret as 'total_epochs'
    tol,
    cap=0,               # unused here; kept for schema compatibility
    alpha=1e-4,
    batch_size=2048,
    test_size=0.2,
    seed=0,
):
    """
    Linear SVM (hinge loss) trained with SGD entirely in float64.
    Returns the standard 11-tuple used by your CSV/plots.
    """
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr); Xte = scaler.transform(Xte)

    classes = np.unique(ytr)

    clf = SGDClassifier(
        loss='hinge',
        alpha=float(alpha),
        learning_rate='optimal',
        tol=float(tol),
        warm_start=True,
        random_state=seed,
    )

    m0 = _rss_mb(); t0 = time.perf_counter()
    it_d = _epochs_fit(
        clf, Xtr, ytr,
        epochs=int(max_iter),
        batch_size=int(batch_size),
        classes=classes,
        dtype=np.float64,
    )
    elapsed = time.perf_counter() - t0; m1 = _rss_mb()

    acc = accuracy_score(yte, clf.predict(Xte))
    return (
        tag, len(X), int(classes.size), float(tol), int(cap),
        0, int(it_d), "Double", elapsed, max(0.0, m1 - m0), float(acc)
    )

# -----------------------
# Hybrid: float32 -> float64 (true IR)
# -----------------------
def svm_hybrid_precision(
    tag, X, y, *,
    max_iter_total,      # total epochs budget
    tol_single,          # SGD tol during float32 stage
    tol_double,          # SGD tol during float64 stage
    single_iter_cap,     # number of epochs to run in float32
    alpha=1e-4,
    batch_size=2048,
    test_size=0.2,
    seed=0,
):
    """
    Iterative refinement for linear SVM:
      Stage-1 (float32): run 'single_iter_cap' epochs quickly
      Stage-2 (float64): continue training the SAME model for the remaining epochs
    """
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr); Xte = scaler.transform(Xte)

    classes = np.unique(ytr)

    cap = int(max(0, min(int(single_iter_cap), int(max_iter_total))))
    rem = int(max(0, int(max_iter_total) - cap))

    clf = SGDClassifier(
        loss='hinge',
        alpha=float(alpha),
        learning_rate='optimal',
        tol=float(tol_single),
        warm_start=True,
        random_state=seed,
    )

    m0 = _rss_mb(); t0 = time.perf_counter()

    # Stage-1 (float32) — fast, coarse
    it_s = _epochs_fit(
        clf, Xtr, ytr,
        epochs=cap,
        batch_size=int(batch_size),
        classes=classes,
        dtype=np.float32,
    )

    # Stage-2 (float64) — polish
    clf.tol = float(tol_double)
    it_d = _epochs_fit(
        clf, Xtr, ytr,
        epochs=rem,
        batch_size=int(batch_size),
        classes=classes,
        dtype=np.float64,
    )

    elapsed = time.perf_counter() - t0; m1 = _rss_mb()

    acc = accuracy_score(yte, clf.predict(Xte))
    return (
        tag, len(X), int(classes.size), float(tol_single), int(cap),
        int(it_s), int(it_d), "Hybrid", elapsed, max(0.0, m1 - m0), float(acc)
    )
