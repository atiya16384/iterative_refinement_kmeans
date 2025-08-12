# experiments/svm_precision.py
from aoclda.sklearn import skpatch; skpatch()
import numpy as np, time, psutil
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024**2)

def _epochs_fit(clf: SGDClassifier, Z, y, epochs, batch_size, classes, dtype):
    """
    Mini-epoch training loop using partial_fit so we can warm-start.
    Works for both float32 and float64 features (dtype controls precision).
    """
    if epochs <= 0: return 0
    rng = np.random.RandomState(0)
    n = Z.shape[0]
    bs = max(1, int(batch_size))
    iters = 0
    # first call of partial_fit must include 'classes'
    first = True
    for _ in range(int(epochs)):
        # one pass (shuffle batches)
        idx = rng.permutation(n)
        for i in range(0, n, bs):
            sl = idx[i:i+bs]
            if sl.size == 0: continue
            Xb = Z[sl].astype(dtype, copy=False)
            yb = y[sl]
            if first:
                clf.partial_fit(Xb, yb, classes=classes)
                first = False
            else:
                clf.partial_fit(Xb, yb)
        iters += 1
    return iters  # epochs consumed

# -----------------------
# Baseline: "Double"
# -----------------------
def rff_sgd_double_precision(
    tag, X, y, *,
    total_epochs,             # total optimization budget (all in float64)
    tol=1e-4,                 # SGD early-stopping tol (validation-based)
    alpha=1e-4,               # L2 reg
    gamma='scale',            # RBF gamma (float or 'scale'/'auto' like SVC)
    n_components=2048,        # RFF feature dimension (bigger = closer to RBF)
    batch_size=2048,
    test_size=0.2,
    seed=0
):
    """
    Double baseline: Standardize -> RFF (float64) -> SGDClassifier(loss='hinge') in float64.
    """
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr); Xte = scaler.transform(Xte)

    # gamma choices mimic SVC: 'scale' = 1/(n_features*var(X)), 'auto' = 1/n_features
    if gamma == 'scale':
        g = 1.0 / (Xtr.shape[1] * Xtr.var())
    elif gamma == 'auto':
        g = 1.0 / Xtr.shape[1]
    else:
        g = float(gamma)

    rff = RBFSampler(gamma=g, n_components=int(n_components), random_state=seed)
    Ztr = rff.fit_transform(Xtr).astype(np.float64, copy=False)
    Zte = rff.transform(Xte).astype(np.float64, copy=False)

    clf = SGDClassifier(loss='hinge', alpha=float(alpha), learning_rate='optimal',
                        tol=float(tol), warm_start=True, random_state=seed)

    m0 = _rss_mb(); t0 = time.perf_counter()
    # Train for 'total_epochs' in float64
    classes = np.unique(ytr)
    it_d = _epochs_fit(clf, Ztr, ytr, epochs=int(total_epochs),
                       batch_size=int(batch_size), classes=classes, dtype=np.float64)
    elapsed = time.perf_counter() - t0; m1 = _rss_mb()

    acc = accuracy_score(yte, clf.predict(Zte))
    # Return schema: (tag, DatasetSize, NumClasses, Tolerance, Cap, iter_single, iter_double, Suite, Time, Memory_MB, Accuracy)
    return (tag, len(X), int(classes.size), float(tol), 0, 0, int(it_d),
            "Double", elapsed, max(0.0, m1 - m0), float(acc))

# -----------------------
# Hybrid IR: float32 -> float64
# -----------------------
def rff_sgd_hybrid_ir(
    tag, X, y, *,
    total_epochs,             # total epochs budget
    cap_epochs,               # Stage-1 epochs in float32 (the "cap")
    tol_single=1e-3,          # tol used in Stage-1 (SGD tol; loose is fine)
    tol_double=1e-4,          # tol in Stage-2 (tighter)
    alpha=1e-4,               # L2 reg
    gamma='scale',
    n_components=2048,
    batch_size=2048,
    test_size=0.2,
    seed=0
):
    """
    True iterative refinement with warm-start:
      - Standardize -> RFF
      - Stage-1: train for cap_epochs using float32 features (fast)
      - Stage-2: cast features to float64 and continue training for (total - cap) epochs
    """
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr); Xte = scaler.transform(Xte)

    if gamma == 'scale':
        g = 1.0 / (Xtr.shape[1] * Xtr.var())
    elif gamma == 'auto':
        g = 1.0 / Xtr.shape[1]
    else:
        g = float(gamma)

    rff = RBFSampler(gamma=g, n_components=int(n_components), random_state=seed)
    # Build both dtypes once to avoid recomputing random features in stage-2
    Ztr32 = rff.fit_transform(Xtr).astype(np.float32, copy=False)
    Zte64 = rff.transform(Xte).astype(np.float64, copy=False)  # prediction in double
    Ztr64 = Ztr32.astype(np.float64, copy=False)               # reuse same features

    classes = np.unique(ytr)
    cap = int(max(0, min(int(cap_epochs), int(total_epochs))))
    rem = int(max(0, int(total_epochs) - cap))

    # One model, warm-started across stages
    clf = SGDClassifier(loss='hinge', alpha=float(alpha), learning_rate='optimal',
                        tol=float(tol_single), warm_start=True, random_state=seed)

    m0 = _rss_mb(); t0 = time.perf_counter()

    # Stage-1 (float32)
    it_s = _epochs_fit(clf, Ztr32, ytr, epochs=cap,
                       batch_size=int(batch_size), classes=classes, dtype=np.float32)

    # Update tol for double stage (polish)
    clf.tol = float(tol_double)

    # Stage-2 (float64), continue same model
    it_d = _epochs_fit(clf, Ztr64, ytr, epochs=rem,
                       batch_size=int(batch_size), classes=classes, dtype=np.float64)

    elapsed = time.perf_counter() - t0; m1 = _rss_mb()
    acc = accuracy_score(yte, clf.predict(Zte64))

    return (tag, len(X), int(classes.size), float(tol_single), int(cap),
            int(it_s), int(it_d), "Hybrid", elapsed, max(0.0, m1 - m0), float(acc))
