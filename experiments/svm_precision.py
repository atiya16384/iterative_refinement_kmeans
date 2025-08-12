# experiments/svm_precision.py
from aoclda.sklearn import skpatch; skpatch()
import numpy as np, time, psutil
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# =========================
# helpers
# =========================
def _rss_mb() -> float:
    """Resident set size in MiB."""
    return psutil.Process().memory_info().rss / (1024**2)

def _split_train_val(X, y, val_frac=0.2, seed=0):
    """Make a validation split from the *training* set (for early stopping in Exp B)."""
    n = X.shape[0]
    m = int(max(1, round(val_frac * n)))
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    val_idx = idx[:m]
    tr_idx  = idx[m:]
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]

def _epochs_fit_fixed(clf: SGDClassifier, X, y, epochs, batch_size, classes, dtype):
    """
    Fixed-budget mini-epoch training with partial_fit (no early stopping).
    dtype: np.float32 for Stage-1 (fast), np.float64 for Stage-2 (polish).
    Returns epochs actually consumed.
    """
    if epochs <= 0: return 0
    rng = np.random.RandomState(0)
    n = X.shape[0]; bs = max(1, int(batch_size))
    first = True; done = 0
    for _ in range(int(epochs)):
        idx = rng.permutation(n)
        for i in range(0, n, bs):
            sl = idx[i:i+bs]
            if sl.size == 0: continue
            Xb = X[sl].astype(dtype, copy=False)
            yb = y[sl]
            if first:
                clf.partial_fit(Xb, yb, classes=classes)
                first = False
            else:
                clf.partial_fit(Xb, yb)
        done += 1
    return done

def _epochs_fit_es(clf: SGDClassifier, X, y, Xv, yv,
                   max_epochs, batch_size, classes, dtype,
                   tol, patience):
    """
    Validation-based early stopping (for Exp B).
    Stop when val accuracy fails to improve by > tol for `patience` epochs.
    """
    if max_epochs <= 0: return 0
    rng = np.random.RandomState(0)
    n = X.shape[0]; bs = max(1, int(batch_size))
    first = True; best = -np.inf; bad = 0; done = 0
    for _ in range(int(max_epochs)):
        idx = rng.permutation(n)
        for i in range(0, n, bs):
            sl = idx[i:i+bs]
            if sl.size == 0: continue
            Xb = X[sl].astype(dtype, copy=False)
            yb = y[sl]
            if first:
                clf.partial_fit(Xb, yb, classes=classes)
                first = False
            else:
                clf.partial_fit(Xb, yb)
        done += 1
        acc_v = accuracy_score(yv, clf.predict(Xv.astype(dtype, copy=False)))
        if acc_v > best + float(tol):
            best = acc_v; bad = 0
        else:
            bad += 1
            if bad >= int(patience):
                break
    return done

# =====================================================================
# Baseline — Double (Exp A & B can both use this)
# Runs a linear SVM (hinge) with SGD entirely in float64 for max_iter epochs
# =====================================================================
def svm_double_precision(
    tag, X, y, *,
    max_iter,            # interpret as "total epochs"
    tol,                 # logged only (we drive epochs externally)
    cap=0,               # unused here; kept for schema compatibility
    alpha=1e-4,
    batch_size=1024,
    test_size=0.2,
    seed=0,
):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr); Xte = scaler.transform(Xte)
    classes = np.unique(ytr)

    clf = SGDClassifier(loss='hinge', alpha=float(alpha),
                        learning_rate='optimal',
                        early_stopping=False,
                        warm_start=True, random_state=seed)

    m0 = _rss_mb(); t0 = time.perf_counter()
    it_d = _epochs_fit_fixed(clf, Xtr, ytr,
                             epochs=int(max_iter),
                             batch_size=int(batch_size),
                             classes=classes, dtype=np.float64)
    elapsed = time.perf_counter() - t0; m1 = _rss_mb()
    acc = accuracy_score(yte, clf.predict(Xte))
    return (
        tag, len(X), int(classes.size), float(tol), int(cap),
        0, int(it_d), "Double", elapsed, max(0.0, m1 - m0), float(acc)
    )

# ======================================================================================
# Hybrid — Simple two-stage IR
# Stage-1: float32 for `single_iter_cap` epochs (or early-stop if early_stop=True)
# Stage-2: fixed small float64 "polish" (polish_epochs)
# ======================================================================================
def svm_hybrid_precision(
    tag, X, y, *,
    max_iter_total,          # overall budget label (not enforced unless you want)
    tol_single,              # logged; used as ES tol only if early_stop=True
    tol_double,              # logged (Stage-2 is fixed polish)
    single_iter_cap,         # Stage-1 epochs in float32 (cap)
    alpha=1e-4,
    batch_size=1024,
    test_size=0.2,
    seed=0,
    polish_epochs=5,         # small float64 polish
    early_stop=False,        # if True, Stage-1 uses validation early stopping
    val_frac=0.2,
    patience=2,
):
    Xtr_full, Xte, ytr_full, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(Xtr_full)
    Xtr_full = scaler.transform(Xtr_full); Xte = scaler.transform(Xte)
    classes = np.unique(ytr_full)

    # Make a val split only if we need early stopping
    if early_stop:
        Xtr, ytr, Xv, yv = _split_train_val(Xtr_full, ytr_full, val_frac=val_frac, seed=seed)
    else:
        Xtr, ytr = Xtr_full, ytr_full
        Xv = yv = None

    cap = int(max(0, int(single_iter_cap)))

    # ---- Stage-1 (float32) ----
    it_s = 0
    coef64 = intercept64 = None
    if cap > 0:
        clf32 = SGDClassifier(loss='hinge', alpha=float(alpha),
                              learning_rate='optimal',
                              early_stopping=False,
                              warm_start=True, random_state=seed, average=True)
        if early_stop:
            it_s = _epochs_fit_es(clf32, Xtr, ytr, Xv, yv,
                                  max_epochs=cap,
                                  batch_size=int(batch_size),
                                  classes=classes, dtype=np.float32,
                                  tol=float(tol_single), patience=int(patience))
        else:
            it_s = _epochs_fit_fixed(clf32, Xtr, ytr,
                                     epochs=cap,
                                     batch_size=int(batch_size),
                                     classes=classes, dtype=np.float32)
        coef64 = clf32.coef_.astype(np.float64, copy=True)
        intercept64 = clf32.intercept_.astype(np.float64, copy=True)

    # ---- Stage-2 (float64 polish) ----
    clf64 = SGDClassifier(loss='hinge', alpha=float(alpha),
                          learning_rate='optimal',
                          early_stopping=False,
                          warm_start=True, random_state=seed, average=True)

    m0 = _rss_mb(); t0 = time.perf_counter()

    # tiny init to allocate arrays
    init_bs = min(32, Xtr_full.shape[0])
    clf64.partial_fit(Xtr_full[:init_bs].astype(np.float64),
                      ytr_full[:init_bs], classes=classes)
    if coef64 is not None:
        clf64.coef_[:] = coef64
        clf64.intercept_[:] = intercept64

    it_d = _epochs_fit_fixed(clf64, Xtr_full, ytr_full,
                             epochs=int(polish_epochs),
                             batch_size=int(batch_size),
                             classes=classes, dtype=np.float64)

    elapsed = time.perf_counter() - t0; m1 = _rss_mb()
    acc = accuracy_score(yte, clf64.predict(Xte))
    return (
        tag, len(X), int(classes.size), float(tol_single), int(cap),
        int(it_s), int(it_d), "Hybrid", elapsed, max(0.0, m1 - m0), float(acc)
    )

