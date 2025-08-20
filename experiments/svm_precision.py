# experiments/svm_precision.py
import numpy as np, time, psutil
from sklearn.svm import SVC, NuSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pathlib, pandas as pd
from visualisations.SVM_visualisations import SVMVisualizer; 
from datasets.utils import generate_synthetic_data, synth_specs

RESULTS_DIR = pathlib.Path("Results")
RESULTS_DIR.mkdir(exist_ok=True)

def print_summary(path, group_by):
    df = pd.read_csv(path)
    print(f"\n==== SUMMARY: {path.name.upper()} ====")
    print(df.groupby(group_by)[['Accuracy','Time','Memory_MB']].mean())

    # Datasets
    for tag, n, d, c, seed in synth_specs:
        X, y = generate_synthetic_data(n, d, c, seed)


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

