import numpy as np
from aoclda.svm import svc
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import time
import pandas as pd


# --- Synthetic dataset ---
def make_synthetic_data(n_samples=2000, n_features=30, random_state=42):
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_informative=int(0.75*n_features),
                               n_redundant=int(0.25*n_features),
                               random_state=random_state)
    y = 2*y - 1   # convert {0,1} to {-1,+1}
    return X, y


# --- Double precision SVM (full solve) ---
def svm_double_precision(X, y, C=1.0, kernel="linear", tol=1e-5, max_iter=5000):
    model = svc(kernel=kernel, precision="double", tol=tol, C=C, max_iter=max_iter)
    model.fit(X, y)
    return model


# --- Hybrid SVM: single precision + iterative refinement ---
def svm_hybrid_precision(X, y, C=1.0, kernel="linear", n_refine=3, max_iter=2000):
    # Step 1: initial training in single precision
    model_sp = svc(kernel=kernel, precision="single", C=C, tol=1e-3, max_iter=max_iter)
    model_sp.fit(X, y)

    # Cast support vectors & coefficients to double
    dual_coef = model_sp.dual_coef_.astype(np.float64)
    support_vectors = model_sp.support_vectors_.astype(np.float64)

    # Step 2: iterative refinement in double precision
    model_dp = svc(kernel=kernel, precision="double", C=C, tol=1e-5, max_iter=max_iter)
    model_dp.fit(X, y, x0=dual_coef)  # warm start from single precision

    for _ in range(n_refine - 1):
        model_dp.fit(X, y, x0=model_dp.dual_coef_)  # re-refine

    return model_dp


# --- Experiment runner ---
def run_experiments():
    X, y = make_synthetic_data()

    results = []

    Cs = [0.1, 1.0, 10.0]
    kernels = ["linear", "rbf"]
    refines = [1, 3, 5]

    # --- Double precision ---
    for C in Cs:
        for kernel in kernels:
            t0 = time.time()
            model = svm_double_precision(X, y, C=C, kernel=kernel)
            t1 = time.time()
            acc = accuracy_score(y, model.predict(X))
            results.append({
                "method": "double",
                "C": C,
                "kernel": kernel,
                "n_refine": None,
                "accuracy": acc,
                "time_sec": t1 - t0
            })

    # --- Hybrid ---
    for C in Cs:
        for kernel in kernels:
            for n_refine in refines:
                t0 = time.time()
                model = svm_hybrid_precision(X, y, C=C, kernel=kernel, n_refine=n_refine)
                t1 = time.time()
                acc = accuracy_score(y, model.predict(X))
                results.append({
                    "method": "hybrid",
                    "C": C,
                    "kernel": kernel,
                    "n_refine": n_refine,
                    "accuracy": acc,
                    "time_sec": t1 - t0
                })

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = run_experiments()
    print(df)# experiments/svm_precision.py
import numpy as np, time, psutil
from sklearn.svm import SVC, NuSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pathlib, pandas as pd
from visualisations.SVM_visualisations import SVMVisualizer; 
from datasets.utils import generate_synthetic_data, synth_specs

RESULTS_DIR = pathlib.Path("../Results")
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

