# experiments/logreg_precision.py
# experiments/logreg_precision.py
import time
import numpy as np
from aoclda.linear_model import linmod
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pathlib
import pandas as pd

from datasets.utils import generate_synthetic_data, synth_specs, lr_columns_A, lr_columns_B
from visualisations.LOGREG_visualisations import LogisticVisualizer

RESULTS_DIR = pathlib.Path("Results")
RESULTS_DIR.mkdir(exist_ok=True)

# Config
rows_A, rows_B = [], []

# Run synthetic datasets
for tag, n, d, k, seed in synth_specs:
    X, y = generate_synthetic_data(n_samples=n, n_features=d, n_clusters=k, random_state=seed)
    n_classes = len(set(y))




# ---------------- helpers ----------------
def _iter_scalar(n_iter_attr) -> int:
    """Normalize scikit/AOCL's n_iter_ (int or array) to a single int."""
    arr = np.asarray(n_iter_attr)
    return int(arr.max()) if arr.ndim else int(arr)

def _mem_mb_from_arrays(*arrays) -> float:
    """Simple peak-ish memory proxy: max of involved array byte sizes."""
    if not arrays:
        return 0.0
    return float(max(getattr(a, "nbytes", 0) for a in arrays) / 1e6)

# ---------------- baseline: full fp64 ----------------
def run_full_double(X, y, n_classes, max_iter, tol):
    """
    Train AOCL logistic regression fully in double precision on the train set,
    evaluate on a held-out test set. Returns:
      (model, it_single=0, it_double, time_s, mem_MB, accuracy)
    """
    # Labels as integers (classification); check class count.
    y = np.asarray(y).astype(int, copy=False)
    if n_classes is None:
        n_classes = int(np.unique(y).size)
    else:
        assert int(n_classes) == int(np.unique(y).size), \
            f"n_classes mismatch: got {n_classes}, found {np.unique(y).size}"

    # Split + scale
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    scaler = StandardScaler().fit(Xtr)
    Xtr64 = scaler.transform(Xtr).astype(np.float64, copy=False)
    Xte64 = scaler.transform(Xte).astype(np.float64, copy=False)

    # Fit (double)
    t0 = time.perf_counter()
    clf = linmod(
        "logistic",
        solver="auto",
        max_iter=int(max_iter),
        precision="double",
    )
    clf.fit(Xtr64, ytr, tol=float(tol))
    elapsed = time.perf_counter() - t0

    # Metrics
    it_double = _iter_scalar(getattr(clf, "n_iter_", 0))
    acc = float(accuracy_score(yte, clf.predict(Xte64)))
    mem_MB = _mem_mb_from_arrays(Xtr64, Xte64)

    return clf, 0, it_double, elapsed, mem_MB, acc

# ---------------- hybrid: fp32 probe -> fp64 fresh ----------------
def run_hybrid(X, y, n_classes, max_iter_total, tol_single, tol_double, single_iter_cap):
    """
    Stage-1: fp32 probe (capped iterations) on train set.
    Stage-2: fp64 from scratch for remaining budget on the same train set.
    Evaluate on held-out test set.

    Returns the 5-tuple your experiment code expects:
      (it_single, it_double, time_total_s, mem_MB_peak, accuracy)
    """
    # Labels as integers (classification); check class count.
    y = np.asarray(y).astype(int, copy=False)
    if n_classes is None:
        n_classes = int(np.unique(y).size)
    else:
        assert int(n_classes) == int(np.unique(y).size), \
            f"n_classes mismatch: got {n_classes}, found {np.unique(y).size}"

    # Split + scale once
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    scaler = StandardScaler().fit(Xtr)
    Xtr32 = scaler.transform(Xtr).astype(np.float32, copy=False)
    Xtr64 = Xtr32.astype(np.float64, copy=False)   # exact same scaling/path
    Xte64 = scaler.transform(Xte).astype(np.float64, copy=False)

    # Cap handling
    if single_iter_cap is None:
        cap = int(max_iter_total)
    else:
        cap = int(max(0, min(int(single_iter_cap), int(max_iter_total))))

    # --- Stage 1: fp32 probe ---
    t0 = time.perf_counter()
    clf32 = linmod(
        "logistic",
        solver="auto",
        max_iter=int(cap),
        precision="single",
    )
    clf32.fit(Xtr32, ytr, tol=float(tol_single))
    t_single = time.perf_counter() - t0
    it_single = _iter_scalar(getattr(clf32, "n_iter_", 0))

    # --- Stage 2: fp64 fresh with remaining budget ---
    remaining = max(1, int(max_iter_total) - it_single)
    t1 = time.perf_counter()
    clf64 = linmod(
        "logistic",
        solver="auto",
        max_iter=int(remaining),
        precision="double",
    )
    clf64.fit(Xtr64, ytr, tol=float(tol_double))
    t_double = time.perf_counter() - t1
    it_double = _iter_scalar(getattr(clf64, "n_iter_", 0))

    # Evaluate on test
    acc = float(accuracy_score(yte, clf64.predict(Xte64)))

    # Simple peak-ish memory proxy (scaled arrays we held concurrently)
    mem_MB = _mem_mb_from_arrays(Xtr32, Xtr64, Xte64)

    return it_single, it_double, (t_single + t_double), mem_MB, acc
# def adaptive_mixed_precision_lr(X, y, switch_tol=1e-3, max_iter=1000):
#     """AOCL-compatible adaptive mixed precision logistic regression"""
#     # Precision conversion
#     X_fp32 = np.asarray(X, dtype=np.float32, order='C')
#     X_fp64 = X.astype(np.float64, copy=False) if X.dtype != np.float64 else X
    
#     # --- FP32 Phase ---
#     clf_fp32 = LogisticRegression(
#         solver='lbfgs',
#         max_iter=max_iter,
#         warm_start=False  # Explicitly disabled for AOCL
#     )
#     t_fp32_start = time.perf_counter()
#     clf_fp32.fit(X_fp32, y)
#     t_fp32 = time.perf_counter() - t_fp32_start
    
#     # --- Switching Logic ---
#     grad_norm = np.linalg.norm(clf_fp32.coef_)
#     needs_refinement = grad_norm > switch_tol
    
#     if not needs_refinement:
#         # Early return if FP32 is sufficient
#         clf_fp32._used_mixed_precision = False
#         clf_fp32._time_fp32 = t_fp32
#         clf_fp32._time_fp64 = 0
#         clf_fp32._final_precision = 'fp32'
#         return clf_fp32
    
#     # --- FP64 Refinement Phase ---
#     t_fp64_start = time.perf_counter()
#     clf_fp64 = LogisticRegression(
#         solver='lbfgs',
#         max_iter=max_iter,
#         warm_start=False,
#         # AOCL-compatible initialization
#         coef_init=clf_fp32.coef_.astype(np.float64),
#         intercept_init=clf_fp32.intercept_.astype(np.float64)
#     )
#     clf_fp64.fit(X_fp64, y)
#     t_fp64 = time.perf_counter() - t_fp64_start
    
#     # --- Instrumentation ---
#     clf_fp64._used_mixed_precision = True
#     clf_fp64._time_fp32 = t_fp32
#     clf_fp64._time_fp64 = t_fp64
#     clf_fp64._final_precision = 'mixed'
    
#     return clf_fp64
