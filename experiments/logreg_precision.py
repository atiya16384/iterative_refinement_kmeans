# experiments/logreg_precision.py
# experiments/logreg_precision.py
import time
import numpy as np
from aoclda.linear_model import linmod
from sklearn.metrics import roc_auc_score
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
    acc = float(roc_auc_score(yte, clf.predict(Xte64)))
    mem_MB = _mem_mb_from_arrays(Xtr64, Xte64)

    return clf, 0, it_double, elapsed, mem_MB, acc

# ---------------- hybrid: fp32 probe -> fp64 fresh ----------------
def run_hybrid(X, y, n_classes, max_iter_total, tol_single, tol_double, single_iter_cap):

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


    t0=time.perf_counter()

    
  

    # Evaluate on test
    acc = float(roc_auc_score(yte, clf64.predict(Xte64)))

    # Simple peak-ish memory proxy (scaled arrays we held concurrently)
    mem_MB = _mem_mb_from_arrays(Xtr32, Xtr64, Xte64)

    return it_single, it_double, (t_single + t_double), mem_MB, acc


    
#     # --- Switching Logic ---
#     grad_norm = np.linalg.norm(clf_fp32.coef_)
#     needs_refinement = grad_norm > switch_tol
    
