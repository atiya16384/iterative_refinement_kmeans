# experiments/enet_precision.py
import time
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_squared_error
# ENET_main.py
import pathlib
import pandas as pd
import numpy as np

from datasets.utils import (
    generate_synthetic_data_en, enet_specs
)

from datasets.utils import (
    enet_columns_A, enet_columns_B
)

RESULTS_DIR = pathlib.Path("Results")
RESULTS_DIR.mkdir(exist_ok=True)

rows_A, rows_B = [], []

# Synthetic datasets
for tag, n, d, sparsity, noise, seed in enet_specs:
    X, y = generate_synthetic_data_en(
        n_samples=n, n_features=d, seed=seed,
        sparsity=sparsity, noise=noise, rho=0.5
    )

def _iter_scalar(n_iter_attr) -> int:
    """sklearn returns an int or an array; make it a single int."""
    arr = np.asarray(n_iter_attr)
    return int(arr.max()) if arr.ndim else int(arr)

# ----------------------------
# Pure double-precision run
# ----------------------------
def run_full_double(
    X, y, *,
    max_iter: int,
    tol: float,
    alpha: float,
    l1_ratio: float,
    random_state: int = 0,
):
    """
    One fp64 baseline. AOCL-safe (no warm_start/coef injection).
    Returns: (it_single=0, it_double, time_s, mem_MB, r2, mse)
    """
    X64 = np.asarray(X, dtype=np.float64)

    t0 = time.perf_counter()
    en64 = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        tol=tol,
        fit_intercept=True,
        random_state=random_state,
    )
    en64.fit(X64, y)
    elapsed = time.perf_counter() - t0

    yhat = en64.predict(X64)
    r2 = r2_score(y, yhat)
    mse = mean_squared_error(y, yhat)

    mem_MB = X64.nbytes / 1e6
    return 0, _iter_scalar(getattr(en64, "n_iter_", max_iter)), elapsed, mem_MB, r2, mse


# ----------------------------
# Hybrid: fp32 (cap) → fp64
# ----------------------------
def run_hybrid(
    X, y, *,
    max_iter_total: int,
    tol_single: float,
    tol_double: float,
    single_iter_cap: int | None,
    alpha: float,
    l1_ratio: float,
    random_state: int = 0,
):
    """
    Stage‑1 (fp32) for up to 'single_iter_cap' iters, then Stage‑2 (fp64)
    for the remaining budget. No skipping and no warm_start tricks,
    fully AOCL‑compatible.

    Returns: (it_single, it_double, time_total_s, mem_MB_peak, r2, mse)
    """
    X32 = np.asarray(X, dtype=np.float32)
    X64 = np.asarray(X, dtype=np.float64)

    # sanitize cap
    cap = max_iter_total if single_iter_cap is None else int(single_iter_cap)
    cap = int(max(0, min(cap, int(max_iter_total))))

    # ---- fp32 stage
    t0 = time.perf_counter()
    en32 = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=cap,
        tol=tol_single,
        fit_intercept=True,
        random_state=random_state,
    )
    en32.fit(X32, y)
