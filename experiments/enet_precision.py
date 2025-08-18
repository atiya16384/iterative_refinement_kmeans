# experiments/enet_precision.py
import time
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_squared_error


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
    t_single = time.perf_counter() - t0
    it_single = _iter_scalar(getattr(en32, "n_iter_", cap))

    # ---- fp64 stage (fresh fit; no weight transfer)
    remaining = max(1, int(max_iter_total) - it_single)
    t1 = time.perf_counter()
    en64 = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=remaining,
        tol=tol_double,
        fit_intercept=True,
        random_state=random_state,
    )
    en64.fit(X64, y)
    t_double = time.perf_counter() - t1
    it_double = _iter_scalar(getattr(en64, "n_iter_", remaining))

    yhat64 = en64.predict(X64)
    r2 = r2_score(y, yhat64)
    mse = mean_squared_error(y, yhat64)

    # peak memory approx: we keep both views during the run
    mem_MB = max(X32.nbytes, X64.nbytes) / 1e6
    return it_single, it_double, (t_single + t_double), mem_MB, r2, mse

