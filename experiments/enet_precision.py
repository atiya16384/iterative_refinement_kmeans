# experiments/enet_precision.py
import time
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_squared_error

def _mem_MB(*arrs) -> float:
    return sum(getattr(a, "nbytes", 0) for a in arrs) / 1e6

# ---------------------------
# Full double-precision run
# ---------------------------
def run_full_double(
    X, y,
    alpha, l1_ratio,
    max_iter, tol
):
    X64 = np.asarray(X, dtype=np.float64)

    t0 = time.perf_counter()
    en = ElasticNet(
        alpha=alpha, l1_ratio=l1_ratio,
        max_iter=max_iter, tol=tol,
        warm_start=False, selection="cyclic"
    )
    en.fit(X64, y)
    t = time.perf_counter() - t0

    yhat = en.predict(X64)
    r2 = r2_score(y, yhat)
    mse = mean_squared_error(y, yhat)
    mem = _mem_MB(X64)

    # elastic net exposes n_iter_ (int)
    return en, 0, int(en.n_iter_), t, mem, r2, mse

# -------------------------------------------------
# Hybrid: float32 (capped)  → warm-started float64
# -------------------------------------------------
def run_hybrid(
    X, y,
    max_iter_total,
    tol_single, tol_double,
    single_iter_cap,
    alpha, l1_ratio,
    random_state=0
):


    # Prepare float32 and float64 versions
    X32 = np.asarray(X, dtype=np.float32)
    X64 = np.asarray(X, dtype=np.float64)

    cap = max_iter_total if single_iter_cap is None else int(single_iter_cap)
    cap = int(max(0, min(cap, int(max_iter_total))))

    # ---- fp32 stage ----
    t0 = time.perf_counter()
    en32 = ElasticNet(
        alpha=alpha, l1_ratio=l1_ratio,
        max_iter=cap, tol=tol_single,
        fit_intercept=True, random_state=random_state
    )
    en32.fit(X32, y)
    t_single = time.perf_counter() - t0
    it_single = int(getattr(en32, "n_iter_", cap))

    # ---- fp64 stage (fresh, always run) ----
    remaining = max(1, int(max_iter_total) - it_single)
    t1 = time.perf_counter()
    en64 = ElasticNet(
        alpha=alpha, l1_ratio=l1_ratio,
        max_iter=remaining, tol=tol_double,
        fit_intercept=True, random_state=random_state
    )
    en64.fit(X64, y)
    t_double = time.perf_counter() - t1
    it_double = int(getattr(en64, "n_iter_", remaining))

    # Evaluate final fp64 model
    yhat64 = en64.predict(X64)
    r2_64  = r2_score(y, yhat64)
    mse_64 = mean_squared_error(y, yhat64)

    mem_MB = max(X32.nbytes, X64.nbytes) / 1e6
    return it_single, it_double, (t_single + t_double), mem_MB, r2_64, mse_64

