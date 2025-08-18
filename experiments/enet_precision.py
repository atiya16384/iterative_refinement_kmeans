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
    alpha, l1_ratio,
    max_iter_total,
    single_iter_cap,
    tol_single, tol_double
):
    X32 = np.asarray(X, dtype=np.float32)
    X64 = np.asarray(X, dtype=np.float64)

    # cap rules
    cap = max_iter_total if single_iter_cap is None else int(single_iter_cap)
    cap = int(max(1, min(cap, int(max_iter_total))))

    # ---- fp32 stage ----
    t0 = time.perf_counter()
    en32 = ElasticNet(
        alpha=alpha, l1_ratio=l1_ratio,
        max_iter=cap, tol=tol_single,
        warm_start=False, selection="cyclic"
    )
    en32.fit(X32, y)
    t_single = time.perf_counter() - t0
    it_single = int(en32.n_iter_)

    # ---- fp64 refinement (true warm-start) ----
    remaining = max(1, int(max_iter_total) - it_single)
    en64 = ElasticNet(
        alpha=alpha, l1_ratio=l1_ratio,
        max_iter=remaining, tol=tol_double,
        warm_start=True, selection="cyclic"
    )
    # Seed state (supported by sklearn + AOCL patch):
    en64.coef_ = en32.coef_.astype(np.float64, copy=True)
    en64.intercept_ = float(en32.intercept_)

    t1 = time.perf_counter()
    en64.fit(X64, y)        # continues from seeded coef_/intercept_
    t_double = time.perf_counter() - t1
    it_double = int(en64.n_iter_)

    yhat = en64.predict(X64)
    r2 = r2_score(y, yhat)
    mse = mean_squared_error(y, yhat)

    # report peak footprint (realistic): the larger of {X32, X64}
    mem_MB = max(_mem_MB(X32), _mem_MB(X64))
    return it_single, it_double, (t_single + t_double), mem_MB, r2, mse
