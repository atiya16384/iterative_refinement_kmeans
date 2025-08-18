# experiments/enet_experiments.py
from experiments.enet_precision import run_full_double, run_hybrid


def _cfg(config: dict, key: str, fallback_key: str | None = None):
    """Small helper to tolerate slightly different key names."""
    if key in config:
        return config[key]
    if fallback_key is not None and fallback_key in config:
        return config[fallback_key]
    raise KeyError(f"Missing '{key}' in config (also looked for '{fallback_key}')")


# ============================================================
# Experiment A: cap sweep (fixed single tolerance)
# ============================================================
def run_experiment_A(ds_name: str, X, y, d: int, config: dict) -> list[list]:
    rows: list[list] = []

    tol_fixed  = _cfg(config, "tol_single_A", "tol_fixed_A")  # single tol
    max_iter   = _cfg(config, "max_iter_A", "max_iter")
    cap_grid   = _cfg(config, "cap_grid", "CapGrid")
    alpha      = _cfg(config, "alpha", "enet_alpha")
    l1_ratio   = _cfg(config, "l1_ratio", "enet_l1_ratio")

    n = len(X)
    p = getattr(X, "shape", [None, d])[1]

    # fp64 baseline once (uses the SAME tolerance)
    it_s_b, it_d_b, t_b, mem_b, r2_b, mse_b = run_full_double(
        X, y,
        max_iter=max_iter,
        tol=tol_fixed,               # <— same tol
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=0,
    )

    for cap in cap_grid:
        rows.append([ds_name, n, p, "Double", cap, tol_fixed,
                     it_s_b, it_d_b, "ENet_ExpA", t_b, mem_b, r2_b, mse_b])

        it_s, it_d, t, mem, r2, mse = run_hybrid(
            X, y,
            max_iter_total=max_iter,
            tol_single=tol_fixed,     # <— same tol
            tol_double=tol_fixed,     # <— same tol
            single_iter_cap=cap,
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=0,
        )
        rows.append([ds_name, n, p, "Hybrid", cap, tol_fixed,
                     it_s, it_d, "ENet_ExpA", t, mem, r2, mse])

    return rows

# ============================================================
# Experiment B: fp32 tolerance sweep (no cap; use full budget)
# ============================================================
def run_experiment_B(ds_name: str, X, y, d: int, config: dict) -> list[list]:
    """
    Sweep fp32 tolerance; no cap (let single use whatever it needs within budget).
    """
    rows: list[list] = []

    tol_grid  = _cfg(config, "tol_single_grid", "TolGrid")
    tol_double = _cfg(config, "tol_double_B", "tol_double")
    max_iter   = _cfg(config, "max_iter_B", "max_iter")
    alpha      = _cfg(config, "alpha", "enet_alpha")
    l1_ratio   = _cfg(config, "l1_ratio", "enet_l1_ratio")

    n = len(X)
    p = getattr(X, "shape", [None, d])[1]

    # fp64 baseline once (repeat per tol for consistent plots)
    it_s_b, it_d_b, t_b, mem_b, r2_b, mse_b = run_full_double(
        X, y,
        max_iter=max_iter,
        tol=tol_double,
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=0,
    )

    for tol_single in tol_grid:
        rows.append([
            ds_name, n, p,
            "Double", tol_single,
            it_s_b, it_d_b, "ENet_ExpB",
            t_b, mem_b, r2_b, mse_b
        ])

        it_s, it_d, t, mem, r2, mse = run_hybrid(
            X, y,
            max_iter_total=max_iter,
            tol_single=tol_single,
            tol_double=tol_double,
            single_iter_cap=None,  # no cap
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=0,
        )
        rows.append([
            ds_name, n, p,
            "Hybrid", tol_single,
            it_s, it_d, "ENet_ExpB",
            t, mem, r2, mse
        ])

    return rows
