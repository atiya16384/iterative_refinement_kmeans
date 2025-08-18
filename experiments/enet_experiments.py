# experiments/enet_experiments.py
from experiments.enet_precision import run_full_double, run_hybrid


def run_experiment_A(ds_name, X, y, n_features, config):
    """
    Cap sweep at fixed fp32 tol; compare Hybrid vs Double with the same total budget.
    """
    rows = []

    alpha      = config.get("alpha", 1e-3)
    l1_ratio   = config.get("l1_ratio", 0.5)      # 1.0 => Lasso
    max_iter   = config["max_iter_A"]
    tol_double = config.get("tol_double_A", 1e-6)
    tol_single = config["tol_fixed_A"]

    # baseline double once per dataset
    _, it_s_b, it_d_b, t_b, mem_b, r2_b, mse_b = run_full_double(
        X, y, alpha, l1_ratio, max_iter=max_iter, tol=tol_double
    )

    for cap in config["cap_grid"]:
        rows.append([
            ds_name, n_features, "Double", cap, tol_single,
            it_s_b, it_d_b, "ENet_ExpA", t_b, mem_b, r2_b, mse_b
        ])

        it_s, it_d, t, mem, r2, mse = run_hybrid(
            X, y, alpha, l1_ratio,
            max_iter_total=max_iter,
            single_iter_cap=cap,
            tol_single=tol_single, tol_double=tol_double
        )
        rows.append([
            ds_name, n_features, "Hybrid", cap, tol_single,
            it_s, it_d, "ENet_ExpA", t, mem, r2, mse
        ])
    return rows


def run_experiment_B(ds_name, X, y, n_features, config):
    """
    fp32 tolerance sweep (no cap). Hybrid uses all budget in fp32 until convergence,
    then uses leftover in fp64.
    """
    rows = []
    alpha      = config.get("alpha", 1e-3)
    l1_ratio   = config.get("l1_ratio", 0.5)
    max_iter   = config["max_iter_B"]
    tol_double = config["tol_double_B"]

    for tol_single in config["tol_single_grid"]:
        # paired double baseline for this budget
        _, it_s_b, it_d_b, t_b, mem_b, r2_b, mse_b = run_full_double(
            X, y, alpha, l1_ratio, max_iter=max_iter, tol=tol_double
        )
        rows.append([
            ds_name, n_features, "Double", tol_single,
            it_s_b, it_d_b, "ENet_ExpB", t_b, mem_b, r2_b, mse_b
        ])

        it_s, it_d, t, mem, r2, mse = run_hybrid(
            X, y, alpha, l1_ratio,
            max_iter_total=max_iter,
            single_iter_cap=None,             # let fp32 run until it stops
            tol_single=tol_single, tol_double=tol_double
        )
        rows.append([
            ds_name, n_features, "Hybrid", tol_single,
            it_s, it_d, "ENet_ExpB", t, mem, r2, mse
        ])
    return rows
