# experiments/logreg_experiments.py
from experiments.logreg_precision import run_full_double, run_hybrid

def run_experiment_A(ds_name, X, y, n_classes, config):
    rows = []
    tol_single = config["tol_fixed_A"]
    tol_double = config["tol_double_B"]
    max_iter   = config["max_iter_A"]

    # baseline (pure double)
    _, it_s_b, it_d_b, t_b, mem_b, acc_b = run_full_double(X, y, n_classes, max_iter, tol=tol_double)

    for cap in config["cap_grid"]:
        rows.append([ds_name, len(X), n_classes, "Double", cap, tol_single,
                     it_s_b, it_d_b, "LR_ExpA", t_b, mem_b, acc_b])

        it_s, it_d, t, mem, acc = run_hybrid(
            X, y, n_classes,
            max_iter_total=max_iter,
            tol_single=tol_single,
            tol_double=tol_double,
            single_iter_cap=cap,
        )
        rows.append([ds_name, len(X), n_classes, "Hybrid", cap, tol_single,
                     it_s, it_d, "LR_ExpA", t, mem, acc])
    return rows

def run_experiment_B(ds_name, X, y, n_classes, config):
    rows = []
    tol_double = config["tol_double_B"]
    max_iter   = config["max_iter_B"]

    for tol_single in config["tol_single_grid"]:
        _, it_s_b, it_d_b, t_b, mem_b, acc_b = run_full_double(X, y, n_classes, max_iter, tol=tol_double)
        rows.append([ds_name, len(X), n_classes, "Double", tol_single,
                     it_s_b, it_d_b, "LR_ExpB", t_b, mem_b, acc_b])

        it_s, it_d, t, mem, acc = run_hybrid(
            X, y, n_classes,
            max_iter_total=max_iter,
            tol_single=tol_single,
            tol_double=tol_double,
            single_iter_cap=None,
        )
        rows.append([ds_name, len(X), n_classes, "Hybrid", tol_single,
                     it_s, it_d, "LR_ExpB", t, mem, acc])
    return rows

