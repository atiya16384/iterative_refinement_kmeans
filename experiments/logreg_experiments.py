# experiments/logreg_experiments.py
def run_experiment_A(ds_name, X, y, n_classes, config):
    rows = []
    tol_single = config["tol_fixed_A"]

    # compute once per dataset
    _, it_s_b, it_d_b, t_b, mem_b, acc_b = run_full_double(
        X, y, n_classes, config["max_iter_A"], tol=config.get("tol_double_B", 1e-6)
    )

    for cap in config["cap_grid"]:
        rows.append([ds_name, len(X), n_classes, "Double", cap, tol_single,
                     it_s_b, it_d_b, "LR_ExpA", t_b, mem_b, acc_b])

        it_s, it_d, t, mem, acc = run_hybrid(
            X, y, n_classes,
            max_iter_total=config["max_iter_A"],
            tol_single=tol_single,
            tol_double=config.get("tol_double_B", 1e-6),
            single_iter_cap=cap,
            min_acc_to_skip=config.get("min_acc_to_skip", None),  # e.g., 0.98
        )
        rows.append([ds_name, len(X), n_classes, "Hybrid", cap, tol_single,
                     it_s, it_d, "LR_ExpA", t, mem, acc])
    return rows

def run_experiment_B(ds_name, X, y, n_classes, config):
    rows = []
    for tol_single in config["tol_single_grid"]:
        _, it_s_b, it_d_b, t_b, mem_b, acc_b = run_full_double(
            X, y, n_classes, config["max_iter_B"], tol=config["tol_double_B"]
        )
        rows.append([ds_name, len(X), n_classes, "Double", tol_single,
                     it_s_b, it_d_b, "LR_ExpB", t_b, mem_b, acc_b])

        it_s, it_d, t, mem, acc = run_hybrid(
            X, y, n_classes,
            max_iter_total=config["max_iter_B"],
            tol_single=tol_single,
            tol_double=config["tol_double_B"],
            single_iter_cap=None,  # no cap
            min_acc_to_skip=config.get("min_acc_to_skip", None),
        )
        rows.append([ds_name, len(X), n_classes, "Hybrid", tol_single,
                     it_s, it_d, "LR_ExpB", t, mem, acc])
    return rows

