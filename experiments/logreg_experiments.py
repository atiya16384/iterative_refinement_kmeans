from experiments.logreg_precision import run_full_double, run_hybrid

# ========================
# Experiment A (Cap sweep)
# ========================
# - Fixed single-precision tolerance: config["tol_fixed_A"]
# - Sweep single-iter cap:           config["cap_grid"]
# - Total iteration budget:          config["max_iter_A"]
def run_experiment_A(ds_name, X, y, n_classes, config):
    rows = []
    tol_single = config["tol_fixed_A"]

    for cap in config["cap_grid"]:
        # Double baseline
        _, it_s, it_d, t, mem, acc = run_full_double(
            X, y, n_classes, config["max_iter_A"], tol=config["tol_fixed_A"]
        )
        rows.append([
            ds_name, len(X), n_classes,
            "Double", cap, tol_single,
            it_s, it_d, "LR_ExpA",
            t, mem, acc
        ])

        # Hybrid with cap
        it_s, it_d, t, mem, acc = run_hybrid(
            X, y, n_classes,
            max_iter_total=config["max_iter_A"],
            tol_single=tol_single,
            tol_double=config["tol_fixed_A"],     # mirror k-means A: same tol used
            single_iter_cap=cap
        )
        rows.append([
            ds_name, len(X), n_classes,
            "Hybrid", cap, tol_single,
            it_s, it_d, "LR_ExpA",
            t, mem, acc
        ])
    return rows

# ===================================
# Experiment B (Tolerance-only sweep)
# ===================================
# - Sweep single-precision tolerance: config["tol_single_grid"]
# - No cap; single runs until tol or budget
# - Total iteration budget:           config["max_iter_B"]
# - Double tolerance:                 config["tol_double_B"]
def run_experiment_B(ds_name, X, y, n_classes, config):
    rows = []
    for tol_single in config["tol_single_grid"]:
        # Double baseline (same every loop; keep it paired for plotting convenience)
        _, it_s, it_d, t, mem, acc = run_full_double(
            X, y, n_classes, config["max_iter_B"], tol=config["tol_double_B"]
        )
        rows.append([
            ds_name, len(X), n_classes,
            "Double", tol_single,
            it_s, it_d, "LR_ExpB",
            t, mem, acc
        ])

        # Hybrid (no cap -> single_iter_cap=None)
        it_s, it_d, t, mem, acc = run_hybrid(
            X, y, n_classes,
            max_iter_total=config["max_iter_B"],
            tol_single=tol_single,
            tol_double=config["tol_double_B"],
            single_iter_cap=None
        )
        rows.append([
            ds_name, len(X), n_classes,
            "Hybrid", tol_single,
            it_s, it_d, "LR_ExpB",
            t, mem, acc
        ])
    return rows

