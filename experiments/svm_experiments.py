from experiments.svm_precision import svm_double_precision, svm_hybrid_precision

class SVMExperimentRunner:
    def __init__(self, config):
        self.config = config
        self.results_A, self.results_B = [], []

    def run_all(self, tag, X, y):
        cfg = self.config

        # ===== A: vary CAP (% of train in Stage-1) =====
        for cap in cfg["caps"]:
            # Double baseline
            self.results_A.append(
                svm_double_precision(
                    tag, X, y,
                    tol=cfg["tol_fixed_A"],
                    max_iter=cfg["epochs_A_total"],
                    C=cfg["C"], kernel=cfg["kernel"], gamma=cfg["gamma"],
                    test_size=cfg["test_size"], seed=cfg["seed"],
                    cache_mb=cfg["cache_mb"],
                )
            )
            # Hybrid with same total budget
            self.results_A.append(
                svm_hybrid_precision(
                    tag, X, y,
                    max_iter_total=cfg["epochs_A_total"],
                    tol_single=cfg["tol_fixed_A"],  # same tol in both stages
                    tol_double=cfg["tol_fixed_A"],
                    single_iter_cap=cap,            # 0, 1, 2, 5, 10, 20 (%)
                    keep_frac=cfg["keep_frac_A"],
                    C=cfg["C"], kernel=cfg["kernel"], gamma=cfg["gamma"],
                    test_size=cfg["test_size"], seed=cfg["seed"],
                    cache_mb=cfg["cache_mb"],
                )
            )

        # ===== B: vary Stage-1 tolerance (CAP fixed) =====
        for tol_s in cfg["tolerances"]:
            # Double baseline
            self.results_B.append(
                svm_double_precision(
                    tag, X, y,
                    tol=cfg["tol_double_B"],
                    max_iter=cfg["epochs_B_total"],
                    C=cfg["C"], kernel=cfg["kernel"], gamma=cfg["gamma"],
                    test_size=cfg["test_size"], seed=cfg["seed"],
                    cache_mb=cfg["cache_mb"],
                )
            )
            # Hybrid with same total budget
            self.results_B.append(
                svm_hybrid_precision(
                    tag, X, y,
                    max_iter_total=cfg["epochs_B_total"],
                    tol_single=tol_s,
                    tol_double=cfg["tol_double_B"],
                    single_iter_cap=cfg["cap_B"],     # e.g. 10% probe
                    keep_frac=cfg["keep_frac_B"],
                    C=cfg["C"], kernel=cfg["kernel"], gamma=cfg["gamma"],
                    test_size=cfg["test_size"], seed=cfg["seed"],
                    cache_mb=cfg["cache_mb"],
                )
            )

    def get_results(self):
        return self.results_A, self.results_B

