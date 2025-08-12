from experiments.svm_precision import svm_double_precision, svm_hybrid_precision

class SVMExperimentRunner:
    def __init__(self, config):
        self.config = config
        self.results_A = []
        self.results_B = []

    def run_all(self, tag, X, y):
        cfg = self.config
        for _ in range(cfg["n_repeats"]):
            # ===== Experiment A: vary cap, fixed total epochs =====
            for cap in cfg["caps"]:
                # Baseline double: full budget in float64
                self.results_A.append(
                    svm_double_precision(
                        tag, X, y,
                        max_iter=cfg["epochs_A_total"],
                        tol=cfg["tol_fixed_A"], cap=cap
                    )
                )
                # Hybrid: cap in f32 + (total - cap) in f64
                self.results_A.append(
                    svm_hybrid_precision(
                        tag, X, y,
                        max_iter_total=cfg["epochs_A_total"],
                        tol_single=cfg["tol_fixed_A"],
                        tol_double=cfg["tol_fixed_A"],
                        single_iter_cap=cap
                    )
                )

            # ===== Experiment B: vary tol_single, fixed stage-1 cap and total =====
            for tol in cfg["tolerances"]:
                # Baseline double: full budget in float64
                self.results_B.append(
                    svm_double_precision(
                        tag, X, y,
                        max_iter=cfg["epochs_B_total"],
                        tol=cfg["tol_double_B"], cap=cfg["cap_B"]
                    )
                )
                # Hybrid: cap_B in f32 + (total - cap_B) in f64
                self.results_B.append(
                    svm_hybrid_precision(
                        tag, X, y,
                        max_iter_total=cfg["epochs_B_total"],
                        tol_single=tol,
                        tol_double=cfg["tol_double_B"],
                        single_iter_cap=cfg["cap_B"]
                    )
                )

    def get_results(self):
        return self.results_A, self.results_B

