# experiments/svm_experiments.py
from experiments.svm_precision import svm_double_precision, svm_hybrid_precision

class SVMExperimentRunner:
    def __init__(self, config):
        self.config = config
        self.results_A = []
        self.results_B = []

    def run_all(self, tag, X, y):
        cfg = self.config
        for _ in range(cfg["n_repeats"]):
            # -----------------
            # Experiment A: vary cap (as % subset for Stage-1)
            # tol is fixed (cfg["tol_fixed_A"])
            # -----------------
            for cap in cfg["caps"]:
                self.results_A.append(
                    svm_double_precision(tag, X, y,
                        max_iter=cfg["max_iter_A"], tol=cfg["tol_fixed_A"], cap=cap)
                )
                self.results_A.append(
                    svm_hybrid_precision(tag, X, y,
                        max_iter_total=cfg["max_iter_A"],
                        tol_single=cfg["tol_fixed_A"], tol_double=cfg["tol_fixed_A"],
                        single_iter_cap=cap)
                )

            # -----------------
            # Experiment B: vary Stage-1 tolerance
            # Stage-2 tol is fixed (cfg["tol_double_B"])
            # -----------------
            for tol in cfg["tolerances"]:
                self.results_B.append(
                    svm_double_precision(tag, X, y,
                        max_iter=cfg["max_iter_B"], tol=cfg["tol_double_B"], cap=cfg["max_iter_B"])
                )
                self.results_B.append(
                    svm_hybrid_precision(tag, X, y,
                        max_iter_total=cfg["max_iter_B"],
                        tol_single=tol, tol_double=cfg["tol_double_B"],
                        single_iter_cap=cfg["cap_B"])  # keep % fixed for fairness
                )

    def get_results(self):
        return self.results_A, self.results_B

