# experiments/svm_experiments.py
from experiments.svm_precision import svm_double_precision, svm_hybrid_precision

class SVMExperimentRunner:
    def __init__(self, config):
        self.config = config
        self.results_A = []
        self.results_B = []

    def run_all(self, tag, X, y):
        cfg = self.config
        # ===== Experiment A: vary subset percentage ("cap") =====
        for cap in cfg["caps"]:
            self.results_A.append(
                svm_double_precision(tag, X, y,
                    max_iter=cfg["max_iter_A"], tol=cfg["tol_fixed_A"], cap=cap,
                    C=cfg.get("C", 1.0), kernel=cfg.get("kernel", "rbf"),
                    gamma=cfg.get("gamma", "scale"))
            )
            self.results_A.append(
                svm_hybrid_precision(tag, X, y,
                    max_iter_total=cfg["max_iter_A"],
                    tol_single=cfg["tol_fixed_A"], tol_double=cfg["tol_fixed_A"],
                    single_iter_cap=cap,
                    C=cfg.get("C", 1.0), kernel=cfg.get("kernel", "rbf"),
                    gamma=cfg.get("gamma", "scale"),
                    keep_frac=cfg.get("keep_frac", 0.40))
            )

        # ===== Experiment B: vary Stage-1 tolerance; fix subset percentage =====
        for tol in cfg["tolerances"]:
            self.results_B.append(
                svm_double_precision(tag, X, y,
                    max_iter=cfg["max_iter_B"], tol=cfg["tol_double_B"], cap=cfg["cap_B"],
                    C=cfg.get("C", 1.0), kernel=cfg.get("kernel", "rbf"),
                    gamma=cfg.get("gamma", "scale"))
            )
            self.results_B.append(
                svm_hybrid_precision(tag, X, y,
                    max_iter_total=cfg["max_iter_B"],
                    tol_single=tol, tol_double=cfg["tol_double_B"],
                    single_iter_cap=cfg["cap_B"],
                    C=cfg.get("C", 1.0), kernel=cfg.get("kernel", "rbf"),
                    gamma=cfg.get("gamma", "scale"),
                    keep_frac=cfg.get("keep_frac", 0.40))
            )

    def get_results(self):
        return self.results_A, self.results_B
