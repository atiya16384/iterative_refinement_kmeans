from experiments.svm_precision import svm_double_precision, svm_hybrid_precision

class SVMExperimentRunner:
    def __init__(self, config):
        self.config = config
        self.results_A = []
        self.results_B = []
        self.results_C = []
        self.results_D = []

    def run_all(self, tag, X, y):
        cfg = self.config
        for _ in range(cfg["n_repeats"]):
            # A: Vary cap
            for cap in cfg["caps"]:
                self.results_A.append(svm_double_precision(tag, X, y, max_iter=300, tol=cfg["tol_fixed_A"], cap=cap))
                self.results_A.append(svm_hybrid_precision(tag, X, y, max_iter_total=300, tol_single=cfg["tol_fixed_A"], tol_double=cfg["tol_fixed_A"], single_iter_cap=cap))
            # B: Vary tolerance
            for tol in cfg["tolerances"]:
                self.results_B.append(svm_double_precision(tag, X, y, max_iter=1000, tol=tol, cap=1000))
                self.results_B.append(svm_hybrid_precision(tag, X, y, max_iter_total=1000, tol_single=tol, tol_double=cfg["tol_double_B"], single_iter_cap=1000))
            # C: Fixed tol, 80% cap
            cap_80 = int(cfg["max_iter_C"] * cfg["perc_C"])
            self.results_C.append(svm_double_precision(tag, X, y, max_iter=cfg["max_iter_C"], tol=cfg["tol_fixed_A"], cap=cap_80))
            self.results_C.append(svm_hybrid_precision(tag, X, y, max_iter_total=cfg["max_iter_C"], tol_single=cfg["tol_fixed_A"], tol_double=cfg["tol_fixed_A"], single_iter_cap=cap_80))
            # D: Fixed tol
            self.results_D.append(svm_double_precision(tag, X, y, max_iter=1000, tol=cfg["tol_D"], cap=1000))
            self.results_D.append(svm_hybrid_precision(tag, X, y, max_iter_total=1000, tol_single=cfg["tol_D"], tol_double=cfg["tol_double_B"], single_iter_cap=1000))

    def get_results(self):
        return self.results_A, self.results_B, self.results_C, self.results_D
