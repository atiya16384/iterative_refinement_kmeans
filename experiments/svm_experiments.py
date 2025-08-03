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

    def get_results(self):
        return self.results_A, self.results_B, 
