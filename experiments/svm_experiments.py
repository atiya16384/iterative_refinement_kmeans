# experiments/svm_experiments.py
from experiments.svm_precision import (
    rff_sgd_double_precision as svm_double_precision,
    rff_sgd_hybrid_ir        as svm_hybrid_precision,
)

class SVMExperimentRunner:
    def __init__(self, config):
        self.config = config
        self.results_A = []
        self.results_B = []

    def run_all(self, tag, X, y):
        cfg = self.config
        for _ in range(cfg["n_repeats"]):
            # A: Vary "cap" = Stage-1 epochs in float32
            for cap in cfg["caps"]:
                # Baseline: total epochs all in double
                self.results_A.append(
                    svm_double_precision(tag, X, y,
                        total_epochs=cfg["epochs_A_total"],
                        tol=cfg["tol_fixed_A"], alpha=cfg["alpha"], gamma=cfg["gamma"],
                        n_components=cfg["n_components"], batch_size=cfg["batch_size"])
                )
                # Hybrid IR: cap epochs in fp32, rest in fp64
                self.results_A.append(
                    svm_hybrid_precision(tag, X, y,
                        total_epochs=cfg["epochs_A_total"], cap_epochs=cap,
                        tol_single=cfg["tol_fixed_A"], tol_double=cfg["tol_fixed_A"],
                        alpha=cfg["alpha"], gamma=cfg["gamma"],
                        n_components=cfg["n_components"], batch_size=cfg["batch_size"])
                )

            # B: Vary tolerance (Stage-1 tol_single), fix cap_epochs
            for tol in cfg["tolerances"]:
                self.results_B.append(
                    svm_double_precision(tag, X, y,
                        total_epochs=cfg["epochs_B_total"],
                        tol=cfg["tol_double_B"], alpha=cfg["alpha"], gamma=cfg["gamma"],
                        n_components=cfg["n_components"], batch_size=cfg["batch_size"])
                )
                self.results_B.append(
                    svm_hybrid_precision(tag, X, y,
                        total_epochs=cfg["epochs_B_total"], cap_epochs=cfg["cap_B"],
                        tol_single=tol, tol_double=cfg["tol_double_B"],
                        alpha=cfg["alpha"], gamma=cfg["gamma"],
                        n_components=cfg["n_components"], batch_size=cfg["batch_size"])
                )

    def get_results(self):
        return self.results_A, self.results_B

