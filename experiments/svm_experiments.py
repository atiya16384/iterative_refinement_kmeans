# experiments/svm_experiments.py
from experiments.svm_precision import (
    svm_double_precision,
    svm_hybrid_precision,
)

class SVMExperimentRunner:
    def __init__(self, config):
        self.config = config
        self.results_A = []
        self.results_B = []

    def run_all(self, tag, X, y):
        cfg = self.config
        alpha = cfg.get("alpha", 1e-4)
        batch_size = cfg.get("batch_size", 1024)

        # =========================
        # Experiment A: vary cap
        # =========================
        for cap in cfg["caps"]:
            # Baseline (double) — fixed epochs in double
            self.results_A.append(
                svm_double_precision(
                    tag, X, y,
                    max_iter=cfg["epochs_A_total"],
                    tol=cfg["tol_fixed_A"],
                    cap=cap,
                    alpha=alpha,
                    batch_size=batch_size,
                )
            )
            # Hybrid — float32 cap, tiny float64 polish
            self.results_A.append(
                svm_hybrid_precision(
                    tag, X, y,
                    max_iter_total=cfg["epochs_A_total"],
                    tol_single=cfg["tol_fixed_A"],
                    tol_double=cfg["tol_fixed_A"],
                    single_iter_cap=cap,
                    alpha=alpha,
                    batch_size=batch_size,
                    polish_epochs=cfg["polish_epochs_A"],
                    early_stop=False,        # A: no ES; isolate cap effect
                )
            )

        # =========================
        # Experiment B: vary tol
        # =========================
        for tol in cfg["tolerances"]:
            # Baseline (double) — same fixed double run for reference
            self.results_B.append(
                svm_double_precision(
                    tag, X, y,
                    max_iter=cfg["epochs_B_total"],
                    tol=cfg["tol_double_B"],
                    cap=cfg["cap_B"],
                    alpha=alpha,
                    batch_size=batch_size,
                )
            )
            # Hybrid — Stage-1 early stop with tol, capped by cap_B; small polish
            self.results_B.append(
                svm_hybrid_precision(
                    tag, X, y,
                    max_iter_total=cfg["epochs_B_total"],
                    tol_single=tol,
                    tol_double=cfg["tol_double_B"],
                    single_iter_cap=cfg["cap_B"],
                    alpha=alpha,
                    batch_size=batch_size,
                    polish_epochs=cfg["polish_epochs_B"],
                    early_stop=True,         # B: ES driven by tol_single
                    patience=cfg.get("patience_B", 2),
                )
            )

    def get_results(self):
        return self.results_A, self.results_B

