# experiments/logreg_experiments.py
from typing import Dict, List, Tuple
import numpy as np
from experiments.logreg_precision import (
    logreg_double_precision, logreg_hybrid_precision
)

# row schema:
# (DatasetName, DatasetSize, ModelDim, Tolerance, Cap,
#  IterSingle, IterDouble, Suite, Time, Memory_MB, Accuracy)
ResultRow = Tuple[str, int, int, float, int, int, int, str, float, float, float]


class LogRegExperimentRunner:
    """
    Runs Experiment A (vary cap) and Experiment B (vary tolerance)
    for Logistic Regression.
    """
    def __init__(self, config: Dict):
        self.cfg = config
        self.results_A: List[ResultRow] = []
        self.results_B: List[ResultRow] = []

    def run_all(self, tag: str, X: np.ndarray, y: np.ndarray) -> None:
        self._run_A(tag, X, y)
        self._run_B(tag, X, y)

    def get_results(self):
        return self.results_A, self.results_B

    # ---------------------- Experiment A: vary CAP ----------------------
    def _run_A(self, tag, X, y):
        c = self.cfg
        caps      = list(c.get("caps", [0, 5, 10, 20]))
        max_iter  = int(c.get("epochs_A_total", 50))
        tol_fixed = float(c.get("tol_fixed_A", 1e-4))
        # shared params
        common = dict(
            C=float(c.get("C", 1.0)),
            solver=str(c.get("solver", "lbfgs")),
            test_size=float(c.get("test_size", 0.2)),
            seed=int(c.get("seed", 0)),
        )
        for cap in caps:
            # baseline (double)
            self.results_A.append(
                logreg_double_precision(tag, X, y,
                    max_iter=max_iter, tol=tol_fixed, cap=cap, **common)
            )
            # hybrid
            self.results_A.append(
                logreg_hybrid_precision(tag, X, y,
                    max_iter_total=max_iter,
                    tol_single=tol_fixed, tol_double=tol_fixed,
                    single_iter_cap=cap, **common)
            )

    # ------------------- Experiment B: vary tol_single -------------------
    def _run_B(self, tag, X, y):
        c = self.cfg
        tols      = [float(t) for t in c.get("tolerances", [5e-3, 1e-3, 5e-4])]
        max_iter  = int(c.get("epochs_B_total", 50))
        tol_dbl   = float(c.get("tol_double_B", 1e-4))
        cap_B     = int(c.get("cap_B", 10))
        common = dict(
            C=float(c.get("C", 1.0)),
            solver=str(c.get("solver", "lbfgs")),
            test_size=float(c.get("test_size", 0.2)),
            seed=int(c.get("seed", 0)),
        )
        for tol_s in tols:
            self.results_B.append(
                logreg_double_precision(tag, X, y,
                    max_iter=max_iter, tol=tol_dbl, cap=cap_B, **common)
            )
            self.results_B.append(
                logreg_hybrid_precision(tag, X, y,
                    max_iter_total=max_iter,
                    tol_single=tol_s, tol_double=tol_dbl,
                    single_iter_cap=cap_B, **common)
            )
