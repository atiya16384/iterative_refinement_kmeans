# experiments/svm_experiments.py
from experiments.svm_precision import svm_double_precision, svm_hybrid_precision

class SVMExperimentRunner:
    def __init__(self, config):
        self.cfg = config
        self.results_A = []  # rows for Experiment A (cap)
        self.results_B = []  # rows for Experiment B (tolerance)

    def run_all(self, tag, X, y):
        n_repeats = int(self.cfg.get("n_repeats", 1))
        for _ in range(n_repeats):
            self._run_exp_A(tag, X, y)
            self._run_exp_B(tag, X, y)

    def get_results(self):
        return self.results_A, self.results_B

    # --- shared kwargs for SVC ---
    def _svc_kwargs(self):
        cfg = self.cfg
        return dict(
            C=float(cfg.get("C", 1.0)),
            kernel=str(cfg.get("kernel", "rbf")),
            gamma=cfg.get("gamma", "scale"),
            test_size=float(cfg.get("test_size", 0.2)),
            seed=int(cfg.get("seed", 0)),
            cache_mb=float(cfg.get("cache_mb", 1024)),
        )

    # ===== Experiment A: vary Stage-1 subset CAP (% of train) =====
    def _run_exp_A(self, tag, X, y):
        cfg = self.cfg
        caps      = list(cfg.get("caps", [0, 1, 2, 5, 10, 20]))
        tol_fixed = float(cfg.get("tol_fixed_A", 1e-4))
        max_iter  = int(cfg.get("epochs_A_total", 300))
        keep_frac = float(cfg.get("keep_frac_A", cfg.get("keep_frac", 0.10)))
        probe_max = int(cfg.get("probe_max", 100_000))
        kw = self._svc_kwargs()

        for cap in caps:
            # Double baseline (cap set so plots match 1:1 by Cap)
            self.results_A.append(
                svm_double_precision(
                    tag, X, y,
                    tol=tol_fixed,
                    max_iter=max_iter,
                    cap=cap,
                    **kw
                )
            )
            # Hybrid for the same cap
            self.results_A.append(
                svm_hybrid_precision(
                    tag, X, y,
                    max_iter_total=max_iter,
                    tol_single=tol_fixed,
                    tol_double=tol_fixed,
                    single_iter_cap=cap,
                    keep_frac=keep_frac,
                    probe_max=probe_max,
                    **kw
                )
            )

    # ===== Experiment B: vary Stage-1 tolerance (CAP fixed) =====
    def _run_exp_B(self, tag, X, y):
        cfg = self.cfg
        tols       = [float(t) for t in cfg.get("tolerances", [1e-3, 5e-3, 1e-2])]
        tol_double = float(cfg.get("tol_double_B", 1e-4))
        max_iter   = int(cfg.get("epochs_B_total", 300))
        cap_B      = cfg.get("cap_B", 10)
        keep_frac  = float(cfg.get("keep_frac_B", cfg.get("keep_frac", 0.10)))
        probe_max  = int(cfg.get("probe_max", 100_000))
        kw = self._svc_kwargs()

        for tol_s in tols:
            # Double baseline (set cap field for schema consistency)
            self.results_B.append(
                svm_double_precision(
                    tag, X, y,
                    tol=tol_double,
                    max_iter=max_iter,
                    cap=int(cap_B),
                    **kw
                )
            )
            # Hybrid with the same total budget
            if bool(cfg.get("add_baseline_point_B", True)):
                self.results_B.append(
                    svm_hybrid_precision(
                        tag, X, y,
                        max_iter_total=max_iter,
                        tol_single=tol_s,
                        tol_double=tol_double,
                        single_iter_cap=0, 
                        keep_frac=keep_frac,
                        probe_max=probe_max,
                        **kw
                    )
                )
