from experiments.svm_precision import svm_double_precision, svm_hybrid_subset_filter

class SVMExperimentRunner:
    def __init__(self, config):
        self.config = config
        self.results_A, self.results_B = [], []

    def run_all(self, tag, X, y):
        cfg = self.config
        for rep in range(cfg["n_repeats"]):
            # ===== A: vary cap (subset size), fix tol_stage1 =====
            tol_s = cfg["tol_fixed_A"]        # e.g., 1e-16
            tol_d = cfg["tol_double_A"]       # e.g., 1e-5
            for cap_frac in cfg["cap_fracs_A"]:
                # Baseline: full SVC with final tol
                self.results_A.append(
                    svm_double_precision(tag, X, y, tol=tol_d, max_iter=cfg["max_iter_A"])
                )
                # Hybrid: subset -> filter -> final
                self.results_A.append(
                    svm_hybrid_subset_filter(
                        tag, X, y,
                        cap_frac=cap_frac,
                        tol_stage1=tol_s, tol_stage2=tol_d,
                        max_iter_stage1=cfg["max_iter_A_stage1"],
                        max_iter_stage2=cfg["max_iter_A"],
                        margin_thresh=cfg.get("margin_thresh", 1.0),
                        seed=rep
                    )
                )

            # ===== B: vary tol_stage1, fix cap =====
            cap_b = cfg["cap_frac_B"]
            tol_dB = cfg["tol_double_B"]      # e.g., 1e-5
            for tol_sweep in cfg["tolerances_B"]:
                self.results_B.append(
                    svm_double_precision(tag, X, y, tol=tol_dB, max_iter=cfg["max_iter_B"])
                )
                self.results_B.append(
                    svm_hybrid_subset_filter(
                        tag, X, y,
                        cap_frac=cap_b,
                        tol_stage1=tol_sweep, tol_stage2=tol_dB,
                        max_iter_stage1=cfg["max_iter_B_stage1"],
                        max_iter_stage2=cfg["max_iter_B"],
                        target_keep_frac = cfg["target_keep_frac"],
                        probe_gamma=cfg.get("probe_gamma", None),
                        cache_size_mb=cfg.get("cache_size_mb", 1000),
                        scale_X=cfg.get("scale_X", True),
                        seed=rep
                    )
                )

    def get_results(self):
        return self.results_A, self.results_B
