



from experiments.svm_precision import svm_double_precision, svm_hybrid_precision


def run_all(tag, X, y):
        for _ in range(n_repeats):
            # A
            for cap in caps:
                results_A.append(svm_double_precision(tag, X, y, max_iter=300, tol=tol_fixed_A, cap=cap))
                results_A.append(svm_hybrid_precision(tag, X, y, max_iter_total=300, tol_single=tol_fixed_A, tol_double=tol_fixed_A, single_iter_cap=cap))
            # B
            for tol in tolerances:
                results_B.append(svm_double_precision(tag, X, y, max_iter=1000, tol=tol, cap=1000))
                results_B.append(svm_hybrid_precision(tag, X, y, max_iter_total=1000, tol_single=tol, tol_double=tol_double_B, single_iter_cap=1000))
            # C
            cap_80 = int(max_iter_C * perc_C)
            results_C.append(svm_double_precision(tag, X, y, max_iter=max_iter_C, tol=tol_fixed_A, cap=cap_80))
            results_C.append(svm_hybrid_precision(tag, X, y, max_iter_total=max_iter_C, tol_single=tol_fixed_A, tol_double=tol_fixed_A, single_iter_cap=cap_80))
            # D
            results_D.append(svm_double_precision(tag, X, y, max_iter=1000, tol=tol_D, cap=1000))
            results_D.append(svm_hybrid_precision(tag, X, y, max_iter_total=1000, tol_single=tol_D, tol_double=tol_double_B, single_iter_cap=1000))
