import numpy as np
from aoclda.svm import SVC
from sklearn.datasets import make_classification

import time
import pandas as pd


# --- Synthetic dataset ---
def make_synthetic_data(n_samples=2000, n_features=30, random_state=42):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, random_state=random_state)
    y = 2*y - 1   # convert {0,1} to {-1,+1}
    return X, y


# --- Double precision SVM (full solve) ---
def svm_double_precision(X, y, C=1.0, kernel="linear", tol=1e-5, max_iter=5000):
    model = SVC(kernel=kernel, precision="double", tol=tol, C=C, max_iter=max_iter)
    model.fit(X, y)
    return model


# --- Hybrid SVM: single precision + iterative refinement ---
def svm_hybrid_precision(X, y, C=1.0, kernel="linear", n_refine=3, max_iter=2000):
    # Step 1: initial training in single precision
    model_sp = SVC(kernel=kernel, precision="single", C=C, tol=1e-3, max_iter=max_iter)
    model_sp.fit(X, y)

    # Cast support vectors & coefficients to double
    dual_coef = model_sp.dual_coef_.astype(np.float64)
    support_vectors = model_sp.support_vectors_.astype(np.float64)

    # Step 2: iterative refinement in double precision
    model_dp = SVC(kernel=kernel, precision="double", C=C, tol=1e-5, max_iter=max_iter)
    model_dp.fit(X, y, x0=dual_coef)  # warm start from single precision

    for _ in range(n_refine - 1):
        model_dp.fit(X, y, x0=model_dp.dual_coef_)  # re-refine

    return model_dp


