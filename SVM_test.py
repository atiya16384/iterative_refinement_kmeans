import numpy as np
import time
from aoclda.sklearn import skpatch
skpatch() # Apply AOCL patch before any SM usage
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Generate synthetic dataset
def generate_dataset(n_samples, n_features, seed=0):
    X, y = make_classification(n_samples=n_samples,n_features=n_features, random_state=seed)
    return X, y

# Double precision baseline
def svm_double_precision(X, y, max_iter, tol, C=1.0, kernel='rbf', test_size=0.2, seed=0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    start_time = time.time()

    svm = SVC(C=C, kernel=kernel, gamma='scale', tol=tol, max_iter=max_iter, random_state=seed)
    print(svm.aocl)
    svm.fit(X_train, y_train)
    elapsed_time = time.time() - start_time

    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    iterations = svm.n_iter_

    print(f"[Double Precision] Kernel: {kernel}, tol_single: {tol}, C: {C}, Single Iterations : 0, Double Iterations: {iterations}, Accuracy: {acc:.4f}, F1-score: {f1:.4f}, Time: {elapsed_time:.2f}s")


# Hybrid precision adaptive method
def svm_hybrid_precision(X, y, max_iter_total, tol_single, tol_double, C=1.0, kernel='rbf', test_size=0.2, seed=0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    X_train_single = X_train.astype(np.float32)

    start_single = time.time()
    svm_single = SVC(C=C, kernel=kernel, gamma='scale', tol=tol_single, max_iter=max_iter_total, random_state=seed)
    svm_single.fit(X_train_single, y_train)
    elapsed_single = time.time() - start_single

    elapsed_double = 0
    iters_single = svm_single.n_iter_

    remaining_iters = max(1, (max_iter_total - iters_single))
    start_double = time.time()
    svm_double = SVC(C=C, kernel=kernel, gamma='scale', tol=tol_double, max_iter=remaining_iters, random_state=seed)
    svm_double.fit(X_train, y_train)
    elapsed_double = time.time() - start_double
   
    y_pred = svm_double.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    iters_double = svm_double.n_iter_
    total_time = elapsed_single + elapsed_double

    print(f"[Hybrid Precision] Kernel: {kernel}, tol: {tol_single},  C: {C}, Single Iterations: {iters_single if iters_single is not None else 'N/A'}, Double Iterations: {iters_double}, Accuracy: {acc:.4f}, F1-score: {f1:.4f}, Time: {total_time:.2f}s")

# Run Experiments A and B
def run_experiments():
    X, y = generate_dataset(n_samples = 40000, n_features = 20, seed = 0)

    # initialise svm in one place
    kernels = ['linear', 'rbf']

    # Experiment A: Fixed tolerance, varying iteration caps
    tol_fixed = 1e-16
    caps = [0, 10, 20, 30, 40, 50]
    print("\nExperiment A: Fixed Tolerance, Varying Iteration Cap")
    for kernel in kernels:
        for cap in caps:
            svm_double_precision(X, y, max_iter=300, tol=tol_fixed, kernel=kernel, C=1.0)
            svm_hybrid_precision(X, y, max_iter_total=cap, tol_single=tol_fixed, tol_double=tol_fixed, kernel=kernel, C=1.0)

    # Experiment B: Fixed iteration cap, varying tolerance
    cap_fixed = 1000
    tolerances = [1e-1, 1e-2, 1e-3, 1e-4]
    print("\nExperiment B: Fixed Iteration Cap, Varying Tolerance")
    for kernel in kernels:
        for tol in tolerances:
            svm_double_precision(X, y, max_iter=cap_fixed, tol=tol, kernel=kernel, C=1.0)
            svm_hybrid_precision(X, y, max_iter_total=cap_fixed, tol_single=tol, tol_double=1e-5, kernel=kernel, C=1.0)

if __name__ == "__main__":
    run_experiments()
