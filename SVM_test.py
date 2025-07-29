import numpy as np
import time
import pandas as pd
from aoclda.sklearn import skpatch
skpatch()
import matplotlib.pyplot as plt
import pathlib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

RESULTS_DIR = pathlib.Path("Results")
RESULTS_DIR.mkdir(exist_ok=True)

def generate_dataset(n_samples, n_features, seed=0):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, random_state=seed)
    return X, y

def svm_double_precision(X, y, max_iter, tol, C=1.0, kernel='rbf', test_size=0.2, seed=0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    start_time = time.time()
    svm = SVC(C=C, kernel=kernel, gamma='scale', tol=tol, max_iter=max_iter, random_state=seed)

    svm.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    iterations = svm.n_iter_
    return (kernel, "Double", 0, tol, acc, f1, elapsed_time, 0,iterations)

def svm_hybrid_precision(X, y, max_iter_total, tol_single, tol_double, single_iter_cap, C=1.0, kernel='rbf', test_size=0.2, seed=0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    X_train_single = X_train.astype(np.float32)
    start_single = time.time()
    svm_single = SVC(C=C, kernel=kernel, gamma='scale', tol=tol_single, max_iter=single_iter_cap, random_state=seed)
    svm_single.fit(X_train_single, y_train)
    elapsed_single = time.time() - start_single
    iters_single = svm_single.n_iter_

    remaining_iters = max(1, max_iter_total - iters_single)
    start_double = time.time()
    svm_double = SVC(C=C, kernel=kernel, gamma='scale', tol=tol_double, max_iter=remaining_iters, random_state=seed)
    svm_double.fit(X_train, y_train)
    elapsed_double = time.time() - start_double
    y_pred = svm_double.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    iters_double = svm_double.n_iter_

    return (kernel, "Hybrid", single_iter_cap, tol_single, acc, f1, elapsed_single + elapsed_double, iters_single,iters_double)

def run_experiments():
    X, y = generate_dataset(n_samples=10000, n_features=20, seed=0)  # Reduce size to speed up
    kernels = ['linear', 'rbf']
    tol_fixed = 1e-3
    caps = [0, 10, 20, 30, 40, 50]
    cap_fixed = 100
    tolerances = [1e-1, 1e-2, 1e-3, 1e-4]
    tol_double = 1e-5
    rows_A, rows_B = [], []

    for kernel in kernels:
        for cap in caps:
            res_dbl = svm_double_precision(X, y, max_iter=300, tol=tol_fixed, kernel=kernel)
            res_hyb = svm_hybrid_precision(X, y, max_iter_total=300, tol_single=tol_fixed, tol_double=tol_fixed, single_iter_cap=cap, kernel=kernel)
            res_dbl["Cap"] = cap
            rows_A.append(res_dbl)
            rows_A.append(res_hyb)

        for tol in tolerances:
            res_dbl = svm_double_precision(X, y, max_iter=cap_fixed, tol=tol, kernel=kernel)
            res_hyb = svm_hybrid_precision(X, y, max_iter_total=cap_fixed, tol_single=tol, tol_double=tol_double, single_iter_cap=cap_fixed, kernel=kernel)
            res_dbl["Tolerance"] = tol
            rows_B.append(res_dbl)
            rows_B.append(res_hyb)

    df_A = pd.DataFrame(rows_A)
    df_B = pd.DataFrame(rows_B)
    df_A.to_csv(RESULTS_DIR / "svm_results_expA.csv", index=False)
    df_B.to_csv(RESULTS_DIR / "svm_results_expB.csv", index=False)
    return df_A, df_B

def plot_svm_cap_vs_accuracy(df):
    df = df[df["Mode"] == "Hybrid"]
    grouped = df.groupby("Cap")["Accuracy"].mean().reset_index()
    plt.plot(grouped["Cap"], grouped["Accuracy"], marker='o')
    plt.title("Cap vs Accuracy (SVM Hybrid)")
    plt.xlabel("Single Precision Iteration Cap")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "svm_cap_vs_accuracy.png")
    plt.close()

def plot_svm_cap_vs_time(df):
    df = df[df["Mode"] == "Hybrid"]
    grouped = df.groupby("Cap")["Time"].mean().reset_index()
    plt.plot(grouped["Cap"], grouped["Time"], marker='o')
    plt.title("Cap vs Time (SVM Hybrid)")
    plt.xlabel("Single Precision Iteration Cap")
    plt.ylabel("Total Time (s)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "svm_cap_vs_time.png")
    plt.close()

def plot_svm_tolerance_vs_accuracy(df):
    df = df[df["Mode"] == "Hybrid"]
    grouped = df.groupby("Tolerance")["Accuracy"].mean().reset_index()
    plt.plot(grouped["Tolerance"], grouped["Accuracy"], marker='o')
    plt.xscale("log")
    plt.title("Tolerance vs Accuracy (SVM Hybrid)")
    plt.xlabel("Single Precision Tolerance")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "svm_tolerance_vs_accuracy.png")
    plt.close()

def plot_svm_tolerance_vs_time(df):
    df = df[df["Mode"] == "Hybrid"]
    grouped = df.groupby("Tolerance")["Time"].mean().reset_index()
    plt.plot(grouped["Tolerance"], grouped["Time"], marker='o')
    plt.xscale("log")
    plt.title("Tolerance vs Time (SVM Hybrid)")
    plt.xlabel("Single Precision Tolerance")
    plt.ylabel("Total Time (s)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "svm_tolerance_vs_time.png")
    plt.close()

if __name__ == "__main__":
    df_A, df_B = run_experiments()
    plot_svm_cap_vs_accuracy(df_A)
    plot_svm_cap_vs_time(df_A)
    plot_svm_tolerance_vs_accuracy(df_B)
    plot_svm_tolerance_vs_time(df_B)

