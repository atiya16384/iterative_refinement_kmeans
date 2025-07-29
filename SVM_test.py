import numpy as np
import time
import pandas as pd
import psutil
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import pathlib

RESULTS_DIR = pathlib.Path("Results")
RESULTS_DIR.mkdir(exist_ok=True)

def generate_dataset(n_samples, n_features, n_classes=2, seed=0):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_classes=n_classes,
        random_state=seed
    )
    return X, y

def measure_memory():
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 2)  # memory in MB

def svm_double_precision(tag, X, y, max_iter, tol, cap=0, C=1.0, kernel='rbf', test_size=0.2, seed=0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    mem_before = measure_memory()
    start_time = time.time()
    svm = SVC(C=C, kernel=kernel, gamma='scale', tol=tol, max_iter=int(max_iter), random_state=seed)
    svm.fit(X_train, y_train)
    elapsed = time.time() - start_time
    mem_after = measure_memory()
    y_pred = svm.predict(X_test)

    return (
        tag, len(X), 0, tol, cap, 0, svm.n_iter_, 'Double',
        elapsed, mem_after - mem_before, accuracy_score(y_test, y_pred)
    )

def svm_hybrid_precision(tag, X, y, max_iter_total, tol_single, tol_double, single_iter_cap, C=1.0, kernel='rbf', test_size=0.2, seed=0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    mem_before = measure_memory()

    X_train_single = X_train.astype(np.float32)
    start_single = time.time()
    svm_single = SVC(C=C, kernel=kernel, gamma='scale', tol=tol_single, max_iter=int(single_iter_cap), random_state=seed)
    svm_single.fit(X_train_single, y_train)
    time_single = time.time() - start_single
    iter_single = svm_single.n_iter_

    remaining_iters = max(1, max_iter_total - iter_single)
    start_double = time.time()
    svm_double = SVC(C=C, kernel=kernel, gamma='scale', tol=tol_double, max_iter=remaining_iters, random_state=seed)
    svm_double.fit(X_train, y_train)
    time_double = time.time() - start_double
    iter_double = svm_double.n_iter_

    total_time = time_single + time_double
    mem_after = measure_memory()
    y_pred = svm_double.predict(X_test)

    return (
        tag, len(X), 0, tol_single, single_iter_cap, iter_single, iter_double, 'Hybrid',
        total_time, mem_after - mem_before, accuracy_score(y_test, y_pred)
    )

def run_experiments():
    tol_fixed_A = 1e-16
    tol_double_B = 1e-5
    caps = [0, 50, 100, 150, 200, 250, 300]
    tolerances = [1e-1, 1e-2, 1e-3, 1e-4]
    max_iter_C = 300
    perc_C = 0.8
    tol_D = 1e-3
    n_repeats = 1

    synth_specs = [
        ("SYNTH_C_5_F_80_n1M", 1_000_000, 80, 5, 0),
        ("SYNTH_C_80_F_5_n1M", 1_000_000, 5, 80, 1),
        ("SYNTH_C_80_30_n1M", 1_000_000, 30, 80, 2)
    ]

    results_A, results_B, results_C, results_D = [], [], [], []

    for tag, n_samples, n_features, n_classes, seed in synth_specs:
        X, y = generate_dataset(n_samples=n_samples, n_features=n_features, n_classes=n_classes, seed=seed)

        for _ in range(n_repeats):
            # Experiment A
            for cap in caps:
                results_A.append(svm_double_precision(tag, X, y, max_iter=300, tol=tol_fixed_A, cap=cap))
                results_A.append(svm_hybrid_precision(tag, X, y, max_iter_total=300, tol_single=tol_fixed_A, tol_double=tol_fixed_A, single_iter_cap=cap))

            # Experiment B
            for tol in tolerances:
                results_B.append(svm_double_precision(tag, X, y, max_iter=1000, tol=tol, cap=1000))
                results_B.append(svm_hybrid_precision(tag, X, y, max_iter_total=1000, tol_single=tol, tol_double=tol_double_B, single_iter_cap=1000))

            # Experiment C (Fixed % of max_iter)
            cap_80 = int(max_iter_C * perc_C)
            results_C.append(svm_double_precision(tag, X, y, max_iter=max_iter_C, tol=tol_fixed_A, cap=cap_80))
            results_C.append(svm_hybrid_precision(tag, X, y, max_iter_total=max_iter_C, tol_single=tol_fixed_A, tol_double=tol_fixed_A, single_iter_cap=cap_80))

            # Experiment D (Fixed Tolerance)
            results_D.append(svm_double_precision(tag, X, y, max_iter=1000, tol=tol_D, cap=1000))
            results_D.append(svm_hybrid_precision(tag, X, y, max_iter_total=1000, tol_single=tol_D, tol_double=tol_double_B, single_iter_cap=1000))

    columns = [
        'DatasetName', 'DatasetSize', 'NumClusters',
        'tolerance_single', 'Cap', 'iter_single', 'iter_double', 'Suite',
        'Time', 'Memory_MB', 'accuracy'
    ]

    df_A = pd.DataFrame(results_A, columns=columns)
    df_B = pd.DataFrame(results_B, columns=columns)
    df_C = pd.DataFrame(results_C, columns=columns)
    df_D = pd.DataFrame(results_D, columns=columns)

    df_A.to_csv(RESULTS_DIR / "svm_expA_caps.csv", index=False)
    df_B.to_csv(RESULTS_DIR / "svm_expB_tol.csv", index=False)
    df_C.to_csv(RESULTS_DIR / "svm_expC_80percent.csv", index=False)
    df_D.to_csv(RESULTS_DIR / "svm_expD_tol_fixed.csv", index=False)

    print("Saved:")
    print("- svm_expA_caps.csv")
    print("- svm_expB_tol.csv")
    print("- svm_expC_80percent.csv")
    print("- svm_expD_tol_fixed.csv")
    print_summary()

    # Print summary for each experiment
def print_summary(path, group_by):
    df = pd.read_csv(path)
    print(f"\n==== SUMMARY: {path.name.upper()} ====")
    summary = df.groupby(group_by)[['accuracy', 'Time', 'Memory_MB']].mean()
    print(summary)

    print_summary(RESULTS_DIR / "svm_expA_caps.csv", ['DatasetName', 'Suite'])
    print_summary(RESULTS_DIR / "svm_expB_tol.csv", ['DatasetName', 'Suite'])
    print_summary(RESULTS_DIR / "svm_expC_80percent.csv", ['DatasetName', 'Suite'])
    print_summary(RESULTS_DIR / "svm_expD_tol_fixed.csv", ['DatasetName', 'Suite'])

    print("\nResults saved to 'Results/' directory.")

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
    run_experiments()
    plot_svm_cap_vs_accuracy(df_A)
    plot_svm_cap_vs_time(df_A)
    plot_svm_tolerance_vs_accuracy(df_B)
    plot_svm_tolerance_vs_time(df_B)

