from aoclda.sklearn import skpatch
skpatch()
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time 
import psutil
import numpy as np

def measure_memory():
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 2)

def _iters_scalar(n_iter_attr) -> int:
    if n_iter_attr is None:
        return 0
    arr = np.atleast_1d(n_iter_attr)
    return int(arr.max()) if arr.size else 0

def svm_double_precision(tag, X, y, max_iter, tol, cap=0, C=1.0, kernel='rbf', test_size=0.2, seed=0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    mem_before = measure_memory()
    start_time = time.time()
    svm = SVC(C=C, kernel=kernel, gamma='scale', tol=tol, max_iter=int(max_iter), random_state=seed)
    svm.fit(X_train, y_train)
    elapsed = time.time() - start_time
    mem_after = measure_memory()
    y_pred = svm.predict(X_test)
    iter_double = _iters_scalar(getattr(svm, "n_iter_", 0))

    return (
        tag, len(X), 0, tol, cap, 0,  _iters_scalar(getattr(svm, "n_iter_", 0)), 'Double',
        elapsed, mem_after - mem_before, accuracy_score(y_test, y_pred)
    )


def svm_hybrid_precision(tag, X, y, max_iter_total, tol_single, tol_double, single_iter_cap,
                         C=1.0, kernel='rbf', test_size=0.2, seed=0, cache_size=400):
    # single split, standardize once for both stages
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std  = scaler.transform(X_test)

    mem_before = measure_memory()

    # -------- stage 1 (probe, float32) --------
    Xtr32 = X_train_std.astype(np.float32, copy=False)
    Xte32 = X_test_std.astype(np.float32, copy=False)
    t1 = time.time()
    svm_single = SVC(C=C, kernel=kernel, gamma='scale', tol=float(tol_single),
                     max_iter=int(single_iter_cap), cache_size=float(cache_size), random_state=seed)
    svm_single.fit(Xtr32, y_train)
    time_single  = time.time() - t1
    iter_single  = _iters_scalar(getattr(svm_single, "n_iter_", 0))

    # Explicitly free probe arrays/models before final memory snapshot
    del Xtr32, Xte32, svm_single
    gc.collect()

    # -------- stage 2 (final, float64) --------
    remaining_iters = max(1, int(max_iter_total) - int(iter_single))
    t2 = time.time()
    svm_double = SVC(C=C, kernel=kernel, gamma='scale', tol=float(tol_double),
                     max_iter=int(remaining_iters), cache_size=float(cache_size), random_state=seed)
    svm_double.fit(X_train_std, y_train)
    time_double  = time.time() - t2
    iter_double  = _iters_scalar(getattr(svm_double, "n_iter_", 0))

    total_time = time_single + time_double
    y_pred = svm_double.predict(X_test_std)
    mem_after = measure_memory()

    return (
        tag, len(X), 0, tol_single, single_iter_cap, iter_single, iter_double, 'Hybrid',
        total_time, mem_after - mem_before, accuracy_score(y_test, y_pred)
    )
