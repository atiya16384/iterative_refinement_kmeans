
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
    # Collapse array of iter counts to one scalar
    iter_single = _iters_scalar(getattr(svm_single, "n_iter_", 0))

    # Remaining budget must be a positive int
    remaining_iters = int(max(1, int(max_iter_total) - int(iter_single)))

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
