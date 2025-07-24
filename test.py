import numpy as np
import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

def svm_double_precision(X, y, max_iter, tol, C=1.0, kernel='rbf', test_size=0.2, seed=0):
    """
    Double precision SVM classification baseline.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    start_time = time.time()
    
    svm = SVC(C=C, kernel=kernel, gamma='scale', tol=tol, 
              max_iter=max_iter, random_state=seed)
    
    svm.fit(X_train, y_train)
    elapsed_time = time.time() - start_time

    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    support_vectors_count = len(svm.support_)

    iterations = svm.n_iter_
    memory_MB = X_train.astype(np.float64).nbytes / 1e6

    return {
        'model': svm,
        'iterations': iterations,
        'elapsed_time': elapsed_time,
        'accuracy': acc,
        'f1_score': f1,
        'memory_MB': memory_MB,
        'num_support_vectors': support_vectors_count
    }

def svm_hybrid_precision(X, y, max_iter_total, tol_single, tol_double, residual_tol=1e-3, single_iter_cap=200, 
                         C=1.0, kernel='rbf', test_size=0.2, seed=0):
    """
    Adaptive hybrid-precision SVM classification.
    """
    # Split dataset into train-test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    # Convert to single precision for initial training
    X_train_single = X_train.astype(np.float32)
    
    # Single-precision phase
    start_single = time.time()
    svm_single = SVC(C=C, kernel=kernel, gamma='scale', tol=tol_single,
                     max_iter=single_iter_cap, random_state=seed)
    svm_single.fit(X_train_single, y_train)
    elapsed_single = time.time() - start_single

    # Evaluate single-precision residual (training error)
    residual_single = 1 - svm_single.score(X_train_single, y_train)

    elapsed_double = 0
    total_double_iters = 0

    # Adaptive switching based on residual
    if residual_single > residual_tol and (max_iter_total - svm_single.n_iter_) > 0:
        remaining_iters = max_iter_total - svm_single.n_iter_
        
        # Double precision refinement
        start_double = time.time()
        svm_double = SVC(C=C, kernel=kernel, gamma='scale', tol=tol_double,
                         max_iter=remaining_iters, random_state=seed)
        svm_double.fit(X_train, y_train)
        elapsed_double = time.time() - start_double
        final_model = svm_double
        total_double_iters = svm_double.n_iter_
    else:
        final_model = svm_single

    # Evaluate final model on test set
    y_pred = final_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    support_vectors_count = len(final_model.support_)

    total_time = elapsed_single + elapsed_double
    memory_MB = (X_train_single.nbytes + X_train.nbytes) / 1e6

    return {
        'model': final_model,
        'single_precision_iters': svm_single.n_iter_,
        'double_precision_iters': total_double_iters,
        'elapsed_time': total_time,
        'accuracy': acc,
        'f1_score': f1,
        'memory_MB': memory_MB,
        'num_support_vectors': support_vectors_count
    }
