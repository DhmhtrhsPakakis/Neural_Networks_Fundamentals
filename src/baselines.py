# src/baselines.py
import time
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.metrics import accuracy_score

def evaluate_baselines_on_features(X_train, y_train, X_test, y_test, feature_name):
    """
    Trains and evaluates 3 baseline classifiers using scikit-learn implementations:
    1. Nearest Class Centroid (NCC)
    2. 1-Nearest Neighbor (1-NN)
    3. 3-Nearest Neighbors (3-NN)
    """
    models = {
        "Nearest Class Centroid (NCC)": NearestCentroid(),
        "1-Nearest Neighbor (1-NN)": KNeighborsClassifier(n_neighbors=1, n_jobs=-1, metric='euclidean'),
        "3-Nearest Neighbors (3-NN)": KNeighborsClassifier(n_neighbors=3, n_jobs=-1, metric='euclidean')
    }
    
    results = []
    predictions = {}
    
    print(f"\n--- Evaluating Baselines for: {feature_name} (Dimensions D={X_train.shape[1]}) ---")
    
    for model_name, model in models.items():
        # 1. Training Time (Fit)
        t0 = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - t0
        
        # 2. Testing Time (Predict on Test set)
        t1 = time.time()
        y_pred_test = model.predict(X_test)
        pred_time = time.time() - t1
        
        # 3. Compute Accuracy
        test_acc = accuracy_score(y_test, y_pred_test) * 100
        
        print(f"{model_name:30s} | Test Acc: {test_acc:5.2f}% | Fit: {fit_time:.3f}s | Predict: {pred_time:.2f}s")
        
        results.append({
            "Features": feature_name,
            "Dimensions (D)": X_train.shape[1],
            "Classifier": model_name,
            "Test Accuracy (%)": round(test_acc, 2),
            "Fit Time (sec)": round(fit_time, 4),
            "Predict Time (sec)": round(pred_time, 3)
        })
        
        predictions[model_name] = y_pred_test
        
    return results, predictions