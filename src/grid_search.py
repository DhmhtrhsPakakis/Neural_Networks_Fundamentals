import itertools
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from src.pytorch_mlp import MLPClassifier
from src.features import extract_flattened, extract_pca, extract_resnet_embeddings

def run_cv_grid_search(X_full, y_full, n_samples=5000, n_splits=5, n_components=53, random_state=42):
    
    # Extract the subset
    X_subset, _, y_subset, _ = train_test_split(
        X_full, y_full, 
        train_size=n_samples, 
        stratify=y_full, 
        random_state=random_state
    )
    
    # Define the hyperparameter space for search
    param_grid = {
        'feature_extraction': ['flatten', 'pca', 'resnet'], 
        'hidden_layers': [[64, 64], [128, 128], [256, 256], [256, 128, 64]],
        'learning_rate': [1e-2, 1e-3, 1e-4],
        'weight_decay': [1e-3, 1e-4, 1e-5]
    }
    
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Starting Grid Search for {len(experiments)} combinations...")
    
    best_overall_f1 = -1.0
    best_params = None
    results = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for i, params in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] Testing config: {params}")
        fold_f1_scores = []
        
        # Cross Validation Loop
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_subset, y_subset)):
            
            X_fold_train_raw, X_fold_val_raw = X_subset[train_idx], X_subset[val_idx]
            y_fold_train, y_fold_val = y_subset[train_idx], y_subset[val_idx]
            
            # Feature Extraction Method
            feat_method = params['feature_extraction']
            
            if feat_method == 'flatten':
                X_train_feats, X_val_feats = extract_flattened(
                    X_fold_train_raw, X_fold_val_raw, standardize=True
                )
            elif feat_method == 'pca':
                X_train_flat, X_val_flat = extract_flattened(
                    X_fold_train_raw, X_fold_val_raw, standardize=True
                )
                X_train_feats, X_val_feats, _ = extract_pca(
                    X_train_flat, X_val_flat, n_components=n_components, random_state=random_state
                )
            elif feat_method == 'resnet':
                X_train_feats, X_val_feats = extract_resnet_embeddings(
                    X_fold_train_raw, X_fold_val_raw
                )
            
            # Model TRaining
            model = MLPClassifier(
                hidden_layers=params['hidden_layers'],
                learning_rate=params['learning_rate'],
                weight_decay=params['weight_decay'],
                epochs=60,          
                T_0=15,             
                batch_size=64,     
                random_state=random_state,
                verbose=0           
            )
            
            model.fit(X_train_feats, y_fold_train, X_val=X_val_feats, y_val=y_fold_val)
            fold_f1_scores.append(model.best_val_f1_)
            
        mean_cv_f1 = np.mean(fold_f1_scores)
        print(f"--> Mean 5-CV Macro F1: {mean_cv_f1:.4f} (±{np.std(fold_f1_scores):.4f})")
        
        params_with_score = params.copy()
        params_with_score['mean_cv_f1'] = mean_cv_f1
        results.append(params_with_score)
        
        if mean_cv_f1 > best_overall_f1:
            best_overall_f1 = mean_cv_f1
            best_params = params
            
    print("=" * 60)
    print(f"Top 5-CV Mean Macro F1 Score: {best_overall_f1:.4f}")
    print(f"Optimal Hyperparameters: {best_params}")
    
    df_results = pd.DataFrame(results).sort_values(by='mean_cv_f1', ascending=False)
    return best_params, df_results