import os 
import numpy as np
from torchvision import datasets
from sklearn.model_selection import train_test_split

def load_svhn_data(data_dir="../data/raw", n_train=1500, n_test=3000, random_state=42, normalize=True):
    """
     Download and load SVHN dataset with stratified subsampling.

    Returns:
        X_train, X_test: numpy arrays with shape: (N,3,32,32) with values [0,1]
        y_train, y_test: numpy arrays with shape (N,) with labels [0-9]
    """

    os.makedirs(data_dir, exist_ok=True)

    print("Loading SVHN Dataset")
    train_dataset = datasets.SVHN(root=data_dir, split="train", download=True)
    test_dataset = datasets.SVHN(root=data_dir, split="test", download=True)

    X_train = train_dataset.data.astype(np.float32)
    y_train = train_dataset.labels.astype(np.int64)
    X_test = test_dataset.data.astype(np.float32)
    y_test = test_dataset.labels.astype(np.int64)
        
    # Normalize Data (Convert from [0-255] to [0-1])
    if normalize:
        X_train /=  255.0
        X_test /=  255.0

    # Stratified Subsampling for fust execution in cpu
    if n_train is not None and n_train < len(X_train):
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=n_train, stratify=y_train, random_state=random_state
        )
        
    if n_test is not None and n_test < len(X_test):
        X_test, _, y_test, _ = train_test_split(
            X_test, y_test, train_size=n_test, stratify=y_test, random_state=random_state
        )
        
    print(f"Dataset Loaded. Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test
    
    