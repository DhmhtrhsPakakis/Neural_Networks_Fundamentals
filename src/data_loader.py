import os 
import numpy as np
from torchvision import datasets
from sklearn.model_selection import train_test_split

def load_svhn_data(data_dir="../data/raw", n_train=None, n_test=None, random_state=42, normalize=True, balanced=False):
    """
     Download and load SVHN dataset with stratified or balanced subsampling.

    Args:
        data_dir: Path to store data.
        n_train: Number of training samples to retain.
        n_test: Number of test samples to retain.
        random_state: Seed for reproducibility.
        normalize: If True, scale pixels to [0, 1].
        balanced: If True, returns a perfectly balanced subset (equal samples per class).
                  If False, returns a stratified subset preserving original class distribution.

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

    np.random.seed(random_state)

    def get_balanced_subset(X, y, n_samples):
        """Extract the same number of classes samples (uniform distribution)."""
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        samples_per_class = n_samples // n_classes
        
        selected_indices = []
        for c in unique_classes:
            class_indices = np.where(y == c)[0]
            # Chose the samples without replace
            chosen = np.random.choice(class_indices, samples_per_class, replace=False)
            selected_indices.extend(chosen)
            
        selected_indices = np.array(selected_indices)
        np.random.shuffle(selected_indices)
        
        return X[selected_indices], y[selected_indices]

    if balanced:
        # Create balanced dataset
        if n_train is not None and n_train < len(X_train):
            X_train, y_train = get_balanced_subset(X_train, y_train, n_train)
        if n_test is not None and n_test < len(X_test):
            X_test, y_test = get_balanced_subset(X_test, y_test, n_test)
    else:
        # Classic Stratified Subsampling
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