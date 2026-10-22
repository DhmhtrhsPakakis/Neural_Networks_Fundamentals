import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models, transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def extract_flattened(X_train, X_test, standardize=True):
    """
    Flattening: Converts images from (N, 3, 32, 32) to vectors of (N, 3072).
    """
    N_train = X_train.shape[0]
    N_test = X_test.shape[0]
    
    X_train_flat = X_train.reshape(N_train, -1)
    X_test_flat = X_test.reshape(N_test, -1)
    
    if standardize:
        scaler = StandardScaler()
        X_train_flat = scaler.fit_transform(X_train_flat)
        X_test_flat = scaler.transform(X_test_flat)
        
    return X_train_flat, X_test_flat


def extract_pca(X_train_flat, X_test_flat, n_components=0.95, random_state=42):
    """
    PCA: Dimensionality Reduction from vectors of 3072 to n_components for 95% variance.
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    
    # Fit on the train set!
    X_train_pca = pca.fit_transform(X_train_flat)
    X_test_pca = pca.transform(X_test_flat)
    
    return X_train_pca, X_test_pca, pca


def extract_resnet_embeddings(X_train, X_test, cache_dir="../data/features", batch_size=128):
    """
    ResNet18 Feature Extraction: Extract vectors of 512 using  ResNet18.
    """
    os.makedirs(cache_dir, exist_ok=True)
    train_cache = os.path.join(cache_dir, f"resnet_train_{len(X_train)}.npy")
    test_cache = os.path.join(cache_dir, f"resnet_test_{len(X_test)}.npy")
    
    if os.path.exists(train_cache) and os.path.exists(test_cache):
        print("Φόρτωση έτοιμων ResNet18 embeddings από τον δίσκο...")
        return np.load(train_cache), np.load(test_cache)
    
    device = torch.device("cpu")
    
    # load resnet and reomve the classification layer
    weights = models.ResNet18_Weights.DEFAULT
    resnet = models.resnet18(weights=weights)
    resnet.fc = nn.Identity()  # Load the vector before classification
    resnet = resnet.to(device)
    resnet.eval()
    
    # Normalization from ImageNet and resize to 64x64
    preprocess = transforms.Compose([
        transforms.Resize((64, 64), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    def _get_embeddings(X_data, desc):
        tensor_x = torch.from_numpy(X_data)
        dataset = TensorDataset(tensor_x)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        embeddings = []
        with torch.no_grad():
            for (batch_x,) in tqdm(loader, desc=desc):
                batch_x = preprocess(batch_x.to(device))
                feats = resnet(batch_x)
                embeddings.append(feats.cpu().numpy())
        return np.vstack(embeddings)
    
    X_train_res = _get_embeddings(X_train, "ResNet Train")
    X_test_res = _get_embeddings(X_test, "ResNet Test")
    
    # Normalize the extracted embedding
    scaler = StandardScaler()
    X_train_res = scaler.fit_transform(X_train_res)
    X_test_res = scaler.transform(X_test_res)
    
    np.save(train_cache, X_train_res)
    np.save(test_cache, X_test_res)
    
    return X_train_res, X_test_res