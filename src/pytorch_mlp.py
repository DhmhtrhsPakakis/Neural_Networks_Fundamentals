"""
PyTorch Multi-Layer Perceptron (MLP) Classifier Module for Multi-Class Classification.
Scikit-Learn compatible wrapper supporting dynamic depth.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

class MLPClassifier(BaseEstimator, ClassifierMixin):
    """
    A Scikit-Learn compatible wrapper for a dynamic PyTorch Multi-Layer Perceptron
    tailored for multi-class classification (e.g., SVHN 10 classes).
    """
    def __init__(self, hidden_layers=[128,64], 
                 activation='relu', learning_rate=1e-3, epochs=100, T_0=15,weight_decay=1e-4,
                 batch_size=64, random_state=42,use_class_weights=False, verbose=0):
        
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.T_0 = T_0
        self.batch_size = batch_size
        self.random_state = random_state
        self.use_class_weights = use_class_weights
        self.verbose = verbose
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.classes_ = None
        self.num_classes_ = None
        self.history_ = {'train_loss': [], 'val_loss': [], 'val_macro_f1': []}
        self.best_val_f1_ = -1.0

    def _build_network(self, input_dim, num_classes):
        """Dynamically instantiates the multi-class neural network architecture."""
        torch.manual_seed(self.random_state)
        
        layers = []
        in_features = input_dim

        # Support both int (e.g., hidden_layers=2, hidden_size=128) and list (e.g., [128, 64])
        if isinstance(self.hidden_layers, (list, tuple)):
            layer_dims = self.hidden_layers
        else:
            layer_dims = [self.hidden_size] * self.hidden_layers

        for out_features in layer_dims:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU() if self.activation == 'relu' else nn.Tanh())

            in_features = out_features

        # Output layer produces raw logits for each class (CrossEntropyLoss applies Softmax internally)
        layers.append(nn.Linear(in_features, num_classes))
        return nn.Sequential(*layers).to(self.device)

    def fit(self, X, y, X_val = None, y_val=None):
        """Trains the Multi-Class Neural Network on the provided dataset."""
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        # Dynamically infer classes (e.g., 0-9 for SVHN)
        self.classes_ = np.unique(y)
        self.num_classes_ = len(self.classes_)
        input_dim = X.shape[1]
        
        self.model = self._build_network(input_dim, self.num_classes_)
        self.history_ = {'train_loss': [], 'val_loss': [], 'val_macro_f1': []}

        # Optimizer with explicit Weight Decay (L2)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # Cosine Annealing with Warm Restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.T_0, T_mult=1)

        # Use Sample Weights
        if self.use_class_weights:
            # Inverse Sample Weights
            class_weights = compute_class_weight(
                class_weight='balanced', 
                classes=self.classes_, 
                y=y
            )
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            # Criterion
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
            if self.verbose >= 1:
                print("Using Class Weights:", np.round(class_weights, 4))
        else:
            criterion = nn.CrossEntropyLoss()
        
        # PyTorch Tensors for Training Set (CrossEntropyLoss requires 1D torch.long targets)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        
        generator = torch.Generator()
        generator.manual_seed(self.random_state)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, generator=generator
        )
        
        # Checkpointing setup (only if validation data is provided)
        do_checkpointing = (X_val is not None) and (y_val is not None)
        if do_checkpointing:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(self.device)
            self.best_val_f1_ = -1.0
            best_model_weights = copy.deepcopy(self.model.state_dict())

        # --- TRAINING LOOP ---
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * x_batch.size(0)

            # Step the scheduler AFTER each epoch
            scheduler.step()

            epoch_train_loss = running_loss / len(X)
            self.history_['train_loss'].append(epoch_train_loss)
            current_lr = optimizer.param_groups[0]['lr']
                
            # --- VALIDATION & CHECKPOINTING ---
            if do_checkpointing:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(X_val_tensor)
                    val_loss = criterion(val_logits, y_val_tensor).item()
                    val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()
                    val_f1 = f1_score(y_val, val_preds, average='macro')
                    
                    self.history_['val_loss'].append(val_loss)
                    self.history_['val_macro_f1'].append(val_f1)
                
                if self.verbose >= 1:
                    print(f"Epoch {epoch+1:02d}/{self.epochs} | LR: {current_lr:.6f} | Train Loss: {epoch_train_loss:.4f} | Val F1: {val_f1:.4f}")
                    
                # Checkpointing based on Macro F1
                if val_f1 > self.best_val_f1_:
                    self.best_val_f1_ = val_f1
                    best_model_weights = copy.deepcopy(self.model.state_dict())
            else:
                if self.verbose >= 1:
                    print(f"Epoch {epoch+1:02d}/{self.epochs} | LR: {current_lr:.6f} | Train Loss: {epoch_train_loss:.4f}")

        # Load the best weights if checkpointing was used
        if do_checkpointing:
            self.model.load_state_dict(best_model_weights)
            if self.verbose >= 1:
                print(f"\nTraining completed. Best Validation Macro F1: {self.best_val_f1_:.4f}")        

        return self

    def predict_proba(self, X):
        """
        Calculates class probabilities of shape (N, num_classes) using Softmax.
        """
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            
        return probs
    
    def predict(self, X):
        """
        Predicts the class label (0-9) with the highest probability.
        """
        probs = self.predict_proba(X)
        class_indices = np.argmax(probs, axis=1)
        return self.classes_[class_indices]