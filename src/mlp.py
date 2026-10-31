# src/mlp_scratch.py
import numpy as np

class MLPFromScratch:
    """
    Multi-Layer Perceptron (1 Hidden Layer) from scratch using NumPy.
    Supports mini-batch Gradient Descent, ReLU activation for the hidden layer,
    and Softmax for the output layer.
    """
    def __init__(self, input_size, hidden_layers, num_classes, learning_rate=0.01, random_state=42, init_method="random", activation_func = "relu"):

        self.lr = learning_rate
        np.random.seed(random_state)
        self.activation_function = activation_func

        self.layers_sizes = [input_size] + hidden_layers + [num_classes]
        self.num_layers = len(self.layers_sizes) - 1 # minus the input layer

        self.W = []
        self.b = []

        self._initialize_weights(method=init_method)


    def _initialize_weights(self, method):

        for i in range(self.num_layers):
            input_dimension = self.layers_sizes[i]
            output_dimension = self.layers_sizes[i+1]

            is_output_layer = (i== self.num_layers - 1)

            if method == "zeros":
                weight_matrix = np.zeros((input_dimension,output_dimension))
            elif method == "random":
                weight_matrix = np.random.randn(input_dimension,output_dimension) * 0.01  # multiply to keep the weights small
            elif method == "Xavier":
                weight_matrix = np.random.randn(input_dimension,output_dimension) * np.sqrt(1. / input_dimension)
            elif method == "he":
                weight_matrix = np.random.randn(input_dimension,output_dimension) * np.sqrt(2. / input_dimension)
            elif method == "optimal":
                if is_output_layer:
                    weight_matrix = np.random.randn(input_dimension,output_dimension) * np.sqrt(1. / input_dimension)
                else :
                    weight_matrix = np.random.randn(input_dimension,output_dimension) * np.sqrt(2. / input_dimension)

            bias_vector = np.zeros((1,output_dimension))
            self.W.append(weight_matrix)
            self.b.append(bias_vector)
                    
    def relu(self, Z):
        """ReLU activation function."""
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        """Derivative of ReLU."""
        return (Z > 0).astype(float)

    def softmax(self, Z):
        """
        Softmax activation function with numerical stability.
        Subtracting the max value prevents exponential overflow.
        """
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def compute_ce_loss(self, Y_pred, Y_true_one_hot):
        m = Y_true_one_hot.shape[0]
        Y_pred = np.clip(Y_pred, 1e-15, 1 - 1e-15)
        loss = -np.sum(Y_true_one_hot * np.log(Y_pred)) / m
        return loss

    def forward(self,X):

        self.A = [X]
        self.Z = []

        for i in range(self.num_layers):

            # Projection
            z = np.dot(self.A[i],self.W[i]) + self.b[i]
            self.Z.append(z)

            # Activation Function
            if i == self.num_layers - 1:
                self.A.append(self.softmax(z))
            else:
                self.A.append(self.relu(z))

        # Return final prediction
        return self.A[-1]

    def backward(self, X, Y_true):

        N = X.shape[0]

        dW = [None]*self.num_layers
        db = [None]*self.num_layers

        # Output Layer Error
        dZ = self.A[-1] - Y_true

        # Backpropgation
        for i in reversed(range(self.num_layers)):

            dW[i] = np.dot(self.A[i].T , dZ)/N
            db[i] = np.sum(dZ, axis=0, keepdims=True)/N

            # If we are not in the first layer we send the error backwards
            if i>0:
                dA_prev = np.dot(dZ,self.W[i].T)
                dZ = dA_prev * self.relu_derivative(self.Z[i-1])


        # Update Weights
        for i in range(self.num_layers):
            self.W[i]-=self.lr * dW[i]
            self.b[i]-=self.lr * db[i]


    def fit(self, X, y, epochs=10, batch_size=32, X_val=None, y_val=None):
        num_samples = X.shape[0]
        num_classes = self.W[-1].shape[1]
        
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        for epoch in range(epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                y_batch_one_hot = np.zeros((y_batch.size, num_classes))
                y_batch_one_hot[np.arange(y_batch.size), y_batch] = 1
                
                self.forward(X_batch)
                self.backward(X_batch, y_batch_one_hot)
            
            train_loss, train_acc = self.evaluate(X, y)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate(X_val, y_val)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                print(f"Epoch {epoch+1:03d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc:.2f}%")
            else:
                print(f"Epoch {epoch+1:03d}/{epochs} | Train Loss: {train_loss:.4f}")
                
        return history

    def evaluate(self, X, y):
        probs = self.forward(X)
        num_classes = self.W[-1].shape[1]
        
        y_one_hot = np.zeros((y.size, num_classes))
        y_one_hot[np.arange(y.size), y] = 1
        
        loss = self.compute_ce_loss(probs, y_one_hot)
        predictions = np.argmax(probs, axis=1)
        accuracy = np.mean(predictions == y) * 100
        
        return loss, accuracy


