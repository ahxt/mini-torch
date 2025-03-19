import numpy as np
from typing import List, Tuple, Union, Optional

class Tensor:
    def __init__(self, data: np.ndarray, requires_grad: bool = False):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None if requires_grad else None

    def __add__(self, other):
        return Tensor(self.data + other.data)

    def __mul__(self, other):
        return Tensor(self.data * other.data)

    def __matmul__(self, other):
        return Tensor(self.data @ other.data)

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

class Module:
    def __init__(self):
        self._parameters = []
        self.input = None
        self.output = None

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, x: Tensor) -> Tensor:
        self.input = x
        self.output = self.forward(x)
        return self.output

class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = Tensor(
            np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features),
            requires_grad=True
        )
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)
        self._parameters = [self.weight, self.bias]

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias

    def backward(self, grad: Tensor) -> Tensor:
        self.bias.grad = np.sum(grad.data, axis=0)
        self.weight.grad = self.input.data.T @ grad.data
        input_grad = grad.data @ self.weight.data.T
        return Tensor(input_grad)

class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return Tensor(1 / (1 + np.exp(-x.data)))

    def backward(self, grad: Tensor) -> Tensor:
        sigmoid_derivative = self.output.data * (1 - self.output.data)
        return Tensor(grad.data * sigmoid_derivative)

class SoftmaxCrossEntropyLoss:
    def __call__(self, logits: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = logits.data.shape[0]
        exp_logits = np.exp(logits.data - np.max(logits.data, axis=1, keepdims=True))
        softmax_pred = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        loss = -np.sum(target.data * np.log(softmax_pred + 1e-7)) / batch_size

        grad = (softmax_pred - target.data) / batch_size
        return Tensor(loss), Tensor(grad)

class Model:
    def __init__(self):
        self.modules = []

    def add(self, module: Module):
        self.modules.append(module)
        return self

    def parameters(self) -> List[Tensor]:
        params = []
        for module in self.modules:
            params.extend(module._parameters)
        return params

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x

    def backward(self, grad: Tensor):
        for module in reversed(self.modules):
            grad = module.backward(grad)

class SGD:
    def __init__(self, parameters: List[Tensor], lr: float = 0.01):
        self.parameters = [p for p in parameters if p.requires_grad]
        self.lr = lr

    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()

    def step(self):
        for param in self.parameters:
            param.data -= self.lr * param.grad

def test_nn_correctness():
    import torch
    import torch.nn as nn
    import torch.optim as optim

    # np.random.seed(42)
    # torch.manual_seed(42)

    # Custom numpy implementation
    model = Model()
    model.add(Linear(2, 4))
    model.add(Sigmoid())
    model.add(Linear(4, 3))
    
    optimizer = SGD(model.parameters(), lr=1)
    criterion = SoftmaxCrossEntropyLoss()
    
    x = Tensor(np.array([[1.0, 2.0], [0.5, 1.5], [2.0, 1.0]]), requires_grad=True)
    target = Tensor(np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]))
    
    # PyTorch implementation with same weights
    class PyTorchModel(nn.Module):
        def __init__(self, numpy_model):
            super().__init__()
            # Copy weights from numpy model
            self.layer1 = nn.Linear(2, 4)
            with torch.no_grad():
                self.layer1.weight = nn.Parameter(torch.tensor(numpy_model.modules[0].weight.data.T, dtype=torch.float32))
                self.layer1.bias = nn.Parameter(torch.tensor(numpy_model.modules[0].bias.data, dtype=torch.float32))
            
            self.sigmoid = nn.Sigmoid()
            
            self.layer2 = nn.Linear(4, 3)
            with torch.no_grad():
                self.layer2.weight = nn.Parameter(torch.tensor(numpy_model.modules[2].weight.data.T, dtype=torch.float32))
                self.layer2.bias = nn.Parameter(torch.tensor(numpy_model.modules[2].bias.data, dtype=torch.float32))
        
        def forward(self, x):
            x = self.layer1(x)
            x = self.sigmoid(x)
            x = self.layer2(x)
            return x
    
    # Create PyTorch model with same weights
    torch_model = PyTorchModel(model)
    torch_optimizer = optim.SGD(torch_model.parameters(), lr=1)
    torch_criterion = nn.CrossEntropyLoss()
    
    # Convert data to PyTorch tensors
    torch_x = torch.tensor(x.data, dtype=torch.float32)
    torch_target = torch.tensor(np.argmax(target.data, axis=1), dtype=torch.long)
    
    print("Starting training comparison:")
    print("=" * 50)
    
    for epoch in range(500):
        # NumPy implementation
        logits = model.forward(x)
        loss, grad = criterion(logits, target)
        
        optimizer.zero_grad()
        model.backward(grad)
        optimizer.step()
        
        # PyTorch implementation
        torch_logits = torch_model(torch_x)
        torch_loss = torch_criterion(torch_logits, torch_target)
        
        torch_optimizer.zero_grad()
        torch_loss.backward()
        torch_optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            # NumPy results
            probs = np.exp(logits.data - np.max(logits.data, axis=1, keepdims=True))
            probs /= np.sum(probs, axis=1, keepdims=True)
            
            # PyTorch results
            torch_probs = torch.softmax(torch_logits.detach(), dim=1).numpy()
            
            print(f"Epoch {epoch + 1}")
            print("NumPy Loss:", loss.data)
            print("PyTorch Loss:", torch_loss.item())
            print("NumPy Predictions:")
            print(probs)
            print("PyTorch Predictions:")
            print(torch_probs)
            print("---")

# if __name__ == "__main__":
#     test_nn_correctness()


def train_mnist():
    """
    Train a simple neural network on the MNIST dataset using our custom implementation.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X, y = mnist.data.astype('float32'), mnist.target.astype('int')
    
    # Normalize data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to one-hot encoding
    def to_one_hot(y, num_classes=10):
        one_hot = np.zeros((y.shape[0], num_classes))
        for i, label in enumerate(y):
            one_hot[i, int(label)] = 1.0
        return one_hot
    
    y_train_one_hot = to_one_hot(y_train)
    
    # Create model using the Model class
    model = Model()
    model.add(Linear(784, 128))
    model.add(Sigmoid())
    model.add(Linear(128, 64))
    model.add(Sigmoid())
    model.add(Linear(64, 10))
    
    criterion = SoftmaxCrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.1)
    
    # Training loop
    batch_size = 64
    epochs = 100
    n_samples = X_train.shape[0]
    n_batches = n_samples // batch_size
    
    train_losses = []
    
    print("Starting training...")
    for epoch in range(epochs):
        epoch_loss = 0
        
        # Shuffle data
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train_one_hot[indices]
        
        for batch in range(n_batches):
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size
            
            # Get batch
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            
            # Convert to tensors
            X_batch_tensor = Tensor(X_batch)
            y_batch_tensor = Tensor(y_batch)
            
            # Forward pass
            logits = model.forward(X_batch_tensor)
            loss, grad = criterion(logits, y_batch_tensor)
            
            # Backward pass
            optimizer.zero_grad()
            model.backward(grad)
            optimizer.step()
            
            epoch_loss += loss.data
            
        avg_loss = epoch_loss / n_batches
        train_losses.append(avg_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Evaluate on test set
    correct = 0
    total = 0
    
    # Process in batches to avoid memory issues
    test_batch_size = 100
    n_test_batches = len(X_test) // test_batch_size
    
    for i in range(n_test_batches):
        start_idx = i * test_batch_size
        end_idx = start_idx + test_batch_size
        
        X_test_batch = X_test[start_idx:end_idx]
        y_test_batch = y_test[start_idx:end_idx]
        
        # Forward pass
        logits = model.forward(Tensor(X_test_batch))
        
        # Get predictions
        predictions = np.argmax(logits.data, axis=1)
        
        # Update metrics
        total += len(y_test_batch)
        correct += np.sum(predictions == y_test_batch.astype(int))
    
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

if __name__ == "__main__":
    test_nn_correctness()
    train_mnist()