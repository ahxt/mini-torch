import numpy as np
from typing import List, Tuple, Union, Optional

class Tensor:
    def __init__(self, data: Union[np.ndarray, list, float], requires_grad: bool = False):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None if requires_grad else None

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.data + other.data)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.data * other.data)

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.data @ other.data)

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

class Module:
    def __init__(self):
        self._parameters = []
        self.input = None
        self.output = None

    def parameters(self) -> List[Tensor]:
        return self._parameters

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
        # Initialize weights using He initialization
        self.weight = Tensor(
            np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features),
            requires_grad=True
        )
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)
        self._parameters = [self.weight, self.bias]

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias

    def backward(self, grad: Tensor) -> Tensor:
        # Gradient for bias
        self.bias.grad = np.sum(grad.data, axis=0)
        
        # Gradient for weights
        self.weight.grad = self.input.data.T @ grad.data
        
        # Gradient for input
        input_grad = grad.data @ self.weight.data.T
        return Tensor(input_grad)

class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return Tensor(1 / (1 + np.exp(-x.data)))

    def backward(self, grad: Tensor) -> Tensor:
        # Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
        sigmoid_derivative = self.output.data * (1 - self.output.data)
        return Tensor(grad.data * sigmoid_derivative)

class CrossEntropyLoss:
    def __call__(self, pred: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        # Compute sigmoid if not already applied
        if isinstance(pred, Tensor):
            pred_data = pred.data
        else:
            pred_data = pred
            
        # Clip predictions to avoid log(0)
        eps = 1e-7
        pred_data = np.clip(pred_data, eps, 1 - eps)
        
        # Compute binary cross entropy loss
        loss = -(target.data * np.log(pred_data) + (1 - target.data) * np.log(1 - pred_data))
        loss = loss.mean()
        
        # Compute gradients
        grad = (pred_data - target.data) / pred_data.shape[0]
        
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
            params.extend(module.parameters())
        return params

    def forward(self, x: Tensor) -> Tensor:
        current = x
        for module in self.modules:
            current = module(current)
        return current

    def backward(self, grad: Tensor):
        current_grad = grad
        for module in reversed(self.modules):
            current_grad = module.backward(current_grad)

class SGD:
    def __init__(self, parameters: List[Tensor], lr: float = 0.01):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()

    def step(self):
        for param in self.parameters:
            if param.grad is None:
                continue
            param.data -= self.lr * param.grad

# Example usage:
if __name__ == "__main__":
    # Create a multi-layer model
    model = Model()
    
    # First layer: 2 inputs -> 4 hidden units
    linear1 = Linear(2, 4)
    sigmoid1 = Sigmoid()
    model.add(linear1)
    model.add(sigmoid1)
    
    # Second layer: 4 hidden units -> 2 hidden units
    linear2 = Linear(4, 2)
    sigmoid2 = Sigmoid()
    model.add(linear2)
    model.add(sigmoid2)
    
    # Output layer: 2 hidden units -> 1 output
    linear3 = Linear(2, 1)
    sigmoid3 = Sigmoid()
    model.add(linear3)
    model.add(sigmoid3)
    
    # Create optimizer with smaller learning rate for deeper network
    optimizer = SGD(model.parameters(), lr=0.1)
    
    # Create loss function
    criterion = CrossEntropyLoss()
    
    # Training data
    x = Tensor([[1.0, 2.0]], requires_grad=True)  # Input
    target = Tensor([[1.0]])  # Target (binary classification)
    
    # Training loop
    for epoch in range(100):  # More epochs for deeper network
        # Forward pass
        y = model.forward(x)
        
        # Compute loss and gradients
        loss, grad = criterion(y, target)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Backward pass
        model.backward(grad)
        
        # Update parameters
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:  # Print every 10 epochs
            print(f"Epoch {epoch + 1}")
            print("Loss:", loss.data)
            print("Prediction:", y.data)
            print("---") 