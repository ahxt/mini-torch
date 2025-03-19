import numpy as np
from typing import List, Tuple, Union, Optional

np.random.seed(42)

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
    def __init__(self, in_features: int, out_features: int, init_weight=None, init_bias=None):
        super().__init__()
        if init_weight is not None:
            self.weight = Tensor(init_weight, requires_grad=True)
        else:
            self.weight = Tensor(
                np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features),
                requires_grad=True
            )
            
        if init_bias is not None:
            self.bias = Tensor(init_bias, requires_grad=True)
        else:
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

if __name__ == "__main__":
    layer1_weight = np.array([
        [ 0.5773503,  0.5773503, -0.5773503,  0.5773503],
        [-0.5773503,  0.5773503,  0.5773503,  0.5773503]
    ])
    layer1_bias = np.array([0., 0., 0., 0.])
    layer2_weight = np.array([
        [ 0.5,  0.5,  0.5],
        [-0.5,  0.5,  0.5],
        [ 0.5, -0.5,  0.5],
        [-0.5, -0.5,  0.5]
    ])
    layer2_bias = np.array([0., 0., 0.])

    model = Model()
    linear1 = Linear(2, 4, init_weight=layer1_weight, init_bias=layer1_bias)
    sigmoid1 = Sigmoid()
    linear2 = Linear(4, 3, init_weight=layer2_weight, init_bias=layer2_bias)
    model.add(linear1)
    model.add(sigmoid1)
    model.add(linear2)
    
    optimizer = SGD(model.parameters(), lr=0.1)
    criterion = SoftmaxCrossEntropyLoss()
    
    x = Tensor(np.array([[1.0, 2.0], [0.5, 1.5], [2.0, 1.0]]), requires_grad=True)
    target = Tensor(np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]))
    
    for epoch in range(5000):
        logits = model.forward(x)
        loss, grad = criterion(logits, target)
        optimizer.zero_grad()
        model.backward(grad)
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            exp_logits = np.exp(logits.data - np.max(logits.data, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            print(f"Epoch {epoch + 1}")
            print("Loss:", loss.data)
            print("Predictions (probabilities):")
            print(probs)
            print("---")