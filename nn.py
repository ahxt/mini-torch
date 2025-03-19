import numpy as np
from typing import List, Tuple

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

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return Tensor(np.maximum(0, x.data))

    def backward(self, grad: Tensor) -> Tensor:
        relu_derivative = (self.input.data > 0).astype(np.float32)
        return Tensor(grad.data * relu_derivative)

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

class Adam:
    def __init__(self, parameters: List[Tensor], lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        self.parameters = [p for p in parameters if p.requires_grad]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [np.zeros_like(param.data) for param in self.parameters]
        self.v = [np.zeros_like(param.data) for param in self.parameters]
        self.t = 0

    def zero_grad(self):
        for param in self.parameters:
            if param.requires_grad:
                param.grad = np.zeros_like(param.data)

    def step(self):
        self.t += 1
        for i, param in enumerate(self.parameters):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * np.square(param.grad)
            m_hat = self.m[i] / (1 - np.power(self.beta1, self.t))
            v_hat = self.v[i] / (1 - np.power(self.beta2, self.t))
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

def test_nn_correctness():
    import torch
    import torch.nn as nn
    import torch.optim as optim

    # np.random.seed(42)
    # torch.manual_seed(42)

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
    
    class PyTorchModel(nn.Module):
        def __init__(self, numpy_model):
            super().__init__()
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
    
    torch_model = PyTorchModel(model)
    torch_optimizer = optim.SGD(torch_model.parameters(), lr=1)
    torch_criterion = nn.CrossEntropyLoss()
    
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
            probs = np.exp(logits.data - np.max(logits.data, axis=1, keepdims=True))
            probs /= np.sum(probs, axis=1, keepdims=True)
            torch_probs = torch.softmax(torch_logits.detach(), dim=1).numpy()
            
            print(f"Epoch {epoch + 1}")
            print("NumPy Loss:", loss.data)
            print("PyTorch Loss:", torch_loss.item())
            print("NumPy Predictions:")
            print(probs)
            print("PyTorch Predictions:")
            print(torch_probs)
            print("---")


def train_mnist():
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    model = Model()
    model.add(Linear(784, 128))
    model.add(ReLU())
    # model.add(Sigmoid())
    model.add(Linear(128, 64))
    model.add(ReLU())
    # model.add(Sigmoid())
    model.add(Linear(64, 10))
    
    criterion = SoftmaxCrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    # optimizer = SGD(model.parameters(), lr=0.001)
    
    epochs = 10
    print("Starting training...")
    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            X = Tensor(data.view(-1, 784).numpy())
            y = Tensor(torch.zeros(target.shape[0], 10).scatter_(1, target.unsqueeze(1), 1).numpy())
            
            logits = model.forward(X)
            loss, grad = criterion(logits, y)
            
            optimizer.zero_grad()
            model.backward(grad)
            optimizer.step()
            
            epoch_loss += loss.data
            batch_count += 1
            
            
        avg_loss = epoch_loss / batch_count
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_loss:.4f}")
    
    correct = 0
    total = 0
    
    print("Evaluating model...")
    for data, target in test_loader:
        X = Tensor(data.view(-1, 784).numpy())
        logits = model.forward(X)
        predictions = np.argmax(logits.data, axis=1)
        
        total += len(target)
        correct += np.sum(predictions == target.numpy())
    
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    # test_nn_correctness()
    train_mnist()