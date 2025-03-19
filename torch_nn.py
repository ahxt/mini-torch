import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define the model
class MultiLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        # First layer: 2 inputs -> 4 hidden units
        self.layer1 = nn.Linear(2, 4)
        self.sigmoid = nn.Sigmoid()
        # Second layer: 4 hidden units -> 3 output classes
        self.layer2 = nn.Linear(4, 3)
        
        # Initialize with fixed weights (transposed to match PyTorch's format)
        layer1_weight = np.array([
            [ 0.5773503, -0.5773503],
            [ 0.5773503,  0.5773503],
            [-0.5773503,  0.5773503],
            [ 0.5773503,  0.5773503]
        ])  # Shape: (4, 2)
        layer1_bias = np.array([0., 0., 0., 0.])
        layer2_weight = np.array([
            [ 0.5, -0.5,  0.5, -0.5],
            [ 0.5,  0.5, -0.5, -0.5],
            [ 0.5,  0.5,  0.5,  0.5]
        ])  # Shape: (3, 4)
        layer2_bias = np.array([0., 0., 0.])
        
        # Set the weights and biases
        with torch.no_grad():
            self.layer1.weight.copy_(torch.from_numpy(layer1_weight))
            self.layer1.bias.copy_(torch.from_numpy(layer1_bias))
            self.layer2.weight.copy_(torch.from_numpy(layer2_weight))
            self.layer2.bias.copy_(torch.from_numpy(layer2_bias))
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.sigmoid(x)
        x = self.layer2(x)
        return x

class CustomCrossEntropyLoss:
    def __call__(self, pred, target):
        batch_size = pred.shape[0]
        # Compute softmax
        exp_pred = torch.exp(pred - torch.max(pred, dim=1, keepdim=True)[0])
        softmax_pred = exp_pred / torch.sum(exp_pred, dim=1, keepdim=True)
        
        # Compute cross entropy loss
        loss = -torch.sum(target * torch.log(softmax_pred + 1e-7)) / batch_size
        
        # Compute gradients
        grad = (softmax_pred - target) / batch_size
        
        return loss, grad

# Create model, loss, and optimizer
model = MultiLayerNet()

# Print initial weights and biases to verify
print("Initial weights and biases:")
print("Layer 1 weights:")
print(model.layer1.weight.data.numpy())
print("Layer 1 bias:")
print(model.layer1.bias.data.numpy())
print("Layer 2 weights:")
print(model.layer2.weight.data.numpy())
print("Layer 2 bias:")
print(model.layer2.bias.data.numpy())
print("---")

criterion = CustomCrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training data
x = torch.tensor([[1.0, 2.0], [0.5, 1.5], [2.0, 1.0]], requires_grad=True)
# One-hot encoded targets
target = torch.tensor([
    [1.0, 0.0, 0.0],  # First sample is class 0
    [0.0, 1.0, 0.0],  # Second sample is class 1
    [0.0, 0.0, 1.0]   # Third sample is class 2
])

# Training loop
for epoch in range(5000):
    # Forward pass
    logits = model(x)
    
    # Compute loss and gradients
    loss, grad = criterion(logits, target)
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Manual backward pass
    logits.backward(grad)
    
    # Update parameters
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        # Convert logits to probabilities
        probs = torch.softmax(logits.detach(), dim=1)
        
        print(f"Epoch {epoch + 1}")
        print("Loss:", loss.item())
        print("Predictions (probabilities):")
        print(probs.numpy())
        print("---") 