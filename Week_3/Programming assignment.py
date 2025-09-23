import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# Data Preparation
def runge_function(x):
    return 1.0 / (1 + 25 * x**2)

def runge_derivative(x):
    return -50 * x / (1 + 25 * x**2)**2

# Sample points
np.random.seed(0)
x_train = np.linspace(-1, 1, 1200)
y_train = runge_function(x_train)
dy_train = runge_derivative(x_train)
x_val = np.linspace(-1, 1, 300)
y_val = runge_function(x_val)
dy_val = runge_derivative(x_val)

# PyTorch tensors
x_train_tensor = torch.FloatTensor(x_train).view(-1, 1)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
dy_train_tensor = torch.FloatTensor(dy_train).view(-1, 1)
x_val_tensor = torch.FloatTensor(x_val).view(-1, 1)
y_val_tensor = torch.FloatTensor(y_val).view(-1, 1)
dy_val_tensor = torch.FloatTensor(dy_val).view(-1, 1)

train_ds = TensorDataset(x_train_tensor, y_train_tensor, dy_train_tensor)
val_ds = TensorDataset(x_val_tensor, y_val_tensor, dy_val_tensor)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

# Model definition (tanh activation)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training loop
num_epochs = 100
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    net.train()
    batch_train_losses = []
    for xb, yb, dyb in train_loader:
        xb_ = xb.clone().detach().requires_grad_(True)
        optimizer.zero_grad()
        out = net(xb_)
        loss_f = criterion(out, yb)
        grad_out = torch.autograd.grad(
            outputs=out,
            inputs=xb_,
            grad_outputs=torch.ones_like(out),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        loss_df = criterion(grad_out, dyb)
        loss = loss_f + loss_df
        loss.backward()
        optimizer.step()
        batch_train_losses.append(loss.item())
    train_losses.append(np.mean(batch_train_losses))

    net.eval()
    with torch.enable_grad():
        x_val_tensor_ = x_val_tensor.clone().detach().requires_grad_(True)
        out_val = net(x_val_tensor_)
        val_loss_f = criterion(out_val, y_val_tensor)
        grad_val = torch.autograd.grad(
            outputs=out_val,
            inputs=x_val_tensor_,
            grad_outputs=torch.ones_like(out_val),
            create_graph=False,
            retain_graph=False,
            only_inputs=True
        )[0]
        val_loss_df = criterion(grad_val, dy_val_tensor)
        val_loss = val_loss_f + val_loss_df
        val_losses.append(val_loss.item())


x_plot = np.linspace(-1, 1, 400)
y_true = runge_function(x_plot)
dy_true = runge_derivative(x_plot)

x_plot_tensor = torch.FloatTensor(x_plot).view(-1,1).requires_grad_(True)
y_pred_tensor = net(x_plot_tensor)
y_pred = y_pred_tensor.detach().numpy()

dy_pred = torch.autograd.grad(
    outputs=y_pred_tensor,
    inputs=x_plot_tensor,
    grad_outputs=torch.ones_like(y_pred_tensor),
    create_graph=False
)[0].detach().numpy()

# Plotting results
plt.figure(figsize=(14, 10))

plt.subplot(3, 1, 1)
plt.plot(x_plot, y_true, label="True f(x)", color='blue', linewidth=2)
plt.plot(x_plot, y_pred, label="NN f(x)", color='red', linewidth=2, alpha=0.7)
plt.legend()
plt.title("Function: True vs Neural Network Prediction")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(x_plot, dy_true, label="True f'(x)", color='green', linewidth=2)
plt.plot(x_plot, dy_pred, label="NN f'(x)", color='orange', linewidth=2, alpha=0.7)
plt.legend()
plt.title("Derivative: True vs Neural Network Prediction")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(x_plot, y_true - y_pred.flatten(), label="Residual f(x)", color='purple', linewidth=2)
plt.plot(x_plot, dy_true - dy_pred.flatten(), label="Residual f'(x)", color='red', linewidth=2)
plt.legend()
plt.title("Residual Errors")
plt.grid(True)

plt.tight_layout()
plt.show()

# Error computation
mse_f = np.mean((y_true - y_pred.flatten())**2)
max_error_f = np.max(np.abs(y_true - y_pred.flatten()))
mse_df = np.mean((dy_true - dy_pred.flatten())**2)
max_error_df = np.max(np.abs(dy_true - dy_pred.flatten()))

print(f"MSE f(x): {mse_f:.6f}, Max Error f(x): {max_error_f:.5f}")
print(f"MSE f'(x): {mse_df:.6f}, Max Error f'(x): {max_error_df:.5f}")

plt.figure(figsize=(8, 4))
plt.plot(train_losses, label="Training Loss", color='blue', linewidth=2)
plt.plot(val_losses, label="Validation Loss", color='orange', linewaidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curves")
plt.legend()
plt.grid(True)
plt.show()
