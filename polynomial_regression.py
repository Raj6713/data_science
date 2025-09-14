import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.linspace(-3,3,200).reshape(-1,1)
y = 0.5*X**3-2*X**2 + X + np.random.normal(0,2,X.shape)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

degree = 3
X_poly = torch.cat([X**i for i in range(1, degree+1)], dim=1)  # [x, x², x³]


class PolynomialRegression(nn.Module):
    def __init__(self, degree):
        super(PolynomialRegression,self).__init__()
        self.linear = nn.Linear(degree,1)
    
    def forward(self, x):
        return self.linear(x)
model = PolynomialRegression(degree)


criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 1000
for epoch in range(epochs):
    # Forward pass
    y_pred = model(X_poly)
    loss = criterion(y_pred, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")


with torch.no_grad():
    y_pred = model(X_poly)
    print(y_pred)
plt.scatter(X.numpy(), y.numpy(), color="blue", label="Data")
plt.plot(X.numpy(), y_pred.numpy(), color="red", linewidth=2, label="Model Prediction")
plt.legend()
plt.show()