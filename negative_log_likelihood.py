import torch

X = torch.tensor(
    [[1.0, 2.0], [2.0, 1.0], [2.0, 3.0], [3.0, 2.0], [3.0, 4.0], [4.0, 3.0]]
)
y = torch.tensor([0, 0, 1, 1, 2, 2])

num_samples, num_features = X.shape
num_classes = 3


torch.manual_seed(0)
W = torch.randn(num_features, num_classes) * 0.01
b = torch.zeros(1, num_classes)

lr = 0.1
epochs = 50

for epoch in range(epochs):
    logits = X @ W + b
    exp_logits = torch.exp(logits)
    probs = exp_logits / exp_logits.sum(dim=1, keepdim=True)
    true_class_probs = probs[torch.arange(num_samples), y]
    loss = -torch.log(true_class_probs).mean()
    y_onehot = torch.zeros_like(probs)
    y_onehot[torch.arange(num_samples), y] = 1.0

    dlogits = (probs - y_onehot) / num_samples
    dW = X.T @ dlogits
    db = dlogits.sum(dim=0, keepdim=True)
    W -= lr * dW
    b -= lr * db
    if (epoch + 1) % 10 == 0 or epoch == 0:
        preds = probs.argmax(dim=1)
        acc = (preds == y).float().mean().item()
        print(f"Epoch {epoch+1:02d} | Loss: {loss.item():.4f} | Acc: {acc:.2f}")
print("\nTrained Weights:\n", W)
print("Trained Bias:\n", b)
