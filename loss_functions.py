import torch
import torch.nn as nn


class CategoricalCrossEntropy(nn.Module):
    def __init__(self):
        super(CategoricalCrossEntropy, self).__init__()

    def forward(self, y_pred, y_true):
        log_probs = torch.log_softmax(y_pred, dim=1)
        loss = -log_probs[range(y_true.shape[0]), y_true]
        return loss.mean()


y_pred = torch.tensor(
    [[2.0, 1.0, 0.1, -1.2], [0.5, 2.3, -0.5, 0.0], [1.2, -0.3, 2.1, 0.8]],
    requires_grad=True,
)
y_true = torch.tensor([0, 1, 2])
criterion = CategoricalCrossEntropy()
loss = criterion(y_pred, y_true)
print("Custom CCE Loss:", loss.item())
exit()


def binary_cross_entropy(y_pred, y_true):
    eps = 1e-8
    y_pred = torch.clamp(y_pred, eps, 1 - eps)
    loss = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
    return loss.mean()


y_true = torch.tensor([1.0, 0.0, 1.0, 1.0])
y_logits = torch.tensor([2.0, -1.0, 1.5, -2.0])
y_pred = torch.sigmoid(y_logits)
loss_custom = binary_cross_entropy(y_pred, y_true)
bce_loss = nn.BCELoss()
loss_builtin = bce_loss(y_pred, y_true)
bce_logits_loss = nn.BCEWithLogitsLoss()
loss_stable = bce_logits_loss(y_logits, y_true)
print("Custom BCE Loss: ", loss_custom.item())
print("nn.BCELoss: ", loss_builtin.item())
print("nn.BCELWithLogitsLoss(recommended): ", loss_stable.item())
exit()


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        squared_diff = diff**2
        loss = torch.mean(squared_diff)
        return loss


criterion = MSELoss()
y_true = torch.tensor([1.0, 2.0, 3.0])
y_pred = torch.tensor([1.5, 2.5, 2.0])
loss = criterion(y_pred, y_true)
print("MSE:", loss.item())
# --------------------------------------------------


class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        mae_diff = torch.abs(diff)
        loss = torch.mean(mae_diff)
        return loss


criterion = MAELoss()
loss = criterion(y_pred, y_true)
print("MAE: ", loss.item())


class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, y_pred, y_true):
        error = y_pred - y_true
        abs_error = torch.abs(error)

        # Quadratic for small errors
        quadratic = torch.minimum(
            abs_error, torch.tensor(self.delta, device=error.device)
        )
        linear = abs_error - quadratic

        loss = 0.5 * quadratic**2 + self.delta * linear
        return torch.mean(loss)


class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, y_pred, y_true):
        error = y_true - y_pred
        abs_error = torch.abs(error)
        quadtric = torch.minimum(
            abs_error, torch.tensor(self.delta, device=error.device)
        )
        linear = abs_error - quadtric
        loss = 0.5 * quadtric**2 + self.delta * linear
        return torch.mean(loss)


criterion = HuberLoss()
y_true = torch.tensor([1.0, 2.0, 3.0])
y_pred = torch.tensor([1.5, 2.5, 2.0])
loss = criterion(y_pred, y_true)
print("Huber Loss: ", loss.item())
