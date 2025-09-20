# import torch
# import torch.nn as nn
import torch
import torch.nn as nn

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
#--------------------------------------------------

class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()
        
    def forward(self, y_pred, y_true):
        diff = y_pred-y_true
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
        quadratic = torch.minimum(abs_error, torch.tensor(self.delta, device=error.device))
        linear = abs_error - quadratic
        
        loss = 0.5 * quadratic**2 + self.delta * linear
        return torch.mean(loss)

class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta
        
    def forward(self, y_pred,y_true):
        error = y_true-y_pred
        abs_error = torch.abs(error)
        quadtric = torch.minimum(abs_error, torch.tensor(self.delta, device=error.device))
        linear = abs_error - quadtric
        loss = 0.5*quadtric**2 + self.delta*linear
        return torch.mean(loss)

criterion = HuberLoss()
y_true = torch.tensor([1.0, 2.0, 3.0])
y_pred = torch.tensor([1.5, 2.5, 2.0])
loss = criterion(y_pred, y_true)
print("Huber Loss: ", loss.item())