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

criterion = MSELoss()
loss = criterion(y_pred, y_true)
print("MSE:", loss.item())