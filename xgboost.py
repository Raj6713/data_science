# import torch
# import math
# from collections import Counter, namedtuple

# # Simple Node structure
# Node = namedtuple("Node", ["feature", "threshold", "left", "right", "value", "is_leaf"])
import torch
import math
from collections import Counter, namedtuple
Node = namedtuple("Node", ["feature", "threshold", "left", "right", "value", "is_leaf"])

# class DecisionTreeTorch:
#     def __init__(self, max_depth=3, min_samples_split=2):
#         self.max_depth = max_depth
#         self.min_samples_split = min_samples_split
#         self.root = None
class DecisionTreeTorch:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        
    def fit(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if depth > self.max_depth or n_samples < self.min_samples_split or torch.var(y)==0:
            return Node(None, None, None, None,torch.mean(y).item(), True)
        
        best_feature, best_threshold, best_loss = None, None, float("inf")
        for feature in range(n_features):
            thresholds = torch.unique(X[:,feature])
            for threshold in thresholds:
                left_idx = X[:, feature] <=threshold
                right_idx = ~left_idx
                if left_idx.sum() == 0 or right_idx.sum() == 0:
                    continue
                loss = torch.var(y[left_idx])*left_idx.sum() + torch.var(y[right_idx])*right_idx.sum()
                if loss < best_loss:
                    best_loss = loss
                    best_feature = feature
                    best_threshold = threshold
        if best_feature is None:
            return Node(None, None, None, None, torch.mean(y).item(), True)
        left_idx = X[:, best_feature] <= best_threshold
        right_idx = ~left_idx
        left = self.fit(X[left_idx], y[left_idx], depth+1)
        right = self.fit(X[right_idx], y[right_idx], depth+1)
        return Node(best_feature, best_threshold, left, right, None, False)
    
    def train(self, X, y):
        self.root = self.fit(X,y)
        
    def predict_one(self, x, node=None):
        if node is None:
            node = self.root
        if node.is_leaf:
            return node.value
        if x[node.feature] <=node.threshold:
            return self.predict_one(x, node.left)
        else:
            return self.predict_one(x, node.right)
        
    def predict(self,X):
        return torch.tensor([self.predict_one(x) for x in X])

class XGBoostTorch:
    def __init__(self, n_estimators=50,learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        
    def fit(self,X,y):
        X,y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        y_pred = torch.zeros_like(y)
        for _ in range(self.n_estimators):
            residuals = y - y_pred
            tree = DecisionTreeTorch(max_depth=self.max_depth)
            tree.train(X, residuals)
            update = tree.predict(X)
            y_pred += self.learning_rate*update
            self.trees.append(tree)

    def predict(self,X):
        X = torch.tensor(X, dtype=torch.float32)
        y_pred = torch.zeros(X.shape[0])
        for tree in self.trees:
            y_pred += self.learning_rate*tree.predict(X)
        return y_pred
    
# Simple regression dataset
X = torch.linspace(0, 10, 100).reshape(-1, 1)
y = torch.sin(X).squeeze() + 0.1 * torch.randn(100)

model = XGBoostTorch(n_estimators=50, learning_rate=0.1, max_depth=3)
model.fit(X, y)

y_pred = model.predict(X)
for item1, item2 in zip(y_pred[:10], y[:10]):
    print(item1,item2)

