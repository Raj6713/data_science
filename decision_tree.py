import torch
import math
from collections import Counter, namedtuple

Node = namedtuple(
    "Node", ["features", "threshold", "left", "right", "value", "is_leaf"]
)


class DecisionTreeTorch:
    def __init__(
        self, max_depth=10, min_samples_split=2, criterion="gini", device=None
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        assert criterion in ("gini", "entropy")
        self.criterion = criterion
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.root = None
        self.n_classes_ = None

    def fit(self, X, y):
        X = self._as_tensor(X).to(self.device)
        y = self._as_tensor(y).to(self.device)
        if y.dim() == 2 and y.size(1) == 1:
            y = y.view(-1)
        classes = torch.unique(y)
        print(classes)
        self.class_map = {c: i for i, c in enumerate(sorted(classes.tolist()))}
        mapped = torch.tensor(
            [self.class_map[int(val)] for val in y],
            device=self.device,
            dtype=torch.long,
        )
        self.n_classes_ = len(self.class_map)
        self.root = self._build_tree(X, mapped, depth=0)
        return self

    def predict(self, X):
        X = self._as_tensor(X).to(self.device)
        preds = []
        for i in range(X.size(0)):
            node = self.root
            xi = X[i]
            while not node.is_leaf:
                val = xi[node.features]
                if val <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            preds.append(node.value)
        return torch.tensor(preds, device=self.device, dtype=torch.long)

    def predict_proba(self, X):
        X = self._as_tensor(X).to(self.device)
        probs = []
        for i in range(X.size(0)):
            node = self.root
            xi = X[i]
            while not node.is_leaf:
                val = xi[node.features]
                if val <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            probs.append(node.value_proba)
        return torch.stack(probs)

    def _as_tensor(self, arr, dtype=torch.float32):
        if isinstance(arr, torch.Tensor):
            if dtype is not None:
                return arr.to(dtype)
            return arr
        else:
            return torch.tensor(arr, dtype=dtype)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.size()
        num_samples_per_class = torch.bincount(y, minlength=self.n_classes_)
        predicted_class = torch.argmax(num_samples_per_class).item()
        value_proba = (num_samples_per_class.float() / n_samples).to(self.device)
        if (
            (depth >= self.max_depth)
            or (n_samples < self.min_samples_split)
            or (num_samples_per_class.max().item() == n_samples)
        ):
            leaf = Node(
                features=None,
                threshold=None,
                left=None,
                right=None,
                value=predicted_class,
                is_leaf=True,
            )
            leaf = leaf._replace(value_proba=value_proba)
            return leaf
        feature, threshold, gain = self._best_split(X, y)
        if feature is None:
            leaf = Node(
                features=None,
                threshold=None,
                left=None,
                right=None,
                value=predicted_class,
                is_leaf=True,
            )
            return leaf
        mask_left = X[:, feature] <= threshold
        X_left, y_left = X[mask_left], y[mask_left]
        X_right, y_right = X[~mask_left], y[~mask_left]
        left_node = self._build_tree(X_left, y_left, depth + 1)
        right_node = self._build_tree(X_right, y_right, depth + 1)
        node = Node(
            features=int(feature),
            threshold=float(threshold),
            left=left_node,
            right=right_node,
            value=None,
            is_leaf=False,
        )
        node = node._replace(value_proba=value_proba)
        return node

    def _best_split(self, X, y):
        n_samples, n_features = X.size()
        if n_samples <= 1:
            return None, None, None
        parent_impurity = self._impurity(y)
        best_gain = 0.0
        best_feat = None
        best_thresh = None

        for feature in range(n_features):
            xi = X[:, feature]
            sorted_vals, indices = torch.sort(xi)
            sorted_labels = y[indices]

            unique_vals, inv_idx, counts = torch.unique_consecutive(
                sorted_vals, return_inverse=True, return_counts=True
            )
            if unique_vals.size(0) == 1:
                continue
            prefix_counts = torch.zeros(
                (self.n_classes_, n_samples), device=self.device, dtype=torch.int32
            )
            for cls in range(self.n_classes_):
                cls_mask = (sorted_labels == cls).to(torch.int32)
                prefix_counts[cls] = torch.cumsum(cls_mask, dim=0)
            diffs = sorted_vals[:-1] != sorted_vals[1:]
            candidate_pos = torch.nonzero(diffs, as_tuple=False).view(-1)
            if candidate_pos.numel() == 0:
                continue
            for pos in candidate_pos:
                left_counts = prefix_counts[:, pos].to(torch.float32)
                n_left = (pos + 1).item()
                n_right = n_samples - n_left
                if n_left < self.min_samples_split or n_right < self.min_samples_split:
                    continue
                right_counts = (prefix_counts[:, -1] - prefix_counts[:, pos]).to(
                    torch.float32
                )
                left_imp = self._impurity_from_counts(left_counts, n_left)
                right_imp = self._impurity_from_counts(right_counts, n_right)
                weighted_imp = (n_left * left_imp + n_right * right_imp) / n_samples
                gain = parent_impurity - weighted_imp
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feature
                    best_thresh = float((sorted_vals[pos] + sorted_vals[pos + 1]) / 2.0)
        return best_feat, best_thresh, best_gain

    def _impurity(self, labels):
        counts = torch.bincount(labels, minlength=self.n_classes_).to(torch.float32)
        n = labels.size(0)
        return self._impurity_from_counts(counts, n)

    def _impurity_from_counts(self, counts, n):
        if n == 0:
            return 0.0
        probs = counts / n
        if self.criterion == "gini":
            return 1.0 - torch.sum(probs**2).item()
        else:
            nz = probs > 0
            return -torch.sum(probs[nz] * torch.log2(probs[nz])).item()

    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.root
        indent = "  " * depth
        if node.is_leaf:
            inv_map = {v: k for k, v in self.class_map.items()}
            label = inv_map.get(int(node.value), int(node.value))
            print(
                f"{indent}Leaf: class: {label}, probs={node.value_proba.cpu().numpy()}"
            )
        else:
            self.print_tree(node.left, depth + 1)
            self.print_tree(node.right, depth + 1)


# toy XOR-like dataset (not separable by single split, but OK for test)
X = torch.tensor(
    [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.2, 0.1],
        [0.9, 1.1],
    ],
    dtype=torch.float32,
)

y = torch.tensor([0, 1, 1, 0, 0, 1], dtype=torch.long)

clf = DecisionTreeTorch(max_depth=3, min_samples_split=1, criterion="gini")
clf.fit(X, y)
preds = clf.predict(X)
print("Preds:", preds.cpu().numpy())
clf.print_tree()
