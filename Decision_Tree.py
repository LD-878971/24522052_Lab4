import numpy as np

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def _build_tree(self, X, y, depth):
        num_samples, _ = X.shape
        num_labels = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_labels)

        node = {
            "type": "leaf",
            "predicted_class": predicted_class,
            "num_samples": num_samples,
            "num_labels": num_labels,
            "gini": self._gini(y),
        }

        should_stop = (
            (self.max_depth is not None and depth >= self.max_depth)
            or num_samples < self.min_samples_split
            or len(np.unique(y)) == 1
        )

        if not should_stop:
            best = self._best_split(X, y)
            if best is not None:
                feature_idx, threshold = best
                left_mask  = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                if (left_mask.sum() >= self.min_samples_leaf and
                        right_mask.sum() >= self.min_samples_leaf):
                    node.update({
                        "type":        "internal",
                        "feature_idx": feature_idx,
                        "threshold":   threshold,
                        "left":  self._build_tree(X[left_mask],  y[left_mask],  depth + 1),
                        "right": self._build_tree(X[right_mask], y[right_mask], depth + 1),
                    })
        return node

    def _best_split(self, X, y):
        m, n = X.shape
        best_gini, best_feature, best_threshold = float("inf"), None, None

        for feat in range(n):
            sorted_idx = np.argsort(X[:, feat])
            X_sorted   = X[sorted_idx, feat]
            y_sorted   = y[sorted_idx]

            for i in range(1, m):
                if X_sorted[i] == X_sorted[i - 1]:
                    continue

                threshold = (X_sorted[i] + X_sorted[i - 1]) / 2
                gini = self._weighted_gini(y_sorted[:i], y_sorted[i:], m)

                if gini < best_gini:
                    best_gini, best_feature, best_threshold = gini, feat, threshold

        return (best_feature, best_threshold) if best_feature is not None else None

    def _weighted_gini(self, y_left, y_right, num_samples):
        n_left, n_right = len(y_left), len(y_right)
        return (n_left  / num_samples) * self._gini(y_left) \
             + (n_right / num_samples) * self._gini(y_right)

    def _gini(self, y):
        n = len(y)
        if n == 0:
            return 0.0
        probs = np.bincount(y, minlength=self.n_classes_) / n
        return 1.0 - np.sum(probs ** 2)

    def _predict_sample(self, x, node):
        if node["type"] == "leaf":
            return node["predicted_class"]
        if x[node["feature_idx"]] <= node["threshold"]:
            return self._predict_sample(x, node["left"])
        return self._predict_sample(x, node["right"])