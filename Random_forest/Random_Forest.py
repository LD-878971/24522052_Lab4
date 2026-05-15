import numpy as np
from Decision_Tree import DecisionTreeClassifier


class RandomForestClassifier:
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",   # "sqrt" | "log2" | float (tỉ lệ) | int
        bootstrap=True,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []          # list of (DecisionTreeClassifier, feature_indices)

    # ── public API ────────────────────────────────────────────────────────────
    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        self.trees = []

        rng = np.random.RandomState(self.random_state)
        seeds = rng.randint(0, 1_000_000, size=self.n_estimators)

        for seed in seeds:
            tree_rng = np.random.RandomState(seed)

            # 1. Bootstrap sampling
            X_sample, y_sample = self._bootstrap_sample(X, y, tree_rng)

            # 2. Feature subsampling
            feat_indices = self._sample_features(tree_rng)

            # 3. Train one tree trên feature subset
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X_sample[:, feat_indices], y_sample)
            self.trees.append((tree, feat_indices))

        return self

    def predict(self, X):
        # Majority vote từ tất cả các cây
        votes = self._collect_votes(X)          # (n_samples, n_estimators)
        return np.array([
            np.bincount(votes[i], minlength=self.n_classes_).argmax()
            for i in range(len(X))
        ])

    def predict_proba(self, X):
        # Trung bình xác suất từ tất cả các cây
        votes = self._collect_votes(X)          # (n_samples, n_estimators)
        proba = np.zeros((len(X), self.n_classes_))
        for i in range(len(X)):
            counts = np.bincount(votes[i], minlength=self.n_classes_)
            proba[i] = counts / self.n_estimators
        return proba

    # ── private helpers ───────────────────────────────────────────────────────
    def _bootstrap_sample(self, X, y, rng):
        n_samples = X.shape[0]
        if self.bootstrap:
            indices = rng.choice(n_samples, size=n_samples, replace=True)
        else:
            indices = np.arange(n_samples)
        return X[indices], y[indices]

    def _sample_features(self, rng):
        n = self.n_features_
        mf = self.max_features

        if mf == "sqrt":
            k = max(1, int(np.sqrt(n)))
        elif mf == "log2":
            k = max(1, int(np.log2(n)))
        elif isinstance(mf, float):
            k = max(1, int(mf * n))
        elif isinstance(mf, int):
            k = max(1, min(mf, n))
        else:
            k = n  # dùng tất cả features

        return rng.choice(n, size=k, replace=False)

    def _collect_votes(self, X):
        # Trả về ma trận (n_samples, n_estimators) chứa class dự đoán của từng cây
        all_preds = np.stack([
            tree.predict(X[:, feat_indices])
            for tree, feat_indices in self.trees
        ], axis=1)
        return all_preds