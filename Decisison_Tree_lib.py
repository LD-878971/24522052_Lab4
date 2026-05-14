import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn import tree
from sklearn.model_selection import StratifiedKFold
from data_preprocessing import X, y, X_test, X_train, y_train, y_test, class_names, combined

model = DecisionTreeClassifier(
    max_depth=5,          # Giới hạn độ sâu cây (tránh overfitting)
    min_samples_split=10, # Số mẫu tối thiểu để split
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

depths           = [2, 3, 5, 7, 10, None]
f1_macros        = []
f1_weighted_list = []

print("=== [SKLearn] F1 Macro by max_depth ===")
print(f"{'Depth':<10} {'F1 Macro':<12} {'F1 Weighted'}")
print("-" * 36)

best_f1, best_depth = 0, None
for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1_mac = f1_score(y_test, y_pred, average="macro")
    f1_wei = f1_score(y_test, y_pred, average="weighted")
    f1_macros.append(f1_mac)
    f1_weighted_list.append(f1_wei)
    if f1_mac > best_f1:
        best_f1, best_depth = f1_mac, depth
    print(f"{str(depth):<10} {f1_mac:.4f}       {f1_wei:.4f}")

# ── 2. Best model ─────────────────────────────────────────────────────────────
print(f"\nBest depth: {best_depth}  (F1 macro = {best_f1:.4f})")

best_clf = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
best_clf.fit(X_train, y_train)
y_pred_best = best_clf.predict(X_test)

print("\n=== [SKLearn] Classification Report ===")
print(classification_report(y_test, y_pred_best, target_names=class_names))

cm = confusion_matrix(y_test, y_pred_best)
print("=== [SKLearn] Confusion Matrix ===")
print(cm)

# ── 3. Cross-Validation ───────────────────────────────────────────────────────
print("\n=== [SKLearn] 5-Fold Cross-Validation (F1 macro) ===")
cv_scores = []
for train_idx, val_idx in StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42).split(X, y):
    clf_cv = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
    clf_cv.fit(X[train_idx], y[train_idx])
    cv_scores.append(
        f1_score(y[val_idx], clf_cv.predict(X[val_idx]), average="macro")
    )

print(f"Scores : {[round(s, 4) for s in cv_scores]}")
print(f"Mean   : {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

