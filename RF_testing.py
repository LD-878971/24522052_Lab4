import numpy as np
from Random_Forest import RandomForestClassifier
from Decision_Tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from data_preprocessing import X, y, class_names

# ── Giả sử X, y, class_names, DecisionTreeClassifier đã được định nghĩa ──────

# ── Train/Test Split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 1. Tìm max_depth tốt nhất ─────────────────────────────────────────────────
depths      = [2, 3, 5, 7, 10, None]
f1_macros   = []
f1_weighted_list = []

print("=== F1 Macro by max_depth ===")
print(f"{'Depth':<10} {'F1 Macro':<12} {'F1 Weighted'}")
print("-" * 36)

best_f1, best_depth = 0, None
for depth in depths:
    clf = RandomForestClassifier(max_depth=depth)
    clf.fit(X_train, y_train)
    y_pred      = clf.predict(X_test)
    f1_mac      = f1_score(y_test, y_pred, average="macro")
    f1_wei      = f1_score(y_test, y_pred, average="weighted")
    f1_macros.append(f1_mac)
    f1_weighted_list.append(f1_wei)
    if f1_mac > best_f1:
        best_f1, best_depth = f1_mac, depth
    print(f"{str(depth):<10} {f1_mac:.4f}       {f1_wei:.4f}")

# ── 2. Best model ─────────────────────────────────────────────────────────────
print(f"\nBest depth: {best_depth}  (F1 macro = {best_f1:.4f})")

best_clf = DecisionTreeClassifier(max_depth=best_depth)
best_clf.fit(X_train, y_train)
y_pred_best = best_clf.predict(X_test)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred_best, target_names=class_names))

cm = confusion_matrix(y_test, y_pred_best)
print("=== Confusion Matrix ===")
print(cm)

# ── 3. Cross-Validation ───────────────────────────────────────────────────────
print("\n=== 5-Fold Cross-Validation (F1 macro) ===")
cv_scores = []
for train_idx, val_idx in StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42).split(X, y):
    clf_cv = DecisionTreeClassifier(max_depth=best_depth)
    clf_cv.fit(X[train_idx], y[train_idx])
    cv_scores.append(
        f1_score(y[val_idx], clf_cv.predict(X[val_idx]), average="macro")
    )

print(f"Scores : {[round(s, 4) for s in cv_scores]}")
print(f"Mean   : {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")