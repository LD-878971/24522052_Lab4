import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from data_preprocessing import X, y, X_test, X_train, y_train, y_test, class_names, combined

# ── 0. Khởi tạo nhanh để xem kết quả ban đầu ─────────────────────────────────
model = RandomForestClassifier(
    max_depth=5,
    min_samples_split=10,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── 1. Tìm max_depth và n_estimators tốt nhất ────────────────────────────────
depths            = [2, 3, 5, 7, 10, None]
n_estimators_list = [50, 100, 200]
f1_macros         = []
f1_weighted_list  = []

print("=== [SKLearn] F1 Macro by max_depth & n_estimators ===")
print(f"{'Depth':<10} {'n_estimators':<15} {'F1 Macro':<12} {'F1 Weighted'}")
print("-" * 52)

best_f1, best_depth, best_n = 0, None, 100
for depth in depths:
    for n in n_estimators_list:
        clf = RandomForestClassifier(max_depth=depth, n_estimators=n, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1_mac = f1_score(y_test, y_pred, average="macro")
        f1_wei = f1_score(y_test, y_pred, average="weighted")
        f1_macros.append(f1_mac)
        f1_weighted_list.append(f1_wei)
        if f1_mac > best_f1:
            best_f1, best_depth, best_n = f1_mac, depth, n
        print(f"{str(depth):<10} {n:<15} {f1_mac:.4f}       {f1_wei:.4f}")

# ── 2. Best model ─────────────────────────────────────────────────────────────
print(f"\nBest depth: {best_depth}, Best n_estimators: {best_n}  (F1 macro = {best_f1:.4f})")

best_clf = RandomForestClassifier(max_depth=best_depth, n_estimators=best_n, random_state=42)
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
    clf_cv = RandomForestClassifier(max_depth=best_depth, n_estimators=best_n, random_state=42)
    clf_cv.fit(X[train_idx], y[train_idx])
    cv_scores.append(
        f1_score(y[val_idx], clf_cv.predict(X[val_idx]), average="macro")
    )

print(f"Scores : {[round(s, 4) for s in cv_scores]}")
print(f"Mean   : {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# ── 4. Bảng so sánh tổng hợp ─────────────────────────────────────────────────
print("\n" + "=" * 55)
print("             SO SÁNH HAI MÔ HÌNH")
print("=" * 55)
print(f"{'Metric':<30} {'Decision Tree':>13} {'Random Forest':>10}")
print("-" * 55)
# Thay các '?' bằng kết quả từ model Decision Tree của bạn
print(f"{'Best depth':<30} {'?':>13} {str(best_depth):>10}")
print(f"{'F1 Macro (test set)':<30} {'?':>13} {best_f1:>10.4f}")
print(f"{'CV Mean F1 Macro':<30} {'?':>13} {np.mean(cv_scores):>10.4f}")
print(f"{'CV Std':<30} {'?':>13} {np.std(cv_scores):>10.4f}")
print("=" * 55)