import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from data_preprocessing import X, y, class_names
from sklearn.metrics import f1_score, classification_report, confusion_matrix as cm
from DT_testing import depths, best_depth, best_f1, f1_score, f1_macros, f1_weighted_list, y_test, y_pred_best

depth_labels = [str(d) if d is not None else "None" for d in depths]

# ── Plot 1: F1 Macro vs max_depth ─────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(8, 5))
colors = ["#e74c3c" if d == best_depth else "#3498db" for d in depths]
bars = ax1.bar(depth_labels, f1_macros, color=colors, edgecolor="white",
               linewidth=1.2, zorder=3)
ax1.set_xlabel("max_depth", fontsize=11)
ax1.set_ylabel("F1 Macro", fontsize=11)
ax1.set_title("F1 Macro theo max_depth", fontsize=13, fontweight="bold")
ax1.set_ylim(0, 1.05)
ax1.axhline(best_f1, color="#e74c3c", linestyle="--", linewidth=1.2,
            label=f"Best = {best_f1:.4f}")
ax1.legend(fontsize=9)
ax1.grid(axis="y", alpha=0.4)
for bar, val in zip(bars, f1_macros):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{val:.3f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("DT_f1_macro_by_depth.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Plot 2: F1 Macro vs F1 Weighted ──────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 5))
x_pos = np.arange(len(depths))
width = 0.38
ax2.bar(x_pos - width/2, f1_macros,         width, label="F1 Macro",
        color="#3498db", alpha=0.85, edgecolor="white")
ax2.bar(x_pos + width/2, f1_weighted_list,  width, label="F1 Weighted",
        color="#2ecc71", alpha=0.85, edgecolor="white")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(depth_labels)
ax2.set_xlabel("max_depth", fontsize=11)
ax2.set_ylabel("F1 Score", fontsize=11)
ax2.set_title("F1 Macro vs F1 Weighted", fontsize=13, fontweight="bold")
ax2.set_ylim(0, 1.1)
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig("DT_f1_macro_vs_weighted.png", dpi=150, bbox_inches="tight")
plt.show()


# ── Plot 3: Per-class Precision / Recall / F1 ─────────────────────────────────
report = classification_report(
    y_test, y_pred_best,
    target_names=class_names,
    output_dict=True   # ← quan trọng, trả về dict thay vì string
)

fig5, ax5 = plt.subplots(figsize=(8, 5))
x_cls   = np.arange(len(class_names))
width5  = 0.25
palette = ["#e67e22", "#1abc9c", "#e74c3c"]
for i, (metric, color) in enumerate(zip(["precision", "recall", "f1-score"], palette)):
    vals = [report[cls][metric] for cls in class_names]
    ax5.bar(x_cls + (i - 1) * width5, vals, width5,
            label=metric.capitalize(), color=color,
            alpha=0.85, edgecolor="white")
ax5.set_xticks(x_cls)
ax5.set_xticklabels(class_names, fontsize=10)
ax5.set_ylabel("Score", fontsize=11)
ax5.set_title("Precision / Recall / F1 theo từng lớp",
              fontsize=13, fontweight="bold")
ax5.set_ylim(0, 1.1)
ax5.legend(fontsize=9)
ax5.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig("DT_per_class_metrics.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Plot 4: Normalized Confusion Matrix (%) ───────────────────────────────────
fig6, ax6 = plt.subplots(figsize=(6, 5))
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Oranges",
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor="white",
            annot_kws={"size": 12}, ax=ax6)
ax6.set_xlabel("Predicted", fontsize=11)
ax6.set_ylabel("Actual", fontsize=11)
ax6.set_title("Confusion Matrix (Normalized %)",
              fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("DT_confusion_matrix_pct.png", dpi=150, bbox_inches="tight")
plt.show()

plt.savefig("wine_evaluation.png", dpi=150, bbox_inches="tight", facecolor="white")
