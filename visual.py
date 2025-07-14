import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_curve,
    precision_score,
    confusion_matrix,
)

# def draw_bar(acc, pre, recall):
#     metrics = ['Accuracy', 'Precision', 'Recall']
#     values = [acc, pre, recall]
#     plt.figure(figsize=(5, 5))
#     bars = plt.bar(metrics, values, color=['blue', 'green', 'red'])

#     for bar in bars:
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2., height,
#                 f'{height:.2f}',
#                 ha='center', va='bottom')

#     plt.title('Classification Metrics', fontsize=14)
#     plt.ylabel('Score', fontsize=12)
#     plt.ylim(0, 1.1)  # 设置y轴范围

#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     plt.show()


def plot_auc(y_pred, y_test):
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label="ROC Curve (AUC = {:.5f})".format(roc_auc))

    plt.plot(recall, precision, label="PR Curve (AUC = {:.5})".format(pr_auc))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random", color="gray")

    # set x-axis range
    plt.xlim([0.0, 1.0])

    # set y-axis range
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate / Recall")
    plt.ylabel("True Positive Rate / Precision")
    plt.title("PR-ROC ")
    plt.grid()
    plt.legend(loc="lower right")
    plt.show()


def plot_confusion_matrix(
    y_true, y_pred, y_prob=None, label_names=None, model_name="Model"
):

    # get confusion_matrix
    confusion_mtx = confusion_matrix(y_true, y_pred)

    f, ax = plt.subplots(figsize=(4, 4), dpi=100)
    sns.heatmap(
        confusion_mtx,
        annot=True,
        linewidths=0.1,
        fmt=".0f",
        ax=ax,
        cbar=False,
        xticklabels=label_names,
        yticklabels=label_names,
    )

    plt.xlabel("Predicted Label", fontsize=10)
    plt.ylabel("True Label", fontsize=10)
    plt.title(f"{model_name} Confusion Matrix", fontsize=13)
    plt.tight_layout()
    plt.show()


def draw_bar(**kwargs):
    """
    绘制一个条形图来展示任意数量的分类指标。

    使用关键字参数来传递指标和它们的值。
    例如: draw_bar(Accuracy=0.95, Precision=0.88, F1_Score=0.90)
    """
    # 如果没有传入任何指标，则直接返回
    if not kwargs:
        print("没有提供任何指标进行绘制。")
        return

    # 从关键字参数中提取指标名称和对应的值
    metrics = list(kwargs.keys())
    values = list(kwargs.values())

    # 动态调整图形宽度，防止标签重叠
    fig_width = max(5, len(metrics) * 2)
    plt.figure(figsize=(fig_width, 5))

    # 提供一组颜色，可以根据需要扩展
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    bars = plt.bar(metrics, values, color=colors[: len(metrics)])

    # 在每个条形上显示具体数值
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",  # 显示三位小数以提高精度
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.title("Classification Metrics", fontsize=15)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0, 1.1)  # 设置y轴范围

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
