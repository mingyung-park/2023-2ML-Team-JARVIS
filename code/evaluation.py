# import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import lines
import numpy as np

# from sklearn.metrics import (
#     average_precision_score,
#     roc_curve,
#     precision_recall_curve,
#     auc,
# )
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    # confusion_matrix,
    # roc_auc_score,
    # RocCurveDisplay,
)

# from sklearn.preprocessing import LabelBinarizer
import warnings
import commonUtils

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def compare_models(model_name, y_pred, y_true):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    return {
        "name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def plot_metrics_comparison_multiclass(metrics_data, title, save_fig_path):
    num_metrics = len(metrics_data[0]) - 1
    num_models = len(metrics_data)

    width = 0.5
    fig, ax = plt.subplots(
        1, 1, figsize=(width * (1 + num_metrics * (1 + num_models)), 5)
    )
    fig.suptitle(f"{title}", fontsize=16)

    colors = plt.cm.viridis(np.linspace(0, 1, num_models + 1))

    metric_names = list(metrics_data[0].keys())[1:]

    for i, model_data in enumerate(metrics_data):
        positions = width * (np.arange(num_metrics) * (1 + num_models) + (i + 1))

        for j, metric_name in enumerate(metric_names):
            name, value = model_data["name"], model_data[metric_name]
            print(f"Model Evaluation for {name} - {metric_name}: {value:.4f}")

            ax.bar(
                positions[j],
                value,
                width=width,
                alpha=0.8,
                label=f"{name}",
                color=colors[i],
            )
            ax.text(
                positions[j],
                value,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(positions - width * (num_models / 2) / 2)
    ax.set_xticklabels(metric_names)
    ax.set_xlabel("Metrics")
    ax.set_ylim(0, 1)
    legend_handles = [
        lines.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=colors[j], markersize=10
        )
        for j in range(num_models)
    ]
    labels = [entry["name"] for entry in metrics_data]
    ax.legend(
        legend_handles,
        labels,
        title="Metrics",
        loc="upper right",
        bbox_to_anchor=(1.20, 1),
    )
    if save_fig_path:
        plt.savefig(save_fig_path)
    else:
        plt.show()


def show_confusion_matrix(
    x, y, title, model=None, model_path=None, normalize=None, save_fig_path=None
):
    if model_path:
        model = commonUtils.load_pickle_file(model_path)
    elif model == None:
        raise ValueError("no model provided")

    fig, axes = plt.subplots(1, 1, figsize=(50, 50))
    disp = ConfusionMatrixDisplay.from_estimator(
        model.model, x, y, ax=axes, normalize=normalize
    )
    axes.set_title(f"{title}", fontsize=50)
    if save_fig_path:
        plt.savefig(save_fig_path)
    else:
        plt.show()


def compare_scores(
    names, datasets, title, models=None, model_paths=None, save_fig_path=None
):
    if model_paths:
        models = [commonUtils.load_pickle_file(x) for x in model_paths]
    elif models == None:
        raise ValueError("no model provided")

    compare_data = []

    for model, name, dataset in zip(models, names, datasets):
        y_pred = model.predict(dataset[0])
        compare_data.append(compare_models(name, y_pred, dataset[1]))
    plot_metrics_comparison_multiclass(compare_data, f"{title}", save_fig_path)


def compare_results_from_prediction(title, data, save_fig_path=None):
    compare_data = [
        compare_models(name, y_true, y_pred) for name, y_true, y_pred in data
    ]
    plot_metrics_comparison_multiclass(compare_data, title, save_fig_path)


def show_confusion_matrix_from_prediction(
    y_true,
    y_pred,
    title,
    normalize=None,
    save_fig_path=None,
):
    _, axes = plt.subplots(1, 1, figsize=(50, 50))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, ax=axes, normalize=normalize
    )
    axes.set_title(title, fontsize=50)
    if save_fig_path:
        plt.savefig(save_fig_path)
    else:
        plt.show()
