"""Consistent publication figures; exploratory analysis uses training rows only."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from .utils import save_json, LABELS

sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.05)
plt.rcParams.update({"svg.fonttype": "none", "figure.dpi": 110})

def save_figure(fig, root, name):
    for extension in ("png", "svg"):
        fig.savefig(root / "figures" / f"{name}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)

def eda(frame, features, train, root):
    data = frame.loc[train]
    summary = {"dataset_rows": len(frame), "input_features": len(features),
               "dataset_class_counts": frame.label.value_counts().to_dict(),
               "dataset_missing_values": int(frame[features].isna().sum().sum()),
               "dataset_duplicate_records": int(frame[features].duplicated().sum()),
               "dataset_constant_features": [c for c in features if frame[c].nunique() <= 1],
               "eda_scope": "Training partition only; fixed feature choices; holdout excluded",
               "training_rows": len(data), "training_class_counts": data.label.value_counts().to_dict(),
               "dtypes": data[features].dtypes.astype(str).to_dict(),
               "missing_values": data[features].isna().sum().to_dict(),
               "duplicate_training_records": int(data[features].duplicated().sum()),
               "constant_training_features": [c for c in features if data[c].nunique() <= 1]}
    save_json(root / "results/dataset_summary.json", summary)
    data[features].describe().T.to_csv(root / "results/descriptive_statistics.csv")
    data.groupby("label")[features].agg(["mean", "std", "median"]).to_csv(root / "results/class_comparisons.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = data.label.value_counts().reindex(["Benign", "Malignant"])
    bars = ax.bar(counts.index, counts, color=["#0072B2", "#D55E00"])
    ax.bar_label(bars, padding=4)
    ax.set(title="Training samples by class", ylabel="Number of samples", ylim=(0, counts.max() * 1.15))
    save_figure(fig, root, "class_distribution")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, feature in zip(axes.flat, ["mean radius", "mean texture", "mean concavity", "worst area"]):
        sns.histplot(data=data, x=feature, hue="label", hue_order=["Benign", "Malignant"],
                     palette=["#0072B2", "#D55E00"], element="step", stat="density", common_norm=False, ax=ax)
        ax.set_title(feature.capitalize())
    fig.suptitle("Cell-nucleus measurements · training partition")
    fig.tight_layout()
    save_figure(fig, root, "feature_distributions")
    fig, ax = plt.subplots(figsize=(19, 16))
    sns.heatmap(data[features].corr(), vmin=-1, vmax=1, center=0, cmap="vlag", annot=True,
                fmt=".1f", annot_kws={"size": 6}, square=True, ax=ax, cbar_kws={"label": "Pearson correlation"})
    ax.tick_params(labelsize=9)
    ax.set_title("Feature correlations · training partition", pad=20)
    save_figure(fig, root, "correlation_heatmap")

def performance_figures(metrics, predictions, root):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=list(LABELS.values()), yticklabels=list(LABELS.values()), ax=ax)
    ax.set(xlabel="Predicted class", ylabel="Actual class", title="Selected model · untouched holdout")
    save_figure(fig, root, "confusion_matrix")
    y, p = predictions.actual, predictions.malignancy_probability
    fig, ax = plt.subplots(figsize=(7, 6))
    display = RocCurveDisplay.from_predictions(y, p, ax=ax, name="Selected model", plot_chance_level=True)
    display.line_.set_label(f"Selected model (AUC = {metrics['roc_auc']:.4f})")
    ax.legend(loc="lower right")
    ax.set_title("Holdout ROC · malignant is positive")
    save_figure(fig, root, "roc_curve")
    fig, ax = plt.subplots(figsize=(7, 6))
    display = PrecisionRecallDisplay.from_predictions(y, p, ax=ax, name="Selected model", plot_chance_level=True)
    display.line_.set_label(f"Selected model (AP = {metrics['average_precision']:.4f})")
    ax.legend(loc="lower left")
    ax.set_title("Holdout precision–recall · malignant is positive")
    save_figure(fig, root, "precision_recall_curve")
