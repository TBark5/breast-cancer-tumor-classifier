"""Malignant-class SHAP with explicit output scale and additivity checks."""
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from .utils import malignancy_probability, save_json, LABELS, SEED
from .visualize import save_figure


def explain_model(model, background, samples):
    transformed_background = model[:-1].transform(background)
    transformed_samples = model[:-1].transform(samples)
    classifier = model[-1]
    positive = list(classifier.classes_).index(1)
    if isinstance(classifier, LogisticRegression):
        if positive != 1:
            raise ValueError("Binary linear decision function must target malignancy")
        explainer = shap.LinearExplainer(classifier, transformed_background)
        raw = explainer(transformed_samples)
        values, bases = np.asarray(raw.values), np.asarray(raw.base_values)
        output = classifier.decision_function(transformed_samples)
        scale = "log-odds"
    else:
        explainer = shap.TreeExplainer(classifier, data=transformed_background,
                                       feature_perturbation="interventional", model_output="probability")
        raw = explainer(transformed_samples, check_additivity=False)
        values, bases = normalize_binary(raw.values, raw.base_values, positive, len(samples))
        output = malignancy_probability(model, samples)
        scale = "probability"
    bases = np.broadcast_to(np.squeeze(bases), (len(samples),)).copy()
    reconstructed = bases + values.sum(axis=1)
    np.testing.assert_allclose(reconstructed, output, atol=1e-6, rtol=1e-6)
    # Display original measurement values, even when the model uses standardized inputs.
    explanation = shap.Explanation(values=values, base_values=bases,
        data=samples.to_numpy(), feature_names=list(samples.columns))
    return explanation, scale, float(np.max(np.abs(reconstructed - output)))


def normalize_binary(values, bases, positive, rows):
    """Support legacy per-class lists and modern (sample, feature, class) arrays."""
    if isinstance(values, list):
        return np.asarray(values[positive]), np.full(rows, np.asarray(bases)[positive])
    values, bases = np.asarray(values), np.asarray(bases)
    if values.ndim == 3:
        values = values[:, :, positive]
        bases = bases[:, positive] if bases.ndim == 2 else np.full(rows, bases[positive])
    return values, bases


def waterfall(explanation, row, title, scale):
    shap.plots.waterfall(explanation[row], max_display=12, show=False)
    fig = plt.gcf()
    fig.set_size_inches(11, 7)
    fig.suptitle(title, fontsize=11, y=1.03)
    fig.axes[0].set_xlabel(f"Contribution to malignant {scale}")
    return fig


def generate_explanations(model, X_train, X_test, predictions, root):
    explanation, scale, error = explain_model(model, X_train, X_test)
    np.savez_compressed(root / "results/shap_values.npz", values=explanation.values,
        base_values=explanation.base_values, data=explanation.data,
        feature_names=np.asarray(explanation.feature_names), sample_ids=X_test.index.to_numpy())
    shap.plots.bar(explanation, max_display=15, show=False)
    fig = plt.gcf(); fig.set_size_inches(10, 7)
    fig.axes[0].set_title("Global malignant-class importance · holdout", pad=14)
    fig.axes[0].set_xlabel(f"Mean absolute SHAP value · malignant {scale}")
    save_figure(fig, root, "shap_global_bar")
    np.random.seed(SEED)  # Reproducible vertical jitter; no influence on fitted model.
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    fig = plt.gcf(); fig.set_size_inches(11, 8)
    fig.axes[0].set_title("Direction of feature associations · holdout", pad=14)
    fig.axes[0].set_xlabel(f"SHAP value · malignant {scale}")
    save_figure(fig, root, "shap_beeswarm")
    leading = np.argsort(np.abs(explanation.values).mean(axis=0))[-2:][::-1]
    for rank, index in enumerate(leading, 1):
        shap.plots.scatter(explanation[:, int(index)], show=False)
        fig = plt.gcf(); fig.set_size_inches(8, 5)
        fig.axes[0].set_title(f"Malignant-class contribution: {explanation.feature_names[index]}")
        fig.axes[0].set_ylabel(f"SHAP value · malignant {scale}")
        save_figure(fig, root, f"shap_dependence_{rank}")
    local = []
    for target in [1, 0]:
        row = int(np.flatnonzero(predictions.actual.to_numpy() == target)[0])
        record = predictions.iloc[row].to_dict()
        title = (f"Sample {int(record['sample_id'])} | Actual: {LABELS[target]} | "
                 f"Predicted: {LABELS[int(record['predicted'])]} | P(malignant)={record['malignancy_probability']:.8f}")
        save_figure(waterfall(explanation, row, title, scale), root, f"shap_local_{LABELS[target].lower()}")
        local.append({**record, "base_value": explanation.base_values[row],
                      "sum_contributions": explanation.values[row].sum()})
    metadata = {"class": "Malignant", "scale": scale, "max_additivity_error": error,
                "background": "SHAP deterministic subsample of training-only transformed features",
                "local_examples": local}
    save_json(root / "results/shap_metadata.json", metadata)
    return metadata
