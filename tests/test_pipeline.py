import json
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from src.train import candidates
from src.utils import ROOT, SEED
from src.workflow import run

FIGURES = ["class_distribution", "feature_distributions", "correlation_heatmap", "confusion_matrix",
           "roc_curve", "precision_recall_curve", "shap_global_bar", "shap_beeswarm",
           "shap_dependence_1", "shap_dependence_2", "shap_local_malignant", "shap_local_benign"]

def test_selection_rule_and_cv_records(report):
    cv = report["cross_validation"]
    expected = max(cv, key=lambda name: cv[name]["metrics"]["roc_auc"]["mean"])
    assert expected == report["selected_model"]
    for entry in cv.values():
        for metric in entry["metrics"].values():
            assert len(metric["folds"]) == 5
            np.testing.assert_allclose(metric["mean"], np.mean(metric["folds"]))
            np.testing.assert_allclose(metric["std"], np.std(metric["folds"]))

def test_preprocessing_fits_each_training_fold_only(dataset):
    frame, features, train, test = dataset
    X, y = frame.loc[train, features], frame.loc[train, "target"]
    model, _ = candidates(True)["logistic_regression"]
    for fit_idx, val_idx in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        fitted = clone(model).fit(X.iloc[fit_idx], y.iloc[fit_idx])
        np.testing.assert_allclose(fitted["imputer"].statistics_, X.iloc[fit_idx].median())
        np.testing.assert_allclose(fitted["scaler"].mean_, X.iloc[fit_idx].mean())
        assert not set(X.iloc[fit_idx].index) & set(test)
        assert not set(fit_idx) & set(val_idx)

def validate_artifacts(root):
    for name in FIGURES:
        for extension in ["png", "svg"]:
            path = root / "figures" / f"{name}.{extension}"
            assert path.stat().st_size > 1000
            if extension == "png":
                with Image.open(path) as image:
                    assert min(image.size) > 500
                    assert image.info["dpi"][0] >= 299
                    image.verify()
            else:
                assert ET.parse(path).getroot().tag.endswith("svg")
    for name in ["metrics", "selection", "dataset_summary", "split", "shap_metadata", "environment"]:
        assert json.loads((root / f"results/{name}.json").read_text())
    for name in ["cv_search", "cv_summary", "holdout_metrics", "holdout_predictions", "descriptive_statistics", "class_comparisons"]:
        assert not pd.read_csv(root / f"results/{name}.csv").empty
    assert (root / "models/selected_pipeline.joblib").stat().st_size > 100
    with np.load(root / "results/shap_values.npz", allow_pickle=False) as archive:
        assert archive["values"].shape == (114, 30)

def test_generated_artifacts():
    validate_artifacts(ROOT)

def test_smaller_end_to_end_workflow(tmp_path, monkeypatch, dataset):
    import src.workflow as workflow
    _, _, train, test = dataset
    original_select, original_evaluate = workflow.select_model, workflow.evaluate
    events = []

    def audit_selection(X, y, smoke):
        np.testing.assert_array_equal(X.index, train)
        np.testing.assert_array_equal(y.index, train)
        assert not set(X.index) & set(test)
        events.append("selection")
        return original_select(X, y, smoke)

    def audit_evaluation(model, X, y):
        assert events == ["selection"]
        assert (tmp_path / "results/selection.json").exists()
        np.testing.assert_array_equal(X.index, test)
        events.append("evaluation")
        return original_evaluate(model, X, y)

    monkeypatch.setattr(workflow, "select_model", audit_selection)
    monkeypatch.setattr(workflow, "evaluate", audit_evaluation)
    metrics = run(tmp_path, smoke=True)
    assert events == ["selection", "evaluation"]
    assert 0 <= metrics["accuracy"] <= 1
    validate_artifacts(tmp_path)
