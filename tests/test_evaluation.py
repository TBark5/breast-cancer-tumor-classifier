import numpy as np
from src.evaluate import evaluate
from src.utils import ROOT

def test_saved_metrics_recompute_exactly(dataset, fitted, report):
    frame, features, _, test = dataset
    metrics, _ = evaluate(fitted, frame.loc[test, features], frame.loc[test, "target"])
    assert metrics == report["holdout"]
    assert sum(map(sum, metrics["confusion_matrix"])) == 114

def test_fitted_preprocessing_uses_full_training_only(dataset, fitted):
    frame, features, train, _ = dataset
    np.testing.assert_allclose(fitted["imputer"].statistics_, frame.loc[train, features].median())
    if "scaler" in fitted.named_steps:
        np.testing.assert_allclose(fitted["scaler"].mean_, frame.loc[train, features].mean())
