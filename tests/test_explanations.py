import json
import numpy as np
from src.explain import explain_model, normalize_binary
from src.train import candidates
from src.utils import ROOT, malignancy_probability

def test_selected_explanation_additivity(dataset, fitted):
    frame, features, train, test = dataset
    X = frame.loc[test, features]
    explanation, scale, error = explain_model(fitted, frame.loc[train, features], X)
    assert explanation.values.shape == (114, 30)
    assert error < 1e-6
    reconstructed = explanation.base_values + explanation.values.sum(axis=1)
    if scale == "log-odds":
        reconstructed = 1 / (1 + np.exp(-reconstructed))
    np.testing.assert_allclose(reconstructed, malignancy_probability(fitted, X), atol=1e-6)
    with np.load(ROOT / "results/shap_values.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["sample_ids"], test)
        np.testing.assert_allclose(saved["values"], explanation.values)
    metadata = json.loads((ROOT / "results/shap_metadata.json").read_text())
    assert {item["actual"] for item in metadata["local_examples"]} == {0, 1}

def test_tree_explanation_path(dataset):
    frame, features, train, test = dataset
    model, _ = candidates(smoke=True)["random_forest"]
    model.set_params(classifier__n_estimators=10)
    model.fit(frame.loc[train, features], frame.loc[train, "target"])
    explanation, scale, error = explain_model(model, frame.loc[train, features], frame.loc[test[:4], features])
    assert scale == "probability" and error < 1e-6
    assert explanation.values.shape == (4, 30)

def test_binary_shap_return_shapes():
    values = np.arange(24).reshape(3, 4, 2)
    selected, bases = normalize_binary(values, np.ones((3, 2)), 1, 3)
    np.testing.assert_array_equal(selected, values[:, :, 1])
    legacy, legacy_bases = normalize_binary([values[:, :, 0], values[:, :, 1]], [0, 1], 1, 3)
    np.testing.assert_array_equal(legacy, selected)
    np.testing.assert_array_equal(legacy_bases, bases)
