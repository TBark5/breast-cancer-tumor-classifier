import numpy as np
import joblib
import pandas as pd
from src.utils import ROOT, malignancy_probability

def test_prediction_shapes_classes_ranges_and_saved_values(dataset, fitted):
    frame, features, _, test = dataset
    X = frame.loc[test, features]
    probability = malignancy_probability(fitted, X)
    prediction = fitted.predict(X)
    assert probability.shape == prediction.shape == (114,)
    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()
    assert set(prediction) == {0, 1}
    saved = pd.read_csv(ROOT / "results/holdout_predictions.csv")
    np.testing.assert_array_equal(saved.sample_id, test)
    np.testing.assert_allclose(saved.malignancy_probability, probability)
    np.testing.assert_array_equal(saved.predicted, prediction)

def test_serialization_round_trip(dataset, fitted, tmp_path):
    frame, features, _, test = dataset
    joblib.dump(fitted, tmp_path / "pipeline.joblib")
    loaded = joblib.load(tmp_path / "pipeline.joblib")
    np.testing.assert_allclose(loaded.predict_proba(frame.loc[test, features]),
                               fitted.predict_proba(frame.loc[test, features]))
