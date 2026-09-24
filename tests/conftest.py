import json
import joblib
import pytest
from src.utils import ROOT
from src.data import split_data

@pytest.fixture(scope="session")
def dataset():
    return split_data()

@pytest.fixture(scope="session")
def fitted():
    path = ROOT / "models/selected_pipeline.joblib"
    if not path.exists():
        pytest.fail("Generate artifacts first: python -m src.workflow")
    return joblib.load(path)

@pytest.fixture(scope="session")
def report():
    return json.loads((ROOT / "results/metrics.json").read_text())
