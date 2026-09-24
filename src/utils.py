"""Shared paths, labels, and artifact serialization."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
LABELS = {0: "Benign", 1: "Malignant"}
MAPPING = {"original": {"0": "Malignant", "1": "Benign"},
           "modeling": {"0": "Benign", "1": "Malignant"}}

def directories(root=ROOT):
    root = Path(root)
    for name in ("models", "results", "figures"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root

def save_json(path, value):
    def convert(x):
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value, indent=2, default=convert, allow_nan=False), encoding="utf-8")

def malignancy_probability(model, X):
    index = list(model.classes_).index(1)
    return model.predict_proba(X)[:, index]
