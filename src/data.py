"""Dataset provenance, centralized recoding, and fixed split."""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from .utils import SEED, LABELS

def load_data():
    raw = load_breast_cancer(as_frame=True)
    frame = raw.data.copy()
    frame["original_target"] = raw.target
    frame["target"] = 1 - raw.target
    frame["label"] = frame.target.map(LABELS)
    return frame, list(raw.feature_names)

def split_data():
    frame, features = load_data()
    train, test = train_test_split(frame.index.to_numpy(), test_size=0.2,
                                   random_state=SEED, stratify=frame.target)
    return frame, features, train, test
