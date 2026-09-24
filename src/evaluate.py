"""One final evaluation at the prespecified probability threshold of 0.5."""
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report)
from .utils import malignancy_probability, LABELS

def evaluate(model, X, y):
    probability = malignancy_probability(model, X)
    prediction = model.predict(X)
    matrix = confusion_matrix(y, prediction, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    metrics = {"accuracy": accuracy_score(y, prediction),
               "precision": precision_score(y, prediction, zero_division=0),
               "recall": recall_score(y, prediction), "f1": f1_score(y, prediction),
               "specificity": tn / (tn + fp), "roc_auc": roc_auc_score(y, probability),
               "average_precision": average_precision_score(y, probability),
               "confusion_matrix": matrix.tolist(), "threshold": 0.5,
               "classification_report": classification_report(y, prediction, labels=[0, 1],
                    target_names=[LABELS[0], LABELS[1]], output_dict=True, zero_division=0)}
    predictions = pd.DataFrame({"sample_id": X.index, "actual": y.to_numpy(),
                                "predicted": prediction, "malignancy_probability": probability})
    return metrics, predictions
