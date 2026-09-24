"""Reproducible stages. Selection is finalized before holdout evaluation."""
import argparse
import importlib.metadata
import platform
import joblib
import pandas as pd
from .data import split_data
from .train import select_model
from .evaluate import evaluate
from .explain import generate_explanations
from .visualize import eda, performance_figures
from .utils import ROOT, SEED, MAPPING, directories, save_json


def run(root=ROOT, smoke=False):
    root = directories(root)
    frame, features, train, test = split_data()
    save_json(root / "results/split.json", {"seed": SEED, "test_size": 0.2, "stratified": True,
                                           "train_indices": train, "test_indices": test})
    X_train, y_train = frame.loc[train, features], frame.loc[train, "target"]
    print("Searching models using training-only five-fold cross-validation...", flush=True)
    selected, model, cv, search = select_model(X_train, y_train, smoke)
    selection = {"selected_model": selected, "selection_rule": "Highest training CV mean ROC-AUC; exact ties favor logistic regression",
                 "seed": SEED, "folds": 5, "mapping": MAPPING, "smoke_configuration": smoke,
                 "cross_validation": cv, "selected_parameters": model[-1].get_params(),
                 "preprocessing": {"imputation": "median; missing marker is NaN",
                                   "standard_scaling": "scaler" in model.named_steps}}
    # Persist the decision before accessing holdout measurements for evaluation.
    save_json(root / "results/selection.json", selection)
    joblib.dump(model, root / "models/selected_pipeline.joblib")
    search.to_csv(root / "results/cv_search.csv", index=False)
    summary_rows = [{"model": name, "metric": metric, **values} for name, entry in cv.items()
                    for metric, values in entry["metrics"].items()]
    pd.DataFrame(summary_rows).to_csv(root / "results/cv_summary.csv", index=False)
    X_test, y_test = frame.loc[test, features], frame.loc[test, "target"]
    metrics, predictions = evaluate(model, X_test, y_test)
    save_json(root / "results/metrics.json", {**selection, "holdout": metrics})
    pd.DataFrame([{k: v for k, v in metrics.items() if isinstance(v, (int, float))}]).to_csv(root / "results/holdout_metrics.csv", index=False)
    predictions.to_csv(root / "results/holdout_predictions.csv", index=False)
    eda(frame, features, train, root)
    performance_figures(metrics, predictions, root)
    metadata = generate_explanations(model, X_train, X_test, predictions, root)
    save_json(root / "results/environment.json", {"python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ["numpy", "pandas", "scikit-learn", "matplotlib", "seaborn", "shap", "streamlit", "joblib", "pytest"]}})
    print(f"Selected {selected}; holdout accuracy={metrics['accuracy']:.6f}, AUC={metrics['roc_auc']:.6f}; SHAP error={metadata['max_additivity_error']:.2e}", flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "eda", "explain"], default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=str, default=str(ROOT))
    args = parser.parse_args()
    root = directories(args.output)
    if args.stage == "all":
        run(root, args.smoke)
    else:
        frame, features, train, test = split_data()
        if args.stage == "eda":
            eda(frame, features, train, root)
        else:
            model = joblib.load(root / "models/selected_pipeline.joblib")
            predictions = pd.read_csv(root / "results/holdout_predictions.csv")
            generate_explanations(model, frame.loc[train, features], frame.loc[test, features], predictions, root)

if __name__ == "__main__":
    main()
