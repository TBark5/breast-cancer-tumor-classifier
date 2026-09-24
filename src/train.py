"""Training-only model search; no holdout features enter this module."""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from .utils import SEED

SCORING = {key: key for key in ("roc_auc", "accuracy", "precision", "recall", "f1")}

def candidates(smoke=False):
    logistic = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler()),
                         ("classifier", LogisticRegression(max_iter=5000, random_state=SEED))])
    forest = Pipeline([("imputer", SimpleImputer(strategy="median")),
                       ("classifier", RandomForestClassifier(random_state=SEED, n_jobs=1))])
    return {
        "logistic_regression": (logistic, {"classifier__C": [0.1, 1] if smoke else [0.01, 0.1, 1, 10, 100]}),
        "random_forest": (forest, {"classifier__n_estimators": [30] if smoke else [150, 300],
                                  "classifier__max_depth": [None] if smoke else [None, 5],
                                  "classifier__min_samples_leaf": [1] if smoke else [1, 3],
                                  "classifier__max_features": ["sqrt"] if smoke else ["sqrt", 0.5]})}

def select_model(X_train, y_train, smoke=False):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    searches, rows, summaries = {}, [], {}
    for name, (pipeline, grid) in candidates(smoke).items():
        search = GridSearchCV(pipeline, grid, scoring=SCORING, refit="roc_auc", cv=cv,
                              n_jobs=2, error_score="raise", return_train_score=False)
        search.fit(X_train, y_train)
        searches[name] = search
        table = pd.DataFrame(search.cv_results_)
        table.insert(0, "model", name)
        rows.append(table)
        i = search.best_index_
        summaries[name] = {"parameters": search.best_params_, "metrics": {
            metric: {"mean": float(search.cv_results_[f"mean_test_{metric}"][i]),
                     "std": float(search.cv_results_[f"std_test_{metric}"][i]),
                     "folds": [float(search.cv_results_[f"split{k}_test_{metric}"][i]) for k in range(5)]}
            for metric in SCORING}}
    # Stable insertion order breaks exact ties in favor of the simpler logistic model.
    selected = max(searches, key=lambda name: searches[name].best_score_)
    return selected, searches[selected].best_estimator_, summaries, pd.concat(rows, ignore_index=True)
