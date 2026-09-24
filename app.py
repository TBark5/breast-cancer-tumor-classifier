"""Read-only dashboard for benchmark artifacts; never trains on widget changes."""
import json
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt
from src.utils import ROOT, LABELS
from src.explain import waterfall

st.set_page_config(page_title="Tumor Classification | Explainable AI", page_icon="🔬", layout="wide")
st.title("Breast Cancer Tumor Classification")
st.caption("EXPLAINABLE AI  /  WISCONSIN DIAGNOSTIC BENCHMARK")
st.warning("Educational use only. This benchmark is not a clinically validated diagnostic system and must not guide patient care.")
REQUIRED = ["results/metrics.json", "results/dataset_summary.json", "results/holdout_predictions.csv",
            "results/shap_values.npz", "results/shap_metadata.json", "models/selected_pipeline.joblib"]
missing = [name for name in REQUIRED if not (ROOT / name).exists()]
if missing:
    st.error("Missing artifacts: " + ", ".join(missing) + ". From the project directory run: python scripts/run_all.py")
    st.stop()

@st.cache_resource
def load_artifacts(signature):
    metrics = json.loads((ROOT / "results/metrics.json").read_text())
    summary = json.loads((ROOT / "results/dataset_summary.json").read_text())
    predictions = pd.read_csv(ROOT / "results/holdout_predictions.csv")
    with np.load(ROOT / "results/shap_values.npz", allow_pickle=False) as values:
        explanation = shap.Explanation(values=values["values"], base_values=values["base_values"],
            data=values["data"], feature_names=values["feature_names"].tolist())
        if not np.array_equal(values["sample_ids"], predictions.sample_id):
            raise ValueError("Explanation and prediction sample IDs do not match; regenerate artifacts")
    metadata = json.loads((ROOT / "results/shap_metadata.json").read_text())
    model = joblib.load(ROOT / "models/selected_pipeline.joblib")
    return metrics, summary, predictions, explanation, metadata, model

metrics, summary, predictions, explanation, metadata, model = load_artifacts(
    tuple((ROOT / name).stat().st_mtime_ns for name in REQUIRED))

def chart(name, caption=None):
    path = ROOT / "figures" / f"{name}.png"
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.error(f"Missing {path.name}. Run: python scripts/run_all.py")

page = st.sidebar.radio("Explore", ["Overview", "Model performance", "Feature analysis", "Explainable AI", "About"])
st.sidebar.caption("Positive class: malignant = 1\n\nFixed seed: 42 · Holdout: 20%")
if page == "Overview":
    st.subheader("From cell measurements to transparent predictions")
    st.write("569 samples and 30 cell-nucleus measurements derived from digitized images of fine-needle aspirates of breast masses. These are tabular measurements, not raw mammograms or a representative screening population.")
    cols = st.columns(4)
    for col, name in zip(cols, ["accuracy", "roc_auc", "recall", "specificity"]):
        col.metric(name.replace("_", " ").title(), f"{metrics['holdout'][name]:.3f}")
    st.info(f"Selected using training-only ROC-AUC: {metrics['selected_model']}. All headline scores are from the 114-sample holdout.")
    left, right = st.columns([1, 1])
    with left: chart("class_distribution")
    with right:
        st.subheader("Dataset composition")
        st.dataframe(pd.Series(summary["dataset_class_counts"], name="All samples"))
        st.write("Workflow: fixed split → pipeline preprocessing within five-fold CV → model selection → final holdout evaluation → SHAP explanations.")
elif page == "Model performance":
    st.subheader("Final holdout evaluation")
    st.write("A false negative is a malignant sample classified as benign; a false positive is a benign sample classified as malignant. These benchmark errors do not establish clinical safety or usefulness. The decision threshold is fixed at 0.5.")
    st.dataframe(pd.read_csv(ROOT / "results/cv_summary.csv").drop(columns="folds"), hide_index=True)
    st.caption("Training CV means and standard deviations for the best searched configuration of each model; these are not nested-CV estimates.")
    chart("confusion_matrix")
    cols = st.columns(2)
    with cols[0]: chart("roc_curve")
    with cols[1]: chart("precision_recall_curve")
    st.dataframe(pd.DataFrame(metrics["holdout"]["classification_report"]).T)
elif page == "Feature analysis":
    st.subheader("Training-partition feature analysis")
    st.write("Correlated measurements capture overlapping aspects of nuclear size, shape, and texture. No feature selection was based on these plots.")
    chart("feature_distributions")
    chart("correlation_heatmap")
    st.dataframe(pd.read_csv(ROOT / "results/descriptive_statistics.csv"))
elif page == "Explainable AI":
    st.subheader("Why did the model assign this score?")
    st.write("Global importance averages absolute contributions over held-out samples. Local explanations decompose one prediction relative to a training-background baseline. SHAP describes model associations, not biological causation.")
    st.info(f"SHAP scale: malignant {metadata['scale']}. For log-odds, probability = 1 / (1 + exp(−score)); contributions do not add directly to probability.")
    cols = st.columns(2)
    with cols[0]: chart("shap_global_bar")
    with cols[1]: chart("shap_beeswarm")
    sample_id = st.selectbox("Held-out sample ID", predictions.sample_id.tolist())
    row = int(np.flatnonzero(predictions.sample_id.to_numpy() == sample_id)[0])
    record = predictions.iloc[row]
    cols = st.columns(3)
    cols[0].metric("Actual class", LABELS[int(record.actual)])
    cols[1].metric("Predicted class", LABELS[int(record.predicted)])
    cols[2].metric("P(malignant)", f"{record.malignancy_probability:.4f}")
    fig = waterfall(explanation, row, f"Held-out sample {sample_id}", metadata["scale"])
    st.pyplot(fig); plt.close(fig)
    st.caption(f"Maximum verified SHAP additivity error across all holdout samples: {metadata['max_additivity_error']:.2e}")
    st.dataframe(pd.DataFrame({"feature": explanation.feature_names, "measurement": explanation.data[row],
                              "SHAP contribution": explanation.values[row]}).sort_values("SHAP contribution"))
elif page == "About":
    st.subheader("Methodology and limitations")
    st.write("Median imputation and logistic scaling are fitted inside each CV fold. Logistic regression and random forests are compared using five-fold stratified cross-validation of the 455 training samples. Highest mean ROC-AUC wins; exact ties favor logistic regression. GridSearchCV refits that pipeline on the complete training partition before a single holdout evaluation.")
    st.write("Small historical dataset; no external validation, subgroup fairness assessment, calibration study, or prospective evaluation. Correlated features can share importance. Test performance has sampling uncertainty and is not screening-population performance.")
    st.markdown("[Dataset provenance](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) · [scikit-learn dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)")
    st.code("python scripts/run_all.py\npython -m pytest -q\npython -m streamlit run app.py")
    st.write("See README.md and src/ for the complete methodology and source. Only load trusted joblib artifacts, which can execute Python code.")
