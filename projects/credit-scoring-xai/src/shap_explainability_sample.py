"""
Credit Risk Scoring - Explainable AI (XAI) Auditing Sample
----------------------------------------------------------
Demonstrates how a serialized XGBoost pipeline is loaded to compute
global and local Shapley values via SHAP (TreeExplainer) for
auditable credit underwriting decisions.
"""

import re
import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path


def sanitize_feature_names(feature_names: list[str]) -> list[str]:
    """
    Sanitizes transformed feature names by removing special characters
    ('[', ']', '<', '>') that trigger parser errors in XGBoost/TreeExplainer.
    """
    return [re.sub(r"[\[\]<>]", "_", str(name)) for name in feature_names]


def load_champion_pipeline(model_path: Path):
    """Loads the serialized Scikit-Learn / XGBoost champion pipeline."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {model_path}")
    return joblib.load(model_path)


def audit_credit_applications(
    pipeline,
    X_sample: pd.DataFrame,
    decision_threshold: float = 0.50,
    top_k_reasons: int = 3,
) -> list[dict]:
    """
    Runs model inference and extracts local SHAP attributions for each
    applicant to provide transparent, human-readable risk justifications.
    """
    # 1. Separate preprocessing step from the trained XGBoost classifier
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # 2. Transform raw application data and align clean feature names
    X_transformed = preprocessor.transform(X_sample)
    raw_names = preprocessor.get_feature_names_out()
    clean_names = sanitize_feature_names(raw_names)

    X_encoded_df = pd.DataFrame(X_transformed, columns=clean_names)

    # 3. Predict default probabilities using the full pipeline
    default_probs = pipeline.predict_proba(X_sample)[:, 1]

    # 4. Compute exact Shapley values via observational TreeSHAP
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_encoded_df)

    # 5. Build structured audit reports per loan application
    audit_reports = []
    for idx in range(len(X_encoded_df)):
        prob = float(default_probs[idx])
        decision = "REJECT / HIGH RISK" if prob >= decision_threshold else "APPROVE / LOW RISK"

        instance_shap = pd.Series(shap_values[idx], index=clean_names)
        top_drivers = (
            instance_shap.reindex(instance_shap.abs().sort_values(ascending=False).index)
            .head(top_k_reasons)
        )

        reasons = [
            {
                "feature": feat,
                "shap_impact": round(float(val), 4),
                "direction": "Increases Default Risk" if val > 0 else "Mitigates Default Risk",
            }
            for feat, val in top_drivers.items()
        ]

        audit_reports.append(
            {
                "applicant_index": idx,
                "default_probability": round(prob, 4),
                "underwriting_decision": decision,
                "key_drivers": reasons,
            }
        )

    return audit_reports


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_PATH = BASE_DIR / "models" / "xgboost_campeon_pipeline.pkl"
    SAMPLE_DATA_PATH = BASE_DIR / "data" / "sample_dataset.csv"

    # Example execution flow
    # pipeline = load_champion_pipeline(MODEL_PATH)
    # df_sample = pd.read_csv(SAMPLE_DATA_PATH).drop(columns=["Default"], errors="ignore")
    # reports = audit_credit_applications(pipeline, df_sample.head(5), decision_threshold=0.35)
    # print(reports[0])