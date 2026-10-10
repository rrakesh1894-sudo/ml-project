"""
ChurnGuard — model evaluation helpers.

Computes all required metrics on the held-out test set and generates:
    artifacts/metrics.json
    artifacts/confusion_matrix.png
    artifacts/roc_curve.png
    artifacts/precision_recall_curve.png

Every number saved here comes from the ACTUAL test set — nothing is
hardcoded or invented.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend: safe inside scripts/Streamlit
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

METRICS_JSON = ARTIFACTS_DIR / "metrics.json"

# Risk thresholds used by app.py and reported alongside metrics so the
# threshold rule is documented next to the model that produced it.
RISK_THRESHOLDS = {"low": 0.40, "high": 0.60}  # low < 0.40 <= medium < 0.60 <= high


def evaluate_pipeline(pipeline, X_test, y_test) -> dict:
    """Compute all required classification metrics on the test set."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]  # P(churn == 1)

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "average_precision": round(float(average_precision_score(y_test, y_proba)), 4),
    }


def generate_evaluation_artifacts(pipeline, X_test, y_test, model_name: str) -> dict:
    """
    Save metrics.json and the three PNG charts for the FINAL model.

    Returns the metrics dict (same content as metrics.json).
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = evaluate_pipeline(pipeline, X_test, y_test)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)

    # Feature importance (linear models: coefficients; trees: importances).
    importances = _get_feature_importances(pipeline)

    payload = {
        "final_model": model_name,
        "test_rows": int(len(y_test)),
        "churn_rate_test": round(float(y_test.mean()), 4),
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "risk_thresholds": RISK_THRESHOLDS,
        "feature_importance": importances,
    }

    with open(METRICS_JSON, "w") as f:
        json.dump(payload, f, indent=2)

    _plot_confusion_matrix(cm)
    _plot_roc(fpr, tpr, metrics["roc_auc"])
    _plot_pr(rec_curve, prec_curve, metrics["average_precision"])

    return payload


def _get_feature_importances(pipeline, top_n: int = 15) -> list:
    """Extract the top-N most influential features from the fitted pipeline."""
    clf = pipeline.named_steps["classifier"]
    pre = pipeline.named_steps["preprocessor"]

    # Get human-readable feature names after preprocessing.
    names = pre.get_feature_names_out()
    clean_names = [n.replace("num__", "").replace("cat__", "") for n in names]

    if hasattr(clf, "feature_importances_"):  # tree ensembles
        values = clf.feature_importances_
    elif hasattr(clf, "coef_"):  # linear models
        values = clf.coef_[0]
        # Show absolute impact but keep sign so direction is interpretable.
        pairs = sorted(zip(clean_names, values), key=lambda p: -abs(p[1]))
        return [{"feature": f, "coef": round(float(v), 4)} for f, v in pairs[:top_n]]
    else:
        return []

    pairs = sorted(zip(clean_names, values), key=lambda p: -p[1])
    return [{"feature": f, "importance": round(float(v), 4)} for f, v in pairs[:top_n]]


def _plot_confusion_matrix(cm) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Stay", "Churn"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title("Confusion Matrix (Test Set)")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def _plot_roc(fpr, tpr, auc: float) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})", color="tab:blue")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Test Set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)


def _plot_pr(rec_curve, prec_curve, ap: float) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec_curve, prec_curve, label=f"PR (AP = {ap:.3f})", color="tab:red")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (Test Set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "precision_recall_curve.png", dpi=150)
    plt.close(fig)
