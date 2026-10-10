"""
ChurnGuard — training entry point.

Run from the project root:
    python -m src.train

What it does (in order):
    1. Load + clean the Telco dataset via src/preprocessing.py.
    2. Stratified train/test split (80/20) — stratify so both splits keep
       the same churn ratio; prevents leakage by only fitting on train.
    3. Train Logistic Regression (baseline) and Random Forest, both with
       class_weight="balanced" for the ~73/27 class imbalance.
       NOTE: no oversampling anywhere — class_weight reweights the loss
       function instead of duplicating rows, so the test set stays pure.
    4. Evaluate both on the SAME held-out test set.
    5. Pick the best model by F1 (primary) then ROC-AUC (tie-break).
    6. Save the fitted full pipeline -> models/churn_pipeline.joblib
    7. Generate artifacts/metrics.json + evaluation PNGs via src/evaluate.py.

Every metric printed/saved comes from the test set — nothing is invented.
"""

import json
import sys
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.evaluate import ARTIFACTS_DIR, evaluate_pipeline, generate_evaluation_artifacts
from src.preprocessing import (
    build_preprocessor,
    load_and_clean,
    split_features_target,
)
from src.predict import make_prediction

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "churn_pipeline.joblib"
COMPARISON_JSON = ARTIFACTS_DIR / "model_comparison.json"

RANDOM_STATE = 42  # reproducibility: fixed seed everywhere


def build_pipeline(clf) -> Pipeline:
    """Attach a classifier to the shared preprocessing ColumnTransformer."""
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier", clf),
    ])


def get_candidates() -> dict:
    """Candidate models with class_weight='balanced' for the 73/27 imbalance."""
    return {
        "LogisticRegression": build_pipeline(LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
        "RandomForest": build_pipeline(RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    }


def main() -> None:
    print("=== ChurnGuard training ===")

    # 1-2) Data + leakage-safe split.
    df = load_and_clean()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Churn rate — train: {y_train.mean():.3f} | test: {y_test.mean():.3f}")

    # 3-4) Fit every candidate, evaluate on the same test set.
    results = {}
    for name, pipeline in get_candidates().items():
        print(f"\n--- Training {name} ---")
        pipeline.fit(X_train, y_train)
        metrics = evaluate_pipeline(pipeline, X_test, y_test)
        results[name] = metrics
        print(f"{name}: {metrics}")

    # 5) Select by F1 (primary), ROC-AUC (tie-break).
    final_name = max(
        results,
        key=lambda n: (results[n]["f1"], results[n]["roc_auc"]),
    )
    final_pipeline = get_candidates()[final_name]
    final_pipeline.fit(X_train, y_train)

    print(f"\n>>> Selected final model: {final_name}")

    # 6) Save fitted pipeline (preprocessing + classifier together).
    joblib.dump(final_pipeline, MODEL_PATH)
    print(f"Saved pipeline -> {MODEL_PATH}")

    # 7) Metrics + charts for the final model.
    payload = generate_evaluation_artifacts(final_pipeline, X_test, y_test, final_name)

    with open(COMPARISON_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved comparison  -> {COMPARISON_JSON}")
    print(f"Saved metrics     -> {ARTIFACTS_DIR / 'metrics.json'}")
    print("Saved charts      -> confusion_matrix.png, roc_curve.png, precision_recall_curve.png")

    # Post-training sanity check: predict one real test row through the
    # SAME helper the app will use.
    sample = X_test.iloc[[0]]
    truth = int(y_test.iloc[0])
    result = make_prediction(final_pipeline, sample)
    print(f"\nSanity check (row 0 of test set):")
    print(f"  truth={truth} | pred={result['prediction']} | prob={result['probability']:.3f}")

    print("\n=== Training complete ===")


if __name__ == "__main__":
    sys.exit(main())
