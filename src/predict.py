"""
ChurnGuard — prediction helpers.

One function, used in two places, so training and the app behave identically:
    1. src/train.py calls it as a post-training sanity check.
    2. app.py calls it for every user prediction.
"""

import numpy as np


def make_prediction(pipeline, input_df):
    """
    Run a fitted sklearn Pipeline on a single-row DataFrame of raw features.

    Returns a dict:
        {
            "prediction": 0|1,           # 1 = predicted to churn
            "probability": float,        # P(churn == 1), in [0, 1]
            "probability_pct": float,    # same, as a human-readable percentage
            "label": "Likely to Churn" | "Likely to Stay",
        }

    Raises ValueError if the pipeline has no predict_proba or the input is
    not a single row — both are programming errors, not user errors.
    """
    if input_df.shape[0] != 1:
        raise ValueError("make_prediction expects exactly one row of input.")

    pred = pipeline.predict(input_df)[0]

    if not hasattr(pipeline, "predict_proba"):
        raise ValueError("Pipeline does not support predict_proba().")

    proba = pipeline.predict_proba(input_df)[0]

    # Identify the POSITIVE class (churn == 1) using the model's own
    # class labels rather than assuming column order.
    classes = list(pipeline.classes_)
    if 1 not in classes:
        raise ValueError("Pipeline classes do not contain the positive label 1.")
    churn_idx = classes.index(1)
    churn_prob = float(proba[churn_idx])

    # Guard against floating point noise.
    churn_prob = float(np.clip(churn_prob, 0.0, 1.0))

    label = "Likely to Churn" if int(pred) == 1 else "Likely to Stay"

    return {
        "prediction": int(pred),
        "probability": churn_prob,
        "probability_pct": round(churn_prob * 100, 1),
        "label": label,
    }
