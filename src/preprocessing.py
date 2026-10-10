"""
ChurnGuard — preprocessing utilities.

Everything here is shared by training (src/train.py) and the Streamlit app
(app.py) so that both apply IDENTICAL transformations to the data.

Key design rule (prevents data leakage and train/serve skew):
    - The ColumnTransformer + any scaling/encoding is ALWAYS wrapped inside a
      scikit-learn Pipeline that is fitted ONLY on the training split.
    - The same fitted Pipeline object is what we save with joblib and later
      load in app.py, so preprocessing seen by users is preprocessing seen
      by the model at training time.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Paths (relative to the project root — works both locally and on Render)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# ---------------------------------------------------------------------------
# Column definitions — single source of truth for the whole project
# ---------------------------------------------------------------------------
TARGET = "Churn"
DROP_COLUMNS = ["customerID"]  # identifier, has no predictive meaning

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_FEATURES  # order given to model


def load_raw_data(csv_path=DATA_PATH) -> pd.DataFrame:
    """Read the raw Telco CSV exactly as downloaded."""
    df = pd.read_csv(csv_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal, documented cleaning.

    Decision (Phase 2, user-confirmed): drop rows with blank TotalCharges.
    These are the 11 brand-new customers with tenure == 0 and a blank
    TotalCharges field — under 0.2% of the data. We DROP rather than impute
    to keep the pipeline simple for a 7-day project.
    """
    df = df.copy()

    # 1) TotalCharges arrives as text; blanks become NaN.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # 2) Drop rows where TotalCharges is missing (11 rows).
    df = df.dropna(subset=["TotalCharges"])

    # 3) Drop the identifier column.
    df = df.drop(columns=DROP_COLUMNS)

    # 4) Map the string target to binary: Yes -> 1 (churn), No -> 0.
    df[TARGET] = (df[TARGET] == "Yes").astype(int)

    return df.reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    """
    Build the ColumnTransformer WITHOUT fitting it.

    Scaling and one-hot encoding are applied here. Missing values are NOT
    a concern at this stage because clean_data() already guarantees they
    are gone, but we still include imputers so the pipeline is robust to
    any accidental NaN reaching it at prediction time.
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),  # helps LogisticRegression; harmless for trees
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    return preprocessor


def load_and_clean() -> pd.DataFrame:
    """Convenience wrapper: load raw data then apply clean_data()."""
    return clean_data(load_raw_data())


def split_features_target(df: pd.DataFrame):
    """Split a cleaned dataframe into X (features) and y (binary target)."""
    X = df[FEATURE_ORDER]
    y = df[TARGET]
    return X, y
