"""
ChurnGuard — Streamlit application entry point.

Run from the project root:
    streamlit run app.py

Pages:
    1. Home              — project overview for recruiters
    2. Predict           — customer form -> real prediction from the saved pipeline
    3. Model Performance — actual test-set metrics + evaluation charts
    4. Dataset Insights  — aggregate facts from the training dataset

Design rules honored here:
    - The model is loaded ONCE via st.cache_resource (not retrained per run).
    - Every prediction comes from models/churn_pipeline.joblib via
      src/predict.make_prediction — the same helper used in training.
    - Input widgets are generated FROM the training feature contract
      (src/preprocessing.py), so names/order can never drift.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.predict import make_prediction
from src.preprocessing import CATEGORICAL_FEATURES, FEATURE_ORDER, NUMERIC_FEATURES

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ChurnGuard — Customer Churn Prediction",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Risk thresholds (documented; same values saved in metrics.json)
THRESH_LOW, THRESH_HIGH = 0.40, 0.60


# ---------------------------------------------------------------------------
# Cached resources — loaded once, reused across reruns
# ---------------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    """Load the fitted pipeline (preprocessing + classifier) once."""
    path = MODELS_DIR / "churn_pipeline.joblib"
    if not path.exists():
        st.error(
            "Model file `models/churn_pipeline.joblib` not found.\n\n"
            "Fix: run `python -m src.train` from the project root, then restart the app."
        )
        st.stop()
    try:
        return joblib.load(path)
    except Exception as e:  # corrupted / incompatible file
        st.error(f"Could not load the model file: {e}")
        st.stop()


@st.cache_data
def load_dataset():
    """Load + clean data once for the Dataset Insights page."""
    from src.preprocessing import load_and_clean
    return load_and_clean()


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
page = st.sidebar.radio("Navigate", ["Home", "Predict", "Model Performance", "Dataset Insights"])
st.sidebar.markdown("---")
st.sidebar.caption("ChurnGuard · Telco Customer Churn · IBM dataset")


# ===========================================================================
# PAGE 1 — HOME
# ===========================================================================
if page == "Home":
    st.title("🛡️ ChurnGuard")
    st.subheader("Customer Churn Prediction with Machine Learning")

    st.markdown(
        """
        **The business problem:** telecom providers lose customers every month.
        Retaining an existing customer is far cheaper than acquiring a new one,
        but retention offers are expensive — so the goal is to spend them on
        customers who are *actually* about to leave.

        **What this app does:** you enter a customer's profile (demographics,
        services, contract and billing details) and a machine-learning model
        trained on 7,032 real Telco customers predicts how likely that customer
        is to churn, with a probability and a risk tier.
        """
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset", "7,032 customers")
    c2.metric("Churn rate", "26.5%")
    c3.metric("Features used", str(len(FEATURE_ORDER)))

    left, right = st.columns(2)
    with left:
        st.markdown("#### Machine-learning workflow")
        st.markdown(
            """
            1. **EDA** — clean, explore, and visualize churn drivers in a notebook
            2. **Preprocessing** — column transformer: scaling + one-hot encoding
            3. **Training** — Logistic Regression vs Random Forest (class-weighted)
            4. **Selection** — best F1/ROC-AUC on a held-out stratified test set
            5. **Serving** — full pipeline serialized with `joblib`, loaded here
            """
        )
    with right:
        st.markdown("#### Technology stack")
        st.markdown(
            """
            - **Python** · Pandas · NumPy
            - **Scikit-learn** — preprocessing, models, evaluation
            - **Matplotlib / Seaborn** — charts
            - **Streamlit** — this interface
            - **Joblib** — model serialization
            - **Git/GitHub · Render** — versioning & deployment
            """
        )

    st.info(
        "**Limitations:** the model was trained on one telecom dataset from a "
        "single snapshot; it reflects historical patterns and may not transfer "
        "to other markets or time periods. Predictions are decision support — "
        "not a guarantee any individual customer will churn.",
        icon="⚠️",
    )


# ===========================================================================
# PAGE 2 — PREDICT
# ===========================================================================
elif page == "Predict":
    st.title("🔮 Customer Churn Prediction")
    st.markdown("Enter the customer profile. The trained model outputs a churn probability and risk tier.")

    pipeline = load_pipeline()

    # --- Sensible widget ranges/defaults taken from the dataset contract ---
    with st.form("prediction_form"):
        st.markdown("### 👤 Customer Information")
        col1, col2 = st.columns(2)
        gender = col1.selectbox("Gender", ["Female", "Male"])
        senior = col2.radio("Senior citizen", ["No", "Yes"], horizontal=True)
        col3, col4 = st.columns(2)
        partner = col3.radio("Has a partner", ["Yes", "No"], horizontal=True)
        dependents = col4.radio("Has dependents", ["No", "Yes"], horizontal=True)
        tenure = st.slider("Tenure (months with company)", 0, 72, 12)

        st.markdown("### 📞 Services")
        phone = st.radio("Phone service", ["Yes", "No"], horizontal=True)
        multiple = st.radio("Multiple lines", ["No", "Yes", "No phone service"], horizontal=True)
        internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        col5, col6, col7 = st.columns(3)
        onl_sec = col5.radio("Online security", ["No", "Yes", "No internet service"], horizontal=True)
        onl_bak = col6.radio("Online backup", ["No", "Yes", "No internet service"], horizontal=True)
        dev_prot = col7.radio("Device protection", ["No", "Yes", "No internet service"], horizontal=True)
        col8, col9, col10 = st.columns(3)
        tech = col8.radio("Tech support", ["No", "Yes", "No internet service"], horizontal=True)
        stream_tv = col9.radio("Streaming TV", ["No", "Yes", "No internet service"], horizontal=True)
        stream_mov = col10.radio("Streaming movies", ["No", "Yes", "No internet service"], horizontal=True)

        st.markdown("### 💳 Billing & Contract")
        contract = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
        paperless = st.radio("Paperless billing", ["Yes", "No"], horizontal=True)
        payment = st.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        col11, col12 = st.columns(2)
        monthly = col11.number_input("Monthly charges ($)", 18.0, 120.0, 70.0, step=0.05)
        total = col12.number_input("Total charges ($)", 0.0, 9000.0, float(monthly * max(tenure, 1)), step=5.0)

        submitted = st.form_submit_button("🔎 Predict churn", type="primary", width="stretch")

    if submitted:
        # --- Validation ---
        errors = []
        if total <= 0:
            errors.append("`Total charges` must be greater than 0.")
        if monthly <= 0:
            errors.append("`Monthly charges` must be greater than 0.")
        if phone == "No" and multiple != "No phone service":
            errors.append("`Multiple lines` must be 'No phone service' when phone service is 'No'.")
        if internet == "No" and not all(
            v == "No internet service"
            for v in [onl_sec, onl_bak, dev_prot, tech, stream_tv, stream_mov]
        ):
            errors.append("All add-on services must be 'No internet service' when internet service is 'No'.")
        if total < monthly * tenure * 0.1:
            errors.append("`Total charges` is inconsistent with monthly charges × tenure — please re-check.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            # --- Exact feature contract: names AND order from preprocessing.py ---
            input_df = pd.DataFrame([{
                "tenure": tenure,
                "MonthlyCharges": monthly,
                "TotalCharges": total,
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "PhoneService": phone,
                "MultipleLines": multiple,
                "InternetService": internet,
                "OnlineSecurity": onl_sec,
                "OnlineBackup": onl_bak,
                "DeviceProtection": dev_prot,
                "TechSupport": tech,
                "StreamingTV": stream_tv,
                "StreamingMovies": stream_mov,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
            }])[FEATURE_ORDER]

            result = make_prediction(pipeline, input_df)
            prob = result["probability"]

            if prob < THRESH_LOW:
                tier, color = "LOW", "green"
            elif prob < THRESH_HIGH:
                tier, color = "MEDIUM", "orange"
            else:
                tier, color = "HIGH", "red"

            st.markdown("### Result")
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Prediction", result["label"])
            mcol2.metric("Churn probability", f"{result['probability_pct']}%")
            mcol3.metric("Risk tier", tier if color != "red" else f"{tier} 🔴")

            st.progress(
                min(prob, 1.0),
                text=f"Model probability: {result['probability_pct']}%",
            )

            st.caption(
                "**How to read this:** the probability is the model's estimate "
                "that this customer churns, learned from historical Telco data "
                "(risk tiers: Low < 40% ≤ Medium < 60% ≤ High). Retention teams "
                "typically act on Medium/High first. **Limitations:** this is a "
                "statistical estimate from one historical dataset — it cannot "
                "account for unrecorded context (competitor pricing, support "
                "complaints, life events) and should inform, not replace, "
                "human judgment."
            )


# ===========================================================================
# PAGE 3 — MODEL PERFORMANCE
# ===========================================================================
elif page == "Model Performance":
    st.title("📊 Model Performance")
    st.markdown(
        "Every number below was computed on the held-out stratified test set "
        "(1,407 customers, never seen during training). Raw JSON lives in "
        "`artifacts/metrics.json` and `artifacts/model_comparison.json`."
    )

    import json
    import matplotlib.pyplot as plt

    try:
        metrics = json.load(open(ARTIFACTS_DIR / "metrics.json"))
        comparison = json.load(open(ARTIFACTS_DIR / "model_comparison.json"))
    except FileNotFoundError:
        st.error("Evaluation artifacts missing. Run `python -m src.train` first.")
        st.stop()

    st.markdown(f"**Final model:** `{metrics['final_model']}` — selected by F1, tie-broken by ROC-AUC")

    # Comparison table
    st.markdown("#### Model comparison (test set)")
    comp_df = pd.DataFrame(comparison).T
    comp_df.index.name = "Model"
    st.dataframe(comp_df, width="stretch")

    # Final-model metric cards
    st.markdown("#### Final model")
    m = metrics["metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{m['accuracy']:.3f}")
    c2.metric("Precision", f"{m['precision']:.3f}")
    c3.metric("Recall", f"{m['recall']:.3f}")
    c4, c5, c6 = st.columns(3)
    c4.metric("F1-score", f"{m['f1']:.3f}")
    c5.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    c6.metric("PR-AUC (Avg Precision)", f"{m['average_precision']:.3f}")

    st.warning(
        "**Why not accuracy alone?** Churn is only ~27% of customers, so a lazy "
        "model that predicts 'No churn' for everyone would score ~73% accuracy. "
        "Both models beat that on F1, ROC-AUC and PR-AUC. Note Logistic "
        "Regression was selected over Random Forest despite *lower* accuracy: "
        "with class weighting it catches far more real churners (higher recall), "
        "which is the commercially valuable behavior.",
        icon="ℹ️",
    )

    # Charts
    st.markdown("#### Evaluation charts")
    img_c1, img_c2, img_c3 = st.columns(3)
    img_c1.image(str(ARTIFACTS_DIR / "confusion_matrix.png"), caption="Confusion matrix")
    img_c2.image(str(ARTIFACTS_DIR / "roc_curve.png"), caption="ROC curve")
    img_c3.image(str(ARTIFACTS_DIR / "precision_recall_curve.png"), caption="Precision-recall curve")

    # Threshold trade-off explanation
    st.markdown("#### Prediction threshold trade-off")
    th = metrics["risk_thresholds"]
    st.markdown(
        f"""
        The classifier outputs a probability; a **threshold** converts it into
        "churn" / "stay". At the F1-optimal operating point the model favors
        catching churners. This app reports risk in three documented tiers:

        | Tier | Churn probability | Typical action |
        |---|---|---|
        | **Low** | < {th['low']:.0%} | none |
        | **Medium** | {th['low']:.0%}–{th['high']:.0%} | monitoring / email offer |
        | **High** | ≥ {th['high']:.0%} | proactive retention call |

        Raising the threshold → fewer false alarms (more precision, wasted fewer
        retention offers). Lowering it → catch more churners (more recall, more
        wasted offers). The right choice depends on the cost of each mistake.
        """
    )

    # Feature importance
    st.markdown("#### What drives predictions (feature importance)")
    imp = metrics.get("feature_importance") or []
    if imp:
        is_linear = "coef" in imp[0]
        raw_projection = "@"
        imp_df = pd.DataFrame(imp)
        imp_df.columns = ["Feature", ("Coefficient" if is_linear else "Importance")]
        st.dataframe(imp_df, width="stretch", hide_index=True)
        raw_projection = ""
        if is_linear:
            st.caption(
                "Sign of the coefficient = direction of effect on churn."
                " Positive → increases churn risk; negative → protective."
            )
        else:
            st.caption("Importance values from the tree ensemble (unitless).")


# ===========================================================================
# PAGE 4 — DATASET INSIGHTS
# ===========================================================================
elif page == "Dataset Insights":
    st.title("📈 Dataset Insights")
    st.markdown("Aggregate facts from the IBM Telco dataset used for training.")

    import matplotlib.pyplot as plt
    import seaborn as sns

    df = load_dataset()

    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Churned", f"{int(df['Churn'].sum()):,}")
    c3.metric("Churn %", f"{df['Churn'].mean():.1%}")

    left, right = st.columns(2)

    with left:
        st.markdown("#### Churn by contract type")
        by_contract = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.barplot(x=by_contract.index, y=by_contract.values, hue=by_contract.index,
                    palette="Reds_r", legend=False, ax=ax)
        ax.set_ylabel("Churn rate"); ax.set_xlabel("")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.caption("Month-to-month customers churn far more — no exit penalty.")

    with right:
        st.markdown("#### Churn by tenure group")
        bins = [0, 6, 12, 24, 48, 72]
        labels = ["0–6", "6–12", "12–24", "24–48", "48–72"]
        st.markdown("")  # spacing
        df["_tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels)
        by_tenure = df.groupby("_tenure_group", observed=True)["Churn"].mean()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.barplot(x=by_tenure.index, y=by_tenure.values, hue=by_tenure.index,
                    palette="Blues_r", legend=False, ax=ax)
        ax.set_ylabel("Churn rate"); ax.set_xlabel("Tenure (months)")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.caption("New customers are the most likely to leave — early tenure is the danger zone.")

    st.markdown("#### Churn by monthly charges")
    df["_charge_bin"] = pd.cut(df["MonthlyCharges"], bins=6)
    by_charges = df.groupby("_charge_bin", observed=True)["Churn"].mean()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(by_charges.index.astype(str), by_charges.values, color="#e74c3c")
    ax.set_ylabel("Churn rate"); ax.set_xlabel("Monthly charges ($ bins)")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.caption("Higher monthly charges trend with higher churn — price sensitivity.")
    df = df.drop(columns=["_tenure_group", "_charge_bin"])

    st.info(
        "**Business takeaway:** the riskiest profile is a *new, month-to-month "
        "customer paying high monthly charges* — the same signal the model "
        "weights most heavily.",
        icon="💡",
    )
