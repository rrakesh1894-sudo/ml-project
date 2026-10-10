# ChurnGuard — Resume Entry & Interview Prep

> All facts below describe features that were **actually implemented and
> verified**. Deployment URL is filled in after the Render deploy completes.

## Resume entry

**ChurnGuard — Customer Churn Prediction** · Python, Scikit-learn, Streamlit, Render
_[GitHub](https://github.com/rrakesh1894-sudo/ml-project)_

- Engineered a leakage-safe end-to-end ML pipeline on 7,032 real Telco
  customers (19 features) — stratified split, `ColumnTransformer` (scaling +
  one-hot), class-weighted model selection chosen by F1/ROC-AUC over accuracy.
- Compared Logistic Regression vs Random Forest on a held-out test set; final
  model reaches **ROC-AUC 0.835** / **recall 0.797** / **PR-AUC 0.618**, with
  an EDA notebook and full evaluation artifacts (confusion matrix, ROC, PR).
- Built a 4-page Streamlit app (prediction form, model performance dashboard,
  dataset insights) that serves the serialized scikit-learn pipeline via
  `joblib` — no retraining on request.
  **Deployed:** [https://churnguard-o1j3.onrender.com/](https://churnguard-o1j3.onrender.com/)

---

## 20 Interview Q&A

### Customer churn prediction

**1. What is customer churn, and why does the class imbalance matter?**
Churn = a customer stops using the service. Here ~27% churn (73/27 split).
Imbalance matters because accuracy becomes misleading: a "never churn" model
scores 73% accuracy while detecting zero churners. Metrics like recall, F1 and
PR-AUC, plus `class_weight="balanced"`, address this.

**2. What patterns did your EDA find?**
Month-to-month contracts, low tenure (<6 months), fiber-optic internet,
electronic-check payment, missing add-on services, and higher monthly charges
all correlate with higher churn. The trained Logistic Regression coefficients
rediscover the same signals (`tenure` −1.25, `Contract_Month-to-month` +0.70),
which is a good sanity check that the model learned real structure, not noise.

### Classification algorithms

**3. How does Logistic Regression work, intuitively?**
It models the log-odds of churn as a weighted sum of features:
`log(p/(1-p)) = w·x + b`. Each coefficient is the change in log-odds per unit
of the (scaled) feature. It assumes a linear decision boundary, is fast,
interpretable, and a strong baseline.

**4. Why Random Forest, and why did it lose here?**
Random Forest averages many decorrelated decision trees — it captures nonlinear
interactions without manual feature engineering. It won on accuracy (0.76 vs
0.73) but lost on F1/ROC-AUC because with class weights, Logistic Regression
traded some overall accuracy for substantially better recall on the minority
class — the commercially valuable behavior for churn.

### Data preprocessing

**5. How did you handle `TotalCharges`?**
It was stored as text with 11 blank strings. I converted with
`pd.to_numeric(errors="coerce")`, confirmed all 11 were `tenure == 0` brand-new
customers, and dropped them (<0.2% of data) — a documented decision in
`clean_data()`.

**6. Why scale numeric features for a tree-based model?**
Trees don't need scaling, but the pipeline is shared with Logistic Regression,
where `StandardScaler` helps gradient-based optimization converge. One pipeline
keeps train/serve behavior identical — a small correctness win over
per-model preprocessing.

### Train/test splitting & leakage

**7. Why stratify the split?**
`stratify=y` keeps the 26.6% churn rate in both train and test sets, so test
metrics reflect the real distribution instead of being skewed by an unlucky
draw — important with an imbalanced target.

**8. Where could data leakage happen in this project, and how did you prevent it?**
Two risks: (a) fitting the scaler/encoder on the full dataset before splitting —
prevented by wrapping preprocessing inside the `Pipeline` fitted only on the
training split; (b) oversampling before splitting — avoided entirely by using
`class_weight="balanced"` instead of resampling, keeping the test set pure.

### Class imbalance

**9. How does `class_weight="balanced"` work?**
It weights each class inversely to its frequency in the loss function
(`n_samples / (n_classes * n_samples_per_class)`), so misclassifying a minority
(churn) example costs the model more. No rows are duplicated or synthesized.

**10. Could you have used SMOTE instead?**
Yes, but only *after* splitting, and only on the training fold — applying it
before the split would leak synthetic minority patterns into the test set.
For a 7-day project, class weights gave most of the benefit with none of that
risk. Also worth mentioning: SMOTE with one-hot-encoded features creates
fraudulent categorical combinations.

### Precision, recall, F1

**11. Precision vs recall — which matters more for churn?**
It depends on the cost of each error. Recall matters more if losing a customer
is expensive (you'd rather over-contact than miss churners); precision matters
if retention offers are costly. My app exposes a three-tier risk system so the
business can choose where to act rather than a single hard threshold.

**12. Why is F1 the selection metric?**
F1 is the harmonic mean of precision and recall, penalizing either being low.
With 27% churn, it focuses the selection on minority-class performance instead
of majority-class accuracy.

### ROC-AUC & PR-AUC

**13. What does ROC-AUC 0.835 mean?**
The probability that the model ranks a random churner above a random
non-churner is 83.5%. It's threshold-independent ranking quality.

**14. Why report PR-AUC too?**
With an imbalanced dataset, ROC-AUC can look optimistic because it includes the
large true-negative pool. PR-AUC (average precision) focuses only on the
positive class — 0.618 vs a no-skill baseline of ~0.27 — giving a harder,
more honest view of minority-class detection.

### Threshold selection

**15. How did you choose the risk thresholds?**
Low <40% ≤ Medium <60% ≤ High, documented in `metrics.json` and shown in the
app. These are business-policy defaults, not fitted numbers — a fuller process
would tune them on a validation set by minimizing expected cost
(`cost_FP * FP + cost_FN * FN`).

### Pipelines & serialization

**16. Why use a scikit-learn `Pipeline` instead of preprocessing separately?**
It bundles preprocessing + classifier into one estimable object, so
`.fit()` on training data is the only place transformations are learned, and
the same object is what gets serialized and served — eliminating train/serve
skew by construction.

**17. Why Joblib over Pickle for the model file?**
Joblib uses a more efficient binary format for large numpy-backed objects
(faster, smaller) and is the sklearn-recommended serializer. The saved file is
only 8.6 KB because it contains the fitted transformers plus the classifier,
not the raw data.

### Streamlit

**18. How does Streamlit's execution model work, and why `st.cache_resource`?**
Streamlit reruns the entire script on every interaction (widget change, form
submit). `@st.cache_resource` memoizes the model load across reruns — without
it, the 8.6 KB model would be re-read from disk on every single interaction.
`@st.cache_data` similarly caches the dataset for the insights page.

**19. What happens in the app when a user submits the form?**
Widgets → a 1-row DataFrame with the exact training column names/order →
`pipeline.predict()` for the class and `predict_proba()` for the probability,
with the positive class located via `pipeline.classes_` (not column-order
assumption) → validation, then risk tiering and display.

### Deployment & limitations

**20. How is the app deployed, and what are its limitations?**
Render Web Service running `streamlit run app.py --server.address 0.0.0.0
--server.port $PORT`, with the model artifact committed to the repo so no
retraining happens on deploy. Limitations: trained on a single historical
dataset snapshot (concept drift risk), static threshold policy, no per-prediction
explanation method (SHAP would be the next step), and no automated retraining.
