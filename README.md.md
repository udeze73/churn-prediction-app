# 📉 Customer Churn Prediction Using Stacked Ensemble Learning, SHAP Explainability & Streamlit Deployment

## 🚀 Project Overview

Customer churn is one of the most critical business challenges faced by subscription-based and service-oriented organizations. Acquiring a new customer is often significantly more expensive than retaining an existing one. Consequently, the ability to identify customers at risk of leaving provides businesses with an opportunity to proactively intervene and improve customer retention.

This project develops an end-to-end machine learning solution for customer churn prediction using stacked ensemble learning, class imbalance handling, threshold optimization, SHAP explainability, and Streamlit deployment.

## 🚀 Live App
> https://churn_prediction_app.streamlit.app

## 📂 Project Structure

```
.
├── churn_notebook_clean.ipynb     # Full training pipeline: EDA, feature engineering,
│                                   # stacking, threshold tuning, SHAP analysis
├── churn_prediction_app.py        # Streamlit app — single customer prediction + SHAP
├── chun_model_bundle.pkl          # Saved trained models (base pipelines + meta model)
├── shap_artifacts.pkl             # Saved SHAP explainer objects
├── requirements.txt               # Python dependencies
├── .streamlit/config.toml         # App theme configuration
└── README.md                      # You are here
```

## 🎯 Project Objectives

- Predict whether a customer is likely to churn.
- Handle severe class imbalance effectively.
- Engineer meaningful business-driven features.
- Compare multiple machine learning algorithms.
- Build a stacked ensemble model for improved predictive performance.
- Explain model decisions using SHAP.
- Deploy the final solution as an interactive Streamlit application.

## ⚠️ Class Imbalance Handling

The dataset is highly imbalanced, making accuracy alone a misleading metric.

To mitigate this issue:

- Logistic Regression → `class_weight='balanced'`
- Random Forest → `class_weight='balanced'`
- SVM → `class_weight='balanced'`
- XGBoost → `scale_pos_weight`

These techniques force the models to pay greater attention to the minority churn class, improving recall and balanced classification performance.

## 🤖 Base Models

The following base learners were trained:

1. Logistic Regression (LR)
2. Random Forest (RF)
3. Support Vector Machine (SVM)
4. XGBoost (XGB)

Each model was embedded within a preprocessing pipeline to eliminate data leakage.

## 🔥 Why Stacking?

Each model learns different patterns:

- LR captures linear relationships.
- RF captures non-linear interactions.
- SVM learns complex decision boundaries.
- XGB captures boosted feature interactions.

Instead of choosing a single model, the project combines their strengths using a stacking framework.

Customer Features
        ↓
 LR + RF + SVM + XGB
        ↓
 Out-of-Fold Predictions
        ↓
 Gradient Boosting Meta Model
        ↓
 Final Churn Prediction

## 📂 Project Structure

```
.
├── churn_notebook_clean.ipynb     # Full training pipeline: EDA, feature engineering,
│                                   # stacking, threshold tuning, SHAP analysis
├── churn_prediction_app.py        # Streamlit app — single customer prediction + SHAP
├── chun_model_bundle.pkl          # Saved trained models (base pipelines + meta model)
├── shap_artifacts.pkl             # Saved SHAP explainer objects
├── requirements.txt               # Python dependencies
├── .streamlit/config.toml         # App theme configuration
└── README.md                      # You are here
```

## 📈 Base Model Performance

| Model | F1 | Precision | Recall | ROC-AUC | Balanced Accuracy |
|---------|------|----------|---------|----------|----------|
| Random Forest | 0.638 | 0.578 | 0.713 | 0.845 | 0.763 |
| XGBoost | 0.617 | 0.555 | 0.694 | 0.839 | 0.747 |
| LogisticRegression | 0.616 | 0.506 | 0.787 | 0.842 | 0.755 |
| SVM | 0.610 | 0.666 | 0.562 | 0.840 | 0.731 |

## 📊 Final stacked Model Performance

| Metric | Score |
|---|---|
| F1 Score | 0.636 |
| Precision | 0.551 |
| Recall | 0.752 |
| ROC-AUC | 0.839 |
| Balanced Accuracy | 0.766 |

## 🔍 Explainability with SHAP

SHAP was implemented using the Random Forest model.

Why Random Forest?

- Optimized TreeExplainer support.
- Easier business interpretation.
- Uses original customer features.
- Produces actionable explanations.

SHAP provides:

- Global feature importance
- Feature impact direction
- Customer-level prediction explanations

## 🌐 Streamlit Deployment

The final application includes:

- Customer churn prediction
- Probability scoring
- Threshold-adjusted decisions
- SHAP explanations
- Interactive UI

## 🏆 Skills Demonstrated

- Data Cleaning
- Feature Engineering
- Class Imbalance Handling
- Ensemble Learning
- Stacking Models
- Threshold Optimization
- SHAP Explainability
- Streamlit Deployment
- Production-Oriented Machine Learning

## 🛠️ Running Locally

```bash
# 1. Clone this repo
git clone (https://github.com/udeze73/churn-prediction-app)>.git
cd <churn-prediction-app>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run churn_prediction_app.py
```

This opens the app at `http://localhost:8501` in your browser.

## ☁️ Deployed on Streamlit Community Cloud

This app is deployed via [Streamlit Community Cloud](https://streamlit.io/cloud),
which auto-builds from this repo's `requirements.txt` and runs
`churn_prediction_app.py` directly — no separate server setup needed.


## 👨‍💻 Author

Joshua Udeze

## 📦 Dataset

[Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) —
7,043 customers, 21 features, publicly available IBM sample dataset.

## 📝 License

This project is for educational/portfolio purposes.
