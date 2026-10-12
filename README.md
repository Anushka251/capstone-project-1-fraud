
cat > README.md <<'EOF'
# Online Fraud Detection

An end-to-end machine learning capstone project for detecting potentially fraudulent financial transactions.

## 🚀 Live Demo

[**Open the Fraud Detection UI**](https://capstone-project-1-fraud.streamlit.app/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://capstone-project-1-fraud.streamlit.app/)

## Project Components

- **Machine Learning:** Trained Random Forest classifier
- **Model inference:** Saved model with fraud probability output
- **API:** FastAPI endpoints for health checks and predictions
- **Interactive UI:** Streamlit transaction form
- **Model artifacts:** Serialized model and feature metadata
- **Deployment:** Streamlit Community Cloud

## Run Locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the UI:

```bash
streamlit run app.py
```

## Example Prediction

The API has been tested with a sample transaction and returned a legitimate classification with a fraud probability of approximately 13.33%.

## Repository

Source code and model artifacts are available in this GitHub repository.
EOF
