
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def init():
    global model, features, threshold

    model_dir = Path(
        os.environ.get("AZUREML_MODEL_DIR", Path(__file__).parent)
    )

    # Find the model even if Azure ML nests the registered artifacts.
    model_path = model_dir / "model.joblib"

    if not model_path.exists():
        matches = list(model_dir.rglob("model.joblib"))

        if not matches:
            raise FileNotFoundError(
                f"model.joblib not found under {model_dir}"
            )

        model_path = matches[0]
        model_dir = model_path.parent

    # Compatibility aliases for NumPy paths used by serialized artifacts.
    import numpy.core

    sys.modules.setdefault("numpy._core", numpy.core)
    sys.modules.setdefault(
        "numpy._core.multiarray",
        numpy.core.multiarray,
    )

    # Load the saved model and metadata.
    model = joblib.load(model_path)

    features_path = model_dir / "features.json"
    metadata_path = model_dir / "metadata.json"

    if not features_path.exists():
        raise FileNotFoundError(
            f"features.json not found under {model_dir}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata.json not found under {model_dir}"
        )

    with features_path.open("r") as file:
        features = json.load(file)

    with metadata_path.open("r") as file:
        metadata = json.load(file)

    threshold = float(metadata.get("threshold", 0.5))

    print(f"Model loaded from: {model_path}")
    print(f"Features: {features}")
    print(f"Classification threshold: {threshold}")


def run(raw_data):
    try:
        # Accept either a JSON string or an already-parsed object.
        data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data

        # Support one transaction or a list of transactions.
        rows = data if isinstance(data, list) else [data]

        if not rows:
            return {"error": "Input must contain at least one transaction"}

        if not all(isinstance(row, dict) for row in rows):
            return {
                "error": "Input must be a JSON object or a list of JSON objects"
            }

        frame = pd.DataFrame(rows)

        missing = [
            column for column in features
            if column not in frame.columns
        ]

        if missing:
            return {
                "error": "Missing required features",
                "missing_features": missing,
            }

        # Preserve the feature order expected by the trained pipeline.
        frame = frame[features]

        probabilities = model.predict_proba(frame)[:, 1]
        predictions = (probabilities >= threshold).astype(int)

        results = [
            {
                "prediction": int(prediction),
                "fraud_probability": float(probability),
                "label": "fraud" if prediction == 1 else "legitimate",
            }
            for prediction, probability in zip(
                predictions, probabilities
            )
        ]

        return {"predictions": results}

    except Exception as exc:
        return {"error": str(exc)}
