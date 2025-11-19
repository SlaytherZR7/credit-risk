import os
import json
import pandas as pd
import numpy as np
import cloudpickle

BASE_DIR = "/app/artifacts"
MODEL_PATH = os.path.join(BASE_DIR, "model_stack_prod.joblib")
META_PATH = os.path.join(BASE_DIR, "model_metadata.json")

_model = None
_threshold = None

def init_model():
    global _model, _threshold

    if _model is not None:
        return _model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Metadata file not found: {META_PATH}")

    print("🔹 Loading stacking model...")
    with open(MODEL_PATH, "rb") as f:
        _model = cloudpickle.load(f)

    print("🔹 Loading metadata...")
    with open(META_PATH) as f:
        metadata = json.load(f)

    _threshold = metadata.get("best_threshold", 0.5)

    print("Model loaded successfully")
    print(f"Threshold: {_threshold}")

    return _model


def preprocess_input(features: dict, expected_cols: list):
    X = pd.DataFrame([features])

    for col in expected_cols:
        if col not in X.columns:
            X[col] = np.nan

    X = X[expected_cols]

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X


def predict_single(features: dict, expected_cols: list):
    model = init_model()
    X = preprocess_input(features, expected_cols)

    proba = float(model.predict_proba(X)[0, 1])

    if not np.isfinite(proba):
        proba = 0.0

    pred = int(proba >= _threshold)

    return {
        "probability": proba,
        "prediction": pred,
        "threshold_used": _threshold
    }


def predict_batch(batch: list, expected_cols: list):
    results = []
    for row in batch:
        results.append(predict_single(row, expected_cols))
    return results
