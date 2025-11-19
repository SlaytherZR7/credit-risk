import os
import json
import pandas as pd
import numpy as np
import cloudpickle

# ======================================================================
# ⭐ CONFIGURACIÓN DE RUTAS (dentro del contenedor Docker)
# ======================================================================

BASE_DIR = "/app/artifacts"

PREPROCESSOR_PATH = os.path.join(BASE_DIR, "preprocessing_pipeline.joblib")
MODEL_PATH = os.path.join(BASE_DIR, "model_stack_prod.pkl")   # ← TU MODELO
META_PATH = os.path.join(BASE_DIR, "model_metadata.json")

# ======================================================================
# ⭐ VARIABLES GLOBALES
# ======================================================================

_preprocessor = None
_model = None
_threshold = None


# ======================================================================
# ⭐ CARGA DE MODELO + PREPROCESSING (solo se ejecuta UNA VEZ)
# ======================================================================

def init_model():
    """
    Carga el pipeline de preprocesamiento + modelo entrenado +
    metadata del threshold. Se ejecuta una sola vez en el startup
    del servicio o antes de la primera predicción.
    """
    global _preprocessor, _model, _threshold

    # Si ya está cargado, no volver a cargarlo
    if _preprocessor is not None and _model is not None:
        return _preprocessor, _model

    # -----------------------
    # Cargar Preprocessing
    # -----------------------
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"❌ Preprocessor not found: {PREPROCESSOR_PATH}")

    print("🔹 Loading preprocessing pipeline...")
    with open(PREPROCESSOR_PATH, "rb") as f:
        _preprocessor = cloudpickle.load(f)

    # -----------------------
    # Cargar Modelo LGBM
    # -----------------------
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")

    print("🔹 Loading LightGBM model...")
    with open(MODEL_PATH, "rb") as f:
        _model = cloudpickle.load(f)

    # -----------------------
    # Cargar Metadata
    # -----------------------
    if os.path.exists(META_PATH):
        with open(META_PATH, "r") as f:
            metadata = json.load(f)
        _threshold = metadata.get("best_threshold", 0.5)
    else:
        _threshold = 0.5   # valor por defecto

    print("✅ Model + preprocessing loaded successfully")
    print(f"🔎 Threshold: {_threshold}")

    return _preprocessor, _model


# ======================================================================
# ⭐ PREDICCIÓN PARA UN SOLO REGISTRO
# ======================================================================

def predict_single(features: dict):
    """
    Flujo de predicción:
    - recibe JSON con features crudas
    - arma DataFrame
    - aplica pipeline de preprocessing
    - aplica modelo LightGBM
    """
    preprocessor, model = init_model()

    # JSON → DataFrame
    X_raw = pd.DataFrame([features])

    # Preprocessing oficial
    X_processed = preprocessor.transform(X_raw)

    # Predict proba
    proba = float(model.predict_proba(X_processed)[0, 1])

    if not np.isfinite(proba):
        proba = 0.0

    # Clasificación con threshold
    pred = int(proba >= _threshold)

    return {
        "probability": proba,
        "prediction": pred,
        "threshold_used": _threshold,
    }


# ======================================================================
# ⭐ PREDICCIÓN BATCH
# ======================================================================

def predict_batch(batch: list):
    """
    Predicción para múltiples filas.
    """
    results = []
    for row in batch:
        results.append(predict_single(row))
    return results
