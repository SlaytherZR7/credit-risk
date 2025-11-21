import os
import json
import pandas as pd
import numpy as np
import joblib
import time

# ======================================================================
# ⭐ CONFIGURACIÓN DE RUTAS (dentro del contenedor Docker)
# ======================================================================

BASE_DIR = "/app/artifacts"

PREPROCESSOR_PATH = os.path.join(BASE_DIR, "preprocessing_pipeline.joblib")
MODEL_PATH = os.path.join(BASE_DIR, "model_stack_prod.pkl")   # ← TU MODELO LGBM
META_PATH = os.path.join(BASE_DIR, "model_metadata.json")

# ======================================================================
# ⭐ VARIABLES GLOBALES (Cache)
# ======================================================================

_preprocessor = None
_model = None
_threshold = None


# ======================================================================
# ⭐ CARGA DE MODELO + PREPROCESSING (1 sola vez)
# ======================================================================

def init_model():
    global _preprocessor, _model, _threshold

    if _preprocessor is not None and _model is not None:
        return _preprocessor, _model

    # -----------------------
    # Preprocessing pipeline
    # -----------------------
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"❌ Preprocessor not found: {PREPROCESSOR_PATH}")

    print("🔹 Loading preprocessing pipeline...")
    _preprocessor = joblib.load(PREPROCESSOR_PATH)

    # -----------------------
    # Modelo LightGBM
    # -----------------------
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")

    print("🔹 Loading LightGBM model...")
    _model = joblib.load(MODEL_PATH)

    # -----------------------
    # Metadata (threshold)
    # -----------------------
    if os.path.exists(META_PATH):
        with open(META_PATH, "r") as f:
            metadata = json.load(f)
        _threshold = metadata.get("best_threshold", 0.5)
    else:
        _threshold = 0.5

    print("✅ Model + preprocessing loaded successfully")
    print(f"🔎 Threshold: {_threshold}")

    return _preprocessor, _model


# ======================================================================
# ⭐ PREDICCIÓN PARA UN SOLO REGISTRO
# ======================================================================

def predict_single(features: dict):
    preprocessor, model = init_model()

    # JSON → DataFrame
    X_raw = pd.DataFrame([features])

    # Preprocessing
    X_processed = preprocessor.transform(X_raw)

    # Predict prob
    proba = float(model.predict_proba(X_processed)[0, 1])
    if not np.isfinite(proba):
        proba = 0.0

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
    start_time = time.time()
    print(f"⏰ [0s] Iniciando predict_batch con {len(batch)} registros")
    
    preprocessor, model = init_model()
    print(f"⏰ [{time.time()-start_time:.1f}s] Modelo cargado")

    df = pd.DataFrame(batch)
    print(f"⏰ [{time.time()-start_time:.1f}s] DataFrame creado: {df.shape}")
    
    X_processed = preprocessor.transform(df)
    print(f"⏰ [{time.time()-start_time:.1f}s] Transformación completa: {X_processed.shape}")

    probas = model.predict_proba(X_processed)[:, 1]
    print(f"⏰ [{time.time()-start_time:.1f}s] Predicción completa")

    results = []
    for proba in probas:
        if not np.isfinite(proba):
            proba = 0.0
        pred = int(proba >= _threshold)
        results.append({
            "probability": float(proba),
            "prediction": pred,
            "threshold_used": _threshold
        })
    
    print(f"⏰ [{time.time()-start_time:.1f}s] Resultados formateados. Total: {len(results)}")
    return results
