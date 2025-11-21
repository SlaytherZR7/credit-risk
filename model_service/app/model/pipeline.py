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
    # ----------------------------------------------------
    # 🔍 VALIDACIÓN DE COLUMNAS
    # ----------------------------------------------------
    expected_cols = None

    # 1. Caso más simple → el preprocessor expone las columnas directamente
    if hasattr(preprocessor, "feature_names_in_"):
        expected_cols = list(preprocessor.feature_names_in_)
        print("📌 Preprocessor exposes feature_names_in_.")
        
    # 2. Caso Pipeline → buscar ColumnTransformer dentro
    elif hasattr(preprocessor, "named_steps"):
        for step_name, step_obj in preprocessor.named_steps.items():
            if hasattr(step_obj, "transformers_"):  # ColumnTransformer detectado
                expected_cols = []
                for name, trans, cols in step_obj.transformers_:
                    if cols != 'drop':
                        if isinstance(cols, list):
                            expected_cols.extend(cols)
                print(f"📌 ColumnTransformer found inside step: {step_name}")
                break

    # 3. Chequeo final
    if expected_cols is None:
        print("❌ ERROR: Could not extract expected columns from the preprocessor.")
        raise ValueError("Preprocessor does not expose required column metadata.")

    # Mostrar columnas esperadas
    print("📌 Expected columns from preprocessor:")
    print(expected_cols)

    # Comparación
    missing = [c for c in expected_cols if c not in df.columns]
    extra   = [c for c in df.columns if c not in expected_cols]

    if missing:
        print("❌ Missing columns:", missing)
    if extra:
        print("⚠️ Extra columns:", extra)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
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
            "risk_score": float(proba),
            "confidence": 1.0,
            "recommendation": "Reject" if pred == 1 else "Approve"
        })
    
    print(f"⏰ [{time.time()-start_time:.1f}s] Resultados formateados. Total: {len(results)}")

    if len(results) == 1:
        return results[0]

    return results

