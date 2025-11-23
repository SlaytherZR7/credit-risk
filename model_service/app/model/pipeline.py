import os
import json
import pandas as pd
import numpy as np
import joblib

# ======================================================================
# ⭐ FORZAR SINGLE-THREADED (evitar deadlocks)
# ======================================================================
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# ======================================================================
# ⭐ CONFIGURACIÓN DE RUTAS (dentro del contenedor Docker)
# ======================================================================

BASE_DIR = "/app/artifacts"

PREPROCESSOR_PATH = os.path.join(BASE_DIR, "preprocessing_pipeline.joblib")
MODEL_PATH = os.path.join(BASE_DIR, "model_stack_prod.pkl")
META_PATH = os.path.join(BASE_DIR, "model_metadata.json")

# ======================================================================
# ⭐ VARIABLES GLOBALES (Cache)
# ======================================================================

_preprocessor = None
_model = None
_threshold = None


# ======================================================================
# ⭐ INIT DEL MODELO (se ejecuta 1 sola vez)
# ======================================================================

def init_model():
    global _preprocessor, _model, _threshold

    if _preprocessor is not None and _model is not None:
        return _preprocessor, _model

    # --- Preprocessor ---
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"❌ Preprocessor not found: {PREPROCESSOR_PATH}")

    print("🔹 Loading preprocessing pipeline...")
    _preprocessor = joblib.load(PREPROCESSOR_PATH)

    # --- Modelo ---
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")

    print("🔹 Loading final model...")
    _model = joblib.load(MODEL_PATH)

    # --- Metadata ---
    if os.path.exists(META_PATH):
        with open(META_PATH, "r") as f:
            metadata = json.load(f)
        _threshold = metadata.get("best_threshold", 0.5)
    else:
        _threshold = 0.5

    print(f"🔎 Threshold: {_threshold}")

    return _preprocessor, _model


# ======================================================================
# ⭐ UTILIDADES PARA VALIDACIÓN
# ======================================================================

def get_required_input_columns(preprocessor):
    """
    Devuelve SOLO las columnas que el JSON debe incluir.
    NO incluye columnas derivadas internas del BaseCleaner.
    """

    base_cleaner = preprocessor.named_steps["base_cleaner"]

    # columnas derivadas dentro de BaseCleaner → NO deben venir en el JSON
    internal_cols = set([
        "AGE_GROUP",
        "TOTAL_INCOME",
        "INCOME_PER_DEPENDANT",
        "LOG_TOTAL_INCOME",
        "N_CARDS",
        "HAS_CARDS",
        "WORKS_SAME_STATE",
    ])

    # columnas que BaseCleaner usa/preprocesa y que SI deben venir
    required = set(
        base_cleaner.state_cols_to_clean +
        base_cleaner.code_cols_to_clean +
        base_cleaner.income_cols
    )

    ct = preprocessor.named_steps["preprocessor"]

    ct_required = set()
    for name, trans, cols in ct.transformers:
        if cols == "drop":
            continue
        if isinstance(cols, list):
            for col in cols:
                if col not in internal_cols:
                    ct_required.add(col)

    return ct_required - internal_cols


def validate_columns(df, preprocessor):
    """
    Valida que el JSON tenga las columnas *necesarias*.
    NO exige columnas internas generadas por BaseCleaner.
    """
    required = get_required_input_columns(preprocessor)

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required JSON columns: {sorted(missing)}")

    return True


# ======================================================================
# ⭐ PREDICCIÓN SINGLE
# ======================================================================

def predict_single(features: dict):
    preprocessor, model = init_model()

    df = pd.DataFrame([features])

    validate_columns(df, preprocessor)

    X_proc = preprocessor.transform(df)

    proba = float(model.predict_proba(X_proc)[0, 1])
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
    preprocessor, model = init_model()

    df = pd.DataFrame(batch)

    validate_columns(df, preprocessor)

    X_proc = preprocessor.transform(df)
    probas = model.predict_proba(X_proc)[:, 1]

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
    return results
