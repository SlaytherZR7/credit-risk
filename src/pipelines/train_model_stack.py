"""
===========================================================
🏋️‍♂️ TRAIN STACKING MODEL (uses YOUR preprocessing)
 - Carga datos crudos
 - Aplica TU preprocessing_pipeline.joblib
 - Entrena el modelo stacking de tu compañero
 - Calibración opcional
 - Genera model_stack.joblib + metadata
===========================================================
"""

import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
import joblib
import cloudpickle

from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
RAW_DATA_PATH = "data/interim/train_clean_headers.parquet"
ARTIFACTS_DIR = "model_service/artifacts"
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, "preprocessing_pipeline.joblib")

TARGET_COL = "TARGET_LABEL_BAD=1"

os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------
def find_best_threshold(y_true, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


# ---------------------------------------------------------
# MAIN TRAINING PIPELINE
# ---------------------------------------------------------
def run_training(calibrate=True):

    # ------------------------------------------
    # 1) Load raw data
    # ------------------------------------------
    logger.info("📥 Cargando dataset RAW...")

    df = pd.read_parquet(RAW_DATA_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"No existe columna target {TARGET_COL}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    logger.info(f"📊 Dataset: {df.shape[0]} filas, {df.shape[1]} columnas")

    # ------------------------------------------
    # 2) Load YOUR preprocessing
    # ------------------------------------------
    logger.info("🔄 Cargando TU preprocessing pipeline...")

    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"No se encontró preprocessing_pipeline.joblib en {PREPROCESSOR_PATH}")

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # Split raw BEFORE processing
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    logger.info("🔄 Aplicando preprocesamiento a train/test...")
    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    logger.info(f"📏 X_train procesado: {X_train.shape}")
    logger.info(f"📏 X_test procesado:  {X_test.shape}")

    # ------------------------------------------
    # 3) Define Stacking model (como el de tu compañero)
    # ------------------------------------------
    logger.info("🧠 Definiendo modelo stacking...")

    xgb_base = XGBClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='auc',
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )

    lgbm_base = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    meta_xgb = XGBClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='auc',
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )

    stack_model = StackingClassifier(
        estimators=[("xgb", xgb_base), ("lgbm", lgbm_base)],
        final_estimator=meta_xgb,
        cv=5,
        passthrough=True,
        n_jobs=-1
    )

    # ------------------------------------------
    # 4) Fit model
    # ------------------------------------------
    logger.info("🏋️ Entrenando modelo stacking...")
    stack_model.fit(X_train, y_train)

    # ------------------------------------------
    # 5) Optionally calibrate
    # ------------------------------------------
    if calibrate:
        logger.info("🎯 Calibrando probabilidades...")
        calibrated = CalibratedClassifierCV(stack_model, method="isotonic", cv=3)
        calibrated.fit(X_train, y_train)
        final_model = calibrated
    else:
        logger.info("⚙️ Sin calibración.")
        final_model = stack_model

    # ------------------------------------------
    # 6) Evaluate
    # ------------------------------------------
    logger.info("📊 Evaluando en test...")
    y_proba = final_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    best_thr, best_f1 = find_best_threshold(y_test, y_proba)

    logger.info(f"🏁 AUC test = {auc:.4f}")
    logger.info(f"🔎 Threshold óptimo = {best_thr:.4f}")
    logger.info(f"🎯 F1 = {best_f1:.4f}")

    # ------------------------------------------
    # 7) Save artifacts
    # ------------------------------------------
    model_path = os.path.join(ARTIFACTS_DIR, "model_stack_prod.pkl")

    with open(model_path, "wb") as f:
        cloudpickle.dump(final_model, f)

    logger.info(f"💾 Modelo guardado en: {model_path}")

    metadata = {
        "auc": auc,
        "best_threshold": best_thr,
        "best_f1": best_f1,
        "calibrated": calibrate,
    }

    meta_path = os.path.join(ARTIFACTS_DIR, "model_metadata.json")

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"📄 Metadata guardada en: {meta_path}")
    logger.info("🏁 Entrenamiento STACK completado con éxito 🎉")


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", type=str, default="true")
    args = parser.parse_args()

    calibrate_flag = args.calibrate.lower() == "true"

    run_training(calibrate=calibrate_flag)
