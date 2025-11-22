import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = "data/processed"
ARTIFACTS_DIR = "model_service/artifacts"

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

TARGET_COL = "TARGET_LABEL_BAD=1"


def find_best_threshold(y_true, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)

    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def run_training():

    logger.info("📥 Cargando datasets procesados...")

    X_train = pd.read_parquet(os.path.join(DATA_DIR, "X_train.parquet"))
    X_test = pd.read_parquet(os.path.join(DATA_DIR, "X_test.parquet"))
    y_train = pd.read_parquet(os.path.join(DATA_DIR, "y_train.parquet"))[TARGET_COL]
    y_test = pd.read_parquet(os.path.join(DATA_DIR, "y_test.parquet"))[TARGET_COL]

    logger.info(f"📊 X_train = {X_train.shape}, X_test = {X_test.shape}")

    logger.info("🎯 Configurando búsqueda de hiperparámetros...")

    param_dist = {
        "num_leaves": [31, 63, 127, 255],
        "max_depth": [-1, 7, 9, 11],
        "learning_rate": [0.1, 0.05, 0.02, 0.01],
        "n_estimators": [400, 800, 1200],
        "min_child_samples": [5, 10, 20, 40],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "scale_pos_weight": [1.5, 2.0, 2.5, 3.0],
    }

    search = RandomizedSearchCV(
        estimator=LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            random_state=42,
            n_jobs=-1,
        ),
        param_distributions=param_dist,
        n_iter=25,
        scoring="roc_auc",
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42,
    )

    logger.info("🚀 Entrenando modelo (puede tardar unos minutos)...")
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    logger.info(f"✅ Best params: {search.best_params_}")
    logger.info(f"📈 Best ROC-AUC CV: {search.best_score_:.4f}")

    logger.info("📊 Evaluando en test...")

    test_proba = best_model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, test_proba)
    logger.info(f"🏁 Test ROC-AUC: {roc:.4f}")

    logger.info("🔎 Calculando mejor threshold según F1...")
    best_threshold, best_f1 = find_best_threshold(y_test, test_proba)

    logger.info(f"🎯 Best threshold = {best_threshold:.4f}")
    logger.info(f"🎯 Best F1 = {best_f1:.4f}")

    model_path = os.path.join(ARTIFACTS_DIR, "model_stack_prod.pkl")
    joblib.dump(best_model, model_path)

    logger.info(f"💾 Modelo guardado en: {model_path}")

    metadata = {
        "best_params": search.best_params_,
        "best_threshold": best_threshold,
        "best_auc": roc,
        "best_f1": best_f1,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    meta_path = os.path.join(ARTIFACTS_DIR, "model_metadata.json")

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"📄 Metadata guardada en: {meta_path}")

    logger.info("🏁 Entrenamiento finalizado con éxito 🎉")


if __name__ == "__main__":
    run_training()
