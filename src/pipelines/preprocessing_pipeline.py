"""
===========================================================
🧼 PREPROCESSING PIPELINE
Ejecuta el pipeline completo de preprocesamiento:
 - carga datos
 - crea pipeline de features
 - ajusta (fit)
 - transforma (transform)
 - hace train/test split
 - guarda datasets procesados
 - guarda pipeline entrenado
===========================================================
"""

import os
import logging
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

# Importamos tu builder del pipeline
from src.features.build_features import build_preprocessing_pipeline

# Importamos tu función de split
from src.utils.split import split_and_save


# ============================================================
# 📝 CONFIGURACIÓN DE LOGGING PROFESIONAL
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


# ============================================================
# 🏗️ FUNCIÓN PRINCIPAL DEL PIPELINE
# ============================================================

def run_preprocessing(
    data_path="data/interim/train_clean_headers.parquet",
    target_col="label",
    processed_dir="data/processed",
    model_dir="models",
):
    """
    Ejecuta el preprocesamiento completo usando tu pipeline + split_and_save.
    """

    logger.info("🏁 Iniciando Preprocessing Pipeline...")

    # --------------------------------------------------------
    # 1️⃣ Cargar datos
    # --------------------------------------------------------
    logger.info(f"📥 Cargando datos desde: {data_path}")

    df = pd.read_parquet(data_path)

    if target_col not in df.columns:
        raise ValueError(f"La columna objetivo '{target_col}' no existe en el dataset.")

    logger.info(f"✅ Datos cargados correctamente: {df.shape[0]} filas, {df.shape[1]} columnas")

    # --------------------------------------------------------
    # 2️⃣ Separar X e y
    # --------------------------------------------------------
    X = df.drop(columns=[target_col])
    y = df[target_col]

    logger.info(f"🔧 Separadas variables predictoras y target: X={X.shape}, y={y.shape}")

    # --------------------------------------------------------
    # 3️⃣ Crear y ajustar pipeline
    # --------------------------------------------------------
    logger.info("🎛️ Construyendo pipeline de preprocesamiento...")
    pipeline = build_preprocessing_pipeline()

    logger.info("⚙️ Ajustando pipeline (fit)... puede tardar unos segundos")
    X_transformed = pipeline.fit_transform(X)

    logger.info(f"✨ Transformación completa. Nueva forma: {X_transformed.shape}")

    # --------------------------------------------------------
    # 4️⃣ Guardar pipeline entrenado
    # --------------------------------------------------------
    os.makedirs(model_dir, exist_ok=True)
    pipeline_path = os.path.join(model_dir, "preprocessing_pipeline.joblib")

    joblib.dump(pipeline, pipeline_path)
    logger.info(f"💾 Pipeline guardado en: {pipeline_path}")

    # --------------------------------------------------------
    # 5️⃣ Train/Test Split + Guardado
    # --------------------------------------------------------
    logger.info("✂️ Realizando train/test split...")

    df_processed = pd.DataFrame(X_transformed)
    df_processed[target_col] = y.values

    X_train, X_test, y_train, y_test = split_and_save(
        df_processed, target_col=target_col, output_dir=processed_dir
    )

    logger.info("📦 Datos procesados y guardados correctamente.")
    logger.info("🏁 Preprocessing Pipeline finalizado con éxito 🎉")


# ============================================================
# 🚀 PUNTO DE ENTRADA PARA EJECUCIÓN DIRECTA
# ============================================================
if __name__ == "__main__":
    run_preprocessing()
