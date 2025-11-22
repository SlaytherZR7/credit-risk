import os
import logging
import pandas as pd
import joblib

from src.features.build_features import build_preprocessing_pipeline
from src.utils.split import split_and_save

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

def run_preprocessing(
    data_path="data/interim/train_clean_headers.parquet",
    target_col="TARGET_LABEL_BAD=1",
    processed_dir="data/processed",
    model_dir="models",
):

    logger.info("🏁 Iniciando Preprocessing Pipeline...")

    logger.info(f"📥 Cargando datos desde: {data_path}")

    df = pd.read_parquet(data_path)

    if target_col not in df.columns:
        raise ValueError(f"La columna objetivo '{target_col}' no existe en el dataset.")

    logger.info(f"✅ Datos cargados correctamente: {df.shape[0]} filas, {df.shape[1]} columnas")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    logger.info(f"🔧 Separadas variables predictoras y target: X={X.shape}, y={y.shape}")

    logger.info("🎛️ Construyendo pipeline de preprocesamiento...")
    pipeline = build_preprocessing_pipeline()

    logger.info("⚙️ Ajustando pipeline (fit)... puede tardar unos segundos")
    X_transformed = pipeline.fit_transform(X)

    logger.info(f"✨ Transformación completa. Nueva forma: {X_transformed.shape}")

    ARTIFACTS_DIR = "model_service/artifacts"
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    pipeline_path = os.path.join(ARTIFACTS_DIR, "preprocessing_pipeline.joblib")

    joblib.dump(pipeline, pipeline_path)
    logger.info(f"💾 Pipeline guardado en: {pipeline_path}")

    logger.info("✂️ Realizando train/test split...")

    df_processed = pd.DataFrame(X_transformed)
    df_processed[target_col] = y.values

    X_train, X_test, y_train, y_test = split_and_save(
        df_processed, target_col=target_col, output_dir=processed_dir
    )

    logger.info("📦 Datos procesados y guardados correctamente.")
    logger.info("🏁 Preprocessing Pipeline finalizado con éxito 🎉")


if __name__ == "__main__":
    run_preprocessing()
