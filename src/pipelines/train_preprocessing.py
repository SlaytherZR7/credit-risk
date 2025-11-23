"""
===========================================================
🧼 PREPROCESSING PIPELINE
Loads data
-Creates the feature pipeline
-Fits the pipeline
-Transforms the data
-Performs train/test split
-Saves the processed datasets
-Saves the trained pipeline
===========================================================
"""

import os
import logging
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
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

    logger.info("🏁 Starting Preprocessing Pipeline...")
    logger.info(f"📥 Loading data from: {data_path}")

    df = pd.read_parquet(data_path)

    if target_col not in df.columns:
        raise ValueError(f"The target column  '{target_col}' does not exist in the dataset.")

    logger.info(f"✅ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    logger.info(f"🔧 Predictor and target variables separated: X={X.shape}, y={y.shape}")
    logger.info("🎛️ Building preprocessing pipeline...")
    pipeline = build_preprocessing_pipeline()

    logger.info("⚙️ Fitting pipeline... this may take a few seconds")
    X_transformed = pipeline.fit_transform(X)

    logger.info(f"✨ Transformation complete. New shape:: {X_transformed.shape}")

    ARTIFACTS_DIR = "model_service/artifacts"
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    pipeline_path = os.path.join(ARTIFACTS_DIR, "preprocessing_pipeline.joblib")

    joblib.dump(pipeline, pipeline_path)
    logger.info(f"💾 Pipeline saved at: {pipeline_path}")
    logger.info("✂️ Performing train/test split...")

    df_processed = pd.DataFrame(X_transformed)
    df_processed[target_col] = y.values

    X_train, X_test, y_train, y_test = split_and_save(
        df_processed, target_col=target_col, output_dir=processed_dir
    )

    logger.info("📦 Processed data saved successfully.")
    logger.info("🏁 Preprocessing Pipeline completed successfully 🎉")

if __name__ == "__main__":
    run_preprocessing()
