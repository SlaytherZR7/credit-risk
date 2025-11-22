import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

def run_data_preparation(
    variables_path=Path("data/raw/PAKDD2010_VariablesList.XLS"),
    raw_train_path=Path("data/raw/PAKDD2010_Modeling_Data.txt"),
    interim_output_path=Path("data/interim/train_clean_headers.parquet")
):
    """
    Lee datos RAW, aplica encabezados corregidos y guarda el dataset INTERIM.
    """

    logger.info("🏁 Iniciando Data Preparation Pipeline...")

    if not variables_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {variables_path}")

    if not raw_train_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {raw_train_path}")

    logger.info(f"📄 Leyendo VariablesList desde: {variables_path}")

    variables_df = pd.read_excel(variables_path)

    colnames = variables_df["Var_Title"].astype(str).tolist()

    if len(colnames) > 43:
        original_name = colnames[43]
        colnames[43] = "MATE_" + colnames[43]
        logger.info(f"🔧 Modificada columna 44: {original_name} → {colnames[43]}")

    logger.info(f"🧩 Total columnas definidas: {len(colnames)}")

    logger.info(f"📥 Leyendo dataset principal desde: {raw_train_path}")

    df_train = pd.read_csv(
        raw_train_path,
        sep="\t",
        low_memory=False,
        encoding="latin1",
        header=None,
        names=colnames
    )

    logger.info(f"✅ Datos cargados: {df_train.shape[0]} filas, {df_train.shape[1]} columnas")
    logger.info(f"📌 Primeras columnas: {colnames[:5]}")

    interim_output_path.parent.mkdir(parents=True, exist_ok=True)

    df_train.to_parquet(interim_output_path, index=False)

    logger.info(f"💾 Dataset INTERIM guardado en: {interim_output_path}")
    logger.info("🏁 Data Preparation Pipeline finalizado con éxito 🎉")


if __name__ == "__main__":
    run_data_preparation()
