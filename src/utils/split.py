# src/utils/split.py

"""
===========================================================
✂️ TRAIN/TEST SPLIT UTILITY
Funciones auxiliares para dividir datos y guardarlos
en data/processed/ de forma profesional.
===========================================================
"""

import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def split_and_save(
    df,
    target_col,
    output_dir="data/processed",
    test_size=0.2,
    random_state=42
):
    """
    Divide un DataFrame en train/test, guarda los conjuntos en disco y devuelve las particiones.

    Parámetros
    ----------
    df : pd.DataFrame
        Dataset procesado completo (incluye X y y).
    target_col : str
        Nombre de la columna objetivo.
    output_dir : str o Path
        Directorio donde se guardarán los archivos.
    test_size : float
        Proporción del test set.
    random_state : int
        Semilla para reproducibilidad.

    Returns
    -------
    X_train, X_test, y_train, y_test : pd.DataFrame / pd.Series
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Separar X e y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # 3. Guardar archivos
    X_train.to_parquet(output_path / "X_train.parquet")
    X_test.to_parquet(output_path / "X_test.parquet")
    y_train.to_frame(name=target_col).to_parquet(output_path / "y_train.parquet")
    y_test.to_frame(name=target_col).to_parquet(output_path / "y_test.parquet")

    print(f"✅ Guardado en {output_path} (train: {X_train.shape}, test: {X_test.shape})")

    return X_train, X_test, y_train, y_test
