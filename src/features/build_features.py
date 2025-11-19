# src/features/build_features.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# ============================================================
# 🧩 1️⃣ BaseCleaner: Limpieza, FE y Drops (Paso 1 del Pipeline)
# ============================================================

class BaseCleaner(BaseEstimator, TransformerMixin):
    """
    Limpieza general, Ingeniería de Features (FE) y Eliminación de Columnas Ruidosas.
    """

    def __init__(self):
        # Columnas a ELIMINAR 
        self.columns_to_drop = [
            'ID_CLIENT', 'CLERK_TYPE', 'FLAG_RG', 'FLAG_CPF',
            'FLAG_INCOME_PROOF', 'FLAG_HOME_ADDRESS_DOCUMENT',
            'FLAG_ACSP_RECORD', 'FLAG_MOBILE_PHONE', 'EDUCATION_LEVEL',
            'QUANT_ADDITIONAL_CARDS', 
            # Flags de tarjetas eliminadas (su valor se captura en N_CARDS)
            'FLAG_DINERS', 'FLAG_AMERICAN_EXPRESS','FLAG_OTHER_CARDS',
            # Ubicación de alta cardinalidad/ruido
            'RESIDENCIAL_BOROUGH', 'PROFESSIONAL_BOROUGH',
            'RESIDENCIAL_CITY', 'PROFESSIONAL_CITY', 'CITY_OF_BIRTH'
        ]

        # Columnas Categóricas/Estado (que se mantienen) y necesitan limpieza de strings a NaN
        self.state_cols_to_clean = [
            'RESIDENCIAL_STATE', 
            'PROFESSIONAL_STATE',
            'STATE_OF_BIRTH',
            'SEX'
        ]
        
        # Columnas de CÓDIGO (Area Code y ZIP) que necesitan limpieza de strings a NaN
        self.code_cols_to_clean = [
            'RESIDENCIAL_PHONE_AREA_CODE', 
            'PROFESSIONAL_PHONE_AREA_CODE',
            'RESIDENCIAL_ZIP_3', 
            'PROFESSIONAL_ZIP_3',
        ] 
        
        # Columnas de Ingreso (que necesitan limpieza de strings a NaN ANTES del cálculo de FE)
        self.income_cols = ['PERSONAL_MONTHLY_INCOME', 'OTHER_INCOMES']

    # ... (self.card_flags ya no es necesario aquí, se define en transform) ...


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
    
        # Valores de error de hoja de cálculo y strings vacíos que deben ser NaN
        EXCEL_ERRORS = ['#DIV/0!', '#N/A', '#NAME?', '#NULL!', '#NUM!', '#REF!', '#VALUE!', ' ']

        # --- 1️⃣ LIMPIEZA DE STRINGS DE ERROR Y CONVERSIÓN (CRÍTICO) ---
        all_numeric_or_code_cols = self.income_cols + self.code_cols_to_clean

        for col in all_numeric_or_code_cols:
            if col in X.columns:
                # Reemplazar errores por NaN
                X[col] = X[col].replace(EXCEL_ERRORS, np.nan)
                # Convertir a numérico (errores → NaN)
                X[col] = pd.to_numeric(X[col], errors='coerce')

        # Limpieza de Strings Categóricos/Estado
        for col in self.state_cols_to_clean:
            if col in X.columns:
                X[col] = X[col].replace([' ', ''], np.nan).str.upper()

        # --- 2️⃣ INGENIERÍA DE FEATURES ---
        # N_CARDS: cantidad total de tarjetas
        card_flags = ['FLAG_VISA', 'FLAG_MASTERCARD', 'FLAG_DINERS', 'FLAG_AMERICAN_EXPRESS', 'FLAG_OTHER_CARDS']
        if all(c in X.columns for c in card_flags):
            X['N_CARDS'] = X[card_flags].fillna(0).sum(axis=1)

        # TOTAL_INCOME, INCOME_PER_DEPENDANT y LOG_TOTAL_INCOME
        if all(c in X.columns for c in self.income_cols):
            X['TOTAL_INCOME'] = X[self.income_cols[0]].fillna(0) + X[self.income_cols[1]].fillna(0)
            X['INCOME_PER_DEPENDANT'] = X['TOTAL_INCOME'] / (X['QUANT_DEPENDANTS'].fillna(0) + 1)
            X['LOG_TOTAL_INCOME'] = np.log1p(X['TOTAL_INCOME'])

        # HAS_CARDS: indicador binario de que posee al menos una tarjeta
        if 'N_CARDS' in X.columns:
            X['HAS_CARDS'] = (X['N_CARDS'] > 0).astype(int)

        # WORKS_SAME_STATE: trabaja en el mismo estado donde vive
        if all(c in X.columns for c in ['RESIDENCIAL_STATE', 'PROFESSIONAL_STATE']):
            X['WORKS_SAME_STATE'] = (
                X['RESIDENCIAL_STATE'].fillna('') == X['PROFESSIONAL_STATE'].fillna('')
            ).astype(int)

        # --- 3️⃣ LIMPIEZA DE OUTLIERS ---
        # QUANT_DEPENDANTS: valores fuera de rango reemplazados por mediana
        if 'QUANT_DEPENDANTS' in X.columns:
            median_dependants = X.loc[X['QUANT_DEPENDANTS'] <= 15, 'QUANT_DEPENDANTS'].median()
            X.loc[X['QUANT_DEPENDANTS'] > 15, 'QUANT_DEPENDANTS'] = median_dependants

        # AGE: eliminar menores de edad (outliers de negocio)
        if 'AGE' in X.columns:
            # Reemplazar edades menores a 18 por 18
            X.loc[X['AGE'] < 18, 'AGE'] = 18

            # Crear grupo de edades (binned feature)
            X['AGE_GROUP'] = pd.cut(
                X['AGE'],
                bins=[18, 25, 35, 45, 55, 65, 120],
                labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'],
                right=False
            )

        # --- 4️⃣ DROP Y MAPEO FINAL ---
        # Eliminación de columnas irrelevantes
        X = X.drop(columns=[col for col in self.columns_to_drop if col in X.columns], errors='ignore')

        # Mapeo de binarias Y/N a 1/0
        yn_cols = ['COMPANY', 'FLAG_RESIDENCIAL_PHONE', 'FLAG_PROFESSIONAL_PHONE']
        binary_map = {'Y': 1, 'N': 0, 1: 1, 0: 0}
        for col in yn_cols:
            if col in X.columns:
                X[col] = X[col].replace({'': 'N', np.nan: 'N', ' ': 'N'}).map(binary_map)

        return X


# ============================================================
# 🧩 2️⃣ CodeImputer (Imputación Centinela con Flag)
# ============================================================
# (Esta clase queda igual)
class CodeImputer(BaseEstimator, TransformerMixin):
    """ Imputa NaNs en códigos numéricos con un valor centinela (-1) y crea un flag."""
    
    def __init__(self, fill_value=-1, add_flag=True):
        self.fill_value = fill_value
        self.add_flag = add_flag

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in X.columns:
            # 1. Crear flag WAS_NULL antes de imputar
            if self.add_flag:
                flag_col = f"{col}_WAS_NULL"
                X[flag_col] = X[col].isna().astype(int)
            
            # 2. Imputar y asegurar tipo entero
            X[col] = X[col].fillna(self.fill_value).astype(int)
        return X

    def get_feature_names_out(self, input_features=None):
        """Devuelve los nombres de las columnas de salida del transformador."""
        # Se requiere este método para la compatibilidad con ColumnTransformer en scikit-learn
        if input_features is None:
            # Si no hay input_features, usamos X.columns si el transformador ya fue ajustado
            try:
                input_features = self.columns_ # asumiendo que el fit guarda las columnas
            except AttributeError:
                # Fallback, aunque no debería ocurrir en un flujo normal
                return np.array([f"{col}" for col in range(len(input_features))])
            
        output_features = list(input_features)
        if self.add_flag:
            output_features += [f"{col}_WAS_NULL" for col in input_features]
        return np.array(output_features)
    
# ============================================================
# 🧩 3️⃣ build_preprocessing_pipeline (Ensamblaje)
# ============================================================

def build_preprocessing_pipeline():
    """Crea el pipeline completo de preprocesamiento, adaptado al negocio."""

    # 1. Definición de Listas de Columnas Finales (Después de BaseCleaner)

    # 1.1 Columnas Numéricas a Imputar con la Mediana y Escalar (CONTINUAS)
    # NOTA: Las columnas de ingreso ya se usaron para FE y no deben estar aquí
    NUMERIC_FEATS = [
        'PERSONAL_ASSETS_VALUE', 
        'QUANT_BANKING_ACCOUNTS', 'QUANT_SPECIAL_BANKING_ACCOUNTS', 
        'AGE', 'MONTHS_IN_RESIDENCE', 'MONTHS_IN_THE_JOB'
    ]

    # 1.2 Columnas de Códigos (Imputación Centinela -1)
    # Incluimos los ZIPs y Area Codes que limpiamos en BaseCleaner
    CODE_FEATS = [
        'PROFESSION_CODE', 'OCCUPATION_TYPE', 'MATE_PROFESSION_CODE', 
        'MATE_EDUCATION_LEVEL', 'RESIDENCE_TYPE',
        'RESIDENCIAL_PHONE_AREA_CODE', 
        'PROFESSIONAL_PHONE_AREA_CODE',
        'RESIDENCIAL_ZIP_3', # <-- AGREGADO
        'PROFESSIONAL_ZIP_3' # <-- AGREGADO
    ]

    # 1.3 Categóricas/Estado (Imputación de Moda + OHE)
    OHE_FEATS = [
        'SEX', 'PRODUCT', 'NACIONALITY', 'MARITAL_STATUS', 
        'APPLICATION_SUBMISSION_TYPE', 'POSTAL_ADDRESS_TYPE',
        'RESIDENCIAL_STATE', 'PROFESSIONAL_STATE', 'STATE_OF_BIRTH',
        'AGE_GROUP' 
    ]


    # --- Definición de Tubos ---
    
    # Tubo para Numéricas (Mediana + Escalado Estándar)
    numeric_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Tubo para Categóricas (Moda + OHE)
    categorical_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Tubo para Códigos (Centinela con Flag)
    code_imputer_pipe = CodeImputer(fill_value=-1, add_flag=True)


    # --- Ensamblaje del ColumnTransformer (Paso 2) ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric_pipe', numeric_pipe, NUMERIC_FEATS),
            ('categorical_pipe', categorical_pipe, OHE_FEATS),
            ('code_imputer_pipe', code_imputer_pipe, CODE_FEATS),
        ], 
        # Features creadas (LOG_INCOME, N_CARDS, INCOME_PER_DEPENDANT) y binarias mapeadas pasan sin alterarse.
        remainder='passthrough' 
    )

    # --- Pipeline Final ---
    pipeline = Pipeline(steps=[
        ('base_cleaner', BaseCleaner()), 
        ('preprocessor', preprocessor) 
    ])

    return pipeline