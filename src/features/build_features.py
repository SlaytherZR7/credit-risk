import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# ============================================================
# 🧹 1️⃣ BaseCleaner: Limpieza, Feature Engineering, Drops
# ============================================================

class BaseCleaner(BaseEstimator, TransformerMixin):
    """
    Limpieza general, Ingeniería de Features (FE) y Eliminación de Columnas Ruidosas.
    """

    def __init__(self):
        self.columns_to_drop = [
            'ID_CLIENT', 'CLERK_TYPE', 'FLAG_RG', 'FLAG_CPF',
            'FLAG_INCOME_PROOF', 'FLAG_HOME_ADDRESS_DOCUMENT',
            'FLAG_ACSP_RECORD', 'FLAG_MOBILE_PHONE', 'EDUCATION_LEVEL',
            'QUANT_ADDITIONAL_CARDS',
            'FLAG_DINERS', 'FLAG_AMERICAN_EXPRESS', 'FLAG_OTHER_CARDS',
            'RESIDENCIAL_BOROUGH', 'PROFESSIONAL_BOROUGH',
            'RESIDENCIAL_CITY', 'PROFESSIONAL_CITY', 'CITY_OF_BIRTH'
        ]

        self.state_cols_to_clean = [
            'RESIDENCIAL_STATE',
            'PROFESSIONAL_STATE',
            'STATE_OF_BIRTH',
            'SEX'
        ]

        self.code_cols_to_clean = [
            'RESIDENCIAL_PHONE_AREA_CODE',
            'PROFESSIONAL_PHONE_AREA_CODE',
            'RESIDENCIAL_ZIP_3',
            'PROFESSIONAL_ZIP_3',
        ]

        self.income_cols = ['PERSONAL_MONTHLY_INCOME', 'OTHER_INCOMES']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        EXCEL_ERRORS = ['#DIV/0!', '#N/A', '#NAME?', '#NULL!', '#NUM!',
                        '#REF!', '#VALUE!', ' ']

        # Limpieza numérica
        all_numeric_or_code_cols = self.income_cols + self.code_cols_to_clean
        for col in all_numeric_or_code_cols:
            if col in X.columns:
                X[col] = X[col].replace(EXCEL_ERRORS, np.nan)
                X[col] = pd.to_numeric(X[col], errors='coerce')

        # Limpieza categórica
        for col in self.state_cols_to_clean:
            if col in X.columns:
                X[col] = X[col].replace([' ', ''], np.nan).str.upper()

        # Ingeniería de features
        card_flags = ['FLAG_VISA', 'FLAG_MASTERCARD',
                      'FLAG_DINERS', 'FLAG_AMERICAN_EXPRESS',
                      'FLAG_OTHER_CARDS']
        if all(c in X.columns for c in card_flags):
            X['N_CARDS'] = X[card_flags].fillna(0).sum(axis=1)

        if all(c in X.columns for c in self.income_cols):
            X['TOTAL_INCOME'] = X[self.income_cols[0]].fillna(0) + \
                                X[self.income_cols[1]].fillna(0)
            X['INCOME_PER_DEPENDANT'] = X['TOTAL_INCOME'] / \
                                        (X['QUANT_DEPENDANTS'].fillna(0) + 1)
            X['LOG_TOTAL_INCOME'] = np.log1p(X['TOTAL_INCOME'])

        if 'N_CARDS' in X.columns:
            X['HAS_CARDS'] = (X['N_CARDS'] > 0).astype(int)

        if all(c in X.columns for c in ['RESIDENCIAL_STATE', 'PROFESSIONAL_STATE']):
            X['WORKS_SAME_STATE'] = (
                X['RESIDENCIAL_STATE'].fillna('') ==
                X['PROFESSIONAL_STATE'].fillna('')
            ).astype(int)

        # Outliers dependants
        if 'QUANT_DEPENDANTS' in X.columns:
            median_dependants = X.loc[X['QUANT_DEPENDANTS'] <= 15,
                                      'QUANT_DEPENDANTS'].median()
            X.loc[X['QUANT_DEPENDANTS'] > 15, 'QUANT_DEPENDANTS'] = median_dependants

        # AGE + binning
        if 'AGE' in X.columns:
            X.loc[X['AGE'] < 18, 'AGE'] = 18
            X['AGE_GROUP'] = pd.cut(
                X['AGE'],
                bins=[18, 25, 35, 45, 55, 65, 120],
                labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'],
                right=False
            )

        # Dropeo
        X = X.drop(columns=[c for c in self.columns_to_drop if c in X.columns],
                   errors='ignore')

        # Map de Y/N → 1/0
        yn_cols = ['COMPANY', 'FLAG_RESIDENCIAL_PHONE', 'FLAG_PROFESSIONAL_PHONE']
        binary_map = {'Y': 1, 'N': 0, 1: 1, 0: 0}
        for col in yn_cols:
            if col in X.columns:
                X[col] = X[col].replace({'': 'N', np.nan: 'N', ' ': 'N'}).map(binary_map)

        return X


# ============================================================
# 🧩 2️⃣ CodeImputer con Flag — COMPATIBLE SKLEARN
# ============================================================

class CodeImputerWithFlag(BaseEstimator, TransformerMixin):
    """
    Imputa códigos numéricos con -1 + agrega flag *_WAS_NULL.
    Totalmente compatible con ColumnTransformer.
    """

    def __init__(self, fill_value=-1):
        self.fill_value = fill_value

    def fit(self, X, y=None):
        self.columns_ = X.columns.tolist()   # Muy importante
        return self

    def transform(self, X):
        X = X.copy()
        out = {}

        for col in self.columns_:
            out[f"{col}_WAS_NULL"] = X[col].isna().astype(int)
            out[col] = X[col].fillna(self.fill_value).astype(int)

        return pd.DataFrame(out)

    def get_feature_names_out(self, input_features=None):
        names = []
        for col in self.columns_:
            names.append(f"{col}_WAS_NULL")
            names.append(col)
        return np.array(names)


# ============================================================
# 🧩 3️⃣ build_preprocessing_pipeline
# ============================================================

def build_preprocessing_pipeline():

    NUMERIC_FEATS = [
        'PERSONAL_ASSETS_VALUE',
        'QUANT_BANKING_ACCOUNTS', 'QUANT_SPECIAL_BANKING_ACCOUNTS',
        'AGE', 'MONTHS_IN_RESIDENCE', 'MONTHS_IN_THE_JOB'
    ]

    CODE_FEATS = [
        'PROFESSION_CODE', 'OCCUPATION_TYPE', 'MATE_PROFESSION_CODE',
        'MATE_EDUCATION_LEVEL', 'RESIDENCE_TYPE',
        'RESIDENCIAL_PHONE_AREA_CODE', 'PROFESSIONAL_PHONE_AREA_CODE',
        'RESIDENCIAL_ZIP_3', 'PROFESSIONAL_ZIP_3'
    ]

    OHE_FEATS = [
        'SEX', 'PRODUCT', 'NACIONALITY', 'MARITAL_STATUS',
        'APPLICATION_SUBMISSION_TYPE', 'POSTAL_ADDRESS_TYPE',
        'RESIDENCIAL_STATE', 'PROFESSIONAL_STATE', 'STATE_OF_BIRTH',
        'AGE_GROUP'
    ]

    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    code_pipe = CodeImputerWithFlag(fill_value=-1)

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric_pipe', numeric_pipe, NUMERIC_FEATS),
            ('categorical_pipe', categorical_pipe, OHE_FEATS),
            ('code_pipe', code_pipe, CODE_FEATS),
        ],
        remainder='passthrough'
    )

    return Pipeline([
        ('base_cleaner', BaseCleaner()),
        ('preprocessor', preprocessor)
    ])
