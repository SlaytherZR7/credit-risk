from pydantic import BaseModel
from typing import Optional, List


class PredictionFeatures(BaseModel):
    NACIONALITY: Optional[int] = None
    OTHER_INCOMES: Optional[float] = None
    FLAG_MASTERCARD: Optional[int] = None
    PERSONAL_ASSETS_VALUE: Optional[float] = None
    QUANT_DEPENDANTS: Optional[int] = None
    MONTHS_IN_RESIDENCE: Optional[float] = None
    PERSONAL_MONTHLY_INCOME: Optional[float] = None
    PROFESSION_CODE: Optional[float] = None
    MATE_PROFESSION_CODE: Optional[float] = None
    AGE: Optional[int] = None
    FLAG_EMAIL: Optional[int] = None
    FLAG_VISA: Optional[int] = None
    QUANT_CARS: Optional[int] = None
    PAYMENT_DAY: Optional[int] = None
    MARITAL_STATUS: Optional[int] = None
    RESIDENCE_TYPE: Optional[float] = None
    QUANT_BANKING_ACCOUNTS: Optional[int] = None
    QUANT_SPECIAL_BANKING_ACCOUNTS: Optional[int] = None
    OCCUPATION_TYPE: Optional[float] = None
    MATE_EDUCATION_LEVEL: Optional[float] = None
    PRODUCT: Optional[int] = None

    APPLICATION_SUBMISSION_TYPE: Optional[str] = None
    SEX: Optional[str] = None
    FLAG_RESIDENCIAL_PHONE: Optional[str] = None
    COMPANY: Optional[str] = None
    FLAG_PROFESSIONAL_PHONE: Optional[str] = None
    STATE_OF_BIRTH: Optional[str] = None
    CITY_OF_BIRTH: Optional[str] = None
    RESIDENCIAL_STATE: Optional[str] = None
    RESIDENCIAL_CITY: Optional[str] = None
    RESIDENCIAL_BOROUGH: Optional[str] = None
    RESIDENCIAL_PHONE_AREA_CODE: Optional[str] = None
    PROFESSIONAL_STATE: Optional[str] = None
    PROFESSIONAL_CITY: Optional[str] = None
    PROFESSIONAL_BOROUGH: Optional[str] = None
    PROFESSIONAL_PHONE_AREA_CODE: Optional[str] = None
    RESIDENCIAL_ZIP_3: Optional[str] = None
    PROFESSIONAL_ZIP_3: Optional[str] = None


class PredictionRequest(BaseModel):
    features: PredictionFeatures


class BatchPredictionRequest(BaseModel):
    features: List[PredictionFeatures]
