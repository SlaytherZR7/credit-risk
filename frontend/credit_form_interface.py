import streamlit as st
import pandas as pd
import os
import json
from typing import Optional

UF_MAP = {
    12: "AC", 27: "AL", 13: "AM", 16: "AP", 29: "BA", 23: "CE",
    53: "DF", 32: "ES", 52: "GO", 21: "MA", 31: "MG", 50: "MS",
    51: "MT", 15: "PA", 25: "PB", 26: "PE", 22: "PI", 41: "PR",
    33: "RJ", 24: "RN", 11: "RO", 14: "RR", 43: "RS", 42: "SC",
    28: "SE", 35: "SP", 17: "TO"
}

UF_SIGLAS = sorted(list(set(UF_MAP.values())))


@st.cache_data
def load_brazil_cities() -> Optional[pd.DataFrame]:
    """Load Brazilian cities from a CSV if available.

    Tries multiple local paths including an env var override (CITIES_CSV_PATH).
    Returns None when not found so the UI can fallback to text inputs.
    Expected CSV columns: codigo_uf, nome
    """
    env_path = os.getenv("CITIES_CSV_PATH")
    candidates = [p for p in [
        env_path,
        os.path.join(os.getcwd(), "data", "raw", "cities.csv"),
        os.path.join(os.path.dirname(__file__), "..", "data", "raw", "cities.csv"),
        "/app/data/raw/cities.csv",
        "./data/raw/cities.csv",
    ] if p]

    for path in candidates:
        try:
            if os.path.exists(path):
                df = pd.read_csv(path, encoding="utf-8")
                if "codigo_uf" in df.columns and "nome" in df.columns:
                    df["STATE"] = df["codigo_uf"].map(UF_MAP)
                    df = df.rename(columns={"nome": "CITY"})
                    df = df.dropna(subset=["STATE"])  # safety
                    return df[["STATE", "CITY"]]
        except Exception:
            continue

    return None


custom_labels = {
    "SEX": "Gender",
    "MARITAL_STATUS": "Marital Status",
    "QUANT_DEPENDANTS": "Number of Dependents",
    "EDUCATION_LEVEL": "Education Level",
    "MATE_EDUCATION_LEVEL": "Mate's Education Level",
    "STATE_OF_BIRTH": "State of Birth",
    "CITY_OF_BIRTH": "City of Birth",
    "NACIONALITY": "Nationality",
    "AGE": "Age",
    "RESIDENCIAL_STATE": "State of Residence",
    "RESIDENCIAL_CITY": "City of Residence",
    "RESIDENCIAL_BOROUGH": "Borough of Residence",
    "RESIDENCE_TYPE": "Type of Residence",
    "MONTHS_IN_RESIDENCE": "Months at Current Residence",
    "FLAG_RESIDENCIAL_PHONE": "Has Home Phone?",
    "RESIDENCIAL_PHONE_AREA_CODE": "Home Phone Area Code",
    "FLAG_MOBILE_PHONE": "Has Mobile Phone?",
    "FLAG_EMAIL": "Has Email Address?",
    "RESIDENCIAL_ZIP_3": "Residential ZIP Code",
    "COMPANY": "Company Provided",
    "PROFESSIONAL_STATE": "Work State",
    "PROFESSIONAL_CITY": "Work City",
    #"PROFESSIONAL_BOROUGH": "Work Borough",
    "FLAG_PROFESSIONAL_PHONE": "Has Work Phone?",
    "PROFESSIONAL_PHONE_AREA_CODE": "Work Phone Area Code",
    "MONTHS_IN_THE_JOB": "Months in Current Job",
    "PROFESSION_CODE": "Profession Code",
    "OCCUPATION_TYPE": "Occupation Type",
    "MATE_PROFESSION_CODE": "Mate's Profession Code",
    "PROFESSIONAL_ZIP_3": "Work ZIP Code",
    "PERSONAL_MONTHLY_INCOME": "Monthly Income (R$)",
    "OTHER_INCOMES": "Other Monthly Income (R$)",
    "QUANT_BANKING_ACCOUNTS": "Number of Bank Accounts",
    "QUANT_SPECIAL_BANKING_ACCOUNTS": "Number of Special Bank Accounts",
    "PERSONAL_ASSETS_VALUE": "Value of Personal Assets (R$)",
    "QUANT_CARS": "Number of Cars Owned",
    "FLAG_VISA": "Has VISA Card?",
    "FLAG_MASTERCARD": "Has MasterCard?",
    "FLAG_DINERS": "Has Diners Club Card?",
    "FLAG_AMERICAN_EXPRESS": "Has American Express Card?",
    "FLAG_OTHER_CARDS": "Has Other Credit Cards?",
    "QUANT_ADDITIONAL_CARDS": "Number of Additional Cards Requested",
    "FLAG_HOME_ADDRESS_DOCUMENT": "Provided Proof of Home Address?",
    "FLAG_RG": "Provided National ID (RG)?",
    "FLAG_CPF": "Provided Taxpayer ID (CPF)?",
    "FLAG_INCOME_PROOF": "Provided Proof of Income?",
    "PAYMENT_DAY": "Preferred Payment Day",
    "APPLICATION_SUBMISSION_TYPE": "Submission Method",
    "POSTAL_ADDRESS_TYPE": "Postal Address Type",
    "PRODUCT": "Financial Product Type",
    "FLAG_ACSP_RECORD": "Has Credit Delinquency Record?"
}

try:
    base_dir = os.path.dirname(__file__)
except NameError:
    base_dir = os.getcwd()

json_path = os.path.join(base_dir, 'field_options.json')
with open(json_path, 'r', encoding='utf-8') as f:
    field_options = json.load(f)



def create_credit_application_form():
    st.header("📝 Credit Application Form")

    uploaded_file = st.file_uploader(
        "Upload CSV or TXT file to auto-fill fields",
        type=['csv', 'txt']
    )

    uploaded_data = {}
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file, delimiter='\t')

            df.columns = [col.strip().upper() for col in df.columns]
            uploaded_data = df.iloc[0].to_dict()
            st.success("✅ File loaded successfully! Fields will be pre-filled.")
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")

    # --- FORM STRUCTURE ---
    sections = {
        'Personal Information': ["AGE","SEX", "MARITAL_STATUS", "QUANT_DEPENDANTS", "EDUCATION_LEVEL",
                                 "STATE_OF_BIRTH", "CITY_OF_BIRTH", "NACIONALITY"],
        'Residential Information': ["RESIDENCIAL_STATE", "RESIDENCIAL_CITY", "RESIDENCIAL_BOROUGH",
                                    "RESIDENCE_TYPE", "MONTHS_IN_RESIDENCE", "FLAG_RESIDENCIAL_PHONE",
                                    "RESIDENCIAL_PHONE_AREA_CODE", "FLAG_MOBILE_PHONE", "FLAG_EMAIL",
                                    "RESIDENCIAL_ZIP_3"],
        'Employment Information': ["COMPANY", "PROFESSIONAL_STATE", "PROFESSIONAL_CITY",
                                   "PROFESSIONAL_BOROUGH", "FLAG_PROFESSIONAL_PHONE",
                                   "PROFESSIONAL_PHONE_AREA_CODE", "MONTHS_IN_THE_JOB",
                                   "PROFESSION_CODE", "OCCUPATION_TYPE" ,"MATE_EDUCATION_LEVEL","MATE_PROFESSION_CODE",
                                   "PROFESSIONAL_ZIP_3"],
        'Financial Information': ["PERSONAL_MONTHLY_INCOME", "OTHER_INCOMES",
                                  "QUANT_BANKING_ACCOUNTS", "QUANT_SPECIAL_BANKING_ACCOUNTS",
                                  "PERSONAL_ASSETS_VALUE", "QUANT_CARS"],
        'Credit Card Information': ["FLAG_VISA", "FLAG_MASTERCARD", "FLAG_DINERS",
                                    "FLAG_AMERICAN_EXPRESS", "FLAG_OTHER_CARDS", "QUANT_ADDITIONAL_CARDS"],
        'Documentation': ["FLAG_HOME_ADDRESS_DOCUMENT", "FLAG_RG", "FLAG_CPF", "FLAG_INCOME_PROOF", "FLAG_ACSP_RECORD"],
        'Application Details': ["PAYMENT_DAY", "APPLICATION_SUBMISSION_TYPE", "POSTAL_ADDRESS_TYPE", "PRODUCT"],
    }

    tabs = st.tabs(list(sections.keys()))
    form_data = {}

    # --- FIELDS ---
    for i, (section, fields) in enumerate(sections.items()):
        with tabs[i]:
            col1, col2 = st.columns(2)
            for j, field in enumerate(fields):
                col = col1 if j % 2 == 0 else col2
                with col:
                    label = field.replace("_", " ").title()
                    default_val = uploaded_data.get(field, "")

                    # --- Render fields ---
                    form_data[field] = st.text_input(label, value=str(default_val))

                    # --- Regla de negocio: Edad ---
                    if field == "AGE":
                        try:
                            age = float(form_data.get("AGE", 0))
                            if age < 18 or age > 80:
                                st.markdown(
                                    """
                                    <div style='background-color:#ffebee; border-left:5px solid #f44336;
                                                padding:10px; border-radius:6px; color:#b71c1c;'>
                                        🚫 <strong>Request denied:</strong> age must be between 18 and 80 years.
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                st.stop()
                        except ValueError:
                            st.warning("⚠️ Invalid age. Please enter a valid number.")

                    # --- Regla de negocio: Campos vacíos ---
                    value = str(form_data.get(field, "")).strip()
                    if value in ["", "None", "NaN", "nan"]:
                        st.markdown(
                            f"""
                            <div style='background-color:#fff3cd; border-left:5px solid #ff9800;
                                        padding:8px; border-radius:6px; color:#795548; margin-top:4px;'>
                                ⚠️ <strong> Empty field:</strong> {label} has not been completed.
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

    return form_data



def create_credit_application_form_m(custom_labels, field_options):
    st.header("📝 Credit Application Form (Manual Input)")
    form_data = {}

    # 🧩 Form sections
    sections = {
        'Personal Information': ["SEX", "MARITAL_STATUS", "QUANT_DEPENDANTS", "EDUCATION_LEVEL",
                                 "STATE_OF_BIRTH", "CITY_OF_BIRTH", "NACIONALITY", "AGE"],
        'Residential Information': ["RESIDENCIAL_STATE", "RESIDENCIAL_CITY", "RESIDENCIAL_BOROUGH",
                                    "RESIDENCE_TYPE", "MONTHS_IN_RESIDENCE", "FLAG_RESIDENCIAL_PHONE",
                                    "RESIDENCIAL_PHONE_AREA_CODE", "FLAG_MOBILE_PHONE", "FLAG_EMAIL",
                                    "RESIDENCIAL_ZIP_3"],
        'Employment Information': ["COMPANY", "PROFESSIONAL_STATE", "PROFESSIONAL_CITY",
                                   "PROFESSIONAL_BOROUGH", "FLAG_PROFESSIONAL_PHONE",
                                   "PROFESSIONAL_PHONE_AREA_CODE", "MONTHS_IN_THE_JOB",
                                   "PROFESSION_CODE", "OCCUPATION_TYPE", "MATE_PROFESSION_CODE",
                                   "PROFESSIONAL_ZIP_3"],
        'Financial Information': ["PERSONAL_MONTHLY_INCOME", "OTHER_INCOMES",
                                  "QUANT_BANKING_ACCOUNTS", "QUANT_SPECIAL_BANKING_ACCOUNTS",
                                  "PERSONAL_ASSETS_VALUE", "QUANT_CARS"],
        'Credit Card Information': ["FLAG_VISA", "FLAG_MASTERCARD", "FLAG_DINERS",
                                    "FLAG_AMERICAN_EXPRESS", "FLAG_OTHER_CARDS", "QUANT_ADDITIONAL_CARDS"],
        'Documentation': ["FLAG_HOME_ADDRESS_DOCUMENT", "FLAG_RG", "FLAG_CPF", "FLAG_INCOME_PROOF", "FLAG_ACSP_RECORD"],
        'Application Details': ["PAYMENT_DAY", "APPLICATION_SUBMISSION_TYPE", "POSTAL_ADDRESS_TYPE", "PRODUCT"],
    }

    # 🧮 Numeric fields
    numeric_fields = [
        "AGE", "QUANT_DEPENDANTS", "MONTHS_IN_RESIDENCE", "RESIDENCIAL_PHONE_AREA_CODE",
        "MONTHS_IN_THE_JOB", "PROFESSIONAL_PHONE_AREA_CODE", "PERSONAL_MONTHLY_INCOME",
        "OTHER_INCOMES", "QUANT_BANKING_ACCOUNTS", "QUANT_SPECIAL_BANKING_ACCOUNTS",
        "PERSONAL_ASSETS_VALUE", "QUANT_CARS", "QUANT_ADDITIONAL_CARDS", "PAYMENT_DAY"
    ]

    # 🗂️ Tabs for sections
    tabs = st.tabs(list(sections.keys()))

    cities_df = load_brazil_cities()
    
    for i, (section, fields) in enumerate(sections.items()):
        with tabs[i]:
            col1, col2 = st.columns(2)
            for j, field in enumerate(fields):
                col = col1 if j % 2 == 0 else col2
                with col:
                    label = custom_labels.get(field, field.replace("_", " ").title())

                    # --- ① Special 3-digit fields (ZIP / DDD) ---
                    if field in ["RESIDENCIAL_PHONE_AREA_CODE", "PROFESSIONAL_PHONE_AREA_CODE",
                                "RESIDENCIAL_ZIP_3", "PROFESSIONAL_ZIP_3"]:
                        form_data[field] = st.text_input(label, key=field)
                        value = str(form_data.get(field, "")).strip()
                        if value:  # Only validate if not empty
                            if not value.isdigit():
                                st.markdown(
                                    f"<div style='color:#b58900; font-size:0.9em; margin-top:2px;'>⚠️ {label} must contain digits only.</div>",
                                    unsafe_allow_html=True
                                )
                            elif len(value) > 3:
                                st.markdown(
                                    f"<div style='color:#b71c1c; font-size:0.9em; margin-top:2px;'>🚫 {label} must have a maximum of 3 digits.</div>",
                                    unsafe_allow_html=True
                                )

                    # --- ② State dropdowns ---
                    elif field in ["STATE_OF_BIRTH", "RESIDENCIAL_STATE", "PROFESSIONAL_STATE"]:
                        if cities_df is not None:
                            state_siglas = sorted(cities_df["STATE"].unique())
                            form_data[field] = st.selectbox(label, state_siglas, key=field)
                        else:
                            form_data[field] = st.selectbox(label, UF_SIGLAS, key=field)

                    # --- ③ City dropdowns (dependent on selected state) ---
                    elif field in ["CITY_OF_BIRTH", "RESIDENCIAL_CITY", "PROFESSIONAL_CITY"]:
                        state_links = {
                            "CITY_OF_BIRTH": "STATE_OF_BIRTH",
                            "RESIDENCIAL_CITY": "RESIDENCIAL_STATE",
                            "PROFESSIONAL_CITY": "PROFESSIONAL_STATE",
                        }
                        state_field = state_links.get(field)
                        # Avoid passing None as a key to session_state.get and handle missing state_field gracefully
                        selected_state = st.session_state.get(state_field) if state_field is not None else None
                        if selected_state:
                            if cities_df is not None:
                                filtered_cities = sorted(
                                    cities_df[cities_df["STATE"] == selected_state]["CITY"].unique()
                                )
                                form_data[field] = st.selectbox(label, filtered_cities, key=field)
                            else:
                                form_data[field] = st.text_input(label, key=field)
                        else:
                            hint_field = state_field.replace('_', ' ').title() if state_field else "the corresponding state"
                            st.info(f"👆 Please select the corresponding state first ({hint_field}).")
                            form_data[field] = ""

                    # --- ④ Fields with predefined options (from field_options.json) ---
                    elif field in field_options:
                        options = field_options[field]
                        if isinstance(options, dict):
                            keys = list(options.keys())
                            labels = list(options.values())
                            selected_label = st.selectbox(label, labels, key=field)
                            form_data[field] = keys[labels.index(selected_label)]
                        else:
                            form_data[field] = st.selectbox(label, options, key=field)

                    # --- ⑤ Numeric fields ---
                    elif field in ["AGE", "QUANT_DEPENDANTS", "MONTHS_IN_RESIDENCE",
                                "MONTHS_IN_THE_JOB", "PERSONAL_MONTHLY_INCOME",
                                "OTHER_INCOMES", "QUANT_BANKING_ACCOUNTS",
                                "QUANT_SPECIAL_BANKING_ACCOUNTS", "PERSONAL_ASSETS_VALUE",
                                "QUANT_CARS", "QUANT_ADDITIONAL_CARDS", "PAYMENT_DAY"]:
                        form_data[field] = st.number_input(label, min_value=0, step=1, key=field)

                    # --- ⑥ Generic text fields ---
                    else:
                        form_data[field] = st.text_input(label, key=field)

                    # --- ⑦ Business rule: AGE validation ---
                    if field == "AGE":
                        age = form_data.get("AGE", 0)
                        if isinstance(age, (int, float)) and (age < 18 or age > 80):
                            st.error("🚫 Invalid age range. Age must be between 18 and 80 years.")
                            st.stop()

                    # --- ⑧ Empty field check ---
                    value = str(form_data.get(field, "")).strip()
                    if value in ["", "None", "NaN"]:
                        st.markdown(
                            f"<div style='background-color:#ffe6e6; color:#b30000; "
                            f"padding:4px; border-radius:4px; font-size:0.9em;'>⚠️ Field '{label}' is empty.</div>",
                            unsafe_allow_html=True
                        )


    # Resumen de campos faltantes
    missing = [f for f, v in form_data.items() if str(v).strip() in ["", "None", "NaN"]]
    st.markdown("---")
    if missing:
        st.warning(f"⚠️ {len(missing)} empty fields. All fields must be completed before proceeding.")
    else:
        st.success("✅ All fields have been completed")

    return form_data

