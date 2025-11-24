# ML Credit Risk Analysis

> Predicción de riesgo crediticio con pipelines reproducibles, importancia de variables y flujo de scoring/visualización. Incluye API/Frontend opcional para demo.

## 📦 Componentes principales

Este proyecto sigue una arquitectura modular inspirada en buenas prácticas de MLOps, separando claramente:

- pipelines de entrenamiento,
- lógica de preprocesamiento,
- API de inferencia,
- frontend de demo,
- artefactos del modelo,
- notebooks exploratorios.

A continuación se detallan los componentes clave del repositorio.

---

### 🧠 `src/` — Lógica principal del proyecto

#### 🔹 `src/features/`
- `build_features.py`  
  Implementa el pipeline de preprocesamiento completo:
  - `BaseCleaner` (limpieza + feature engineering)
  - `CodeImputerWithFlag` (imputación robusta de códigos)
  - `build_preprocessing_pipeline()` (ColumnTransformer final)

#### 🔹 `src/pipelines/`
Incluye los **pipelines reproducibles de entrenamiento**:

- `data_preparation_pipeline.py`  
  Limpieza inicial y generación de datasets en `data/interim/` y `data/processed/`.

- `train_preprocessing.py`  
  Entrena el preprocesador completo y guarda el artefacto:  
  `model_service/artifacts/preprocessing_pipeline.joblib`

- `train_model_stack.py`  
  Entrena el modelo de stacking (XGBoost + LightGBM + meta-XGB)  
  y genera:  
  - `model_stack_prod.pkl`  
  - `model_metadata.json`

#### 🔹 `src/utils/`
- `split.py`  
  Funciones auxiliares para dividir dataset en train/test.

---

### 🤖 `model_service/` — Servicio de inferencia (API + worker)

Contiene todo lo necesario para correr el modelo en producción dentro de Docker.

#### 🔹 `model_service/app/`
- `main.py`  
  Servicio FastAPI principal: carga modelo + preprocesador.
- `worker.py`  
  Worker RQ para tareas en background (inferencias asincrónicas).

##### `model_service/app/model/`
- `pipeline.py`  
  Inicialización del modelo y preprocesador.
- `preprocess.py`  
  Utilidades para aplicar transformaciones y validaciones en inferencia.

##### `model_service/app/utils/`
- `schema.py`  
  Esquemas Pydantic para requests (`PredictionRequest`, batch, etc.)
- `utils.py`  
  Funciones auxiliares del servicio.

#### 🔹 `model_service/artifacts/`
Contiene los artefactos entrenados:

- `preprocessing_pipeline.joblib`
- `model_stack_prod.pkl`
- `model_metadata.json`

---

### 🧩 `api/` — API completa con autenticación (opcional)

Una API alternativa, con estructura clásica de FastAPI:

- `app/main.py` — punto de entrada
- `auth/` — login, JWT, dependencias, validadores
- `users/` — modelos y repositorio de usuarios
- `predictions/` — endpoints de scoring

> Esta API no es necesaria para el scoring del modelo,  
> pero se mantiene como módulo separado para demo con autenticación.

---

### 🎨 `frontend/` — Aplicación Streamlit para demo

- `streamlit_app.py` — interfaz principal
- `credit_form_interface.py` — mapeo de campos → payload para modelo
- `field_options.json` — catálogos (sex, estados, productos, etc.)
- `utils.py` — funciones del frontend

El frontend permite:
- cargar datos manualmente
- obtener un scoring del modelo
- visualizar métricas y simulaciones simples

---

### 📊 `notebooks/` — Exploración y prototipos

- `01_EDA.ipynb` — análisis exploratorio
- `03_Model_Visualization.ipynb` — análisis exploratorio

> El entrenamiento final NO depende del notebook,  
> sino de los scripts en `src/pipelines/`.

---

### 📁 `data/` — Dataset estructurado por etapas

- `raw/` — datos originales  
- `interim/` — datos intermedios limpios  
- `processed/` — datasets finales para entrenamiento (X_train, y_train, etc.)

Más detalle en la sección **🗂️ Datos**.


## 🧰 Requisitos y entorno

- Python 3.10 recomendado (Windows soportado)
- Instalar dependencias:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Notas:
- Para leer el archivo .XLS del dataset se fija xlrd==1.2.0 (las versiones ≥2.0 ya no soportan .xls).
- LightGBM y CatBoost están incluidos en requirements y se usan si están instalados.

## 🗂️ Datos

El proyecto utiliza una estructura clara para la gestión del dataset **PAKDD 2010**, separando datos **raw**, **interim** y **processed** para mantener un flujo limpio, ordenado y reproducible.

---

### 📁 Ubicación principal

Los datos se encuentran dentro de la carpeta:

---

### 📌 `data/raw/` — Datos originales (sin modificar)

Archivos requeridos:

- `PAKDD2010_VariablesList.XLS` — Diccionario de variables (nombres y descripciones)
- `PAKDD2010_Modeling_Data.txt` — Dataset para entrenamiento
- `PAKDD2010_Prediction_Data.txt` — Dataset para scoring/predicción

Archivo adicional utilizado por el frontend:

- `cities.csv` — Catálogo opcional para autocompletado de ciudades

---

### 📁 `data/interim/` — Datos intermedios

Archivos generados durante la etapa de limpieza inicial:

- `train_clean_headers.parquet` — Versión del dataset con encabezados corregidos y estructura estandarizada

---

### 📁 `data/processed/` — Datos finales para modelado

Archivos generados automáticamente por los pipelines de preprocesamiento:

- `X_train.parquet`
- `X_test.parquet`
- `y_train.parquet`
- `y_test.parquet`

Estos archivos representan los datasets listos para entrenamiento y evaluación de modelos.


## 🧪 Uso del notebook principal

1) Abrir `notebooks/02_Feature_Engineering_Modelado.ipynb` y ejecutar en orden:
     - Celda 1: carga de datos.
     - Celda 2: agrupación de variables, exclusiones y DataFrame FINAL (auditable).
     - Celda 3: construcción del preprocesador y resumen de columnas generadas.
     - Celda 4: importancia con XGBoost; umbral configurable; crea `preprocessor_filtered` con variables ≥ umbral (por defecto 0.02 en el cuaderno; se puede ajustar).
     - Celda 5: entrenamiento y evaluación de modelos activos (RF, XGBoost, LightGBM, CatBoost).
     - Celda 6: predicciones sobre `Prediction_Data.txt` y columnas `score_*` en `df_pred`.
     - Celda 7: histogramas de scores por modelo.

2) Búsqueda de hiperparámetros (RandomizedSearchCV):
     - La celda de HPO incluye un flag `HPO_ENABLED = False` para evitar ejecuciones largas. Cambiar a `True` para activar.
     - Los mejores pipelines quedan en `tuned_models`.
     - En la celda de predicciones, `USE_TUNED_MODELS = False` por defecto. Cambiar a `True` para usar `tuned_models` si existen.

## 🔧 Pipelines de Entrenamiento (MLOps)

Además del flujo interactivo en notebooks, este proyecto incluye pipelines reproducibles en src/pipelines/ que permiten entrenar el modelo de manera estandarizada, sin depender del notebook.

Estos scripts orquestan el flujo completo:

1. Preparación de Datos

python src/pipelines/data_preparation_pipeline.py

Limpia y organiza los datos raw, generando los datasets listos para preprocesamiento.

2. Entrenamiento del Preprocesador

python src/pipelines/train_preprocessing.py

Entrena el ColumnTransformer final y guarda el artefacto:
model_service/artifacts/preprocessing_pipeline.joblib

3. Entrenamiento del Modelo (Stacking)

python src/pipelines/train_model_stack.py

Entrena el modelo de producción y genera:

model_service/artifacts/model_stack_prod.pkl
model_service/artifacts/model_metadata.json

Estos artefactos son cargados automáticamente por el servicio FastAPI al iniciar, permitiendo usar el modelo entrenado sin depender del notebook.

## 🤖 Modelos incluidos

- Random Forest (scikit-learn)
- XGBoost (xgboost)
- LightGBM (lightgbm) – opcional si instalado
- CatBoost (catboost) – opcional si instalado
- (Opcional) GB leaves → OneHot → LR (útil para calibración y capturar interacciones de árboles)

## 🧩 Diseño del preprocesamiento


El proyecto utiliza un pipeline de preprocesamiento **100% reproducible y compatible con scikit-learn**, construido mediante:

- **`BaseCleaner`** → limpieza avanzada + feature engineering
- **`CodeImputerWithFlag`** → imputación robusta para códigos numéricos con flags
- **`ColumnTransformer`** → preprocesamiento paralelo por tipo de variable
- **`build_preprocessing_pipeline()`** → ensamblado final listo para entrenamiento e inferencia

---

### 🔹 1. Limpieza y Feature Engineering — `BaseCleaner`

`BaseCleaner` aplica transformaciones consistentes sobre los datos raw:

- Conversión de errores de Excel (`#N/A`, `#DIV/0!`, etc.) a `NaN`
- Normalización de estados, códigos y columnas categóricas problemáticas
- Generación de nuevas features:
  - `N_CARDS` (conteo total de tarjetas)
  - `TOTAL_INCOME`, `INCOME_PER_DEPENDANT`, `LOG_TOTAL_INCOME`
  - `HAS_CARDS`
  - `WORKS_SAME_STATE`
  - Binning de edad → `AGE_GROUP`
- Corrección de outliers específicos (`QUANT_DEPENDANTS > 15`)
- Dropeo de columnas ruidosas/irrelevantes (IDs, boroughs, flags redundantes, etc.)
- Conversión de Y/N → 1/0 en variables binarias

> Este paso concentra toda la ingeniería de features previa al ColumnTransformer.

---

### 🔹 2. Imputación especializada de códigos — `CodeImputerWithFlag`

Las columnas de códigos numéricos reciben un tratamiento especial:

- Imputación con un valor fijo (`-1`)
- Creación automática de un flag `<col>_WAS_NULL`
- Salida 100% numérica y consistente
- Compatible con scikit-learn y modelado basado en árboles

Beneficios:
- Preserva información sobre valores faltantes  
- Mantiene compatibilidad con modelos tree-based  
- Mejora estabilidad e interpretabilidad  

---

### 🔹 3. ColumnTransformer — Preprocesamiento unificado

Las columnas se agrupan en tres bloques:

#### **➤ NUMERIC_FEATS**
- Imputación: mediana  
- Escalado: `StandardScaler`

#### **➤ OHE_FEATS**
- Imputación: moda  
- Codificación: `OneHotEncoder(handle_unknown='ignore')`

#### **➤ CODE_FEATS**
- Imputación + flags: `CodeImputerWithFlag`

El resultado es un preprocesamiento robusto, interpretable y listo para producción.

---

### 🔹 4. Pipeline final

El pipeline completo se arma así:

BaseCleaner
↓
ColumnTransformer
(numeric_pipe + categorical_pipe + code_pipe)
↓
Dataset final listo para modelado

Este pipeline se serializa como artefacto para inferencia:

- `model_service/artifacts/preprocessing_pipeline.joblib`

---

### 🧠 Resumen

A diferencia del preprocesamiento tradicional (winsor, target encoding, cuantiles), este proyecto implementa un pipeline propio:

- Limpieza manual detallada  
- Feature engineering guiado por lógica de negocio  
- ColumnTransformer transparente  
- Imputación con flags para códigos numéricos  
- Total compatibilidad con scikit-learn y MLOps

El resultado es un preprocesamiento **robusto, reproducible y listo para producción**.

## 📊 Métricas e información del modelo (model_metadata.json)

Durante el entrenamiento del modelo stacking, el proyecto genera un archivo:

- `model_service/artifacts/model_metadata.json`


Este archivo contiene **métricas clave del modelo final**, calculadas sobre el set de test:

- **AUC (`auc`)**  
  Medida general de discriminación del modelo.

- **Mejor umbral (`best_threshold`)**  
  Obtenido maximizando el F1-score mediante la curva Precision-Recall.

- **F1-score óptimo (`best_f1`)**

- **Flag de calibración (`calibrated`)**  
  Indica si el modelo final usa calibración de probabilidades vía Isotonic Regression.

Ejemplo real generado por el pipeline:

```json
{
    "auc": 0.6476,
    "best_threshold": 0.2466,
    "best_f1": 0.4550,
    "calibrated": true
}

Nota:
A diferencia de otros enfoques, este proyecto no aplica filtrado de variables por importancia.
El archivo model_metadata.json se utiliza para auditoría del modelo, selección del umbral óptimo y trazabilidad del entrenamiento.

## ▶️ API / Frontend (opcional)

Para demo rápida (cuando quieras mostrar un servicio):

```powershell
uvicorn api.main:app --reload --port 8000
# Docs: http://localhost:8000/docs

streamlit run frontend/streamlit_app.py --server.port 8501
# App: http://localhost:8501
```

Autenticación (demo):
- Usuarios válidos: `admin/admin123` y `analyst/analyst456`.
- Si `USE_BACKEND=false`, el login se valida localmente en el frontend (modo simulado).
- Si `USE_BACKEND=true`, el frontend llama a `POST /login` en la API y guarda un `access_token` de sesión (sin autorización estricta para esta demo).

Esquema de entrada del modelo (`POST /predict`):
- Campos requeridos del JSON:
    - `income` (float)
    - `age` (int)
    - `credit_amount` (float)
    - `employment_length` (int, en años)
    - `debt_ratio` (float, 0–1)

La UI de Streamlit mapea automáticamente el formulario de “Credit Application Form (Manual Input)” a estos 5 campos:
- `income` = `PERSONAL_MONTHLY_INCOME` + `OTHER_INCOMES`
- `age` = `AGE`
- `employment_length` = floor(`MONTHS_IN_THE_JOB` / 12)
- `credit_amount` ≈ 20% de `PERSONAL_ASSETS_VALUE` (si falta, usa 10000)
- `debt_ratio` ≈ `credit_amount` / (`income`*12 + `PERSONAL_ASSETS_VALUE`) recortado a [0, 0.9]

## 🐳 Ejecutar con Docker Compose

Requisitos: Docker Desktop y Docker Compose.

1) Construir y levantar servicios (API + Frontend):
```powershell
docker compose up --build
```

2) URLs:
- Frontend: http://localhost:8501
- FastAPI: http://localhost:8000
- Docs API: http://localhost:8000/docs

Notas:
- El frontend se conecta a la API vía `API_BASE_URL` (definido en docker-compose como `http://api:8000`).
- Los volúmenes montan `./models` y `./data` dentro de los contenedores (`/app/models`, `/app/data`).
- Healthchecks validan que cada servicio esté listo antes de exponerlo.

Archivos de datos auxiliares (opcional):
- La UI puede cargar un catálogo de ciudades de Brasil desde `data/raw/cities.csv`. Rutas soportadas automáticamente:
    - `./data/raw/cities.csv` (host)
    - `/app/data/raw/cities.csv` (contenedor)
    - o define `CITIES_CSV_PATH` con la ruta al CSV
- Si el archivo no existe, la UI hace fallback: Estados por sigla fija y ciudades como texto libre (no falla).

### Variables de entorno útiles
- API (servicio `api`):
    - `MODEL_PATH`: ruta al artefacto del modelo o pipeline (por ejemplo, `/app/models/pipeline.joblib`).
    - `PREPROCESSOR_PATH`: ruta al preprocesador si el modelo no lo incluye.
    - `API_HOST`, `API_PORT`, `API_DEBUG` (ya preconfigurados para Docker).
- Frontend (servicio `frontend`):
    - `API_BASE_URL`: URL de la API dentro de la red de Docker (`http://api:8000`).
    - `USE_BACKEND`: `true` para consultar la API real.
    - `CITIES_CSV_PATH`: ruta al CSV de ciudades (opcional; si no existe, hay fallback seguro).

Puedes añadir estas variables bajo `environment:` en `docker-compose.yml` o usar un archivo `.env`.

### Endpoints principales de la API
- `POST /login` → autenticación demo (devuelve `access_token` si usuario/clave válidos).
- `POST /predict` → scoring individual con el esquema de 5 campos indicado arriba.
- `POST /predict/batch` → scoring por lote (`{"profiles": [ ... ]}`).
- `POST /simulate` → simulación de decisiones; parámetros:
    - `profiles`: lista de perfiles con al menos `credit_amount` si quieres métricas monetarias
    - `decision_threshold` (float, default 0.5): aprueba cuando `risk_score <= threshold`
    - `profit_margin` (float, default 0.05)
- `GET /model/info` y `GET /health` → info básica y healthcheck.

### Cambiar al modelo real
1. Copia tu artefacto entrenado a `./models` (por ejemplo `./models/pipeline_real.joblib`).
2. Edita `docker-compose.yml` → `MODEL_PATH=/app/models/pipeline_real.joblib` (y opcional `PREPROCESSOR_PATH` si usas artefactos separados).
3. Reconstruye y levanta:
     ```powershell
     docker compose up -d --build
     ```
4. Valida `/health`, `/model/info` y una predicción simple.

## 🛠️ Troubleshooting (solución de problemas)

Estos son los errores más comunes y cómo resolverlos rápidamente.

1) Error 422 Unprocessable Entity en `/predict`
- Síntomas: la app muestra “API Error: 422 …” o el detalle pide campos faltantes.
- Causa: el payload no cumple el esquema del endpoint (faltan campos o nombres distintos).
- Solución: asegúrate de enviar exactamente estos 5 campos: `income` (float), `age` (int), `credit_amount` (float), `employment_length` (int), `debt_ratio` (float). La UI ya lo mapea automáticamente; si pruebas con herramientas externas, respeta el esquema.

2) FileNotFoundError con `cities.csv`
- Síntomas: traza en `frontend/credit_form_interface.py` al leer `cities.csv`.
- Causa: archivo ausente o ruta local no válida en el contenedor.
- Solución: coloca el archivo en `data/raw/cities.csv` (se monta en `/app/data/raw/cities.csv`) o define `CITIES_CSV_PATH`. Si no existe el archivo, la UI hace fallback a siglas de estados y ciudades como texto (no se rompe).

3) “Invalid credentials” al hacer login
- Síntomas: el login falla siempre.
- Causas: (a) `USE_BACKEND=true` pero el endpoint `/login` no está en la imagen en ejecución (falta rebuild); (b) credenciales distintas a las de demo; (c) API no alcanzable.
- Solución: rebuild de API/Frontend, usar usuarios de demo `admin/admin123` o `analyst/analyst456`, verificar `/openapi.json` incluye `/login` y que `API_BASE_URL` apunte a la API (en Compose: `http://api:8000`).

4) Modelo no cargado / `/model/info` falla
- Síntomas: `/health` indica `model_loaded: false` o el endpoint de predicción falla.
- Causa: `MODEL_PATH` o `PREPROCESSOR_PATH` apuntan a rutas inexistentes.
- Solución: copia el artefacto real a `./models`, actualiza `MODEL_PATH` en `docker-compose.yml` (por ejemplo, `/app/models/pipeline_real.joblib`) y reconstruye.

5) Simulación con `approved_applications=0` o ROI negativa
- Causa: con el pipeline “dummy” los `risk_score` ≈ 0.5; si el umbral es muy estricto, no hay aprobados; además con `profit_margin` 0.05 y `risk_score` 0.5, la pérdida esperada puede superar la ganancia.
- Solución: ajusta el slider `decision_threshold` (la regla es `score <= threshold`) y/o `profit_margin`, o usa tu modelo real para scores más informativos.

6) El Frontend no conecta con la API
- Síntomas: “API Error … conexión” o métricas que no cargan.
- Causa: `API_BASE_URL` incorrecto. Dentro de Docker Compose debe ser `http://api:8000`; en local, `http://localhost:8000`.
- Solución: verifica variables de entorno y reconstruye si cambiaste el compose.

7) Puertos en uso (8000/8501)
- Síntomas: Docker no puede publicar puertos.
- Solución: cierra procesos que usan esos puertos o cambia el mapeo en `docker-compose.yml`.

8) Contenedores “unhealthy”
- Causa: healthcheck falla por API caída o Frontend sin levantar.
- Solución: revisa logs (`docker compose logs -f api` / `frontend`), valida rutas de modelo/datos, reintenta el build.

9) Batch `/predict/batch` devuelve error
- Causa: formato incorrecto.
- Solución: envía `{ "profiles": [ { five fields }, ... ] }` con el mismo esquema de `/predict` por perfil.

10) Lectura de `.xls` falla en el notebook
- Causa: `xlrd>=2.0` no soporta `.xls`.
- Solución: usa `xlrd==1.2.0` (ya está en `requirements.txt`).

## ✅ Checklist de reproducibilidad

- [x] requirements.txt actualizado (incluye xlrd==1.2.0, sklearn, xgboost, lightgbm, catboost, scipy, etc.)
- [x] Paquete `ml_creditrisk` con docstrings y funciones reutilizables
- [x] Notebook principal orquestando el flujo E2E
- [x] Flags para activar/desactivar HPO y usar modelos tuneados

## 📄 Licencia

MIT (ver archivo LICENSE).
