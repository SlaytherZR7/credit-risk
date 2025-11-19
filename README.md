# ML Credit Risk Analysis

> Predicción de riesgo crediticio con pipelines reproducibles, importancia de variables y flujo de scoring/visualización. Incluye API/Frontend opcional para demo.

## 📦 Componentes principales

- `ml_creditrisk/` (paquete Python):
    - `feature_grouping.py`: utilidades para agrupar variables y auditar Missing%.
    - `preprocessing.py`: ColumnTransformer desde grupos (imputación, winsor, OHE, target-encoding) y discretizador robusto por cuantiles.
    - `importance.py`: entrenamiento XGBoost + agregado de importancias a variables originales, filtrado por umbral y plotting.
    - `models.py`: modelos base (RF, XGBoost, LightGBM, CatBoost) + evaluador y pipeline “GB leaves → OneHot → LR”.
- `notebooks/02_Feature_Engineering_Modelado.ipynb`: orquesta el flujo E2E (carga → grupos → preprocesamiento → importancia → modelos → predicciones → gráficos).
- `api/` y `frontend/`: demo opcional con FastAPI/Streamlit (no requerida para el notebook).

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

- Ubicación: `data/raw/`
- Archivos esperados:
    - `PAKDD2010_VariablesList.XLS` (nombres de columnas)
    - `PAKDD2010_Modeling_Data.txt` (modelado)
    - `PAKDD2010_Prediction_Data.txt` (scoring)

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

## 🤖 Modelos incluidos

- Random Forest (scikit-learn)
- XGBoost (xgboost)
- LightGBM (lightgbm) – opcional si instalado
- CatBoost (catboost) – opcional si instalado
- (Opcional) GB leaves → OneHot → LR (útil para calibración y capturar interacciones de árboles)

## 🧩 Diseño del preprocesamiento

- Numéricas reales: imputación (−1) → winsor (cuantiles) → robust scaler
- Numéricas con prioridad (AGE, MONTHS_IN_THE_JOB): discretización por cuantiles (robusta)
- Categóricas baja cardinalidad y binarias: OneHotEncoder
- Categóricas alta cardinalidad: TargetEncoder (smoothing=0.3)
- Todas las decisiones dependen de `df_groups_final` como “single source of truth”.

## Importancia y filtrado

- Importancias por feature output se agregan a variables raw originales.
- Se construye `preprocessor_filtered` con variables ≥ umbral.
- Tabla de variables eliminadas incluida para auditoría.

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

## Guardar y usar el preprocesador (.joblib)

En la última sección del notebook `02_Feature_Engineering_Modelado.ipynb` se incluye una celda para entrenar el preprocesador activo y guardarlo como artefacto reutilizable.

- Qué guarda:
    - `models/preprocessor_active_<timestamp>.joblib`: el `ColumnTransformer`/`Pipeline` final ajustado sobre el set de entrenamiento.
    - `models/preprocessor_active_<timestamp>.json`: metadatos (umbral de importancia, tamaños, fecha, hash de columnas, etc.).

- Cómo cargarlo y usarlo en otro script o sesión:
  
    Ejemplo mínimo en Python:
  
    1) Cargar el artefacto
    2) Aplicar `transform` sobre un DataFrame con el mismo esquema de columnas que el de entrenamiento

    Notas:
    - El artefacto espera las mismas columnas “raw” de entrada que se usaron en el entrenamiento (mismo `df_groups_final`).
    - Si cambian los grupos o el umbral de importancia, se debe volver a entrenar y guardar un nuevo artefacto.

- Versionado y .gitignore:
    - Por defecto, `models/*.joblib` y `models/*.json` están ignorados en `.gitignore` para evitar subir artefactos pesados.
    - Si necesitas versionar un artefacto concreto, puedes:
        - Forzar el agregado con `git add -f models/preprocessor_active_YYYYMMDD_HHMMSS.joblib` y su `.json`, o
        - Quitar/ajustar la regla de `.gitignore` para `models/*`.

Sugerencia: mantén un naming consistente y documenta en el metadato el dataset y los flags utilizados (por ejemplo, `IMPORTANCE_THRESHOLD`).

## 📄 Licencia

MIT (ver archivo LICENSE).
