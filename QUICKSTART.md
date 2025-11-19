# 🚀 Guía de Inicio Rápido - Credit Risk Analysis

> ¿Solo quieres correr el flujo de modelado y scoring? Abre `notebooks/02_Feature_Engineering_Modelado.ipynb` y sigue las celdas en orden (agrupación → preprocesamiento → importancia → modelos → predicciones → histogramas). Flags útiles: `HPO_ENABLED=False` y `USE_TUNED_MODELS=False` por defecto para ejecuciones rápidas.

## 📋 Resumen del Sistema

Este proyecto combina **FastAPI** para el backend y **Streamlit** para la interfaz de usuario, creando un sistema completo de análisis de riesgo crediticio.

### 🏗️ Arquitectura del Sistema

```
┌─────────────────┐    HTTP/REST    ┌─────────────────┐
│                 │ ◄──────────────► │                 │
│   Streamlit UI  │                 │   FastAPI       │
│   (Frontend)    │                 │   (Backend)     │
│   Puerto: 8501  │                 │   Puerto: 8000  │
└─────────────────┘                 └─────────────────┘
                                            │
                                            ▼
                                    ┌─────────────────┐
                                    │  ML Models      │
                                    │  (Scikit-learn, │
                                    │   LightGBM,     │
                                    │   XGBoost, etc) │
                                    └─────────────────┘
```

## 🛠️ Instalación y Configuración

### 1. Clonar el repositorio
```bash
git clone <repository-url>
cd ML_CreditRisk
```

### 2. Crear entorno virtual (Python 3.10+)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Instalar el proyecto
```bash
pip install -e .
```

## 🚀 Iniciar el Sistema

### Opción A: Script Automático (Recomendado)

**Windows:**
```bash
start_system.bat
```

**Linux/Mac:**
```bash
chmod +x start_system.sh
./start_system.sh
```

### Opción B: Manual

**1. Iniciar FastAPI (Terminal 1):**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**2. Iniciar Streamlit (Terminal 2):**
```bash
cd frontend
streamlit run streamlit_app.py
```

## 🌐 URLs del Sistema

| Servicio | URL | Descripción |
|----------|-----|-------------|
| **Streamlit UI** | http://localhost:8501 | Interfaz principal del usuario |
| **FastAPI** | http://localhost:8000 | API REST del backend |
| **API Docs** | http://localhost:8000/docs | Documentación interactiva (Swagger) |
| **ReDoc** | http://localhost:8000/redoc | Documentación alternativa |
| **Health Check** | http://localhost:8000/health | Estado del sistema |

## 🎯 Funcionalidades Principales

### 1. 🔍 Análisis Individual
- Formulario interactivo para capturar datos del perfil crediticio
- Predicción en tiempo real del riesgo crediticio
- Visualización del risk score con gauge interactivo
- Recomendación automática (APROBAR/RECHAZAR/REVISAR)

### 2. 📊 Análisis en Lote
- Carga de archivos CSV con múltiples perfiles
- Procesamiento masivo de predicciones
- Generación de datos sintéticos para pruebas
- Visualizaciones de distribución de riesgo
- Exportación de resultados

### 3. 🎯 Simulación de Decisiones
- Simulación de impacto financiero
- Configuración de parámetros (umbral de decisión, margen de ganancia)
- Cálculo de ROI esperado
- Recomendaciones basadas en resultados

### 4. 📈 Dashboard de Métricas
- Información del modelo cargado
- Métricas del sistema en tiempo real
- Monitoreo de performance

## 📁 Estructura de Archivos Clave

```
├── api/
│   ├── main.py              # Aplicación FastAPI principal
│   └── models.py            # Modelos Pydantic para validación
├── frontend/
│   ├── streamlit_app.py     # Aplicación Streamlit principal
│   ├── utils.py             # Utilidades del frontend
│   └── .streamlit/
│       └── config.toml      # Configuración de Streamlit
├── src/
│   ├── models/
│   │   ├── train_model.py   # Entrenamiento de modelos
│   │   └── predict_model.py # Predicciones
│   └── features/
│       └── build_features.py # Construcción de features
├── start_system.bat         # Script de inicio (Windows)
└── start_system.sh          # Script de inicio (Linux/Mac)
```

## 🔧 Configuración Avanzada

### Variables de Entorno
Crea un archivo `.env` basado en `.env.example`:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=True

# Model Configuration
MODEL_PATH=models/
DEFAULT_MODEL=best_model.pkl

# Logging
LOG_LEVEL=INFO
```

### Configuración de Streamlit
Edita `frontend/.streamlit/config.toml` para personalizar la interfaz:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

## 🧪 Testing

### Ejecutar Tests de la API
```bash
# Instalar pytest si no está instalado
pip install pytest pytest-cov

# Ejecutar tests
pytest tests/ -v

# Con cobertura
pytest tests/ --cov=api --cov=src
```

### Test Manual de la API
```bash
# Health check
curl http://localhost:8000/health

# Predicción individual
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "income": 50000,
       "age": 35,
       "credit_amount": 15000,
       "employment_length": 5,
       "debt_ratio": 0.3
     }'
```

## 🐳 Deployment con Docker

### Construcción y Ejecución
```bash
cd deployment
docker-compose up --build
```

### Deployment en Producción
```bash
./deployment/deploy.sh production
```

## 📊 Flujo de Trabajo del Proyecto

### 1. Desarrollo del Modelo
```bash
# 1. Procesar datos
python src/data/make_dataset.py

# 2. Construir features
python src/features/build_features.py

# 3. Entrenar modelos
python src/models/train_model.py

# 4. Evaluar modelos
python src/models/predict_model.py
```

### 2. Desarrollo de la API
```bash
# Iniciar en modo desarrollo
uvicorn api.main:app --reload
```

### 3. Desarrollo del Frontend
```bash
# Iniciar Streamlit
streamlit run frontend/streamlit_app.py
```

## 🔍 Troubleshooting

### Problemas Comunes

**1. Error: "Modelo no encontrado"**
- Asegúrate de haber entrenado un modelo ejecutando `python src/models/train_model.py`
- Verifica que existe el archivo `models/best_model.pkl`

**2. Error de conexión Streamlit ↔ FastAPI**
- Verifica que FastAPI esté corriendo en puerto 8000
- Revisa el health check: `curl http://localhost:8000/health`

**3. Error de puertos ocupados**
```bash
# Verificar puertos en uso
netstat -ano | findstr :8000  # Windows
netstat -tulpn | grep :8000   # Linux

# Cambiar puertos si es necesario
uvicorn api.main:app --port 8001
streamlit run frontend/streamlit_app.py --server.port 8502
```

**4. Error de dependencias**
```bash
# Reinstalar dependencias
pip install --upgrade -r requirements.txt

# Limpiar cache de pip
pip cache purge
```

## 📚 Documentación Adicional

- **API Docs**: http://localhost:8000/docs (cuando FastAPI esté ejecutándose)
- **Streamlit Docs**: https://docs.streamlit.io/
- **FastAPI Docs**: https://fastapi.tiangolo.com/

## 💾 Guardar y usar el preprocesador (.joblib)

En la última celda del notebook `02_Feature_Engineering_Modelado.ipynb` puedes entrenar el preprocesador activo y guardarlo como artefacto reutilizable:

- Salida:
  - `models/preprocessor_active_<timestamp>.joblib` (preprocesador ajustado)
  - `models/preprocessor_active_<timestamp>.json` (metadatos)

- Carga y uso básico (en Python):
  - Carga el `.joblib` y usa `.transform(X)` sobre datos con el mismo esquema de columnas de entrenamiento.
  - Si cambiaste los grupos (`df_groups_final`) o el umbral de importancia, vuelve a entrenar y guardar.

- Versionado:
  - Por defecto, `models/*.joblib` y `.json` están ignorados en `.gitignore`.
  - Para versionar uno concreto, usa `git add -f` o ajusta `.gitignore`.

## 🤝 Contribución

1. Crear una branch para tu feature
2. Hacer cambios y agregar tests
3. Verificar que todos los tests pasen
4. Crear pull request

## 📞 Soporte

Si encuentras problemas:
1. Revisa esta documentación
2. Verifica los logs de la aplicación
3. Consulta las issues en el repositorio de GitHub